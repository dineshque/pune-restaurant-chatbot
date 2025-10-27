
import re, json
from typing import Dict


# 1️ Regex-based fallback parser

def parse_intent(user_query: str) -> Dict:
    """Fallback regex-based parser for local/offline use."""
    print(" [RegexParser] Starting fallback intent parsing...")

    # Defensive: ensure query is string
    if isinstance(user_query, dict):
        user_query = json.dumps(user_query)
    q = str(user_query).lower()

    # --- Cuisine detection ---
    cuisines = []
    cuisine_list = [
        "chinese","italian","north indian","south indian","mughlai","seafood",
        "cafe","vegetarian","veg","continental","thai","japanese","pizza",
        "biryani","burger","punjabi","maharashtrian","mexican","lebanese",
        "bbq","dessert"
    ]
    for c in cuisine_list:
        if c in q:
            cuisines.append(c.title())

    # --- Sentiment ---
    positive = any(w in q for w in ["best","top","amazing","great","famous","highest"])
    negative = any(w in q for w in ["worst","bad","poor","terrible","minimum","lowest","below","less than"])

    # --- Rating filter ---
    min_rating, max_rating = 0.0, 5.0
    rating_pattern = re.findall(r"(\d\.\d|\d)", q)
    if rating_pattern:
        try:
            val = float(rating_pattern[0])
            if any(x in q for x in ["below","under","less"]):
                max_rating = val
            else:
                min_rating = val
        except:
            pass

    # --- Distance ---
    near_me = any(x in q for x in ["near me","around me","close by","nearby","around"])
    distance_km = 5.0
    m = re.search(r"within\s*(\d+)\s*(km|kilometers|kms|miles)?", q)
    if m:
        val = float(m.group(1))
        if m.group(2) and "mile" in m.group(2):
            val *= 1.6
        distance_km = val

    # --- Meal time ---
    meal_time = None
    if "breakfast" in q:
        meal_time = "breakfast"
    elif "lunch" in q:
        meal_time = "lunch"
    elif "dinner" in q:
        meal_time = "dinner"

    # --- Location extraction ---
    location_text = None
    loc = re.search(r"near\s+([a-z\s]+)", q)
    if loc:
        location_text = loc.group(1).strip().split("within")[0].strip()

    result = {
        "cuisines": cuisines or None,
        "sentiment": "negative" if negative else ("positive" if positive else "neutral"),
        "rating_target": "low" if negative else ("high" if positive else None),
        "min_rating": min_rating,
        "max_rating": max_rating,
        "near_me": near_me,
        "distance_km": distance_km,
        "meal_time": meal_time,
        "location_text": location_text
    }

    print(f" [RegexParser] Parsed result → {result}\n")
    return result



# 2️  Groq-based structured parser

def refine_query(user_query: str, groq_api_key: str):
    """
    Use Groq LLM to parse intent.
    Automatically repairs minor JSON formatting issues (pipes, trailing commas, etc.).
    Falls back to regex-based parse_intent() if unrecoverable.
    """
    from groq import Groq
    import json, re

    print("\n [GroqParser] Starting structured intent parsing...")

    def safe_json_parse(raw_text, fallback=None):
        """Heuristic JSON repair and parsing."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            #  Attempt auto-repair for common LLM formatting issues
            fixed = raw_text
            fixed = fixed.replace("’", "'").replace("“", '"').replace("”", '"')
            fixed = re.sub(r"\|\s*['\"]?\w+['\"]?", "", fixed)     # remove | lunch etc
            fixed = re.sub(r",\s*}", "}", fixed)                   # trailing comma in dict
            fixed = re.sub(r",\s*]", "]", fixed)                   # trailing comma in list
            fixed = re.sub(r"None", "null", fixed)
            fixed = re.sub(r"True", "true", fixed)
            fixed = re.sub(r"False", "false", fixed)
            try:
                return json.loads(fixed)
            except Exception as e:
                print(f" [safe_json_parse] Failed even after repair: {e}")
                return fallback

    try:
        client = Groq(api_key=groq_api_key)
        system_prompt = """
        You are a restaurant search assistant for Pune.
        If the query is about restaurants, return ONLY a valid JSON object:
        {
          "intent_type": "restaurant_search",
          "cuisines": [list or null],
          "sentiment": "positive" | "negative" | "neutral",
          "rating_target": "high" | "low" | null,
          "min_rating": number or null,
          "max_rating": number or null,
          "distance_km": number or null,
          "location_text": string or null,
          "meal_time": "breakfast" | "lunch" | "dinner" | null
        }
        Output **pure JSON** only. 
        If the user query is NOT about restaurants, reply in plain text.
        """

        print(" Sending query to Groq model...")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2
        )
        print(" Using Groq model: llama-3.1-8b-instant")

        raw = response.choices[0].message.content.strip()
        print(f" [GroqParser] Raw response: {raw}")

        #  Check if JSON-like
        if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
            result = safe_json_parse(raw, fallback=None)
            if result is not None:
                print(" [GroqParser] Parsed structured JSON intent successfully.")
                if "intent_type" not in result:
                    result["intent_type"] = "restaurant_search"
                return result
            else:
                print(" [GroqParser] Could not parse JSON, falling back to regex parser.")
                return parse_intent(user_query)

        #  Otherwise, plain text conversational reply
        print(" [GroqParser] Non-JSON text detected — treating as conversational reply.")
        return raw

    except Exception as e:
        print(f" [GroqParser] Fatal error → fallback to regex parser: {e}")
        return parse_intent(user_query)
