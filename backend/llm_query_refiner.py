
# backend/llm_query_refiner.py (add/replace parse_intent)
import re
# backend/llm_query_refiner.py
import re, json
from typing import Dict

# ------------------------------
# 1️⃣  Regex-based fallback parser
# ------------------------------
def parse_intent(user_query: str) -> Dict:
    """Fallback regex-based parser for local/offline use."""
    q = user_query.lower()

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
        except: pass

    # --- Budget detection ---
    budget = "any"
    if any(x in q for x in ["cheap","budget","affordable","low cost","economy"]): budget="low"
    elif any(x in q for x in ["premium","expensive","fine dine","fine-dine","luxury"]): budget="high"

    # --- Open now ---
    open_now = any(x in q for x in ["open now","open","tonight","today"])

    # --- Distance ---
    near_me = any(x in q for x in ["near me","around me","close by","nearby","around"])
    distance_km = 5.0
    m = re.search(r"within\s*(\d+)\s*(km|kilometers|kms|miles)?", q)
    if m:
        val = float(m.group(1))
        if m.group(2) and "mile" in m.group(2): val *= 1.6
        distance_km = val

    # --- Meal time ---
    meal_time = None
    if "breakfast" in q: meal_time="breakfast"
    elif "lunch" in q: meal_time="lunch"
    elif "dinner" in q: meal_time="dinner"

    # --- Location extraction ---
    location_text = None
    loc = re.search(r"near\s+([a-z\s]+)", q)
    if loc:
        location_text = loc.group(1).strip().split("within")[0].strip()

    return {
        "cuisines": cuisines or None,
        "sentiment": "negative" if negative else ("positive" if positive else "neutral"),
        "rating_target": "low" if negative else ("high" if positive else None),
        "min_rating": min_rating,
        "max_rating": max_rating,
        "budget": budget,
        "open_now": open_now,
        "near_me": near_me,
        "distance_km": distance_km,
        "meal_time": meal_time,
        "location_text": location_text
    }


# ------------------------------
# 2️⃣  Groq-based structured parser
# ------------------------------
def refine_query(user_query: str, groq_api_key: str):
    """
    Use Groq LLM to parse intent. Falls back to regex parser on error.
    """
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)

        system_prompt = """
        You are a restaurant search assistant for Pune.
        Return only a valid JSON with the following fields:
        {
          "cuisines": [list or null],
          "sentiment": "positive" | "negative" | "neutral",
          "rating_target": "high" | "low" | null,
          "min_rating": number or null,
          "max_rating": number or null,
          "budget": "low" | "medium" | "high" | "any",
          "open_now": true | false,
          "distance_km": number or null,
          "location_text": string or null,
          "meal_time": "breakfast" | "lunch" | "dinner" | null
        }
        Output pure JSON, no text.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",

            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2
        )
        
        print("🧠 Using Groq model: llama-3.1-8b-instant")


        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        return result

    except Exception as e:
        print(f"⚠️  Groq error or unavailable → fallback to regex. Reason: {e}")
        return parse_intent(user_query)
