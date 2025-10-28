from backend.feedback_utils import send_feedback_request, save_feedback, ensure_feedback_file
from backend.semantic_search import load_df_with_embeddings, hybrid_rank
from backend.llm_query_refiner import refine_query, parse_intent
from twilio.twiml.messaging_response import MessagingResponse
from backend.logging_utils import log_query
from flask import Flask, request
import time, pandas as pd
import os

# Load dataset once at startup
DF = load_df_with_embeddings()

app = Flask(__name__)

# --- Detect if user is sending feedback ---
def is_feedback_message(msg: str, intent: dict) -> bool:
    """Return True if message looks like feedback rather than a query."""
    msg_lower = msg.lower().strip()
    words = msg_lower.split()
    short_message = len(words) <= 5
    feedback_keywords = ["good", "bad", "average", "nice", "poor", "amazing", "great", "ok", "fine", "comment"]
    cuisine_keywords = ["indian", "chinese", "biryani", "south", "north", "italian", "pizza", "burger", "restaurant", "food"]
    location_keywords = ["near", "in", "around", "within", "km", "road", "park"]

    # Feedback if short + contains sentiment word + no cuisine/location context + not restaurant intent
    if short_message and any(k in msg_lower for k in feedback_keywords):
        if not any(k in msg_lower for k in cuisine_keywords + location_keywords):
            if not (isinstance(intent, dict) and intent.get("intent_type") == "restaurant_search"):
                return True
    return False


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    start = time.time()
    user_msg = request.form.get("Body", "").strip()
    user_phone = request.form.get("From", "").replace("whatsapp:", "")
    ensure_feedback_file()
    print(f"\n📩 Incoming WhatsApp message: {user_msg}")


    # --- Optional: short phrases like “hi”, “thanks” shouldn’t trigger LLM ---
    if user_msg.lower() in ["hi", "hello", "thanks", "thank you", "ok"]:
        resp = MessagingResponse()
        resp.message("👋 Hi there! Ask me about restaurants in Pune — like 'best Italian near me'.")
        return str(resp)

    # --- Extract optional coordinates from message like "loc:18.52,73.85" ---
    user_lat = user_lng = None
    if "loc:" in user_msg.lower():
        try:
            coord = user_msg.lower().split("loc:")[1].strip().split(",")
            user_lat, user_lng = float(coord[0]), float(coord[1])
            user_msg = user_msg.split("loc:")[0].strip()
            print(f"📍 Parsed user coordinates: ({user_lat}, {user_lng})")
        except Exception as e:
            print(f"⚠️  Failed to parse coordinates: {e}")

    # --- Get intent from Groq (only once) ---
    refined_query = refine_query(
        user_msg,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # --- Handle Feedback After Intent Parsing ---
    if is_feedback_message(user_msg, refined_query):
        save_feedback(user_phone, f"Comment: {user_msg}")
        resp = MessagingResponse()
        resp.message("Thank you for sharing your feedback 💬!")
        print(f"📝 Feedback saved safely (true feedback): {user_msg}")
        return str(resp)
    
    # Case A — Non-restaurant / conversational text reply
    if isinstance(refined_query, str):
        print("💬 Conversational query detected. Sending Groq's text reply directly.")
        resp = MessagingResponse()
        resp.message(refined_query)
        return str(resp)

    # Case B — Restaurant search intent (structured)
    if isinstance(refined_query, dict) and refined_query.get("intent_type") == "restaurant_search":
        print("🍽️ Detected structured restaurant search intent.")
        intent = refined_query or parse_intent(user_msg)
        print(f"🎯 Parsed Intent: {intent}")

        df = DF.copy()

        # Hybrid ranking (skip redundant LLM call)
        from backend.semantic_search import embed_query
        embed_query.cache_clear()
        print("🧹 Cleared cached query embeddings.")

        hybrid_top = hybrid_rank(
            user_query=user_msg,
            df=df,
            user_lat=user_lat,
            user_lng=user_lng,
            top_k=5,
            intent=intent,
            groq_api_key=None
        )

        # Prepare WhatsApp response
        reply_lines = [f"🔎 Results for: '{user_msg}'"]
        for _, r in hybrid_top.iterrows():
            dist = f" • {r['distance_km']:.1f} km" if not pd.isna(r.get("distance_km")) else ""
        
            contact = r.get("phone", "N/A") if pd.notna(r.get("phone")) else "N/A"
            reply_lines.append(
                f"🍽️ {r['name']} ({r['cuisine']})\n"
                f"⭐ {r['rating']}  |  Trust {r.get('authenticity_score', r['rating']):.2f}{dist}\n"
                f"📍 {r['address']}\n"
                f"📞 Contact: {contact}\n"
            )


        resp = MessagingResponse()
        resp.message("\n".join(reply_lines))

        # Schedule feedback reminder (after 60s)
        from backend.feedback_tasks import schedule_feedback_request
        schedule_feedback_request(user_phone, user_msg, hybrid_top["name"].tolist(), delay_sec=60)

        # Log query
        latency = time.time() - start
        log_query(user_msg, hybrid_top[["name"]].to_dict("records"), latency, extras=intent)
        print(f"✅ Response sent successfully in {latency:.2f}s.")
        return str(resp)

    # Case C — Unrecognized or malformed structure
    print("⚠️ Unrecognized structure. Sending fallback reply.")
    resp = MessagingResponse()
    resp.message("Sorry, I didn’t quite understand that. Try asking about cuisines, e.g. 'top North Indian near FC Road'.")
    return str(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
