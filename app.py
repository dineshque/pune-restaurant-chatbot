from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import time, pandas as pd
import os
from backend.semantic_search import load_df_with_embeddings, hybrid_rank
# from backend.reranker import rerank
from backend.llm_query_refiner import refine_query, parse_intent
from backend.logging_utils import log_query

# Load dataset once at startup
DF = load_df_with_embeddings()

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    start = time.time()
    user_msg = request.form.get("Body", "")
    print(f"\n Incoming WhatsApp message: {user_msg}")

    # Extract optional coordinates from message like "loc:18.52,73.85"
    user_lat = user_lng = None
    # if "loc:" in user_msg.lower():
    #     try:
    #         coord = user_msg.lower().split("loc:")[1].strip().split(",")
    #         user_lat, user_lng = float(coord[0]), float(coord[1])
    #         user_msg = user_msg.split("loc:")[0].strip()
    #         print(f" Parsed user coordinates: ({user_lat}, {user_lng})")
    #     except Exception as e:
    #         print(f"  Failed to parse coordinates: {e}")

    #  Step 1: Get intent from Groq (only once)
    refined_query = refine_query(
        user_msg,
        groq_api_key= os.getenv("GROQ_API_KEY")
    )

    #  Case A — Non-restaurant / conversational text reply
    if isinstance(refined_query, str):
        print(" Conversational query detected. Sending Groq's text reply directly.")
        resp = MessagingResponse()
        resp.message(refined_query)
        return str(resp)

    #  Case B — Restaurant search intent (structured)
    if isinstance(refined_query, dict) and refined_query.get("intent_type") == "restaurant_search":
        print(" Detected structured restaurant search intent.")

        intent = refined_query or parse_intent(user_msg)
        print(f" Parsed Intent: {intent}")

        df = DF.copy()

        # Step 2: Hybrid ranking — pass parsed intent instead of re-calling Groq
        hybrid_top = hybrid_rank(
            user_query=user_msg,
            df=df,
            user_lat=user_lat,
            user_lng=user_lng,
            top_k=5,
            intent=intent,       #  Pass parsed intent here
            groq_api_key=None    #  Skip redundant Groq call inside hybrid_rank
        )

        # Step 3: Re-rank top results with cross-encoder
        final = hybrid_top #rerank(user_msg, hybrid_top, out_k=5)
        # final = rerank(user_msg, hybrid_top, out_k=5)
        # print("Final Result ---------->",final )

        # Step 4: Prepare WhatsApp response
        reply_lines = [f"🔎 Results for: '{user_msg}'"]
        for _, r in final.iterrows():
            dist = f" • {r['distance_km']:.1f} km" if not pd.isna(r.get("distance_km")) else ""
            reply_lines.append(
                f"🍽️ {r['name']} ({r['cuisine']})\n"
                f"⭐ {r['rating']}  |  Trust {r.get('authenticity_score', r['rating']):.2f}{dist}\n"
                f"📍 {r['address']}\n"
            )

        resp = MessagingResponse()
        resp.message("\n".join(reply_lines))

        # Step 5: Log query + results
        latency = time.time() - start
        log_query(user_msg, final[["name"]].to_dict("records"), latency, extras=intent)
        print(f" Response sent successfully in {latency:.2f}s.")
        return str(resp)

    # 🔸 Case C — Unrecognized or malformed structure
    print(" Unrecognized structure. Sending fallback reply.")
    resp = MessagingResponse()
    resp.message("Sorry, I didn’t quite understand that. Please ask about restaurants or cuisines.")
    return str(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
