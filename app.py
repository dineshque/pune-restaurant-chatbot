# # backend/app.py
# from flask import Flask, request
# from twilio.twiml.messaging_response import MessagingResponse
# import os
# import pandas as pd
# import torch
# from semantic_search import semantic_search, model

# # Load processed data and embeddings once at startup
# df = pd.read_csv("data/processed/pune_restaurants_with_embeddings.csv")
# df["embedding"] = torch.load("data/processed/embeddings.pt")

# app = Flask(__name__)
# @app.route("/")
# def home():
#     return 'Welcome'
# @app.route("/whatsapp", methods=["POST"])
# def whatsapp_reply():
#     """
#     Receive incoming WhatsApp message, run search, and reply with top results.
#     """
#     user_msg = request.form.get("Body")
#     print(f"📩 User Query: {user_msg}")

#     # Run semantic search
#     results = semantic_search(user_msg, df, top_k=3)

#     # Format results nicely
#     reply_text = f"🔎 Top Restaurants for: '{user_msg}'\n\n"
#     for i, row in results.iterrows():
#         reply_text += (
#             f"🍽️ {row['name']} ({row['cuisine']})\n"
#             f"⭐ Rating: {row['rating']}\n"
#             f"📍 Address: {row['address']}\n\n"
#         )

#     # Send reply back via Twilio
#     resp = MessagingResponse()
#     resp.message(reply_text)
#     return str(resp)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

# backend/app.py
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import time, pandas as pd, torch

from backend.semantic_search import load_df_with_embeddings, hybrid_rank
from backend.reranker import rerank
from backend.llm_query_refiner import refine_query, parse_intent
from backend.logging_utils import log_query

# Load once at startup
DF = load_df_with_embeddings()

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    start = time.time()
    user_msg = request.form.get("Body", "")
    # optional: parse user-provided lat,lng if they send "loc:lat,lng"
    user_lat = user_lng = None
    if "loc:" in user_msg.lower():
        try:
            coord = user_msg.lower().split("loc:")[1].strip().split(",")
            user_lat, user_lng = float(coord[0]), float(coord[1])
            user_msg = user_msg.split("loc:")[0].strip()
        except: pass

    refined_query = refine_query(user_msg)

    # simple pre-filter: if min_rating set, keep only those
    df = DF
    intent = parse_intent(refined_query)

    if intent["min_rating"] > 0:
        df = df[df["rating"] >= intent["min_rating"]]

    # hybrid semantic ranking
    hybrid_top = hybrid_rank(user_msg, df, user_lat, user_lng, top_k=10)

    # cross-encoder rerank (final 5)
    final = rerank(user_msg, hybrid_top, out_k=5)

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

    latency = time.time() - start
    # convert pandas rows to dicts for logger
    log_query(user_msg, final[["name"]].to_dict("records"), latency, extras=intent)
    return str(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
