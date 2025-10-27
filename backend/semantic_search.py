import os, math, time
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import torch
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
from backend.llm_query_refiner import refine_query


#  MODEL SETUP

MODEL_NAME = "thenlper/gte-large"  # high-performance 1024D model
_model = SentenceTransformer(MODEL_NAME)



#  UTILITY HELPERS

def _normalize(col):
    x = pd.to_numeric(col, errors="coerce").fillna(0).astype(float)
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-9)

def haversine_km(lat1, lon1, lat2, lon2):
    if pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

@lru_cache(maxsize=512)
def embed_query(text: str):
    """Embed a single query string (cached for efficiency)."""
    return _model.encode(text, convert_to_tensor=True)



#  LOAD & PRECOMPUTE EMBEDDINGS

def generate_embeddings_if_missing():
    emb_path = "data/processed/embeddings.pt"
    csv_path = "data/processed/pune_restaurants_with_embeddings.csv"
    input_path = "data/processed/pune_restaurants_cleaned.csv"

    if os.path.exists(emb_path) and os.path.exists(csv_path):
        print(" Found existing embeddings, skipping regeneration.")
        return

    if not os.path.exists(input_path):
        raise FileNotFoundError(f" Input file not found: {input_path}")

    print(" Generating restaurant embeddings...")
    df = pd.read_csv(input_path)
    df["summary_text"] = (
        df["name"].fillna("") + " " +
        df["cuisine"].fillna("") + " " +
        df["address"].fillna("") + " Rating: " +
        df["rating"].astype(str)
    )

    embeddings = _model.encode(df["summary_text"].tolist(), convert_to_tensor=True)
    os.makedirs("data/processed", exist_ok=True)
    torch.save(embeddings, emb_path)
    df.to_csv(csv_path, index=False)
    print(f" Generated embeddings for {len(df)} restaurants.\n")

def load_df_with_embeddings():
    generate_embeddings_if_missing()
    df = pd.read_csv("data/processed/pune_restaurants_with_embeddings.csv")
    embeddings = torch.load("data/processed/embeddings.pt", weights_only=False)

    if isinstance(embeddings, torch.Tensor):
        embeddings = [embeddings[i] for i in range(embeddings.shape[0])]

    if len(df) != len(embeddings):
        raise ValueError(f" Mismatch: {len(df)} rows vs {len(embeddings)} embeddings.")

    df = df.copy()
    df["embedding"] = embeddings
    print(f" Loaded {len(df)} rows with embeddings (dim={embeddings[0].shape[-1]}).")
    return df



#  SEMANTIC SEARCH + HYBRID LOGIC

def semantic_candidates(user_query: str, df: pd.DataFrame, k: int = 50):
    """Retrieve top-k semantically similar restaurants."""
    q_emb = embed_query(user_query)
    df = df.copy()
    df["similarity"] = df["embedding"].apply(lambda e: float(util.cos_sim(q_emb, e)))
    return df.sort_values("similarity", ascending=False).head(k)


def hybrid_rank(user_query, df, user_lat=None, user_lng=None, top_k=5, intent=None, groq_api_key=None):
    """
    Hybrid ranking with:
      - Structured or LLM intent (if provided)
      - Semantic similarity
      - Sentiment & distance weighting
      Handles:
      • “near me” → use user coords if provided; else Pune center; default radius 3 km
      • Textual location → geocode with Pune bias (e.g., “FC Road”)
      • If neither provided → no hard radius filter; distance used as soft score if coords available
    
    """
    print("\n Starting hybrid_rank()")
    print(f" Query: {user_query}")

    
    # Step 1: Intent handling
    
    if intent:
        print(" Using pre-parsed intent passed from app.py")
    else:
        print(" No pre-parsed intent, calling Groq/regex inside hybrid_rank()")
        intent = refine_query(user_query, groq_api_key)

    print(f" Intent received: {intent}")

    qlow = user_query.lower()
    sentiment = intent.get("sentiment", "neutral")
    rating_target = intent.get("rating_target") or (
        "low" if any(w in qlow for w in ["worst", "bad", "below", "less"]) else
        ("high" if any(w in qlow for w in ["best", "top", "great"]) else None)
    )

    # near_me = bool(intent.get("near_me", False))
    # if near_me or intent.get("location_text") == 'near me':
    #     distance_km = 20.0
    # else:
    #     distance_km = float(intent.get("distance_km") or 0)
    # enforce_distance = near_me or (distance_km > 0)
    # print("distance_km -------------->", distance_km)
    # print(f" Sentiment={sentiment} | RatingTarget={rating_target} | NearMe={near_me} | DistanceFilter={enforce_distance} | Radius={distance_km}")

    # 2) Location logic
    near_me_flag = bool(intent.get("near_me", False) or str(intent.get("location_text","")).strip().lower() == "near me")

    # Decide radius: explicit distance_km wins; else default 3 km for “near me”; else None (no hard filter)
    explicit_radius = intent.get("distance_km", None)
    if explicit_radius is not None and str(explicit_radius).strip() != "":
        try:
            distance_km = float(explicit_radius)
        except:
            distance_km = 5.0 if near_me_flag else None
    else:
        distance_km = 5.0 if near_me_flag else None

    # If “near me”: prefer given user_lat/lng; if absent, use Pune center
    if near_me_flag:
        if user_lat is None or user_lng is None:
            print("⚠️ ‘near me’ used but no coords provided — defaulting to Pune center")
            user_lat, user_lng = 18.5204, 73.8567
        else:
            print(f"📍 Using user coords for ‘near me’: ({user_lat},{user_lng})")
    else:
        # textual location workflow (geocode if coords not provided)
        loc_text = intent.get("location_text")
        if (user_lat is None or user_lng is None) and loc_text:
            lat, lng = geocode_location(loc_text)
            if lat and lng:
                user_lat, user_lng = lat, lng
                print(f"📍 Geocoded '{loc_text}' → ({user_lat},{user_lng})")
            else:
                print(f"⚠️ Couldn’t geocode '{loc_text}', falling back to Pune center")
                user_lat, user_lng = 18.5204, 73.8567

    # Hard filter only if a radius is decided (either explicit or near_me defaulted)
    enforce_distance = distance_km is not None
    print(f"🎯 Sentiment={sentiment} | RatingTarget={rating_target} | NearMe={near_me_flag} | EnforceRadius={enforce_distance} | Radius={distance_km}")

    # Step 3: Semantic pool
    
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    print(f" Valid ratings range: {df['rating'].min()}–{df['rating'].max()}")

    k_pool = len(df) if rating_target == "low" else min(400, len(df))
    candidates = semantic_candidates(user_query, df, k=k_pool)
    print(f" Retrieved {len(candidates)} semantic candidates")

    
    # Step 4: Cuisine Filter
    
    cuisines = intent.get("cuisines")
    if cuisines:
        before = len(candidates)
        cuisines_lower = [c.lower() for c in cuisines]
        candidates = candidates[candidates["cuisine"].str.lower().apply(lambda x: any(c in str(x) for c in cuisines_lower))]
        print(f" Cuisine filter: {before} → {len(candidates)} for {cuisines}")
    else:
        print(" No specific cuisine filter applied")

    
    # Step 5: Rating Filter
    
    if rating_target == "high":
        candidates = candidates[candidates["rating"] >= 4.0]
        print(f" High rating filter (>=4.0) → {len(candidates)} remain")
    elif rating_target == "low":
        candidates = candidates[candidates["rating"] <= 3.5]
        print(f" Low rating filter (<=3.5) → {len(candidates)} remain")
    else:
        print(" No rating filter applied")

    
    # Step 6: Distance Handling
    
    from backend.geo_utils import geocode_location

    if (user_lat is None or user_lng is None) and intent.get("location_text"):
        loc = intent["location_text"]
        lat, lng = geocode_location(loc)
        if lat and lng:
            user_lat, user_lng = lat, lng
            print(f" Geocoded '{loc}' → ({user_lat}, {user_lng})")
        else:
            print(f" Geocoding failed for '{loc}', defaulting to Pune center")
            user_lat, user_lng = 18.5204, 73.8567

    if user_lat and user_lng:
        candidates["distance_km"] = candidates.apply(
            lambda r: haversine_km(user_lat, user_lng, r["latitude"], r["longitude"]), axis=1
        )
        candidates["proximity_score"] = np.exp(-candidates["distance_km"] / 5.0)
        if enforce_distance and distance_km:
            before = len(candidates)
            candidates = candidates[candidates["distance_km"] <= distance_km + 0.5]
            print(f" Radius filter ≤ {distance_km} km → {len(candidates)} (dropped {before - len(candidates)})")
    else:
        candidates["distance_km"] = np.nan
        candidates["proximity_score"] = 0.0
        print(" No user location provided — skipping distance filter")

    
    # Step 6: Final Scoring
    
    sim = _normalize(candidates["similarity"])
    auth = _normalize(candidates["rating"])
    prox = _normalize(candidates["proximity_score"])

    w_sem, w_auth, w_prox = ((0.5, 0.3, 0.2) if enforce_distance else (0.6, 0.3, 0.1))
    candidates["final_score"] = w_sem * sim + w_auth * auth + w_prox * prox
    print(f" Weights: semantic={w_sem}, rating={w_auth}, distance={w_prox}")

    ascending = sentiment == "negative"
    result = candidates.sort_values("final_score", ascending=ascending).head(top_k)

    print(f" Final {len(result)} results (sentiment={sentiment})")
    return result[["name", "cuisine", "address","number", "rating", "distance_km", "final_score"]]


if __name__ == "__main__":
    generate_embeddings_if_missing()
