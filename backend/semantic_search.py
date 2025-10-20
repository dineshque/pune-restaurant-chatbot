# # backend/semantic_search.py
# import os
# import torch
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util

# # ✅ Hide TensorFlow logs for cleaner output
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # ✅ Load model
# print("🔹 Loading sentence transformer model...")
# model = SentenceTransformer("all-MiniLM-L6-v2")
# print("✅ Model loaded successfully.\n")

# # ✅ Define file paths
# input_file = "data/processed/pune_restaurants_cleaned.csv"
# output_csv = "data/processed/pune_restaurants_with_embeddings.csv"
# output_pt = "data/processed/embeddings.pt"

# # ✅ Check input file exists
# if not os.path.exists(input_file):
#     raise FileNotFoundError(f"❌ Input file not found: {input_file}")

# # ✅ Load data
# df = pd.read_csv(input_file)
# print(f"📄 Loaded {len(df)} restaurant records.")

# # ✅ Create summary text for embeddings
# df["summary_text"] = (
#     df["name"].fillna("") + " " +
#     df["cuisine"].fillna("") + " " +
#     df["address"].fillna("") + " Rating: " +
#     df["rating"].astype(str)
# )

# # ✅ Generate embeddings with progress logs
# embeddings = []
# for i, text in enumerate(df["summary_text"]):
#     if i % 25 == 0:
#         print(f"➡️ Processing row {i+1}/{len(df)}...")
#     emb = model.encode(text, convert_to_tensor=True)
#     embeddings.append(emb)

# # ✅ Save embeddings
# os.makedirs("data/processed", exist_ok=True)
# torch.save(embeddings, output_pt)
# df.to_csv(output_csv, index=False)

# print("\n✅ Done!")
# print(f"🔸 Saved embeddings to: {output_pt}")
# print(f"🔸 Saved dataset to: {output_csv}")
# print(f"Total embeddings generated: {len(embeddings)}")

# import os

# # ✅ Disable TensorFlow backend for SentenceTransformers
# os.environ["USE_TF"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TF logs if it tries to load

# from sentence_transformers import SentenceTransformer, util
# import pandas as pd
# import torch

# # Load model globally once (PyTorch backend only)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def generate_embeddings():
#     """
#     Create and store embeddings for semantic search.
#     """
#     df = pd.read_csv("data/processed/pune_restaurants_cleaned.csv")

#     # Prepare unified text column for embeddings
#     df["summary_text"] = (
#         df["name"].fillna("") + " " +
#         df["cuisine"].fillna("") + " " +
#         df["address"].fillna("") + " " +
#         df["rating"].astype(str)
#     )

#     print("🔹 Generating embeddings...")
#     df["embedding"] = df["summary_text"].apply(
#         lambda x: model.encode(x, convert_to_tensor=True)
#     )

#     # Save embeddings
#     torch.save(df["embedding"].tolist(), "data/processed/embeddings.pt")
#     df.to_csv("data/processed/pune_restaurants_with_embeddings.csv", index=False)
#     print(f"✅ Embeddings generated and saved for {len(df)} restaurants.")
#     return df


# def semantic_search(user_query, df, top_k=5):
#     """
#     Perform semantic similarity search using cosine similarity.
#     """
#     query_embedding = model.encode(user_query, convert_to_tensor=True)
#     df["similarity"] = df["embedding"].apply(lambda x: float(util.cos_sim(query_embedding, x)))
#     results = df.sort_values("similarity", ascending=False).head(top_k)
#     return results[["name", "cuisine", "address", "rating", "similarity"]]


# if __name__ == "__main__":
#     generate_embeddings()


# # backend/semantic_search.py
# import os, math, time
# os.environ["USE_TF"] = "0"            # force PyTorch backend
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import numpy as np
# import pandas as pd
# import torch
# from functools import lru_cache
# from sentence_transformers import SentenceTransformer, util

# MODEL_NAME = "thenlper/gte-large"     # higher quality than MiniLM
# # MODEL_NAME = "all-MiniLM-L6-v2"     # fallback if RAM is tight

# # Load model once
# _model = SentenceTransformer(MODEL_NAME)

# def _normalize(col):
#     x = pd.to_numeric(col, errors="coerce").fillna(0).astype(float)
#     rng = x.max() - x.min()
#     return (x - x.min()) / (rng + 1e-9)

# @lru_cache(maxsize=512)
# def embed_query(text: str):
#     return _model.encode(text, convert_to_tensor=True)

# def load_df_with_embeddings():
#     df = pd.read_csv("data/processed/pune_restaurants_with_embeddings.csv")
#     df["embedding"] = torch.load("data/processed/embeddings.pt")
#     return df

# def haversine_km(lat1, lon1, lat2, lon2):
#     if pd.isna(lat2) or pd.isna(lon2):
#         return np.nan
#     R=6371
#     dlat = math.radians(lat2 - lat1)
#     dlon = math.radians(lon2 - lon1)
#     a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
#     return 2*R*math.asin(math.sqrt(a))

# def semantic_candidates(user_query:str, df:pd.DataFrame, k:int=50):
#     q_emb = embed_query(user_query)
#     # cosine similarity with precomputed doc embeddings
#     df = df.copy()
#     df["similarity"] = df["embedding"].apply(lambda e: float(util.cos_sim(q_emb, e)))
#     return df.sort_values("similarity", ascending=False).head(k)

# def hybrid_rank(user_query:str,
#                 df:pd.DataFrame,
#                 user_lat:float|None=None,
#                 user_lng:float|None=None,
#                 top_k:int=5):
#     candidates = semantic_candidates(user_query, df, k=50)

#     # proximity
#     if user_lat is not None and user_lng is not None:
#         candidates["distance_km"] = candidates.apply(
#             lambda r: haversine_km(user_lat, user_lng, r.get("latitude"), r.get("longitude")), axis=1)
#         candidates["proximity_score"] = np.exp(-(candidates["distance_km"].fillna(50))/5.0)  # 5km decay
#     else:
#         candidates["distance_km"] = np.nan
#         candidates["proximity_score"] = 0.0

#     # authenticity (from preprocessing)
#     if "authenticity_score" not in candidates.columns:
#         candidates["authenticity_score"] = candidates["rating"]

#     # normalize signals
#     sim = _normalize(candidates["similarity"])
#     auth = _normalize(candidates["authenticity_score"])
#     prox = _normalize(candidates["proximity_score"])

#     # weights (tune with a small eval set)
#     w_sem, w_auth, w_prox = 0.6, 0.3, 0.1
#     candidates["final_score"] = w_sem*sim + w_auth*auth + w_prox*prox

#     out = candidates.sort_values("final_score", ascending=False).head(top_k)
#     return out[["name","cuisine","address","rating","authenticity_score","distance_km","final_score"]]

# backend/semantic_search.py
# from backend.llm_query_refiner import parse_intent
import os, math, time
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import torch
from functools import lru_cache
from backend.llm_query_refiner import refine_query
# from llm_query_refiner import refine_query
from sentence_transformers import SentenceTransformer, util

# 🔹 Choose model — must stay consistent for both generation & search
# MODEL_NAME = "all-MiniLM-L6-v2"  # ✅ simpler + smaller (384D)
MODEL_NAME = "thenlper/gte-large"  # use this if you have more GPU/RAM

_model = SentenceTransformer(MODEL_NAME)

# =============================
#  UTILITY HELPERS
# =============================

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

# =============================
#  MAIN FUNCTIONS
# =============================

def generate_embeddings_if_missing():
    """Generate embeddings once if not already available."""
    emb_path = "data/processed/embeddings.pt"
    csv_path = "data/processed/pune_restaurants_with_embeddings.csv"
    input_path = "data/processed/pune_restaurants_cleaned.csv"

    if os.path.exists(emb_path) and os.path.exists(csv_path):
        print("✅ Found existing embeddings, skipping regeneration.")
        return

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    print("⚙️ Generating restaurant embeddings...")
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
    print(f"✅ Generated embeddings for {len(df)} restaurants.\n")

def load_df_with_embeddings():
    """Load dataset and attach embeddings properly."""
    generate_embeddings_if_missing()

    df = pd.read_csv("data/processed/pune_restaurants_with_embeddings.csv")

    # Load embeddings from torch file
    embeddings = torch.load("data/processed/embeddings.pt", weights_only=False)

    # ✅ If embeddings are a single 2D tensor (N x D), convert to list of row tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = [embeddings[i] for i in range(embeddings.shape[0])]

    # ✅ Sanity check
    if len(df) != len(embeddings):
        raise ValueError(f"❌ Mismatch: Data has {len(df)} rows, embeddings have {len(embeddings)} vectors.")

    # ✅ Assign one embedding per restaurant
    df = df.copy()
    df["embedding"] = embeddings

    print(f"✅ Loaded {len(df)} rows with {len(embeddings)} embeddings (dim={embeddings[0].shape[-1]}).")
    return df


def semantic_candidates(user_query: str, df: pd.DataFrame, k: int = 50):
    """Retrieve top-k semantically similar restaurants."""
    q_emb = embed_query(user_query)
    df = df.copy()
    df["similarity"] = df["embedding"].apply(lambda e: float(util.cos_sim(q_emb, e)))
    return df.sort_values("similarity", ascending=False).head(k)

def hybrid_rank(user_query, df, user_lat=None, user_lng=None, top_k=5, groq_api_key=None):
    """
    Hybrid ranking with:
      - LLM/regex intent
      - semantic similarity
      - (optional) distance-aware scoring
      - sentiment-aware sort
      - progressive backoff to avoid empty results
    Includes rich debug prints at each step.
    """
    print(df.columns)
    print("\n🧠 Starting hybrid_rank()")
    print(f"🔍 Query: {user_query}")

    # 1) Intent
    print("➡️  Calling LLM/regex-based intent parser...")
    intent = refine_query(user_query, groq_api_key)
    print(f"🧩 Parsed intent: {intent}")

    qlow = user_query.lower()
    sentiment = intent.get("sentiment", "neutral")
    rating_target = intent.get("rating_target") or (
        "low" if any(w in qlow for w in ["worst","bad","minimum","below","less"]) else
        ("high" if any(w in qlow for w in ["best","top","highest"]) else None)
    )
    # Only treat distance as a hard FILTER if user asked for it (near_me or explicit radius)
    near_me = bool(intent.get("near_me", False))
    req_radius = intent.get("distance_km", None)
    enforce_distance_filter = near_me or (req_radius is not None)
    # If no explicit radius, we will NOT filter; we’ll just use distance as a soft score.
    distance_km = float(req_radius) if req_radius not in (None, "", False) else None

    print(f"🎯 Sentiment={sentiment} | RatingTarget={rating_target} | NearMe={near_me} | DistanceFilter={enforce_distance_filter} | Radius={distance_km}")

    # 2) Ensure numeric rating
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["rating"])
    print(f"🧹 Ratings normalized → dropped {before - len(df)} rows without ratings. Range: {df['rating'].min()}–{df['rating'].max()}")

    # 3) Candidate pool size: make it BIG for "worst" so low-rated can appear
    # At ~2000 rows, full similarity is acceptable; if dataset grows, set k=1000 cap.
    if rating_target == "low" or sentiment == "negative":
        k_pool = len(df)
    else:
        k_pool = min(400, len(df))  # a bit larger than 100 to reduce bias toward only "popular" items
    print(f"⚙️  Building semantic candidate pool of size: {k_pool}")

    # Build semantic candidates: compute similarity on the chosen pool
    # We reuse your semantic_candidates but widen its k; if k==len(df) we’ll compute for all.
    candidates = semantic_candidates(user_query, df, k=k_pool)
    print(f"✅ Retrieved {len(candidates)} semantic candidates")
    print(f"   Similarity range: {candidates['similarity'].min():.4f}–{candidates['similarity'].max():.4f}")

    # === 4A) Cuisine-based filtering (before rating) ===
    cuisines = intent.get("cuisines")
    if cuisines:
        cuisines_lower = [c.lower() for c in cuisines]
        before = len(candidates)
        candidates = candidates[
            candidates["cuisine"].str.lower().apply(
                lambda x: any(c in str(x) for c in cuisines_lower)
            )
        ]
        print(f"🥢 Cuisine filter → {len(candidates)} remain (from {before}) for {cuisines}")
    else:
        print("ℹ️  No specific cuisine requested — keeping all cuisines")

    # 4) Rating filter (robust)
    # First do a copy so we can roll back during backoff if needed.
    base_candidates = candidates.copy()

    if rating_target == "high":
        candidates = candidates[candidates["rating"] >= 4.0]
        print(f"⭐ HIGH-rating filter (>=4.0) → {len(candidates)} remain")
    elif rating_target == "low":
        low_cut = 3.5  # primary threshold
        low_rated = candidates[candidates["rating"] <= low_cut]
        if len(low_rated) > 0:
            candidates = low_rated
            print(f"📉 LOW-rating filter (<= {low_cut}) → {len(candidates)} remain")
        else:
            # fallback: bottom 20% by rating inside semantic pool
            q20 = candidates["rating"].quantile(0.20)
            candidates = candidates[candidates["rating"] <= q20]
            print(f"⚠️  No items <= {low_cut}. Using bottom 20% fallback (cutoff={q20:.2f}) → {len(candidates)} remain")
    else:
        print("ℹ️  No rating filter applied")

        # 6️⃣ Distance computation (handles naming, type, and hidden spaces)
    from backend.geo_utils import geocode_location

# 🔍 Step 0 — Auto-detect location if provided in intent
    if (user_lat is None or user_lng is None) and intent.get("location_text"):
        location_text = intent["location_text"]
        lat, lng = geocode_location(location_text)
        if lat and lng:
            user_lat, user_lng = lat, lng
            print(f"✅ Using geocoded coordinates for '{location_text}' → ({user_lat}, {user_lng})")
        else:
            print(f"⚠️ Could not geocode '{location_text}', using default Pune center (18.5204, 73.8567)")
            user_lat, user_lng = 18.5204, 73.8567  # Default Pune coordinates


        # Normalize column names just once
        candidates.columns = candidates.columns.str.strip().str.lower()

        # Try to detect latitude/longitude column variations
        lat_col = next((c for c in candidates.columns if "lat" in c), None)
        lng_col = next((c for c in candidates.columns if "lon" in c or "lng" in c), None)

        if not lat_col or not lng_col:
            raise KeyError(f"❌ Latitude/Longitude columns not found in {candidates.columns.tolist()}")

        # Ensure numeric types
        candidates[lat_col] = pd.to_numeric(candidates[lat_col], errors="coerce")
        candidates[lng_col] = pd.to_numeric(candidates[lng_col], errors="coerce")

        # Compute distances safely
        candidates["distance_km"] = candidates.apply(
            lambda r: haversine_km(user_lat, user_lng, r[lat_col], r[lng_col]),
            axis=1
        )

        n_missing = candidates["distance_km"].isna().sum()
        print(f"✅ Distances computed: {len(candidates)-n_missing}/{len(candidates)} (NaN={n_missing})")

        # Fill NaN with max (neutral for ranking)
        candidates["distance_km"] = candidates["distance_km"].fillna(candidates["distance_km"].max())

        # Compute proximity score (soft factor)
        candidates["proximity_score"] = np.exp(-(candidates["distance_km"]) / 5.0)

        if enforce_distance_filter and distance_km is not None:
            before = len(candidates)
            candidates = candidates[candidates["distance_km"] <= distance_km + 0.5]
            print(f"📏 Radius filter ≤ {distance_km} km → {len(candidates)} remain (dropped {before - len(candidates)})")
        else:
            print("ℹ️  No hard radius filter applied (soft scoring only)")

        print(f"📊 Distance range: {candidates['distance_km'].min():.2f}–{candidates['distance_km'].max():.2f} km")

    else:
        candidates["distance_km"] = np.nan
        candidates["proximity_score"] = 0.0
        print("ℹ️  No user location provided — skipping distance computation.")



    # 6) Progressive backoff if we’re at risk of returning 0 results
    # a) If empty after rating/distance filters, relax distance, then relax rating.
    if len(candidates) == 0:
        print("🚨 Backoff: No candidates remain. Removing distance filter and retrying rating-only...")
        candidates = base_candidates.copy()
        if rating_target == "low":
            low_rated = candidates[candidates["rating"] <= 3.5]
            if len(low_rated) > 0:
                candidates = low_rated
                print(f"↩️  Backoff result: LOW-rating only → {len(candidates)} remain")
            else:
                q20 = candidates["rating"].quantile(0.20)
                candidates = candidates[candidates["rating"] <= q20]
                print(f"↩️  Backoff result: bottom 20% (cutoff={q20:.2f}) → {len(candidates)} remain")
        elif rating_target == "high":
            candidates = candidates[candidates["rating"] >= 4.0]
            print(f"↩️  Backoff result: HIGH-rating only → {len(candidates)} remain")
        else:
            print("↩️  Backoff result: no rating filter → using semantic pool")

    # If still too few (< top_k), relax further: use pure rating sort from the semantic pool
    if len(candidates) < top_k:
        print(f"⚠️  Only {len(candidates)} candidates. Expanding from base pool by rating to fill {top_k}.")
        # Merge with a few more from base pool (ensure no duplicates)
        need = top_k - len(candidates)
        # Choose the bottom or top by rating depending on sentiment
        if sentiment == "negative":
            filler = base_candidates.sort_values("rating", ascending=True).head(max(need, 10))
        else:
            filler = base_candidates.sort_values("rating", ascending=False).head(max(need, 10))
        candidates = pd.concat([candidates, filler]).drop_duplicates(subset=["name","address"], keep="first")
        print(f"🧩 After filler merge → {len(candidates)} candidates")
    
    # 🩹 Ensure every row has a valid distance_km and proximity_score
    print("🩹 Validating distance and proximity columns...")

    # Create missing columns if absent
    if "distance_km" not in candidates.columns:
        candidates["distance_km"] = np.nan
    if "proximity_score" not in candidates.columns:
        candidates["proximity_score"] = 0.0

    # Detect possible latitude / longitude column names
    cols_lower = [c.lower() for c in candidates.columns]
    lat_col = next((c for c in cols_lower if c.startswith("lat")), None)
    lng_col = next((c for c in cols_lower if c.startswith("lon") or "lng" in c), None)

    if user_lat is not None and user_lng is not None and lat_col and lng_col:
        lat_col = candidates.columns[cols_lower.index(lat_col)]
        lng_col = candidates.columns[cols_lower.index(lng_col)]

        # Recompute distance for missing rows
        missing_mask = candidates["distance_km"].isna() | (candidates["distance_km"] <= 0)
        if missing_mask.any():
            print(f"📏 Recomputing distances for {missing_mask.sum()} filler rows using '{lat_col}', '{lng_col}'...")
            candidates.loc[missing_mask, "distance_km"] = candidates[missing_mask].apply(
                lambda r: haversine_km(user_lat, user_lng, r.get(lat_col), r.get(lng_col)),
                axis=1
            )

        # Fill any remaining NaN with max distance (neutral)
        max_d = candidates["distance_km"].max(skipna=True) or 50.0
        candidates["distance_km"] = candidates["distance_km"].fillna(max_d)
        candidates["proximity_score"] = np.exp(-(candidates["distance_km"]) / 5.0)
        print(f"✅ Distances verified. Range: {candidates['distance_km'].min():.2f}–{candidates['distance_km'].max():.2f} km")
    else:
        print("⚠️ User coordinates or lat/lng columns missing — skipping distance recomputation.")
        max_d = 50.0
        candidates["distance_km"] = candidates["distance_km"].fillna(max_d)
        candidates["proximity_score"] = np.exp(-(candidates["distance_km"]) / 5.0)

    # 7) Compute hybrid score and sort
    print("🧮 Computing hybrid score...")
    sim  = _normalize(candidates["similarity"])
    auth = _normalize(candidates["rating"])
    prox = _normalize(candidates["proximity_score"])

    # Weight distance more only if user asked for near me / radius
    w_sem, w_auth, w_prox = ((0.5, 0.3, 0.2) if (enforce_distance_filter and user_lat is not None and user_lng is not None)
                             else (0.6, 0.3, 0.1))
    candidates["final_score"] = w_sem*sim + w_auth*auth + w_prox*prox
    print(f"✅ Weights → semantic={w_sem}, rating={w_auth}, distance={w_prox}")

    ascending = (sentiment == "negative")
    print(f"🔃 Sorting {'ASC' if ascending else 'DESC'} by final_score (sentiment={sentiment})")
    result = candidates.sort_values("final_score", ascending=ascending).head(top_k)

    # 8) Final debug
    if len(result) == 0:
        print("❌ Still 0 results after backoff. Returning best-effort global rating slice.")
        # Absolute fallback: return global bottom/top by rating from df
        if sentiment == "negative":
            result = df.sort_values("rating", ascending=True).head(top_k)
        else:
            result = df.sort_values("rating", ascending=False).head(top_k)

    print(f"🏁 Final results: {len(result)}")
    print(f"   Ratings: {result['rating'].min()}–{result['rating'].max()}")
    if "distance_km" in result.columns:
        print(f"   Distances: {result['distance_km'].min()}–{result['distance_km'].max()}")

    return result[["name","cuisine","address","rating","distance_km","final_score"] if "final_score" in result.columns else ["name","cuisine","address","rating","distance_km"]]


if __name__ == "__main__":
    # Run standalone to precompute embeddings
    generate_embeddings_if_missing()
