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

import os

# ✅ Disable TensorFlow backend for SentenceTransformers
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TF logs if it tries to load

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Load model globally once (PyTorch backend only)
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings():
    """
    Create and store embeddings for semantic search.
    """
    df = pd.read_csv("data/processed/pune_restaurants_cleaned.csv")

    # Prepare unified text column for embeddings
    df["summary_text"] = (
        df["name"].fillna("") + " " +
        df["cuisine"].fillna("") + " " +
        df["address"].fillna("") + " " +
        df["rating"].astype(str)
    )

    print("🔹 Generating embeddings...")
    df["embedding"] = df["summary_text"].apply(
        lambda x: model.encode(x, convert_to_tensor=True)
    )

    # Save embeddings
    torch.save(df["embedding"].tolist(), "data/processed/embeddings.pt")
    df.to_csv("data/processed/pune_restaurants_with_embeddings.csv", index=False)
    print(f"✅ Embeddings generated and saved for {len(df)} restaurants.")
    return df


def semantic_search(user_query, df, top_k=5):
    """
    Perform semantic similarity search using cosine similarity.
    """
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    df["similarity"] = df["embedding"].apply(lambda x: float(util.cos_sim(query_embedding, x)))
    results = df.sort_values("similarity", ascending=False).head(top_k)
    return results[["name", "cuisine", "address", "rating", "similarity"]]


if __name__ == "__main__":
    generate_embeddings()
