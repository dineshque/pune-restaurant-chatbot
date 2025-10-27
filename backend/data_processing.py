"""
STEP 2 – DATA PROCESSING
-------------------------
Cleans and preprocesses the raw Pune restaurant data, normalizes cuisine names,
removes duplicates, validates ratings, and computes an authenticity score.
The final dataset is ready for Typesense indexing and embedding generation.
"""

import pandas as pd
import numpy as np
import re
import os

def clean_and_preprocess():
    raw_path = "data/raw/pune_restaurants_.csv"
    processed_path = "data/processed/pune_restaurants_cleaned.csv"
    os.makedirs("data/processed", exist_ok=True)

    #  Load Data
    
    df = pd.read_csv(raw_path)
    print(f"📂 Loaded {len(df)} raw records from {raw_path}")

    
    # Remove Duplicates
    
    before = len(df)
    df.drop_duplicates(subset=["name", "address"], inplace=True)
    after = len(df)
    print(f"🧹 Removed {before - after} duplicates — Remaining: {after}")

    
    # Validate and Normalize Ratings
    
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].clip(0, 5)
    df["rating"].fillna(df["rating"].median(), inplace=True)

    
    # Clean Text Fields
    
    def clean_text(t):
        if isinstance(t, str):
            t = re.sub(r"\s+", " ", t)
            return t.strip()
        return t

    for col in ["name", "address"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    
    # Handle Optional Fields
    
    for col in ["phone", "website"]:
        if col in df.columns:
            df[col].fillna("Not Available", inplace=True)

    
    # Keep Relevant Columns Only
    
    keep_cols = [
        "name", "cuisine", "address", "rating",
        "latitude", "longitude", "phone",
        "user_ratings_total","website"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    
    # Compute Authenticity Score
    
    if "user_ratings_total" in df.columns and "rating" in df.columns:
        C = df["rating"].mean()
        m = 100  # minimum reviews threshold for confidence
        df["authenticity_score"] = (
            (df["user_ratings_total"] / (df["user_ratings_total"] + m)) * df["rating"]
            + (m / (df["user_ratings_total"] + m)) * C
        ).clip(0, 5).round(2)
        print("✅ Authenticity scores computed successfully.")
    else:
        df["authenticity_score"] = df["rating"]
        print("⚠️ Could not compute authenticity scores (missing columns).")

    
    #  Normalize Column Names for Indexing
    
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    
    #  Save Clean Dataset
    
    df.to_csv(processed_path, index=False)
    print(f"✅ Cleaned dataset saved to {processed_path}")
    print(f"Final record count: {len(df)}")

    return df


if __name__ == "__main__":
    clean_and_preprocess()
