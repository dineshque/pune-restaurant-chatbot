import typesense
import pandas as pd
import os

def create_client():
    """
    Connect to your Typesense cluster.
    Replace host and api_key with your actual credentials.
    """
    return typesense.Client({
        'nodes': [{
            'host': 'iad9pvfewscx5b04p-1.a1.typesense.net',  
            'port': '443',
            'protocol': 'https'
        }],
        'api_key': os.getenv("TYPESENSE_API_KEY") or "KHLMZS7UIoLIlTJeA05yhvQE79DXBaLG" ,
        'connection_timeout_seconds': 10
    })

def create_schema(client):
    schema = {
        "name": "restaurants",
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "name", "type": "string"},
            {"name": "cuisine", "type": "string"},              # comma-separated is fine
            {"name": "address", "type": "string"},
            {"name": "rating", "type": "float"},
            {"name": "authenticity_score", "type": "float"},    # ✅ new
            {"name": "user_ratings_total", "type": "int32"},
            {"name": "price_level", "type": "int32"},
            {"name": "latitude", "type": "float"},
            {"name": "longitude", "type": "float"},
            {"name": "phone", "type": "string", "optional": True},
            {"name": "website", "type": "string", "optional": True},
            {"name": "summary_text", "type": "string"},
        ],
        # keep rating; we’ll sort/score in hybrid layer anyway
        "default_sorting_field": "rating"
    }
    try:
        client.collections.create(schema)
        print(" Collection created.")
    except Exception as e:
        print(" Schema create skipped (probably exists):", e)

def index_data(client):
    df = pd.read_csv("data/processed/pune_restaurants_cleaned.csv")
    df["summary_text"] = (
        df["name"].fillna("") + " " +
        df["cuisine"].fillna("") + " restaurant near " +
        df["address"].fillna("") + " rated " + df["rating"].astype(str)
    )
    df["id"] = df.index.astype(str)
    records = df.to_dict(orient="records")
    client.collections["restaurants"].documents.import_(records, {'action':'upsert'})
    print(f" {len(df)} docs indexed.")

if __name__ == "__main__":
    c = create_client()
    create_schema(c)
    index_data(c)