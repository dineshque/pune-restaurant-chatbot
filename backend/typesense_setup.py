# backend/typesense_setup.py
import typesense
import pandas as pd

def create_client():
    """
    Connect to your Typesense cluster.
    Replace host and api_key with your actual credentials.
    """
    return typesense.Client({
        'nodes': [{
            'host': 'iad9pvfewscx5b04p-1.a1.typesense.net',   # from your Typesense dashboard
            'port': '443',
            'protocol': 'https'
        }],
        'api_key': 'KHLMZS7UIoLIlTJeA05yhvQE79DXBaLG',
        'connection_timeout_seconds': 10
    })

def create_schema(client):
    """
    Define the schema for restaurant collection.
    """
    schema = {
        "name": "restaurants",
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "name", "type": "string"},
            {"name": "cuisine", "type": "string"},
            {"name": "address", "type": "string"},
            {"name": "rating", "type": "float"},
            {"name": "user_ratings_total", "type": "int32"},
            {"name": "price_level", "type": "int32"},
            {"name": "latitude", "type": "float"},
            {"name": "longitude", "type": "float"},
            {"name": "phone", "type": "string", "optional": True},
            {"name": "website", "type": "string", "optional": True},
            {"name": "summary_text", "type": "string"},
        ],
        "default_sorting_field": "rating"
    }

    try:
        client.collections.create(schema)
        print("✅ Typesense collection created successfully.")
    except Exception as e:
        print("⚠️ Collection may already exist:", e)

def index_data(client):
    """
    Load cleaned CSV and index into Typesense.
    """
    df = pd.read_csv("data/processed/pune_restaurants_cleaned.csv")

    # Create a searchable summary text
    df["summary_text"] = (
        df["name"].fillna("") + " " +
        df["cuisine"].fillna("") + " " +
        df["address"].fillna("") + " " +
        df["rating"].astype(str)
    )

    # Prepare records
    df["id"] = df.index.astype(str)
    records = df.to_dict(orient="records")

    # Upload data
    client.collections["restaurants"].documents.import_(records, {'action': 'upsert'})
    print(f"✅ {len(df)} records indexed into Typesense.")

if __name__ == "__main__":
    client = create_client()
    create_schema(client)
    index_data(client)
