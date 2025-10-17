# embeddings/embed_utils.py
from sentence_transformers import SentenceTransformer

def load_model():
    """
    Loads and returns a pre-trained SentenceTransformer model.
    """
    print("🔹 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded.")
    return model
