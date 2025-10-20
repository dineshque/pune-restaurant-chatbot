# # test_search.py
# import pandas as pd
# from backend.reranker import rerank

# from backend.semantic_search import load_df_with_embeddings, hybrid_rank

# df = load_df_with_embeddings()
# query = "worst restaurant in pune"
# hybrid = hybrid_rank(query, df, user_lat=18.59409, user_lng=73.7253163, top_k=10)
# final = rerank(query, hybrid, out_k=5)
# print(final[["name","cuisine","rating","final_score"]])

# test_query.py
# from backend.semantic_search import load_df_with_embeddings, hybrid_rank
# df = load_df_with_embeddings()
# print(hybrid_rank("worst restaurant in pune", df))
# print(hybrid_rank("best chinese in pune", df))
from backend.semantic_search import hybrid_rank, load_df_with_embeddings
import os

df = load_df_with_embeddings()

groq_api_key = os.getenv("GROQ_API_KEY")  # export before running

queries = [
    "best chinese restaurants near me",
    "worst restaurant in Pune",
    "restaurants within 3 km open now",
    "cheap south indian for dinner near FC Road",
]

for q in queries:
    print(f"\n🔍 Query: {q}")
    res = hybrid_rank(q, df, user_lat=18.5204, user_lng=73.8567, groq_api_key=groq_api_key)
    print(res.head(5))
