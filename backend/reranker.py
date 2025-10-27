# # backend/reranker.py
# from sentence_transformers import CrossEncoder
# import pandas as pd

# # Small, fast, decent quality
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# _reranker = CrossEncoder(RERANK_MODEL)

# def rerank(query:str, df_topk:pd.DataFrame, text_col:str="summary_text", out_k:int=5):
#     if text_col not in df_topk.columns:
#         # build an ad-hoc summary if not present
#         df_topk = df_topk.copy()
#         df_topk[text_col] = (
#             df_topk["name"].fillna("") + " " +
#             df_topk["cuisine"].fillna("") + " restaurant near " +
#             df_topk["address"].fillna("") + " rated " +
#             df_topk["rating"].astype(str)
#         )
#     pairs = [(query, s) for s in df_topk[text_col].tolist()]
#     scores = _reranker.predict(pairs)
#     df_topk = df_topk.copy()
#     df_topk["rerank_score"] = scores
#     return df_topk.sort_values("rerank_score", ascending=False).head(out_k)
