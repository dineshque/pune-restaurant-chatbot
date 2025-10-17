# backend/llm_query_refiner.py
from openai import OpenAI

def refine_query(user_query, api_key):
    """
    Use LLM to clarify or reformulate vague queries.
    """
    client = OpenAI(api_key=api_key)
    prompt = f"""
    You are a restaurant search assistant for Pune.
    Rewrite this user query into a clear search description:
    '{user_query}'
    Example output: 'Chinese restaurants with rating above 4.0 in Pune'
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
