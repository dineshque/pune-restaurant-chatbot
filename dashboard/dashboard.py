# # dashboard/dashboard.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import os, psutil, torch
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import plotly.express as px

# # ---------------------------
# # ⚙️ Streamlit Config
# # ---------------------------
# st.set_page_config(page_title="Restaurant Chatbot Dashboard", layout="wide")

# st.title("🍽️ Pune Restaurant Chatbot — Admin Dashboard")

# # Auto-refresh every 60 seconds (live monitoring)
# st_autorefresh = st.sidebar.checkbox("Auto-refresh every 60 sec", value=True)
# if st_autorefresh:
#     st.experimental_rerun()

# # ---------------------------
# # 📂 Data Load Helpers
# # ---------------------------
# @st.cache_data
# def load_data():
#     logs_path = "data/processed/query_logs.csv"
#     feedback_path = "data/processed/feedback_log.csv"
#     restaurants_path = "data/processed/pune_restaurants_with_embeddings.csv"

#     df_logs = pd.read_csv(logs_path) if os.path.exists(logs_path) else pd.DataFrame()
#     df_fb = pd.read_csv(feedback_path) if os.path.exists(feedback_path) else pd.DataFrame()
#     df_rest = pd.read_csv(restaurants_path) if os.path.exists(restaurants_path) else pd.DataFrame()
#     return df_logs, df_fb, df_rest


# df_logs, df_fb, df_rest = load_data()

# # ---------------------------
# # 🧭 Sidebar Navigation
# # ---------------------------
# st.sidebar.title("📊 Dashboard Navigation")
# page = st.sidebar.radio(
#     "Go to section",
#     ["Overview", "User Queries", "Restaurant Insights", "Customer Feedback", "System Metrics"]
# )

# # ---------------------------
# # 🏠 OVERVIEW
# # ---------------------------
# if page == "Overview":
#     st.subheader("System Overview & Daily Insights")

#     total_queries = len(df_logs)
#     unique_users = len(df_logs["user_phone"].unique()) if "user_phone" in df_logs else 0
#     avg_latency = df_logs["latency"].mean() if "latency" in df_logs else 0

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total Queries", total_queries)
#     col2.metric("Unique Users", unique_users)
#     col3.metric("Avg Response Time", f"{avg_latency:.2f}s")

#     if not df_logs.empty:
#         df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")
#         df_daily = df_logs.groupby(df_logs["timestamp"].dt.date).size()
#         st.markdown("### 📅 Daily Query Volume")
#         st.line_chart(df_daily)

#         st.markdown("### 🔥 Top Cuisines Queried")
#         df_logs["query"] = df_logs["query"].astype(str)
#         cuisines = df_logs["query"].str.extractall(r"(?i)(chinese|south indian|italian|pizza|burger|north indian|biryani|cafe|mughlai)").value_counts().head(10)
#         st.bar_chart(cuisines)

#     else:
#         st.warning("No query log data available yet.")

# # ---------------------------
# # 💬 USER QUERIES
# # ---------------------------
# elif page == "User Queries":
#     st.subheader("Recent User Queries 📜")

#     if df_logs.empty:
#         st.warning("No query logs found.")
#     else:
#         df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")
#         df_logs = df_logs.sort_values("timestamp", ascending=False)
#         st.dataframe(df_logs.head(50), use_container_width=True)

#         st.markdown("### ⏱️ Response Time Trends")
#         if "latency" in df_logs:
#             st.line_chart(df_logs.tail(100)["latency"])
#         else:
#             st.info("Latency data unavailable.")

# # ---------------------------
# # 🍴 RESTAURANT INSIGHTS
# # ---------------------------
# elif page == "Restaurant Insights":
#     st.subheader("Restaurant Insights 🍴")

#     if df_rest.empty:
#         st.warning("No restaurant dataset found.")
#     else:
#         st.markdown("### 🌍 Geographic Heatmap of Queried Restaurants")

#         if {"latitude", "longitude"}.issubset(df_rest.columns):
#             # Create weighted heatmap
#             df_rest["weight"] = np.random.randint(1, 5, len(df_rest))  # demo weights
#             fig = px.density_mapbox(
#                 df_rest,
#                 lat="latitude",
#                 lon="longitude",
#                 z="weight",
#                 radius=15,
#                 center=dict(lat=18.52, lon=73.85),
#                 zoom=10,
#                 mapbox_style="carto-positron",
#                 title="Restaurant Density Across Pune"
#             )
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Latitude/Longitude columns missing for mapping.")

#         # Top Rated
#         st.markdown("### 🌟 Top Rated Restaurants")
#         top_rated = df_rest.sort_values("rating", ascending=False).head(10)
#         st.table(top_rated[["name", "cuisine", "rating", "address"]])

#         st.markdown("### 🍽️ Cuisine Distribution")
#         st.bar_chart(df_rest["cuisine"].value_counts().head(10))

# # ---------------------------
# # 💬 CUSTOMER FEEDBACK
# # ---------------------------
# elif page == "Customer Feedback":
#     st.subheader("Customer Feedback 💬")

#     if df_fb.empty:
#         st.warning("No feedback data found.")
#     else:
#         st.dataframe(df_fb.tail(30), use_container_width=True)

#         avg_rating = (
#             df_fb["feedback"].str.extract(r"(\d)").dropna().astype(float).mean()[0]
#             if not df_fb.empty else np.nan
#         )
#         st.metric("Average Rating", f"{avg_rating:.2f} ⭐")

#         st.markdown("### 📈 Rating Trends")
#         rating_trend = (
#             df_fb["feedback"].str.extract(r"(\d)").dropna().astype(float).rename(columns={0: "rating"})
#         )
#         st.line_chart(rating_trend)

#         # Word cloud
#         st.markdown("### ☁️ Feedback Word Cloud")
#         text = " ".join(df_fb["feedback"].dropna().astype(str))
#         wc = WordCloud(width=800, height=300, background_color="white").generate(text)
#         fig, ax = plt.subplots()
#         ax.imshow(wc, interpolation="bilinear")
#         ax.axis("off")
#         st.pyplot(fig)

# # ---------------------------
# # ⚙️ SYSTEM METRICS
# # ---------------------------
# elif page == "System Metrics":
#     st.subheader("System Health & Performance ⚙️")

#     col1, col2, col3 = st.columns(3)
#     col1.metric("CPU Usage", f"{psutil.cpu_percent()}%")
#     col2.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
#     col3.metric("GPU Available", torch.cuda.is_available())

#     st.markdown("### 🧠 Model & Embedding Info")
#     st.text("Embedding Model: thenlper/gte-large")
#     st.text("LLM: Groq Llama 3.1 8B Instant")
#     st.text("Dataset Size: 2000+ Restaurants")
#     st.text("Embeddings Cached: ✅")

#     st.markdown("### 📊 System Load Trends (Live)")
#     cpu_load = [psutil.cpu_percent(interval=0.5) for _ in range(20)]
#     st.line_chart(cpu_load)

# # ---------------------------
# # 📅 Footer
# # ---------------------------
# st.sidebar.markdown("---")
# st.sidebar.markdown("Developed by **Digitree Labs** © 2025")
# st.sidebar.info("Monitor chatbot usage, user feedback, and restaurant insights in real-time.")
# dashboard/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import psutil
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------------------
# ⚙️ Streamlit Configuration
# ---------------------------
st.set_page_config(page_title="Restaurant Chatbot Dashboard", layout="wide")
st.title("🍽️ Pune Restaurant Chatbot — Admin Dashboard")
# Auto-refresh every 10 seconds
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=10 * 1000, key="refresh")

# ---------------------------
# 📁 Data Load Helpers with Validation
# ---------------------------
@st.cache_data
def load_data():
    """Loads and validates required data files."""
    paths = {
        "logs": "data/processed/query_logs.csv",
        "feedback": "data/processed/feedback_log.csv",
        "restaurants": "data/processed/pune_restaurants_with_embeddings.csv"
    }
    dfs = {}
    for key, path in paths.items():
        if os.path.exists(path):
            try:
                dfs[key] = pd.read_csv(path)
            except Exception as e:
                st.error(f"Error loading {key} data: {e}")
                dfs[key] = pd.DataFrame()
        else:
            dfs[key] = pd.DataFrame()
    return dfs["logs"], dfs["feedback"], dfs["restaurants"]

df_logs, df_fb, df_rest = load_data()

# ---------------------------
# 🧭 Sidebar Navigation
# ---------------------------
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio(
    "Go to section",
    ["Overview", "User Queries", "Restaurant Insights", "Customer Feedback", "System Metrics"]
)

# ---------------------------
# 🏠 OVERVIEW
# ---------------------------
if page == "Overview":
    st.subheader("System Overview & Daily Insights")

    total_queries = len(df_logs)
    unique_users = len(df_logs["user_phone"].unique()) if "user_phone" in df_logs else 0
    avg_latency = df_logs["latency"].mean() if "latency" in df_logs else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", total_queries)
    col2.metric("Unique Users", unique_users)
    col3.metric("Avg Response Time", f"{avg_latency:.2f}s")

    if not df_logs.empty:
        # Parse timestamps
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")
        # Group by date for daily queries
        df_daily = df_logs["timestamp"].dt.date.value_counts().sort_index()
        st.markdown("### 📅 Daily Query Volume")
        st.line_chart(df_daily)

        # Extract top cuisines
        st.markdown("### 🔥 Top Cuisines Queried")
        cuisine_regex = r"(?i)(chinese|south indian|italian|pizza|burger|north indian|biryani|cafe|mughlai)"
        cuisine_hits = df_logs["query"].str.extractall(cuisine_regex).value_counts().head(10)
        
        df_cuisine_hits = cuisine_hits.reset_index()
        df_cuisine_hits.columns = ["Cuisine", "Count"]
        st.bar_chart(df_cuisine_hits.set_index("Cuisine"))

        # st.bar_chart(cuisine_hits.set_index("Cuisine"))  # Wrong: Series has no set_index
    else:
        st.warning("No query log data available yet.")

# ---------------------------
# 💬 USER QUERIES
# ---------------------------
elif page == "User Queries":
    st.subheader("Recent User Queries 📜")

    if df_logs.empty:
        st.warning("No query logs found.")
    else:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")
        df_logs = df_logs.sort_values("timestamp", ascending=False)
        st.dataframe(df_logs.head(50), use_container_width=True)

        st.markdown("### ⏱️ Response Time Trends")
        if "latency" in df_logs:
            latency_plot = df_logs.tail(100)[["timestamp", "latency"]].set_index("timestamp")
            st.line_chart(latency_plot)
        else:
            st.info("Latency data unavailable.")

# ---------------------------
# 🍴 RESTAURANT INSIGHTS
# ---------------------------
elif page == "Restaurant Insights":
    st.subheader("Restaurant Insights 🍴")

    if df_rest.empty:
        st.warning("No restaurant dataset found.")
    else:
        # Geographic heatmap
        st.markdown("### 🌍 Geographic Heatmap of Queried Restaurants")
        if set(["latitude", "longitude"]).issubset(df_rest.columns):
            df_rest["weight"] = np.random.randint(1, 5, len(df_rest))
            fig = px.density_mapbox(
                df_rest,
                lat="latitude",
                lon="longitude",
                z="weight",
                radius=15,
                center=dict(lat=18.52, lon=73.85),
                zoom=10,
                mapbox_style="carto-positron",
                title="Restaurant Density Across Pune"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Latitude/Longitude columns missing for mapping.")

        # Top Rated Restaurants
        st.markdown("### 🌟 Top Rated Restaurants")
        top_rated = df_rest.sort_values("rating", ascending=False).head(10)
        st.table(top_rated[["name", "cuisine", "rating", "address"]])

        # Cuisine Distribution
        st.markdown("### 🍽️ Cuisine Distribution")
        st.bar_chart(df_rest["cuisine"].value_counts().head(10))

# ---------------------------
# 💬 CUSTOMER FEEDBACK
# ---------------------------
elif page == "Customer Feedback":
    st.subheader("Customer Feedback 💬")

    if df_fb.empty:
        st.warning("No feedback data found.")
    else:
        st.dataframe(df_fb.tail(30), use_container_width=True)

        # Safely extract ratings
        rating_series = df_fb["feedback"].str.extract(r"(\d(?:\.\d)?)")[0].dropna().astype(float)
        avg_rating = rating_series.mean() if not rating_series.empty else np.nan
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")

        st.markdown("### 📈 Rating Trends")
        st.line_chart(rating_series)

        # Word cloud
        st.markdown("### ☁️ Feedback Word Cloud")
        text = " ".join(df_fb["feedback"].dropna().astype(str))
        wc = WordCloud(width=800, height=300, background_color="white").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# ---------------------------
# ⚙️ SYSTEM METRICS
# ---------------------------
elif page == "System Metrics":
    st.subheader("System Health & Performance ⚙️")

    col1, col2, col3 = st.columns(3)
    # Protect against environments without torch or GPU
    gpu_status = torch.cuda.is_available() if hasattr(torch, "cuda") else False
    col1.metric("CPU Usage", f"{psutil.cpu_percent()}%")
    col2.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
    col3.metric("GPU Available", "✅" if gpu_status else "❌")

    st.markdown("### 🧠 Model & Embedding Info")
    st.text("Embedding Model: thenlper/gte-large")
    st.text("LLM: Groq Llama 3.1 8B Instant")
    st.text("Dataset Size: 2000+ Restaurants")
    st.text("Embeddings Cached: ✅")

    st.markdown("### 📊 System Load Trends (Live)")
    cpu_load = [psutil.cpu_percent(interval=0.5) for _ in range(20)]
    st.line_chart(cpu_load)

# ---------------------------
# 📅 Footer
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.info("Monitor chatbot usage, user feedback, and restaurant insights in real-time.")
