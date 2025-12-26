import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline

# ---- CPU/Render friendly ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")

# -----------------------------
# Helpers: load JSON
# -----------------------------
@st.cache_data
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_load_json(path: str):
    try:
        return load_json(path)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return []

# -----------------------------
# Sentiment pipeline (lazy load)
# -----------------------------
@st.cache_resource
def get_sentiment_pipeline():
    # Loads only when called
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU
    )

def run_sentiment(texts, batch_size: int = 16):
    clf = get_sentiment_pipeline()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        out.extend(clf(chunk, truncation=True))
    return out

# -----------------------------
# Load data files
# -----------------------------
products = safe_load_json("products.json")
testimonials = safe_load_json("testimonials.json")
reviews = safe_load_json("reviews.json")

products_df = pd.DataFrame(products)
testimonials_df = pd.DataFrame(testimonials)
reviews_df = pd.DataFrame(reviews)

# -----------------------------
# UI: header + nav
# -----------------------------
st.title("E-commerce Brand Reputation Monitor (2023)")
st.caption("Scraped from web-scraping.dev (Products / Testimonials / Reviews) + Sentiment Analysis")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

# -----------------------------
# PAGE: Products
# -----------------------------
if page == "Products":
    st.header("Products")
    if products_df.empty:
        st.warning("products.json is empty/missing. Run `python scraper.py` locally/GitHub Actions and commit the JSON.")
    else:
        st.dataframe(products_df, use_container_width=True)

# -----------------------------
# PAGE: Testimonials
# -----------------------------
elif page == "Testimonials":
    st.header("Testimonials")
    if testimonials_df.empty:
        st.warning("testimonials.json is empty/missing. Run `python scraper.py` locally/GitHub Actions and commit the JSON.")
    else:
        st.dataframe(testimonials_df, use_container_width=True)

# -----------------------------
# PAGE: Reviews (core feature)
# -----------------------------
else:
    st.header("Reviews — Sentiment Analysis")

    if reviews_df.empty:
        st.warning("reviews.json is empty/missing. Run `python scraper.py` and commit the JSON.")
        st.stop()

    # Normalize columns to expected names
    if "text" not in reviews_df.columns and "review_text" in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={"review_text": "text"})

    if "date" not in reviews_df.columns or "text" not in reviews_df.columns:
        st.error("reviews.json must contain fields: 'date' and 'text'.")
        st.stop()

    # Parse dates safely
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")
    reviews_df = reviews_df.dropna(subset=["date"]).copy()
    reviews_df["text"] = reviews_df["text"].astype(str)

    # Month selector (strict 2023)
    months = [datetime(2023, m, 1).strftime("%b %Y") for m in range(1, 13)]
    chosen_label = st.select_slider("Select a month (2023)", options=months, value="Jan 2023")
    chosen_month = months.index(chosen_label) + 1

    filtered = reviews_df[
        (reviews_df["date"].dt.year == 2023) &
        (reviews_df["date"].dt.month == chosen_month)
    ].copy()

    if filtered.empty:
        st.info(f"No reviews found for {chosen_label}.")
        st.stop()

    # Memory-safe limit (Render free tier friendly)
    st.sidebar.subheader("Reviews limits")
    max_n = st.sidebar.slider("Max reviews to analyze", min_value=20, max_value=200, value=80, step=10)

    if len(filtered) > max_n:
        st.warning(
            f"This month has {len(filtered)} reviews. "
            f"To avoid Render memory crashes, analyzing first {max_n}. "
            f"(Increase limit in sidebar if you upgraded RAM.)"
        )
        filtered = filtered.head(max_n).copy()

    # Optional: prevent auto-run on every slider move
    auto_run = st.sidebar.checkbox("Auto-run sentiment", value=True)
    run_now = True
    if not auto_run:
        run_now = st.button("Run sentiment analysis")

    if not run_now:
        st.info("Sentiment analysis is paused. Enable Auto-run or press the button.")
        st.dataframe(filtered[["date", "text"]].sort_values("date", ascending=False), use_container_width=True)
        st.stop()

    # Sentiment analysis
    with st.spinner("Running HuggingFace sentiment analysis... (first run may take longer)"):
        texts = filtered["text"].tolist()
        preds = run_sentiment(texts, batch_size=16)

    filtered["sentiment"] = [p.get("label") for p in preds]  # POSITIVE / NEGATIVE
    filtered["confidence"] = [float(p.get("score", 0.0)) for p in preds]

    # Visualization: counts + avg confidence
    counts = (
        filtered["sentiment"]
        .value_counts()
        .reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
    )
    avg_conf = float(filtered["confidence"].mean()) if len(filtered) else 0.0

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sentiment Distribution")
        st.bar_chart(counts)

    with col2:
        st.subheader("Metrics")
        st.metric("Total Reviews Analyzed", int(len(filtered)))
        st.metric("Positive Reviews", int(counts.get("POSITIVE", 0)))
        st.metric("Negative Reviews", int(counts.get("NEGATIVE", 0)))
        st.metric("Avg. Confidence Score", f"{avg_conf:.2%}")

    # Table with predictions
    st.subheader("Filtered Reviews & Analysis")
    st.dataframe(
        filtered[["date", "text", "sentiment", "confidence"]]
        .sort_values("date", ascending=False),
        use_container_width=True
    )
