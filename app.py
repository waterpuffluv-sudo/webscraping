import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline

# ---- Render/CPU-friendly env ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load(path: str):
    try:
        return load_json(path)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return []


# -----------------------------
# Sentiment model (lazy)
# -----------------------------
@st.cache_resource
def get_sentiment_pipeline():
    # IMPORTANT: model loads only when this function is called
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU
    )


def run_sentiment(texts, batch_size: int = 16):
    clf = get_sentiment_pipeline()
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        # truncation avoids very long texts causing extra memory usage
        preds.extend(clf(chunk, truncation=True))
    return preds


# -----------------------------
# UI
# -----------------------------
st.title("E-commerce Brand Reputation Monitor (2023)")
st.caption("Scraped from web-scraping.dev (Products / Testimonials / Reviews) + Sentiment Analysis")

products = safe_load("products.json")
testimonials = safe_load("testimonials.json")
reviews = safe_load("reviews.json")

products_df = pd.DataFrame(products)
testimonials_df = pd.DataFrame(testimonials)
reviews_df = pd.DataFrame(reviews)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])


# -----------------------------
# Pages
# -----------------------------
if page == "Products":
    st.header("Products")
    if products_df.empty:
        st.warning("products.json is empty or missing. Run `python scraper.py` first and push JSON to GitHub.")
    else:
        st.dataframe(products_df, use_container_width=True)

elif page == "Testimonials":
    st.header("Testimonials")
    if testimonials_df.empty:
        st.warning("testimonials.json is empty or missing. Run `python scraper.py` first and push JSON to GitHub.")
    else:
        st.dataframe(testimonials_df, use_container_width=True)

else:
    st.header("Reviews — Sentiment Analysis")

    if reviews_df.empty:
        st.warning("reviews.json is empty or missing. Run `python scraper.py` first and push JSON to GitHub.")
        st.stop()

    # Normalize columns
    if "text" not in reviews_df.columns and "review_text" in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={"review_text": "text"})

    if "date" not in reviews_df.columns or "text" not in reviews_df.columns:
        st.error("reviews.json must contain 'date' and 'text' fields.")
        st.stop()

    # Parse date
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")
    reviews_df = reviews_df.dropna(subset=["date"])
    reviews_df["text"] = reviews_df["text"].astype(str)

    # Month selector (strictly 2023)
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

    # Memory protection: limit max reviews per run
    st.subheader(f"Filtered reviews for {chosen_label}")
    max_n = st.sidebar.slider("Max reviews to analyze (memory-safe)", 20, 200, 80, step=10)

    if len(filtered) > max_n:
        st.warning(f"Too many reviews for this month ({len(filtered)}). "
                   f"To avoid Render memory crashes, analyzing first {max_n}. "
                   f"(You can increase the limit in sidebar.)")
        filtered = filtered.head(max_n).copy()

    # Optional manual trigger to prevent accidental OOM on first load
    auto_run = st.sidebar.checkbox("Auto-run sentiment", value=True)

    run_now = True
    if not auto_run:
        run_now = st.button("Run sentiment analysis")

    if run_now:
        with st.spinner("Running HuggingFace sentiment analysis..."):
            try:
                texts = filtered["text"].tolist()
                preds = run_sentiment(texts, batch_size=16)
            except Exception as e:
                st.error(f"Sentiment analysis failed (likely memory/torch issue): {e}")
                st.stop()

        filtered["sentiment"] = [p.get("label") for p in preds]          # POSITIVE / NEGATIVE
        filtered["confidence"] = [float(p.get("score", 0.0)) for p in preds]

        # Visualization
        counts = filtered["sentiment"].value_counts().reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
        avg_conf = float(filtered["confidence"].mean()) if len(filtered) else 0.0

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Sentiment Distribution")
            st.bar_chart(counts)

        with col2:
            st.subheader("Metrics")
            st.metric("Total analyzed", int(len(filtered)))
            st.metric("Positive", int(counts.get("POSITIVE", 0)))
            st.metric("Negative", int(counts.get("NEGATIVE", 0)))
            st.metric("Avg. confidence", f"{avg_conf:.2%}")

        st.subheader("Reviews + Predictions")
        st.dataframe(
            filtered[["date", "text", "sentiment", "confidence"]].sort_values("date", ascending=False),
            use_container_width=True
        )

    else:
        st.info("Sentiment analysis is paused. Enable 'Auto-run sentiment' or press the button.")
        st.dataframe(filtered[["date", "text"]].sort_values("date", ascending=False), use_container_width=True)

