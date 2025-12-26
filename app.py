import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")


@st.cache_data
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_sentiment_pipeline():
    # требование: HuggingFace transformers pipeline + пример SST-2
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def safe_load(path: str):
    return load_json(path) if os.path.exists(path) else []


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


if page == "Products":
    st.header("Products")
    if products_df.empty:
        st.warning("products.json is empty or missing. Run `python scraper.py` first.")
    else:
        st.dataframe(products_df, use_container_width=True)

elif page == "Testimonials":
    st.header("Testimonials")
    if testimonials_df.empty:
        st.warning("testimonials.json is empty or missing. Run `python scraper.py` first.")
    else:
        st.dataframe(testimonials_df, use_container_width=True)

else:
    st.header("Reviews — Sentiment Analysis")

    if reviews_df.empty:
        st.warning("reviews.json is empty or missing. Run `python scraper.py` first.")
        st.stop()

    # date -> datetime
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")
    reviews_df = reviews_df.dropna(subset=["date"])

    # Month selector for 2023 only
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

    with st.spinner("Running HuggingFace sentiment analysis..."):
        clf = get_sentiment_pipeline()
        texts = filtered["text"].astype(str).tolist()
        preds = clf(texts)

    filtered["sentiment"] = [p["label"] for p in preds]          # POSITIVE / NEGATIVE
    filtered["confidence"] = [float(p["score"]) for p in preds]  # confidence score

    counts = filtered["sentiment"].value_counts().reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
    avg_conf = float(filtered["confidence"].mean())

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Sentiment Distribution")
        st.bar_chart(counts)

    with col2:
        st.subheader("Metrics")
        st.metric("Total reviews", int(len(filtered)))
        st.metric("Positive", int(counts["POSITIVE"]))
        st.metric("Negative", int(counts["NEGATIVE"]))
        st.metric("Avg. confidence", f"{avg_conf:.2%}")

    st.subheader("Filtered Reviews")
    st.dataframe(
        filtered[["date", "text", "sentiment", "confidence"]].sort_values("date", ascending=False),
        use_container_width=True
    )
