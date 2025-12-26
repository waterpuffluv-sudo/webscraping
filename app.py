import json
import os

import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Brand Reputation Monitor", layout="wide")

@st.cache_data
def load_json(path: str):
    """Load JSON from disk with a friendly error if missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def get_sentiment():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

st.title("E-commerce Brand Reputation Monitor (2023)")
st.caption("Data source: web-scraping.dev — Products / Testimonials / Reviews")

products = load_json("products.json")
testimonials = load_json("testimonials.json")
reviews = load_json("reviews.json")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

if page == "Products":
    st.header("Products")
    df = pd.DataFrame(products)
    if df.empty:
        st.warning("products.json is empty. Run scraper.py first.")
    else:
        st.dataframe(df, use_container_width=True)

elif page == "Testimonials":
    st.header("Testimonials")
    df = pd.DataFrame(testimonials)
    if df.empty:
        st.warning("testimonials.json is empty. Run scraper.py first.")
    else:
        st.dataframe(df, use_container_width=True)

else:
    st.header("Reviews — Sentiment Analysis")

    df = pd.DataFrame(reviews)
    if df.empty:
        st.warning("reviews.json is empty. Run scraper.py first.")
        st.stop()

    # Expect date as YYYY-MM-DD from scraper
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Month selector for 2023 only
    months = pd.period_range("2023-01", "2023-12", freq="M")
    month_labels = [m.strftime("%b %Y") for m in months]
    chosen_label = st.select_slider("Select month (2023)", options=month_labels, value=month_labels[0])
    chosen_period = months[month_labels.index(chosen_label)]

    df["YearMonth"] = df["date"].dt.to_period("M")
    filtered = df[(df["YearMonth"] == chosen_period)].copy()

    if filtered.empty:
        st.info(f"No reviews found for {chosen_label}.")
        st.stop()

    with st.spinner("Running sentiment analysis (HuggingFace Transformers)..."):
        clf = get_sentiment()
        texts = filtered["text"].astype(str).tolist()
        preds = clf(texts)

    filtered["sentiment"] = [p["label"] for p in preds]      # POSITIVE / NEGATIVE
    filtered["confidence"] = [float(p["score"]) for p in preds]

    # Counts + avg confidence
    counts = filtered["sentiment"].value_counts().reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
    avg_conf = filtered["confidence"].mean()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Sentiment distribution")
        chart_df = (
            counts.rename("count")
            .reset_index()
            .rename(columns={"index": "sentiment"})
        )
        chart_df["avg_confidence"] = float(avg_conf)

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("sentiment:N", title="Sentiment"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("sentiment:N", title="Sentiment"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("avg_confidence:Q", title="Avg. confidence", format=".2%"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Avg. confidence for {chosen_label}: {avg_conf:.2%}")

    with col2:
        st.subheader("Metrics")
        st.metric("Total reviews", len(filtered))
        st.metric("Positive", int(counts["POSITIVE"]))
        st.metric("Negative", int(counts["NEGATIVE"]))
        st.metric("Avg. confidence", f"{avg_conf:.2%}")

    st.subheader("Filtered reviews")
    show_cols = ["date", "text", "sentiment", "confidence"]
    st.dataframe(filtered[show_cols].sort_values("date", ascending=False), use_container_width=True)
