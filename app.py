import json
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")

DATA_DIR = Path(".")  # если сделаешь папку data/, поменяй на Path("data")

PRODUCTS_PATH = DATA_DIR / "products.json"
TESTIMONIALS_PATH = DATA_DIR / "testimonials.json"
REVIEWS_PATH = DATA_DIR / "reviews.json"

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_products(raw):
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    # подстраховка под разные ключи
    rename_map = {}
    if "title" in df.columns and "name" not in df.columns:
        rename_map["title"] = "name"
    if "Product Name" in df.columns:
        rename_map["Product Name"] = "name"
    if "Description" in df.columns:
        rename_map["Description"] = "description"
    df = df.rename(columns=rename_map)
    return df

def normalize_testimonials(raw):
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    rename_map = {}
    if "Testimonial" in df.columns:
        rename_map["Testimonial"] = "text"
    if "Author" in df.columns:
        rename_map["Author"] = "author"
    df = df.rename(columns=rename_map)
    return df

def parse_review_date(x):
    """
    Поддержка:
    - '2023-04-25'
    - 'Reviewed on Apr 25 2023'
    - 'Apr 25 2023'
    """
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    s = s.replace("Reviewed on", "").replace("reviewed on", "").replace("on", "").strip()
    return pd.to_datetime(s, errors="coerce")

def normalize_reviews(raw):
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    # под разные ключи
    rename_map = {}
    if "review_text" in df.columns and "text" not in df.columns:
        rename_map["review_text"] = "text"
    if "date" not in df.columns and "review_date" in df.columns:
        rename_map["review_date"] = "date"
    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = df["date"].apply(parse_review_date)
    else:
        df["date"] = pd.NaT

    if "text" not in df.columns:
        df["text"] = ""

    return df

@st.cache_resource
def get_sentiment_pipe():
    # фиксируем модель (как в требованиях)
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

def run_sentiment(df: pd.DataFrame):
    if df.empty:
        return df
    pipe = get_sentiment_pipe()
    texts = df["text"].astype(str).tolist()
    results = pipe(texts)
    df = df.copy()
    df["sentiment"] = [r["label"] for r in results]
    df["confidence"] = [float(r["score"]) for r in results]
    return df

# ----------------------------
# Load data
# ----------------------------
products_raw = load_json(PRODUCTS_PATH)
testimonials_raw = load_json(TESTIMONIALS_PATH)
reviews_raw = load_json(REVIEWS_PATH)

products_df = normalize_products(products_raw)
testimonials_df = normalize_testimonials(testimonials_raw)
reviews_df = normalize_reviews(reviews_raw)

# ----------------------------
# UI
# ----------------------------
st.title("E-commerce Brand Reputation Monitor (2023)")
st.caption("Data source: web-scraping.dev (Products / Testimonials / Reviews) + HuggingFace sentiment analysis")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"], index=2)

if page == "Products":
    st.header("Products")
    if products_df.empty:
        st.warning("products.json is empty or missing.")
    else:
        st.dataframe(products_df, use_container_width=True)

elif page == "Testimonials":
    st.header("Testimonials")
    if testimonials_df.empty:
        st.warning("testimonials.json is empty or missing.")
    else:
        st.dataframe(testimonials_df, use_container_width=True)

else:
    st.header("Reviews — Sentiment Analysis")

    if reviews_df.empty:
        st.warning("reviews.json is empty or missing.")
        st.stop()

    # только 2023
    reviews_2023 = reviews_df.dropna(subset=["date"]).copy()
    reviews_2023 = reviews_2023[reviews_2023["date"].dt.year == 2023].copy()

    if reviews_2023.empty:
        st.warning("No valid 2023 dates found in reviews.json.")
        st.stop()

    # Month selector (Jan-Dec 2023)
    months = pd.period_range("2023-01", "2023-12", freq="M")
    month_labels = [m.strftime("%b %Y") for m in months]

    selected_label = st.select_slider("Select a month (2023)", options=month_labels, value=month_labels[0])
    selected_period = pd.Period(selected_label, freq="M")

    filtered = reviews_2023[reviews_2023["date"].dt.to_period("M") == selected_period].copy()

    # controls to reduce memory/compute
    st.sidebar.subheader("Reviews limits")
    max_n = st.sidebar.slider("Max reviews to analyze", 5, 60, 20, step=5)
    auto_run = st.sidebar.checkbox("Auto-run sentiment", value=True)

    if filtered.empty:
        st.info(f"No reviews found for {selected_label}.")
        st.stop()

    filtered = filtered.sort_values("date", ascending=False).head(max_n)

    if auto_run:
        with st.spinner("Running HuggingFace sentiment analysis..."):
            analyzed = run_sentiment(filtered)
    else:
        if st.button("Run sentiment analysis"):
            with st.spinner("Running HuggingFace sentiment analysis..."):
                analyzed = run_sentiment(filtered)
        else:
            analyzed = filtered.copy()

    # If sentiment not computed yet
    if "sentiment" not in analyzed.columns:
        st.subheader("Filtered reviews (no sentiment yet)")
        st.dataframe(analyzed[["date", "text"]], use_container_width=True)
        st.stop()

    # Metrics + chart
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Metrics")
        st.metric("Reviews analyzed", len(analyzed))
        st.metric("Avg confidence", f"{analyzed['confidence'].mean():.2%}")

    with col2:
        st.subheader("Positive vs Negative")
        counts = analyzed["sentiment"].value_counts().reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
        st.bar_chart(counts)

    st.subheader("Reviews (filtered) + sentiment")
    st.dataframe(analyzed[["date", "text", "sentiment", "confidence"]], use_container_width=True)
