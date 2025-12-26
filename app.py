import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

# меньше лишних потоков токенайзера
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")

DATA_DIR = Path(".")  # если сделаешь папку data/, поменяй на Path("data")

PRODUCTS_PATH = DATA_DIR / "products.json"
TESTIMONIALS_PATH = DATA_DIR / "testimonials.json"
REVIEWS_PATH = DATA_DIR / "reviews.json"


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
    df = df.rename(columns={
        "Product Name": "name",
        "Description": "description",
        "title": "name",
    })
    return df


def normalize_testimonials(raw):
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    df = df.rename(columns={
        "Author": "author",
        "Testimonial": "text",
        "testimonial": "text",
    })
    return df


def parse_review_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    s = s.replace("Reviewed on", "").replace("reviewed on", "").strip()
    s = s.replace("on", "").strip()
    return pd.to_datetime(s, errors="coerce")


def normalize_reviews(raw):
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    df = df.rename(columns={
        "review_text": "text",
        "review_date": "date",
    })

    if "date" not in df.columns:
        df["date"] = pd.NaT
    df["date"] = df["date"].apply(parse_review_date)

    if "text" not in df.columns:
        df["text"] = ""

    return df


@st.cache_resource
def get_sentiment_pipe():
    # грузится ТОЛЬКО когда реально запускаешь анализ (по кнопке)
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_data(show_spinner=False)
def analyze_texts(texts: list[str]):
    pipe = get_sentiment_pipe()
    # batch + truncation для стабильности на хостинге
    results = pipe(
        texts,
        batch_size=8,
        truncation=True,
        max_length=256,
    )
    return results


# ----------------------------
# Load data
# ----------------------------
products_df = normalize_products(load_json(PRODUCTS_PATH))
testimonials_df = normalize_testimonials(load_json(TESTIMONIALS_PATH))
reviews_df = normalize_reviews(load_json(REVIEWS_PATH))

# ----------------------------
# UI
# ----------------------------
st.title("E-commerce Brand Reputation Monitor (2023)")
st.caption("web-scraping.dev (Products / Testimonials / Reviews) + HuggingFace sentiment analysis")

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

    reviews_2023 = reviews_df.dropna(subset=["date"]).copy()
    reviews_2023 = reviews_2023[reviews_2023["date"].dt.year == 2023].copy()

    if reviews_2023.empty:
        st.warning("No valid 2023 dates found in reviews.json.")
        st.stop()

    months = pd.period_range("2023-01", "2023-12", freq="M")
    month_labels = [m.strftime("%b %Y") for m in months]

    selected_label = st.select_slider(
        "Select a month (2023)",
        options=month_labels,
        value=month_labels[0],
    )
    selected_period = pd.Period(selected_label, freq="M")

    filtered = reviews_2023[reviews_2023["date"].dt.to_period("M") == selected_period].copy()
    if filtered.empty:
        st.info(f"No reviews found for {selected_label}.")
        st.stop()

    # Оптимизация по умолчанию: показываем мало
    st.sidebar.subheader("Reviews limits")
    max_n = st.sidebar.slider("Max reviews to analyze", 5, 30, 10, step=5)   # <= дефолт 10
    auto_run = st.sidebar.checkbox("Auto-run sentiment", value=False)        # <= дефолт OFF

    filtered = filtered.sort_values("date", ascending=False).head(max_n)

    st.subheader("Preview (fast)")
    st.dataframe(filtered[["date", "text"]], use_container_width=True)

    run_now = auto_run or st.button("Run sentiment analysis")

    if not run_now:
        st.info("Sentiment analysis is paused. Enable Auto-run or press the button.")
        st.stop()

    with st.spinner("Running HuggingFace sentiment analysis (first run may download the model)..."):
        texts = filtered["text"].astype(str).tolist()
        results = analyze_texts(texts)

    analyzed = filtered.copy()
    analyzed["sentiment"] = [r["label"] for r in results]
    analyzed["confidence"] = [float(r["score"]) for r in results]

    st.subheader("Results")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Reviews analyzed", len(analyzed))
        st.metric("Avg confidence", f"{analyzed['confidence'].mean():.2%}")

    with col2:
        counts = analyzed["sentiment"].value_counts().reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
        st.bar_chart(counts)

    st.subheader("Reviews + sentiment")
    st.dataframe(analyzed[["date", "text", "sentiment", "confidence"]], use_container_width=True)
