import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Важно: не плодим лишние потоки токенайзера
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")

DATA_DIR = Path(".")
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
    # на всякий случай поддержка разных ключей
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


def normalize_reviews(raw):
    """
    Ожидаемый формат (как у тебя):
    {
      "date": "2023-05-18",
      "text": "...",
      "source_url": "https://web-scraping.dev/reviews"
    }
    """
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    # гарантируем нужные колонки
    if "date" not in df.columns:
        df["date"] = None
    if "text" not in df.columns:
        df["text"] = ""
    if "source_url" not in df.columns:
        df["source_url"] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # ISO отлично парсится
    df["text"] = df["text"].astype(str)

    return df


# --- Sentiment (ленивая загрузка) ---
@st.cache_resource
def get_sentiment_pipe():
    # импорт внутри, чтобы приложение не падало до клика и не тратило память на старте
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_data(show_spinner=False)
def analyze_texts(texts: tuple[str, ...]):
    pipe = get_sentiment_pipe()
    # батчи + обрезка текста = быстрее и стабильнее на Render
    results = pipe(
        list(texts),
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
st.caption("Scraped from web-scraping.dev (Products / Testimonials / Reviews) + Sentiment Analysis")

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
        st.warning("No valid 2023 reviews found.")
        st.stop()

    # выбор месяца
    months = pd.period_range("2023-01", "2023-12", freq="M")
    month_labels = [m.strftime("%b %Y") for m in months]

    selected_label = st.select_slider(
        "Select a month (2023)",
        options=month_labels,
        value=month_labels[0],
    )
    selected_period = pd.Period(selected_label, freq="M")

    filtered = reviews_2023[reviews_2023["date"].dt.to_period("M") == selected_period].copy()
    filtered = filtered.sort_values("date", ascending=False)

    if filtered.empty:
        st.info(f"No reviews found for {selected_label}.")
        st.stop()

    # ---- Оптимизация: по умолчанию минимум ----
    st.sidebar.subheader("Reviews limits")
    max_n = st.sidebar.slider("Max reviews to analyze", 5, 30, 5, step=5)  # дефолт 5
    auto_run = st.sidebar.checkbox("Auto-run sentiment", value=False)       # дефолт OFF

    # берем только N последних
    filtered_n = filtered.head(max_n).copy()

    st.subheader("Preview (fast)")
    st.dataframe(
        filtered_n[["date", "text", "source_url"]],
        use_container_width=True,
        height=320
    )

    run_now = auto_run or st.button("Run sentiment analysis")

    if not run_now:
        st.info("Sentiment analysis is paused. Enable Auto-run or press the button.")
        st.stop()

    # ---- Анализ ----
    try:
        with st.spinner("Running sentiment analysis (first run may download the model)..."):
            texts = tuple(filtered_n["text"].astype(str).tolist())
            results = analyze_texts(texts)

        analyzed = filtered_n.copy()
        analyzed["sentiment"] = [r["label"] for r in results]
        analyzed["confidence"] = [float(r["score"]) for r in results]

    except Exception as e:
        st.error("Sentiment model failed to load/run. Check requirements/runtime on Render.")
        st.exception(e)
        st.stop()

    # ---- Визуализация ----
    st.subheader("Results")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Reviews analyzed", len(analyzed))
        st.metric("Avg confidence", f"{analyzed['confidence'].mean():.2%}")

    with col2:
        counts = analyzed["sentiment"].value_counts().reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
        st.bar_chart(counts)

    st.subheader("Reviews + sentiment")
    st.dataframe(
        analyzed[["date", "text", "sentiment", "confidence", "source_url"]],
        use_container_width=True,
        height=420
    )

