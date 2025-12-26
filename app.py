import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="E-commerce Brand Reputation Monitor (2023)",
    layout="wide",
)

DATA_DIR = Path(".")
PRODUCTS_FILE = DATA_DIR / "products.json"
TESTIMONIALS_FILE = DATA_DIR / "testimonials.json"

# Variant A: precomputed sentiment file
REVIEWS_SCORED_FILE = DATA_DIR / "reviews_scored.json"

# Fallback (if you still keep raw reviews.json)
REVIEWS_RAW_FILE = DATA_DIR / "reviews.json"


# ----------------------------
# HELPERS
# ----------------------------
@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_df(items, kind: str) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return df

    # Normalize expected columns for each dataset
    if kind == "products":
        # common columns: name/title, description, price, source_url, page
        # keep whatever exists, but try to order nicely
        preferred = [c for c in ["name", "title", "description", "price", "page", "source_url"] if c in df.columns]
        rest = [c for c in df.columns if c not in preferred]
        return df[preferred + rest]

    if kind == "testimonials":
        preferred = [c for c in ["author", "text", "source_url"] if c in df.columns]
        rest = [c for c in df.columns if c not in preferred]
        return df[preferred + rest]

    if kind == "reviews":
        # date parsing
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # standardize text col name
        if "review_text" in df.columns and "text" not in df.columns:
            df["text"] = df["review_text"]
        # order columns
        preferred = [c for c in ["date", "text", "sentiment", "confidence", "source_url"] if c in df.columns]
        rest = [c for c in df.columns if c not in preferred]
        return df[preferred + rest]

    return df


def month_options_2023():
    months = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    return [m.strftime("%b %Y") for m in months]


def filter_reviews_2023_by_month(df: pd.DataFrame, month_label: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df

    month_dt = pd.to_datetime(month_label, format="%b %Y", errors="coerce")
    if pd.isna(month_dt):
        return df

    start = month_dt
    end = month_dt + pd.offsets.MonthBegin(1)
    mask = (df["date"] >= start) & (df["date"] < end)
    return df.loc[mask].copy()


# ----------------------------
# LOAD DATA
# ----------------------------
products_raw = load_json(PRODUCTS_FILE)
testimonials_raw = load_json(TESTIMONIALS_FILE)

# Prefer scored reviews; fallback to raw
reviews_raw = load_json(REVIEWS_SCORED_FILE)
if not reviews_raw:
    reviews_raw = load_json(REVIEWS_RAW_FILE)

products_df = to_df(products_raw, "products")
testimonials_df = to_df(testimonials_raw, "testimonials")
reviews_df = to_df(reviews_raw, "reviews")


# ----------------------------
# UI: HEADER + NAV
# ----------------------------
st.title("E-commerce Brand Reputation Monitor (2023)")
st.caption("Scraped from web-scraping.dev (Products / Testimonials / Reviews)")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"], index=2)

# ----------------------------
# PAGE: PRODUCTS
# ----------------------------
if page == "Products":
    st.header("Products")

    if products_df.empty:
        st.warning("products.json is empty or missing.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            q = st.text_input("Search (name/description)", "")
        with col2:
            sort_col = st.selectbox("Sort by", options=[c for c in products_df.columns], index=0)

        df = products_df.copy()
        if q.strip():
            ql = q.strip().lower()
            cols = [c for c in ["name", "title", "description"] if c in df.columns]
            if cols:
                mask = False
                for c in cols:
                    mask = mask | df[c].astype(str).str.lower().str.contains(ql, na=False)
                df = df.loc[mask]

        if sort_col in df.columns:
            df = df.sort_values(sort_col, kind="stable")

        st.dataframe(df, use_container_width=True)

# ----------------------------
# PAGE: TESTIMONIALS
# ----------------------------
elif page == "Testimonials":
    st.header("Testimonials")

    if testimonials_df.empty:
        st.warning("testimonials.json is empty or missing.")
    else:
        q = st.text_input("Search (author/text)", "")
        df = testimonials_df.copy()

        if q.strip():
            ql = q.strip().lower()
            cols = [c for c in ["author", "text"] if c in df.columns]
            if cols:
                mask = False
                for c in cols:
                    mask = mask | df[c].astype(str).str.lower().str.contains(ql, na=False)
                df = df.loc[mask]

        st.dataframe(df, use_container_width=True)

# ----------------------------
# PAGE: REVIEWS (NO MODEL RUN)
# ----------------------------
else:
    st.header("Reviews — Sentiment Analysis (precomputed)")

    if reviews_df.empty:
        st.warning("No reviews found (reviews_scored.json or reviews.json missing/empty).")
        st.stop()

    # Sidebar controls (make it fast by default)
    st.sidebar.subheader("Reviews limits")
    default_max = 10  # minimal by default
    max_to_show = st.sidebar.slider("Max reviews to show", 5, 50, default_max, step=5)

    # Month selector
    months = month_options_2023()
    selected_month = st.select_slider("Select a month (2023)", options=months, value="Jan 2023")

    filtered = filter_reviews_2023_by_month(reviews_df, selected_month)

    if filtered.empty:
        st.info(f"No reviews found for {selected_month}.")
        st.stop()

    # If sentiment exists -> summary + chart
    has_sentiment = "sentiment" in filtered.columns
    has_conf = "confidence" in filtered.columns

    if has_sentiment:
        left, right = st.columns([2, 1])
        with left:
            counts = filtered["sentiment"].value_counts()
            st.subheader("Sentiment distribution")
            st.bar_chart(counts)

        with right:
            st.subheader("Metrics")
            st.metric("Total reviews in month", int(len(filtered)))
            if "POSITIVE" in counts.index:
                st.metric("Positive", int(counts.get("POSITIVE", 0)))
            if "NEGATIVE" in counts.index:
                st.metric("Negative", int(counts.get("NEGATIVE", 0)))
            if has_conf:
                avg_conf = float(pd.to_numeric(filtered["confidence"], errors="coerce").mean())
                if pd.notna(avg_conf):
                    st.metric("Avg confidence", f"{avg_conf:.2%}")

        # Optional filter
        sentiments_available = sorted(filtered["sentiment"].dropna().unique().tolist())
        chosen = st.multiselect("Filter by sentiment", sentiments_available, default=sentiments_available)
        if chosen:
            filtered = filtered[filtered["sentiment"].isin(chosen)]

    # Sort newest first, then limit (FAST DEFAULT)
    if "date" in filtered.columns:
        filtered = filtered.sort_values("date", ascending=False, kind="stable")

    st.subheader(f"Reviews for {selected_month} (showing up to {max_to_show})")
    st.dataframe(filtered.head(max_to_show), use_container_width=True)
