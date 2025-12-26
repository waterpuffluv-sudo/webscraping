## What is this?
Streamlit dashboard that displays **Products**, **Testimonials**, and **Reviews (2023)** scraped from `https://web-scraping.dev/`.
For reviews, the app runs **HuggingFace Transformers** sentiment analysis (Positive/Negative) and shows counts + average confidence.

## 1) Scrape data (run once)
The assignment recommends scraping ahead of time and saving the results to JSON.

```bash
pip install -r requirements.txt
python scraper.py
```

This creates/updates:
- `products.json`
- `testimonials.json`
- `reviews.json`

## 2) Run the Streamlit app
```bash
streamlit run app.py
```

**Note:** the first time you open the Reviews page, the sentiment model may take longer to load because it downloads and caches the model.

## Deploy (Render)
Build command:
```bash
pip install -r requirements.txt
```

Start command:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
```
