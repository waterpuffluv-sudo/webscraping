import json
from pathlib import Path

from transformers import pipeline

IN_PATH = Path("reviews.json")
OUT_PATH = Path("reviews_scored.json")

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError("reviews.json not found")

    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    if not data:
        OUT_PATH.write_text("[]", encoding="utf-8")
        print("No reviews to analyze.")
        return

    texts = [str(x.get("text", "")) for x in data]

    pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    results = pipe(
        texts,
        batch_size=8,
        truncation=True,
        max_length=256
    )

    for item, r in zip(data, results):
        item["sentiment"] = r["label"]
        item["confidence"] = float(r["score"])

    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(data)} reviews to {OUT_PATH}")

if __name__ == "__main__":
    main()
