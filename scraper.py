import json
import re
import time
import shutil
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


BASE = "https://web-scraping.dev"


def create_driver() -> webdriver.Chrome:
    chrome_path = shutil.which("chromium") or shutil.which("chromium-browser")
    driver_path = shutil.which("chromedriver")

    if not chrome_path:
        raise RuntimeError("Chromium not found. Add pkgs.chromium to dev.nix and rebuild.")
    if not driver_path:
        raise RuntimeError("chromedriver not found. Add pkgs.chromedriver to dev.nix and rebuild.")

    options = webdriver.ChromeOptions()
    options.binary_location = chrome_path
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")

    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(40)
    return driver


def _clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "")).strip()


def _extract_price(text: str) -> Optional[str]:
    m = re.search(r"\$\s?\d+(?:\.\d{2})?", text)
    return m.group(0).replace(" ", "") if m else None


def _safe_get_text(el) -> str:
    return _clean_text(el.get_text(" ", strip=True)) if el else ""


# -------------------- PRODUCTS (pagination pages) --------------------

def parse_products_html(html: str, page: int) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")

    # Robust: product cards are often "div.card"
    cards = soup.select("div.card")
    if not cards:
        # fallback
        cards = soup.select("[class*='product'], div[class*='Product'], article")

    items = []
    for c in cards:
        # name
        name_el = c.select_one("h2, h3, a h2, a h3, h2 a, h3 a, a")
        name = _safe_get_text(name_el)
        if not name or len(name) < 2:
            continue

        # description: first meaningful paragraph
        p_el = c.select_one("p")
        desc = _safe_get_text(p_el)

        # price: try dedicated class, else regex from whole card text
        price_el = c.select_one(".price, [class*='price'], [data-testid*='price']")
        price = _safe_get_text(price_el) if price_el else _extract_price(_safe_get_text(c))

        items.append({
            "name": name,
            "description": desc,
            "price": price,
            "page": page
        })

    # Deduplicate внутри страницы
    seen = set()
    uniq = []
    for it in items:
        key = (it["name"], it.get("price"))
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    return uniq


def scrape_products(driver: webdriver.Chrome, max_pages: int = 30) -> List[Dict]:
    all_items: List[Dict] = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/products?page={page}"
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        page_items = parse_products_html(driver.page_source, page=page)
        if not page_items:
            break

        all_items.extend(page_items)

    return all_items


# -------------------- REVIEWS ("Load More" until button disappears) --------------------

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def parse_reviews_html(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")

    # Reviews grid/cards
    cards = soup.select("div.card")
    if not cards:
        cards = soup.select("[class*='review'], article, li")

    items = []
    for c in cards:
        txt = _safe_get_text(c)
        date_m = DATE_RE.search(txt)
        if not date_m:
            continue
        date_str = date_m.group(1)

        # review text: pick a paragraph that is not empty and not just date
        p_candidates = c.select("p")
        review_text = ""
        for p in p_candidates:
            t = _safe_get_text(p)
            if t and not DATE_RE.search(t):
                review_text = t
                break

        if not review_text:
            # fallback: remove date from full text
            review_text = _clean_text(DATE_RE.sub("", txt))

        # rating: count star svgs if present (optional)
        rating = None
        star_svgs = c.select("svg")
        if star_svgs:
            # usually 5 stars exist, but sometimes only filled stars are present
            # We'll approximate by counting svgs near top if many exist
            if len(star_svgs) >= 3:
                rating = min(5, len(star_svgs))

        items.append({
            "date": date_str,
            "text": review_text,
            "rating": rating
        })

    # Deduplicate
    seen = set()
    uniq = []
    for it in items:
        key = (it["date"], it["text"][:80])
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    return uniq


def scrape_reviews(driver: webdriver.Chrome, max_clicks: int = 100) -> List[Dict]:
    url = f"{BASE}/reviews"
    driver.get(url)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    def get_count() -> int:
        return len(parse_reviews_html(driver.page_source))

    prev_count = get_count()

    for _ in range(max_clicks):
        # try find "Load More" button by visible text
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Load More')]"))
            )
        except Exception:
            # no button -> finish
            break

        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
        time.sleep(0.2)
        btn.click()
        time.sleep(1.0)

        new_count = get_count()
        if new_count <= prev_count:
            # защита от бесконечного цикла
            break
        prev_count = new_count

    return parse_reviews_html(driver.page_source)


# -------------------- TESTIMONIALS (scroll to end) --------------------

def parse_testimonials_html(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")

    blocks = soup.select("[class*='testimonial'], div.card, blockquote")
    items = []

    for b in blocks:
        text = ""
        author = ""

        # Try find author-like element
        author_el = b.select_one("[class*='author'], strong, h3, h4")
        author = _safe_get_text(author_el)

        # Text: best paragraph / blockquote content
        if b.name == "blockquote":
            text = _safe_get_text(b)
        else:
            p = b.select_one("p")
            text = _safe_get_text(p) if p else _safe_get_text(b)

        text = _clean_text(text)
        author = _clean_text(author)

        # simple validation
        if len(text) < 10:
            continue

        items.append({"author": author, "text": text})

    # Deduplicate
    seen = set()
    uniq = []
    for it in items:
        key = (it.get("author", ""), it["text"][:80])
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    return uniq


def scrape_testimonials(driver: webdriver.Chrome, max_scrolls: int = 60) -> List[Dict]:
    url = f"{BASE}/testimonials"
    driver.get(url)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    last_height = driver.execute_script("return document.body.scrollHeight")
    stable_rounds = 0

    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.0)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stable_rounds += 1
            if stable_rounds >= 3:  # 3 раза подряд не меняется -> конец
                break
        else:
            stable_rounds = 0
            last_height = new_height

    return parse_testimonials_html(driver.page_source)


# -------------------- MAIN --------------------

def save_json(path: str, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    driver = create_driver()
    try:
        products = scrape_products(driver)
        save_json("products.json", products)
        print(f"Products scraped: {len(products)}")

        reviews = scrape_reviews(driver)
        save_json("reviews.json", reviews)
        print(f"Reviews scraped: {len(reviews)}")

        testimonials = scrape_testimonials(driver)
        save_json("testimonials.json", testimonials)
        print(f"Testimonials scraped: {len(testimonials)}")

    finally:
        driver.quit()
