import json
import re
import time
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
    TimeoutException,
)

BASE = "https://web-scraping.dev"


# -----------------------------
# Helpers
# -----------------------------
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_date_to_iso(s: str) -> Optional[str]:
    """Returns YYYY-MM-DD if possible, else None."""
    s = _clean(s)

    # Already ISO inside the text?
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Strip common prefixes
    s2 = s.replace("Reviewed on", "").replace("on", "").strip()

    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s2, fmt).date().isoformat()
        except Exception:
            pass

    return None


def _safe_text(el) -> str:
    try:
        return _clean(el.get_text(" ", strip=True))
    except Exception:
        try:
            return _clean(el.text)
        except Exception:
            return ""


def save_json(path: str, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------
# Driver
# -----------------------------
def create_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Works in:
    - Firebase/IDX (chromium + chromedriver from nix)
    - GitHub Actions (Chrome exists; Selenium Manager can handle driver)
    """
    chrome_bin = (
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
    )
    chromedriver_bin = shutil.which("chromedriver")

    options = webdriver.ChromeOptions()
    if chrome_bin:
        options.binary_location = chrome_bin

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")

    # If chromedriver exists -> use it, else let Selenium Manager handle it
    if chromedriver_bin:
        service = Service(chromedriver_bin)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)

    driver.set_page_load_timeout(45)
    return driver


def wait_page(driver, timeout: int = 15):
    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))


def safe_click(driver, element):
    """
    Robust click that avoids ElementClickInterceptedException in headless/CI.
    """
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    time.sleep(0.2)
    driver.execute_script("window.scrollBy(0, -120);")  # avoid sticky footer/header overlap
    time.sleep(0.15)

    try:
        ActionChains(driver).move_to_element(element).pause(0.1).click(element).perform()
        return
    except Exception:
        pass

    # JS click fallback
    driver.execute_script("arguments[0].click();", element)


def find_clickable_by_text(driver, text: str):
    """
    Find visible, enabled button or link with exact text (case-insensitive).
    """
    t = text.strip().lower()
    for el in driver.find_elements(By.CSS_SELECTOR, "button, a"):
        try:
            if el.is_displayed() and el.is_enabled() and el.text.strip().lower() == t:
                return el
        except StaleElementReferenceException:
            continue
    return None


# -----------------------------
# PRODUCTS (pagination pages ?page=)
# -----------------------------
def extract_products(html: str, page: int, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")

    # Common pattern: cards
    cards = soup.select("div.card, article, div[class*='product']")
    items = []

    for c in cards:
        name_el = c.select_one("h2, h3, h2 a, h3 a, a")
        name = _safe_text(name_el)
        if not name:
            continue

        desc_el = c.select_one("p")
        desc = _safe_text(desc_el)

        price_el = c.select_one(".price, [class*='price']")
        price = _safe_text(price_el)

        items.append(
            {
                "name": name,
                "description": desc,
                "price": price if price else None,
                "page": page,
                "source_url": url,
            }
        )

    # Dedup
    seen = set()
    out = []
    for it in items:
        key = (it["name"], it.get("price"))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def scrape_products(driver, max_pages: int = 50) -> List[Dict]:
    all_items: List[Dict] = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/products?page={page}"
        driver.get(url)
        wait_page(driver)
        time.sleep(0.7)

        items = extract_products(driver.page_source, page=page, url=url)
        if not items:
            break
        all_items.extend(items)

    return all_items


# -----------------------------
# REVIEWS (Load More until disappears)
# -----------------------------
DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def extract_reviews(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.review, div[class*='review'], div.card, article, li")

    items = []
    for c in cards:
        text_all = _safe_text(c)
        m = DATE_RE.search(text_all)
        if not m:
            continue

        iso_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        # pick the longest paragraph as review text
        ps = c.select("p")
        best = ""
        for p in ps:
            t = _safe_text(p)
            if t and not DATE_RE.search(t) and len(t) > len(best):
                best = t

        if not best:
            # fallback: remove date from full text
            best = _clean(DATE_RE.sub("", text_all))

        if len(best) < 5:
            continue

        items.append({"date": iso_date, "text": best, "source_url": url})

    # Dedup
    seen = set()
    out = []
    for it in items:
        key = (it["date"], it["text"][:80])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def scrape_reviews(driver, max_clicks: int = 250) -> List[Dict]:
    url = f"{BASE}/reviews"
    driver.get(url)
    wait_page(driver)
    time.sleep(1.0)

    def reviews_count() -> int:
        return len(extract_reviews(driver.page_source, url=url))

    prev = reviews_count()

    for _ in range(max_clicks):
        btn = find_clickable_by_text(driver, "Load More")
        if not btn:
            break

        try:
            safe_click(driver, btn)
        except ElementClickInterceptedException:
            # last resort: JS click
            try:
                driver.execute_script("arguments[0].click();", btn)
            except Exception:
                break
        except StaleElementReferenceException:
            continue

        # wait until more reviews appear or button disappears
        try:
            WebDriverWait(driver, 10).until(
                lambda d: reviews_count() > prev or find_clickable_by_text(d, "Load More") is None
            )
        except TimeoutException:
            # no change => stop (avoids infinite loops)
            cur = reviews_count()
            if cur <= prev:
                break

        cur = reviews_count()
        if cur <= prev:
            break
        prev = cur
        time.sleep(0.25)

    return extract_reviews(driver.page_source, url=url)


# -----------------------------
# TESTIMONIALS (scroll to end)
# -----------------------------
def extract_testimonials(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.select("div.testimonial, div[class*='testimonial'], blockquote, div.card, article")

    items = []
    for b in blocks:
        # author
        author_el = b.select_one("[class*='author'], strong, h3, h4")
        author = _safe_text(author_el) or None

        # text
        text_el = b.select_one("p") if b.name != "blockquote" else b
        text = _safe_text(text_el)
        if len(text) < 10:
            continue

        items.append({"author": author, "text": text, "source_url": url})

    # Dedup
    seen = set()
    out = []
    for it in items:
        key = (it.get("author") or "", it["text"][:80])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def scrape_testimonials(driver, max_scrolls: int = 200) -> List[Dict]:
    url = f"{BASE}/testimonials"
    driver.get(url)
    wait_page(driver)
    time.sleep(1.0)

    last_height = driver.execute_script("return document.body.scrollHeight")
    stable = 0

    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.9)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
            last_height = new_height

    return extract_testimonials(driver.page_source, url=url)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    driver = create_driver(headless=True)
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
