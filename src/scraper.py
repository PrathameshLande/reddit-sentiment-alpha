# scraper.py — Phase 2
# Hits Reddit's public JSON API (no credentials needed) and saves
# WallStreetBets posts to data/raw/wsb_posts.csv
#
# We fetch from MULTIPLE endpoints to build up historical data:
#   - hot   → what's popular right now
#   - top?t=month → top posts from the past 30 days (already has future prices!)
#   - top?t=week  → top posts from the past 7 days

import requests
import pandas as pd
import re
import os
from datetime import datetime, timezone

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SUBREDDIT  = "wallstreetbets"
HEADERS    = {"User-Agent": "python:alpha_sentiment_bot:v1.0 (personal learning project)"}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE= os.path.join(OUTPUT_DIR, "wsb_posts.csv")

# Each entry = (sort_type, time_filter)
# time_filter only applies to "top" and "controversial" sort
ENDPOINTS = [
    ("hot",  None),       # current hot posts
    ("top",  "week"),     # best posts of the past 7 days
    ("top",  "month"),    # best posts of the past 30 days
    ("top",  "year"),     # best posts of the past 12 months ← more historical data
]

NOISE_WORDS = {
    "DD","YOLO","CEO","IPO","EPS","ATH","WSB","SEC","FDA",
    "USA","GDP","ETF","IMO","OP","OTM","ITM","DFV","RH",
    "AM","PM","ALL","NEW","FOR","THE","ARE","BIG","NOW","AI","US"
}

# ── TICKER EXTRACTION ─────────────────────────────────────────────────────────
def extract_tickers(text):
    if not isinstance(text, str):
        return []
    raw = re.findall(r'\$([A-Z]{1,5})', text)
    return [t for t in raw if t not in NOISE_WORDS]

# ── FETCH FROM ONE ENDPOINT ───────────────────────────────────────────────────
def fetch_posts(sort="hot", time_filter=None, limit=100):
    """Fetches up to `limit` posts from one Reddit sort endpoint."""
    url = f"https://www.reddit.com/r/{SUBREDDIT}/{sort}.json?limit={limit}"
    if time_filter:
        url += f"&t={time_filter}"

    label = f"{sort}{'/' + time_filter if time_filter else ''}"
    print(f"  📡 Fetching r/{SUBREDDIT}/{label} ...")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"     ❌ Status {resp.status_code}")
            return []
        posts = resp.json()["data"]["children"]
        print(f"     ✅ {len(posts)} posts")
        return posts
    except Exception as e:
        print(f"     ❌ {e}")
        return []

# ── PARSE RAW POSTS ───────────────────────────────────────────────────────────
def parse_posts(posts_raw):
    records = []
    for post in posts_raw:
        p = post["data"]
        created_dt = datetime.fromtimestamp(p["created_utc"], tz=timezone.utc)
        date_str   = created_dt.strftime("%Y-%m-%d")
        tickers    = extract_tickers(p.get("title", ""))
        records.append({
            "date":         date_str,
            "title":        p.get("title", ""),
            "score":        p.get("score", 0),
            "num_comments": p.get("num_comments", 0),
            "tickers":      ",".join(tickers),
            "ticker_count": len(tickers),
            "url":          "https://reddit.com" + p.get("permalink", ""),
        })
    return pd.DataFrame(records)

# ── SAVE ──────────────────────────────────────────────────────────────────────
def save(df, filepath=OUTPUT_FILE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df]).drop_duplicates(subset=["url"])
    df.to_csv(filepath, index=False)
    print(f"\n💾 Saved {len(df)} total posts → {filepath}")
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Scraping WallStreetBets across multiple time windows...\n")

    all_posts = []
    for sort, time_filter in ENDPOINTS:
        raw   = fetch_posts(sort=sort, time_filter=time_filter)
        if raw:
            all_posts.extend(raw)

    if not all_posts:
        print("❌ No posts fetched.")
    else:
        df = parse_posts(all_posts)

        # De-duplicate across endpoints (same post can appear in hot + top)
        df = df.drop_duplicates(subset=["url"])

        print(f"\n── Summary ─────────────────────────────────────────")
        print(f"  Total unique posts:    {len(df)}")
        print(f"  Posts with tickers:    {len(df[df['ticker_count'] > 0])}")
        print(f"  Date range:            {df['date'].min()} → {df['date'].max()}")
        print(f"  Unique tickers found:  {df['tickers'].str.split(',').explode().nunique()}")
        print(f"\n  Post date distribution:")
        print(df['date'].value_counts().sort_index().to_string())

        save(df)
