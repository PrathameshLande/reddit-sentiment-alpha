# sentiment.py — Phase 3
# Scores every Reddit post title with two models:
#   1. VADER   — fast, rule-based baseline
#   2. FinBERT — finance-trained AI model (the main scorer)
# Output saved to data/processed/scored_posts.csv

import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch

# ── FILE PATHS ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "..", "data", "raw",       "wsb_posts.csv")
OUTPUT_FILE= os.path.join(BASE_DIR, "..", "data", "processed", "scored_posts.csv")

# ── 1. VADER SCORER ───────────────────────────────────────────────────────────
# VADER gives us a "compound" score from -1.0 (most negative) to +1.0 (most positive)
# It's fast — no GPU, no download, runs instantly
vader = SentimentIntensityAnalyzer()

def vader_score(text):
    """Returns a single compound score between -1.0 and +1.0"""
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return vader.polarity_scores(text)["compound"]


# ── 2. FINBERT SCORER ─────────────────────────────────────────────────────────
# FinBERT is a BERT model fine-tuned on ~10,000 financial news headlines.
# It outputs three probabilities: positive, negative, neutral.
# We combine them into one score: positive_prob - negative_prob
#
# First run: downloads ~440MB model from HuggingFace automatically (free).
# Subsequent runs: loads from local cache instantly.

print("⏳ Loading FinBERT model (downloads ~440MB on first run — please wait)...")

# torch.device picks GPU if you have one (CUDA), otherwise falls back to CPU
# For most laptops, this will be CPU — totally fine for our data size
device = 0 if torch.cuda.is_available() else -1

finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None,           # return scores for ALL three labels (pos/neg/neutral)
    device=device
)
print("✅ FinBERT loaded\n")


def finbert_score(text):
    """
    Runs FinBERT on a piece of text.
    Returns a score from -1.0 (bearish) to +1.0 (bullish).

    How it works:
      FinBERT outputs something like:
        [{"label": "positive", "score": 0.82},
         {"label": "negative", "score": 0.12},
         {"label": "neutral",  "score": 0.06}]
      We compute: positive_score - negative_score = 0.82 - 0.12 = +0.70
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0

    # FinBERT has a 512-token limit — titles are short so this is rarely an issue
    text = text[:512]

    try:
        results = finbert(text)[0]   # [0] because we passed one string
        # Convert list of dicts into a simple label→score lookup
        scores = {r["label"]: r["score"] for r in results}
        return scores.get("positive", 0.0) - scores.get("negative", 0.0)
    except Exception as e:
        print(f"  ⚠️  FinBERT error on: '{text[:60]}...' → {e}")
        return 0.0


# ── 3. SCORE INTERPRETATION HELPER ───────────────────────────────────────────
def interpret(score):
    """Turns a numeric score into a human-readable label."""
    if score >  0.15: return "BULLISH"
    if score < -0.15: return "BEARISH"
    return "NEUTRAL"


# ── 4. MAIN SCORING PIPELINE ─────────────────────────────────────────────────
def score_all(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """
    Reads the raw CSV, scores every title, saves enriched CSV.
    """
    # Load raw posts
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("   Run scraper.py first!")
        return None

    df = pd.read_csv(input_file)
    print(f"📂 Loaded {len(df)} posts from {input_file}\n")

    # ── VADER (fast — score all at once) ──────────────────────────────────────
    print("🔄 Running VADER scorer...")
    df["vader_score"] = df["title"].apply(vader_score)
    print("✅ VADER done\n")

    # ── FinBERT (slower — shows progress) ────────────────────────────────────
    print("🤖 Running FinBERT scorer (this takes ~1-3 mins on CPU)...")
    scores = []
    total = len(df)
    for i, title in enumerate(df["title"], 1):
        score = finbert_score(title)
        scores.append(score)
        # Progress indicator every 10 posts
        if i % 10 == 0 or i == total:
            print(f"   {i}/{total} posts scored...", end="\r")

    df["finbert_score"] = scores
    print("\n✅ FinBERT done\n")

    # ── Labels ────────────────────────────────────────────────────────────────
    df["vader_label"]   = df["vader_score"].apply(interpret)
    df["finbert_label"] = df["finbert_score"].apply(interpret)

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("── Sentiment Summary ────────────────────────────────────")
    print(f"  FinBERT avg score :  {df['finbert_score'].mean():.3f}")
    print(f"  VADER avg score   :  {df['vader_score'].mean():.3f}")
    print()
    print("  FinBERT label breakdown:")
    print(df["finbert_label"].value_counts().to_string())
    print()
    print("  Sample (highest scoring posts):")
    top = df.nlargest(3, "finbert_score")[["finbert_score", "finbert_label", "title"]]
    for _, row in top.iterrows():
        print(f"  [{row['finbert_label']:7s} {row['finbert_score']:+.2f}]  {row['title'][:70]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved scored data → {output_file}")

    return df


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    score_all()
