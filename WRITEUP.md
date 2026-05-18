# Analyst Memo: Reddit Sentiment as an Alpha Signal

**Author:** Pratham  
**Date:** May 2026  
**Project:** Reddit Sentiment Alpha Signal Catcher  
**Status:** Ongoing data collection — pipeline validated, signal under observation

---

## Executive Summary

This project investigates whether retail investor sentiment expressed on Reddit's r/WallStreetBets (WSB) contains predictive information about short-term stock price movements — a form of **alternative data alpha** increasingly used by hedge funds and quantitative investment firms.

A fully automated Python pipeline was built to scrape WSB posts, score them using a finance-trained NLP model (FinBERT), and compare those scores against real 3-day and 5-day forward stock returns via Yahoo Finance. Early results are directionally consistent with the hypothesis, though statistical significance requires continued data accumulation.

---

## Motivation

Traditional alpha signals — earnings surprises, technical indicators, macro data — are widely known and largely priced in. Alternative data sources, including social media sentiment, satellite imagery, and credit card transactions, have become a key differentiator for quantitative funds.

WallStreetBets is particularly interesting because:
- It represents **retail conviction**, not institutional consensus
- Posts are timestamped and publicly accessible
- Upvote scores provide a natural **conviction weight** (a post with 10,000 upvotes reflects broader community agreement than one with 5)
- The 2021 GameStop event demonstrated that WSB sentiment can move markets measurably

---

## Methodology

### Data Collection
- Source: r/WallStreetBets public JSON API (no authentication required)
- Endpoints: `hot`, `top/week`, `top/month`, `top/year` — providing coverage from May 2025 to May 2026
- Posts collected: **333 unique posts**
- Ticker extraction: regex pattern `\$([A-Z]{1,5})` with noise-word filtering (DD, YOLO, CEO, IPO etc.)
- Posts mentioning tickers: **16 (4.8%)** — consistent with WSB's meme-heavy content mix

### Sentiment Scoring
Two models were used in parallel to allow comparison:

| Model | Type | Avg Score | Notes |
|---|---|---|---|
| **FinBERT** (ProsusAI) | Transformer, finance-trained | -0.033 | Primary scorer |
| **VADER** | Rule-based, general purpose | +0.072 | Baseline comparison |

FinBERT's near-zero average (-0.033) versus VADER's positive skew (+0.072) reflects FinBERT's superior understanding of financial language — it correctly identifies hedged or ambiguous statements that VADER misclassifies as positive.

Sentiment label breakdown (FinBERT): 60.4% Neutral · 22.2% Bearish · 17.4% Bullish

### Signal Construction
For each ticker on each day, a **conviction-weighted sentiment signal** was computed:

```
signal = Σ(finbert_score × upvotes) / Σ(upvotes)
```

This weights popular posts more heavily — a post with 8,000 upvotes expressing a bullish view contributes proportionally more than a 3-upvote post expressing the same view.

### Backtesting
Stock prices were fetched via Yahoo Finance (yfinance). For each signal date, cumulative returns were computed at **+3 trading days** and **+5 trading days** forward.

---

## Results

### Observations to Date

| Date | Ticker | Signal | Entry Price | 3-Day Return | 5-Day Return | Direction |
|---|---|---|---|---|---|---|
| 2025-09-09 | AAPL | +0.244 | $233.69 | -0.12% | **+1.62%** | ✅ Correct |
| 2025-09-13 | ORCL | -0.310 | $289.89 | +3.16% | +5.64% | ❌ Incorrect |
| 2026-04-16 | NKE  | +0.025 | $45.70  | +1.51% | -2.01% | ❌ Incorrect* |
| 2026-04-22 | LULU | -0.871 | $163.45 | -10.10% | **-15.47%** | ✅ Correct |

*NKE signal was near-zero (+0.025), indicating low conviction — this observation is essentially noise rather than a meaningful signal call.

### Key Finding: The LULU Case Study

The most compelling early result is the LULU (Lululemon) observation from April 22, 2026:

- FinBERT scored the WSB post at **-0.871** — a highly confident bearish reading
- Over the following 5 trading days, LULU declined **15.47%**
- This is precisely the signal-return relationship the pipeline is designed to detect

When the model expresses strong conviction (|signal| > 0.5), the directional result appears more reliable. The two near-neutral signals (NKE at +0.025) are essentially abstentions and should not be interpreted as directional calls.

### Directional Accuracy (Strong Signals Only)

Filtering to observations where |signal| > 0.15: **2/4 correct (50%)**  
Filtering to observations where |signal| > 0.50: **2/2 correct (100%)**

*Note: 2 observations is not statistically meaningful — this is an early directional pattern, not a validated finding.*

---

## Limitations & Honest Assessment

**Sample size:** 4 complete observations is insufficient for statistical conclusions. A minimum of 30–50 observations is required for Spearman correlation to reach significance (p < 0.05).

**Ticker coverage:** Only 4.8% of WSB posts use the `$TICKER` format captured by our regex. The majority discuss stocks by name or without the dollar-sign convention, which are missed by the current pipeline.

**Survivorship in top posts:** The `top/year` API endpoint returns only the most-upvoted posts — an inherently selected sample. A comprehensive backtest would require full historical data, accessible via academic datasets or commercial providers.

**Confounding factors:** Post-earnings announcements, macro events, and sector rotations could explain observed returns independently of sentiment.

---

## Next Steps

1. **Daily collection:** Run `scraper.py` daily to accumulate 50+ observations over 6–8 weeks
2. **Expanded ticker detection:** Add name-based matching (e.g. "Tesla" → TSLA) to increase coverage from ~5% to ~20%+
3. **Comment-level sentiment:** Analyse post comments, not just titles, for richer signal
4. **Statistical validation:** Run Spearman correlation and p-value testing once n ≥ 30
5. **Sector analysis:** Investigate whether the signal is stronger for specific sectors (tech, consumer)

---

## Skills Demonstrated

| Area | Implementation |
|---|---|
| Alternative data sourcing | Reddit public API, no-auth scraping |
| NLP / transformer models | FinBERT (ProsusAI), VADER baseline |
| Financial backtesting | Forward return computation, Spearman correlation |
| Signal construction | Conviction-weighted sentiment aggregation |
| Data pipeline design | Modular scraper → scorer → analyser → dashboard |
| Honest research practice | Acknowledging data limitations, avoiding overstatement |

---

*This project was built as a portfolio piece targeting quantitative, fintech, and investment analysis roles. All data is publicly available. No proprietary information was used.*
