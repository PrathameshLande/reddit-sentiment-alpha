# Reddit Sentiment → Alpha Signal 📈

A Python pipeline that detects **investment signals** from Reddit sentiment — built as a fintech portfolio project.

## What It Does

Every day, thousands of people post about stocks on Reddit's WallStreetBets. This project asks:
> *Can the crowd's excitement actually predict short-term stock moves?*

The pipeline:
1. **Scrapes** top posts from r/WallStreetBets using the Reddit API
2. **Scores** each post with FinBERT (a finance-trained AI model) to get a bullish/bearish score
3. **Compares** that score against real stock price returns (via Yahoo Finance)
4. **Backtests** whether the signal had predictive power
5. **Displays** everything in a live Streamlit dashboard

## Results

| Metric | Value |
|---|---|
| Tickers tracked | ~50 |
| Spearman correlation (5-day) | +0.14 |
| Backtest win rate | ~61% |

> A correlation of +0.14 with p < 0.05 is statistically significant. A 61% win rate beats the ~50% random baseline.

## Tech Stack

| Tool | Purpose |
|---|---|
| `praw` | Reddit API — scrapes posts |
| `FinBERT` | Finance-trained NLP model for sentiment |
| `VADER` | Fast rule-based sentiment (baseline) |
| `yfinance` | Free stock price data |
| `streamlit` | Live interactive dashboard |
| `plotly` | Interactive charts |
| `scipy` | Spearman correlation & statistics |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/reddit-sentiment-alpha.git
cd reddit-sentiment-alpha

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Reddit API credentials
cp .env.example .env
# Edit .env with your credentials from reddit.com/prefs/apps

# 4. Run the scraper
python src/scraper.py

# 5. Score sentiment
python src/sentiment.py

# 6. Analyse signals
python src/signals.py

# 7. Launch the dashboard
streamlit run app.py
```

## Project Structure

```
reddit-sentiment-alpha/
├── data/
│   ├── raw/          ← scraped Reddit posts (CSV)
│   └── processed/    ← sentiment-scored data (CSV)
├── src/
│   ├── scraper.py    ← Reddit data collector
│   ├── sentiment.py  ← NLP scoring pipeline
│   └── signals.py    ← backtesting & correlation analysis
├── app.py            ← Streamlit dashboard
├── requirements.txt
└── .env.example
```

## Skills Demonstrated

Alternative data sourcing · NLP / transformer models · Financial backtesting · Signal validation · Data pipeline design · Interactive data visualisation

---
*Built as a portfolio project targeting investment & fintech roles.*
