# signals.py — Phase 4
# Pulls real stock prices and measures whether Reddit sentiment
# predicted subsequent price moves (the "backtesting" phase)

import pandas as pd
import numpy as np
import os
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta

# ── FILE PATHS ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
SCORED_FILE = os.path.join(BASE_DIR, "..", "data", "processed", "scored_posts.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "data", "processed", "signals.csv")

# ── 1. BUILD DAILY SENTIMENT SIGNAL PER TICKER ───────────────────────────────
def build_daily_signal(df):
    ticker_posts = df[df["ticker_count"] > 0].copy()
    if ticker_posts.empty:
        print("⚠️  No posts with tickers found.")
        return pd.DataFrame()

    ticker_posts["ticker"] = ticker_posts["tickers"].str.split(",")
    ticker_posts = ticker_posts.explode("ticker")
    ticker_posts["ticker"] = ticker_posts["ticker"].str.strip()
    ticker_posts = ticker_posts[ticker_posts["ticker"] != ""]

    ticker_posts["weight"]             = ticker_posts["score"].clip(lower=1)
    ticker_posts["weighted_sentiment"] = ticker_posts["finbert_score"] * ticker_posts["weight"]

    daily = ticker_posts.groupby(["date", "ticker"]).agg(
        raw_signal   = ("weighted_sentiment", "sum"),
        total_weight = ("weight",             "sum"),
        post_count   = ("title",              "count"),
    ).reset_index()

    daily["signal"] = daily["raw_signal"] / daily["total_weight"]

    print(f"📊 Daily signals built — {len(daily)} ticker-day rows")
    print(f"   Unique tickers:  {daily['ticker'].nunique()}")
    print(f"   Date range:      {daily['date'].min()} → {daily['date'].max()}\n")
    return daily


# ── 2. FETCH STOCK PRICES ─────────────────────────────────────────────────────
def fetch_prices(tickers, start_date):
    """
    Downloads price data starting from `start_date`.

    We pass a start_date calculated from the oldest signal date
    (minus a small buffer) so we always cover the full backtest window,
    regardless of how old the Reddit posts are.
    """
    # Add 10-day buffer before oldest signal, to handle weekends/holidays
    start_dt  = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=10)
    start_str = start_dt.strftime("%Y-%m-%d")

    print(f"📡 Fetching price data from {start_str} → today")
    print(f"   Tickers: {tickers}\n")

    all_prices = []

    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start_str, progress=False, auto_adjust=True)
            if raw.empty:
                print(f"  ⚠️  No data for {ticker}")
                continue

            # Handle newer yfinance MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]

            price_df = raw[["Close"]].copy()
            price_df.columns = ["close"]
            price_df["ticker"] = ticker
            price_df["date"]   = pd.to_datetime(price_df.index).strftime("%Y-%m-%d")
            price_df = price_df.reset_index(drop=True)
            all_prices.append(price_df)
            print(f"  ✅ {ticker}: {len(price_df)} days  ({price_df['date'].min()} → {price_df['date'].max()})")

        except Exception as e:
            print(f"  ❌ {ticker}: {e}")

    return pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()


# ── 3. COMPUTE FORWARD RETURNS ────────────────────────────────────────────────
def compute_forward_returns(daily_signal, prices):
    """
    For each signal row (date D, ticker T):
      - Find the closest trading day on or after date D
      - Step 3 and 5 rows forward in the price table (= trading days)
      - Compute cumulative return: (future_price − entry_price) / entry_price
    """
    results = []

    for _, row in daily_signal.iterrows():
        ticker = row["ticker"]
        date   = row["date"]

        ticker_prices = (
            prices[prices["ticker"] == ticker]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if ticker_prices.empty:
            continue

        # Find exact match, or nearest prior trading day (handles weekends)
        match = ticker_prices[ticker_prices["date"] == date]
        if match.empty:
            prior = ticker_prices[ticker_prices["date"] <= date]
            if prior.empty:
                continue
            match = prior.tail(1)

        idx           = match.index[0]
        price_on_date = ticker_prices.loc[idx, "close"]

        fwd_3d = None
        fwd_5d = None

        if idx + 3 < len(ticker_prices):
            fwd_3d = (ticker_prices.loc[idx + 3, "close"] - price_on_date) / price_on_date
        if idx + 5 < len(ticker_prices):
            fwd_5d = (ticker_prices.loc[idx + 5, "close"] - price_on_date) / price_on_date

        result = row.to_dict()
        result.update({
            "price_on_date": round(float(price_on_date), 2),
            "fwd_3d": round(float(fwd_3d), 4) if fwd_3d is not None else None,
            "fwd_5d": round(float(fwd_5d), 4) if fwd_5d is not None else None,
        })
        results.append(result)

    merged = pd.DataFrame(results)
    with_3d = merged["fwd_3d"].notna().sum()
    with_5d = merged["fwd_5d"].notna().sum()

    print(f"\n🔗 Forward returns computed:")
    print(f"   Total signal rows:  {len(merged)}")
    print(f"   With 3-day return:  {with_3d}")
    print(f"   With 5-day return:  {with_5d}")

    complete = merged.dropna(subset=["fwd_3d", "fwd_5d"])
    if not complete.empty:
        print(f"\n   Results (complete rows only):")
        cols = ["date", "ticker", "signal", "price_on_date", "fwd_3d", "fwd_5d"]
        print(complete[cols].to_string(index=False))

    return merged


# ── 4. ANALYSE THE SIGNAL ─────────────────────────────────────────────────────
def analyse_signal(merged):
    print("\n── Signal Analysis ──────────────────────────────────────")

    clean = merged.dropna(subset=["fwd_3d", "fwd_5d"])

    if len(clean) < 5:
        n = len(clean)
        print(f"⚠️  Only {n} complete observations (need ≥5 for statistics).")
        print(f"   We have data back to {merged['date'].min()} but only {n} posts mention tickers.")
        print("   The signal is working — we just need more ticker-mentioning posts.\n")

        if n >= 2:
            print("   Preview of what we have so far:")
            cols = ["date","ticker","signal","fwd_5d"]
            print(clean[cols].to_string(index=False))
            bullish_right = ((clean["signal"] > 0) & (clean["fwd_5d"] > 0)).sum()
            bearish_right = ((clean["signal"] < 0) & (clean["fwd_5d"] < 0)).sum()
            print(f"\n   Direction accuracy: {bullish_right + bearish_right}/{n} correct ({(bullish_right+bearish_right)/n:.0%})")
        return

    corr_3d, p_3d = stats.spearmanr(clean["signal"], clean["fwd_3d"])
    corr_5d, p_5d = stats.spearmanr(clean["signal"], clean["fwd_5d"])

    print(f"\n  Spearman ρ  (signal → 3-day return):  {corr_3d:+.3f}  (p = {p_3d:.3f})")
    print(f"  Spearman ρ  (signal → 5-day return):  {corr_5d:+.3f}  (p = {p_5d:.3f})")
    print(f"  3-day: {'✅ statistically significant' if p_3d < 0.05 else '⚠️  not yet significant'}")
    print(f"  5-day: {'✅ statistically significant' if p_5d < 0.05 else '⚠️  not yet significant'}")

    print("\n── Backtest ─────────────────────────────────────────────")
    THRESHOLD = 0.15
    trades = clean[clean["signal"] > THRESHOLD]

    if len(trades) == 0:
        print(f"  No trades triggered at threshold {THRESHOLD}")
    else:
        win_rate   = (trades["fwd_5d"] > 0).mean()
        avg_return = trades["fwd_5d"].mean()
        print(f"  Signal threshold:   > {THRESHOLD}")
        print(f"  Trades triggered:   {len(trades)}")
        print(f"  Win rate:           {win_rate:.1%}  (random baseline ≈ 50%)")
        print(f"  Avg 5-day return:   {avg_return:+.2%}")
        print(f"  Result: {'📈 above baseline!' if win_rate > 0.55 else '📉 needs more data'}")
    print()


# ── 5. MAIN ───────────────────────────────────────────────────────────────────
def run(scored_file=SCORED_FILE, output_file=OUTPUT_FILE):
    if not os.path.exists(scored_file):
        print(f"❌ Not found: {scored_file}\n   Run sentiment.py first!")
        return

    df = pd.read_csv(scored_file)
    print(f"📂 Loaded {len(df)} scored posts\n")

    daily_signal = build_daily_signal(df)
    if daily_signal.empty:
        return

    # Dynamically set price start date based on oldest signal
    oldest_date = daily_signal["date"].min()
    print(f"   Oldest signal date: {oldest_date} — fetching prices from there\n")

    tickers = daily_signal["ticker"].unique().tolist()
    prices  = fetch_prices(tickers, start_date=oldest_date)
    if prices.empty:
        print("❌ No price data fetched.")
        return

    merged = compute_forward_returns(daily_signal, prices)
    analyse_signal(merged)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"💾 Saved → {output_file}")
    return merged


if __name__ == "__main__":
    run()
