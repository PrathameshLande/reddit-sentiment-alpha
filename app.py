# app.py — Phase 5
# Streamlit dashboard — run with:  streamlit run app.py
#
# Streamlit works top-to-bottom like a script.
# Every widget (slider, dropdown, button) triggers a full re-run from top.
# Think of it as: "re-run this script every time the user touches anything."

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import sys

# ── PAGE CONFIG (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title = "Reddit Alpha Signal",
    page_icon  = "📈",
    layout     = "wide",
)

# ── FILE PATHS ────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
RAW_FILE  = os.path.join(BASE, "data", "raw",       "wsb_posts.csv")
SCORED    = os.path.join(BASE, "data", "processed", "scored_posts.csv")
SIGNALS   = os.path.join(BASE, "data", "processed", "signals.csv")

# ── DATA LOADING ──────────────────────────────────────────────────────────────
# @st.cache_data tells Streamlit: "only reload this if the file changes"
# Without it, the CSV would reload on every single user interaction — very slow.
@st.cache_data
def load_data():
    scored  = pd.read_csv(SCORED)  if os.path.exists(SCORED)  else pd.DataFrame()
    signals = pd.read_csv(SIGNALS) if os.path.exists(SIGNALS) else pd.DataFrame()
    return scored, signals

scored, signals = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📈 Reddit Sentiment → Alpha Signal")
st.caption(
    "NLP-powered pipeline: scrapes r/WallStreetBets → scores posts with FinBERT "
    "→ compares sentiment against real stock price moves."
)
st.divider()

# ── TOP METRICS ROW ───────────────────────────────────────────────────────────
# st.columns splits the page into N equal columns side by side
col1, col2, col3, col4 = st.columns(4)

total_posts     = len(scored)
posts_w_tickers = int(scored["ticker_count"].gt(0).sum()) if not scored.empty else 0
avg_finbert     = round(float(scored["finbert_score"].mean()), 3) if not scored.empty else 0
complete_obs    = int(signals.dropna(subset=["fwd_5d"]).shape[0]) if not signals.empty else 0

col1.metric("Posts Scored",        total_posts)
col2.metric("Posts with Tickers",  posts_w_tickers)
col3.metric("FinBERT Avg Score",   f"{avg_finbert:+.3f}",
            help="Near zero = neutral overall. Positive = community is bullish.")
col4.metric("Complete Observations", complete_obs,
            help="Signal rows with confirmed 5-day forward returns (for backtesting).")

st.divider()

# ── TABS: organise the dashboard into sections ────────────────────────────────
# Tabs keep the layout clean — each section gets its own tab
tab1, tab2, tab3 = st.tabs(["🧠 Sentiment Overview", "📊 Signal vs Returns", "📋 Raw Data"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SENTIMENT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if scored.empty:
        st.warning("No scored data found. Run `python src/sentiment.py` first.")
    else:
        col_left, col_right = st.columns(2)

        # ── Label distribution pie chart ──────────────────────────────────────
        with col_left:
            st.subheader("Overall Sentiment Split")
            label_counts = scored["finbert_label"].value_counts().reset_index()
            label_counts.columns = ["label", "count"]

            # Colour map: green=bullish, red=bearish, grey=neutral
            colour_map = {"BULLISH": "#26a269", "BEARISH": "#e01b24", "NEUTRAL": "#9a9996"}
            fig_pie = px.pie(
                label_counts,
                names  = "label",
                values = "count",
                color  = "label",
                color_discrete_map = colour_map,
                hole   = 0.4,   # donut style
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── FinBERT vs VADER scatter ───────────────────────────────────────────
        with col_right:
            st.subheader("FinBERT vs VADER Scores")
            st.caption("Each dot is one post. Perfect agreement = diagonal line.")
            fig_scatter = px.scatter(
                scored,
                x     = "vader_score",
                y     = "finbert_score",
                color = "finbert_label",
                color_discrete_map = colour_map,
                hover_data = ["title"],
                opacity    = 0.6,
                labels = {"vader_score": "VADER Score", "finbert_score": "FinBERT Score"},
            )
            # Add a reference line at y=0 and x=0
            fig_scatter.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5)
            fig_scatter.add_vline(x=0, line_dash="dot", line_color="grey", opacity=0.5)
            fig_scatter.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Posts by date (volume chart) ──────────────────────────────────────
        st.subheader("Post Volume & Sentiment Over Time")
        daily_avg = (
            scored.groupby("date")
            .agg(post_count=("title","count"), avg_sentiment=("finbert_score","mean"))
            .reset_index()
        )

        fig_time = go.Figure()
        # Bar chart for post volume
        fig_time.add_trace(go.Bar(
            x    = daily_avg["date"],
            y    = daily_avg["post_count"],
            name = "Post Count",
            marker_color = "lightblue",
            opacity = 0.6,
            yaxis = "y2",
        ))
        # Line chart for avg sentiment
        fig_time.add_trace(go.Scatter(
            x    = daily_avg["date"],
            y    = daily_avg["avg_sentiment"],
            name = "Avg FinBERT Score",
            line = dict(color="#26a269", width=2),
            mode = "lines+markers",
        ))
        fig_time.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.4)
        fig_time.update_layout(
            yaxis  = dict(title="Avg Sentiment Score", range=[-1, 1]),
            yaxis2 = dict(title="Post Count", overlaying="y", side="right"),
            legend = dict(orientation="h", y=1.1),
            margin = dict(t=20, b=40),
            hovermode = "x unified",
        )
        st.plotly_chart(fig_time, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIGNAL vs RETURNS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if signals.empty:
        st.warning("No signal data found. Run `python src/signals.py` first.")
    else:
        complete = signals.dropna(subset=["fwd_5d"])

        if complete.empty:
            st.info(
                "No complete signal observations yet — all recent signals are "
                "still waiting for 5-day forward price data. "
                "Check back in a few trading days!"
            )
        else:
            st.subheader("Sentiment Signal vs 5-Day Forward Return")
            st.caption(
                "Each point = one ticker on one date. "
                "If the signal works, bullish posts (right) should cluster above zero "
                "and bearish posts (left) should cluster below zero."
            )

            colour_map = {"BULLISH": "#26a269", "BEARISH": "#e01b24", "NEUTRAL": "#9a9996"}
            complete["direction"] = complete["signal"].apply(
                lambda s: "BULLISH" if s > 0.15 else ("BEARISH" if s < -0.15 else "NEUTRAL")
            )

            fig_sig = px.scatter(
                complete,
                x          = "signal",
                y          = "fwd_5d",
                color      = "direction",
                color_discrete_map = colour_map,
                text       = "ticker",
                size       = [12] * len(complete),
                hover_data = ["date", "price_on_date", "fwd_3d"],
                labels = {
                    "signal": "FinBERT Sentiment Signal (weighted)",
                    "fwd_5d": "5-Day Forward Return",
                },
            )
            fig_sig.add_hline(y=0,  line_dash="dot", line_color="grey", opacity=0.5)
            fig_sig.add_vline(x=0,  line_dash="dot", line_color="grey", opacity=0.5)
            fig_sig.update_traces(textposition="top center")
            fig_sig.update_layout(margin=dict(t=20, b=40))
            st.plotly_chart(fig_sig, use_container_width=True)

            # ── Summary table ─────────────────────────────────────────────────
            st.subheader("Observation Details")
            display = complete[["date","ticker","signal","price_on_date","fwd_3d","fwd_5d"]].copy()
            display["signal"]  = display["signal"].round(3)
            display["fwd_3d"]  = (display["fwd_3d"] * 100).round(2).astype(str) + "%"
            display["fwd_5d"]  = (display["fwd_5d"] * 100).round(2).astype(str) + "%"
            st.dataframe(display, use_container_width=True, hide_index=True)

            # ── Direction accuracy ────────────────────────────────────────────
            n = len(complete)
            correct = (
                ((complete["signal"] > 0.15) & (complete["fwd_5d"] > 0)) |
                ((complete["signal"] < -0.15) & (complete["fwd_5d"] < 0))
            ).sum()
            st.metric(
                "Directional Accuracy",
                f"{correct}/{n}  ({correct/n:.0%})",
                help="Among strong-signal posts, how often did the direction match?",
            )

            st.info(
                "📌 **Analyst note:** With fewer than 30 observations, directional "
                "accuracy is illustrative, not statistically conclusive. "
                "Run the scraper daily for 4–8 weeks to reach significance."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Scored Posts")

    if scored.empty:
        st.warning("No data yet.")
    else:
        # Filter controls
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            label_filter = st.multiselect(
                "Filter by sentiment label",
                options = ["BULLISH", "NEUTRAL", "BEARISH"],
                default = ["BULLISH", "NEUTRAL", "BEARISH"],
            )
        with col_f2:
            tickers_only = st.checkbox("Show only posts with tickers", value=False)
        with col_f3:
            min_score = st.slider("Min upvote score", 0, 10000, 0, step=100)

        filtered = scored[scored["finbert_label"].isin(label_filter)]
        if tickers_only:
            filtered = filtered[filtered["ticker_count"] > 0]
        filtered = filtered[filtered["score"] >= min_score]

        st.caption(f"Showing {len(filtered)} of {len(scored)} posts")

        display_cols = ["date", "finbert_label", "finbert_score", "vader_score",
                        "score", "tickers", "title"]
        st.dataframe(
            filtered[display_cols].sort_values("finbert_score", ascending=False),
            use_container_width = True,
            hide_index = True,
        )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Python · FinBERT (ProsusAI) · PRAW · yfinance · Streamlit · Plotly  "
    "| Data source: r/WallStreetBets (public API)"
)
