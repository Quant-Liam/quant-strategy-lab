from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from btc15m import (
    CoinbaseClient,
    append_trade,
    compute_market_features,
    compute_order_book_features,
    current_15m_window,
    decide_trade_side,
    ensure_log,
    infer_window_start_price,
    kelly_fraction_binary,
    load_trades,
    order_book_ladder,
    predict_up_probability,
    settle_open_trades,
    summarize,
)
from btc15m.external import (
    fetch_binance_funding_zscore,
    fetch_binance_liquidation_imbalance,
    fetch_cryptocompare_news_shock,
)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_candles_cached(product_id: str, granularity: int, limit: int) -> pd.DataFrame:
    return CoinbaseClient(timeout=10).fetch_candles(
        product_id=product_id,
        granularity=granularity,
        limit=limit,
    )


@st.cache_data(ttl=2, show_spinner=False)
def fetch_live_price_cached(product_id: str) -> float:
    return CoinbaseClient(timeout=10).fetch_live_price(product_id=product_id)


@st.cache_data(ttl=3, show_spinner=False)
def fetch_order_book_cached(product_id: str, level: int):
    return CoinbaseClient(timeout=10).fetch_order_book(product_id=product_id, level=level)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_funding_cached(symbol: str, lookback: int) -> float:
    return fetch_binance_funding_zscore(symbol=symbol, lookback=lookback)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_liquidation_cached(symbol: str, lookback_hours: int) -> float:
    return fetch_binance_liquidation_imbalance(symbol=symbol, lookback_hours=lookback_hours)


@st.cache_data(ttl=90, show_spinner=False)
def fetch_news_cached(lookback_minutes: int) -> float:
    return fetch_cryptocompare_news_shock(lookback_minutes=lookback_minutes)


st.set_page_config(page_title="BTC 15m Decision Console", layout="wide")

st.title("BTC 15m Decision Console")
st.caption(
    "Research dashboard for short-horizon BTC direction signals. "
    "Do not deploy with real capital without paper testing and controls."
)

with st.sidebar:
    st.header("Runtime")
    product_id = st.text_input("Coinbase product", value="BTC-USD")
    refresh_sec = st.slider("Auto refresh (seconds)", min_value=3, max_value=60, value=8, step=1)
    candles_1m_limit = st.slider(
        "1m candles to pull",
        min_value=180,
        max_value=1200,
        value=300,
        step=60,
        help="Larger values are slower because Coinbase only returns candles in chunks.",
    )
    candles_15m_limit = st.slider(
        "15m candles to pull",
        min_value=120,
        max_value=1200,
        value=300,
        step=60,
    )
    order_book_band_bps = st.slider("Order-book wall band (bps)", min_value=10, max_value=150, value=50, step=5)
    depth_levels = st.slider("Order-book levels to display", min_value=10, max_value=75, value=30, step=5)

    st.header("Market Odds")
    market_up_price = st.number_input(
        "UP share price",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        format="%.2f",
    )
    derive_down = st.checkbox("Set DOWN = 1 - UP", value=True)
    market_down_price = 1.0 - market_up_price
    if not derive_down:
        market_down_price = st.number_input(
            "DOWN share price",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            format="%.2f",
        )

    st.header("Risk")
    fee_rate = st.number_input(
        "Fee rate",
        min_value=0.0,
        max_value=0.05,
        value=0.0156,
        step=0.0001,
        format="%.4f",
        help="Use your all-in effective fee estimate.",
    )
    edge_buffer = st.slider("Min edge before trade", min_value=0.0, max_value=0.10, value=0.02, step=0.005)
    fractional_kelly = st.slider("Fractional Kelly", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    max_fraction = st.slider("Max bankroll at risk", min_value=0.01, max_value=1.0, value=0.20, step=0.01)

    st.header("External Signals")
    st.caption("Automatic fetch uses Binance and CryptoCompare public feeds.")

    auto_funding = st.checkbox("Auto funding z-score", value=True)
    manual_funding_zscore = st.slider(
        "Manual funding z-score",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        help="Positive means crowded longs (contrarian bearish).",
    )

    auto_liquidation = st.checkbox("Auto liquidation imbalance", value=True)
    manual_liquidation_imbalance = st.slider(
        "Manual liquidation imbalance",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="(short liquidations - long liquidations) / total.",
    )
    liquidation_lookback_hours = st.slider(
        "Liquidation lookback (hours)",
        min_value=1,
        max_value=72,
        value=24,
        step=1,
    )

    auto_news = st.checkbox("Auto news shock", value=True)
    manual_news_shock = st.slider(
        "Manual news shock",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Positive = bullish fresh news, negative = bearish.",
    )
    news_lookback_minutes = st.slider(
        "News lookback (minutes)",
        min_value=5,
        max_value=240,
        value=30,
        step=5,
    )

    st.header("Paper Trading")
    paper_log_path = st.text_input("Trade log file", value="paper_trades.csv")
    paper_bankroll = st.number_input(
        "Paper bankroll (USD)",
        min_value=10.0,
        max_value=100_000_000.0,
        value=1000.0,
        step=10.0,
    )
    auto_settle = st.checkbox("Auto settle open paper trades", value=True)
    notes = st.text_input("Paper trade note", value="")
    log_trade_clicked = st.button("Log current recommendation")
    settle_now_clicked = st.button("Settle open trades now")

st_autorefresh(interval=refresh_sec * 1000, limit=None, key="auto-refresh")

try:
    candles_1m = fetch_candles_cached(product_id=product_id, granularity=60, limit=candles_1m_limit)
    candles_15m = fetch_candles_cached(product_id=product_id, granularity=900, limit=candles_15m_limit)
    live_price = fetch_live_price_cached(product_id=product_id)
    snapshot = fetch_order_book_cached(product_id=product_id, level=2)
except Exception as exc:
    st.error(f"Failed to pull Coinbase data: {exc}")
    st.stop()

window_start, window_end = current_15m_window()
window_start_price = infer_window_start_price(
    candles_1m=candles_1m,
    window_start=window_start,
    fallback_price=live_price,
)

order_book_features = compute_order_book_features(
    snapshot=snapshot,
    band_bps=order_book_band_bps,
)

funding_zscore = manual_funding_zscore
funding_source = "manual"
if auto_funding:
    try:
        funding_zscore = fetch_funding_cached(symbol="BTCUSDT", lookback=120)
        funding_source = "binance"
    except Exception:
        funding_zscore = manual_funding_zscore
        funding_source = "manual (auto failed)"

liquidation_imbalance = manual_liquidation_imbalance
liquidation_source = "manual"
if auto_liquidation:
    try:
        liquidation_imbalance = fetch_liquidation_cached(
            symbol="BTCUSDT",
            lookback_hours=liquidation_lookback_hours,
        )
        liquidation_source = "binance"
    except Exception:
        liquidation_imbalance = manual_liquidation_imbalance
        liquidation_source = "manual (auto failed)"

news_shock = manual_news_shock
news_source = "manual"
if auto_news:
    try:
        news_shock = fetch_news_cached(lookback_minutes=news_lookback_minutes)
        news_source = "cryptocompare"
    except Exception:
        news_shock = manual_news_shock
        news_source = "manual (auto failed)"

market_features = compute_market_features(
    candles_15m=candles_15m,
    live_price=live_price,
    order_book_features=order_book_features,
    funding_zscore=funding_zscore,
    liquidation_imbalance=liquidation_imbalance,
    news_shock=news_shock,
)

model_out = predict_up_probability(market_features)
p_up = model_out.p_up
p_down = 1.0 - p_up

decision = decide_trade_side(
    p_up=p_up,
    market_up_price=market_up_price,
    market_down_price=market_down_price,
    edge_buffer=edge_buffer,
)

kelly_up = kelly_fraction_binary(
    p_win=p_up,
    share_price=market_up_price,
    fee_rate=fee_rate,
    fractional_kelly=fractional_kelly,
    max_fraction=max_fraction,
)

kelly_down = kelly_fraction_binary(
    p_win=p_down,
    share_price=market_down_price,
    fee_rate=fee_rate,
    fractional_kelly=fractional_kelly,
    max_fraction=max_fraction,
)

selected_kelly = None
if decision.side == "UP":
    selected_kelly = kelly_up
elif decision.side == "DOWN":
    selected_kelly = kelly_down

paper_path = ensure_log(Path(paper_log_path))

should_run_settle = settle_now_clicked
if auto_settle:
    now_utc = pd.Timestamp.now(tz="UTC")
    last_check = st.session_state.get("last_settle_check_utc")
    if last_check is None or (now_utc - last_check).total_seconds() >= 60:
        should_run_settle = True
        st.session_state["last_settle_check_utc"] = now_utc

if should_run_settle:
    _, settled_count = settle_open_trades(path=paper_path, candles_1m=candles_1m)
    if settled_count > 0:
        st.info(f"Settled {settled_count} open paper trade(s).")

if log_trade_clicked:
    if decision.side not in {"UP", "DOWN"}:
        st.warning("No trade was logged because decision is NO_TRADE.")
    elif selected_kelly is None or selected_kelly.fraction <= 0:
        st.warning("No trade was logged because Kelly sizing is 0%.")
    else:
        stake_fraction = float(selected_kelly.fraction)
        stake_usd = float(paper_bankroll * stake_fraction)
        trade_id = append_trade(
            path=paper_path,
            record={
                "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "product_id": product_id,
                "window_start_utc": window_start.isoformat(),
                "window_end_utc": window_end.isoformat(),
                "side": decision.side,
                "market_price": decision.market_price,
                "fee_rate": fee_rate,
                "p_up": p_up,
                "p_down": p_down,
                "p_side": decision.p_side,
                "edge": decision.edge,
                "kelly_fraction": selected_kelly.raw_kelly,
                "stake_fraction": stake_fraction,
                "bankroll_usd": float(paper_bankroll),
                "stake_usd": stake_usd,
                "start_price": float(window_start_price),
                "status": "OPEN",
                "notes": notes,
            },
        )
        st.success(f"Logged paper trade {trade_id} with stake ${stake_usd:,.2f}.")

paper_summary = summarize(paper_path)
paper_trades = load_trades(paper_path)

seconds_left = max(0, int((window_end - pd.Timestamp.now(tz="UTC")).total_seconds()))
price_delta = live_price - window_start_price
price_delta_pct = (price_delta / window_start_price) if window_start_price else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Live Price", f"${live_price:,.2f}", delta=f"{price_delta_pct * 100:.3f}% vs 15m open")
col2.metric("15m Window Open", f"${window_start_price:,.2f}")
col3.metric("P(UP)", f"{p_up * 100:.2f}%")
col4.metric("P(DOWN)", f"{p_down * 100:.2f}%")
if selected_kelly is None:
    col5.metric("Suggested Bet", "0.00%", delta="NO_TRADE")
else:
    col5.metric(
        "Suggested Bet",
        f"{selected_kelly.fraction * 100:.2f}%",
        delta=f"{decision.side} | edge {decision.edge * 100:.2f}%",
    )

st.write(
    f"Active window: `{window_start.strftime('%Y-%m-%d %H:%M:%S UTC')}` -> "
    f"`{window_end.strftime('%Y-%m-%d %H:%M:%S UTC')}` "
    f"({seconds_left}s remaining)"
)

left, right = st.columns((3, 2))

with left:
    st.subheader("Live BTC Price")
    chart_df = candles_1m.tail(180).copy()

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="BTC-USD (1m)",
        )
    )

    fig_price.add_hline(
        y=window_start_price,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="15m start",
        annotation_position="top left",
    )
    fig_price.add_hline(
        y=live_price,
        line_dash="dot",
        line_color="#10b981",
        annotation_text="live",
        annotation_position="bottom left",
    )
    fig_price.add_vline(x=window_start, line_dash="dot", line_color="#93c5fd")
    fig_price.add_vline(x=window_end, line_dash="dot", line_color="#93c5fd")

    fig_price.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=25, b=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_price, use_container_width=True)

with right:
    st.subheader("Order-Book Wall Map")
    ladder = order_book_ladder(snapshot=snapshot, band_bps=order_book_band_bps, levels=depth_levels)
    bids = ladder[ladder["side"] == "bid"].sort_values("price", ascending=False)
    asks = ladder[ladder["side"] == "ask"].sort_values("price", ascending=True)

    fig_book = go.Figure()
    fig_book.add_trace(
        go.Bar(
            x=bids["notional"],
            y=bids["price"],
            orientation="h",
            name="Bids",
            marker_color="#10b981",
            opacity=0.75,
        )
    )
    fig_book.add_trace(
        go.Bar(
            x=-asks["notional"],
            y=asks["price"],
            orientation="h",
            name="Asks",
            marker_color="#ef4444",
            opacity=0.75,
        )
    )
    fig_book.add_hline(y=snapshot.mid, line_dash="dot", line_color="#93c5fd")
    fig_book.update_layout(
        barmode="overlay",
        xaxis_title="Notional (USD), asks shown as negative",
        yaxis_title="Price",
        margin=dict(l=10, r=10, t=25, b=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_book, use_container_width=True)

st.subheader("Decision + Sizing")
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.write(
        {
            "decision": decision.side,
            "reason": decision.reason,
            "edge_up": round(decision.edge_up, 5),
            "edge_down": round(decision.edge_down, 5),
            "selected_edge": round(decision.edge, 5),
            "market_up_price": market_up_price,
            "market_down_price": market_down_price,
            "funding_zscore_used": round(funding_zscore, 4),
            "liquidation_imbalance_used": round(liquidation_imbalance, 4),
            "news_shock_used": round(news_shock, 4),
            "funding_source": funding_source,
            "liquidation_source": liquidation_source,
            "news_source": news_source,
        }
    )

with summary_col2:
    st.write(
        {
            "kelly_up_fraction": round(kelly_up.fraction, 5),
            "kelly_down_fraction": round(kelly_down.fraction, 5),
            "kelly_up_edge": round(kelly_up.edge, 5),
            "kelly_down_edge": round(kelly_down.edge, 5),
            "fee_rate": fee_rate,
            "fractional_kelly": fractional_kelly,
            "max_fraction": max_fraction,
        }
    )

st.subheader("Paper Trading Summary")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Logged Trades", paper_summary.total)
p2.metric("Open", paper_summary.open_count)
p3.metric("Settled", paper_summary.settled_count)
p4.metric("Win Rate", f"{paper_summary.win_rate * 100:.2f}%")
p5.metric("Total PnL", f"${paper_summary.total_pnl:,.2f}")

if not paper_trades.empty:
    show_cols = [
        "created_at_utc",
        "window_start_utc",
        "window_end_utc",
        "side",
        "market_price",
        "stake_usd",
        "status",
        "outcome_side",
        "pnl_usd",
        "return_pct",
        "notes",
    ]
    existing = [c for c in show_cols if c in paper_trades.columns]
    st.dataframe(paper_trades[existing].tail(200), use_container_width=True, hide_index=True)
else:
    st.caption(f"No paper trades logged yet. Log file: {paper_path}")

st.subheader("Feature Contributions")
feature_rows = []
for feature_name, raw_value in market_features.items():
    feature_rows.append(
        {
            "feature": feature_name,
            "raw": raw_value,
            "normalized": model_out.normalized_features.get(feature_name, 0.0),
            "contribution": model_out.contributions.get(feature_name, 0.0),
        }
    )

feature_df = pd.DataFrame(feature_rows)
feature_df["abs_contribution"] = feature_df["contribution"].abs()
feature_df = feature_df.sort_values("abs_contribution", ascending=False).drop(columns=["abs_contribution"])
st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.caption(
    "Signal notes: funding and liquidation use Binance futures data; "
    "news shock uses CryptoCompare headlines with keyword sentiment and recency weighting."
)
