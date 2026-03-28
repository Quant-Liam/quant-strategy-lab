from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from btc15m import (
    CoinbaseClient,
    ParticleFilterConfig,
    append_trade,
    compute_kelly_from_pf,
    compute_particle_filter_frame,
    compute_regime_frame,
    current_15m_window,
    ensure_log,
    infer_window_start_price,
    is_price_above_pf_fair_value,
    is_price_below_pf_fair_value,
    load_trades,
    project_particle_filter_to_time,
    snapshot_from_particle_filter_frame,
    snapshot_from_regime_frame,
    settle_open_trades,
    summarize,
)

CHART_WINDOW_MINUTES = 120
ONE_MINUTE_FETCH_CANDLES = 720
FIFTEEN_MINUTE_MODEL_CANDLES = 320
REGIME_LOOKBACK = 240
PARTICLE_FILTER_LOOKBACK = 240
PARTICLE_FILTER_PARTICLES = 300


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


@st.cache_data(ttl=15, show_spinner=False)
def compute_regime_frame_cached(candles_15m: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return compute_regime_frame(candles_15m=candles_15m, lookback=lookback)


@st.cache_data(ttl=15, show_spinner=False)
def fit_particle_filter_cached(
    candles_15m: pd.DataFrame,
    regime_frame: pd.DataFrame,
    lookback: int,
    num_particles: int,
) -> pd.DataFrame:
    return compute_particle_filter_frame(
        candles_15m=candles_15m,
        regime_frame=regime_frame,
        config=ParticleFilterConfig(
            lookback=lookback,
            num_particles=num_particles,
        ),
    )


def build_pf_kelly_table(sizing_result, bankroll_usd: float) -> pd.DataFrame:
    def _fmt_price(value: float | None) -> str:
        if value is None or not pd.notna(value):
            return "n/a"
        return f"${value:,.2f}"

    def _fmt_pct(value: float | None, decimals: int = 2) -> str:
        if value is None or not pd.notna(value):
            return "n/a"
        return f"{value * 100:.{decimals}f}%"

    return pd.DataFrame(
        [
            {"item": "Trade side", "value": sizing_result.trade_side or "NO_TRADE"},
            {"item": "Regime", "value": sizing_result.regime_label.upper()},
            {"item": "Live BTC price", "value": _fmt_price(sizing_result.live_price)},
            {"item": "PF fair value", "value": _fmt_price(sizing_result.fair_price_pf)},
            {"item": "Raw PF gap", "value": _fmt_price(sizing_result.raw_gap)},
            {"item": "Normalized gap", "value": f"{sizing_result.normalized_gap:.4f}" if pd.notna(sizing_result.normalized_gap) else "n/a"},
            {"item": "PF base win probability", "value": _fmt_pct(sizing_result.p_base, decimals=2)},
            {"item": "Final win probability", "value": _fmt_pct(sizing_result.p_final, decimals=2)},
            {"item": "PF confidence", "value": _fmt_pct(sizing_result.pf_confidence, decimals=1)},
            {"item": "Regime confidence", "value": _fmt_pct(sizing_result.regime_confidence, decimals=1)},
            {"item": "Confidence multiplier", "value": f"{sizing_result.confidence_multiplier:.4f}"},
            {"item": "Binary share price", "value": _fmt_pct(sizing_result.market_share_price, decimals=2)},
            {"item": "Break-even probability", "value": _fmt_pct(sizing_result.break_even_prob, decimals=2)},
            {"item": "Raw Kelly", "value": _fmt_pct(sizing_result.raw_kelly, decimals=2)},
            {"item": "Optimal bet", "value": _fmt_pct(sizing_result.kelly_fraction, decimals=2)},
            {"item": "Optimal bet (USD)", "value": _fmt_price(bankroll_usd * sizing_result.kelly_fraction)},
            {"item": "Expected log growth", "value": f"{sizing_result.expected_log_growth:.6f}"},
            {"item": "No-trade reason", "value": sizing_result.no_trade_reason or "active"},
        ]
    )


st.set_page_config(page_title="BTC 15m Regime Console", layout="wide")

st.title("BTC 15m Regime Console")
st.caption(
    "A live BTC dashboard centered on market regimes, particle-filter fair value, "
    "and simple paper-trade logging."
)

with st.sidebar:
    st.header("Trade Action")
    paper_log_path = st.text_input("Trade log file", value="paper_trades.csv")
    paper_side = st.radio("Paper side", options=["UP", "DOWN"], horizontal=True)
    notes = st.text_input("Paper trade note", value="")
    log_trade_clicked = st.button("Log current trade", use_container_width=True)
    settle_now_clicked = st.button("Settle open trades now", use_container_width=True)
    auto_settle = st.checkbox("Auto settle open paper trades", value=True)

    st.header("Runtime")
    product_id = st.text_input("Coinbase product", value="BTC-USD")
    refresh_sec = st.slider("Auto refresh (seconds)", min_value=3, max_value=60, value=8, step=1)
    st.caption(
        f"The chart shows the last {CHART_WINDOW_MINUTES // 60} hours of BTC price action. "
        "The regime and particle-filter models still run on the full historical lookback."
    )

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

    st.header("Kelly Sizing")
    bankroll_usd = st.number_input(
        "Sizing bankroll (USD)",
        min_value=10.0,
        max_value=100_000_000.0,
        value=1000.0,
        step=10.0,
    )
    fee_rate = st.number_input(
        "Fee rate",
        min_value=0.0,
        max_value=0.05,
        value=0.0156,
        step=0.0001,
        format="%.4f",
        help="All-in fee estimate for the binary contract.",
    )
    alpha = st.number_input(
        "PF sigmoid alpha",
        min_value=0.1,
        max_value=10.0,
        value=1.5,
        step=0.1,
        format="%.2f",
        help="Controls how aggressively the normalized PF gap maps into p(win).",
    )
    min_gap_scale = st.number_input(
        "PF min gap scale",
        min_value=0.0001,
        max_value=0.05,
        value=0.0010,
        step=0.0001,
        format="%.4f",
        help="Minimum denominator scale as a fraction of BTC price.",
    )
    use_confidence_shrink = st.checkbox("Shrink probability by PF x regime confidence", value=True)
    fractional_kelly = st.slider("Fractional Kelly", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    max_fraction = st.slider("Max bankroll at risk", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
    st.caption(
        "Win probability is derived from the particle-filter fair value gap, "
        "then Kelly sizes the binary market position against the current share price."
    )

st_autorefresh(interval=refresh_sec * 1000, limit=None, key="auto-refresh")

try:
    candles_1m = fetch_candles_cached(product_id=product_id, granularity=60, limit=ONE_MINUTE_FETCH_CANDLES)
    candles_15m = fetch_candles_cached(
        product_id=product_id,
        granularity=900,
        limit=FIFTEEN_MINUTE_MODEL_CANDLES,
    )
    live_price = fetch_live_price_cached(product_id=product_id)
except Exception as exc:
    st.error(f"Failed to pull Coinbase data: {exc}")
    st.stop()

window_start, window_end = current_15m_window()
window_start_price = infer_window_start_price(
    candles_1m=candles_1m,
    window_start=window_start,
    fallback_price=live_price,
)

regime_frame = compute_regime_frame_cached(candles_15m=candles_15m, lookback=REGIME_LOOKBACK)
regime_snapshot = snapshot_from_regime_frame(regime_frame)
pf_frame = fit_particle_filter_cached(
    candles_15m=candles_15m,
    regime_frame=regime_frame,
    lookback=PARTICLE_FILTER_LOOKBACK,
    num_particles=PARTICLE_FILTER_PARTICLES,
)
pf_snapshot = snapshot_from_particle_filter_frame(pf_frame)

pf_last_timestamp = candles_15m.index[-1]
if pf_last_timestamp.tzinfo is None:
    pf_last_timestamp = pf_last_timestamp.tz_localize("UTC")
else:
    pf_last_timestamp = pf_last_timestamp.tz_convert("UTC")

live_pf_fair_price = project_particle_filter_to_time(
    snapshot=pf_snapshot,
    last_observation_time=pf_last_timestamp,
    target_time=pd.Timestamp.now(tz="UTC"),
)
live_pf_gap = live_price - live_pf_fair_price
live_pf_gap_pct = (live_pf_gap / live_pf_fair_price) if live_pf_fair_price > 0 else 0.0
pf_price_below_live_fair = is_price_below_pf_fair_value(live_price, live_pf_fair_price)
pf_price_above_live_fair = is_price_above_pf_fair_value(live_price, live_pf_fair_price)

active_share_price = market_up_price
if regime_snapshot.allowed_side == "DOWN":
    active_share_price = market_down_price

pf_kelly = compute_kelly_from_pf(
    live_price=live_price,
    fair_price_pf=live_pf_fair_price,
    pf_uncertainty=pf_snapshot.uncertainty,
    pf_confidence=pf_snapshot.confidence,
    regime_label=regime_snapshot.regime,
    regime_confidence=regime_snapshot.confidence,
    market_share_price=active_share_price,
    fee_rate=fee_rate,
    alpha=alpha,
    min_gap_scale=min_gap_scale,
    fractional_kelly=fractional_kelly,
    max_fraction=max_fraction,
    use_confidence_shrink=use_confidence_shrink,
)

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
    entry_time = pd.Timestamp.now(tz="UTC")
    trade_id = append_trade(
        path=paper_path,
        record={
            "created_at_utc": entry_time.isoformat(),
            "product_id": product_id,
            "window_start_utc": entry_time.isoformat(),
            "window_end_utc": window_end.isoformat(),
            "side": paper_side,
            "regime": regime_snapshot.regime,
            "regime_allowed_side": regime_snapshot.allowed_side,
            "regime_confidence": regime_snapshot.confidence,
            "start_price": float(live_price),
            "spot_price": float(live_price),
            "status": "OPEN",
            "notes": notes,
        },
    )
    st.success(
        f"Logged {paper_side} paper trade {trade_id} at BTC ${live_price:,.2f}. "
        f"It will settle at the close of this 15-minute window."
    )

paper_summary = summarize(paper_path)
paper_trades = load_trades(paper_path)

seconds_left = max(0, int((window_end - pd.Timestamp.now(tz="UTC")).total_seconds()))
price_delta = live_price - window_start_price
price_delta_pct = (price_delta / window_start_price) if window_start_price else 0.0
pf_win_label = "n/a" if pf_kelly.p_final is None else f"{pf_kelly.p_final * 100:.2f}%"
optimal_bet_usd = bankroll_usd * pf_kelly.kelly_fraction

metric_cols = st.columns(8)
metric_cols[0].metric("Live Coinbase BTC", f"${live_price:,.2f}", delta=f"{price_delta_pct * 100:.3f}% vs 15m open")
metric_cols[1].metric("15m Window Open", f"${window_start_price:,.2f}")
metric_cols[2].metric(
    "Regime",
    regime_snapshot.regime.upper(),
    delta=f"{regime_snapshot.allowed_side} | {regime_snapshot.confidence * 100:.1f}%",
)
metric_cols[3].metric("PF Fair Price", f"${live_pf_fair_price:,.2f}", delta=f"{pf_snapshot.regime.upper()} context")
metric_cols[4].metric("Spot - PF", f"${live_pf_gap:,.2f}", delta=f"{live_pf_gap_pct * 100:.3f}%")
metric_cols[5].metric("P(Win)", pf_win_label)
metric_cols[6].metric("Optimal Bet", f"{pf_kelly.kelly_fraction * 100:.2f}%", delta=f"${optimal_bet_usd:,.2f}")
metric_cols[7].metric("Window Time Left", f"{seconds_left}s")

st.write(
    f"Active window: `{window_start.strftime('%Y-%m-%d %H:%M:%S UTC')}` -> "
    f"`{window_end.strftime('%Y-%m-%d %H:%M:%S UTC')}` "
    f"({seconds_left}s remaining)"
)
st.caption(
    f"Regime reason: {regime_snapshot.reason} "
    f"Chart is focused on the last {CHART_WINDOW_MINUTES // 60} hours of BTC price action. "
    f"Live PF gap is price minus fair value."
)

if regime_snapshot.allowed_side == "NO_TRADE":
    st.info("Neutral regime is active, so directional regime trading is paused until the model leaves neutral.")
elif regime_snapshot.allowed_side == "UP":
    st.success("Bull regime is active, so the regime layer currently favors UP setups.")
else:
    st.success("Bear regime is active, so the regime layer currently favors DOWN setups.")

if pf_kelly.no_trade_reason:
    st.info(f"Kelly sizing is inactive: `{pf_kelly.no_trade_reason}`.")
else:
    st.success(
        f"PF-derived Kelly sizing is active for {pf_kelly.trade_side}. "
        f"Optimal bet is {pf_kelly.kelly_fraction * 100:.2f}% of bankroll."
    )

if regime_snapshot.regime == "bull":
    if pf_price_below_live_fair:
        st.success("Particle filter says live price is below bull-regime fair value, so the regime-consistent long setup is active.")
    else:
        st.info("Particle filter says live price is not below bull-regime fair value right now.")
elif regime_snapshot.regime == "bear":
    if pf_price_above_live_fair:
        st.success("Particle filter says live price is above bear-regime fair value, so the regime-consistent short setup is active.")
    else:
        st.info("Particle filter says live price is not above bear-regime fair value right now.")

chart_col, status_col = st.columns((7, 4))

with chart_col:
    st.subheader("Live BTC Price")
    now_utc = pd.Timestamp.now(tz="UTC")
    chart_start = now_utc - pd.Timedelta(minutes=CHART_WINDOW_MINUTES)
    chart_df = candles_1m.loc[candles_1m.index >= chart_start].copy()
    if chart_df.empty:
        chart_df = candles_1m.tail(CHART_WINDOW_MINUTES).copy()
    chart_start = chart_df.index.min()
    chart_end = max(now_utc, chart_df.index.max())

    pf_plot = pf_frame.loc[pf_frame.index >= chart_start].copy()
    if not pf_plot.empty and now_utc > pf_plot.index.max():
        live_pf_row = pd.DataFrame(
            {"pf_fair_price": [live_pf_fair_price]},
            index=pd.DatetimeIndex([now_utc]),
        )
        pf_plot = pd.concat([pf_plot[["pf_fair_price"]], live_pf_row])

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
    if not pf_plot.empty:
        fig_price.add_trace(
            go.Scatter(
                x=pf_plot.index,
                y=pf_plot["pf_fair_price"],
                mode="lines+markers",
                name="PF fair price",
                line=dict(color="#f97316", width=2),
                marker=dict(size=6),
            )
        )
    fig_price.add_trace(
        go.Scatter(
            x=[window_start],
            y=[window_start_price],
            mode="markers",
            name="15m open",
            marker=dict(color="#f59e0b", size=8, symbol="diamond"),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=[now_utc],
            y=[live_price],
            mode="markers",
            name="Live BTC",
            marker=dict(color="#10b981", size=9),
        )
    )
    fig_price.add_hline(
        y=window_start_price,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="current 15m open",
        annotation_position="top left",
    )
    fig_price.add_vrect(
        x0=window_start,
        x1=window_end,
        fillcolor="#dbeafe",
        opacity=0.08,
        line_width=0,
    )
    fig_price.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=25, b=10),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_price.update_xaxes(range=[chart_start, chart_end], title_text="UTC time")
    fig_price.update_yaxes(title_text="Price")
    st.plotly_chart(fig_price, use_container_width=True)

with status_col:
    st.subheader("Signal Status")
    status_metrics = st.columns(2)
    status_metrics[0].metric("Allowed Side", regime_snapshot.allowed_side)
    status_metrics[1].metric("PF Gap", f"${live_pf_gap:,.2f}")

    status_metrics = st.columns(2)
    status_metrics[0].metric("PF Confidence", f"{pf_snapshot.confidence * 100:.1f}%")
    status_metrics[1].metric("PF Drift", f"{pf_snapshot.drift * 100:.3f}% / 15m")

    status_metrics = st.columns(2)
    status_metrics[0].metric("PF Uncertainty", f"${pf_snapshot.uncertainty:,.2f}")
    status_metrics[1].metric("Window Time Left", f"{seconds_left}s")

    st.markdown(f"**Regime:** `{regime_snapshot.regime.upper()}`")
    st.markdown(f"**Reason:** {regime_snapshot.reason}")

    st.subheader("Kelly Sizing")
    st.dataframe(build_pf_kelly_table(pf_kelly, bankroll_usd), use_container_width=True, hide_index=True)

    st.subheader("Particle Filter")
    st.dataframe(
        pd.DataFrame(
            [
                {"item": "PF fair price", "value": f"${live_pf_fair_price:,.2f}"},
                {"item": "Real price - PF fair price", "value": f"${live_pf_gap:,.2f}"},
                {"item": "Gap percent", "value": f"{live_pf_gap_pct * 100:.3f}%"},
                {"item": "Price below PF fair value", "value": pf_price_below_live_fair},
                {"item": "Price above PF fair value", "value": pf_price_above_live_fair},
                {"item": "Bull long setup", "value": regime_snapshot.regime == "bull" and pf_price_below_live_fair},
                {"item": "Bear short setup", "value": regime_snapshot.regime == "bear" and pf_price_above_live_fair},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Regime Model")
    regime_probs = pd.DataFrame(
        [
            {"state": "bull", "probability": regime_snapshot.probabilities["bull"]},
            {"state": "neutral", "probability": regime_snapshot.probabilities["neutral"]},
            {"state": "bear", "probability": regime_snapshot.probabilities["bear"]},
        ]
    )
    regime_probs["probability"] = regime_probs["probability"].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(regime_probs, use_container_width=True, hide_index=True)
    st.caption(
        f"GARCH volatility: {regime_snapshot.garch_volatility * 100:.3f}% per 15m | "
        f"vol ratio: {regime_snapshot.garch_volatility_ratio:.2f} | "
        f"fit status: {regime_snapshot.fit_status}"
    )

st.subheader("Paper Trading")
paper_cols = st.columns(6)
paper_cols[0].metric("Logged Trades", paper_summary.total)
paper_cols[1].metric("Open", paper_summary.open_count)
paper_cols[2].metric("Settled", paper_summary.settled_count)
paper_cols[3].metric("Win Rate", f"{paper_summary.win_rate * 100:.2f}%")
paper_cols[4].metric("Avg Return", f"{paper_summary.avg_return_pct * 100:.3f}%")
paper_cols[5].metric("Net Move", f"${paper_summary.net_price_move:,.2f}")

if not paper_trades.empty:
    show_cols = [
        "created_at_utc",
        "window_end_utc",
        "side",
        "start_price",
        "end_price",
        "status",
        "outcome_side",
        "return_pct",
        "notes",
    ]
    existing = [column for column in show_cols if column in paper_trades.columns]
    paper_display = paper_trades[existing].tail(200).copy()
    paper_display = paper_display.rename(
        columns={
            "created_at_utc": "logged_at_utc",
            "window_end_utc": "settles_at_utc",
            "start_price": "entry_price",
            "end_price": "exit_price",
            "return_pct": "trade_return_pct",
        }
    )
    st.dataframe(paper_display, use_container_width=True, hide_index=True)
else:
    st.caption(f"No paper trades logged yet. Log file: {paper_path}")
