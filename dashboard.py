from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from btc15m import (
    CoinbaseClient,
    ParticleFilterConfig,
    append_trade,
    compute_particle_filter_frame,
    compute_regime_frame,
    compute_market_features,
    compute_order_book_features,
    current_15m_window,
    decide_trade_side,
    ensure_log,
    infer_window_start_price,
    is_price_above_pf_fair_value,
    is_price_below_pf_fair_value,
    kelly_fraction_binary,
    load_trades,
    predict_up_probability,
    project_particle_filter_to_time,
    snapshot_from_particle_filter_frame,
    snapshot_from_regime_frame,
    settle_open_trades,
    summarize,
)
from btc15m.external import (
    fetch_binance_funding_zscore,
    fetch_binance_liquidation_imbalance,
    fetch_cryptocompare_news_shock,
)

ONE_MINUTE_CHART_CANDLES = 720
FIFTEEN_MINUTE_MODEL_CANDLES = 320
ORDER_BOOK_BAND_BPS = 50
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


def build_feature_table(
    market_features: dict[str, float],
    normalized_features: dict[str, float],
    contributions: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for feature_name, raw_value in market_features.items():
        rows.append(
            {
                "feature": feature_name,
                "raw": raw_value,
                "normalized": normalized_features.get(feature_name, 0.0),
                "contribution": contributions.get(feature_name, 0.0),
            }
        )

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        return feature_df
    feature_df["abs_contribution"] = feature_df["contribution"].abs()
    feature_df = feature_df.sort_values("abs_contribution", ascending=False).drop(columns=["abs_contribution"])
    return feature_df


def build_kelly_table(
    p_win: float,
    share_price: float,
    fee_rate: float,
    fractional_kelly: float,
    max_fraction: float,
    kelly_result,
    applied_fraction: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Model win probability", "value": f"{p_win * 100:.2f}%"},
            {"item": "Market share price", "value": f"{share_price * 100:.2f}%"},
            {"item": "Fee rate", "value": f"{fee_rate * 100:.2f}%"},
            {"item": "Effective share price (with fees)", "value": f"{kelly_result.effective_share_price * 100:.2f}%"},
            {"item": "Break-even probability", "value": f"{kelly_result.break_even_prob * 100:.2f}%"},
            {"item": "Net odds", "value": f"{kelly_result.net_odds:.4f}"},
            {"item": "Raw Kelly", "value": f"{kelly_result.raw_kelly * 100:.2f}%"},
            {"item": "Fractional Kelly setting", "value": f"{fractional_kelly * 100:.0f}%"},
            {"item": "Max bankroll cap", "value": f"{max_fraction * 100:.2f}%"},
            {"item": "Kelly candidate size", "value": f"{kelly_result.fraction * 100:.2f}%"},
            {"item": "Applied trade size", "value": f"{applied_fraction * 100:.2f}%"},
            {"item": "Expected log growth", "value": f"{kelly_result.expected_log_growth:.6f}"},
        ]
    )


st.set_page_config(page_title="BTC 15m Regime Console", layout="wide")

st.title("BTC 15m Regime Console")
st.caption(
    "A cleaner live BTC dashboard with regime-gated trading, visible Kelly sizing, "
    "and paper-trade logging. Bull trades only UP, bear trades only DOWN, neutral pauses trading."
)

with st.sidebar:
    st.header("Trade Action")
    paper_log_path = st.text_input("Trade log file", value="paper_trades.csv")
    paper_bankroll = st.number_input(
        "Paper bankroll (USD)",
        min_value=10.0,
        max_value=100_000_000.0,
        value=1000.0,
        step=10.0,
    )
    notes = st.text_input("Paper trade note", value="")
    log_trade_clicked = st.button("Log current trade", use_container_width=True)
    settle_now_clicked = st.button("Settle open trades now", use_container_width=True)
    auto_settle = st.checkbox("Auto settle open paper trades", value=True)

    st.header("Runtime")
    product_id = st.text_input("Coinbase product", value="BTC-USD")
    refresh_sec = st.slider("Auto refresh (seconds)", min_value=3, max_value=60, value=8, step=1)
    st.caption(
        f"The price chart always uses {ONE_MINUTE_CHART_CANDLES} live 1-minute candles. "
        "The signal model uses a fixed 15-minute lookback."
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
    fee_rate = st.number_input(
        "Fee rate",
        min_value=0.0,
        max_value=0.05,
        value=0.0156,
        step=0.0001,
        format="%.4f",
        help="Your all-in effective fee estimate.",
    )
    edge_buffer = st.slider("Min edge before trade", min_value=0.0, max_value=0.10, value=0.02, step=0.005)
    fractional_kelly = st.slider("Fractional Kelly", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    max_fraction = st.slider("Max bankroll at risk", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
    st.caption(
        "Kelly sizing uses the side probability versus market price after fees, "
        "then applies your fractional Kelly and max-risk cap."
    )

    with st.expander("External Signals", expanded=False):
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

st_autorefresh(interval=refresh_sec * 1000, limit=None, key="auto-refresh")

try:
    candles_1m = fetch_candles_cached(product_id=product_id, granularity=60, limit=ONE_MINUTE_CHART_CANDLES)
    candles_15m = fetch_candles_cached(
        product_id=product_id,
        granularity=900,
        limit=FIFTEEN_MINUTE_MODEL_CANDLES,
    )
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
    band_bps=ORDER_BOOK_BAND_BPS,
)

funding_zscore = 0.0
funding_source = "off"
if "auto_funding" in locals():
    funding_zscore = manual_funding_zscore
    funding_source = "manual"
    if auto_funding:
        try:
            funding_zscore = fetch_funding_cached(symbol="BTCUSDT", lookback=120)
            funding_source = "binance"
        except Exception:
            funding_zscore = manual_funding_zscore
            funding_source = "manual (auto failed)"

liquidation_imbalance = 0.0
liquidation_source = "off"
if "auto_liquidation" in locals():
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

news_shock = 0.0
news_source = "off"
if "auto_news" in locals():
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

decision = decide_trade_side(
    p_up=p_up,
    market_up_price=market_up_price,
    market_down_price=market_down_price,
    edge_buffer=edge_buffer,
    regime=regime_snapshot.regime,
    allowed_side=regime_snapshot.allowed_side,
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

regime_kelly = {"UP": kelly_up, "DOWN": kelly_down}.get(regime_snapshot.allowed_side)
display_kelly = selected_kelly or regime_kelly
display_kelly_side = decision.side if decision.side in {"UP", "DOWN"} else regime_snapshot.allowed_side

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
        st.warning(f"No trade was logged. {decision.reason}")
    elif selected_kelly is None or selected_kelly.fraction <= 0:
        st.warning("No trade was logged because the applied Kelly size is 0%.")
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
                "regime": regime_snapshot.regime,
                "regime_allowed_side": regime_snapshot.allowed_side,
                "regime_confidence": regime_snapshot.confidence,
                "market_price": decision.market_price,
                "spot_price": float(live_price),
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
        st.success(
            f"Logged paper trade {trade_id} at live BTC ${live_price:,.2f} "
            f"with stake ${stake_usd:,.2f}."
        )

paper_summary = summarize(paper_path)
paper_trades = load_trades(paper_path)

seconds_left = max(0, int((window_end - pd.Timestamp.now(tz="UTC")).total_seconds()))
price_delta = live_price - window_start_price
price_delta_pct = (price_delta / window_start_price) if window_start_price else 0.0
applied_fraction = float(selected_kelly.fraction) if selected_kelly is not None else 0.0
candidate_fraction = float(display_kelly.fraction) if display_kelly is not None else 0.0
candidate_stake_usd = float(paper_bankroll * candidate_fraction)

metric_cols = st.columns(7)
metric_cols[0].metric("Live Coinbase BTC", f"${live_price:,.2f}", delta=f"{price_delta_pct * 100:.3f}% vs 15m open")
metric_cols[1].metric("15m Window Open", f"${window_start_price:,.2f}")
metric_cols[2].metric(
    "Regime",
    regime_snapshot.regime.upper(),
    delta=f"{regime_snapshot.allowed_side} | {regime_snapshot.confidence * 100:.1f}%",
)
metric_cols[3].metric("PF Fair Price", f"${live_pf_fair_price:,.2f}", delta=f"{pf_snapshot.regime.upper()} context")
metric_cols[4].metric("Spot - PF", f"${live_pf_gap:,.2f}", delta=f"{live_pf_gap_pct * 100:.3f}%")
metric_cols[5].metric("P(UP)", f"{p_up * 100:.2f}%")
metric_cols[6].metric(
    "Applied Trade Size",
    f"{applied_fraction * 100:.2f}%",
    delta=f"{decision.side} | ${paper_bankroll * applied_fraction:,.2f}",
)

st.write(
    f"Active window: `{window_start.strftime('%Y-%m-%d %H:%M:%S UTC')}` -> "
    f"`{window_end.strftime('%Y-%m-%d %H:%M:%S UTC')}` "
    f"({seconds_left}s remaining)"
)
st.caption(
    f"Regime reason: {regime_snapshot.reason} "
    f"Chart is fixed to {ONE_MINUTE_CHART_CANDLES} one-minute candles. "
    f"Live PF gap is price minus fair value."
)

if decision.side == "NO_TRADE":
    if regime_snapshot.allowed_side == "NO_TRADE":
        st.info("Neutral regime is active, so trading is paused until the model leaves neutral.")
    else:
        st.warning(decision.reason)
else:
    st.success(
        f"Current trade is {decision.side}. "
        f"Market price {decision.market_price:.2f}, applied Kelly {applied_fraction * 100:.2f}%."
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
    chart_df = candles_1m.tail(ONE_MINUTE_CHART_CANDLES).copy()
    pf_plot = pf_frame.loc[pf_frame.index >= chart_df.index.min()].copy()

    fig_price = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
    )
    fig_price.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="BTC-USD (1m)",
        ),
        row=1,
        col=1,
    )

    if not pf_plot.empty:
        fig_price.add_trace(
            go.Scatter(
                x=pf_plot.index,
                y=pf_plot["pf_fair_price"],
                mode="lines",
                name="PF fair price",
                line=dict(color="#f97316", width=2),
            ),
            row=1,
            col=1,
        )
        fig_price.add_trace(
            go.Scatter(
                x=pf_plot.index,
                y=pf_plot["pf_gap"],
                mode="lines",
                name="Price - PF",
                line=dict(color="#38bdf8", width=2),
            ),
            row=2,
            col=1,
        )

    fig_price.add_trace(
        go.Scatter(
            x=[pd.Timestamp.now(tz="UTC")],
            y=[live_pf_fair_price],
            mode="markers",
            name="PF live fair",
            marker=dict(color="#f97316", size=9, symbol="diamond"),
        ),
        row=1,
        col=1,
    )
    fig_price.add_hline(
        y=window_start_price,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="15m start",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig_price.add_hline(
        y=live_price,
        line_dash="dot",
        line_color="#10b981",
        annotation_text="live",
        annotation_position="bottom left",
        row=1,
        col=1,
    )
    fig_price.add_hline(y=0, line_dash="dot", line_color="#94a3b8", row=2, col=1)
    fig_price.add_vline(x=window_start, line_dash="dot", line_color="#93c5fd", row=1, col=1)
    fig_price.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=25, b=10),
        height=690,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_price.update_yaxes(title_text="Price", row=1, col=1)
    fig_price.update_yaxes(title_text="Gap", row=2, col=1)
    st.plotly_chart(fig_price, use_container_width=True)

with status_col:
    st.subheader("Trade Setup")
    status_metrics = st.columns(2)
    status_metrics[0].metric("UP Edge", f"{decision.edge_up * 100:.2f}%")
    status_metrics[1].metric("DOWN Edge", f"{decision.edge_down * 100:.2f}%")

    status_metrics = st.columns(2)
    status_metrics[0].metric("UP Market Price", f"{market_up_price * 100:.2f}%")
    status_metrics[1].metric("DOWN Market Price", f"{market_down_price * 100:.2f}%")

    status_metrics = st.columns(2)
    status_metrics[0].metric("Allowed Side", regime_snapshot.allowed_side)
    status_metrics[1].metric("PF Gap", f"${live_pf_gap:,.2f}")

    st.markdown(f"**Decision:** `{decision.side}`")
    st.markdown(f"**Why:** {decision.reason}")

    st.subheader("Kelly Model")
    st.caption(
        "Raw Kelly = (b * p - q) / b. This dashboard then applies your fractional Kelly setting, "
        "caps the size at the bankroll limit, and finally blocks trades that fail the regime or edge rules."
    )
    if display_kelly is None:
        st.caption("Kelly is inactive because the current regime does not allow any trade direction.")
    else:
        kelly_table = build_kelly_table(
            p_win=decision.p_side if decision.side in {"UP", "DOWN"} else (p_up if display_kelly_side == "UP" else p_down),
            share_price=decision.market_price if decision.side in {"UP", "DOWN"} else (market_up_price if display_kelly_side == "UP" else market_down_price),
            fee_rate=fee_rate,
            fractional_kelly=fractional_kelly,
            max_fraction=max_fraction,
            kelly_result=display_kelly,
            applied_fraction=applied_fraction,
        )
        st.dataframe(kelly_table, use_container_width=True, hide_index=True)
        st.caption(f"Kelly candidate stake on the current bankroll: ${candidate_stake_usd:,.2f}")

    st.subheader("Particle Filter")
    pf_metrics = st.columns(2)
    pf_metrics[0].metric("PF Fair Price", f"${live_pf_fair_price:,.2f}")
    pf_metrics[1].metric("PF Confidence", f"{pf_snapshot.confidence * 100:.1f}%")

    pf_metrics = st.columns(2)
    pf_metrics[0].metric("PF Drift", f"{pf_snapshot.drift * 100:.3f}% / 15m")
    pf_metrics[1].metric("PF Uncertainty", f"${pf_snapshot.uncertainty:,.2f}")

    st.dataframe(
        pd.DataFrame(
            [
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
paper_cols = st.columns(5)
paper_cols[0].metric("Logged Trades", paper_summary.total)
paper_cols[1].metric("Open", paper_summary.open_count)
paper_cols[2].metric("Settled", paper_summary.settled_count)
paper_cols[3].metric("Win Rate", f"{paper_summary.win_rate * 100:.2f}%")
paper_cols[4].metric("Total PnL", f"${paper_summary.total_pnl:,.2f}")

if not paper_trades.empty:
    show_cols = [
        "created_at_utc",
        "side",
        "regime",
        "market_price",
        "spot_price",
        "stake_usd",
        "status",
        "outcome_side",
        "pnl_usd",
        "return_pct",
        "notes",
    ]
    existing = [column for column in show_cols if column in paper_trades.columns]
    st.dataframe(paper_trades[existing].tail(200), use_container_width=True, hide_index=True)
else:
    st.caption(f"No paper trades logged yet. Log file: {paper_path}")

with st.expander("Signal Diagnostics", expanded=False):
    st.write(
        {
            "funding_zscore_used": round(funding_zscore, 4),
            "liquidation_imbalance_used": round(liquidation_imbalance, 4),
            "news_shock_used": round(news_shock, 4),
            "funding_source": funding_source,
            "liquidation_source": liquidation_source,
            "news_source": news_source,
            "spread_bps": round(order_book_features.get("spread_bps", 0.0), 3),
            "depth_imbalance": round(order_book_features.get("depth_imbalance", 0.0), 4),
        }
    )

with st.expander("Model Details", expanded=False):
    feature_df = build_feature_table(
        market_features=market_features,
        normalized_features=model_out.normalized_features,
        contributions=model_out.contributions,
    )
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    st.caption(
        "Funding and liquidation use Binance futures data; "
        "news shock uses CryptoCompare headlines with keyword sentiment and recency weighting. "
        "The regime model fits a 3-state Markov switch on returns standardized by a GARCH volatility filter."
    )
