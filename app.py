from __future__ import annotations

from datetime import datetime, timezone
import math
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Forex Scanner Pro",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD",
    "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/CHF",
    "GBP/CHF", "CAD/JPY", "CHF/JPY", "AUD/CAD", "EUR/CAD", "GBP/CAD",
    "NZD/JPY", "AUD/NZD", "EUR/AUD", "GBP/AUD", "NZD/CAD", "CAD/CHF",
    "AUD/CHF", "NZD/CHF", "EUR/NZD", "GBP/NZD",
]

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 12% 0%, #142b49 0, #08111f 45%);}
[data-testid="stHeader"] {background: rgba(8,17,31,.78);}
.block-container {padding-top:1.3rem; max-width:1550px;}
.hero {padding:1.25rem 1.4rem;border:1px solid #294866;border-radius:18px;
background:linear-gradient(115deg,#132944,#101c30);margin-bottom:1rem;}
.hero h1 {margin:0;color:#f3f7ff;font-size:2.1rem;}
.hero p {margin:.35rem 0 0;color:#b4c7df;}
.note {padding:.8rem 1rem;border-radius:12px;background:#1b2b42;
border:1px solid #355374;color:#dce9f8;margin-bottom:.8rem;}
.metric {background:#111f33;border:1px solid #2b4564;border-radius:14px;padding:1rem;}
.good {color:#70edb0;font-weight:800;}
.bad {color:#ff8994;font-weight:800;}
.smallmuted {color:#a9bbd2;font-size:.86rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💱 FOREX SCANNER PRO</h1>
  <p>Intraday + swing setups · Live-feed API connection · Technical ranking · $500 USD risk controls</p>
</div>
""", unsafe_allow_html=True)

def api_key_value():
    typed = st.session_state.get("td_api_key", "").strip()
    try:
        saved = st.secrets.get("TWELVE_DATA_API_KEY", "")
    except Exception:
        saved = ""
    return typed or saved

def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001

def fmt_price(pair: str, value: float) -> str:
    return f"{value:.3f}" if pair.endswith("JPY") else f"{value:.5f}"

@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(pair: str, interval: str, api_key: str, outputsize: int = 300):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": "UTC",
        "format": "JSON",
        "apikey": api_key,
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        message = payload.get("message") or payload.get("status") or str(payload)
        raise RuntimeError(message)
    df = pd.DataFrame(payload["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").set_index("datetime")
    return df

def indicators(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["EMA20"] = d.close.ewm(span=20, adjust=False).mean()
    d["EMA50"] = d.close.ewm(span=50, adjust=False).mean()
    d["EMA200"] = d.close.ewm(span=200, adjust=False).mean()
    delta = d.close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)
    d.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    ema12 = d.close.ewm(span=12, adjust=False).mean()
    ema26 = d.close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d.MACD.ewm(span=9, adjust=False).mean()
    prev = d.close.shift(1)
    tr = pd.concat([
        d.high - d.low,
        (d.high - prev).abs(),
        (d.low - prev).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d["PRIOR20_HIGH"] = d.high.shift(1).rolling(20).max()
    d["PRIOR20_LOW"] = d.low.shift(1).rolling(20).min()
    return d.dropna()

def analyze(pair: str, raw: pd.DataFrame, timeframe: str):
    if len(raw) < 210:
        return None
    d = indicators(raw)
    if len(d) < 3:
        return None
    row, prev = d.iloc[-1], d.iloc[-2]
    price = float(row.close)
    atr = float(row.ATR)
    rsi = float(row.RSI)
    uptrend = price > row.EMA50 > row.EMA200 and row.EMA20 > row.EMA50
    downtrend = price < row.EMA50 < row.EMA200 and row.EMA20 < row.EMA50
    bullish_momentum = row.MACD > row.MACD_SIGNAL
    bearish_momentum = row.MACD < row.MACD_SIGNAL

    near_ema_long = price >= row.EMA20 * 0.997 and price <= row.EMA20 * 1.006
    near_ema_short = price <= row.EMA20 * 1.003 and price >= row.EMA20 * 0.994
    breakout_long = price > row.PRIOR20_HIGH
    breakout_short = price < row.PRIOR20_LOW

    long_setup = uptrend and 48 <= rsi <= 68 and bullish_momentum and (near_ema_long or breakout_long)
    short_setup = downtrend and 32 <= rsi <= 52 and bearish_momentum and (near_ema_short or breakout_short)

    score_buy = 0
    score_sell = 0
    if uptrend: score_buy += 30
    if downtrend: score_sell += 30
    if 50 <= rsi <= 65: score_buy += 20
    elif 45 <= rsi <= 70: score_buy += 10
    if 35 <= rsi <= 50: score_sell += 20
    elif 30 <= rsi <= 55: score_sell += 10
    if bullish_momentum: score_buy += 15
    if bearish_momentum: score_sell += 15
    if near_ema_long: score_buy += 15
    if near_ema_short: score_sell += 15
    if breakout_long: score_buy += 20
    if breakout_short: score_sell += 20
    if atr / price < 0.012: 
        score_buy += 5
        score_sell += 5

    direction = "WAIT"
    score = max(score_buy, score_sell)
    setup = "No confirmed setup"
    if long_setup:
        direction, score, setup = "BUY", score_buy, "Trend pullback" if near_ema_long else "20-bar breakout"
    elif short_setup:
        direction, score, setup = "SELL", score_sell, "Trend pullback" if near_ema_short else "20-bar breakdown"
    elif uptrend:
        setup = "Uptrend — waiting for trigger"
    elif downtrend:
        setup = "Downtrend — waiting for trigger"

    grade = "A" if direction != "WAIT" and score >= 80 else (
        "B" if score >= 60 else ("C" if score >= 40 else "D")
    )
    stop_dist = 1.5 * atr
    if direction == "BUY":
        entry, stop = price, price - stop_dist
        target1, target2 = price + 2*stop_dist, price + 3*stop_dist
    elif direction == "SELL":
        entry, stop = price, price + stop_dist
        target1, target2 = price - 2*stop_dist, price - 3*stop_dist
    else:
        entry = stop = target1 = target2 = np.nan

    ps = pip_size(pair)
    return {
        "Pair": pair, "Signal": direction, "Grade": grade, "Score": int(score),
        "Setup": setup, "Price": price, "RSI": rsi, "ATR_pips": atr/ps,
        "Entry": entry, "Stop": stop, "Target_1": target1, "Target_2": target2,
        "Risk_pips": stop_dist/ps, "Time": d.index[-1],
        "Chart": d[["close", "EMA20", "EMA50", "EMA200"]].tail(120),
    }

def scan_pairs(pairs, interval, api_key, timeframe):
    output, errors = [], []
    progress = st.progress(0, text="Connecting to market data…")
    status = st.empty()
    for i, pair in enumerate(pairs, 1):
        status.write(f"Fetching {pair} ({i}/{len(pairs)}). The provider's free tier may enforce request limits.")
        try:
            candles = fetch_candles(pair, interval, api_key)
            item = analyze(pair, candles, timeframe)
            if item:
                output.append(item)
            else:
                errors.append(f"{pair}: not enough candle history returned")
        except Exception as exc:
            errors.append(f"{pair}: {exc}")
        progress.progress(i / len(pairs), text=f"Analysed {i}/{len(pairs)} pairs")
        if i < len(pairs):
            time.sleep(8.1)  # Stay under a typical 8-requests/minute free-tier limit.
    progress.empty()
    status.empty()
    output.sort(key=lambda x: (x["Signal"] != "WAIT", x["Score"]), reverse=True)
    return output, errors

with st.sidebar:
    st.header("Scanner controls")
    st.text_input("Twelve Data API key", type="password", key="td_api_key",
                  help="Paste your API key. It is held only in this browser session unless you add it to Streamlit secrets.")
    key = api_key_value()
    st.caption("Data provider: Twelve Data")
    st.divider()
    capital = st.number_input("Account capital (USD)", min_value=50.0, max_value=1000000.0,
                              value=500.0, step=50.0)
    risk_pct = st.slider("Risk per trade (%)", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    risk_cash = capital * risk_pct / 100
    st.metric("Planned risk per trade", f"${risk_cash:.2f}")
    pair_count = st.selectbox("Scan universe", [6, 12, 28], index=0,
                              format_func=lambda x: f"{x} major pairs" if x == 6 else ("12 liquid pairs" if x == 12 else "28 major + cross pairs"))
    selected_pairs = PAIRS[:pair_count]
    st.caption("Each pair uses one API request per scan. Requests are spaced to reduce rate-limit errors.")
    st.divider()
    st.caption("No broker connection or automatic order placement.")

if not key:
    st.warning("Add your Twelve Data API key in the sidebar to enable market-data scans.")
    st.markdown("""
    **How to get connected**
    1. Create an account with [Twelve Data](https://twelvedata.com/).
    2. Copy your API key from its dashboard.
    3. Paste the key into the password field in the sidebar.
    4. Start with the 6-pair scan to check your plan's request allowance.
    """)
    st.stop()

tab_intraday, tab_swing, tab_risk = st.tabs(["⚡ Intraday scanner", "📈 Swing scanner", "🧮 Position sizing"])

def show_scan(tab, label, interval):
    with tab:
        st.subheader(label)
        st.caption(f"Timeframe: {interval} candles · 300-bar history · Signals are candidates, not orders.")
        c1, c2 = st.columns([1, 3])
        with c1:
            go = st.button(f"Scan {label}", type="primary", key=f"scan_{interval}", use_container_width=True)
        with c2:
            st.caption("Manual scans only. Refresh when you want a new snapshot; this avoids burning API credits in the background.")
        state_key = f"results_{interval}"
        errors_key = f"errors_{interval}"
        if go:
            fetch_candles.clear()
            with st.spinner("Scanning selected currency pairs…"):
                results, errors = scan_pairs(selected_pairs, interval, key, label)
            st.session_state[state_key] = results
            st.session_state[errors_key] = errors
            st.session_state[f"time_{interval}"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        results = st.session_state.get(state_key, [])
        errors = st.session_state.get(errors_key, [])
        if results:
            stamp = st.session_state.get(f"time_{interval}", "")
            a_count = sum(x["Grade"] == "A" and x["Signal"] != "WAIT" for x in results)
            buys = sum(x["Signal"] == "BUY" for x in results)
            sells = sum(x["Signal"] == "SELL" for x in results)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pairs analysed", len(results))
            m2.metric("BUY setups", buys)
            m3.metric("SELL setups", sells)
            m4.metric("A-grade setups", a_count)
            st.caption(f"Last scan: {stamp}")
            actionable = [x for x in results if x["Signal"] != "WAIT"]
            if not actionable:
                st.warning("No confirmed entry setup passed all rules. Do not force a trade.")
            elif a_count:
                st.success(f"{a_count} A-grade setup(s) passed the scanner rules. Verify spread, news and price before considering a trade.")
            else:
                st.info("Setups found, but none reached A grade. Treat them as watchlist candidates.")
            st.subheader("Top opportunities")
            cards = st.columns(3)
            for col, item in zip(cards, results[:3]):
                with col:
                    color = "good" if item["Signal"] == "BUY" else ("bad" if item["Signal"] == "SELL" else "")
                    st.markdown(f"**{item['Pair']}** · <span class='{color}'>{item['Signal']}</span> · **{item['Grade']} ({item['Score']}/100)**", unsafe_allow_html=True)
                    st.caption(item["Setup"])
                    if item["Signal"] != "WAIT":
                        st.write(f"Entry: `{fmt_price(item['Pair'], item['Entry'])}`")
                        st.write(f"Stop: `{fmt_price(item['Pair'], item['Stop'])}`")
                        st.write(f"Target 1: `{fmt_price(item['Pair'], item['Target_1'])}`")
                        st.write(f"Target 2: `{fmt_price(item['Pair'], item['Target_2'])}`")
                        st.write(f"Stop distance: `{item['Risk_pips']:.1f} pips`")
                    st.write(f"RSI: {item['RSI']:.1f} · ATR: {item['ATR_pips']:.1f} pips")
            rows = []
            for x in results:
                rows.append({k: x[k] for k in ["Pair","Signal","Grade","Score","Setup","Price","RSI","ATR_pips","Risk_pips","Time"]})
            frame = pd.DataFrame(rows)
            st.subheader("Full ranked scan")
            st.dataframe(frame, use_container_width=True, hide_index=True,
                         column_config={"Price": st.column_config.NumberColumn(format="%.5f"),
                                        "RSI": st.column_config.NumberColumn(format="%.1f"),
                                        "ATR_pips": st.column_config.NumberColumn(format="%.1f"),
                                        "Risk_pips": st.column_config.NumberColumn(format="%.1f")})
            chart_pair = st.selectbox("Chart pair", [x["Pair"] for x in results], key=f"chart_{interval}")
            chosen = next(x for x in results if x["Pair"] == chart_pair)
            st.subheader(f"{chart_pair} · price and EMAs")
            st.line_chart(chosen["Chart"], height=320)
            csv = frame.to_csv(index=False).encode("utf-8")
            st.download_button("Download scan CSV", csv, file_name=f"forex_{interval}_scan.csv",
                               mime="text/csv", key=f"csv_{interval}")
        elif errors:
            st.warning("No scan results were produced. Check the API key, provider plan, and pair availability.")
        if errors:
            with st.expander(f"Provider messages ({len(errors)})"):
                for e in errors:
                    st.write("• " + e)

show_scan(tab_intraday, "Intraday setups", "15min")
show_scan(tab_swing, "Swing setups", "1day")

with tab_risk:
    st.subheader("Position sizing — $500 USD account")
    st.write("Enter the planned stop distance. This calculator gives a rough sizing estimate for USD-quoted pairs (EUR/USD, GBP/USD, AUD/USD, NZD/USD).")
    pair = st.selectbox("Pair", ["EUR/USD","GBP/USD","AUD/USD","NZD/USD","USD/JPY","USD/CAD","USD/CHF"])
    stop_pips = st.number_input("Stop distance (pips)", min_value=1.0, max_value=500.0, value=20.0, step=1.0)
    risk_amount = capital * risk_pct / 100
    if pair.endswith("/USD"):
        pip_value_per_standard_lot = 10.0
        lots = risk_amount / (stop_pips * pip_value_per_standard_lot)
        units = lots * 100000
        st.metric("Estimated position size", f"{lots:.3f} standard lots")
        st.metric("Approximate units", f"{units:,.0f}")
        st.caption(f"At {risk_pct:.2f}% risk, planned loss at the stop is about ${risk_amount:.2f}, before spread, commission and slippage.")
    else:
        st.info("For this pair, pip value changes with the exchange rate. Use your broker's pip-value calculator rather than this simplified USD-quote calculation.")

st.divider()
st.markdown("""
<div class="note">
<b>Important:</b> A-grade is a technical score, not a win probability. Forex is leveraged and can lose more quickly than expected.
The scanner does not account for your broker's exact bid/ask spread, swaps, commissions, execution slippage, or all scheduled economic releases.
Live-feed access and request limits depend on your Twelve Data account and subscription. Confirm that your account's FX entitlement is real-time before trading.
No trades are placed and no broker account is connected.
</div>
""", unsafe_allow_html=True)
