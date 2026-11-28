from __future__ import annotations

from datetime import datetime, timezone, timedelta, time as clock_time
from zoneinfo import ZoneInfo
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

    # Keep displayed scores on a true 0–100 scale.
    score_buy = min(100, score_buy)
    score_sell = min(100, score_sell)
    direction = "WAIT"
    score = max(score_buy, score_sell)
    setup = "No confirmed setup"
    if long_setup:
        direction, score, setup = "BUY", score_buy, "Price pulled back during an upward trend" if near_ema_long else "Price moved above recent highs"
    elif short_setup:
        direction, score, setup = "SELL", score_sell, "Price pulled back during a downward trend" if near_ema_short else "Price moved below recent lows"
    elif uptrend:
        setup = "Price is generally rising — waiting for a good entry"
    elif downtrend:
        setup = "Price is generally falling — waiting for a good entry"

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
        "Setup": setup, "Price": price, "Pressure": rsi, "Typical_move_pips": atr/ps,
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

tab_intraday, tab_swing, tab_risk, tab_duration = st.tabs([
    "⚡ Intraday scanner", "📈 Swing scanner", "🧮 Position sizing", "⏳ Trade duration calculator"
])

def show_scan(tab, label, interval):
    with tab:
        st.subheader(label)
        interval_label = "15-minute candles" if interval == "15min" else "daily candles"
        st.caption(f"Chart interval: {interval_label} · Up to 300 candles of history. Signals are watchlist ideas, not orders.")
        with st.expander("Technical terms explained in simple language"):
            st.markdown("""
            - **RSI — buying/selling pressure:** A 0–100 reading showing which side has been stronger recently. It can show when a move is stretched, but it does not predict a reversal.
            - **ATR — typical movement:** An estimate of how far price usually moves in one candle. It measures movement, not direction.
            - **EMA — average price line:** A smoothed line that helps show price direction. The 20 line reacts faster; the 50 and 200 show broader direction.
            - **MACD — momentum check:** Compares faster and slower averages to show whether upward or downward momentum is strengthening.
            - **Pip:** A small standard unit of currency movement. Most pairs use 0.0001; yen pairs usually use 0.01.
            - **Breakout:** Price moves above a recent high. A breakdown is a move below a recent low.
            - **Stop-loss:** The planned exit level intended to limit a loss.
            - **Target:** A planned price level where you may take profit. It is not guaranteed to be reached.
            - **A/B/C/D grade:** A summary of how many scanner rules agree. It is not a win rate or a probability of profit.
            """)
        if not key:
            st.warning("Add your Twelve Data API key in the sidebar to use the live currency scanner. The trading-time calculator works without an API key.")
            st.markdown("""
            1. Create an account with [Twelve Data](https://twelvedata.com/).
            2. Copy your API key.
            3. Paste it into the password field in the sidebar.
            """)
            return
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
            m4.metric("Best-quality setups", a_count)
            st.caption(f"Last scan: {stamp}")
            actionable = [x for x in results if x["Signal"] != "WAIT"]
            if not actionable:
                st.warning("No confirmed entry setup passed all rules. Do not force a trade.")
            elif a_count:
                st.success(f"{a_count} high-quality setup(s) matched the scanner rules. Check trading costs, news and the live price before considering a trade.")
            else:
                st.info("Some possible setups appeared, but none met the highest quality level. Treat them as watchlist candidates.")
            st.subheader("Best-looking opportunities")
            cards = st.columns(3)
            for col, item in zip(cards, results[:3]):
                with col:
                    color = "good" if item["Signal"] == "BUY" else ("bad" if item["Signal"] == "SELL" else "")
                    st.markdown(f"**{item['Pair']}** · <span class='{color}'>{item['Signal']}</span> · **{item['Grade']} · {item['Score']}/100 strength**", unsafe_allow_html=True)
                    st.caption(item["Setup"])
                    if item["Signal"] != "WAIT":
                        st.write(f"Entry: `{fmt_price(item['Pair'], item['Entry'])}`")
                        st.write(f"Stop: `{fmt_price(item['Pair'], item['Stop'])}`")
                        st.write(f"Target 1: `{fmt_price(item['Pair'], item['Target_1'])}`")
                        st.write(f"Target 2: `{fmt_price(item['Pair'], item['Target_2'])}`")
                        st.write(f"Stop distance: `{item['Risk_pips']:.1f} pips`")
                    st.write(f"Buying/selling pressure: {item['Pressure']:.1f}/100")
                    st.write(f"Typical movement per candle: {item['Typical_move_pips']:.1f} pips")
            rows = []
            for x in results:
                rows.append({k: x[k] for k in ["Pair","Signal","Grade","Score","Setup","Price","Pressure","Typical_move_pips","Risk_pips","Time"]})
            frame = pd.DataFrame(rows).rename(columns={
                "Grade": "Quality level", "Score": "Rule match score", "Setup": "Why it appeared",
                "Pressure": "Buying/selling pressure (0–100)", "Typical_move_pips": "Usual movement (pips)",
                "Risk_pips": "Stop distance (pips)", "Time": "Last candle time"
            })
            st.subheader("All results, ranked from strongest match")
            st.dataframe(frame, use_container_width=True, hide_index=True,
                         column_config={"Price": st.column_config.NumberColumn("Current price", format="%.5f"),
                                        "Buying/selling pressure (0–100)": st.column_config.NumberColumn(format="%.1f"),
                                        "Usual movement (pips)": st.column_config.NumberColumn(format="%.1f"),
                                        "Stop distance (pips)": st.column_config.NumberColumn(format="%.1f")})
            chart_pair = st.selectbox("Chart pair", [x["Pair"] for x in results], key=f"chart_{interval}")
            chosen = next(x for x in results if x["Pair"] == chart_pair)
            st.subheader(f"{chart_pair} · price direction")
            chart = chosen["Chart"].rename(columns={
                "close": "Closing price", "EMA20": "Short-term average (20)",
                "EMA50": "Medium-term average (50)", "EMA200": "Long-term average (200)"
            })
            st.line_chart(chart, height=320)
            with st.expander("What do these numbers mean?"):
                st.markdown("""
                - **Buying/selling pressure:** A 0–100 reading of which side has been stronger recently. It is not a prediction. Very high or low readings can mean a move is stretched.
                - **Typical movement:** About how far the pair has usually moved in one candle, measured in pips. It helps estimate how much room a stop-loss may need.
                - **Price averages:** Smoothed price lines that help show the general direction over short, medium and longer periods.
                - **Setup strength:** How many of the scanner's conditions agree. It is **not** the percentage chance of winning.
                """)
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
    st.subheader("How much should I trade? — position size calculator")
    st.write("Enter how far away your stop-loss will be. The calculator estimates a trade size so that a stop-loss would cost about your chosen risk amount. This simplified version is accurate only for pairs where USD is the second currency (for example EUR/USD).")
    pair = st.selectbox("Pair", ["EUR/USD","GBP/USD","AUD/USD","NZD/USD","USD/JPY","USD/CAD","USD/CHF"])
    stop_pips = st.number_input("Distance from entry to stop-loss (pips)", min_value=1.0, max_value=500.0, value=20.0, step=1.0)
    risk_amount = capital * risk_pct / 100
    if pair.endswith("/USD"):
        pip_value_per_standard_lot = 10.0
        lots = risk_amount / (stop_pips * pip_value_per_standard_lot)
        units = lots * 100000
        st.metric("Estimated trade size", f"{lots:.3f} standard lots")
        st.metric("Approximate units", f"{units:,.0f}")
        st.caption(f"With {risk_pct:.2f}% risk, the planned loss at the stop is about ${risk_amount:.2f}. Real costs and currency conversion can change the result.")
    else:
        st.info("The value of a pip for this pair changes with the exchange rate. This basic calculator does not yet calculate that conversion. Check the pip value with your broker before placing a trade.")


# Trade-duration calculator: estimates a rough holding-time range from the target distance
# and the recent/entered average movement per candle. It is a planning aid, not a forecast.
with tab_duration:
    st.subheader("How long might I stay in this trade?")
    st.caption(
        "Estimate a rough time range for price to travel from your entry to your take-profit target. "
        "The estimate uses the target distance and the pair's usual movement per candle."
    )

    scan_choices = []
    for interval_key, interval_name in [("15min", "Intraday · 15-minute candles"), ("1day", "Swing · daily candles")]:
        for scanned in st.session_state.get(f"results_{interval_key}", []):
            if scanned.get("Signal") in ("BUY", "SELL") and np.isfinite(scanned.get("Entry", np.nan)):
                scan_choices.append({
                    "label": f"{scanned['Pair']} · {scanned['Signal']} · {interval_name} · {scanned['Grade']} grade",
                    "record": scanned,
                    "interval_key": interval_key,
                    "interval_name": interval_name,
                })

    use_scan = False
    selected_scan = None
    if scan_choices:
        source = st.radio(
            "Where should the numbers come from?",
            ["Use a setup from my latest scan", "Enter my own trade details"],
            horizontal=True,
            key="duration_source",
        )
        use_scan = source == "Use a setup from my latest scan"
    else:
        st.info("Run an intraday or swing scan first to use its suggested entry, target and typical movement automatically. You can also enter the numbers manually below.")

    if use_scan:
        selected_label = st.selectbox(
            "Choose a scanned setup",
            [x["label"] for x in scan_choices],
            key="duration_scan_choice",
        )
        selected_scan = next(x for x in scan_choices if x["label"] == selected_label)
        scanned = selected_scan["record"]
        pair = scanned["Pair"]
        interval_key = selected_scan["interval_key"]
        timeframe_minutes = 15 if interval_key == "15min" else 1440
        timeframe_label = "15 minutes" if interval_key == "15min" else "1 trading day"
        atr_pips = float(scanned["Typical_move_pips"])
        target_choice = st.radio(
            "Which profit target should I estimate?",
            ["Target 1 (closer target)", "Target 2 (further target)"],
            horizontal=True,
            key="duration_target_choice",
        )
        target_price = float(scanned["Target_1"] if target_choice.startswith("Target 1") else scanned["Target_2"])
        target_distance_pips = abs(target_price - float(scanned["Entry"])) / pip_size(pair)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pair and direction", f"{pair} · {scanned['Signal']}")
        c2.metric("Entry price", fmt_price(pair, float(scanned["Entry"])))
        c3.metric("Target price", fmt_price(pair, target_price))
        st.caption(
            f"Using {timeframe_label} candles. The scanner's recent usual movement is "
            f"{atr_pips:.1f} pips per candle. Target distance: {target_distance_pips:.1f} pips."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            pair = st.selectbox("Currency pair", PAIRS, index=0, key="duration_pair")
            timeframe_options = {
                "5 minutes": 5,
                "15 minutes": 15,
                "30 minutes": 30,
                "1 hour": 60,
                "4 hours": 240,
                "1 trading day": 1440,
            }
            timeframe_label = st.selectbox(
                "Chart candle size",
                list(timeframe_options.keys()),
                index=1,
                key="duration_timeframe",
            )
            timeframe_minutes = timeframe_options[timeframe_label]
        with c2:
            target_distance_pips = st.number_input(
                "Distance from entry to profit target (pips)",
                min_value=0.1,
                max_value=10000.0,
                value=30.0,
                step=1.0,
                key="duration_target_pips",
                help="For example, if your entry is 1.0800 and your target is 1.0830 on EUR/USD, the distance is 30 pips.",
            )
            atr_pips = st.number_input(
                "Usual price movement per candle (pips)",
                min_value=0.1,
                max_value=5000.0,
                value=8.0,
                step=0.5,
                key="duration_typical_move",
                help="Use the scanner's 'usual movement per candle' value where available. It is the average size of movement, not a promise of movement in your trade's direction.",
            )

    st.markdown("#### How the estimate is calculated")
    st.write(
        "A market can move back and forth, so it will not travel toward your target on every candle. "
        "The calculator shows three planning scenarios: slower progress (25% of usual movement), "
        "middle progress (50%), and faster progress (75%). These are assumptions, not probabilities."
    )

    if atr_pips <= 0 or target_distance_pips <= 0 or timeframe_minutes <= 0:
        st.error("Enter a target distance and usual candle movement greater than zero.")
    else:
        slow_bars = target_distance_pips / (atr_pips * 0.25)
        middle_bars = target_distance_pips / (atr_pips * 0.50)
        fast_bars = target_distance_pips / (atr_pips * 0.75)

        def describe_holding_time(bars: float, candle_minutes: int) -> str:
            if candle_minutes == 1440:
                return f"{bars:.1f} trading days ({bars:.1f} daily candles)"
            total_minutes = bars * candle_minutes
            total_hours = total_minutes / 60
            if total_hours < 1:
                return f"about {total_minutes:.0f} minutes ({bars:.1f} candles)"
            if total_hours < 24:
                return f"about {total_hours:.1f} trading hours ({bars:.1f} candles)"
            return f"about {total_hours / 24:.1f} 24-hour periods ({bars:.1f} candles)"

        m1, m2, m3 = st.columns(3)
        m1.metric("Faster-market scenario", describe_holding_time(fast_bars, timeframe_minutes))
        m2.metric("Middle planning estimate", describe_holding_time(middle_bars, timeframe_minutes))
        m3.metric("Slower-market scenario", describe_holding_time(slow_bars, timeframe_minutes))

        st.success(
            f"A practical planning range is roughly {describe_holding_time(fast_bars, timeframe_minutes)} "
            f"to {describe_holding_time(slow_bars, timeframe_minutes)}. "
            f"The middle scenario is {describe_holding_time(middle_bars, timeframe_minutes)}."
        )

        st.markdown("#### Set a time-based review point")
        default_review_bars = 16 if timeframe_minutes <= 15 else (8 if timeframe_minutes <= 60 else (6 if timeframe_minutes <= 240 else 5))
        review_bars = st.number_input(
            f"After how many {timeframe_label} candles should I review the trade if neither target nor stop has been reached?",
            min_value=1,
            max_value=500,
            value=default_review_bars,
            step=1,
            key="duration_review_bars",
        )
        st.info(
            "Review reminder: " + describe_holding_time(float(review_bars), timeframe_minutes) +
            ". At that point, check whether the original reason for entering still holds. This is a review point, "
            "not an instruction to close automatically."
        )

        with st.expander("What do these numbers mean?"):
            st.markdown("""
            - **Target distance:** How far price must move from your planned entry to reach your profit target.
            - **Usual movement per candle:** How much the pair has typically moved during one candle. The scanner uses ATR for this figure.
            - **Candle:** One bar on your chart. A 15-minute candle represents 15 minutes; a daily candle represents one trading day.
            - **Middle estimate:** A planning example that assumes price makes progress toward the target equal to half its usual candle movement on average.
            - **Review point:** A time to check whether the trade still makes sense. It is not a guaranteed exit time.
            """)

    st.warning(
        "Important: this calculator cannot know how long a trade will actually take. Price may reach the stop-loss first, "
        "move sideways, or never reach the target. Use your stop-loss and trading plan—not this time estimate—to control risk. "
        "Weekend closures, news, spreads and changing volatility can make the actual holding time longer or shorter."
    )

st.divider()
st.markdown("""
<div class="note">
<b>Important:</b> The A/B/C/D grade is a setup-strength score, not a win probability. Forex is leveraged and can lose more quickly than expected.
The scanner does not yet account for your broker's exact spread, swaps, commissions, execution slippage, or all scheduled economic releases.
Live-feed access and request limits depend on your Twelve Data account and subscription. Confirm that your account's FX entitlement is real-time before trading.
No trades are placed and no broker account is connected.
</div>
""", unsafe_allow_html=True)
