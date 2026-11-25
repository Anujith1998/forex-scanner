# Forex Scanner Pro

A Streamlit Forex scanner with separate intraday (15-minute) and swing (daily) scans.

## Included
- Major and cross currency pairs (6 / 12 / 28 pair scan universes)
- EMA 20/50/200, RSI 14, MACD, ATR and 20-bar breakout/pullback checks
- BUY / SELL / WAIT classification and A-D technical grades
- Entry, ATR-based stop, 2R and 3R target guides
- $500 USD default account, adjustable risk percentage
- Approximate position sizing for USD-quoted pairs
- CSV export and price/EMA chart
- No broker integration and no automatic order placement

## Market data setup
This version uses the Twelve Data time-series API. A provider API key is required.
1. Create an account at https://twelvedata.com/
2. Review the plan's Forex real-time entitlement and API request limits.
3. Paste your API key in the app sidebar.

**Important:** Do not assume every free account receives unrestricted real-time FX data. The data freshness and allowed requests depend on your provider plan. The app spaces requests to reduce the chance of hitting a common free-tier rate limit. A 28-pair scan can take several minutes.

For Streamlit Community Cloud, you can optionally store the key in App settings → Secrets:
```toml
TWELVE_DATA_API_KEY = "paste-your-key-here"
```
Do not commit your API key to GitHub.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a new GitHub repository, e.g. `forex-scanner`.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. In Streamlit Community Cloud, deploy the repository with `app.py` as the main file.
4. Add your Twelve Data key through Streamlit Secrets or enter it in the sidebar.

## Important limitations
- Technical grades are not win probabilities or trade recommendations.
- Signals use the latest completed candle returned by the provider; a provider's latest candle may still be forming depending on API behavior.
- No broker spread, swaps, commission, slippage, or economic calendar filter is built into this first version.
- Position sizing is a simplified estimate only for USD-quoted pairs. Use your broker's own pip-value calculation for other pairs.
- No orders are sent to any broker.
