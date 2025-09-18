# Triangular Arbitrage Bot (Binance)

A lightweight, asyncio-based triangular arbitrage scanner for Binance spot markets. It listens to real-time bookTicker streams for a configurable universe of symbols, evaluates both directions of each triangle, and logs the best opportunity in basis points (bps). By default it runs in paper mode (no orders are sent).

## What it does
- Discovers viable triangles across quotes (USDT, BTC, ETH, BNB by default)
- Subscribes to combined WebSocket streams for all required symbols
- Continuously prices triangles and picks the best opportunity
- Applies taker fees and basic exchange filters in calculations
- Paper trading by default; prints opportunities and estimated PnL

## Project layout
- `tri_arb_bot.py` — main asyncio app: websockets, pricing, and event loop
- `config_symbols.py` — base/quote universes and triangle generation

## Requirements
- Python 3.10+ (3.11 recommended)
- Packages: `aiohttp`, `websockets`, `python-dotenv`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or install directly without a requirements file:

```bash
pip install aiohttp websockets python-dotenv
```

## Configuration
Create a `.env` file with your Binance API credentials (used only if you enable live trading):

```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

Edit trading parameters in `tri_arb_bot.py` if needed:
- `START_QUOTE_USDT` — notional to simulate per opportunity
- `TAKER_FEE` — taker fee applied in the math (default 0.10%)
- `PROFIT_BPS_THRESHOLD` — minimum net bps to log/trade
- `LIVE` — set to `True` to prepare for live trading (orders not implemented here)

Tweak symbol universes in `config_symbols.py`:
- `QUOTES` — quote assets used to form triangles (default: USDT, BTC, ETH, BNB)
- `BASES` — base assets considered for triangles

## Run (paper mode)

```bash
python tri_arb_bot.py
```

You’ll see periodic logs with the best triangle, estimated bps, and simulated PnL. Live order placement across arbitrary quote legs is not implemented in this script (see `maybe_trade`). If you plan to go live, implement quote-aware execution for each leg and add robust risk checks.


## Notes
- For education only. Markets are risky. Backtest, simulate, and add robust safety checks before any live trading.
- Respect exchange rate limits. Consider adding throttling and retries.
- Network or stream hiccups happen; reconnect logic exists but should be hardened for production.
