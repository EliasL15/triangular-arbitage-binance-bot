# tri_arb_bot.py
import os
import hmac
import time
import json
import asyncio
import hashlib
import logging
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Tuple, List

import aiohttp
import websockets
from dotenv import load_dotenv

from config_symbols import build_available_pairs, build_triangles, symbol_name, Triangle

# -------------------
# Global Config
# -------------------
REST_BASE = "https://api.binance.com"

# Trading params
START_QUOTE_USDT = Decimal("100")
TAKER_FEE = Decimal("0.0010")          # 0.10% taker
PROFIT_BPS_THRESHOLD = Decimal("1")    # require > 5 bps net
MAX_STALENESS_SEC = 2.0
COOLDOWN_SEC = 0.2
LIVE = False

# Risk
ENABLE_KILL_SWITCH = True
DAILY_MAX_LOSS_USDT = Decimal("200")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")

load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# -------------------
# Helpers
# -------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def hmac_sig(query_string: str, secret: str) -> str:
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def d(v) -> Decimal:
    return Decimal(str(v))

def floor_step(value: Decimal, step: Decimal) -> Decimal:
    quant = (value / step).to_integral_exact(rounding=ROUND_DOWN)
    return (quant * step).quantize(step)

# -------------------
# Exchange Info (filters)
# -------------------
class Filters:
    def __init__(self):
        self.data: Dict[str, Dict] = {}

    async def load_for(self, session: aiohttp.ClientSession, symbols: List[str]):
        # Binance allows ?symbols=["SYMA","SYMB",...] in one call
        params = {"symbols": json.dumps(symbols, separators=(',', ':'))}
        async with session.get(f"{REST_BASE}/api/v3/exchangeInfo", params=params, timeout=15) as r:
            r.raise_for_status()
            info = await r.json()

        for s in info["symbols"]:
            lot_step = tick_size = min_notional = None
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    lot_step = d(f["stepSize"])
                elif f["filterType"] == "PRICE_FILTER":
                    tick_size = d(f["tickSize"])
                elif f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
                    mn = f.get("minNotional") or f.get("notional")
                    min_notional = d(mn)

            self.data[s["symbol"]] = {
                "base": s["baseAsset"],
                "quote": s["quoteAsset"],
                "lot_step": lot_step or Decimal("0.00000001"),
                "tick_size": tick_size or Decimal("0.00000001"),
                "min_notional": min_notional or Decimal("10"),
            }

    def ensure_qty(self, symbol: str, qty: Decimal) -> Decimal:
        step = self.data[symbol]["lot_step"]
        return floor_step(qty, step)

    def meets_min_notional(self, symbol: str, qty: Decimal, price: Decimal) -> bool:
        return qty * price >= self.data[symbol]["min_notional"]

# -------------------
# REST Trading Client
# -------------------
class BinanceREST:
    def __init__(self, session: aiohttp.ClientSession, api_key: str, api_secret: str):
        self.session = session
        self.api_key = api_key
        self.api_secret = api_secret

    async def _signed_post(self, path: str, params: dict):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API keys missing. Set BINANCE_API_KEY/SECRET or use LIVE=False.")
        params["timestamp"] = now_ms()
        qs = "&".join([f"{k}={params[k]}" for k in sorted(params)])
        signature = hmac_sig(qs, self.api_secret)
        url = f"{REST_BASE}{path}?{qs}&signature={signature}"
        headers = {"X-MBX-APIKEY": self.api_key}
        async with self.session.post(url, headers=headers, timeout=10) as r:
            text = await r.text()
            if r.status >= 400:
                logging.error("Order error %s: %s", r.status, text)
                r.raise_for_status()
            return json.loads(text)

    async def market_buy_quote(self, symbol: str, quote_qty: Decimal):
        params = {"symbol": symbol, "side": "BUY", "type": "MARKET", "quoteOrderQty": str(quote_qty.normalize())}
        return await self._signed_post("/api/v3/order", params)

    async def market_sell_base(self, symbol: str, quantity: Decimal):
        params = {"symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": str(quantity.normalize())}
        return await self._signed_post("/api/v3/order", params)

# -------------------
# Multi-Triangle Tri-Arb Bot
# -------------------
class TriArbMulti:
    def __init__(self):
        self.filters = Filters()
        self.rest: BinanceREST | None = None
        self.books: Dict[str, Dict] = {}
        self.triangles: List[Triangle] = []
        self.realized_pnl = Decimal("0")

    def update_book(self, symbol: str, bid: Decimal, ask: Decimal):
        slot = self.books.setdefault(symbol, {"bid": None, "ask": None, "time": 0.0})
        slot["bid"] = bid
        slot["ask"] = ask
        slot["time"] = time.time()

    def book_ready(self, required_symbols: List[str]) -> bool:
        return all(self.books.get(s, {}).get("bid") is not None and self.books.get(s, {}).get("ask") is not None
                   for s in required_symbols)

    def _get_cross_price(self, base: str, quote: str) -> Tuple[Decimal, Decimal] | None:
        """
        Return (bid, ask) for BASE/QUOTE using direct symbol if exists,
        else inverted from QUOTE/BASE if that exists.
        """
        direct = symbol_name(base, quote)
        inverse = symbol_name(quote, base)

        if direct in self.books and self.books[direct]["bid"] is not None:
            return self.books[direct]["bid"], self.books[direct]["ask"]

        if inverse in self.books and self.books[inverse]["bid"] is not None:
            # invert: (BASE/QUOTE) = 1 / (QUOTE/BASE)
            bid_inv = self.books[inverse]["bid"]
            ask_inv = self.books[inverse]["ask"]
            if bid_inv and ask_inv and bid_inv > 0 and ask_inv > 0:
                # For inversion: bid_{B/Q} = 1/ask_{Q/B}, ask_{B/Q} = 1/bid_{Q/B}
                return (Decimal(1) / ask_inv, Decimal(1) / bid_inv)

        return None

    def evaluate_triangle(self, tri: Triangle, usdt_start: Decimal) -> Dict:
        """
        Compute both directions' end-USD(T) result if Q1 or Q2 is USDT.
        If neither quote is USDT, we'll convert end quote back to USDT via a cross.
        """
        fee = TAKER_FEE

        # Helper to fetch bid/ask for any base/quote (inverting if needed)
        def px(b: str, q: str) -> Tuple[Decimal, Decimal]:
            p = self._get_cross_price(b, q)
            if p is None:
                raise KeyError(f"Price missing for {b}/{q}")
            return p

        # Convert any currency X to USDT via best available route (direct or via inversion)
        def to_usdt(amount: Decimal, asset: str) -> Decimal:
            if asset == "USDT":
                return amount
            b = self._get_cross_price(asset, "USDT")
            if b is None:
                # try via BTC as a fallback bridge
                b1 = self._get_cross_price(asset, "BTC")
                b2 = self._get_cross_price("BTC", "USDT")
                if b1 is None or b2 is None:
                    raise KeyError(f"No route to USDT for {asset}")
                # amount * bid(asset/BTC) * bid(BTC/USDT) (rough)
                return amount * b1[0] * b2[0]
            return amount * b[0]  # sell asset for USDT at bid

        # Path A: q1 as the “starting quote”
        # Start with USDT; if q1 != USDT, we'll first convert USDT->q1 at ask (buy q1 using USDT)
        # Then USDT/q1 -> base (buy base at ask base/q1), then base -> q2, then q2 -> USDT.
        def simulate(path_q1_first: bool) -> Tuple[Decimal, str]:
            q_start, q_mid = (tri.q1, tri.q2) if path_q1_first else (tri.q2, tri.q1)
            path_name = f"{tri.base}:{q_start}->{q_mid}"

            # USDT -> q_start (if needed)
            qty_qstart = usdt_start
            if q_start != "USDT":
                # buy q_start with USDT at ask (USDT is quote, so we need q_start/USDT ask)
                _, ask_qstart_usdt = px(q_start, "USDT")
                qty_qstart = (usdt_start / ask_qstart_usdt) * (1 - fee)

            # q_start -> BASE (buy BASE with q_start at ask)
            _, ask_base_qstart = px(tri.base, q_start)
            qty_base = (qty_qstart / ask_base_qstart) * (1 - fee)

            # BASE -> q_mid
            bid_base_qmid, ask_base_qmid = px(tri.base, q_mid)
            # if we are SELLING base to receive q_mid, we use bid
            qty_qmid = (qty_base * bid_base_qmid) * (1 - fee)

            # q_mid -> USDT (if needed)
            usdt_end = to_usdt(qty_qmid, q_mid)

            return usdt_end, path_name

        try:
            usdt_end_a, name_a = simulate(True)
            usdt_end_b, name_b = simulate(False)
        except KeyError:
            return {"ok": False}

        def bps(x: Decimal) -> Decimal:
            return (x / usdt_start - 1) * Decimal("10000")

        return {
            "ok": True,
            "A": {"name": name_a, "usdt_end": usdt_end_a, "bps": bps(usdt_end_a)},
            "B": {"name": name_b, "usdt_end": usdt_end_b, "bps": bps(usdt_end_b)},
        }

    async def maybe_trade(self, label: str, usdt_start: Decimal):
        if not LIVE:
            logging.info("[PAPER %s] start=%s USDT", label, usdt_start)
            return True
        if self.rest is None:
            logging.error("LIVE=True but REST client is None.")
            return False
        # NOTE: Live execution across arbitrary quotes is non-trivial;
        # for production you should add quote-aware execution legs like in your single-triangle bot.
        logging.warning("LIVE execution for multi-triangle not implemented here.")
        return False

    async def run(self):
        async with aiohttp.ClientSession() as session:
            # 1) Discover all spot pairs & triangles
            async with session.get(f"{REST_BASE}/api/v3/exchangeInfo", timeout=15) as r:
                r.raise_for_status()
                exi_all = await r.json()

            available_pairs = build_available_pairs(exi_all)
            triangles, needed_symbols = build_triangles(available_pairs)
            self.triangles = triangles

            if not triangles:
                logging.error("No valid triangles found. Adjust BASES/QUOTES.")
                return

            logging.info("Triangles: %d | Needed symbols: %d", len(triangles), len(needed_symbols))

            # 2) Load filters for all needed symbols
            await self.filters.load_for(session, sorted(list(needed_symbols)))

            # 3) Prepare combined WebSocket stream URL
            streams = "/".join([f"{s.lower()}@bookTicker" for s in sorted(needed_symbols)])
            stream_url = f"wss://stream.binance.com:9443/stream?streams={streams}"

            # 4) Optional REST client (LIVE only)
            self.rest = BinanceREST(session, API_KEY, API_SECRET) if LIVE else None

            while True:
                try:
                    async with websockets.connect(stream_url, ping_interval=15, ping_timeout=15, max_size=None) as ws:
                        logging.info("Connected to combined stream (%d symbols).", len(needed_symbols))
                        async for raw in ws:
                            msg = json.loads(raw)
                            data = msg.get("data", {})
                            sym = data.get("s")
                            b = data.get("b")
                            a = data.get("a")
                            if sym and b and a:
                                self.update_book(sym, d(b), d(a))

                            # Staleness guard
                            tnow = time.time()
                            # Ensure we at least have recent data for cross quotes (this is a soft guard)
                            # For speed, we don't block on "all" symbols; triangle eval will fail gracefully if missing.
                            # Evaluate every triangle and pick the best opportunity.
                            best = None
                            for tri in self.triangles:
                                res = self.evaluate_triangle(tri, START_QUOTE_USDT)
                                if not res.get("ok"):
                                    continue
                                for leg in ("A", "B"):
                                    stats = res[leg]
                                    if best is None or stats["bps"] > best["bps"]:
                                        best = {"tri": tri, "leg": leg, **stats}

                            if not best:
                                continue
                                
                            now = time.time()
                            if not hasattr(self, "last_log"):
                                self.last_log = 0
                            if now - self.last_log > 1:
                                logging.info("Best triangle: %s leg=%s | %.2f bps | est=%.2f USDT",
                                            best["tri"].base, best["leg"], best["bps"], best["usdt_end"])
                                self.last_log = now
                                                            

                            # Log / trade if above threshold
                            if best["bps"] >= PROFIT_BPS_THRESHOLD:
                                logging.info("OPP %s | %s: %.2f bps | %s → %s USDT",
                                             best["leg"], best["name"], best["bps"],
                                             START_QUOTE_USDT, best["usdt_end"].quantize(Decimal('0.01')))
                                ok = await self.maybe_trade(best["name"], START_QUOTE_USDT)
                                if ok:
                                    pnl = best["usdt_end"] - START_QUOTE_USDT
                                    self.realized_pnl += pnl
                                    logging.info("PnL est: %+0.2f | Cum: %+0.2f USDT", pnl, self.realized_pnl)
                                    if ENABLE_KILL_SWITCH and self.realized_pnl <= -DAILY_MAX_LOSS_USDT:
                                        logging.error("Kill switch hit (<= -%s). Stopping.", DAILY_MAX_LOSS_USDT)
                                        return
                                    await asyncio.sleep(COOLDOWN_SEC)

                except Exception as e:
                    logging.exception("WebSocket loop error: %s. Reconnecting in 2s...", e)
                    await asyncio.sleep(2)

# -------------------
# Entrypoint
# -------------------
if __name__ == "__main__":
    bot = TriArbMulti()
    asyncio.run(bot.run())
