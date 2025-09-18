# config_symbols.py
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

# Quote assets we’ll use to form triangles
QUOTES: List[str] = ["USDT", "BTC", "ETH", "BNB"]

# A reasonably liquid base universe to start with (extend as you like)
BASES: List[str] = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "LTC", "LINK",
    "MATIC", "DOT", "AVAX", "ATOM", "XMR", "ETC", "AAVE", "NEAR", "APT", "OP",
    "ARB", "FTM", "SUI", "SEI", "TIA"
]

@dataclass(frozen=True)
class Triangle:
    """
    A triangle like (BASE, Q1, Q2) meaning:
      P1: BASE/Q1 -> BASE/Q2 -> Q2/Q1
      P2: BASE/Q1 -> Q1/Q2 -> BASE/Q2
    We’ll compute both legs using best bid/ask per pair.
    """
    base: str
    q1: str
    q2: str

def symbol_name(base: str, quote: str) -> str:
    return f"{base}{quote}"

def build_available_pairs(exchange_info: Dict) -> Set[str]:
    """
    Build the set of available spot symbols like 'ETHUSDT', 'ETHBTC', 'ETHBNB', 'BTCUSDT', etc.
    Only use symbols that are 'TRADING' and 'SPOT'.
    """
    available = set()
    for s in exchange_info["symbols"]:
        if s.get("status") == "TRADING" and s.get("isSpotTradingAllowed", True):
            available.add(s["symbol"])
    return available

def build_triangles(available_pairs: Set[str]) -> Tuple[List[Triangle], Set[str]]:
    """
    Create triangles for every base in BASES across QUOTES where all three legs exist.
    Triangles considered:
      (base, q1, q2) requires:
        base/q1, base/q2, and q2/q1  (or q1/q2 depending on direction — we'll handle both)
    We'll include a triangle only if we can price BOTH directions using existing symbols:
      Need BOTH q2/q1 OR q1/q2 (we will use whichever is available).
    Return:
      - list of Triangle(base, q1, q2)
      - set of all symbol strings we must subscribe to (bookTicker) & load filters for
    """
    triangles: List[Triangle] = []
    needed_symbols: Set[str] = set()

    # Always include cross-quotes among QUOTES themselves (e.g., BTCUSDT, ETHUSDT, ETHBTC, BNBUSDT, BNBETH, BNBBTC, etc.)
    for q_a in QUOTES:
        for q_b in QUOTES:
            if q_a == q_b:
                continue
            s = symbol_name(q_a, q_b)
            s_rev = symbol_name(q_b, q_a)
            if s in available_pairs or s_rev in available_pairs:
                needed_symbols.add(s if s in available_pairs else s_rev)

    for base in BASES:
        for i in range(len(QUOTES)):
            for j in range(i + 1, len(QUOTES)):
                q1, q2 = QUOTES[i], QUOTES[j]
                if base in (q1, q2):
                    # skip degenerate triangles like base==quote
                    continue

                s_bq1 = symbol_name(base, q1)
                s_bq2 = symbol_name(base, q2)
                s_q2q1 = symbol_name(q2, q1)
                s_q1q2 = symbol_name(q1, q2)

                # Need base/q1 & base/q2
                if not (s_bq1 in available_pairs and s_bq2 in available_pairs):
                    continue

                # Need one of q2/q1 or q1/q2 to close the loop
                if not (s_q2q1 in available_pairs or s_q1q2 in available_pairs):
                    continue

                # Valid triangle
                triangles.append(Triangle(base=base, q1=q1, q2=q2))
                # Collect required symbols for streams/filters
                needed_symbols.update([s_bq1, s_bq2])
                # Add whichever exists for the cross leg
                if s_q2q1 in available_pairs:
                    needed_symbols.add(s_q2q1)
                else:
                    needed_symbols.add(s_q1q2)

    # Deduplicate triangles (dataclass + set would work, but list is fine)
    return triangles, needed_symbols
