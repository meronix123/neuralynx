"""Marktinnenleben direkt vor dem Einstieg: Orderbuch, Taker-Fluss, Open Interest, Funding.

Diese Daten gibt es NICHT rueckwirkend - sie werden deshalb nur live geprueft und bei
jedem Trade mitgespeichert, damit man spaeter auswerten kann, ob sie helfen.

Ergebnis je Richtung: ein Wert von -1 (spricht klar dagegen) bis +1 (spricht klar dafuer).
"""
import logging
import time

log = logging.getLogger("bot")


def book_imbalance(book: dict, depth_pct: float = 0.5) -> float | None:
    """(Kaufvolumen - Verkaufsvolumen) / Summe im Bereich +-depth_pct % um den Mittelkurs."""
    bids, asks = book.get("bids") or [], book.get("asks") or []
    if not bids or not asks:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    lo, hi = mid * (1 - depth_pct / 100), mid * (1 + depth_pct / 100)
    b = sum(q for p, q, *_ in bids if p >= lo)
    a = sum(q for p, q, *_ in asks if p <= hi)
    return (b - a) / (a + b) if a + b > 0 else None


def book_walls(book: dict, range_pct: float = 2.0, bucket_pct: float = 0.1, factor: float = 3.0,
               top: int = 3) -> dict:
    """Grosse Orderbuch-Waende: Preisbereiche (je bucket_pct %), in denen deutlich mehr liegt
    (>= factor x Median) als sonst im Bereich +-range_pct % um den Kurs.
    Rueckgabe: {"bids": [...], "asks": [...], "range_pct": abgedeckter Bereich} - Groesse in USDT."""
    bids, asks = book.get("bids") or [], book.get("asks") or []
    if not bids or not asks:
        return {"bids": [], "asks": [], "range_pct": 0.0}
    mid = (bids[0][0] + asks[0][0]) / 2
    step = mid * bucket_pct / 100
    out = {}
    buckets_all = []
    for side, rows in (("bids", bids), ("asks", asks)):
        b: dict[int, list[float]] = {}
        for p, q, *_ in rows:
            if abs(p - mid) / mid * 100 > range_pct:
                continue
            k = int((p - mid) // step)
            x = b.setdefault(k, [0.0, 0.0])
            x[0] += p * q          # USDT
            x[1] += p * p * q      # fuer volumengewichteten Preis
        out[side] = b
        buckets_all += [v[0] for v in b.values()]
    if not buckets_all:
        return {"bids": [], "asks": [], "range_pct": 0.0}
    med = float(sorted(buckets_all)[len(buckets_all) // 2])
    res = {}
    for side in ("bids", "asks"):
        walls = [{"price": v[1] / v[0], "usdt": v[0], "x_median": v[0] / med if med else None}
                 for v in out[side].values() if v[0] >= factor * med]
        res[side] = sorted(walls, key=lambda w: -w["usdt"])[:top]
    lo = min(p for p, *_ in bids)
    hi = max(p for p, *_ in asks)
    res["range_pct"] = round(min(mid - lo, hi - mid) / mid * 100, 3)
    return res


def fetch_walls(client, symbol: str, cfg: dict) -> dict:
    """Orderbuch moeglichst tief laden (Bitget fasst Preisstufen zusammen) und Waende suchen."""
    best = None
    for params in ({"precision": "scale2", "limit": "max"}, {"precision": "scale1", "limit": "max"}, {}):
        try:
            book = client.fetch_order_book(symbol, None if params else 100, params)
        except Exception as e:  # noqa: BLE001 - nicht jede Stufe gibt es fuer jeden Markt
            log.debug("Orderbuch %s %s: %s", symbol, params, e)
            continue
        w = book_walls(book, cfg.get("wall_range_pct", 2.0), cfg.get("wall_bucket_pct", 0.1),
                       cfg.get("wall_factor", 3.0))
        if best is None or w["range_pct"] > best["range_pct"]:
            best = w
        if w["range_pct"] >= cfg.get("wall_range_pct", 2.0) * 0.8:
            break
    return best or {"bids": [], "asks": [], "range_pct": 0.0}


def wall_blocks(side: str, price: float, tp: float, walls: dict | None, cfg: dict) -> str:
    """Optional (flow.wall_filter): grosse Gegen-Wand zwischen Einstieg und Ziel -> kein Trade."""
    if not cfg.get("wall_filter") or not walls:
        return ""
    min_x = cfg.get("wall_block_x", 5.0)
    rows = walls.get("asks" if side == "long" else "bids") or []
    for w in rows:
        between = price < w["price"] < tp if side == "long" else tp < w["price"] < price
        if between and (w.get("x_median") or 0) >= min_x:
            return f"grosse {'Verkaufs' if side == 'long' else 'Kauf'}-Wand bei {w['price']:.6g} vor dem Ziel"
    return ""


def taker_flow(trades: list) -> float | None:
    """Anteil aggressiver Kaeufe minus Verkaeufe der letzten Trades (-1..+1)."""
    buy = sum(t["amount"] for t in trades if t.get("side") == "buy")
    sell = sum(t["amount"] for t in trades if t.get("side") == "sell")
    return (buy - sell) / (buy + sell) if buy + sell > 0 else None


class FlowMonitor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.oi_hist: dict[str, list[tuple[float, float]]] = {}

    def _oi_change(self, client, symbol: str) -> float | None:
        """Veraenderung des Open Interest in % seit ca. einer Stunde (eigene Messreihe)."""
        try:
            oi = client.fetch_open_interest(symbol)
            val = float(oi.get("openInterestAmount") or oi.get("openInterestValue") or 0)
        except Exception as e:  # noqa: BLE001
            log.debug("OI %s: %s", symbol, e)
            return None
        now = time.time()
        h = self.oi_hist.setdefault(symbol, [])
        h.append((now, val))
        while h and now - h[0][0] > 3 * 3600:
            h.pop(0)
        old = [v for t, v in h if now - t >= 3600]
        return (val / old[-1] - 1) * 100 if old and old[-1] else None

    def measure(self, client, symbol: str, funding: float | None) -> dict:
        m = {"imbalance": None, "taker_flow": None, "oi_change_pct": None, "funding": funding}
        try:
            m["imbalance"] = book_imbalance(client.fetch_order_book(symbol, limit=100),
                                            self.cfg.get("book_depth_pct", 0.5))
        except Exception as e:  # noqa: BLE001
            log.debug("Orderbuch %s: %s", symbol, e)
        try:
            m["taker_flow"] = taker_flow(client.fetch_trades(symbol, limit=200))
        except Exception as e:  # noqa: BLE001
            log.debug("Trades %s: %s", symbol, e)
        m["oi_change_pct"] = self._oi_change(client, symbol)
        try:  # Verhaeltnis Long- zu Short-Konten der Trader (nur Info / Auswertung)
            hist = client.fetch_long_short_ratio_history(symbol, "5m", None, 1)
            m["long_short_ratio"] = float(hist[-1]["longShortRatio"]) if hist else None
        except Exception as e:  # noqa: BLE001
            m["long_short_ratio"] = None
            log.debug("Long/Short %s: %s", symbol, e)
        return m


def flow_score(side: str, m: dict) -> float:
    """-1..+1: wie gut passen Orderbuch & Fluss zur geplanten Richtung?"""
    sign = 1 if side == "long" else -1
    parts = []
    if m.get("imbalance") is not None:
        parts.append(sign * m["imbalance"])
    if m.get("taker_flow") is not None:
        parts.append(sign * m["taker_flow"])
    return sum(parts) / len(parts) if parts else 0.0


def flow_verdict(side: str, m: dict, cfg: dict) -> tuple[bool, str, float]:
    """-> (Einstieg erlaubt?, Begruendung, Risiko-Faktor)"""
    score = flow_score(side, m)
    if score <= -cfg.get("flow_block", 0.35):
        return False, f"Orderbuch/Fluss spricht dagegen ({score:+.2f})", 0.0
    factor = 1.0
    if score < 0:
        factor = 0.5  # leicht dagegen -> halbes Risiko
    return True, f"Orderbuch/Fluss {score:+.2f}", factor
