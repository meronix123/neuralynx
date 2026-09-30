"""Derivate-Daten: Funding-Rate-Historie als Mass fuer 'ueberfuellte' Maerkte.

Hohe positive Funding-Rate = sehr viele gehebelte Longs zahlen fuer ihre Position.
Solche Maerkte kippen oft, weil die Longs bei kleinen Ruecksetzern liquidiert werden.
Der Filter handelt deshalb nicht mit der Masse, wenn das Funding im Vergleich zum
letzten Monat extrem ist. Die Daten gibt es rueckwirkend -> im Backtest pruefbar.
"""
import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger("bot")

RANK_WINDOW = 90   # Funding-Zeitpunkte (bei 8h-Funding ca. 30 Tage)


def fetch_funding_history(client, symbol: str, days: int, data_dir=None) -> pd.DataFrame:
    """Funding-Historie (neueste zuerst seitenweise) bis `days` zurueck. Spalten: ts, rate."""
    cache = None
    if data_dir is not None:
        data_dir.mkdir(exist_ok=True)
        cache = data_dir / f"{symbol.replace('/', '_').replace(':', '_')}_funding_{days}d.csv"
        if cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
            cached = pd.read_csv(cache)
            if len(cached) > 1:
                return cached
    since = client.milliseconds() - days * 86_400_000
    rows: dict[int, float] = {}
    for page in range(1, 80):
        batch = client.fetch_funding_rate_history(symbol, None, 100, {"pageNo": page})
        if not batch:
            break
        new = {int(b["timestamp"]): float(b["fundingRate"]) for b in batch if b.get("fundingRate") is not None}
        before = len(rows)
        rows.update(new)
        if len(rows) == before or min(new) <= since:
            break
    df = pd.DataFrame(sorted(rows.items()), columns=["ts", "rate"])
    df = df[df.ts >= since].reset_index(drop=True)
    if cache is not None and len(df):
        df.to_csv(cache, index=False)
    return df


def with_rank(fdf: pd.DataFrame, window: int = RANK_WINDOW) -> pd.DataFrame:
    """Rang der aktuellen Funding-Rate im Vergleich zu den letzten `window` Werten (0..1)."""
    out = fdf[["ts", "rate"]].copy().sort_values("ts").reset_index(drop=True)
    out["rank"] = out["rate"].rolling(window, min_periods=20).rank(pct=True)
    return out


def merge_funding(bars: pd.DataFrame, fdf: pd.DataFrame | None) -> pd.Series:
    """Funding-Rang je Bar - nur Werte, die zum Bar-Schluss schon bekannt waren."""
    if fdf is None or len(fdf) == 0:
        return pd.Series(np.nan, index=bars.index)
    ranked = with_rank(fdf)
    left = pd.DataFrame({"avail": bars["avail"].to_numpy(), "_i": np.arange(len(bars))})
    m = pd.merge_asof(left.sort_values("avail"), ranked[["ts", "rank"]].rename(columns={"ts": "avail"}),
                      on="avail", direction="backward").sort_values("_i")
    return pd.Series(m["rank"].to_numpy(), index=bars.index)


def funding_blocks(side: str, rank: float | None, s: dict) -> bool:
    """Markt in Signalrichtung ueberfuellt? (Rang >= Grenze bei Long, <= 1-Grenze bei Short)"""
    if not s.get("funding_filter") or rank is None or rank != rank:
        return False
    lim = s.get("funding_block_rank", 0.9)
    return rank >= lim if side == "long" else rank <= 1 - lim


def live_rank(history: pd.DataFrame | None, current: float | None) -> float | None:
    """Rang der aktuellen Funding-Rate gegen die letzten Werte (live)."""
    if history is None or current is None or len(history) < 20:
        return None
    last = history["rate"].tail(RANK_WINDOW).to_numpy()
    return float((last <= current).mean())
