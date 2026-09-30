"""Automatische Wahl der Zeiteinheit.

Der Bot beobachtet mehrere Einstiegs-Zeiteinheiten gleichzeitig (z. B. 15m ... 4h).
Jede fuehrt ein "Schatten-Konto": Jedes Signal wird auf dem Papier bis Stop, Ziel oder
Zeit-Stop verfolgt - egal ob echt gehandelt wurde. Echt gehandelt wird nur auf den
Zeiteinheiten, deren letzte Schatten-Trades im Plus liegen. So wechselt der Bot selbst
zwischen schnellen und langsamen Trades, je nachdem was der Markt gerade hergibt.
Backtest und Live nutzen dieselben Funktionen.
"""
import copy
from bisect import bisect_right

from .filters import leader_blocks, mtf_blocks
from .strategy import TF_MS

# Einstiegs-Zeiteinheit -> (Trend-Zeiteinheit, Trend-EMA schnell, langsam)
TREND_FOR = {
    "5m": ("1h", 50, 200),
    "15m": ("1h", 50, 200),
    "30m": ("4h", 50, 200),
    "1h": ("4h", 50, 200),
    "2h": ("1d", 20, 50),
    "4h": ("1d", 20, 50),
}
MAX_SHADOW_BARS = 100


def multi(cfg: dict) -> bool:
    return cfg.get("tf_select", "fixed") in ("adaptive", "all") and bool(cfg.get("timeframes"))


def adaptive(cfg: dict) -> bool:
    return cfg.get("tf_select", "fixed") == "adaptive" and bool(cfg.get("timeframes"))


def active_tfs(cfg: dict) -> list[str]:
    """Beobachtete Einstiegs-Zeiteinheiten, groesste zuerst (bei Gleichstand bevorzugt)."""
    if not multi(cfg):
        return [cfg["timeframe"]]
    return sorted(dict.fromkeys(cfg["timeframes"]), key=lambda t: TF_MS[t], reverse=True)


def tf_cfg(cfg: dict, tf: str) -> dict:
    """Einstellungen fuer eine Zeiteinheit (eigene Trend-Zeiteinheit und Trend-EMAs)."""
    c = copy.deepcopy(cfg)
    c["timeframe"] = tf
    if tf in TREND_FOR and (multi(cfg) or tf != cfg["timeframe"]):
        ttf, fast, slow = TREND_FOR[tf]
        c["trend_timeframe"] = ttf
        c["strategy"].update(trend_ema_fast=fast, trend_ema_slow=slow)
    return c


def shadow_results(d: dict, tf: str, s: dict, fee: float, leader: list | None = None) -> list[tuple[int, float]]:
    """Schatten-Trades einer Zeitreihe. d: Spalten als Listen (ts, high, low, close, signal,
    sl_dist, tp_dist, optional mtf_score). Rueckgabe: (Zeitpunkt bekannt, Ergebnis in R) je Trade,
    Gebuehren eingerechnet. Nur Trades, die im Datenbereich abgeschlossen wurden."""
    tf_ms = TF_MS[tf]
    ts, hi, lo, cl, sig = d["ts"], d["high"], d["low"], d["close"], d["signal"]
    sl_d, tp_d, mtf = d["sl_dist"], d["tp_dist"], d.get("mtf_score")
    hold = s.get("max_hold_bars", 0)
    n = len(ts)
    out = []
    for i in range(n):
        if not sig[i]:
            continue
        side = "long" if sig[i] == 1 else "short"
        if mtf is not None and mtf_blocks(side, mtf[i], s):
            continue
        if leader is not None and s.get("leader_filter") and leader_blocks(side, leader[i]):
            continue
        e, r = cl[i], sl_d[i]
        if not r or r != r or r <= 0:
            continue
        sign = 1 if side == "long" else -1
        sl, tp = e - sign * r, e + sign * tp_d[i]
        fee_r = 2 * fee * e / r
        for j in range(i + 1, min(n, i + 1 + MAX_SHADOW_BARS)):
            res = None
            if (lo[j] <= sl) if sign == 1 else (hi[j] >= sl):
                res = -1.0
            elif (hi[j] >= tp) if sign == 1 else (lo[j] <= tp):
                res = tp_d[i] / r
            elif hold and j - i >= hold and sign * (cl[j] - e) / r < 0.5:
                res = sign * (cl[j] - e) / r
            if res is not None:
                out.append((int(ts[j]) + tf_ms, res - fee_r))
                break
    return out


def profit_factor(rs: list[float]) -> float | None:
    if not rs:
        return None
    loss = -sum(x for x in rs if x <= 0)
    win = sum(x for x in rs if x > 0)
    return win / loss if loss > 0 else (99.0 if win > 0 else 0.0)


def recent(rs: list[float], s: dict) -> list[float]:
    return rs[-s.get("tf_window", 20):]


def tf_allowed(rs: list[float], s: dict) -> bool:
    """Genug Schatten-Trades und die letzten davon im Plus? (Anlaufphase: alles erlaubt)"""
    if len(rs) < s.get("tf_warmup", 10):
        return True
    return profit_factor(recent(rs, s)) >= s.get("tf_min_pf", 1.0)


class ShadowBook:
    """Schatten-Ergebnisse je Zeiteinheit (alle Maerkte zusammen), nach Bekanntwerden sortiert."""

    def __init__(self, per_tf: dict[str, list[tuple[int, float]]]):
        self.times, self.rs = {}, {}
        for tf, rows in per_tf.items():
            rows = sorted(rows)
            self.times[tf] = [t for t, _ in rows]
            self.rs[tf] = [r for _, r in rows]

    def known(self, tf: str, now_ms: int) -> list[float]:
        k = bisect_right(self.times.get(tf, []), now_ms)
        return self.rs.get(tf, [])[:k]

    def stats(self, tf: str, now_ms: int, s: dict) -> dict:
        rs = self.known(tf, now_ms)
        pf = profit_factor(recent(rs, s))
        return {"n": len(rs), "pf": None if pf is None else round(pf, 2), "ok": tf_allowed(rs, s)}
