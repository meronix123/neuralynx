"""Einstiegslogik: erst Marktlage erkennen, dann passende Strategie.

Marktlage (je Bar):
  Chaos        extreme Volatilitaet (ATR ueber Grenze / in den obersten 3 %) -> nichts tun
  Ruhephase    Bollinger-Baender so eng wie selten (unterste 20 %)  -> Ausbruch vorbereiten
  Trend        ADX >= 22 und Trend der hoeheren Zeiteinheit klar  -> Trend-Ruecksetzer
  Seitwaerts   ADX < 18                                          -> Rueckkehr zur Mitte

Strategien:
  Trend-Ruecksetzer  Ausloeser (RSI/MACD/EMA21) + mind. `min_score` von 6 Punkten,
                     SL sl_atr x ATR, TP tp_atr x ATR
  Rueckkehr zur Mitte  Kurs war unter dem unteren Band (RSI ueberverkauft) und kehrt
                     zurueck -> Ziel: mittleres Band; Short spiegelbildlich
  Ausbruch           Nach Ruhephase Schluss ueber dem 20-Bar-Hoch mit 1,5-fachem Volumen,
                     nicht gegen den Trend der hoeheren Zeiteinheit

Dieselbe Funktion `compute_signals` wird im Backtest und live benutzt,
damit beide exakt gleich entscheiden.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, ema, macd_hist, rsi, vwap_daily

TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass
class Signal:
    side: str  # "long" | "short"
    price: float
    atr: float
    sl: float
    tp: float
    ts: int
    score: int = 0
    strategy: str = "trend"
    regime: str = ""


def _trend(trend_df: pd.DataFrame, s: dict, trend_tf: str) -> pd.DataFrame:
    t = trend_df[["ts", "close"]].copy()
    fast = ema(t["close"], s["trend_ema_fast"])
    slow = ema(t["close"], s["trend_ema_slow"])
    up = (fast > slow) & (t["close"] > fast)
    down = (fast < slow) & (t["close"] < fast)
    t["trend"] = np.where(up, 1, np.where(down, -1, 0))
    # zu wenig Historie fuer die langsame EMA -> kein Trend
    t.loc[t.index[: s["trend_ema_slow"]], "trend"] = 0
    # ein Trend-Bar ist erst nach seinem Schluss nutzbar
    t["avail"] = t["ts"] + TF_MS[trend_tf]
    return t[["avail", "trend"]]


REGIMES = {
    "trend_up": "Trend aufwaerts",
    "trend_down": "Trend abwaerts",
    "range": "Seitwaerts",
    "squeeze": "Ruhephase - Ausbruch moeglich",
    "chaos": "Chaos - zu wild",
    "unclear": "Unklar",
}
STRATEGY_NAMES = {"trend": "Trend-Ruecksetzer", "range": "Rueckkehr zur Mitte", "breakout": "Ausbruch"}


def _rolling_rank(x: pd.Series, window: int) -> pd.Series:
    """Wo liegt der aktuelle Wert im Vergleich zu den letzten `window` Werten? (0..1)"""
    return x.rolling(window, min_periods=window // 2).rank(pct=True)


def compute_signals(df: pd.DataFrame, trend_df: pd.DataFrame, s: dict,
                    tf: str, trend_tf: str) -> pd.DataFrame:
    """Marktlage erkennen und je Lage die passende Strategie pruefen.

    Ergebnis-Spalten: regime, signal (+1/-1/0), strategy, sl_dist, tp_dist, score
    """
    d = df.copy().reset_index(drop=True)
    d["avail"] = d["ts"] + TF_MS[tf]
    d = pd.merge_asof(
        d.sort_values("avail"),
        _trend(trend_df, s, trend_tf).sort_values("avail"),
        on="avail",
        direction="backward",
    )
    d["trend"] = d["trend"].fillna(0)

    c = d["close"]
    d["ema_f"] = ema(c, s["ema_fast"])
    d["ema_s"] = ema(c, s["ema_slow"])
    d["rsi"] = rsi(c, s["rsi_period"])
    d["macd_h"] = macd_hist(c)
    d["adx"] = adx(d, s["atr_period"])
    d["bb_up"], d["bb_lo"] = bollinger(c)
    d["bb_mid"] = (d["bb_up"] + d["bb_lo"]) / 2
    d["bb_width"] = (d["bb_up"] - d["bb_lo"]) / d["bb_mid"]
    d["vwap"] = vwap_daily(d)
    d["atr"] = atr(d, s["atr_period"])
    d["atr_pct"] = d["atr"] / c * 100
    d["vol_ma"] = d["volume"].rolling(20).mean()
    n_don = s.get("donchian", 20)
    d["don_hi"] = d["high"].rolling(n_don).max().shift()
    d["don_lo"] = d["low"].rolling(n_don).min().shift()
    width_rank = _rolling_rank(d["bb_width"], 120)
    atr_rank = _rolling_rank(d["atr_pct"], 120)
    prev = d.shift()

    # ---------------- Marktlage ----------------
    chaos = (d["atr_pct"] > s["max_atr_pct"]) | (atr_rank > s.get("chaos_atr_rank", 0.97))
    squeeze = width_rank <= s.get("squeeze_pct", 0.2)
    trending = d["adx"] >= s.get("adx_trend", 22)
    ranging = d["adx"] < s.get("adx_range", 18)
    d["regime"] = np.select(
        [chaos, squeeze, trending & (d["trend"] == 1), trending & (d["trend"] == -1), ranging],
        ["chaos", "squeeze", "trend_up", "trend_down", "range"],
        default="unclear",
    )
    tradable = ~chaos & (d["atr_pct"] >= s["min_atr_pct"])
    vol_ok = d["volume"] >= s["volume_factor"] * d["vol_ma"]
    enabled = set(s.get("strategies", ["trend", "range", "breakout"]))

    # ---------------- 1) Trend-Ruecksetzer ----------------
    long_event = (
        ((prev["rsi"] < s["rsi_long_trigger"]) & (d["rsi"] >= s["rsi_long_trigger"]))
        | ((prev["macd_h"] <= 0) & (d["macd_h"] > 0))
        | ((prev["close"] <= prev["ema_f"]) & (c > d["ema_f"]))
    )
    short_event = (
        ((prev["rsi"] > s["rsi_short_trigger"]) & (d["rsi"] <= s["rsi_short_trigger"]))
        | ((prev["macd_h"] >= 0) & (d["macd_h"] < 0))
        | ((prev["close"] >= prev["ema_f"]) & (c < d["ema_f"]))
    )
    long_score = (
        ((c > d["ema_s"]) & (d["ema_f"] > d["ema_s"])).astype(int)   # Struktur
        + (d["macd_h"] > 0).astype(int)                                # Momentum
        + (d["adx"] >= s["adx_min"]).astype(int)                       # Trendstaerke
        + vol_ok.astype(int)                                           # Volumen
        + ((d["rsi"] < s["rsi_overbought"]) & (c < d["bb_up"])).astype(int)  # nicht ueberkauft
        + (c > d["vwap"]).astype(int)                                  # ueber VWAP
    )
    short_score = (
        ((c < d["ema_s"]) & (d["ema_f"] < d["ema_s"])).astype(int)
        + (d["macd_h"] < 0).astype(int)
        + (d["adx"] >= s["adx_min"]).astype(int)
        + vol_ok.astype(int)
        + ((d["rsi"] > s["rsi_oversold"]) & (c > d["bb_lo"])).astype(int)
        + (c < d["vwap"]).astype(int)
    )
    in_trend_lage = d["regime"].isin(["trend_up", "trend_down", "unclear"])
    t_on = "trend" in enabled
    trend_long = (d["trend"] == 1) & in_trend_lage & long_event & (long_score >= s["min_score"]) & t_on
    trend_short = (d["trend"] == -1) & in_trend_lage & short_event & (short_score >= s["min_score"]) & t_on

    # ---------------- 2) Rueckkehr zur Mitte (nur seitwaerts) ----------------
    is_range = (d["regime"] == "range") & ("range" in enabled)
    was_oversold = d["rsi"].rolling(3).min() < s["rsi_oversold"]
    was_overbought = d["rsi"].rolling(3).max() > s["rsi_overbought"]
    mr_sl = s.get("mr_sl_atr", 1.0) * d["atr"]
    mr_long = is_range & (prev["close"] < prev["bb_lo"]) & (c > d["bb_lo"]) & was_oversold \
        & ((d["bb_mid"] - c) >= s.get("mr_min_rr", 1.0) * mr_sl)
    mr_short = is_range & (prev["close"] > prev["bb_up"]) & (c < d["bb_up"]) & was_overbought \
        & ((c - d["bb_mid"]) >= s.get("mr_min_rr", 1.0) * mr_sl)

    # ---------------- 3) Ausbruch nach Ruhephase ----------------
    recent_squeeze = squeeze.astype(int).rolling(s.get("squeeze_lookback", 6)).max().shift() == 1
    bo_vol = d["volume"] >= s.get("bo_volume_factor", 1.5) * d["vol_ma"]
    bo_ok = recent_squeeze & bo_vol & ("breakout" in enabled)
    bo_long = bo_ok & (c > d["don_hi"]) & (prev["close"] <= prev["don_hi"]) & (d["trend"] != -1)
    bo_short = bo_ok & (c < d["don_lo"]) & (prev["close"] >= prev["don_lo"]) & (d["trend"] != 1)

    # ---------------- zusammenfuehren (Ausbruch > Trend > Mitte) ----------------
    conds = [bo_long & tradable, bo_short & tradable, trend_long & tradable, trend_short & tradable,
             mr_long & tradable, mr_short & tradable]
    d["signal"] = np.select(conds, [1, -1, 1, -1, 1, -1], default=0)
    d["strategy"] = np.select(conds, ["breakout", "breakout", "trend", "trend", "range", "range"], default="")
    d["sl_dist"] = np.select(
        conds,
        [s.get("bo_sl_atr", 1.5) * d["atr"]] * 2 + [s["sl_atr"] * d["atr"]] * 2 + [mr_sl] * 2,
        default=np.nan,
    )
    d["tp_dist"] = np.select(
        conds,
        [s.get("bo_tp_atr", 3.0) * d["atr"]] * 2 + [s["tp_atr"] * d["atr"]] * 2
        + [d["bb_mid"] - c, c - d["bb_mid"]],
        default=np.nan,
    )
    d["score"] = np.select(conds, [6, 6, long_score, short_score, 6, 6], default=0)
    warm = max(s["ema_slow"] * 2, 120)
    d.loc[d.index[:warm], "signal"] = 0
    return d


def levels(side: str, price: float, atr_value: float, s: dict) -> tuple[float, float]:
    """SL/TP der Trend-Strategie (ATR-Vielfache)."""
    return levels_from(side, price, s["sl_atr"] * atr_value, s["tp_atr"] * atr_value)


def levels_from(side: str, price: float, sl_dist: float, tp_dist: float) -> tuple[float, float]:
    if side == "long":
        return price - sl_dist, price + tp_dist
    return price + sl_dist, price - tp_dist


def last_closed_signal(sig_df: pd.DataFrame, s: dict, funding: float | None = None) -> Signal | None:
    """Signal des letzten ABGESCHLOSSENEN Bars (der letzte Bar ist noch offen)."""
    if len(sig_df) < 2:
        return None
    row = sig_df.iloc[-2]
    if row["signal"] == 0:
        return None
    side = "long" if row["signal"] == 1 else "short"
    if funding is not None:
        # sehr hohe Funding-Rate = Markt einseitig ueberhebelt -> nicht mitlaufen
        if side == "long" and funding > s["max_funding_rate"]:
            return None
        if side == "short" and funding < -s["max_funding_rate"]:
            return None
    price = float(row["close"])
    sl, tp = levels_from(side, price, float(row["sl_dist"]), float(row["tp_dist"]))
    return Signal(side, price, float(row["atr"]), sl, tp, int(row["ts"]), int(row["score"]),
                  str(row["strategy"]), str(row["regime"]))


def snapshot(sig_df: pd.DataFrame, s: dict) -> dict:
    """Was der Bot gerade 'sieht' (letzter abgeschlossener Bar) - fuer die Oberflaeche."""
    row = sig_df.iloc[-2]
    trend = {1: "aufwaerts", -1: "abwaerts", 0: "seitwaerts"}[int(row["trend"])]
    regime = str(row["regime"])
    watching = {
        "trend_up": "Trend-Ruecksetzer LONG", "trend_down": "Trend-Ruecksetzer SHORT",
        "range": "Rueckkehr zur Mitte (Kauf am unteren / Verkauf am oberen Bollinger-Band)",
        "squeeze": "Ausbruch aus der Ruhephase (mit Volumen)", "chaos": "nichts - Markt zu wild",
        "unclear": "Trend-Ruecksetzer, falls der Tagestrend passt",
    }[regime]
    bias = "long" if row["trend"] == 1 else "short" if row["trend"] == -1 else None
    view = {
        "ts": int(row["ts"]),
        "close": float(row["close"]),
        "trend_1h": trend,
        "regime": regime,
        "regime_text": REGIMES[regime],
        "watching": watching,
        "rsi": round(float(row["rsi"]), 1),
        "macd_hist": float(row["macd_h"]),
        "adx": round(float(row["adx"]), 1),
        "atr_pct": round(float(row["atr_pct"]), 3),
        "bb_width_pct": round(float(row["bb_width"]) * 100, 2) if pd.notna(row["bb_width"]) else None,
        "vwap": float(row["vwap"]) if pd.notna(row["vwap"]) else None,
        "bias": bias,
        "signal": int(row["signal"]),
    }
    if bias and regime in ("trend_up", "trend_down", "unclear"):
        sl, tp = levels(bias, float(row["close"]), float(row["atr"]), s)
        view["plan"] = {"side": bias, "entry": float(row["close"]), "sl": sl, "tp": tp}
    return view


def trail_stop(side: str, entry: float, sl_init: float, current_sl: float,
               price: float, atr_value: float, s: dict, fee: float) -> float | None:
    """Neuer Stop (nur in Gewinnrichtung) oder None.

    Ab +breakeven_at_r: Stop mindestens auf Einstand (+Gebuehren),
    danach im Abstand trail_atr x ATR hinter dem Kurs.
    """
    r = abs(entry - sl_init)
    if r <= 0:
        return None
    if side == "long":
        if (price - entry) / r < s["breakeven_at_r"]:
            return None
        cand = max(entry * (1 + 2 * fee), price - s["trail_atr"] * atr_value)
        return cand if cand > current_sl and cand < price else None
    if (entry - price) / r < s["breakeven_at_r"]:
        return None
    cand = min(entry * (1 - 2 * fee), price + s["trail_atr"] * atr_value)
    return cand if cand < current_sl and cand > price else None
