"""Einstiegslogik: Trend-Pullback mit Punktesystem.

Long (Short spiegelbildlich):
  Pflicht:     1h-Trend aufwaerts (EMA50 > EMA200, Kurs > EMA50)
               Volatilitaet (ATR) im erlaubten Bereich
  Ausloeser:   RSI kreuzt ueber 40  ODER  MACD-Histogramm dreht ueber 0
               ODER  Kurs erobert die EMA21 zurueck
  Punkte:      Struktur (Kurs/EMA21 ueber EMA50), MACD > 0, ADX >= Mindestwert,
               Volumen >= Schnitt, nicht ueberkauft (RSI, Bollinger), Kurs > VWAP
  Einstieg ab `min_score` Punkten (von 6). Niedriger = mehr Trades.

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


def compute_signals(df: pd.DataFrame, trend_df: pd.DataFrame, s: dict,
                    tf: str, trend_tf: str) -> pd.DataFrame:
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
    d["vwap"] = vwap_daily(d)
    d["atr"] = atr(d, s["atr_period"])
    d["atr_pct"] = d["atr"] / c * 100
    atr_ok = d["atr_pct"].between(s["min_atr_pct"], s["max_atr_pct"])
    vol_ok = d["volume"] >= s["volume_factor"] * d["volume"].rolling(20).mean()
    trend_ok = d["adx"] >= s["adx_min"]
    prev = d.shift()

    # Ausloeser (Timing): mindestens eines muss in DIESEM Bar passieren
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

    # Bestaetigungen: je erfuellter Punkt +1
    long_score = (
        ((c > d["ema_s"]) & (d["ema_f"] > d["ema_s"])).astype(int)   # Struktur
        + (d["macd_h"] > 0).astype(int)                                # Momentum
        + trend_ok.astype(int)                                         # Trendstaerke
        + vol_ok.astype(int)                                           # Volumen
        + ((d["rsi"] < s["rsi_overbought"]) & (c < d["bb_up"])).astype(int)  # nicht ueberkauft
        + (c > d["vwap"]).astype(int)                                  # ueber VWAP
    )
    short_score = (
        ((c < d["ema_s"]) & (d["ema_f"] < d["ema_s"])).astype(int)
        + (d["macd_h"] < 0).astype(int)
        + trend_ok.astype(int)
        + vol_ok.astype(int)
        + ((d["rsi"] > s["rsi_oversold"]) & (c > d["bb_lo"])).astype(int)
        + (c < d["vwap"]).astype(int)
    )

    long_ = (d["trend"] == 1) & atr_ok & long_event & (long_score >= s["min_score"])
    short = (d["trend"] == -1) & atr_ok & short_event & (short_score >= s["min_score"])
    d["signal"] = np.where(long_, 1, np.where(short, -1, 0))
    d["score"] = np.where(long_, long_score, np.where(short, short_score, 0))
    d.loc[d.index[: s["ema_slow"] * 2], "signal"] = 0
    return d


def levels(side: str, price: float, atr_value: float, s: dict) -> tuple[float, float]:
    if side == "long":
        return price - s["sl_atr"] * atr_value, price + s["tp_atr"] * atr_value
    return price + s["sl_atr"] * atr_value, price - s["tp_atr"] * atr_value


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
    sl, tp = levels(side, float(row["close"]), float(row["atr"]), s)
    return Signal(side, float(row["close"]), float(row["atr"]), sl, tp, int(row["ts"]), int(row["score"]))


def snapshot(sig_df: pd.DataFrame, s: dict) -> dict:
    """Was der Bot gerade 'sieht' (letzter abgeschlossener Bar) - fuer die Oberflaeche."""
    row = sig_df.iloc[-2]
    trend = {1: "aufwaerts", -1: "abwaerts", 0: "seitwaerts"}[int(row["trend"])]
    bias = "long" if row["trend"] == 1 else "short" if row["trend"] == -1 else None
    view = {
        "ts": int(row["ts"]),
        "close": float(row["close"]),
        "trend_1h": trend,
        "rsi": round(float(row["rsi"]), 1),
        "macd_hist": float(row["macd_h"]),
        "adx": round(float(row["adx"]), 1),
        "atr_pct": round(float(row["atr_pct"]), 3),
        "vwap": float(row["vwap"]) if pd.notna(row["vwap"]) else None,
        "bias": bias,
        "signal": int(row["signal"]),
    }
    if bias:
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
