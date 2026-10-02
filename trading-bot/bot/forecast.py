"""KI-Kursprognose: wohin koennte der Kurs in den naechsten 30 Minuten laufen?

Lernt aus den letzten ~1000 5-Minuten-Kerzen eines Markts (ohne Zusatzpakete):
Merkmale (Renditen 5 min ... 2 h, RSI, Abstand zu EMAs in ATR, Bollinger-Lage, MACD, Volumen,
Volatilitaet, Tageszeit) -> je Abstand (5, 10, ... 30 min) eine Ridge-Regression auf die
kuenftige Rendite. Dazu ein Unsicherheitsband aus den Fehlern auf ungesehenen Daten.

Ehrlich bleiben: Gelernt wird nur mit den aelteren 80 %, geprueft an den neuesten 20 %.
Ist das Modell dort nicht besser als "Kurs bleibt gleich", wird das angezeigt
(Vorhersagekraft "keine"). Jede Live-Prognose wird gespeichert und nach 30 Minuten mit dem
echten Kurs verglichen (Live-Trefferquote).
"""
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import atr, bollinger, ema, macd_hist, rsi

log = logging.getLogger(__name__)

STEP_MIN = 5
HORIZONS = [1, 2, 3, 4, 5, 6]            # in 5-Minuten-Schritten -> 5 ... 30 min
BAND_Z = 1.28                            # 80-%-Band
RIDGE = 5.0
FEATURE_NAMES = ["r1", "r3", "r6", "r12", "r24", "rsi", "d_ema20", "d_ema50", "bb_pos", "macd",
                 "vol", "atr_pct", "hour_sin", "hour_cos"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merkmale je Kerze (nur Vergangenheit bis einschliesslich dieser Kerze)."""
    c = df["close"].astype(float)
    a = atr(df, 14).replace(0, np.nan)
    lr = np.log(c)
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24):
        f[f"r{n}"] = (lr - lr.shift(n)) * 100
    f["rsi"] = (rsi(c, 14) - 50) / 50
    f["d_ema20"] = (c - ema(c, 20)) / a
    f["d_ema50"] = (c - ema(c, 50)) / a
    up, lo = bollinger(c, 20, 2.0)
    f["bb_pos"] = ((c - lo) / (up - lo).replace(0, np.nan) - 0.5) * 2
    f["macd"] = macd_hist(c) / a
    vol = df["volume"].astype(float)
    f["vol"] = np.log((vol + 1e-9) / (vol.rolling(48).mean() + 1e-9))
    f["atr_pct"] = a / c * 100
    hours = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.hour + pd.to_datetime(df["ts"], unit="ms", utc=True).dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    return f[FEATURE_NAMES].clip(-10, 10)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    reg = lam * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return w, mu, sd


def _predict(model, X: np.ndarray) -> np.ndarray:
    w, mu, sd = model
    return np.c_[np.ones(len(X)), (X - mu) / sd] @ w


def build(df: pd.DataFrame, test_share: float = 0.2) -> dict | None:
    """Modell aus 5-Minuten-Kerzen (Spalten ts, open, high, low, close, volume; letzte Kerze darf offen sein).
    Rueckgabe: Prognosepfad, Band und Guete auf ungesehenen Daten - oder None bei zu wenig Daten."""
    df = df.reset_index(drop=True)
    closed = df.iloc[:-1]                         # offene Kerze nicht zum Lernen
    if len(closed) < 300:
        return None
    feats = make_features(df)
    X_all = feats.to_numpy(float)
    lr = np.log(df["close"].astype(float).to_numpy())
    n_closed = len(closed)
    path, band, quality = [], [], []
    for h in HORIZONS:
        # Ziel: Rendite (in %) von Kerze i bis i+h - nur wo i+h noch eine abgeschlossene Kerze ist
        y = np.full(len(df), np.nan)
        y[: n_closed - h] = (lr[h:n_closed] - lr[: n_closed - h]) * 100
        ok = ~np.isnan(X_all).any(1) & ~np.isnan(y)
        idx = np.where(ok)[0]
        if len(idx) < 200:
            return None
        split = int(len(idx) * (1 - test_share))
        tr, te = idx[:split], idx[split:]
        model = _ridge(X_all[tr], y[tr])
        pred_te = _predict(model, X_all[te])
        err = y[te] - pred_te
        mae_model, mae_naive = float(np.mean(np.abs(err))), float(np.mean(np.abs(y[te])))
        hits = np.sign(pred_te) == np.sign(y[te])
        moved = y[te] != 0
        quality.append({
            "min": h * STEP_MIN,
            "skill": round(1 - mae_model / mae_naive, 4) if mae_naive > 0 else 0.0,   # >0 = besser als "bleibt gleich"
            "hit": round(float(hits[moved].mean()), 3) if moved.any() else None,
            "n_test": int(len(te)),
        })
        final = _ridge(X_all[idx], y[idx])        # fuer die Live-Prognose mit allen Daten lernen
        x_now = X_all[-1:]
        if np.isnan(x_now).any():
            return None
        path.append(float(_predict(final, x_now)[0]))
        band.append(float(np.std(err)) * BAND_Z)
    last = float(df["close"].iloc[-1])
    t0 = int(df["ts"].iloc[-1]) // 1000
    pts = [{"time": t0, "mid": last, "lo": last, "hi": last}]
    for h, r, b in zip(HORIZONS, path, band):
        mid = last * math.exp(r / 100)
        pts.append({"time": t0 + h * STEP_MIN * 60, "mid": mid, "lo": last * math.exp((r - b) / 100),
                    "hi": last * math.exp((r + b) / 100)})
    q30 = quality[-1]
    skill = float(np.mean([q["skill"] for q in quality]))
    return {
        "price": last, "time": t0, "points": pts,
        "change_pct": round(path[-1], 3), "band_pct": round(band[-1], 3),
        "direction": "hoch" if path[-1] > band[-1] * 0.25 else "runter" if path[-1] < -band[-1] * 0.25 else "seitwaerts",
        "quality": quality, "skill": round(skill, 4), "hit_30m": q30["hit"],
        "useful": skill > 0.005 and (q30["hit"] or 0) > 0.52,
        "trained_on": int(n_closed),
    }


class ForecastLog:
    """Live-Prognosen merken und nach 30 Minuten mit dem echten Kurs vergleichen (Trefferquote)."""

    def __init__(self, path: Path | None = None, keep: int = 500):
        self.path, self.keep = path, keep
        self.items: list[dict] = []
        if path and path.exists():
            try:
                self.items = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.items = []

    def add(self, sym: str, fc: dict) -> None:
        # hoechstens eine Prognose je Markt und 5-Minuten-Kerze
        if any(x["sym"] == sym and x["time"] == fc["time"] for x in self.items[-50:]):
            return
        self.items.append({"sym": sym, "time": fc["time"], "price": fc["price"], "target": fc["points"][-1]["mid"],
                           "lo": fc["points"][-1]["lo"], "hi": fc["points"][-1]["hi"], "actual": None})
        self.items = self.items[-self.keep:]
        self._save()

    def resolve(self, sym: str, candles: pd.DataFrame) -> None:
        """Offene Prognosen auswerten, deren 30 Minuten vorbei sind."""
        closes = dict(zip((candles["ts"] // 1000).astype(int), candles["close"].astype(float)))
        changed = False
        for x in self.items:
            if x["sym"] == sym and x["actual"] is None:
                t = x["time"] + (HORIZONS[-1] - 1) * STEP_MIN * 60   # Kerze, die 30 min spaeter schliesst
                if t in closes:
                    x["actual"] = closes[t]
                    changed = True
        if changed:
            self._save()

    def stats(self, sym: str | None = None) -> dict:
        done = [x for x in self.items if x["actual"] is not None and (sym is None or x["sym"] == sym)]
        moved = [x for x in done if x["actual"] != x["price"] and x["target"] != x["price"]]
        hits = [(x["target"] - x["price"]) * (x["actual"] - x["price"]) > 0 for x in moved]
        inband = [x["lo"] <= x["actual"] <= x["hi"] for x in done]
        return {"n": len(done), "hit": round(sum(hits) / len(hits), 3) if hits else None,
                "in_band": round(sum(inband) / len(inband), 3) if inband else None}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items), encoding="utf-8")
        except OSError as e:
            log.debug("Prognose-Log: %s", e)


class Forecaster:
    """Prognose je Markt, neu berechnet sobald eine neue 5-Minuten-Kerze da ist (sonst aus dem Speicher)."""

    def __init__(self, candles_fn, log_path: Path | None = None):
        self.candles_fn = candles_fn
        self.cache: dict[str, tuple[float, dict]] = {}
        self.log = ForecastLog(log_path)

    def get(self, sym: str) -> dict:
        hit = self.cache.get(sym)
        if hit and time.time() - hit[0] < 20:
            return hit[1]
        df = self.candles_fn(sym, "5m", 1000)
        self.log.resolve(sym, df.iloc[:-1])
        fc = build(df)
        if fc is None:
            out = {"symbol": sym, "ok": False, "msg": "Zu wenig Kursdaten fuer eine Prognose"}
        else:
            self.log.add(sym, fc)
            out = {"symbol": sym, "ok": True, **fc, "live": self.log.stats(sym)}
        self.cache[sym] = (time.time(), out)
        return out
