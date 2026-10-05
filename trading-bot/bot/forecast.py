"""KI-Kursprognose: Entscheidung LONG/SHORT fuer die naechsten 30 Minuten + Prognose-Kerzen.

Je Markt (ohne Zusatzpakete):
- Merkmale aus 5-Minuten-Kerzen: Renditen 5 min ... 1 Tag, RSI, Abstand zu EMAs (auch die der
  1h-/4h-Zeitebene) in ATR, Bollinger-Lage, MACD, Lage in der Tagesspanne, Kerzenform,
  Volumen und Kauf-/Verkaufsdruck, Volatilitaet, Tageszeit - bei Altcoins zusaetzlich BTC.
- Je Abstand (5 ... 30 min) eine logistische Regression: Wahrscheinlichkeit, dass der Kurs
  dann HOEHER steht. Daraus die Entscheidung (immer LONG oder SHORT) und der Prognosepfad.
- Lernt laufend: Jede neue Kerze wird gespeichert (Gedaechtnis waechst bis ~70 Tage), bei jeder
  neuen 5-Minuten-Kerze wird neu gelernt, neuere Daten zaehlen staerker.
- Prueft sich selbst: Guete an ungesehenen Daten (letzte 20 %), und jede Live-Entscheidung wird
  nach 30 Minuten mit dem echten Kurs verglichen. Liegt sie zuletzt oft falsch, sinkt die
  angezeigte Sicherheit (Kalibrierung) und es erscheint eine Warnung.
"""
import json
import logging
import math
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import atr, bollinger, ema, macd_hist, rsi
from .orderblocks import detect as ob_detect
from .patterns import analyze as pattern_analyze

log = logging.getLogger(__name__)

STEP_MIN = 5
HORIZONS = [1, 2, 3, 4, 5, 6]            # in 5-Minuten-Schritten -> 5 ... 30 min
MAX_HISTORY = 20_000                     # gespeicherte 5-Minuten-Kerzen je Markt (~70 Tage)
HALF_LIFE = 3_000                        # Gewicht neuerer Kerzen: halbiert sich alle ~10 Tage
L2 = 2.0
BAND_Z = 1.28                            # 80-%-Band
BOOST_ROWS = 12_000                      # Baum-Modell lernt aus den neuesten 12.000 Kerzen (Rechenzeit)
MIN_BOOK_BARS = 800                      # Orderbuch-Druck erst ab ~3 Tagen Aufzeichnung als Merkmal
BACKFILL_DAYS = 70                       # beim ersten Start so viel Vergangenheit von Bitget laden
GRID = [(0.5, 1500), (2.0, 1500), (2.0, 6000), (8.0, 6000), (8.0, 20000)]   # (L2, Halbwertszeit in Kerzen)


def make_features(df: pd.DataFrame, leader=None, funding: pd.DataFrame | None = None,
                  macro: pd.DataFrame | None = None, gold: bool = False, step_min: int = None) -> pd.DataFrame:
    """leader: DataFrame (BTC) oder {Name: DataFrame} (z. B. BTC und ETH); funding: Spalten ts, rate."""
    """Merkmale je Kerze (nur Vergangenheit bis einschliesslich dieser Kerze)."""
    c, h, lo, o = (df[k].astype(float) for k in ("close", "high", "low", "open"))
    a = atr(df, 14).replace(0, np.nan)
    lr = np.log(c)
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24, 48, 96, 288):          # 5 min ... 1 Tag
        f[f"r{n}"] = (lr - lr.shift(n)) * 100
    for k in (1, 2):                                   # die einzelnen vorigen Kerzen (fuer Wechselwirkungen)
        f[f"r1_lag{k}"] = f["r1"].shift(k)
    f["rsi"] = (rsi(c, 14) - 50) / 50
    f["rsi_1h"] = (rsi(c, 168) - 50) / 50
    for span, name in ((20, "e20"), (50, "e50"), (600, "e1h50"), (2400, "e4h50")):
        f[f"d_{name}"] = (c - ema(c, span)) / a
    up, dn = bollinger(c, 20, 2.0)
    f["bb_pos"] = ((c - dn) / (up - dn).replace(0, np.nan) - 0.5) * 2
    f["macd"] = macd_hist(c) / a
    f["macd_1h"] = macd_hist(c, 144, 312, 108) / a
    hi288, lo288 = h.rolling(288, min_periods=48).max(), lo.rolling(288, min_periods=48).min()
    f["day_pos"] = ((c - lo288) / (hi288 - lo288).replace(0, np.nan) - 0.5) * 2
    rng = (h - lo).replace(0, np.nan)
    f["body"] = (c - o) / rng                          # Kerzenform: Koerper ...
    f["wick_up"] = (h - np.maximum(c, o)) / rng        # ... und Dochte
    f["wick_dn"] = (np.minimum(c, o) - lo) / rng
    vol = df["volume"].astype(float)
    f["vol"] = np.log((vol + 1e-9) / (vol.rolling(48).mean() + 1e-9))
    press = np.sign(c - o) * vol                        # Kauf-/Verkaufsdruck
    f["press"] = press.rolling(12).sum() / (vol.rolling(12).sum() + 1e-9)
    f["atr_pct"] = a / c * 100
    f["atr_trend"] = np.log(a / a.rolling(288, min_periods=48).mean())
    t = pd.to_datetime(df["ts"], unit="ms", utc=True)
    hours = t.dt.hour + t.dt.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
    leaders = {"btc": leader} if isinstance(leader, pd.DataFrame) else (leader or {})
    for name, ld in leaders.items():
        if ld is None or not len(ld):
            continue
        lc = pd.Series(ld["close"].astype(float).to_numpy(), index=ld["ts"].to_numpy())
        lc = lc[~lc.index.duplicated()].reindex(df["ts"].to_numpy()).ffill()
        llr = np.log(lc.to_numpy())
        for n in (1, 3, 12, 48):
            f[f"{name}_r{n}"] = (llr - np.r_[np.full(n, np.nan), llr[:-n]]) * 100
    if funding is not None and len(funding):
        fr = pd.Series(funding["rate"].astype(float).to_numpy(), index=funding["ts"].astype("int64").to_numpy())
        fr = fr[~fr.index.duplicated()].sort_index()
        pos = np.searchsorted(fr.index.to_numpy(), df["ts"].to_numpy(), side="right") - 1
        vals = np.where(pos >= 0, fr.to_numpy()[np.clip(pos, 0, None)], np.nan)
        f["funding"] = vals * 1e4                                     # in Basispunkten
        f["funding_chg"] = f["funding"] - f["funding"].shift(96)
    # Order Blocks wie im Chart: Abstand zur naechsten Gegen-/Stuetz-Zone (in ATR) und Antests
    o_, h_, l_, c_ = (df[k].astype(float).to_numpy() for k in ("open", "high", "low", "close"))
    above, below, sup, _ = ob_detect(o_, h_, l_, c_, a.to_numpy(), 1.0, 3)
    av = a.to_numpy()
    f["ob_up"] = np.where(np.isnan(above), 10.0, (above - c_) / av)
    f["ob_dn"] = np.where(np.isnan(below), 10.0, (c_ - below) / av)
    f["ob_sup"] = sup.astype(float)
    if len(df) > 400:                                                  # Order Blocks der 1h-Zeitebene
        t1 = pd.to_datetime(df["ts"], unit="ms", utc=True)
        g = df.set_index(t1).resample("1h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        if len(g) > 30:
            a1 = atr(g.reset_index(drop=True), 14).to_numpy()
            ab1, be1, su1, _ = ob_detect(*(g[k].to_numpy(float) for k in ("open", "high", "low", "close")), a1, 1.5, 3)
            known = (g.index + pd.Timedelta(hours=1)).as_unit("ns").asi8 // 1_000_000   # erst nach Kerzenschluss bekannt
            pos = np.searchsorted(known, df["ts"].to_numpy(), side="right") - 1
            okp = pos >= 0
            pc = np.clip(pos, 0, None)
            f["ob1h_up"] = np.where(okp & ~np.isnan(ab1[pc]), (ab1[pc] - c_) / av, 10.0)
            f["ob1h_dn"] = np.where(okp & ~np.isnan(be1[pc]), (c_ - be1[pc]) / av, 10.0)
            f["ob1h_sup"] = np.where(okp, su1[pc], 0).astype(float)
    # Kerzen- und Chart-Muster auf 5 min, 1 h und 4 h (hoehere erst nach Kerzenschluss bekannt)
    pa = pattern_analyze(df)
    for col in ("cdl_score", "pat_score", "pattern_score", "sr_up", "sr_dn", "pat_dbl_top", "pat_dbl_bottom",
                "pat_break_up", "pat_break_dn", "pat_triangle", "cdl_doji", "cdl_inside"):
        f[f"m5_{col}"] = pa[col].to_numpy()
    if len(df) > 600:
        t1 = pd.to_datetime(df["ts"], unit="ms", utc=True)
        for rule, hours, name in (("1h", 1, "m1h"), ("4h", 4, "m4h")):
            g = df.set_index(t1).resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            if len(g) < 40:
                continue
            pg = pattern_analyze(g.reset_index(drop=True))
            known = (g.index + pd.Timedelta(hours=hours)).as_unit("ns").asi8 // 1_000_000
            pos = np.searchsorted(known, df["ts"].to_numpy(), side="right") - 1
            okp, pc = pos >= 0, np.clip(pos, 0, None)
            for col in ("pattern_score", "cdl_score", "pat_score", "pat_hh_hl", "pat_lh_ll"):
                f[f"{name}_{col}"] = np.where(okp, pg[col].to_numpy()[pc], 0.0)
    if macro is not None and len(macro):
        from .macro import merge_macro
        bars = pd.DataFrame({"avail": df["ts"].astype("int64").to_numpy() + (step_min or STEP_MIN) * 60_000},
                            index=df.index)
        f["macro"] = np.asarray(merge_macro(bars, macro, gold=gold), dtype=float)   # bekannt ab Kerzenschluss
    for col, name in (("wall_bid", "wall_bid"), ("wall_ask", "wall_ask")):          # Orderbuch-Waende (live)
        if col in df and df[col].notna().sum() >= MIN_BOOK_BARS:
            f[name] = df[col].astype(float).fillna(5.0)
    if "book_imb" in df:                                              # live aufgezeichneter Orderbuch-Druck
        has = df["book_imb"].notna()
        if has.sum() >= MIN_BOOK_BARS:
            f["book_imb"] = df["book_imb"].astype(float).fillna(0.0) * 3
            f["book_has"] = has.astype(float)
    return f.replace([np.inf, -np.inf], np.nan).clip(-10, 10)


def _logit_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = L2, iters: int = 25):
    """Gewichtete logistische Regression (Newton-Verfahren) mit L2-Regularisierung."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    beta = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[0] = 0.0
    w = w / w.mean()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (w * (p - y)) + reg * beta
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + np.diag(reg) + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-6:
            break
    return beta, mu, sd


def _logit_p(model, X: np.ndarray) -> np.ndarray:
    beta, mu, sd = model
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ beta, -30, 30)))


def _calibrate(p: float, live: dict) -> tuple[float, str]:
    """Sicherheit an die echte Live-Trefferquote anpassen (lernen aus Fehlern)."""
    n, hit = live.get("n_recent", 0), live.get("hit_recent")
    if n < 20 or hit is None:
        return p, ""
    # Liegt die KI zuletzt oft daneben, wird sie vorsichtiger (Sicherheit Richtung 50 %)
    trust = max(0.0, min(1.0, (hit - 0.45) / 0.15))
    note = "" if hit >= 0.5 else f"lag zuletzt oft falsch ({round(hit * 100)} % von {n}) - Sicherheit gesenkt"
    return 0.5 + (p - 0.5) * trust, note


class Boost:
    """Gradient Boosting mit kleinen Entscheidungsbaeumen (Tiefe 2) - findet Zusammenhaenge, die eine
    lineare Regression nicht sieht (z. B. "hoher RSI nur bei fallendem Volumen gefaehrlich").
    Merkmale werden in Stufen (Quantile) eingeteilt, damit es auch ohne Zusatzpakete schnell geht."""

    def __init__(self, rounds: int = 80, lr: float = 0.1, bins: int = 16, min_leaf: float = 0.02, lam: float = 5.0):
        self.rounds, self.lr, self.bins, self.min_leaf, self.lam = rounds, lr, bins, min_leaf, lam
        self.edges, self.trees, self.f0 = [], [], 0.0

    def _bin(self, X):
        return np.stack([np.searchsorted(e, X[:, j]) for j, e in enumerate(self.edges)], 1).astype(np.int16)

    def _best(self, Xb, g, h, rows):
        """Bester Schnitt (Merkmal, Stufe) fuer die Zeilen `rows` - oder None."""
        G, H = g[rows].sum(), h[rows].sum()
        best, min_h = None, self.min_leaf * h.sum()
        base = G * G / (H + self.lam)
        for j in range(Xb.shape[1]):
            xb = Xb[rows, j]
            gl = np.cumsum(np.bincount(xb, g[rows], self.bins + 1))[:-1]
            hl = np.cumsum(np.bincount(xb, h[rows], self.bins + 1))[:-1]
            ok = (hl >= min_h) & (H - hl >= min_h)
            if not ok.any():
                continue
            gain = np.where(ok, gl ** 2 / (hl + self.lam) + (G - gl) ** 2 / (H - hl + self.lam) - base, -1)
            k = int(gain.argmax())
            if gain[k] > 0 and (best is None or gain[k] > best[0]):
                best = (gain[k], j, k)
        return best

    def fit(self, X, y, w):
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.bins + 1)[1:-1])) for j in range(X.shape[1])]
        Xb = self._bin(X)
        w = w / w.mean()
        m = float(np.clip(np.average(y, weights=w), 0.01, 0.99))
        self.f0 = math.log(m / (1 - m))
        F = np.full(len(y), self.f0)
        self.trees = []
        for _ in range(self.rounds):
            p = 1 / (1 + np.exp(-F))
            g, h = (p - y) * w, p * (1 - p) * w
            allr = np.arange(len(y))
            root = self._best(Xb, g, h, allr)
            if root is None:
                break
            _, j, k = root
            tree = {"j": j, "k": k, "kids": []}
            for side in (Xb[:, j] <= k, Xb[:, j] > k):
                rows = allr[side]
                sub = self._best(Xb, g, h, rows)
                if sub is None:
                    v = -g[rows].sum() / (h[rows].sum() + self.lam) * self.lr
                    tree["kids"].append({"leaf": (v, v), "j": 0, "k": self.bins})
                    F[rows] += v
                    continue
                _, j2, k2 = sub
                left = rows[Xb[rows, j2] <= k2]
                right = rows[Xb[rows, j2] > k2]
                vl = -g[left].sum() / (h[left].sum() + self.lam) * self.lr
                vr = -g[right].sum() / (h[right].sum() + self.lam) * self.lr
                F[left] += vl
                F[right] += vr
                tree["kids"].append({"leaf": (vl, vr), "j": j2, "k": k2})
            self.trees.append(tree)
        return self

    def proba(self, X):
        Xb = self._bin(X)
        F = np.full(len(X), self.f0)
        for t in self.trees:
            side = Xb[:, t["j"]] > t["k"]
            for s_, kid in ((~side, t["kids"][0]), (side, t["kids"][1])):
                go_right = Xb[:, kid["j"]] > kid["k"]
                F += np.where(s_, np.where(go_right, kid["leaf"][1], kid["leaf"][0]), 0.0)
        return 1 / (1 + np.exp(-F))


def _sr_pct(feats: pd.DataFrame, col: str) -> float | None:
    """Abstand zum naechsten Widerstand/zur naechsten Unterstuetzung in % (letzte abgeschlossene Kerze)."""
    if col not in feats or "atr_pct" not in feats or len(feats) < 2:
        return None
    v, a = float(feats[col].iloc[-2]), float(feats["atr_pct"].iloc[-2])
    return None if v != v or a != a or v >= 10 else round(v * a, 4)


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _local_cal(p_te: np.ndarray, yt: np.ndarray, p_now: float, h: int) -> tuple[float, float, int]:
    """Sicherheit je Meinungsstaerke: Wie oft lag die KI an ungesehenen Daten richtig, wenn sie eine
    aehnlich starke Meinung hatte wie jetzt? -> (kalibrierte Wahrscheinlichkeit "hoeher", z-Wert, Anzahl).
    Ohne nachweisbaren Vorsprung (z < 1) bleibt es bei 0,5."""
    n = len(p_te)
    k = int(min(n, max(300, n * 0.12)))
    near = np.argsort(np.abs(p_te - p_now))[:k]            # die k aehnlichsten Vorhersagen
    up = float(yt[near].mean())
    n_eff = max(1.0, k / h)                                 # ueberlappende Zeitraeume zaehlen weniger
    z = (up - 0.5) / math.sqrt(0.25 / n_eff)
    shrunk = (up * n_eff + 0.5 * 30) / (n_eff + 30)         # bei wenig Daten Richtung 0,5 ziehen
    # strenger als bei einer einzelnen Pruefung: es wird unter 6 Vorhersagezeiten die beste genommen
    trust = max(0.0, min(1.0, (abs(z) - 1.8) / 2.0))
    return 0.5 + (shrunk - 0.5) * trust, z, k


def selective(p_te: np.ndarray, yt: np.ndarray, h: int, thresholds=(0.52, 0.54, 0.56, 0.58, 0.60),
              step_min: int = None) -> list[dict]:
    """Wie oft haette die KI an ungesehenen Daten mindestens X % Sicherheit gezeigt - und wie oft lag sie
    dann richtig? (Sicherheit je Meinungsstaerke wie live, aus den jeweils aehnlichsten Vorhersagen.)"""
    n = len(p_te)
    if n < 300:
        return []
    order = np.argsort(p_te)
    ys = yt[order]
    k = int(min(n, max(300, n * 0.12)))
    csum = np.r_[0.0, np.cumsum(ys)]
    pos = np.arange(n)
    lo = np.clip(pos - k // 2, 0, n - k)
    up = (csum[lo + k] - csum[lo]) / k                     # Anteil "hoeher" bei aehnlicher Meinung
    n_eff = max(1.0, k / h)
    z = (up - 0.5) / math.sqrt(0.25 / n_eff)
    shrunk = (up * n_eff + 0.5 * 30) / (n_eff + 30)
    trust = np.clip((np.abs(z) - 1.8) / 2.0, 0, 1)
    p_cal = 0.5 + (shrunk - 0.5) * trust
    conf = np.maximum(p_cal, 1 - p_cal)
    right = (p_cal >= 0.5) == (ys == 1)
    days = n * (step_min or STEP_MIN) / 1440
    out = []
    for t in thresholds:
        sel = conf >= t
        out.append({"min_conf": t, "per_day": round(float(sel.sum()) / days, 1) if days else 0.0,
                    "hit": round(float(right[sel].mean()), 3) if sel.any() else None, "n": int(sel.sum())})
    return out


def build(df: pd.DataFrame, leader=None, test_share: float = 0.2, live: dict | None = None,
          funding: pd.DataFrame | None = None, macro: pd.DataFrame | None = None, gold: bool = False,
          boost: bool = True, keep_test: bool = False, move_min: float = 0.1, step_min: int = None,
          horizons: list[int] | None = None) -> dict | None:
    """Modell aus 5-Minuten-Kerzen (ts, open, high, low, close, volume; die letzte darf offen sein).

    Drei Abschnitte: Lernen (erste 70 %), Auswahl (naechste 10 %: Einstellungen + Mischung linear/Baeume),
    Test (letzte 20 %: nur zum ehrlichen Messen). Danach wird mit allen Daten fuer die Live-Prognose gelernt.
    Rueckgabe: Entscheidung, Pfad, Band und Guete - oder None bei zu wenig Daten."""
    df = df.reset_index(drop=True)
    n_closed = len(df) - 1                        # offene Kerze nicht zum Lernen
    if n_closed < 300:
        return None
    STEP = step_min or STEP_MIN
    HZ = list(horizons or HORIZONS)
    feats = make_features(df, leader, funding, macro, gold, STEP)
    X_all = feats.to_numpy(float)
    keep = ~np.isnan(X_all[:-1]).all(0)           # Merkmale ohne Daten weglassen
    names = [n for n, k in zip(feats.columns, keep) if k]
    X_all = np.nan_to_num(X_all[:, keep], nan=0.0)
    lr = np.log(df["close"].astype(float).to_numpy())
    age = np.arange(len(df))[::-1].astype(float)
    warm = min(300, n_closed // 5)                 # die ersten Kerzen haben unvollstaendige Indikatoren
    H = HZ[-1]
    # Nur aus Bewegungen lernen, die nach Gebuehren zaehlen (|Bewegung| >= move_min %), und nie aus
    # Zeitraeumen, in denen der Markt still stand / geschlossen war (Gold/Silber nachts, Wochenende)
    from .hours import flat_mask
    flat_c = np.r_[0, np.cumsum(flat_mask(df))]

    def labeled(idx, h):
        ret_ = (lr[idx + h] - lr[idx]) * 100
        alive = (flat_c[idx + h + 1] - flat_c[idx]) == 0
        ok = alive & (np.abs(ret_) >= move_min)
        return idx[ok], ret_[ok], ret_[alive]

    idx30, ret30, _ = labeled(np.arange(warm, n_closed - H), H)
    if len(idx30) < 200:
        return None
    n = len(idx30)
    a_end, v_end = int(n * (1 - test_share - 0.1)), int(n * (1 - test_share))

    def cut(idx, h):          # Abschnitte mit Abstand h, damit sich Ziel-Zeitraeume nicht ueberlappen
        return idx[:a_end - h], idx[a_end:v_end - h], idx[v_end:]

    # 1) Einstellungen (Regularisierung, wie stark neuere Daten zaehlen) an der 30-min-Prognose waehlen
    y30_all = (ret30 > 0).astype(float)
    tr, va, te = cut(np.arange(n), H)
    best = None
    for l2, hl in GRID:
        wgt = 0.5 ** (age / hl)
        m = _logit_fit(X_all[idx30[tr]], y30_all[tr], wgt[idx30[tr]], l2)
        ll = _logloss(_logit_p(m, X_all[idx30[va]]), y30_all[va])
        if best is None or ll < best[0]:
            best = (ll, l2, hl)
    _, L2_, HL_ = best
    weight = 0.5 ** (age / HL_)
    # 2) Baeume dazu? Mischung an der Auswahl-Strecke waehlen (0 = nur linear, 1 = nur Baeume)
    mix = 0.0
    if boost and len(tr) >= 1000:
        lin = _logit_fit(X_all[idx30[tr]], y30_all[tr], weight[idx30[tr]], L2_)
        tb = tr[-BOOST_ROWS:]
        bst = Boost().fit(X_all[idx30[tb]], y30_all[tb], weight[idx30[tb]])
        pl, pb = _logit_p(lin, X_all[idx30[va]]), bst.proba(X_all[idx30[va]])
        mix = min((0.0, 0.5, 1.0), key=lambda a_: _logloss((1 - a_) * pl + a_ * pb, y30_all[va]))

    probs, raw, moves, quality = [], [], [], []
    test_keep: dict = {}
    for h in HZ:
        idx, ret, ret_alive = labeled(np.arange(warm, n_closed - h), h)
        if len(idx) < 200:
            return None
        y = (ret > 0).astype(float)
        nn = len(idx)
        t_end = int(nn * (1 - test_share))
        tr_i, te_i = np.arange(t_end - h), np.arange(t_end, nn)
        m = _logit_fit(X_all[idx[tr_i]], y[tr_i], weight[idx[tr_i]], L2_)
        p_te = _logit_p(m, X_all[idx[te_i]])
        final = _logit_fit(X_all[idx], y, weight[idx], L2_)
        p_now = float(_logit_p(final, X_all[n_closed - 1:n_closed])[0])     # letzte ABGESCHLOSSENE Kerze
        if h == H and mix > 0:
            tb = tr_i[-BOOST_ROWS:]
            b_te = Boost().fit(X_all[idx[tb]], y[tb], weight[idx[tb]]).proba(X_all[idx[te_i]])
            p_te = (1 - mix) * p_te + mix * b_te
            fb = np.arange(len(idx))[-BOOST_ROWS:]
            b_now = float(Boost().fit(X_all[idx[fb]], y[fb], weight[idx[fb]]).proba(X_all[n_closed - 1:n_closed])[0])
            p_now = (1 - mix) * p_now + mix * b_now
        yt = y[te_i]
        hit = float(((p_te >= 0.5) == (yt == 1)).mean())
        base = max(y[tr_i].mean(), 1 - y[tr_i].mean())   # "immer die haeufigere Richtung"
        brier = float(np.mean((p_te - yt) ** 2))
        brier0 = float(np.mean((y[tr_i].mean() - yt) ** 2))
        # nur so sicher, wie die KI an ungesehenen Daten NACHWEISLICH besser als der Zufall war:
        # Vorsprung gegen den Zufall in Standardfehlern (sich ueberlappende Zeitraeume zaehlen weniger)
        n_eff = max(1.0, len(te_i) / h)
        z = (hit - max(0.5, float(base))) / math.sqrt(0.25 / n_eff)
        trust = max(0.0, min(1.0, (z - 1.0) / 2.0))
        # Trefferquote in den 20 % sichersten Momenten (dort steigt der Autopilot ein)
        strong = np.abs(p_te - 0.5) >= np.quantile(np.abs(p_te - 0.5), 0.8)
        hit_top = float(((p_te[strong] >= 0.5) == (yt[strong] == 1)).mean()) if strong.any() else None
        p_loc, z_loc, _ = _local_cal(p_te, yt, p_now, h)
        if keep_test:
            test_keep[h] = (p_te.copy(), yt.copy())
        quality.append({"min": h * STEP, "hit": round(hit, 3), "base": round(float(base), 3),
                        "skill": round(1 - brier / brier0, 4) if brier0 > 0 else 0.0, "n_test": int(len(te_i)),
                        "trust": round(trust, 2), "z": round(z, 2),
                        "hit_top20": None if hit_top is None else round(hit_top, 3), "z_now": round(z_loc, 2)})
        # Sicherheit: je Meinungsstaerke gemessen; die globale Messung zaehlt, wenn sie mehr hergibt
        p_glob = 0.5 + (p_now - 0.5) * trust
        probs.append(p_loc if abs(p_loc - 0.5) > abs(p_glob - 0.5) else p_glob)
        raw.append(p_now)
        moves.append((float(np.mean(np.abs(ret_alive[-2000:]))), float(np.std(ret_alive[-2000:]))))
    live = live or {}
    # Entscheidung auf der Vorhersagezeit, auf der die KI gerade nachweislich am staerksten ist
    best = max(range(len(HZ)), key=lambda i: abs(probs[i] - 0.5))
    if abs(probs[best] - 0.5) < 1e-9:
        best = len(HZ) - 1
    p30, note = _calibrate(probs[best], live)
    lean = p30 if abs(p30 - 0.5) > 1e-9 else raw[best]     # ohne Vorsprung: Richtung des Rohmodells
    decision = "LONG" if lean >= 0.5 else "SHORT"
    anchor_ts = int(df["ts"].iloc[n_closed]) // 1000          # Beginn der offenen Kerze
    # Pfad = Meinung des Modells (roh) in Richtung der Entscheidung; wie sehr man ihr trauen kann,
    # sagt die (kalibrierte) Sicherheit
    nodes = [(0, 0.0, 0.0)]
    sign = 1 if decision == "LONG" else -1
    for h, p, (mabs, sd) in zip(HZ, raw, moves):
        nodes.append((h * STEP * 60, sign * abs(2 * p - 1) * mabs, sd * BAND_Z))
    q30 = quality[best]                                     # Guete auf der Vorhersagezeit der Entscheidung
    useful = q30["z"] >= 2.0 and q30["skill"] > 0          # deutlich (2 Standardfehler) besser als Zufall
    rng = (df["high"] - df["low"]).astype(float).to_numpy()[-300:-1] / df["close"].astype(float).to_numpy()[-300:-1]
    return {
        "time": anchor_ts, "decision": decision,
        "confidence": round(max(p30, 1 - p30), 3), "p_up": round(p30, 3), "p_up_raw": round(raw[best], 3),
        "decision_min": HZ[best] * STEP,
        "by_horizon": [{"min": h * STEP, "p_up": round(pp, 3)} for h, pp in zip(HZ, probs)],
        "nodes": [{"sec": s_, "ret": round(r, 4), "band": round(b_, 4)} for s_, r, b_ in nodes],
        "change_pct": round(nodes[-1][1], 3), "band_pct": round(nodes[-1][2], 3),
        "candle_range_pct": round(float(np.median(rng)) * 100, 4),
        "typical_30_pct": round(moves[-1][0], 4), "sd_5m_pct": round(moves[0][1], 4),
        "quality": quality, "hit_30m": q30["hit"], "base_30m": q30["base"], "useful": bool(useful),
        "note": note, "trained_on": int(n_closed), "features": int(X_all.shape[1]), "feature_names": names,
        **({"test": test_keep} if keep_test else {}),
        "move_min": move_min, "labeled": int(len(idx30)), "step_min": STEP, "horizons": HZ,
        "sr_up_pct": _sr_pct(feats, "m5_sr_up"), "sr_dn_pct": _sr_pct(feats, "m5_sr_dn"),
        "model": {"l2": L2_, "half_life_days": round(HL_ * STEP / 1440, 1), "trees": mix,
                  "kind": "linear" if mix == 0 else "Baeume" if mix == 1 else "linear + Baeume"},
    }


def path(fc: dict, price: float, now_s: int, seed: int) -> list[dict]:
    """Szenario fuer die Prognose-Kerzen (1-Minuten-Verlauf ab jetzt): So saehe es wahrscheinlich aus, wenn
    die KI recht hat - Richtung der Entscheidung, Groesse der typischen 30-Minuten-Bewegung dieses Markts,
    unterwegs Auf und Ab mit der echten Schwankung (Brownsche Bruecke, je 5-Minuten-Kerze gleich)."""
    start = fc["time"]
    step = fc.get("step_min") or STEP_MIN
    hz = fc.get("horizons") or HORIZONS
    end = start + hz[-1] * step * 60
    sign = 1 if fc.get("decision") == "LONG" else -1
    target = sign * float(fc.get("typical_30_pct") or abs(fc["nodes"][-1]["ret"]))      # in %
    sd_min = float(fc.get("sd_5m_pct") or 0.1) / math.sqrt(step)                           # Schwankung je Minute in %
    band_end = fc["nodes"][-1]["band"]
    rs = np.random.default_rng(seed)
    n = hz[-1] * step                                                                      # Minuten-Schritte
    steps = rs.normal(0, sd_min, n)
    walk = np.r_[0.0, np.cumsum(steps)]
    tt = np.arange(n + 1) / n
    bridge = walk - tt * walk[-1] + tt * target          # beginnt bei 0, endet genau beim Szenario-Ziel
    t0 = now_s - now_s % 60
    out = []
    for k in range(n + 1):
        t = start + k * 60
        if t < t0 and k < n:
            continue
        out.append({"time": t, "price": price * math.exp(bridge[k] / 100), "band": band_end * math.sqrt(max(tt[k], 1e-9))})
    if out:
        shift = price / out[0]["price"]                  # vom aktuellen Kurs aus starten
        for p_ in out:
            p_["price"] *= shift
        out[-1]["time"] = end
    return out


class ForecastLog:
    """Live-Entscheidungen merken und nach 30 Minuten mit dem echten Kurs vergleichen."""

    def __init__(self, path: Path | None = None, keep: int = 3000):
        self.path, self.keep = path, keep
        self.items: list[dict] = []
        if path and path.exists():
            try:
                self.items = [x for x in json.loads(path.read_text(encoding="utf-8")) if "decision" in x]
            except (OSError, ValueError):
                self.items = []

    def add(self, sym: str, fc: dict, price: float) -> None:
        if any(x["sym"] == sym and x["time"] == fc["time"] for x in self.items[-200:]):
            return                                   # eine Entscheidung je Markt und 5-Minuten-Kerze
        self.items.append({"sym": sym, "time": fc["time"], "price": price, "decision": fc["decision"],
                           "conf": fc["confidence"], "band": fc["band_pct"], "actual": None,
                           "p_raw": fc.get("p_up_raw"), "h_min": fc.get("decision_min", 30),
                           "step": fc.get("step_min", STEP_MIN)})
        self.items = self.items[-self.keep:]
        self._save()

    def resolve(self, sym: str, candles: pd.DataFrame) -> None:
        closes = dict(zip((candles["ts"] // 1000).astype(int), candles["close"].astype(float)))
        changed = False
        for x in self.items:
            if x["sym"] == sym and x["actual"] is None:
                step = x.get("step", STEP_MIN)
                t = x["time"] + int(x.get("h_min") or HORIZONS[-1] * STEP_MIN) * 60 - step * 60   # schliesst nach h_min
                if t in closes:
                    x["actual"] = closes[t]
                    changed = True
        if changed:
            self._save()

    def stats(self, sym: str | None = None, recent: int = 50) -> dict:
        done = [x for x in self.items if x["actual"] is not None and (sym is None or x["sym"] == sym)
                and x["actual"] != x["price"]]
        hits = [(x["actual"] > x["price"]) == (x["decision"] == "LONG") for x in done]
        last = hits[-recent:]
        sure = [h for h, x in zip(hits, done) if x["conf"] >= 0.6]
        return {"n": len(hits), "hit": round(sum(hits) / len(hits), 3) if hits else None,
                "n_recent": len(last), "hit_recent": round(sum(last) / len(last), 3) if last else None,
                "n_sure": len(sure), "hit_sure": round(sum(sure) / len(sure), 3) if sure else None,
                "last": [{"time": x["time"], "decision": x["decision"], "hit": h}
                         for x, h in list(zip(done, hits))[-10:]]}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items), encoding="utf-8")
        except OSError as e:
            log.debug("Prognose-Log: %s", e)


class CandleMemory:
    """Gedaechtnis: abgeschlossene 5-Minuten-Kerzen je Markt (waechst mit jeder Kerze) plus live
    aufgezeichnete Orderbuch-Werte je Kerze (Druck, Abstand zur naechsten Kauf-/Verkaufs-Wand)."""

    def __init__(self, folder: Path | None, tf: str = "5m", step_min: int = STEP_MIN, max_rows: int = None):
        self.folder, self.tf, self.step, self.max_rows = folder, tf, step_min, max_rows or MAX_HISTORY
        self.mem: dict[str, pd.DataFrame] = {}
        self.book_acc: dict[str, dict[int, list[float]]] = {}
        self.meta = {}
        if folder is not None and (folder / "lernen.json").exists():
            try:
                self.meta = json.loads((folder / "lernen.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.meta = {}

    def _file(self, sym: str) -> Path | None:
        return self.folder / f"kerzen_{sym.split(':')[0].replace('/', '')}_{self.tf}.csv" if self.folder else None

    def save_meta(self) -> None:
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "lernen.json").write_text(json.dumps(self.meta), encoding="utf-8")
        except OSError as e:
            log.debug("KI-Meta: %s", e)

    def record_book(self, sym: str, bar_ts: int, imb: float | None, wall_bid: float | None, wall_ask: float | None):
        """Orderbuch-Werte waehrend einer Kerze sammeln (Mittelwert wird beim Kerzenschluss gespeichert)."""
        acc = self.book_acc.setdefault(sym, {})
        x = acc.setdefault(int(bar_ts), [0.0, 0, 0.0, 0, 0.0, 0])
        for i, v in ((0, imb), (2, wall_bid), (4, wall_ask)):
            if v is not None and v == v:
                x[i] += float(v)
                x[i + 1] += 1
        for t in [t for t in acc if t < bar_ts - 3_600_000]:
            del acc[t]

    def merge(self, sym: str, fresh: pd.DataFrame) -> pd.DataFrame:
        """Gespeicherte + frische Kerzen; die letzte (offene) Kerze bleibt am Ende, wird aber nicht gespeichert."""
        old = self.mem.get(sym)
        f = self._file(sym)
        if old is None and f is not None and f.exists():
            try:
                old = pd.read_csv(f)
            except (OSError, ValueError):
                old = None
        closed = fresh.iloc[:-1].copy()
        acc = self.book_acc.get(sym, {})
        for col, i in (("book_imb", 0), ("wall_bid", 2), ("wall_ask", 4)):
            closed[col] = [acc[t][i] / acc[t][i + 1] if t in acc and acc[t][i + 1] else np.nan
                           for t in closed["ts"].astype("int64")]
        if old is not None and len(old):
            # neue Kurse gewinnen, aufgezeichnete Orderbuch-Werte der alten Zeilen bleiben erhalten
            allc = closed.set_index("ts").combine_first(old.set_index("ts")).reset_index()
        else:
            allc = closed
        allc = allc.sort_values("ts").tail(self.max_rows).reset_index(drop=True)
        # nur lueckenlose Daten verwenden (nach laengerer Pause, > 2 h, beginnt das Gedaechtnis dort neu)
        gaps = np.where(np.diff(allc["ts"].to_numpy()) > 2 * 3_600_000)[0]
        if len(gaps):
            allc = allc.iloc[gaps[-1] + 1:].reset_index(drop=True)
        if len(allc) < len(closed):
            allc = closed.reset_index(drop=True)
        self.mem[sym] = allc
        m = self.meta.setdefault(sym, {})
        m.setdefault("since", int(time.time()))
        if f is not None and (old is None or len(allc) != len(old) or not m.get("saved")):
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                allc.to_csv(f, index=False)
                m["saved"] = 1
            except OSError as e:
                log.debug("Kerzen-Gedaechtnis %s: %s", sym, e)
        out = pd.concat([allc, fresh.iloc[-1:]]).reset_index(drop=True)
        return out

    def backfill(self, sym: str, ohlcv_fn, days: int = BACKFILL_DAYS, ref_ms: int | None = None) -> int:
        """Vergangenheit von Bitget nachladen (einmalig), damit die KI nicht bei null anfaengt.
        ref_ms: Zeit der neuesten Bitget-Kerze (unabhaengig von der PC-Uhr)."""
        have = self.mem.get(sym)
        ref = ref_ms or int(time.time() * 1000)
        start = ref - days * 86_400_000
        if have is not None and len(have) and int(have["ts"].iloc[0]) <= start + 86_400_000:
            return 0
        rows, since = [], start
        end = int(have["ts"].iloc[0]) if have is not None and len(have) else ref
        while since < end:
            batch = ohlcv_fn(sym, self.tf, since, 200)
            if not batch:
                since += 200 * self.step * 60_000
                continue
            rows += batch
            nxt = batch[-1][0] + self.step * 60_000
            if nxt <= since:
                break
            since = nxt
        if not rows:
            return 0
        hist = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        hist = hist.drop_duplicates("ts").sort_values("ts")
        base = self.mem.get(sym)
        allc = hist.set_index("ts").combine_first(base.set_index("ts")).reset_index() if base is not None else hist
        allc = allc.sort_values("ts").tail(self.max_rows).reset_index(drop=True)
        self.mem[sym] = allc
        f = self._file(sym)
        if f is not None:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                allc.to_csv(f, index=False)
            except OSError as e:
                log.debug("Kerzen-Gedaechtnis %s: %s", sym, e)
        return len(hist)


class Forecaster:
    """KI je Markt: lernt im Hintergrund bei jeder neuen 5-Minuten-Kerze neu (die Oberflaeche wartet nie)."""

    def __init__(self, candles_fn, folder: Path | None = None, mode: str = "paper", leader: str | None = None,
                 price_fn=None, ohlcv_fn=None, book_fn=None, funding_fn=None, macro_fn=None,
                 leaders: list[str] | None = None, background: bool = True, tf: str = "5m",
                 step_min: int = STEP_MIN, horizons: list[int] | None = None, build_opts: dict | None = None,
                 backfill_days: int = BACKFILL_DAYS, max_rows: int | None = None, name: str = "ki"):
        self.candles_fn, self.price_fn, self.leader = candles_fn, price_fn, leader
        self.tf, self.step, self.horizons = tf, step_min, horizons or HORIZONS
        self.build_opts, self.backfill_days, self.max_rows = build_opts or {}, backfill_days, max_rows
        self.ohlcv_fn, self.book_fn, self.funding_fn, self.macro_fn = ohlcv_fn, book_fn, funding_fn, macro_fn
        self.leaders = [x for x in (leaders or ([leader] if leader else [])) if x]
        self.background = background
        self.memory = CandleMemory(folder / name if folder else None, tf, step_min, max_rows)
        self.log = ForecastLog(folder / f"{name}_entscheidungen_{mode}.json" if folder else None)
        self.models: dict[str, tuple[int, dict]] = {}
        self.fresh: dict[str, tuple[float, pd.DataFrame]] = {}
        self.busy: set[str] = set()
        self.errors: dict[str, str] = {}
        self.filled: set[str] = set()
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[float, object]] = {}

    def _cached(self, key: str, ttl: float, fn):
        hit = self.cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        try:
            val = fn()
        except Exception as e:  # noqa: BLE001 - Zusatzdaten sind optional
            log.debug("KI %s: %s", key, e)
            val = hit[1] if hit else None
        self.cache[key] = (time.time(), val)
        return val

    def _record_book(self, sym: str, bar_ts: int) -> None:
        if not self.book_fn:
            return
        book = self._cached(f"book|{sym}", 10, lambda: self.book_fn(sym))
        if not book or not book.get("bids") or not book.get("asks"):
            return
        from .flow import book_imbalance, book_walls
        mid = (float(book["bids"][0][0]) + float(book["asks"][0][0])) / 2
        w = book_walls(book)
        wb = min((abs(mid - x["price"]) / mid * 100 for x in w.get("bids") or []), default=None)
        wa = min((abs(x["price"] - mid) / mid * 100 for x in w.get("asks") or []), default=None)
        self.memory.record_book(sym, bar_ts, book_imbalance(book), wb, wa)

    def _candles(self, sym: str) -> pd.DataFrame:
        hit = self.fresh.get(sym)
        if hit and time.time() - hit[0] < 10:
            return hit[1]
        raw = self.candles_fn(sym, self.tf, 1000).reset_index(drop=True)
        self._record_book(sym, int(raw["ts"].iloc[-1]))
        if self.ohlcv_fn and sym not in self.filled:          # einmalig Vergangenheit nachladen
            self.filled.add(sym)
            self._spawn(self._backfill, sym, int(raw["ts"].iloc[0]))
        df = self.memory.merge(sym, raw)
        self.fresh[sym] = (time.time(), df)
        return df

    def _spawn(self, fn, *args):
        if self.background:
            threading.Thread(target=fn, args=args, daemon=True).start()
        else:
            fn(*args)

    def _backfill(self, sym: str, ref_ms: int | None = None) -> None:
        try:
            n = self.memory.backfill(sym, self.ohlcv_fn, self.backfill_days, ref_ms=ref_ms)
            if n:
                log.info("KI %s %s: %d Kerzen Vergangenheit nachgeladen (%d Tage)", sym, self.tf, n, self.backfill_days)
                self.fresh.pop(sym, None)
                self.models.pop(sym, None)                    # mit mehr Daten neu lernen
        except Exception as e:  # noqa: BLE001
            log.warning("KI %s: Vergangenheit nicht ladbar (%s)", sym, e)

    def _train(self, sym: str, df: pd.DataFrame, bar: int) -> None:
        try:
            lead = {}
            if "XAU" not in sym and "XAG" not in sym:
                for ls in self.leaders:
                    if ls != sym:
                        try:
                            lead[ls.split("/")[0].lower()] = self._candles(ls)
                        except Exception as e:  # noqa: BLE001 - ohne Leitwaehrung weiter
                            log.debug("KI: %s fehlt (%s)", ls, e)
            funding = self._cached(f"funding|{sym}", 3600, lambda: self.funding_fn(sym)) if self.funding_fn else None
            macro = self._cached("macro", 6 * 3600, self.macro_fn) if self.macro_fn else None
            t0 = time.time()
            fc = build(df, lead, live=self.log.stats(sym), funding=funding, macro=macro,
                       gold="XAU" in sym or "XAG" in sym, step_min=self.step, horizons=self.horizons,
                       **self.build_opts)
            if fc:
                fc["learn_s"] = round(time.time() - t0, 1)
                self.log.add(sym, fc, float(df["close"].iloc[-1]))
                m = self.memory.meta.setdefault(sym, {"since": int(time.time())})
                m["retrains"] = m.get("retrains", 0) + 1
                day = time.strftime("%Y-%m-%d")
                hist = m.setdefault("test_hit", {})
                hist[day] = fc["hit_30m"]                       # Entwicklung der Test-Trefferquote je Tag
                m["test_hit"] = dict(list(hist.items())[-60:])
                self.memory.save_meta()
            self.models[sym] = (bar, fc)
            self.errors.pop(sym, None)
        except Exception as e:  # noqa: BLE001
            log.warning("KI %s lernen: %s", sym, e, exc_info=True)
            self.errors[sym] = f"{type(e).__name__}: {e}"
            self.models.setdefault(sym, (bar, None))
        finally:
            self.busy.discard(sym)

    def info(self, sym: str, df: pd.DataFrame) -> dict:
        """Wie lange und womit die KI schon lernt."""
        m = self.memory.meta.get(sym, {})
        stats = self.log.stats(sym)
        days = (int(df["ts"].iloc[-1]) - int(df["ts"].iloc[0])) / 86_400_000 if len(df) > 1 else 0
        book_bars = int(df["book_imb"].notna().sum()) if "book_imb" in df else 0
        weekly = {}
        for x in self.log.items:
            if x["sym"] == sym and x["actual"] is not None and x["actual"] != x["price"]:
                wk = time.strftime("%d.%m.", time.localtime(x["time"] - (x["time"] % (7 * 86400))))
                ok = (x["actual"] > x["price"]) == (x["decision"] == "LONG")
                w = weekly.setdefault(wk, [0, 0])
                w[0] += ok
                w[1] += 1
        return {"since": m.get("since"), "retrains": m.get("retrains", 0), "bars": int(len(df) - 1),
                "data_days": round(days, 1), "book_bars": book_bars, "book_needed": MIN_BOOK_BARS,
                "checked": stats["n"], "weeks": [{"week": k, "hit": round(v[0] / v[1], 3), "n": v[1]}
                                                 for k, v in list(weekly.items())[-8:]]}

    def get(self, sym: str) -> dict:
        df = self._candles(sym)
        bar = int(df["ts"].iloc[-1]) // 1000
        self.log.resolve(sym, df.iloc[:-1])
        cached = self.models.get(sym)
        with self.lock:
            stale = not cached or cached[0] != bar
            if stale and sym not in self.busy:              # neue 5-Minuten-Kerze -> im Hintergrund neu lernen
                self.busy.add(sym)
                self._spawn(self._train, sym, df, bar)
        cached = self.models.get(sym)
        info = self.info(sym, df)
        fc = cached[1] if cached else None
        if fc is None:
            msg = ("KI lernt gerade ..." if sym in self.busy else
                   f"Fehler beim Lernen ({self.errors[sym]}) - bitte melden" if sym in self.errors else
                   "Zu wenig Kursdaten - die KI sammelt noch")
            return {"symbol": sym, "ok": False, "learning": info, "msg": msg}
        price = float(df["close"].iloc[-1])
        if self.price_fn:
            try:
                price = float(self.price_fn(sym) or price)
            except Exception:  # noqa: BLE001
                pass
        now = int(time.time())
        seed = zlib.crc32(f"{sym}|{fc['time']}".encode())
        start = min(max(now, fc["time"]), fc["time"] + self.step * 60 - 1)
        out = {k: v for k, v in fc.items() if k != "feature_names"}
        return {"symbol": sym, "ok": True, **out, "price": price, "now": now,
                # Start innerhalb der aktuellen 5-Minuten-Kerze (robust gegen Uhr-Abweichungen)
                "path": path(fc, price, start, seed), "live": self.log.stats(sym), "learning": info,
                "relearning": sym in self.busy, "valid_until": fc["time"] + self.step * 60}


def ki_active(fc: dict | None, s: dict) -> tuple[bool, str]:
    """Darf der Bot die KI-Entscheidung gerade benutzen? auto = erst wenn sie sich live bewiesen hat."""
    mode = s.get("ki_filter", "auto")
    if mode == "off" or not fc or not fc.get("ok"):
        return False, "KI-Filter aus" if mode == "off" else "KI-Prognose nicht verfuegbar"
    if mode == "on":
        return True, "KI-Filter an"
    live = fc.get("live") or {}
    need_n, need_hit = s.get("ki_min_live", 100), s.get("ki_min_hit", 0.53)
    if live.get("n", 0) < need_n:
        return False, f"KI sammelt noch Erfahrung ({live.get('n', 0)}/{need_n} gepruefte Entscheidungen)"
    if (live.get("hit") or 0) < need_hit or not fc.get("useful"):
        return False, f"KI noch nicht gut genug (live {round((live.get('hit') or 0) * 100)} %, noetig {round(need_hit * 100)} %)"
    return True, f"KI bewaehrt (live {round(live['hit'] * 100)} % von {live['n']})"


def ki_blocks(side: str, fc: dict | None, s: dict) -> str:
    """Grund, warum die KI einen Einstieg verhindert ('' = erlaubt)."""
    active, _ = ki_active(fc, s)
    if not active:
        return ""
    want = "LONG" if side == "long" else "SHORT"
    if fc["decision"] != want and fc["confidence"] >= s.get("ki_min_conf", 0.55):
        return f"KI dagegen: {fc['decision']} mit {round(fc['confidence'] * 100)} % Sicherheit (naechste 30 min)"
    return ""


def report(folder: Path, mode: str, symbols: list[str] | None = None, name: str = "ki", step_min: int = STEP_MIN,
           horizons: list[int] | None = None, build_opts: dict | None = None, title: str = "KI-BERICHT") -> str:
    """KI-Bericht aus den Daten auf diesem PC (python run.py ki-bericht)."""
    lines = []
    w = lines.append
    lg = ForecastLog(folder / f"{name}_entscheidungen_{mode}.json")
    meta = {}
    try:
        meta = json.loads((folder / name / "lernen.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    syms = symbols or sorted({x["sym"] for x in lg.items} | set(meta))
    pct = lambda v: "-" if v is None else f"{v * 100:.1f} %"
    w(f"===== {title} ({mode}) =====")
    done_all = [x for x in lg.items if x["actual"] is not None and x["actual"] != x["price"]]
    w(f"Gepruefte Live-Entscheidungen insgesamt: {len(done_all)}")
    for sym in syms:
        w("")
        w(f"--- {sym} ---")
        m = meta.get(sym, {})
        if m.get("since"):
            w(f"lernt seit {time.strftime('%d.%m. %H:%M', time.localtime(m['since']))}, {m.get('retrains', 0)}x neu gelernt")
        th = m.get("test_hit") or {}
        if th:
            w("Test-Trefferquote je Tag: " + ", ".join(f"{k[5:]}: {pct(v)}" for k, v in list(th.items())[-10:]))
        done = [x for x in done_all if x["sym"] == sym]
        if done:
            ok = [(x["actual"] > x["price"]) == (x["decision"] == "LONG") for x in done]
            w(f"Live: {len(done)} Entscheidungen geprueft, {pct(sum(ok) / len(ok))} richtig")
            for lo_, hi_, name in ((0.0, 0.52, "unter 52 %"), (0.52, 0.54, "52-54 %"), (0.54, 0.56, "54-56 %"),
                                   (0.56, 0.60, "56-60 %"), (0.60, 1.01, "ueber 60 %")):
                sel = [o for o, x in zip(ok, done) if lo_ <= x["conf"] < hi_]
                if sel:
                    w(f"   Sicherheit {name:<11} {len(sel):>5}x  davon richtig {pct(sum(sel) / len(sel))}")
            raw = [(o, x) for o, x in zip(ok, done) if x.get("p_raw") is not None]
            if len(raw) >= 50:
                strong = sorted(raw, key=lambda t: -abs(t[1]["p_raw"] - 0.5))[:max(10, len(raw) // 5)]
                okr = [(t[1]["actual"] > t[1]["price"]) == (t[1]["p_raw"] >= 0.5) for t in strong]
                w(f"   Rohmeinung, staerkste 20 %: {len(strong)}x, Richtung richtig {pct(sum(okr) / len(okr))}")
        tfn = "1m" if step_min == 1 else f"{step_min}m"
        f = folder / name / f"kerzen_{sym.split(':')[0].replace('/', '')}_{tfn}.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
            lead = {}
            for ls in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
                lf = folder / name / f"kerzen_{ls.split(':')[0].replace('/', '')}_{tfn}.csv"
                if ls != sym and lf.exists() and "XAU" not in sym and "XAG" not in sym:
                    lead[ls.split("/")[0].lower()] = pd.read_csv(lf)
            df = pd.concat([df, df.iloc[-1:]]).reset_index(drop=True)      # letzte Zeile als "offene" Kerze
            fc = build(df, lead, keep_test=True, step_min=step_min, horizons=horizons, **(build_opts or {}))
        except Exception as e:  # noqa: BLE001
            w(f"Frischer Test nicht moeglich: {e}")
            continue
        if not fc:
            w("Zu wenig gespeicherte Kerzen fuer einen Test")
            continue
        w(f"Frischer Test mit {fc['trained_on']} Kerzen ({fc['trained_on'] * step_min / 1440:.0f} Tage), "
          f"{fc['features']} Merkmale, Modell {fc['model']['kind']}:")
        for q in fc["quality"]:
            w(f"   {q['min']:>2} min: Treffer {pct(q['hit'])} (Zufall {pct(q['base'])}), "
              f"sicherste 20 %: {pct(q.get('hit_top20'))}, Vorsprung z={q['z']:+.1f}")
        best = max(fc["test"].items(), key=lambda kv: (kv[1][0] - 0.5).std())
        w("Autopilot haette (ungesehene Daten, beste Vorhersagezeit) eingestiegen:")
        for h, (p_te, yt) in sorted(fc["test"].items()):
            sel = selective(p_te, yt, h, step_min=step_min)
            if not sel:
                continue
            w(f"   {h * step_min:>2} min: " + " | ".join(
                f"ab {round(r['min_conf'] * 100)} %: {r['per_day']:.1f}/Tag, richtig {pct(r['hit'])}" for r in sel[1:]))
        _ = best
        w(f"Jetzt: {fc['decision']} mit {pct(fc['confidence'])} (auf {fc['decision_min']} min)")
    w("")
    w("Faustregel: Eine Schwelle lohnt sich, wenn 'richtig' dort klar ueber 55 % liegt und genug Trades/Tag kommen.")
    return "\n".join(lines)
