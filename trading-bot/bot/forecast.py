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
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import atr, bollinger, ema, macd_hist, rsi

log = logging.getLogger(__name__)

STEP_MIN = 5
HORIZONS = [1, 2, 3, 4, 5, 6]            # in 5-Minuten-Schritten -> 5 ... 30 min
MAX_HISTORY = 20_000                     # gespeicherte 5-Minuten-Kerzen je Markt (~70 Tage)
HALF_LIFE = 3_000                        # Gewicht neuerer Kerzen: halbiert sich alle ~10 Tage
L2 = 2.0
BAND_Z = 1.28                            # 80-%-Band


def make_features(df: pd.DataFrame, leader: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merkmale je Kerze (nur Vergangenheit bis einschliesslich dieser Kerze)."""
    c, h, lo, o = (df[k].astype(float) for k in ("close", "high", "low", "open"))
    a = atr(df, 14).replace(0, np.nan)
    lr = np.log(c)
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24, 48, 96, 288):          # 5 min ... 1 Tag
        f[f"r{n}"] = (lr - lr.shift(n)) * 100
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
    if leader is not None and len(leader):
        lc = pd.Series(leader["close"].astype(float).to_numpy(), index=leader["ts"].to_numpy())
        lc = lc[~lc.index.duplicated()].reindex(df["ts"].to_numpy()).ffill()
        llr = np.log(lc.to_numpy())
        for n in (1, 3, 12, 48):
            f[f"btc_r{n}"] = (llr - np.r_[np.full(n, np.nan), llr[:-n]]) * 100
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


def build(df: pd.DataFrame, leader: pd.DataFrame | None = None, test_share: float = 0.2,
          live: dict | None = None) -> dict | None:
    """Modell aus 5-Minuten-Kerzen (ts, open, high, low, close, volume; die letzte darf offen sein).
    Rueckgabe: Entscheidung, Pfad, Band und Guete - oder None bei zu wenig Daten."""
    df = df.reset_index(drop=True)
    n_closed = len(df) - 1                        # offene Kerze nicht zum Lernen
    if n_closed < 300:
        return None
    X_all = make_features(df, leader).to_numpy(float)
    keep = ~np.isnan(X_all[:-1]).all(0)           # Merkmale ohne Daten weglassen
    X_all = np.nan_to_num(X_all[:, keep], nan=0.0)
    lr = np.log(df["close"].astype(float).to_numpy())
    age = np.arange(len(df))[::-1].astype(float)
    weight = 0.5 ** (age / HALF_LIFE)
    warm = 60                                      # die ersten Kerzen haben unvollstaendige Indikatoren
    probs, raw, moves, quality = [], [], [], []
    for h in HORIZONS:
        idx = np.arange(warm, n_closed - h)
        if len(idx) < 200:
            return None
        ret = (lr[idx + h] - lr[idx]) * 100
        y = (ret > 0).astype(float)
        split = int(len(idx) * (1 - test_share))
        tr, te = idx[:split], idx[split:]
        m = _logit_fit(X_all[tr], y[:split], weight[tr])
        p_te = _logit_p(m, X_all[te])
        hit = float(((p_te >= 0.5) == (y[split:] == 1)).mean())
        base = max(y[:split].mean(), 1 - y[:split].mean())   # "immer die haeufigere Richtung"
        brier = float(np.mean((p_te - y[split:]) ** 2))
        brier0 = float(np.mean((y[:split].mean() - y[split:]) ** 2))
        quality.append({"min": h * STEP_MIN, "hit": round(hit, 3), "base": round(float(base), 3),
                        "skill": round(1 - brier / brier0, 4) if brier0 > 0 else 0.0, "n_test": int(len(te))})
        final = _logit_fit(X_all[idx], y, weight[idx])
        p_now = float(_logit_p(final, X_all[n_closed - 1:n_closed])[0])          # letzte ABGESCHLOSSENE Kerze
        # nur so sicher, wie die KI an ungesehenen Daten wirklich besser als der Zufall war
        trust = max(0.0, min(1.0, (hit - max(0.5, float(base))) / 0.05))
        quality[-1]["trust"] = round(trust, 2)
        probs.append(0.5 + (p_now - 0.5) * trust)
        raw.append(p_now)
        recent = np.abs(ret[-2000:])
        moves.append((float(np.mean(recent)), float(np.std(ret[-2000:]))))
    live = live or {}
    p30, note = _calibrate(probs[-1], live)
    lean = p30 if abs(p30 - 0.5) > 1e-9 else raw[-1]       # ohne Vorsprung: Richtung des Rohmodells
    decision = "LONG" if lean >= 0.5 else "SHORT"
    # Pfad: erwartete Bewegung je Abstand = (2p-1) x uebliche Bewegung; Band = uebliche Schwankung
    anchor_ts = int(df["ts"].iloc[n_closed]) // 1000          # Beginn der offenen Kerze
    nodes = [(0, 0.0, 0.0)]
    # Pfad = Meinung des Modells (roh) in Richtung der Entscheidung; wie sehr man ihr trauen kann,
    # sagt die (kalibrierte) Sicherheit
    sign = 1 if decision == "LONG" else -1
    for h, p, (mabs, sd) in zip(HORIZONS, raw, moves):
        nodes.append((h * STEP_MIN * 60, sign * abs(2 * p - 1) * mabs, sd * BAND_Z))
    q30 = quality[-1]
    useful = q30["hit"] > max(0.52, q30["base"] + 0.01) and q30["skill"] > 0
    rng = (df["high"] - df["low"]).astype(float).to_numpy()[-300:-1] / df["close"].astype(float).to_numpy()[-300:-1]
    return {
        "time": anchor_ts, "decision": decision,
        "confidence": round(max(p30, 1 - p30), 3), "p_up": round(p30, 3), "p_up_raw": round(raw[-1], 3),
        "nodes": [{"sec": s, "ret": round(r, 4), "band": round(b, 4)} for s, r, b in nodes],
        "change_pct": round(nodes[-1][1], 3), "band_pct": round(nodes[-1][2], 3),
        "candle_range_pct": round(float(np.median(rng)) * 100, 4),
        "quality": quality, "hit_30m": q30["hit"], "base_30m": q30["base"], "useful": bool(useful),
        "note": note, "trained_on": int(n_closed), "features": int(X_all.shape[1]),
    }


def path(fc: dict, price: float, now_s: int, seed: int) -> list[dict]:
    """Prognose als 1-Minuten-Verlauf ab jetzt (fuer die Prognose-Kerzen jeder Zeiteinheit).
    Zwischen den 5-Minuten-Punkten etwas uebliche Schwankung, damit es wie ein Chart aussieht -
    die Punkte selbst (Richtung, Ziel) kommen aus dem Modell."""
    nodes = fc["nodes"]
    start = fc["time"]
    end = start + nodes[-1]["sec"]
    rs = np.random.default_rng(seed)
    noise_sd = fc.get("candle_range_pct", 0.1) / 100 / 3
    out = []
    times = list(range(now_s - now_s % 60, end, 60)) + [end]
    for t in times:
        x = t - start
        k = max(0, min(len(nodes) - 2, int(x // (STEP_MIN * 60))))
        a, b = nodes[k], nodes[k + 1]
        frac = 0.0 if b["sec"] == a["sec"] else min(1.0, max(0.0, (x - a["sec"]) / (b["sec"] - a["sec"])))
        r = a["ret"] + (b["ret"] - a["ret"]) * frac
        wiggle = rs.normal(0, noise_sd) * math.sin(math.pi * frac) if 0 < frac < 1 else 0.0
        out.append({"time": t, "price": price * math.exp(r / 100 + wiggle),
                    "band": b["band"] * frac + a["band"] * (1 - frac)})
    # vom aktuellen Kurs aus starten (die Prognose wurde zu Beginn der Kerze berechnet)
    if out:
        shift = price / out[0]["price"]
        for p in out:
            p["price"] *= shift
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
                           "conf": fc["confidence"], "band": fc["band_pct"], "actual": None})
        self.items = self.items[-self.keep:]
        self._save()

    def resolve(self, sym: str, candles: pd.DataFrame) -> None:
        closes = dict(zip((candles["ts"] // 1000).astype(int), candles["close"].astype(float)))
        changed = False
        for x in self.items:
            if x["sym"] == sym and x["actual"] is None:
                t = x["time"] + (HORIZONS[-1] - 1) * STEP_MIN * 60     # Kerze, die 30 min spaeter schliesst
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
    """Gedaechtnis: abgeschlossene 5-Minuten-Kerzen je Markt sammeln (waechst mit jeder Kerze)."""

    def __init__(self, folder: Path | None):
        self.folder = folder
        self.mem: dict[str, pd.DataFrame] = {}

    def _file(self, sym: str) -> Path | None:
        return self.folder / f"kerzen_{sym.split(':')[0].replace('/', '')}_5m.csv" if self.folder else None

    def merge(self, sym: str, fresh: pd.DataFrame) -> pd.DataFrame:
        """Gespeicherte + frische Kerzen; die letzte (offene) Kerze bleibt am Ende, wird aber nicht gespeichert."""
        old = self.mem.get(sym)
        f = self._file(sym)
        if old is None and f is not None and f.exists():
            try:
                old = pd.read_csv(f)
            except (OSError, ValueError):
                old = None
        closed = fresh.iloc[:-1]
        allc = pd.concat([old, closed]) if old is not None else closed
        allc = allc.drop_duplicates("ts", keep="last").sort_values("ts").tail(MAX_HISTORY).reset_index(drop=True)
        # nur lueckenlose Daten verwenden (nach laengerer Pause beginnt das Gedaechtnis neu)
        gaps = np.where(np.diff(allc["ts"].to_numpy()) > STEP_MIN * 60_000 * 1.5)[0]
        if len(gaps):
            allc = allc.iloc[gaps[-1] + 1:].reset_index(drop=True)
        if len(allc) < len(closed):
            allc = closed.reset_index(drop=True)
        self.mem[sym] = allc
        if f is not None and (old is None or len(allc) != len(old)):
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                allc.to_csv(f, index=False)
            except OSError as e:
                log.debug("Kerzen-Gedaechtnis %s: %s", sym, e)
        return pd.concat([allc, fresh.iloc[-1:]]).reset_index(drop=True)


class Forecaster:
    """Entscheidung je Markt, neu gelernt bei jeder neuen 5-Minuten-Kerze."""

    def __init__(self, candles_fn, folder: Path | None = None, mode: str = "paper", leader: str | None = None,
                 price_fn=None):
        self.candles_fn, self.price_fn, self.leader = candles_fn, price_fn, leader
        self.memory = CandleMemory(folder / "ki" if folder else None)
        self.log = ForecastLog(folder / f"ki_entscheidungen_{mode}.json" if folder else None)
        self.models: dict[str, tuple[int, dict]] = {}
        self.fresh: dict[str, tuple[float, pd.DataFrame]] = {}

    def _candles(self, sym: str) -> pd.DataFrame:
        hit = self.fresh.get(sym)
        if hit and time.time() - hit[0] < 15:
            return hit[1]
        df = self.memory.merge(sym, self.candles_fn(sym, "5m", 1000).reset_index(drop=True))
        self.fresh[sym] = (time.time(), df)
        return df

    def get(self, sym: str) -> dict:
        df = self._candles(sym)
        bar = int(df["ts"].iloc[-1]) // 1000
        self.log.resolve(sym, df.iloc[:-1])
        cached = self.models.get(sym)
        if not cached or cached[0] != bar:                 # neue 5-Minuten-Kerze -> neu lernen
            lead = None
            if self.leader and sym != self.leader and "XAU" not in sym and "XAG" not in sym:
                try:
                    lead = self._candles(self.leader)
                except Exception as e:  # noqa: BLE001 - ohne BTC weiter
                    log.debug("KI: BTC-Daten fehlen (%s)", e)
            fc = build(df, lead, live=self.log.stats(sym))
            cached = (bar, fc)
            self.models[sym] = cached
            if fc:
                self.log.add(sym, fc, float(df["close"].iloc[-1]))
        fc = cached[1]
        if fc is None:
            return {"symbol": sym, "ok": False, "msg": "Zu wenig Kursdaten - die KI sammelt noch"}
        price = float(df["close"].iloc[-1])
        if self.price_fn:
            try:
                price = float(self.price_fn(sym) or price)
            except Exception:  # noqa: BLE001
                pass
        now = int(time.time())
        seed = zlib.crc32(f"{sym}|{fc['time']}".encode())
        return {"symbol": sym, "ok": True, **fc, "price": price, "now": now,
                # Start innerhalb der aktuellen 5-Minuten-Kerze (robust gegen Uhr-Abweichungen)
                "path": path(fc, price, min(max(now, fc["time"]), fc["time"] + STEP_MIN * 60 - 1), seed),
                "live": self.log.stats(sym),
                "valid_until": fc["time"] + STEP_MIN * 60}


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
