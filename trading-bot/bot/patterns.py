"""Muster-Erkennung: Kerzen-Muster und Chart-Muster - fuer KI, Bot und Chart.

Alles ist kausal: Ein Muster gilt erst ab der Kerze, an der es abgeschlossen bzw. bestaetigt ist
(Wendepunkte z. B. erst `PIVOT` Kerzen spaeter). So gibt es im Backtest keinen Blick in die Zukunft.

Kerzen-Muster (je Kerze): Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star,
Drei weisse Soldaten / drei schwarze Kraehen, Inside Bar.
Chart-Muster: Trend-Struktur (hoehere Hochs/Tiefs), Doppel-Hoch/-Tief mit Bruch der Nackenlinie,
Ausbruch aus der Spanne mit Volumen, Dreieck (zusammenlaufend), Abstand zu Unterstuetzung/Widerstand.
"""
import numpy as np
import pandas as pd

PIVOT = 3            # Wendepunkt = hoechstes/tiefstes Hoch/Tief von 3 Kerzen links und rechts
RANGE = 20           # Spanne fuer Ausbrueche

CANDLE_NAMES = {
    "hammer": ("Hammer", 1), "shooting": ("Shooting Star", -1),
    "engulf_up": ("Bullish Engulfing", 1), "engulf_dn": ("Bearish Engulfing", -1),
    "morning": ("Morning Star", 1), "evening": ("Evening Star", -1),
    "soldiers": ("3 weisse Soldaten", 1), "crows": ("3 schwarze Kraehen", -1),
    "doji": ("Doji", 0), "inside": ("Inside Bar", 0),
}
CHART_NAMES = {
    "dbl_bottom": ("Doppel-Tief (Nackenlinie gebrochen)", 1), "dbl_top": ("Doppel-Hoch (Nackenlinie gebrochen)", -1),
    "break_up": ("Ausbruch nach oben", 1), "break_dn": ("Ausbruch nach unten", -1),
    "hh_hl": ("Aufwaertsstruktur (hoehere Hochs/Tiefs)", 1), "lh_ll": ("Abwaertsstruktur (tiefere Hochs/Tiefs)", -1),
    "triangle": ("Dreieck (Kurs laeuft zusammen)", 0),
}


def _atr(h, lo, c, n=14):
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - lo, np.maximum(abs(h - prev), abs(lo - prev)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def candles(o, h, lo, c) -> dict[str, np.ndarray]:
    """Kerzen-Muster je Kerze (bool), bewertet am Schluss der jeweiligen Kerze."""
    body = np.abs(c - o)
    rng = np.maximum(h - lo, 1e-12)
    up_w = h - np.maximum(c, o)
    dn_w = np.minimum(c, o) - lo
    green, red = c > o, c < o
    n = len(c)

    def p(x, k=1):                                                 # Wert k Kerzen vorher
        return np.r_[np.full(min(k, n), np.nan), x[:max(0, n - k)]]
    avg_body = pd.Series(body).rolling(20, min_periods=5).mean().to_numpy()
    fell = p(c) < p(c, 5)                                          # vorher gefallen / gestiegen
    rose = p(c) > p(c, 5)
    out = {
        "doji": body <= 0.1 * rng,
        "hammer": (dn_w >= 2 * body) & (up_w <= np.maximum(body, 0.1 * rng)) & (body > 0.05 * rng) & fell,
        "shooting": (up_w >= 2 * body) & (dn_w <= np.maximum(body, 0.1 * rng)) & (body > 0.05 * rng) & rose,
        "engulf_up": green & (p(c) < p(o)) & (c >= p(o)) & (o <= p(c)) & (body > p(body)) & (body >= avg_body) & fell,
        "engulf_dn": red & (p(c) > p(o)) & (c <= p(o)) & (o >= p(c)) & (body > p(body)) & (body >= avg_body) & rose,
        "inside": (h < p(h)) & (lo > p(lo)),
    }
    small_mid = p(body) <= 0.4 * p(avg_body)
    out["morning"] = (p(c, 2) < p(o, 2)) & (p(body, 2) >= p(avg_body, 2)) & small_mid & green \
        & (c > (p(o, 2) + p(c, 2)) / 2)
    out["evening"] = (p(c, 2) > p(o, 2)) & (p(body, 2) >= p(avg_body, 2)) & small_mid & red \
        & (c < (p(o, 2) + p(c, 2)) / 2)
    g3 = green & (p(c) > p(o)) & (p(c, 2) > p(o, 2)) & (c > p(c)) & (p(c) > p(c, 2))
    r3 = red & (p(c) < p(o)) & (p(c, 2) < p(o, 2)) & (c < p(c)) & (p(c) < p(c, 2))
    big = (body >= 0.6 * avg_body) & (p(body) >= 0.6 * p(avg_body)) & (p(body, 2) >= 0.6 * p(avg_body, 2))
    out["soldiers"], out["crows"] = g3 & big, r3 & big
    return {k: np.nan_to_num(v.astype(float)).astype(bool) for k, v in out.items()}


def _pivots(h, lo, k=PIVOT):
    """Bestaetigte Wendepunkte: (Index des Wendepunkts, bekannt ab Index) fuer Hochs und Tiefs."""
    n = len(h)
    highs, lows = [], []
    for i in range(k, n - k):
        win_h, win_l = h[i - k:i + k + 1], lo[i - k:i + k + 1]
        # bei gleichen Werten zaehlt die erste Kerze (links echt kleiner/groesser, rechts nicht hoeher/tiefer)
        if h[i] == win_h.max() and h[i] > h[i - k:i].max():
            highs.append((i, i + k))
        if lo[i] == win_l.min() and lo[i] < lo[i - k:i].min():
            lows.append((i, i + k))
    return highs, lows


def chart(o, h, lo, c, v) -> dict[str, np.ndarray]:
    """Chart-Muster je Kerze (bool bzw. Abstaende in ATR), nur aus bereits bestaetigten Wendepunkten."""
    n = len(c)
    a = _atr(h, lo, c)
    highs, lows = _pivots(h, lo)
    out = {k: np.zeros(n, bool) for k in ("dbl_bottom", "dbl_top", "hh_hl", "lh_ll", "triangle", "break_up", "break_dn")}
    sr_up, sr_dn = np.full(n, 10.0), np.full(n, 10.0)
    hi_k, lo_k = [], []            # bisher bekannte Wendepunkte (Index, Preis)
    ih = il = 0
    for i in range(n):
        while ih < len(highs) and highs[ih][1] <= i:
            hi_k.append((highs[ih][0], h[highs[ih][0]]))
            ih += 1
        while il < len(lows) and lows[il][1] <= i:
            lo_k.append((lows[il][0], lo[lows[il][0]]))
            il += 1
        ai = a[i] if a[i] > 0 else max(c[i] * 1e-3, 1e-12)
        above = [p for _, p in hi_k[-12:] if p > c[i]]
        below = [p for _, p in lo_k[-12:] if p < c[i]]
        if above:
            sr_up[i] = min(10.0, (min(above) - c[i]) / ai)
        if below:
            sr_dn[i] = min(10.0, (c[i] - max(below)) / ai)
        if len(hi_k) >= 2 and len(lo_k) >= 2:
            (_, h1), (_, h2) = hi_k[-2], hi_k[-1]
            (_, l1), (_, l2) = lo_k[-2], lo_k[-1]
            out["hh_hl"][i] = h2 > h1 and l2 > l1
            out["lh_ll"][i] = h2 < h1 and l2 < l1
            out["triangle"][i] = h2 < h1 and l2 > l1 and (h2 - l2) < 0.7 * (h1 - l1)
        # Doppel-Hoch: zwei aehnliche Hochs, dazwischen ein Tief (Nackenlinie), Schluss darunter
        if len(hi_k) >= 2 and abs(hi_k[-1][1] - hi_k[-2][1]) <= 0.5 * ai:
            neck = [p for j, p in lo_k if hi_k[-2][0] < j < hi_k[-1][0]]
            if neck and c[i] < min(neck) and (i == 0 or c[i - 1] >= min(neck)):
                out["dbl_top"][i] = True
        if len(lo_k) >= 2 and abs(lo_k[-1][1] - lo_k[-2][1]) <= 0.5 * ai:
            neck = [p for j, p in hi_k if lo_k[-2][0] < j < lo_k[-1][0]]
            if neck and c[i] > max(neck) and (i == 0 or c[i - 1] <= max(neck)):
                out["dbl_bottom"][i] = True
        if i > RANGE:
            hh, ll = h[i - RANGE:i].max(), lo[i - RANGE:i].min()
            vol_ok = v[i] >= 1.5 * v[i - RANGE:i].mean()
            out["break_up"][i] = c[i] > hh and vol_ok
            out["break_dn"][i] = c[i] < ll and vol_ok
    out["sr_up"], out["sr_dn"] = sr_up, sr_dn
    return out


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Alle Muster je Kerze als Spalten + zusammengefasste Bewertung (-1 baerisch ... +1 bullisch).
    Kerzen-Muster wirken 3 Kerzen nach, Chart-Muster (Ausbruch/Doppel) 5 Kerzen."""
    o, h, lo, c = (df[k].astype(float).to_numpy() for k in ("open", "high", "low", "close"))
    v = df["volume"].astype(float).to_numpy() if "volume" in df else np.ones(len(df))
    cd, ch = candles(o, h, lo, c), chart(o, h, lo, c, v)
    out = pd.DataFrame(index=df.index)

    def decay(x, k):        # Signal wirkt noch k Kerzen nach (schwaecher werdend)
        s = pd.Series(x.astype(float))
        return np.maximum.reduce([s.shift(j).fillna(0).to_numpy() * (1 - j / (k + 1)) for j in range(k + 1)])
    cs = np.zeros(len(df))
    for key, (_, sign) in CANDLE_NAMES.items():
        out[f"cdl_{key}"] = cd[key].astype(float)
        if sign:
            cs += sign * decay(cd[key], 3)
    out["cdl_score"] = np.clip(cs, -1, 1)
    ps = np.zeros(len(df))
    for key, (_, sign) in CHART_NAMES.items():
        out[f"pat_{key}"] = ch[key].astype(float)
        if sign:
            ps += sign * (decay(ch[key], 5) if key in ("dbl_bottom", "dbl_top", "break_up", "break_dn")
                          else ch[key].astype(float) * 0.5)
    out["pat_score"] = np.clip(ps, -1, 1)
    out["sr_up"], out["sr_dn"] = ch["sr_up"], ch["sr_dn"]
    out["pattern_score"] = np.clip(0.4 * out["cdl_score"] + 0.6 * out["pat_score"], -1, 1)
    return out


def recent(df: pd.DataFrame, tf_seconds: int, bars: int = 80) -> list[dict]:
    """Erkannte Muster der letzten Kerzen fuer Chart-Markierungen und das Bot-Gehirn."""
    if len(df) < 30:
        return []
    pa = analyze(df)
    out = []
    names = {**{f"cdl_{k}": v for k, v in CANDLE_NAMES.items()}, **{f"pat_{k}": v for k, v in CHART_NAMES.items()}}
    start = max(0, len(df) - bars)
    for col, (label, sign) in names.items():
        if col in ("pat_hh_hl", "pat_lh_ll", "pat_triangle", "cdl_doji", "cdl_inside"):
            continue        # Zustaende/haeufige Muster nicht einzeln markieren
        hits = np.where(pa[col].to_numpy()[start:] > 0)[0] + start
        for i in hits:
            strong = col.startswith("pat_") or col in ("cdl_morning", "cdl_evening", "cdl_soldiers", "cdl_crows")
            out.append({"time": int(df["ts"].iloc[i]) // 1000, "name": label, "dir": sign,
                        "kind": "chart" if col.startswith("pat_") else "kerze", "strong": strong})
    out.sort(key=lambda x: x["time"])
    last = pa.iloc[-2] if len(pa) > 1 else pa.iloc[-1]    # letzte abgeschlossene Kerze
    state = [CHART_NAMES[k][0] for k in ("hh_hl", "lh_ll", "triangle") if last[f"pat_{k}"] > 0]
    return [{"state": state, "score": round(float(last["pattern_score"]), 2),
             "sr_up": round(float(last["sr_up"]), 2), "sr_dn": round(float(last["sr_dn"]), 2)}] + out[-25:]


def pattern_blocks(side: str, score: float, s: dict) -> str:
    """Muster-Filter fuer den Bot. pattern_filter: off | avoid (klares Gegen-Muster -> kein Trade) |
    confirm (nur mit passendem Muster). Rueckgabe: Grund oder ''."""
    mode = s.get("pattern_filter") or "off"          # YAML: off ohne Anfuehrungszeichen = False
    if mode in ("off", False) or score is None or score != score:
        return ""
    sign = 1 if side == "long" else -1
    if mode == "avoid" and sign * score <= -s.get("pattern_avoid", 0.5):
        return f"Chart-/Kerzen-Muster dagegen (Muster-Wert {score:+.2f})"
    if mode == "confirm" and sign * score < s.get("pattern_confirm", 0.3):
        return f"kein bestaetigendes Muster (Muster-Wert {score:+.2f})"
    return ""
