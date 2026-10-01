"""Order Blocks (Smart-Money-Zonen) aus den Kerzen.

Bullischer Order Block: die letzte ROTE Kerze, bevor der Kurs kraeftig nach oben schiesst
(mind. `ob_disp_atr` x ATR ueber ihr Hoch innerhalb von `ob_look` Kerzen). Dort haben grosse
Marktteilnehmer oft gekauft - kommt der Kurs zurueck, reagiert er dort haeufig (Unterstuetzung).
Baerischer Order Block umgekehrt (letzte gruene Kerze vor einem kraeftigen Fall -> Widerstand).

Eine Zone gilt, bis ein Bar dahinter SCHLIESST (dann ist sie "gebrochen") oder sie zu alt ist.
Bekannt ist eine Zone erst, wenn die kraeftige Bewegung bestaetigt ist - kein Blick in die Zukunft.

Je Bar entstehen drei Werte (Backtest und live gleich):
  ob_above   Unterkante der naechsten baerischen Zone ueber dem Kurs (Widerstand) oder NaN
  ob_below   Oberkante der naechsten bullischen Zone unter/um den Kurs (Unterstuetzung) oder NaN
  ob_support +1 = Kurs hat in den letzten Bars eine bullische Zone angetestet und gehalten,
             -1 = baerische Zone angetestet und abgeprallt, 0 = nichts
"""
import numpy as np

MAX_ACTIVE = 20      # je Richtung hoechstens so viele offene Zonen merken
MAX_AGE = 500        # aeltere Zonen verfallen (Bars)
SUPPORT_BARS = 3     # Antest zaehlt so viele Bars lang als Bestaetigung


def detect(o, h, lo, c, atr, disp_atr: float = 1.5, look: int = 3):
    """-> (ob_above, ob_below, ob_support, zones). zones = am Ende noch gueltige Zonen."""
    n = len(c)
    above = np.full(n, np.nan)
    below = np.full(n, np.nan)
    support = np.zeros(n, dtype=int)
    bull: list[dict] = []
    bear: list[dict] = []
    used: set = set()
    last_touch_bull, last_touch_bear = -10**9, -10**9
    for i in range(n):
        a = atr[i]
        # 1) neue Zonen bestaetigen (letzte Gegenkerze der letzten `look` Bars vor der Bewegung)
        if a == a and a > 0:
            for j in range(i - 1, max(i - look, 0) - 1, -1):
                if c[j] < o[j]:
                    if j not in used and c[i] - h[j] >= disp_atr * a:
                        used.add(j)
                        bull.append({"kind": "bull", "top": h[j], "bottom": lo[j], "i": j, "confirm": i, "touches": 0,
                                     "strength": (c[i] - h[j]) / a})
                    break
            for j in range(i - 1, max(i - look, 0) - 1, -1):
                if c[j] > o[j]:
                    if -j - 1 not in used and lo[j] - c[i] >= disp_atr * a:
                        used.add(-j - 1)
                        bear.append({"kind": "bear", "top": h[j], "bottom": lo[j], "i": j, "confirm": i, "touches": 0,
                                     "strength": (lo[j] - c[i]) / a})
                    break
        # 2) gebrochene / alte Zonen entfernen, Antests zaehlen (nur Zonen, die vor diesem Bar bekannt waren)
        keep = []
        for z in bull:
            if c[i] < z["bottom"] or i - z["i"] > MAX_AGE:
                continue
            if z["confirm"] < i and lo[i] <= z["top"]:
                z["touches"] += 1
                last_touch_bull = i
            keep.append(z)
        bull = keep[-MAX_ACTIVE:]
        keep = []
        for z in bear:
            if c[i] > z["top"] or i - z["i"] > MAX_AGE:
                continue
            if z["confirm"] < i and h[i] >= z["bottom"]:
                z["touches"] += 1
                last_touch_bear = i
            keep.append(z)
        bear = keep[-MAX_ACTIVE:]
        # 3) naechste Zonen zum Kurs
        tops = [z["top"] for z in bull if z["bottom"] <= c[i]]
        if tops:
            below[i] = min(max(tops), c[i])
        bottoms = [z["bottom"] for z in bear if z["top"] >= c[i]]
        if bottoms:
            above[i] = max(min(bottoms), c[i])
        tb, ts = i - last_touch_bull < SUPPORT_BARS, i - last_touch_bear < SUPPORT_BARS
        support[i] = 1 if tb and not ts else -1 if ts and not tb else 0
    return above, below, support, bull + bear


def add_columns(d, s: dict):
    """Spalten ob_above / ob_below / ob_support an ein Signal-DataFrame haengen."""
    above, below, support, _ = detect(d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(),
                                      d["close"].to_numpy(), d["atr"].to_numpy(),
                                      s.get("ob_disp_atr", 1.5), s.get("ob_look", 3))
    d["ob_above"], d["ob_below"], d["ob_support"] = above, below, support
    return d


def zones_for_chart(d, s: dict, max_each: int = 4, disp: float | None = None) -> list[dict]:
    """Noch gueltige Zonen (nur abgeschlossene Bars), die dem Kurs am naechsten liegen.
    disp: Mindest-Staerke fuer die Anzeige (Standard wie der Filter); 'strong' = so stark wie der Filter verlangt."""
    closed = d.iloc[:-1]
    if len(closed) < 20:
        return []
    *_, zones = detect(closed["open"].to_numpy(), closed["high"].to_numpy(), closed["low"].to_numpy(),
                       closed["close"].to_numpy(), closed["atr"].to_numpy(),
                       disp if disp is not None else s.get("ob_disp_atr", 1.5), s.get("ob_look", 3))
    price = float(closed["close"].iloc[-1])
    ts = closed["ts"].to_numpy()
    out = []
    for kind in ("bull", "bear"):
        zs = sorted((z for z in zones if z["kind"] == kind), key=lambda z: abs((z["top"] + z["bottom"]) / 2 - price))
        for z in zs[:max_each]:
            out.append({"kind": kind, "top": float(z["top"]), "bottom": float(z["bottom"]),
                        "from_ts": int(ts[z["i"]]), "touches": z["touches"],
                        "strength": round(float(z["strength"]), 2),
                        "strong": bool(z["strength"] >= s.get("ob_disp_atr", 1.5))})
    return out


def ob_blocks(side: str, close: float, tp_dist: float, above, below, support, s: dict) -> str:
    """Order-Block-Filter. Rueckgabe: Grund fuer die Ablehnung oder '' (erlaubt).

    ob_filter: off | avoid (Gegen-Zone versperrt den Weg zum Ziel) | confirm (nur nach Antest
    einer passenden Zone) | both
    """
    mode = s.get("ob_filter", "off")
    if mode == "off" or close != close:
        return ""
    room = s.get("ob_room", 0.6)
    if mode in ("avoid", "both") and tp_dist == tp_dist and tp_dist > 0:
        if side == "long" and above == above and above - close < room * tp_dist:
            return "Order Block (Widerstand) versperrt den Weg zum Ziel"
        if side == "short" and below == below and close - below < room * tp_dist:
            return "Order Block (Unterstuetzung) versperrt den Weg zum Ziel"
    if mode in ("confirm", "both"):
        need = 1 if side == "long" else -1
        if support != need:
            return "kein Antest eines passenden Order Blocks"
    return ""
