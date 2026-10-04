"""Handelszeiten fuer Gold und Silber (und erkennen, wann ein Markt still steht).

Krypto handelt rund um die Uhr. Gold/Silber folgen den Zeiten der Metall-Boersen: Wochenende zu,
taeglich eine kurze Pause. Feste Zeiten (UTC) stehen in config.yaml unter `hours` und lassen sich
anpassen. Zusaetzlich wird automatisch erkannt, wenn sich der Kurs nicht mehr bewegt (Markt zu,
Feiertag): Dann handelt der Bot dort nicht, und die KI lernt nicht aus diesen Kerzen.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .exchange import is_metal

DEFAULT = {
    # Metalle: Wochenende von Freitag 21:00 bis Sonntag 22:00 UTC zu, taeglich Pause 21:00-22:00 UTC
    "metal_week_close": "Fri 21:00", "metal_week_open": "Sun 22:00",
    "metal_daily_pause": ["21:00", "22:00"],
    "no_entry_before_close_min": 30,     # so lange vor Schluss keine neuen Positionen
    "close_before_weekend": True,        # Speed/Autopilot schliessen Metall-Positionen vor dem Wochenende
}
DAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _week_min(s: str) -> int:
    d, t = s.split()
    return DAYS[d] * 1440 + _hm(t)


def _cfg(cfg: dict | None) -> dict:
    return {**DEFAULT, **((cfg or {}).get("hours") or {})}


def minutes_to_close(symbol: str, now: datetime, cfg: dict | None = None) -> float | None:
    """Minuten bis der Markt schliesst (None = Krypto, nie zu). Negativ/0 = gerade zu."""
    if not is_metal(symbol):
        return None
    c = _cfg(cfg)
    now = now.astimezone(timezone.utc)
    wm = now.weekday() * 1440 + now.hour * 60 + now.minute
    wc, wo = _week_min(c["metal_week_close"]), _week_min(c["metal_week_open"])
    # Wochenende (kann ueber den Wochenwechsel gehen)
    in_weekend = (wc <= wm < wo) if wc < wo else (wm >= wc or wm < wo)
    if in_weekend:
        return 0.0
    p0, p1 = (_hm(x) for x in c["metal_daily_pause"])
    dm = now.hour * 60 + now.minute
    if p0 <= dm < p1:
        return 0.0
    to_pause = (p0 - dm) % 1440
    to_week = (wc - wm) % (7 * 1440)
    return float(min(to_pause, to_week))


def is_open(symbol: str, now: datetime | None = None, cfg: dict | None = None) -> bool:
    m = minutes_to_close(symbol, now or datetime.now(timezone.utc), cfg)
    return m is None or m > 0


def entry_allowed(symbol: str, now: datetime | None = None, cfg: dict | None = None) -> tuple[bool, str]:
    """Darf jetzt eine NEUE Position eroeffnet werden? (offen und nicht kurz vor Schluss)"""
    now = now or datetime.now(timezone.utc)
    m = minutes_to_close(symbol, now, cfg)
    if m is None:
        return True, ""
    if m <= 0:
        return False, "Markt geschlossen (Handelszeiten Gold/Silber)"
    if m < _cfg(cfg)["no_entry_before_close_min"]:
        return False, f"Markt schliesst in {m:.0f} min - keine neue Position"
    return True, ""


def close_before_weekend(symbol: str, now: datetime | None = None, cfg: dict | None = None) -> bool:
    """Sollen offene Metall-Positionen jetzt geschlossen werden (letzte Minuten vor dem Wochenende)?"""
    c = _cfg(cfg)
    if not c["close_before_weekend"] or not is_metal(symbol):
        return False
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    wm = now.weekday() * 1440 + now.hour * 60 + now.minute
    to_week = (_week_min(c["metal_week_close"]) - wm) % (7 * 1440)
    return 0 < to_week <= 10


def flat_mask(df: pd.DataFrame, run: int = 3) -> np.ndarray:
    """Kerzen, in denen der Markt still stand (High = Low ueber mind. `run` Kerzen in Folge, oder kein
    Volumen) - typisch fuer geschlossene Metall-Maerkte. Diese Kerzen lernt die KI nicht."""
    h, lo = df["high"].astype(float).to_numpy(), df["low"].astype(float).to_numpy()
    dead = (h - lo) <= 1e-12
    if "volume" in df:
        dead |= df["volume"].astype(float).to_numpy() <= 0
    out = np.zeros(len(df), bool)
    cnt = 0
    for i, d in enumerate(dead):
        cnt = cnt + 1 if d else 0
        if cnt >= run:
            out[i - run + 1:i + 1] = True
    return out


def recently_flat(df: pd.DataFrame, bars: int = 3) -> bool:
    """Steht der Markt gerade still (die letzten abgeschlossenen Kerzen ohne Bewegung)?"""
    if len(df) < bars + 1:
        return False
    return bool(flat_mask(df.iloc[-(bars + 1):-1], run=bars).all())


def next_change(symbol: str, now: datetime, cfg: dict | None = None) -> str:
    """Kurztext fuer die Oberflaeche: 'offen bis Fr 21:00' / 'zu bis So 22:00'."""
    if not is_metal(symbol):
        return "rund um die Uhr"
    m = minutes_to_close(symbol, now, cfg)
    if m and m > 0:
        t = now + timedelta(minutes=m)
        return f"offen bis {t.strftime('%a %H:%M')} UTC"
    for k in range(1, 7 * 1440, 5):
        t = now + timedelta(minutes=k)
        if (minutes_to_close(symbol, t, cfg) or 0) > 0:
            return f"geschlossen bis {t.strftime('%a %H:%M')} UTC"
    return "geschlossen"
