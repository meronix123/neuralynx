"""Marktumfeld: Wirtschaftskalender und Angst-&-Gier-Index.

Der Bot holt die Daten selbst (stuendlich zwischengespeichert):
- Wirtschaftskalender (ForexFactory-Feed): rund um Termine mit hoher Wirkung
  (z. B. Zinsentscheid, Inflation/CPI, Arbeitsmarkt/NFP) werden KEINE neuen
  Trades eroeffnet - dort springen Kurse oft ueber jeden Stop hinweg.
- Crypto Fear & Greed Index (alternative.me): bei extremer Angst oder Gier
  wird das Risiko pro Trade halbiert.
"""
import logging
import time
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("bot")

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FNG_URL = "https://api.alternative.me/fng/?limit=1"


def parse_calendar(rows: list, currencies: list[str], impacts: list[str]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("country") not in currencies or r.get("impact") not in impacts:
            continue
        try:
            when = datetime.fromisoformat(r["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        out.append({"title": r.get("title", "?"), "country": r["country"],
                    "impact": r["impact"], "time": when})
    return sorted(out, key=lambda e: e["time"])


class MarketContext:
    def __init__(self, cfg: dict, http=requests):
        self.cfg = cfg
        self.http = http
        self.events: list[dict] = []
        self.fng: dict | None = None
        self._cal_at = 0.0
        self._fng_at = 0.0
        self.cal_ok = False

    def refresh(self) -> None:
        now = time.time()
        if now - self._cal_at > 3600:
            self._cal_at = now
            try:
                rows = self.http.get(CALENDAR_URL, timeout=15).json()
                self.events = parse_calendar(rows, self.cfg["currencies"], self.cfg["impacts"])
                self.cal_ok = True
                log.info("Wirtschaftskalender: %d wichtige Termine diese Woche", len(self.events))
            except Exception as e:  # noqa: BLE001
                self.cal_ok = False
                self._cal_at = now - 3300  # in 5 Minuten erneut versuchen
                log.warning("Wirtschaftskalender nicht abrufbar: %s", e)
        if now - self._fng_at > 3600:
            self._fng_at = now
            try:
                d = self.http.get(FNG_URL, timeout=15).json()["data"][0]
                self.fng = {"value": int(d["value"]), "label": d["value_classification"]}
            except Exception as e:  # noqa: BLE001
                log.warning("Fear & Greed nicht abrufbar: %s", e)

    def blocking_event(self, now: datetime) -> dict | None:
        before = timedelta(minutes=self.cfg["minutes_before"])
        after = timedelta(minutes=self.cfg["minutes_after"])
        for e in self.events:
            if e["time"] - before <= now <= e["time"] + after:
                return e
        return None

    def next_events(self, now: datetime, n: int = 5) -> list[dict]:
        return [e for e in self.events if e["time"] + timedelta(minutes=self.cfg["minutes_after"]) >= now][:n]

    def check(self, now: datetime) -> tuple[bool, str, float]:
        """-> (neue Trades erlaubt?, Grund, Risiko-Faktor)"""
        self.refresh()
        if not self.cal_ok and self.cfg["block_if_calendar_unavailable"]:
            return False, "Wirtschaftskalender nicht verfuegbar", 0.0
        ev = self.blocking_event(now)
        if ev:
            return False, f"Wirtschaftstermin: {ev['country']} {ev['title']} um {ev['time']:%H:%M} UTC", 0.0
        factor = 1.0
        if self.fng and not (self.cfg["fng_extreme_low"] < self.fng["value"] < self.cfg["fng_extreme_high"]):
            factor = 0.5
        return True, "", factor
