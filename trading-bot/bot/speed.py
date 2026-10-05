"""Speed-Trading: fuer eine begrenzte Zeit (z. B. 5 Minuten) sehr schnell kleine Bewegungen handeln.

- Signal: 1-, 2-, 3- und 5-Minuten-Chart (schnelle EMAs + Schwung) und Orderbuch-Druck.
  Einstieg, wenn genug davon in dieselbe Richtung zeigen (Aggressivitaet einstellbar).
- Einstieg mit Post-Only-Limit-Order am besten Preis (Maker-Gebuehr); nicht ausgefuehrt nach
  `entry_timeout_s` oder bei Richtungswechsel -> storniert.
- Ziel als Limit-Order (Maker), Stop als an die Position gebundener Bitget-Stop. Abstaende aus der
  1-Minuten-Volatilitaet; das Ziel ist immer mind. `min_tp_fee_x` x die Gebuehren.
- Je Trade Zeit-Limit, je Sitzung Verlust-Limit und Hoechstzahl an Trades. Am Ende der Sitzung
  werden offene Speed-Positionen geschlossen.
- Laeuft in der Simulation (echte Kurse, Spielgeld) oder auf dem verbundenen Bitget-Konto.
"""
import logging
import threading
import time

import numpy as np
import pandas as pd

from .indicators import atr, ema

log = logging.getLogger(__name__)

DEFAULTS = {
    "minutes": 0, "margin_usdt": 5.0, "leverage": 10, "aggressiveness": 3,   # 1 = vorsichtig ... 4 = sehr schnell
    "entry_timeout_s": 20, "max_hold_s": 0, "max_trades": 1000, "max_loss_pct": 5.0,
    "tp_atr": 1.0, "sl_atr": 1.0, "min_tp_fee_x": 3.0, "loop_s": 1.5,
    "maker": 0.0002, "taker": 0.0006, "slippage": 0.0003, "min_notional": 5.0,
    # Positionsgroesse: usdt = fester Einsatz | pct = % vom Konto als Einsatz | risk = % vom Konto Verlust am Stop
    "size_mode": "usdt", "size_pct": 10.0, "risk_pct": 1.0, "max_margin_pct": 50.0,
    "hours": {}, "log_path": None,
}
# KI-Autopilot: handelt nach der KI-Prognose (30 min) und fuehrt die Position selbst
AUTO_DEFAULTS = {
    "minutes": 0, "loop_s": 5.0, "entry_timeout_s": 60, "max_hold_s": 0, "max_trades": 1000,
    "min_conf": 0.56,          # Mindest-Sicherheit der KI fuer einen Einstieg
    "use_raw": False,          # Rohsignal auch ohne nachgewiesene Treffsicherheit (nur Simulation)
    "sl_band": 1.0,            # Stop = 1 x die 80-%-Schwankung der naechsten 30 min
    "tp_r": 2.0,               # Ziel = 2 x Risiko
    "be_r": 0.7,               # ab +0,7 R Stop auf Einstand (inkl. Gebuehren)
    "partial_r": 1.0,          # ab +1 R ...
    "partial_frac": 0.5,       # ... die Haelfte verkaufen
    "trail_r": 1.0,            # danach Stop im Abstand 1 R hinter dem besten Kurs nachziehen
    "fast": False,             # Fast-Modus: 5-10-min-Prognose, enge Stops/Ziele, sehr schnelle Reaktion
    "scale_in": False,         # Teilkauf: erst einen Teil kaufen, Rest nur im GEWINN nachkaufen (nie im Verlust)
    "first_frac": 0.5,         #   erster Teil (Anteil der Positionsgroesse)
    "add_at_r": 0.5,           #   Rest bei +0,5 R, wenn die KI weiter dafuer ist
    "gate_min_n": 150,         # Markt-Sperre: ab so vielen live geprueften KI-Entscheidungen ...
    "gate_min_hit": 0.5,       # ... und Trefferquote darunter handelt der Autopilot dort nicht
}
FAST_AUTO = {"loop_s": 1.5, "entry_timeout_s": 20, "be_r": 0.5, "partial_r": 0.8, "trail_r": 0.7, "tp_r": 1.5}
# Hebel-Speed-KI (Turbo): eigenes 1-Minuten-Modell, noch engere Stops, Pruefung jede Sekunde
TURBO_AUTO = {"loop_s": 1.0, "entry_timeout_s": 15, "be_r": 0.5, "partial_r": 0.8, "trail_r": 0.6, "tp_r": 1.5,
              "fast": True}
TFS = {"1m": 1, "2m": 2, "3m": 3, "5m": 5}


def micro_signal(df1m: pd.DataFrame, book: dict | None, aggressiveness: int = 3) -> tuple[int, float, dict]:
    """Richtung (+1 long, -1 short, 0 nichts), Staerke 0..1 und Einzelwerte.
    df1m: 1-Minuten-Kerzen (ts, open, high, low, close, volume), die letzte darf offen sein."""
    df1m = df1m.reset_index(drop=True)
    if len(df1m) < 40:
        return 0, 0.0, {}
    votes = {}
    idx = pd.to_datetime(df1m["ts"], unit="ms", utc=True)
    base = df1m.set_index(idx)
    for name, n in TFS.items():
        d = base if n == 1 else base.resample(f"{n}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        c = d["close"].astype(float)
        if len(c) < 8:
            continue
        fast, slow = ema(c, 3), ema(c, 8)
        trend = np.sign(fast.iloc[-1] - slow.iloc[-1])
        push = np.sign(c.iloc[-1] - c.iloc[-3])            # Schwung der letzten Kerzen
        votes[name] = float(trend) if trend == push else 0.0
    imb = 0.0
    if book and book.get("bids") and book.get("asks"):
        b = sum(float(x[1]) for x in book["bids"][:20])
        a = sum(float(x[1]) for x in book["asks"][:20])
        imb = (b - a) / (b + a) if b + a > 0 else 0.0
    votes["buch"] = 1.0 if imb > 0.15 else -1.0 if imb < -0.15 else 0.0
    score = sum(votes.values())
    need = {1: 5, 2: 4, 3: 3, 4: 2}.get(int(aggressiveness), 3)   # wie viele Stimmen einig sein muessen
    side = 1 if score >= need else -1 if score <= -need else 0
    return side, min(1.0, abs(score) / 5), {**votes, "orderbuch": round(imb, 2), "punkte": score}


def levels(df1m: pd.DataFrame, price: float, side: int, p: dict) -> tuple[float, float]:
    """Stop und Ziel aus der 1-Minuten-Volatilitaet; das Ziel deckt die Gebuehren mehrfach."""
    a = float(atr(df1m.reset_index(drop=True), 14).iloc[-2])
    if not a == a or a <= 0:
        a = price * 0.001
    fees = price * (2 * p["maker"])
    tp_d = max(p["tp_atr"] * a, p["min_tp_fee_x"] * fees)
    sl_d = max(p["sl_atr"] * a, price * 0.0008)
    return price - side * sl_d, price + side * tp_d


class PaperBroker:
    """Simulation mit echten Kursen: Limit-Orders werden nur ausgefuehrt, wenn der Kurs durchlaeuft."""

    def __init__(self, market_fn, p: dict):
        self.market_fn, self.p = market_fn, p
        self.orders: dict[str, dict] = {}
        self.n = 0

    def _id(self):
        self.n += 1
        return f"sp{self.n}"

    def place_entry(self, sym, side, qty, price):
        oid = self._id()
        self.orders[oid] = {"sym": sym, "side": side, "qty": qty, "price": price, "status": "open"}
        return oid

    def entry_status(self, sym, oid):
        o = self.orders[oid]
        if o["status"] == "open":
            last = self.market_fn(sym)["last"]
            if (o["side"] == 1 and last < o["price"]) or (o["side"] == -1 and last > o["price"]):
                o["status"] = "filled"
        return o["status"], o["price"], o["qty"] if o["status"] == "filled" else 0.0

    def cancel(self, sym, oid):
        if oid in self.orders and self.orders[oid]["status"] == "open":
            self.orders[oid]["status"] = "canceled"

    def protect(self, sym, side, qty, sl, tp):
        return {"sl": sl, "tp": tp}

    def exit_status(self, sym, side, qty, prot):
        last = self.market_fn(sym)["last"]
        if (side == 1 and last > prot["tp"]) or (side == -1 and last < prot["tp"]):
            return "ziel", prot["tp"], True
        if (side == 1 and last <= prot["sl"]) or (side == -1 and last >= prot["sl"]):
            fill = min(last, prot["sl"]) if side == 1 else max(last, prot["sl"])
            return "stop", fill * (1 - side * self.p["slippage"]), False
        return None

    def close(self, sym, side, qty, prot):
        last = self.market_fn(sym)["last"]
        return last * (1 - side * self.p["slippage"])

    def move_stop(self, sym, side, qty, prot, sl):
        prot["sl"] = sl

    def close_part(self, sym, side, qty, prot):
        last = self.market_fn(sym)["last"]
        return last * (1 - side * self.p["slippage"])

    def add(self, sym, side, qty):
        last = self.market_fn(sym)["last"]
        return last * (1 + side * self.p["slippage"])

    def set_leverage(self, sym, lev):
        pass

    def resize(self, sym, side, qty, prot):
        pass


class BitgetBroker:
    """Echtes Bitget-Konto (ccxt-Client mit mode_safe). Stop = an die Position gebundener Bitget-Stop."""
    GRACE_S = 10       # so lange nach dem Absichern gilt eine "fehlende" Position noch nicht als geschlossen
    MISS_CHECKS = 3    # so oft hintereinander muss die Position fehlen

    def __init__(self, client, p: dict, margin_mode: str = "isolated"):
        self.c, self.p, self.mm = client, p, margin_mode

    def place_entry(self, sym, side, qty, price):
        o = self.c.create_order(sym, "limit", "buy" if side == 1 else "sell", qty, price,
                                {"postOnly": True, "marginMode": self.mm})
        return str(o["id"])

    def entry_status(self, sym, oid):
        o = self.c.fetch_order(oid, sym)
        filled = float(o.get("filled") or 0)
        st = o.get("status")
        if st == "closed" or (st in ("canceled", "cancelled", "expired") and filled > 0):
            return "filled", float(o.get("average") or o.get("price")), filled
        if st in ("canceled", "cancelled", "expired", "rejected"):
            return "canceled", 0.0, 0.0
        return "open", 0.0, filled

    def cancel(self, sym, oid):
        try:
            self.c.cancel_order(oid, sym)
        except Exception as e:  # noqa: BLE001 - schon ausgefuehrt/erledigt
            log.debug("Speed Storno %s: %s", oid, e)

    def protect(self, sym, side, qty, sl, tp):
        from .exchange import place_size_tpsl
        sl_id = place_size_tpsl(self.c, sym, "long" if side == 1 else "short", "loss_plan", sl, qty, self.mm)
        try:
            tp_o = self.c.create_order(sym, "limit", "sell" if side == 1 else "buy", qty,
                                       float(self.c.price_to_precision(sym, tp)),
                                       {"reduceOnly": True, "marginMode": self.mm})
        except Exception as e:  # noqa: BLE001 - Ziel nicht setzbar: Stop bleibt, Zeit-Limit schliesst
            log.warning("Speed: Ziel %s nicht gesetzt: %s", sym, e)
            tp_o = {"id": ""}
        return {"sl": sl, "tp": tp, "sl_id": sl_id, "tp_id": str(tp_o.get("id") or ""), "t": time.time(), "miss": 0}

    def _pos_size(self, sym, side):
        want = "long" if side == 1 else "short"
        return sum(float(p.get("contracts") or 0) for p in self.c.fetch_positions([sym]) if p.get("side") == want)

    def exit_status(self, sym, side, qty, prot):
        if prot.get("tp_id"):
            o = self.c.fetch_order(prot["tp_id"], sym)
            if o.get("status") == "closed":
                self._cleanup(sym, prot, keep="tp")
                return "ziel", float(o.get("average") or prot["tp"]), True
        if self._pos_size(sym, side) < qty * 0.5:          # Stop ausgeloest (Position weg)?
            # Bitget meldet eine neue Position oft erst nach ein paar Sekunden: erst nach mehreren
            # Fehlmeldungen hintereinander (und nicht direkt nach dem Kauf) als geschlossen werten,
            # sonst wuerde die Position vergessen, obwohl sie auf Bitget offen ist.
            prot["miss"] = prot.get("miss", 0) + 1
            if prot["miss"] < self.MISS_CHECKS or time.time() - prot.get("t", 0) < self.GRACE_S:
                return None
            self._cleanup(sym, prot, keep="sl")
            return "stop", prot["sl"] * (1 - side * self.p["slippage"]), False
        prot["miss"] = 0
        return None

    def _cleanup(self, sym, prot, keep=""):
        from .exchange import cancel_plan
        if keep != "tp" and prot.get("tp_id"):
            self.cancel(sym, prot["tp_id"])
        if keep != "sl" and prot.get("sl_id"):
            try:
                cancel_plan(self.c, sym, prot["sl_id"])
            except Exception as e:  # noqa: BLE001
                log.debug("Speed Stop-Storno: %s", e)

    def close(self, sym, side, qty, prot):
        self._cleanup(sym, prot)
        o = self.c.create_order(sym, "market", "sell" if side == 1 else "buy", qty, None,
                                {"reduceOnly": True, "marginMode": self.mm})
        return float(o.get("average") or self.c.fetch_ticker(sym)["last"])

    def move_stop(self, sym, side, qty, prot, sl):
        from .exchange import cancel_plan, place_size_tpsl
        new = place_size_tpsl(self.c, sym, "long" if side == 1 else "short", "loss_plan", sl, qty, self.mm)
        try:
            cancel_plan(self.c, sym, prot["sl_id"])
        except Exception as e:  # noqa: BLE001
            log.debug("Speed alter Stop: %s", e)
        prot.update(sl=sl, sl_id=new)

    def set_leverage(self, sym, lev):
        """Hebel fuer diesen Markt setzen (nur wenn er sich aendert)."""
        cache = self.__dict__.setdefault("_lev", {})
        if cache.get(sym) == lev:
            return
        for hs in ("long", "short"):
            try:
                self.c.set_leverage(int(lev), sym, {"holdSide": hs, "marginMode": self.mm})
            except Exception as e:  # noqa: BLE001 - z. B. schon gesetzt
                log.debug("Hebel %s: %s", sym, e)
        cache[sym] = lev

    def balance(self):
        """(Gesamt, frei) in USDT - hoechstens alle 5 s neu abgefragt."""
        hit = getattr(self, "_bal", None)
        if hit and time.time() - hit[0] < 5:
            return hit[1]
        b = self.c.fetch_balance({"type": "swap"}).get("USDT", {}) or {}
        val = (float(b.get("total") or 0), float(b.get("free") or 0))
        self._bal = (time.time(), val)
        return val

    def add(self, sym, side, qty):
        """Nachkauf (Teilkauf) zum Marktpreis - nur im Gewinn aufgerufen."""
        o = self.c.create_order(sym, "market", "buy" if side == 1 else "sell", qty, None, {"marginMode": self.mm})
        self._bal = None
        return float(o.get("average") or self.c.fetch_ticker(sym)["last"])

    def close_part(self, sym, side, qty, prot):
        """Teilverkauf zum Marktpreis (nur reduzierend)."""
        o = self.c.create_order(sym, "market", "sell" if side == 1 else "buy", qty, None,
                                {"reduceOnly": True, "marginMode": self.mm})
        return float(o.get("average") or self.c.fetch_ticker(sym)["last"])

    def resize(self, sym, side, qty, prot):
        """Nach einem Teilverkauf: Stop und Ziel auf die Restmenge setzen (neue zuerst, dann alte weg)."""
        from .exchange import cancel_plan, place_size_tpsl
        new_sl = place_size_tpsl(self.c, sym, "long" if side == 1 else "short", "loss_plan", prot["sl"], qty, self.mm)
        try:
            cancel_plan(self.c, sym, prot["sl_id"])
        except Exception as e:  # noqa: BLE001
            log.debug("Autopilot alter Stop: %s", e)
        prot["sl_id"] = new_sl
        if prot.get("tp_id"):
            self.cancel(sym, prot["tp_id"])
            try:
                o = self.c.create_order(sym, "limit", "sell" if side == 1 else "buy", qty,
                                        float(self.c.price_to_precision(sym, prot["tp"])),
                                        {"reduceOnly": True, "marginMode": self.mm})
                prot["tp_id"] = str(o.get("id") or "")
            except Exception as e:  # noqa: BLE001 - Stop bleibt, Ziel fehlt
                log.warning("Autopilot: Ziel nach Teilverkauf nicht gesetzt: %s", e)
                prot["tp_id"] = ""


class SpeedTrader:
    """Eine Handels-Sitzung aus der Oberflaeche: Speed-Trading (kind='speed') oder KI-Autopilot (kind='ki')."""

    def __init__(self, data_fn, cfg: dict | None = None, forecast_fn=None, kind: str = "speed"):
        """data_fn(sym) -> (1-Minuten-Kerzen, Orderbuch, {'last','bid','ask'}) - fuer Signal und Simulation.
        forecast_fn(sym) -> KI-Prognose (nur KI-Autopilot)."""
        self.data_fn, self.forecast_fn, self.kind = data_fn, forecast_fn, kind
        self.name = "KI-Autopilot" if kind == "ki" else "Speed-Trading"
        self.p = {**DEFAULTS, **(AUTO_DEFAULTS if kind == "ki" else {}), **(cfg or {})}
        self.lock = threading.RLock()
        self.active, self.thread = False, None
        self.session: dict = {}
        self.slots: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.events: list[str] = []
        self.market: dict[str, dict] = {}

    # --- Steuerung -------------------------------------------------------
    def start(self, symbols: list[str], broker, equity: float, label: str, free_fn=None, **opts) -> str:
        """free_fn(sym) -> True, wenn im Konto in diesem Markt nichts anderes offen ist (sonst wird dort gewartet)."""
        with self.lock:
            if self.active:
                raise RuntimeError(f"{self.name} laeuft bereits")
            if not symbols:
                raise ValueError("Mindestens einen Markt waehlen")
            p = {**self.p, **{k: v for k, v in opts.items() if v is not None}}
            if self.kind == "ki" and p.get("turbo"):
                p.update(TURBO_AUTO)
                if getattr(self, "turbo_fn", None) is None:
                    raise ValueError("Turbo-KI ist hier nicht verfuegbar")
            elif self.kind == "ki" and p.get("fast"):
                p.update(FAST_AUTO)
            minutes = int(p.get("minutes") or 0)              # 0 = laeuft, bis "Aus" gedrueckt wird
            if self.kind == "ki" and p.get("use_raw") and label != "Simulation":
                raise ValueError("Rohsignal (unbewaehrte KI) nur in der Simulation")
            if p.get("size_mode", "usdt") == "usdt":
                if p["margin_usdt"] * p["leverage"] < p["min_notional"]:
                    raise ValueError(f"Einsatz x Hebel muss mind. {p['min_notional']:g} USDT sein (Bitget-Minimum)")
                if p["margin_usdt"] > equity:
                    raise ValueError(f"Einsatz {p['margin_usdt']:g} USDT ist mehr als verfuegbar ({equity:.2f})")
            self.cur, self.broker, self.free_fn = p, broker, free_fn
            now = time.time()
            self.session = {"symbols": list(symbols), "start": now, "label": label, "stopping": False,
                            "end": now + minutes * 60 if minutes > 0 else float("inf"),
                            "equity0": equity, "params": {k: p.get(k) for k in ("minutes", "margin_usdt", "leverage",
                                                                                 "aggressiveness", "min_conf", "use_raw", "fast", "turbo",
                                                                                 "size_mode", "size_pct", "risk_pct",
                                                                                 "scale_in", "partial_frac")}}
            self.slots = {s: {"state": "idle"} for s in symbols}
            self.trades, self.events = [], []
            self.active = True
            self._event(f"Ein: {', '.join(x.split(':')[0] for x in symbols)} ({label})")
            self.thread = threading.Thread(target=self._run, daemon=True, name=self.kind)
            self.thread.start()
            return f"{self.name} ist an ({label}) - laeuft, bis du Aus drueckst"

    def stop(self, why: str = "Aus gedrueckt", close: bool = False) -> str:
        """Aus: keine neuen Trades mehr, wartende Limit-Orders weg. Offene Positionen behalten Stop und Ziel
        und werden weiter gefuehrt, bis sie zu sind (close=True: sofort alle zum Marktpreis schliessen)."""
        with self.lock:
            if not self.active:
                return f"{self.name} ist aus"
            self.session["stop_reason"] = why
            self.session["stopping"] = True
            if close:
                self.session["close_now"] = True
        if close:
            return f"{self.name} aus - offene Positionen der Sitzung werden jetzt geschlossen"
        n = sum(1 for v in self.slots.values() if v.get("state") == "open")
        return (f"{self.name} aus - keine neuen Trades. " +
                (f"{n} offene Position(en) laufen mit Stop und Ziel weiter, bis sie zu sind" if n else "Keine offene Position"))

    def busy(self, sym: str) -> bool:
        """Haelt die Sitzung in diesem Markt gerade selbst eine Position/Order auf dem echten Konto?
        (Simulation sperrt nichts; ein Markt, in dem nur gesucht wird, ist frei.)"""
        return (self.active and self.session.get("label") != "Simulation"
                and (self.slots.get(sym) or {}).get("state") in ("pending", "open"))

    # --- Ablauf ------------------------------------------------------------
    def _event(self, text: str) -> None:
        last = getattr(self, "_last_ev", ("", 0.0))
        if text == last[0] and time.time() - last[1] < 30:       # gleiche Meldung nicht jede Sekunde wiederholen
            return
        self._last_ev = (text, time.time())
        self.events.append(f"{time.strftime('%H:%M:%S')} {text}")
        self.events = self.events[-60:]
        log.info("%s: %s", self.name, text)

    def _run(self) -> None:
        try:
            while True:
                ses = self.session
                if ses.get("close_now"):
                    break
                if not ses["stopping"]:
                    if time.time() >= ses["end"]:
                        ses.update(stopping=True, stop_reason="Zeit abgelaufen")
                    elif self._net() <= -ses["equity0"] * self.cur["max_loss_pct"] / 100:
                        ses.update(stopping=True, stop_reason=f"Verlust-Limit {self.cur['max_loss_pct']:g} % erreicht")
                if ses["stopping"]:
                    self._cancel_pending()
                    if not any(v.get("state") == "open" for v in self.slots.values()):
                        break                                   # nichts mehr offen -> Sitzung zu Ende
                for sym in ses["symbols"]:
                    try:
                        self.step(sym)
                    except Exception as e:  # noqa: BLE001 - ein Fehler darf die Sitzung nicht beenden
                        self._event(f"{sym.split(':')[0]} Fehler: {e}")
                time.sleep(self.cur["loop_s"])
        finally:
            self._wind_down(close=bool(self.session.get("close_now")))

    def _cancel_pending(self) -> None:
        for sym, sl in self.slots.items():
            if sl.get("state") != "pending":
                continue
            try:
                self.broker.cancel(sym, sl["oid"])
                st, avg, filled = self.broker.entry_status(sym, sl["oid"])
                if filled > 0:                            # kurz vor dem Storno ausgefuehrt -> absichern
                    sl.update(state="open", entry=avg, qty=filled, opened=time.time(), r0=abs(avg - sl["sl"]),
                              best=avg, prot=self.broker.protect(sym, sl["side"], filled, sl["sl"], sl["tp"]))
                else:
                    sl.clear()
                    sl["state"] = "idle"
            except Exception as e:  # noqa: BLE001
                self._event(f"{sym.split(':')[0]} Storno: {e}")

    def _wind_down(self, close: bool = False) -> None:
        self._cancel_pending()
        for sym, sl in self.slots.items():
            if sl.get("state") != "open":
                continue
            try:
                if close:
                    px = self.broker.close(sym, sl["side"], sl["qty"], sl["prot"])
                    self._book(sym, sl, px, "von Hand geschlossen", maker=False)
                else:
                    self._event(f"{sym.split(':')[0]}: Position bleibt offen (Stop und Ziel liegen auf Bitget)")
            except Exception as e:  # noqa: BLE001
                self._event(f"{sym.split(':')[0]} beim Beenden: {e} - bitte im Konto pruefen!")
        self.active = False
        self._event(f"Aus ({self.session.get('stop_reason', '-')}): {len(self.trades)} Abschluesse, "
                    f"netto {self._net():+.4f} USDT")

    def _market(self, sym):
        df, book, tick = self.data_fn(sym)
        self.market[sym] = tick
        return df, book, tick

    def _signal(self, sym, df, book):
        """-> (Richtung, Einzelwerte, KI-Prognose oder None)."""
        if self.kind != "ki":
            side, _, votes = micro_signal(df, book, self.cur["aggressiveness"])
            return side, votes, None
        fn = getattr(self, "turbo_fn", None) if self.cur.get("turbo") else self.forecast_fn
        fc = fn(sym) if fn else None
        if not fc or not fc.get("ok"):
            return 0, {"KI": (fc or {}).get("msg", "keine Prognose")}, None
        live = fc.get("live") or {}
        if not self.cur.get("use_raw") and live.get("n", 0) >= self.cur.get("gate_min_n", 150) \
                and (live.get("hit") or 0) < self.cur.get("gate_min_hit", 0.5):
            return 0, {"KI": f"Markt gesperrt: trifft live nur {round((live.get('hit') or 0) * 100)} % "
                             f"von {live['n']}"}, fc
        p_up = fc["p_up_raw"] if self.cur.get("use_raw") else fc["p_up"]
        if self.cur.get("fast") and not self.cur.get("turbo") and not self.cur.get("use_raw"):
            # Fast-Modus: die staerkste Prognose auf 5-10 Minuten
            short = [x for x in fc.get("by_horizon") or [] if x["min"] <= 10]
            if short:
                p_up = max(short, key=lambda x: abs(x["p_up"] - 0.5))["p_up"]
        conf = max(p_up, 1 - p_up)
        side = (1 if p_up >= 0.5 else -1) if conf >= self.cur["min_conf"] else 0
        return side, {"KI": "LONG" if p_up >= 0.5 else "SHORT", "Sicherheit": round(conf, 3)}, fc

    def _levels(self, df, price, side, fc):
        p = self.cur
        if self.kind != "ki" or not fc:
            return levels(df, price, side, p)
        fees = price * (p["maker"] + p["taker"])
        if p.get("turbo"):         # Turbo: Stop aus der 1-Minuten-Schwankung (x1,5), mind. 2x Gebuehren
            r = max(price * float(fc.get("sd_5m_pct") or 0.05) / 100 * 1.5, price * 0.0005, 2 * fees)
        elif p.get("fast"):        # enge Stops aus der 5-Minuten-Schwankung
            r = max(price * float(fc.get("sd_5m_pct") or 0.1) / 100, price * 0.0008, 2 * fees)
        else:
            r = max(price * fc["band_pct"] / 100 * p["sl_band"], price * 0.0015, 2 * fees)
        tp_d = max(r * p["tp_r"], p["min_tp_fee_x"] * fees)
        # Ziel vor dem naechsten Widerstand (Long) / der naechsten Unterstuetzung (Short) - lohnt es nicht, kein Trade
        wall = fc.get("sr_up_pct") if side == 1 else fc.get("sr_dn_pct")
        if wall and p.get("size_mode") == "auto":
            room = price * wall / 100 * 0.9
            if room < tp_d:
                if room < 1.2 * r:
                    return None, None
                tp_d = room
        return price - side * r, price + side * tp_d

    def step(self, sym: str) -> None:
        sl = self.slots[sym]
        p = self.cur
        df, book, tick = self._market(sym)
        side, votes, fc = self._signal(sym, df, book)
        sl["signal"] = {"side": side, "votes": votes}
        now = time.time()
        if sl["state"] == "idle":
            if side == 0 or self.session.get("stopping") or len(self.trades) >= p["max_trades"]:
                return
            free_fn = getattr(self, "free_fn", None)
            if free_fn is not None and not free_fn(sym):
                sl["signal"]["votes"] = {**votes, "wartet": "andere Position in diesem Markt offen"}
                return
            from .hours import entry_allowed, recently_flat
            ok, why = entry_allowed(sym, None, {"hours": p.get("hours")})
            if not ok or recently_flat(df):
                sl["signal"]["votes"] = {**votes, "wartet": why or "Markt steht still (geschlossen?)"}
                return
            price = tick["bid"] if side == 1 else tick["ask"]
            stop, target = self._levels(df, price, side, fc)
            if stop is None:
                sl["signal"]["votes"] = {**votes, "wartet": "Widerstand/Unterstuetzung zu nah - Ziel lohnt nicht"}
                return
            lev = p["leverage"]
            if p.get("size_mode") == "auto":
                qty, lev = self._auto_size(sym, price, stop, votes.get("Sicherheit") or 0.55)
                if qty > 0:
                    self.broker.set_leverage(sym, lev)
            else:
                qty = self._qty(sym, price, stop)
            if qty <= 0:
                sl["signal"]["votes"] = {**votes, "wartet": "zu wenig freies Guthaben fuer die Mindestgroesse"}
                return
            full = qty
            if self.kind == "ki" and p.get("scale_in"):
                part = self._round(sym, qty * p["first_frac"])
                if part > 0 and part * price >= p["min_notional"]:
                    qty = part                                   # Teilkauf: erst ein Teil, Rest im Gewinn
            oid = self.broker.place_entry(sym, side, qty, price)
            sl.update(state="pending", oid=oid, side=side, qty=qty, full=full, price=price, sl=stop, tp=target,
                      placed=now, lev=lev, why_in={k: v for k, v in votes.items() if k != "wartet"},
                      h_min=(fc or {}).get("decision_min"))
            self._event(f"{sym.split(':')[0]} {'LONG' if side == 1 else 'SHORT'} Limit {price:.6g} x{lev} "
                        f"(Ziel {target:.6g}, Stop {stop:.6g})")
        elif sl["state"] == "pending":
            st, avg, filled = self.broker.entry_status(sym, sl["oid"])
            if st == "filled":
                stop, target = sl["sl"] + (avg - sl["price"]), sl["tp"] + (avg - sl["price"])
                sl.update(state="open", entry=avg, qty=filled, opened=now, sl=stop, tp=target, r0=abs(avg - stop),
                          best=avg, prot=self.broker.protect(sym, sl["side"], filled, stop, target))
                self._event(f"{sym.split(':')[0]} ausgefuehrt @ {avg:.6g}")
            elif st == "canceled" or now - sl["placed"] > p["entry_timeout_s"] or side == -sl["side"]:
                self.broker.cancel(sym, sl["oid"])
                st, avg, filled = self.broker.entry_status(sym, sl["oid"])
                if filled > 0:
                    sl.update(state="open", entry=avg, qty=filled, opened=now, r0=abs(avg - sl["sl"]), best=avg,
                              prot=self.broker.protect(sym, sl["side"], filled, sl["sl"], sl["tp"]))
                    self._event(f"{sym.split(':')[0]} teilweise ausgefuehrt ({filled:g}) - abgesichert")
                else:
                    sl.clear()
                    sl["state"] = "idle"
        elif sl["state"] == "open":
            from .hours import close_before_weekend
            done = self.broker.exit_status(sym, sl["side"], sl["qty"], sl["prot"])
            if not done and close_before_weekend(sym, None, {"hours": p.get("hours")}):
                px = self.broker.close(sym, sl["side"], sl["qty"], sl["prot"])
                self._book(sym, sl, px, "vor dem Wochenende geschlossen", maker=False)
                return
            if done:
                why, px, maker = done
                self._book(sym, sl, px, why, maker)
            elif p.get("max_hold_s") and now - sl["opened"] > p["max_hold_s"]:
                px = self.broker.close(sym, sl["side"], sl["qty"], sl["prot"])
                self._book(sym, sl, px, "Zeit-Limit", maker=False)
            elif self.kind == "ki":
                self._manage(sym, sl, tick, side, votes)

    def _manage(self, sym, sl, tick, side, votes) -> None:
        """KI-Autopilot fuehrt die Position: Einstand, Teilverkauf, Nachziehen, Ausstieg bei KI-Wende."""
        p = self.cur
        d = sl["side"]
        last = tick["last"]
        r0 = sl.get("r0") or abs(sl["entry"] - sl["sl"]) or sl["entry"] * 0.002
        sl["best"] = max(sl.get("best", last), last) if d == 1 else min(sl.get("best", last), last)
        gain = d * (last - sl["entry"]) / r0
        name = sym.split(":")[0]
        # KI dreht klar in die Gegenrichtung -> ganz verkaufen
        if side == -d:
            px = self.broker.close(sym, d, sl["qty"], sl["prot"])
            self._book(sym, sl, px, f"KI dreht ({votes.get('KI')} {round((votes.get('Sicherheit') or 0) * 100)} %)", False)
            return
        # Teilkauf: Rest nur im GEWINN nachkaufen, wenn die KI weiter in diese Richtung zeigt
        if p.get("scale_in") and not sl.get("added") and sl.get("full", 0) > sl["qty"] and gain >= p["add_at_r"] \
                and side == d:
            add = self._round(sym, sl["full"] - sl["qty"])
            if add > 0 and add * last >= p["min_notional"]:
                px = self.broker.add(sym, d, add)
                sl["entry"] = (sl["entry"] * sl["qty"] + px * add) / (sl["qty"] + add)
                sl["qty"] += add
                sl["added"] = True
                be0 = sl["entry"] - d * 0.5 * r0                     # Risiko nach dem Nachkauf begrenzen
                if d * (be0 - sl["sl"]) > 0 and d * (last - be0) > 0:
                    sl["sl"] = be0
                    sl["prot"]["sl"] = be0
                self.broker.resize(sym, d, sl["qty"], sl["prot"])
                self._event(f"{name} Teilkauf {add:g} @ {px:.6g} (im Gewinn, +{gain:.1f} R)")
                return
        # Teilverkauf bei +partial_r
        if not sl.get("partial_done") and p["partial_frac"] > 0 and gain >= p["partial_r"]:
            part = self._round(sym, sl["qty"] * p["partial_frac"])
            rest = sl["qty"] - part
            if part > 0 and part * last >= p["min_notional"] and rest * last >= p["min_notional"]:
                px = self.broker.close_part(sym, d, part, sl["prot"])
                self._book_part(sym, sl, px, part)
                sl["qty"] = rest
                sl["partial_done"] = True
                self.broker.resize(sym, d, rest, sl["prot"])
                self._event(f"{name} Teilverkauf {part:g} @ {px:.6g} (+{gain:.1f} R)")
        # Stop auf Einstand (inkl. Gebuehren) und danach nachziehen
        be = sl["entry"] * (1 + d * (p["maker"] + p["taker"]))
        want = None
        if gain >= p["be_r"] and d * (sl["sl"] - be) < 0:
            want = be
        if sl.get("partial_done"):
            trail = sl["best"] - d * p["trail_r"] * r0
            if d * (trail - (want if want is not None else sl["sl"])) > 0:
                want = trail
        if want is not None and d * (want - sl["sl"]) >= 0.1 * r0 and d * (last - want) > 0:
            self.broker.move_stop(sym, d, sl["qty"], sl["prot"], want)
            self._event(f"{name} Stop nachgezogen auf {want:.6g}")
            sl["sl"] = want

    def _round(self, sym, qty) -> float:
        try:
            return float(self.broker.c.amount_to_precision(sym, qty)) if hasattr(self.broker, "c") else qty
        except Exception:  # noqa: BLE001
            return 0.0

    def _book_part(self, sym, sl, px, part) -> None:
        p = self.cur
        gross = sl["side"] * (px - sl["entry"]) * part
        fees = sl["entry"] * part * p["maker"] + px * part * p["taker"]
        self.trades.append({"symbol": sym, "side": "long" if sl["side"] == 1 else "short", "entry": sl["entry"],
                            "exit": px, "qty": part, "gross": round(gross, 6), "fees": round(fees, 6),
                            "net": round(gross - fees, 6), "why": "Teilverkauf",
                            "secs": round(time.time() - sl.get("opened", time.time())), "time": int(time.time() * 1000),
                            "kind": self.kind, "label": self.session.get("label"), "why_in": sl.get("why_in"),
                            "h_min": sl.get("h_min")})
        self._save_trade(self.trades[-1])

    def _balance(self) -> tuple[float, float]:
        """(Gesamt, frei) - Bitget live, Simulation: Startkapital + Ergebnis - gebundene Margin."""
        if hasattr(self.broker, "balance"):
            try:
                return self.broker.balance()
            except Exception as e:  # noqa: BLE001
                log.debug("Guthaben: %s", e)
        total = self.session.get("equity0", 0) + self._net()
        used = sum((v.get("qty") or 0) * (v.get("entry") or v.get("price") or 0) / (v.get("lev") or self.cur["leverage"])
                   for v in self.slots.values() if v.get("state") in ("open", "pending"))
        return total, total - used

    def _auto_size(self, sym, price, stop, conf) -> tuple[float, int]:
        """KI entscheidet Groesse und Hebel: Risiko = risk_pct % vom Konto am Stop, je nach Sicherheit
        halb bis voll; Hebel nur so hoch wie noetig, hoechstens `leverage`, und die Liquidation liegt
        mindestens doppelt so weit weg wie der Stop."""
        p = self.cur
        total, free = self._balance()
        stop_pct = abs(price - stop) / price
        if stop_pct <= 0 or total <= 0:
            return 0.0, p["leverage"]
        factor = max(0.5, min(1.0, 0.5 + (float(conf) - 0.55) / 0.14))     # 55 % -> halb, ab ~62 % -> voll
        risk = total * p["risk_pct"] / 100 * factor
        notional = risk / stop_pct
        lev_safe = max(1, int(1 / (2 * stop_pct + 0.005)))                # Liquidation >= 2x so weit wie der Stop
        cap = max(1, min(int(p["leverage"]), lev_safe))
        budget = max(0.0, min(free * 0.95, total * p.get("max_margin_pct", 50) / 100))
        need = max(1, -(-notional // max(budget, 1e-9)))                  # aufgerundet
        lev = int(min(cap, need))
        notional = min(notional, budget * lev)
        qty = self._round(sym, notional / price)
        return (qty if qty * price >= p["min_notional"] else 0.0), lev

    def _qty(self, sym, price, stop=None) -> float:
        """Positionsgroesse nach Einstellung, immer begrenzt auf freies Guthaben (Puffer 5 %) und
        hoechstens max_margin_pct % des Kontos als Einsatz je Position."""
        p = self.cur
        total, free = self._balance()
        mode = p.get("size_mode", "usdt")
        if mode == "pct":
            margin = total * p["size_pct"] / 100
        elif mode == "risk" and stop is not None and abs(price - stop) > 0:
            margin = (total * p["risk_pct"] / 100) / abs(price - stop) * price / p["leverage"]
        else:
            margin = p["margin_usdt"]
        margin = min(margin, free * 0.95, total * p.get("max_margin_pct", 50) / 100)
        if margin <= 0:
            return 0.0
        qty = margin * p["leverage"] / price
        qty = self._round(sym, qty)
        return qty if qty * price >= p["min_notional"] else 0.0

    def _book(self, sym, sl, px, why, maker) -> None:
        p = self.cur
        gross = sl["side"] * (px - sl["entry"]) * sl["qty"]
        fees = sl["entry"] * sl["qty"] * p["maker"] + px * sl["qty"] * (p["maker"] if maker else p["taker"])
        t = {"symbol": sym, "side": "long" if sl["side"] == 1 else "short", "entry": sl["entry"], "exit": px,
             "qty": sl["qty"], "gross": round(gross, 6), "fees": round(fees, 6), "net": round(gross - fees, 6),
             "why": why, "secs": round(time.time() - sl.get("opened", time.time())), "time": int(time.time() * 1000),
             "kind": self.kind, "label": self.session.get("label"), "why_in": sl.get("why_in"), "h_min": sl.get("h_min"),
             "partial": bool(sl.get("partial_done")), "added": bool(sl.get("added"))}
        self.trades.append(t)
        self._save_trade(t)
        self._event(f"{sym.split(':')[0]} {why}: netto {t['net']:+.4f} USDT")
        sl.clear()
        sl["state"] = "idle"

    def _save_trade(self, t: dict) -> None:
        """Jeden Abschluss dauerhaft speichern (fuer python run.py handelsbericht)."""
        path = self.cur.get("log_path")
        if not path:
            return
        try:
            import json
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(t, default=str) + "\n")
        except OSError as e:
            log.debug("Trade speichern: %s", e)

    def _net(self) -> float:
        return sum(t["net"] for t in self.trades)

    def status(self) -> dict:
        tr = self.trades
        wins = [t for t in tr if t["net"] > 0]
        return {
            "active": self.active, "kind": self.kind, "name": self.name, "label": self.session.get("label"), "symbols": self.session.get("symbols", []),
            "stopping": bool(self.session.get("stopping")),
            "running_s": int(time.time() - self.session.get("start", time.time())) if self.active else 0,
            "params": self.session.get("params", {}), "stop_reason": self.session.get("stop_reason"),
            "trades": len(tr), "wins": len(wins), "hit": round(len(wins) / len(tr), 3) if tr else None,
            "gross": round(sum(t["gross"] for t in tr), 4), "fees": round(sum(t["fees"] for t in tr), 4),
            "net": round(self._net(), 4), "last_trades": tr[-12:][::-1], "events": self.events[-15:][::-1],
            "slots": {s: {"state": v.get("state"), "side": v.get("side"), "entry": v.get("entry") or v.get("price"),
                          "sl": v.get("sl"), "tp": v.get("tp"), "qty": v.get("qty"), "partial": v.get("partial_done"),
                          "lev": v.get("lev"), "added": v.get("added"),
                          "signal": (v.get("signal") or {}).get("votes")}
                      for s, v in self.slots.items()},
        }
