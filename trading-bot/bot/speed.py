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
    "minutes": 5, "margin_usdt": 5.0, "leverage": 10, "aggressiveness": 3,   # 1 = vorsichtig ... 4 = sehr schnell
    "entry_timeout_s": 20, "max_hold_s": 300, "max_trades": 60, "max_loss_pct": 5.0,
    "tp_atr": 1.0, "sl_atr": 1.0, "min_tp_fee_x": 3.0, "loop_s": 1.5,
    "maker": 0.0002, "taker": 0.0006, "slippage": 0.0003, "min_notional": 5.0,
}
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


class BitgetBroker:
    """Echtes Bitget-Konto (ccxt-Client mit mode_safe). Stop = an die Position gebundener Bitget-Stop."""

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
        return {"sl": sl, "tp": tp, "sl_id": sl_id, "tp_id": str(tp_o.get("id") or "")}

    def _pos_size(self, sym, side):
        want = "long" if side == 1 else "short"
        return sum(float(p.get("contracts") or 0) for p in self.c.fetch_positions([sym]) if p.get("side") == want)

    def exit_status(self, sym, side, qty, prot):
        if prot.get("tp_id"):
            o = self.c.fetch_order(prot["tp_id"], sym)
            if o.get("status") == "closed":
                self._cleanup(sym, prot, keep="tp")
                return "ziel", float(o.get("average") or prot["tp"]), True
        if self._pos_size(sym, side) < qty * 0.5:          # Stop ausgeloest (Position weg)
            self._cleanup(sym, prot, keep="sl")
            return "stop", prot["sl"] * (1 - side * self.p["slippage"]), False
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


class SpeedTrader:
    """Eine Speed-Trading-Sitzung (Start/Stopp aus der Oberflaeche)."""

    def __init__(self, data_fn, cfg: dict | None = None):
        """data_fn(sym) -> (1-Minuten-Kerzen, Orderbuch, {'last','bid','ask'}) - fuer Signal und Simulation."""
        self.data_fn = data_fn
        self.p = {**DEFAULTS, **(cfg or {})}
        self.lock = threading.RLock()
        self.active, self.thread = False, None
        self.session: dict = {}
        self.slots: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.events: list[str] = []
        self.market: dict[str, dict] = {}

    # --- Steuerung -------------------------------------------------------
    def start(self, symbols: list[str], broker, equity: float, label: str, **opts) -> str:
        with self.lock:
            if self.active:
                raise RuntimeError("Speed-Trading laeuft bereits")
            if not symbols:
                raise ValueError("Mindestens einen Markt waehlen")
            p = {**self.p, **{k: v for k, v in opts.items() if v is not None}}
            p["minutes"] = max(1, min(60, int(p["minutes"])))
            if p["margin_usdt"] * p["leverage"] < p["min_notional"]:
                raise ValueError(f"Einsatz x Hebel muss mind. {p['min_notional']:g} USDT sein (Bitget-Minimum)")
            if p["margin_usdt"] > equity:
                raise ValueError(f"Einsatz {p['margin_usdt']:g} USDT ist mehr als verfuegbar ({equity:.2f})")
            self.cur, self.broker = p, broker
            now = time.time()
            self.session = {"symbols": list(symbols), "start": now, "end": now + p["minutes"] * 60, "label": label,
                            "equity0": equity, "params": {k: p[k] for k in ("minutes", "margin_usdt", "leverage",
                                                                             "aggressiveness")}}
            self.slots = {s: {"state": "idle"} for s in symbols}
            self.trades, self.events = [], []
            self.active = True
            self._event(f"Start: {', '.join(x.split(':')[0] for x in symbols)} fuer {p['minutes']} min ({label})")
            self.thread = threading.Thread(target=self._run, daemon=True, name="speed")
            self.thread.start()
            return f"Speed-Trading laeuft {p['minutes']} Minuten ({label})"

    def stop(self, why: str = "von Hand beendet") -> str:
        with self.lock:
            if not self.active:
                return "Speed-Trading laeuft nicht"
            self.session["stop_reason"] = why
            self.session["end"] = 0      # Schleife raeumt auf und endet
        return "Speed-Trading wird beendet - offene Speed-Positionen werden geschlossen"

    def busy(self, sym: str) -> bool:
        return self.active and sym in self.session.get("symbols", [])

    # --- Ablauf ------------------------------------------------------------
    def _event(self, text: str) -> None:
        last = getattr(self, "_last_ev", ("", 0.0))
        if text == last[0] and time.time() - last[1] < 30:       # gleiche Meldung nicht jede Sekunde wiederholen
            return
        self._last_ev = (text, time.time())
        self.events.append(f"{time.strftime('%H:%M:%S')} {text}")
        self.events = self.events[-60:]
        log.info("Speed: %s", text)

    def _run(self) -> None:
        try:
            while time.time() < self.session["end"]:
                if self._net() <= -self.session["equity0"] * self.cur["max_loss_pct"] / 100:
                    self.session["stop_reason"] = f"Verlust-Limit {self.cur['max_loss_pct']:g} % erreicht"
                    break
                for sym in self.session["symbols"]:
                    try:
                        self.step(sym)
                    except Exception as e:  # noqa: BLE001 - ein Fehler darf die Sitzung nicht beenden
                        self._event(f"{sym.split(':')[0]} Fehler: {e}")
                time.sleep(self.cur["loop_s"])
        finally:
            self._wind_down()

    def _wind_down(self) -> None:
        for sym, sl in self.slots.items():
            try:
                if sl["state"] == "pending":
                    self.broker.cancel(sym, sl["oid"])
                    st, avg, filled = self.broker.entry_status(sym, sl["oid"])
                    if filled > 0:                        # kurz vor dem Storno ausgefuehrt
                        sl.update(state="open", entry=avg, qty=filled, opened=time.time(),
                                  prot=self.broker.protect(sym, sl["side"], filled, sl["sl"], sl["tp"]))
                if sl["state"] == "open":
                    px = self.broker.close(sym, sl["side"], sl["qty"], sl["prot"])
                    self._book(sym, sl, px, "Sitzungsende", maker=False)
            except Exception as e:  # noqa: BLE001
                self._event(f"{sym.split(':')[0]} beim Beenden: {e} - bitte im Konto pruefen!")
        self.active = False
        self._event(f"Ende ({self.session.get('stop_reason', 'Zeit abgelaufen')}): {len(self.trades)} Trades, "
                    f"netto {self._net():+.4f} USDT")

    def _market(self, sym):
        df, book, tick = self.data_fn(sym)
        self.market[sym] = tick
        return df, book, tick

    def step(self, sym: str) -> None:
        sl = self.slots[sym]
        p = self.cur
        df, book, tick = self._market(sym)
        side, strength, votes = micro_signal(df, book, p["aggressiveness"])
        sl["signal"] = {"side": side, "votes": votes}
        now = time.time()
        if sl["state"] == "idle":
            if side == 0 or len(self.trades) >= p["max_trades"] or now > self.session["end"] - 30:
                return
            price = tick["bid"] if side == 1 else tick["ask"]
            qty = self._qty(sym, price)
            if qty <= 0:
                return
            stop, target = levels(df, price, side, p)
            oid = self.broker.place_entry(sym, side, qty, price)
            sl.update(state="pending", oid=oid, side=side, qty=qty, price=price, sl=stop, tp=target, placed=now)
            self._event(f"{sym.split(':')[0]} {'LONG' if side == 1 else 'SHORT'} Limit {price:.6g} "
                        f"(Ziel {target:.6g}, Stop {stop:.6g})")
        elif sl["state"] == "pending":
            st, avg, filled = self.broker.entry_status(sym, sl["oid"])
            if st == "filled":
                stop, target = sl["sl"] + (avg - sl["price"]), sl["tp"] + (avg - sl["price"])
                sl.update(state="open", entry=avg, qty=filled, opened=now, sl=stop, tp=target,
                          prot=self.broker.protect(sym, sl["side"], filled, stop, target))
                self._event(f"{sym.split(':')[0]} ausgefuehrt @ {avg:.6g}")
            elif st == "canceled" or now - sl["placed"] > p["entry_timeout_s"] or side == -sl["side"]:
                self.broker.cancel(sym, sl["oid"])
                st, avg, filled = self.broker.entry_status(sym, sl["oid"])
                if filled > 0:
                    sl.update(state="open", entry=avg, qty=filled, opened=now,
                              prot=self.broker.protect(sym, sl["side"], filled, sl["sl"], sl["tp"]))
                    self._event(f"{sym.split(':')[0]} teilweise ausgefuehrt ({filled:g}) - abgesichert")
                else:
                    sl.clear()
                    sl["state"] = "idle"
        elif sl["state"] == "open":
            done = self.broker.exit_status(sym, sl["side"], sl["qty"], sl["prot"])
            if done:
                why, px, maker = done
                self._book(sym, sl, px, why, maker)
            elif now - sl["opened"] > p["max_hold_s"]:
                px = self.broker.close(sym, sl["side"], sl["qty"], sl["prot"])
                self._book(sym, sl, px, "Zeit-Limit", maker=False)

    def _qty(self, sym, price) -> float:
        p = self.cur
        qty = p["margin_usdt"] * p["leverage"] / price
        try:
            qty = float(self.broker.c.amount_to_precision(sym, qty)) if hasattr(self.broker, "c") else qty
        except Exception:  # noqa: BLE001 - zu klein fuer die Mindestmenge
            return 0.0
        return qty if qty * price >= p["min_notional"] else 0.0

    def _book(self, sym, sl, px, why, maker) -> None:
        p = self.cur
        gross = sl["side"] * (px - sl["entry"]) * sl["qty"]
        fees = sl["entry"] * sl["qty"] * p["maker"] + px * sl["qty"] * (p["maker"] if maker else p["taker"])
        t = {"symbol": sym, "side": "long" if sl["side"] == 1 else "short", "entry": sl["entry"], "exit": px,
             "qty": sl["qty"], "gross": round(gross, 6), "fees": round(fees, 6), "net": round(gross - fees, 6),
             "why": why, "secs": round(time.time() - sl.get("opened", time.time())), "time": int(time.time() * 1000)}
        self.trades.append(t)
        self._event(f"{sym.split(':')[0]} {why}: netto {t['net']:+.4f} USDT")
        sl.clear()
        sl["state"] = "idle"

    def _net(self) -> float:
        return sum(t["net"] for t in self.trades)

    def status(self) -> dict:
        tr = self.trades
        wins = [t for t in tr if t["net"] > 0]
        return {
            "active": self.active, "label": self.session.get("label"), "symbols": self.session.get("symbols", []),
            "left_s": max(0, int(self.session.get("end", 0) - time.time())) if self.active else 0,
            "params": self.session.get("params", {}), "stop_reason": self.session.get("stop_reason"),
            "trades": len(tr), "wins": len(wins), "hit": round(len(wins) / len(tr), 3) if tr else None,
            "gross": round(sum(t["gross"] for t in tr), 4), "fees": round(sum(t["fees"] for t in tr), 4),
            "net": round(self._net(), 4), "last_trades": tr[-12:][::-1], "events": self.events[-15:][::-1],
            "slots": {s: {"state": v.get("state"), "side": v.get("side"), "entry": v.get("entry") or v.get("price"),
                          "sl": v.get("sl"), "tp": v.get("tp"), "signal": (v.get("signal") or {}).get("votes")}
                      for s, v in self.slots.items()},
        }
