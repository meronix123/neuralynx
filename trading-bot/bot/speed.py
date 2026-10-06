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
import math
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
    "minutes": 0, "loop_s": 3.0, "entry_timeout_s": 60, "max_hold_s": 0, "max_trades": 1000,
    "min_conf": 0.56,          # Mindest-Sicherheit der KI fuer einen Einstieg
    "use_raw": False,          # Rohsignal auch ohne nachgewiesene Treffsicherheit (nur Simulation)
    "sl_band": 1.0,            # Stop = 1 x die 80-%-Schwankung der naechsten 30 min
    "tp_r": 2.0,               # Ziel = 2 x Risiko
    "be_r": 1.3,               # ab +1,3 R Stop auf Einstand (Studien: frueherer Einstand kostet Rendite)
    "partial_r": 1.5,          # ab +1,5 R ...
    "partial_frac": 0.5,       # ... die Haelfte verkaufen
    "trail_r": 1.0,            # danach Stop im Abstand 1 R hinter dem besten Kurs nachziehen
    "fast": False,             # Fast-Modus: 5-10-min-Prognose, enge Stops/Ziele, sehr schnelle Reaktion
    "scale_in": False,         # Teilkauf: erst einen Teil kaufen, Rest nur im GEWINN nachkaufen (nie im Verlust)
    "first_frac": 0.5,         #   erster Teil (Anteil der Positionsgroesse)
    "add_at_r": 0.5,           #   Rest bei +0,5 R, wenn die KI weiter dafuer ist
    "gate_min_n": 150,         # Markt-Sperre: ab so vielen live geprueften KI-Entscheidungen ...
    "gate_min_hit": 0.5,       # ... und Trefferquote darunter handelt der Autopilot dort nicht
    # Kosten-Schutz: nur Trades, bei denen nach Gebuehren rechnerisch etwas uebrig bleiben kann
    "cost_guard": True,
    "fee_r_x": 4.0,            # Stop mind. 4 x die Kosten eines Trades (Gebuehren + Schlupf) entfernt
    "chase": True,             # Limit nicht ausgefuehrt, Signal steht noch -> zum Marktpreis einsteigen
    "chase_max_r": 0.3,        #   ... wenn der Kurs hoechstens 0,3 R vom Limit weggelaufen ist (Limit liegt 0,15 R tiefer)
    "chase_avoid_marks": True, #   ... nicht um :00/:15/:30/:45 und Funding (Spread springt dort)
    "entry_pullback_r": 0.15,  # Limit 0,15 R unter dem Kurs (Long) / darueber (Short): Einstieg im kleinen Ruecksetzer
    # Netz-Modus (auf ausdruecklichen Wunsch, siehe README "Netz"): viele kleine Einheiten statt einer Position
    "grid": False,
    "grid_unit_margin": 0.10,  # Margin je Einheit in USDT (Einheit = Margin x Hebel, mind. Mindestposition 5 USDT)
    "grid_budget": 10.0,       # Margin je Markt hoechstens (USDT)
    "grid_step_pct": 0.6,      # naechste Einheit, wenn der Kurs um so viel Prozent gegen das Netz gelaufen ist (Test: 0,6)
    "grid_take_pct": 1.0,      # Einheit verkaufen ab so viel Prozent Gewinn (Test: 1,0; weite Stufen = weniger Gebuehren)
    "grid_limit": True,        # Nachkauf und Mitnahme als Limit-Orders (Maker, 1/3 der Gebuehren) statt Marktpreis
    "grid_max_loss_pct": 50,   # Netz-Stop: ganzes Netz schliessen, wenn der offene Verlust so viel % des Budgets erreicht
    "grid_keep_base": True,    # die erste Einheit (Hauptposition) bleibt offen, bis die KI dreht oder der Netz-Stop greift
    "grid_flip_conf": 0.53,    # Trendwende: KI neigt ab dieser Sicherheit zur Gegenseite -> ganzes Netz schliessen
    "grid_stale_x": 12.0,      # Netz ohne Gewinn nach 12 x Vorhersagezeit schliessen (nicht endlos aussitzen; Test: 12)
    "brain_veto": True,        # Supergehirn: klar gegen die Richtung (Gewicht >= brain_margin) -> kein Einstieg / Netz zu
    "brain_margin": 1.0,
    "fast_entry": True,        # Schnell-Einstieg: deutlich ueber der noetigen Sicherheit -> sofort zum Marktpreis
    "fast_entry_margin": 0.03, # ... ab noetige Sicherheit + 3 Prozentpunkte (Taker-Gebuehr wird mitgerechnet)
    "flow_confirm": 0.30,      # Einstieg nur, wenn der Taker-Fluss nicht KLAR dagegen laeuft (unter -0,30 = dagegen)
    "flip_extra": 0.03,        # Ausstieg bei KI-Wende nur mit 3 Punkten mehr Sicherheit als fuer den Einstieg
    "min_hold_frac": 0.5,      #   ... und fruehestens nach der halben Vorhersagezeit
    "learn_n": 12,             # Lernen aus eigenen Trades: ab 12 Abschluessen in einem Markt (letzte 3 Tage) ...
    "learn_pf": 0.7,           # ... und Gewinnfaktor darunter pausiert der Markt
    # Gewinne sichern
    "keep": 0.5,               # ab +1 R hoechstens die Haelfte des besten Gewinns wieder hergeben ...
    "keep_2r": 0.7,            # ... ab +2 R hoechstens 30 %
    "atr_k": 2.5,              # Nachzieh-Stop: bester Kurs minus 2,5 x ATR (5-Minuten-Kerzen) ...
    "atr_k_tight": 1.2,        # ... bei Erschoepfung (RSI-Extrem, Umkehrkerze, EMA-Wende) nur 1,2 x ATR
    "time_take_x": 2.0,        # nach 2 x Vorhersagezeit: Gewinn mitnehmen, wenn die KI nicht mehr dafuer ist
    "time_stale_x": 4.0,       # nach 4 x Vorhersagezeit: Position ohne klaren Gewinn schliessen
    "max_open_risk_pct": 3.0,  # alle offenen Positionen zusammen hoechstens 3 % des Kontos am Stop (gleiche Wette!)
    "unprotected_s": 60,       # Stop laesst sich so lange nicht setzen -> Position zur Sicherheit schliessen
    # Erwartungswert und Groesse (Ziel-vor-Stop-Modell der KI, Platt-kalibriert)
    "ev_min_r": 0.05,          # Trade nur, wenn p*Ziel - (1-p)*Stop - Kosten >= 0,05 R
    "kelly_frac": 0.25,        # Viertel-Kelly auf die kalibrierte Wahrscheinlichkeit (Rest: risk_pct als Deckel)
    "trail_adx": 20,           # ATR-Nachzieh-Stop nur im Trend (ADX ab 20) - im Seitwaertsmarkt kostet er Rendite
    # Schutz vor Boersen-Anomalien (Bitget-Vorfaelle: Flash-Wicks, zurueckgerollte Trades, Wartung)
    "spread_max": {"BTC": 0.0003, "ETH": 0.0003, "SOL": 0.0006, "XRP": 0.0006, "XAU": 0.0005, "XAG": 0.001},
    "move_sigma_x": 4.0,       # 1-Minuten-Bewegung ueber 4 x ihrer Schwankung -> 10 min keine neuen Einstiege
    "calm_s": 600,
    "mark_div_max": 0.003,     # Letztkurs weicht > 0,3 % vom Mark-Preis ab -> Anomalie, kein Einstieg
    "ki_err_n": 20,            # Not-Aus: ueber 20 Trades liegt der echte Fehler der KI-Sicherheit ...
    "ki_err_x": 0.15,          # ... um 0,15 (Log-Loss) ueber dem erwarteten -> Markt 12 h pausieren
}
# Rechnung (Simulation mit Bitget-Gebuehren 0,02 % Maker / 0,06 % Taker + Schlupf): ins Plus kommt die KI nur,
# wenn sie bei 20-30 min Vorhersage mind. ~57 % trifft, bei 10-15 min mind. ~60 %; bei 1-5 min frisst der
# Handel selbst bei 60 % Treffern mehr Gebuehren als er bringt. Diese Grenzen setzt der Kosten-Schutz durch.
def cost_need_conf(h_min: float | None) -> float | None:
    """Mindest-Sicherheit je Vorhersagezeit, damit nach Kosten etwas uebrig bleibt (None = lohnt nie)."""
    h = float(h_min or 30)
    if h < 10:
        return None
    return 0.60 if h < 20 else 0.57
FAST_AUTO = {"loop_s": 1.5, "entry_timeout_s": 20, "be_r": 1.0, "partial_r": 1.2, "trail_r": 0.8, "tp_r": 1.5}
# Hebel-Speed-KI (Turbo): eigenes 1-Minuten-Modell, noch engere Stops, Pruefung jede Sekunde
TURBO_AUTO = {"loop_s": 1.0, "entry_timeout_s": 15, "be_r": 1.0, "partial_r": 1.2, "trail_r": 0.8, "tp_r": 1.5,
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


def exit_view(df1m: pd.DataFrame, side: int) -> dict:
    """Indikatoren fuer das Gewinn-Sichern auf abgeschlossenen 5-Minuten-Kerzen: ATR (Schwankung),
    RSI (ueberkauft/-verkauft), EMA 9/21 (Trendwende) und Umkehrkerzen gegen die Position."""
    from .indicators import rsi
    from .patterns import candles
    out = {"atr": None, "rsi": None, "ema_against": False, "reversal": "", "tired": False, "adx": None}
    try:
        d = df1m.reset_index(drop=True)
        idx = pd.to_datetime(d["ts"], unit="ms", utc=True)
        m5 = d.set_index(idx).resample("5min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().iloc[:-1]
        if len(m5) < 15:
            return out
        a = float(atr(m5.reset_index(drop=True), 14).iloc[-1])
        out["atr"] = a if a == a and a > 0 else None
        c = m5["close"].astype(float)
        r = float(rsi(c, 14).iloc[-1])
        out["rsi"] = round(r, 1) if r == r else None
        out["ema_against"] = bool(side * (ema(c, 9).iloc[-1] - ema(c, 21).iloc[-1]) < 0)
        from .indicators import adx as _adx
        av = float(_adx(m5.reset_index(drop=True), 14).iloc[-1])
        out["adx"] = round(av, 1) if av == av else None
        cs = candles(*(m5[k].astype(float).to_numpy() for k in ("open", "high", "low", "close")))
        against = ("shooting", "engulf_dn", "evening", "crows") if side == 1 else ("hammer", "engulf_up", "morning", "soldiers")
        names = {"shooting": "Shooting Star", "engulf_dn": "Bearish Engulfing", "evening": "Evening Star",
                 "crows": "3 schwarze Kraehen", "hammer": "Hammer", "engulf_up": "Bullish Engulfing",
                 "morning": "Morning Star", "soldiers": "3 weisse Soldaten"}
        out["reversal"] = next((names[k] for k in against if cs[k][-1]), "")
        hot = out["rsi"] is not None and (out["rsi"] >= 72 if side == 1 else out["rsi"] <= 28)
        out["tired"] = bool(hot or out["reversal"] or out["ema_against"])
    except Exception as e:  # noqa: BLE001 - Indikatoren sind Hilfe, kein Muss
        log.debug("exit_view: %s", e)
    return out


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
        return int(lev)

    def resize(self, sym, side, qty, prot):
        pass

    def protect_stop(self, sym, side, qty, sl):
        return {"sl": sl, "tp": None}

    def place_exit(self, sym, side, qty, price):
        """Reduzierende Limit-Order (Netz: Einheit im Gewinn verkaufen) - fuellt, wenn der Kurs durchlaeuft."""
        oid = self._id()
        self.orders[oid] = {"sym": sym, "side": side, "qty": qty, "price": price, "status": "open", "exit": True}
        return oid

    def _settle(self, sym):
        last = self.market_fn(sym)["last"]
        for o in self.orders.values():
            if o["sym"] != sym or o["status"] != "open":
                continue
            if o.get("exit"):
                if (o["side"] == 1 and last >= o["price"]) or (o["side"] == -1 and last <= o["price"]):
                    o["status"] = "filled"
            elif (o["side"] == 1 and last < o["price"]) or (o["side"] == -1 and last > o["price"]):
                o["status"] = "filled"

    def open_orders(self, sym):
        self._settle(sym)
        return {k for k, o in self.orders.items() if o["sym"] == sym and o["status"] == "open"}

    def order_fill(self, sym, oid):
        self._settle(sym)
        o = self.orders[oid]
        return o["status"], o["price"]


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

    def protect_stop(self, sym, side, qty, sl):
        """Nur einen Stop fuer GENAU qty setzen (Netz-Modus: kein festes Ziel, Einheiten werden einzeln verkauft)."""
        from .exchange import place_size_tpsl
        sl_id = place_size_tpsl(self.c, sym, "long" if side == 1 else "short", "loss_plan", sl, qty, self.mm)
        return {"sl": sl, "tp": None, "sl_id": sl_id, "tp_id": "", "t": time.time(), "miss": 0}

    def pos_size(self, sym, side):
        return self._pos_size(sym, side)

    def place_exit(self, sym, side, qty, price):
        """Reduzierende Limit-Order (Maker) fuer genau qty."""
        o = self.c.create_order(sym, "limit", "sell" if side == 1 else "buy", qty,
                                float(self.c.price_to_precision(sym, price)),
                                {"reduceOnly": True, "marginMode": self.mm, "postOnly": True})
        return str(o.get("id") or "")

    def open_orders(self, sym):
        """Offene Auftraege eines Marktes (eine Anfrage je Runde, hoechstens alle 2 s)."""
        cache = self.__dict__.setdefault("_oo", {})
        hit = cache.get(sym)
        if hit and time.time() - hit[0] < 2:
            return hit[1]
        ids = {str(o.get("id")) for o in self.c.fetch_open_orders(sym)}
        cache[sym] = (time.time(), ids)
        return ids

    def order_fill(self, sym, oid):
        o = self.c.fetch_order(oid, sym)
        st = o.get("status")
        return ("filled" if st == "closed" else "canceled" if st in ("canceled", "rejected", "expired") else "open",
                float(o.get("average") or o.get("price") or 0))

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
            # Stop ausgeloest oder von aussen geschlossen (von Hand / "alle schliessen")? Am Kurs erkennen,
            # damit das Ergebnis stimmt
            try:
                last = float(self.c.fetch_ticker(sym)["last"])
            except Exception:  # noqa: BLE001
                last = prot["sl"]
            span = abs(prot["tp"] - prot["sl"]) or abs(prot["sl"]) * 0.01
            if abs(last - prot["sl"]) <= 0.25 * span:
                return "stop", prot["sl"] * (1 - side * self.p["slippage"]), False
            return "von aussen geschlossen", last, False
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
        self._bal = None
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
        """Hebel fuer diesen Markt setzen und den ECHTEN Hebel laut Bitget zurueckgeben (None = unbekannt).
        Kein Zwischenspeicher: Schnell-Orders oder der Bot koennen den Hebel im selben Markt aendern."""
        from .exchange import apply_leverage
        return apply_leverage(self.c, sym, int(lev), self.mm)

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
        self._bal = None
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
    def start(self, symbols: list[str], broker, equity: float, label: str, free_fn=None, scan_fn=None, **opts) -> str:
        """free_fn(sym) -> True, wenn im Konto in diesem Markt nichts anderes offen ist (sonst wird dort gewartet).
        scan_fn() -> aktuelle Markt-Liste des Scanners; die Sitzung tauscht Maerkte dann alle 10 min aus (Rotation)."""
        with self.lock:
            if self.active:
                raise RuntimeError(f"{self.name} laeuft bereits")
            if not symbols:
                raise ValueError("Mindestens einen Markt waehlen")
            p = {**self.p, **{k: v for k, v in opts.items() if v is not None}}
            if self.kind == "ki" and p.get("grid"):
                unit = float(p["grid_unit_margin"]) * float(p["leverage"])
                if unit < float(p["min_notional"]) * 0.99:
                    raise ValueError(f"Netz: Einheit {p['grid_unit_margin']:g} USDT x{p['leverage']} = {unit:.2f} USDT - "
                                     f"Bitget-Mindestposition ist {p['min_notional']:g} USDT (Hebel oder Einheit erhoehen)")
                if float(p["grid_take_pct"]) / 100 <= 2 * p["taker"]:
                    raise ValueError("Netz: Mitnahme muss ueber den Hin- und Rueckgebuehren liegen")
                if float(p["grid_budget"]) < float(p["grid_unit_margin"]):
                    raise ValueError("Netz: Budget kleiner als eine Einheit")
            if self.kind == "ki" and p.get("turbo"):
                p.update(TURBO_AUTO)
                if getattr(self, "turbo_fn", None) is None:
                    raise ValueError("Turbo-KI ist hier nicht verfuegbar")
            elif self.kind == "ki" and p.get("fast"):
                p.update(FAST_AUTO)
            minutes = int(p.get("minutes") or 0)              # 0 = laeuft, bis "Aus" gedrueckt wird
            if self.kind == "ki" and p.get("use_raw") and label != "Simulation":
                raise ValueError("Rohsignal (unbewaehrte KI) nur in der Simulation")
            free = p.pop("free", None)
            free = equity if free is None else float(free)
            if p.get("size_mode", "usdt") == "usdt":
                if p["margin_usdt"] * p["leverage"] < p["min_notional"]:
                    raise ValueError(f"Einsatz x Hebel muss mind. {p['min_notional']:g} USDT sein (Bitget-Minimum)")
                if p["margin_usdt"] > free:
                    raise ValueError(f"Einsatz {p['margin_usdt']:g} USDT ist mehr als verfuegbar ({free:.2f})")
            self.cur, self.broker, self.free_fn, self.scan_fn = p, broker, free_fn, scan_fn
            now = time.time()
            self.session = {"symbols": list(symbols), "start": now, "label": label, "stopping": False,
                            "end": now + minutes * 60 if minutes > 0 else float("inf"),
                            "equity0": equity, "params": {k: p.get(k) for k in ("minutes", "margin_usdt", "leverage",
                                                                                 "aggressiveness", "min_conf", "use_raw", "fast", "turbo",
                                                                                 "size_mode", "size_pct", "risk_pct",
                                                                                 "scale_in", "partial_frac")}}
            self.slots = {s: {"state": "idle"} for s in symbols}
            self.trades, self.events = [], []
            self.history = self._load_history(label)
            self.active = True
            self._event(f"Ein: {', '.join(x.split(':')[0] for x in symbols)} ({label})")
            self._persist()
            self.thread = threading.Thread(target=self._run, daemon=True, name=self.kind)
            self.thread.start()
            return f"{self.name} ist an ({label}) - laeuft, bis du Aus drueckst"

    # --- Sitzung ueber einen Neustart retten -----------------------------
    def _state_path(self):
        from pathlib import Path
        lp = self.cur.get("log_path") if getattr(self, "cur", None) else None
        return Path(lp).with_name(f"sitzung_{self.kind}_{Path(lp).stem.split('_')[-1]}.json") if lp else None

    def _persist(self) -> None:
        """Sitzung (Einstellungen, Positionen, Abschluesse) auf Platte - damit nach einem Neustart des Bots
        offene Positionen weiter gefuehrt werden statt vergessen zu sein."""
        path = self._state_path()
        if not path:
            return
        try:
            import json
            if not self.active:
                path.unlink(missing_ok=True)
                return
            snap = {"session": {k: v for k, v in self.session.items() if k != "end" or v != float("inf")},
                    "cur": {k: v for k, v in self.cur.items() if not callable(v)},
                    "slots": self.slots, "trades": self.trades[-200:], "events": self.events[-30:], "saved": time.time()}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snap, default=str), encoding="utf-8")
            tmp.replace(path)
        except (OSError, TypeError, ValueError) as e:
            log.debug("Sitzung speichern: %s", e)

    def saved_session(self, log_path: str | None) -> dict | None:
        """Gespeicherte Sitzung auf Platte (None = keine)."""
        import json
        from pathlib import Path
        if not log_path:
            return None
        path = Path(log_path).with_name(f"sitzung_{self.kind}_{Path(log_path).stem.split('_')[-1]}.json")
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, ValueError):
            return None

    def resume(self, snap: dict, broker, free_fn=None) -> str:
        """Nach einem Neustart weitermachen: offene Positionen auf Bitget pruefen und wieder fuehren."""
        with self.lock:
            if self.active:
                raise RuntimeError(f"{self.name} laeuft bereits")
            self.cur, self.broker, self.free_fn = dict(snap["cur"]), broker, free_fn
            self.session = dict(snap["session"])
            self.session.setdefault("end", float("inf"))
            self.session["stopping"] = False
            self.session.pop("close_now", None)
            self.trades = list(snap.get("trades") or [])
            self.events = list(snap.get("events") or [])
            self.history = self._load_history(self.session.get("label", ""))
            self.slots = {}
            kept, gone = [], []
            for sym, sl in (snap.get("slots") or {}).items():
                sl = dict(sl)
                if sl.get("state") == "open":
                    try:
                        still = self.broker._pos_size(sym, sl["side"]) >= sl["qty"] * 0.5 if hasattr(self.broker, "_pos_size") else True
                    except Exception as e:  # noqa: BLE001 - lieber weiter fuehren als vergessen
                        log.warning("Neustart %s: Position nicht pruefbar (%s)", sym, e)
                        still = True
                    if still:
                        kept.append(sym.split(":")[0])
                    else:
                        gone.append(sym.split(":")[0])
                        sl = {"state": "idle"}
                elif sl.get("state") == "pending":
                    try:
                        self.broker.cancel(sym, sl["oid"])
                    except Exception as e:  # noqa: BLE001
                        log.debug("Neustart Storno: %s", e)
                    sl = {"state": "idle"}
                self.slots[sym] = sl
            self.active = True
            self._event("Nach Neustart fortgesetzt" + (f" - offen: {', '.join(kept)}" if kept else "")
                        + (f" - inzwischen zu (Stop/Ziel auf Bitget): {', '.join(gone)}" if gone else ""))
            self._persist()
            self.thread = threading.Thread(target=self._run, daemon=True, name=self.kind)
            self.thread.start()
            return f"{self.name} nach Neustart fortgesetzt ({self.session.get('label')})"

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
                and (self.slots.get(sym) or {}).get("state") in ("pending", "open", "grid"))

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
                    elif self._day_net() <= -ses["equity0"] * self.cur["max_loss_pct"] / 100 \
                            and time.time() >= ses.get("pause_until", 0):
                        # Tagesverlust-Limit: heute keine neuen Trades mehr, offene Positionen laufen weiter,
                        # morgen (UTC) geht es weiter - die Sitzung bleibt an
                        ses["pause_until"] = (int(time.time()) // 86400 + 1) * 86400
                        self._cancel_pending()
                        self._event(f"Tagesverlust-Limit {self.cur['max_loss_pct']:g} % erreicht - Pause bis "
                                    f"{time.strftime('%d.%m. %H:%M', time.gmtime(ses['pause_until']))} UTC")
                if ses["stopping"]:
                    self._cancel_pending()
                    if not any(v.get("state") in ("open", "grid") for v in self.slots.values()):
                        break                                   # nichts mehr offen -> Sitzung zu Ende
                self._rotate()
                wait = self.cur["loop_s"]
                for sym in list(ses["symbols"]):
                    try:
                        self.step(sym)
                    except Exception as e:  # noqa: BLE001 - ein Fehler darf die Sitzung nicht beenden
                        self._event(f"{sym.split(':')[0]} Fehler: {e}")
                        wait = max(wait, self._backoff(e))
                self._persist_maybe()
                time.sleep(wait)
        finally:
            self._wind_down(close=bool(self.session.get("close_now")))

    def _rotate(self) -> None:
        """Markt-Scanner: alle 10 min neue Maerkte aufnehmen, nicht mehr gelistete ohne Position entfernen.
        Die fest eingestellten Maerkte bleiben immer."""
        fn = getattr(self, "scan_fn", None)
        if not fn or self.session.get("stopping") or time.time() - self.session.get("scan_at", 0) < 600:
            return
        self.session["scan_at"] = time.time()
        try:
            wanted = list(fn() or [])
        except Exception as e:  # noqa: BLE001
            log.debug("Scanner: %s", e)
            return
        fixed = self.session.setdefault("fixed", list(self.session["symbols"]))
        added, dropped = [], []
        for s in wanted:
            if s not in self.slots:
                self.slots[s] = {"state": "idle"}
                self.session["symbols"].append(s)
                added.append(s.split("/")[0])
        for s in list(self.slots):
            if s not in fixed and s not in wanted and self.slots[s].get("state") == "idle":
                del self.slots[s]
                self.session["symbols"].remove(s)
                dropped.append(s.split("/")[0])
        if added or dropped:
            self._event("Scanner: " + (f"neu {', '.join(added)}" if added else "") + (" · " if added and dropped else "")
                        + (f"raus {', '.join(dropped)}" if dropped else ""))
            self._persist()

    def _backoff(self, e: Exception) -> float:
        """Bitget bremst (Ratenlimit) oder Netz weg -> kurz Pause statt weiter hammern (sonst IP-Sperre)."""
        name = type(e).__name__
        text = str(e)
        if name in ("RateLimitExceeded", "DDoSProtection") or "429" in text or "too many" in text.lower():
            self._event("Bitget-Ratenlimit - 20 s Pause")
            return 20.0
        if name in ("NetworkError", "RequestTimeout", "ExchangeNotAvailable", "OnMaintenance") \
                or "timed out" in text.lower() or "connection" in text.lower():
            return 5.0
        return 0.0

    def _day_net(self) -> float:
        day0 = (int(time.time()) // 86400) * 86400 * 1000
        return sum(t["net"] for t in self.trades if (t.get("time") or 0) >= day0)

    def _persist_maybe(self) -> None:
        """Sitzung regelmaessig (alle 20 s) und bei jeder Aenderung einer Position speichern."""
        key = tuple((s, v.get("state"), v.get("sl"), v.get("qty")) for s, v in self.slots.items())
        last = getattr(self, "_persisted", (None, 0.0))
        if key != last[0] or time.time() - last[1] > 20:
            self._persist()
            self._persisted = (key, time.time())

    def _open_risk(self) -> float:
        """Summe des Verlusts am Stop aller offenen/wartenden Positionen (USDT)."""
        out = 0.0
        for v in self.slots.values():
            if v.get("state") == "grid":
                out += float(self.cur.get("grid_budget", 0)) * float(self.cur.get("grid_max_loss_pct", 100)) / 100
            elif v.get("state") in ("open", "pending") and v.get("sl") is not None:
                px = v.get("entry") or v.get("price") or 0
                out += (v.get("qty") or 0) * abs(px - v["sl"])
        return out

    def _cancel_pending(self) -> None:
        for sym, sl in self.slots.items():
            if sl.get("state") != "pending":
                continue
            try:
                self.broker.cancel(sym, sl["oid"])
                st, avg, filled = self.broker.entry_status(sym, sl["oid"])
                if filled > 0:                            # kurz vor dem Storno ausgefuehrt -> absichern
                    sl.update(state="open", entry=avg, qty=filled, opened=time.time(), r0=abs(avg - sl["sl"]), best=avg)
                    self._protect(sym, sl)
                else:
                    sl.clear()
                    sl["state"] = "idle"
            except Exception as e:  # noqa: BLE001
                self._event(f"{sym.split(':')[0]} Storno: {e}")

    def _wind_down(self, close: bool = False) -> None:
        self._cancel_pending()
        for sym, sl in self.slots.items():
            if sl.get("state") not in ("open", "grid"):
                continue
            try:
                if close and sl["state"] == "grid":
                    self._grid_close(sym, sl, "von Hand geschlossen")
                elif close:
                    px = self.broker.close(sym, sl["side"], sl["qty"], sl["prot"])
                    self._book(sym, sl, px, "von Hand geschlossen", maker=False)
                else:
                    self._event(f"{sym.split(':')[0]}: Position bleibt offen (Stop und Ziel liegen auf Bitget)")
            except Exception as e:  # noqa: BLE001
                self._event(f"{sym.split(':')[0]} beim Beenden: {e} - bitte im Konto pruefen!")
        self.active = False
        self._event(f"Aus ({self.session.get('stop_reason', '-')}): {len(self.trades)} Abschluesse, "
                    f"netto {self._net():+.4f} USDT")
        self._persist()

    def _market(self, sym):
        df, book, tick = self.data_fn(sym)
        self.market[sym] = tick
        return df, book, tick

    def _signal(self, sym, df, book):
        """-> (Richtung, Einzelwerte, KI-Prognose oder None)."""
        if self.kind != "ki":
            side, _, votes = micro_signal(df, book, self.cur["aggressiveness"])
            return side, votes, None
        if self.cur.get("turbo"):
            fc = self._blend(sym)                   # Supergehirn: 1-Minuten-Modell + 5-Minuten-KI
        else:
            fc = self.forecast_fn(sym) if self.forecast_fn else None
        if not fc or not fc.get("ok"):
            return 0, {"KI": (fc or {}).get("msg", "keine Prognose")}, None
        if fc.get("veto"):
            return 0, {"KI": fc["decision"], "Sicherheit": fc["confidence"], "wartet": fc["veto"]}, fc
        live = fc.get("live") or {}
        if not self.cur.get("use_raw") and live.get("n", 0) >= self.cur.get("gate_min_n", 150) \
                and (live.get("hit") or 0) < self.cur.get("gate_min_hit", 0.5):
            return 0, {"KI": f"Markt gesperrt: trifft live nur {round((live.get('hit') or 0) * 100)} % "
                             f"von {live['n']}"}, fc
        p_up = fc["p_up_raw"] if self.cur.get("use_raw") else fc["p_up"]
        h_min = fc.get("decision_min")
        if self.cur.get("fast") and not self.cur.get("turbo") and not self.cur.get("use_raw"):
            # Fast-Modus: die staerkste Prognose auf 5-10 Minuten
            short = [x for x in fc.get("by_horizon") or [] if x["min"] <= 10]
            if short:
                best = max(short, key=lambda x: abs(x["p_up"] - 0.5))
                p_up, h_min = best["p_up"], best["min"]
        conf = max(p_up, 1 - p_up)
        votes = {"KI": ("LONG" if p_up >= 0.5 else "SHORT") + (f" (KI {fc['source']})" if fc.get("source") else ""),
                 "Sicherheit": round(conf, 3)}
        self._has_tb = bool((fc.get("tb") or {}).get("long"))
        need = self._need_conf(sym, h_min, live)
        if need is None:
            return 0, {**votes, "wartet": f"Kosten-Schutz: {h_min:g}-min-Prognose bringt nach Gebuehren nichts "
                                          f"(erst ab nachgewiesenen 60 % Treffern)"}, fc
        blocked = self._learned_block(sym)
        if blocked:
            return 0, {**votes, "wartet": blocked}, fc
        side = (1 if p_up >= 0.5 else -1) if conf >= need else 0
        # Schnell-Einstieg: deutlich ueber der noetigen Sicherheit -> sofort zum Marktpreis statt auf den
        # Ruecksetzer zu warten (die besten Signale laufen sonst weg); die Taker-Gebuehr wird unten mitgerechnet
        quick = side != 0 and bool(self.cur.get("fast_entry", True)) and conf >= need + float(self.cur.get("fast_entry_margin", 0.03))
        if side == 0 and conf >= self.cur["min_conf"]:
            votes["wartet"] = f"Kosten-Schutz: braucht {round(need * 100)} % Sicherheit"
        elif side == 0:
            q = next((x for x in fc.get("quality") or [] if x.get("min") == fc.get("decision_min")), None)
            if conf < 0.505:
                votes["wartet"] = ("KI sieht gerade keinen Vorteil gegenueber Zufall"
                                   + (f" (Test: {round(q['hit'] * 100)} % Treffer, Vorsprung z={q['z']:+.1f})" if q else ""))
            else:
                votes["wartet"] = f"KI nur {conf * 100:.1f} % sicher - braucht {need * 100:.0f} %"
        # Orderfluss-Bestaetigung: aggressive Kaeufe/Verkaeufe der letzten Trades duerfen nicht klar dagegen sein
        flow = (fc.get("flow_now") or {}).get("taker")
        thr = self.cur.get("flow_confirm")
        if side != 0 and flow is not None and thr and side * float(flow) < -float(thr):
            votes["Fluss"] = round(float(flow), 2)
            votes["wartet"] = f"Orderfluss dagegen (Taker {float(flow):+.2f}) - wartet auf Bestaetigung"
            side = 0
        # Ziel-vor-Stop-Modell: Erwartungswert nach Kosten in R muss positiv sein
        tb = fc.get("tb") or {}
        head = tb.get("long" if p_up >= 0.5 else "short")
        guard = self.cur.get("cost_guard", True)                       # ohne Kosten-Schutz: bewusst freigeschaltet
        if side != 0 and head and not self.cur.get("use_raw") and guard:
            p = float(head["p"])
            tp_r = float(tb.get("tp_r") or self.cur["tp_r"])
            r_frac = max(float(tb.get("r_pct") or 0.3) / 100, 1e-4)
            # Kosten je Trade in R: Einstieg Maker; Ziel = Maker-Limit, Stop = Taker + Schlupf
            cost_r = (self.cur["maker"] + p * self.cur["maker"] + (1 - p) * (self.cur["taker"] + self.cur["slippage"])) / r_frac
            ev = p * tp_r - (1 - p) - cost_r
            votes["Kosten"] = f"{cost_r:.2f} R"
            votes["Ziel vor Stop"] = round(p, 3)
            votes["EV"] = round(ev, 2)
            if ev < self.cur.get("ev_min_r", 0.05):
                side = 0
                votes["wartet"] = f"Erwartungswert nach Kosten {ev:+.2f} R (Ziel vor Stop {round(p * 100)} %)"
            elif quick and ev - (self.cur["taker"] - self.cur["maker"]) / r_frac < self.cur.get("ev_min_r", 0.05):
                quick = False                                              # Taker-Gebuehr wuerde den Vorteil auffressen
        if quick and side != 0 and flow is not None and side * float(flow) < 0:
            quick = False                                                  # Orderfluss nicht dafuer: lieber im Ruecksetzer
        if quick and side != 0:
            votes["Schnell"] = "ja"
        return side, votes, fc

    BLEND_MAX_MIN = 15          # 5-Minuten-KI im Turbo: nur ihre kurzen Vorhersagezeiten (bis 15 min)
    BLEND_VETO = 0.03           # beide KIs klar uneins (>= 53 % gegeneinander) -> warten

    def _blend(self, sym) -> dict | None:
        """Turbo = Supergehirn: das 1-Minuten-Modell UND die 5-Minuten-KI (mit viel mehr Geschichte, laufend
        innerhalb der Kerze bewertet) schauen auf den Markt. Es zaehlt die staerkere ehrliche (kalibrierte,
        gedeckelte) Sicherheit; sind beide klar uneins, wird gewartet. Jede Quelle bringt ihre eigene Live-
        Bilanz und Kostenpruefung mit, so bleibt die Sperre je Quelle ehrlich."""
        fn1, fn5 = getattr(self, "turbo_fn", None), self.forecast_fn
        fc1 = fn1(sym) if fn1 else None
        try:
            fc5 = fn5(sym) if fn5 else None
        except Exception as e:  # noqa: BLE001 - 5-Minuten-KI ist Zusatz
            log.debug("Turbo-Blend %s: %s", sym, e)
            fc5 = None
        raw = self.cur.get("use_raw")

        def pick(fc, max_min):
            if not fc or not fc.get("ok"):
                return None
            p, h = float(fc["p_up_raw"] if raw else fc["p_up"]), fc.get("decision_min")
            if max_min and not raw:
                short = [x for x in fc.get("by_horizon") or [] if x["min"] <= max_min]
                if short:
                    best = max(short, key=lambda x: abs(x["p_up"] - 0.5))
                    p, h = float(best["p_up"]), best["min"]
                elif (h or 99) > max_min:
                    return None
            return abs(p - 0.5), p, h, fc
        c1, c5 = pick(fc1, None), pick(fc5, self.BLEND_MAX_MIN)
        cands = [c for c in (c1, c5) if c]
        if not cands:
            return fc1 or fc5
        best = max(cands, key=lambda c: c[0])
        src = "1 min" if best is c1 else "5 min"
        out = dict(best[3])
        out.update({"p_up": round(best[1], 3), "decision_min": best[2], "source": src,
                    "decision": "LONG" if best[1] >= 0.5 else "SHORT", "confidence": round(max(best[1], 1 - best[1]), 3)})
        if best[2] != best[3].get("decision_min"):
            out["p_up_raw"] = out["p_up"]
        if src == "5 min":
            out["tb"] = {}                          # Ziel-vor-Stop der 5-min-KI passt nicht zum engen Turbo-Stop
        other = next((c for c in cands if c is not best), None)
        if other and other[0] >= self.BLEND_VETO and (other[1] - 0.5) * (best[1] - 0.5) < 0:
            o_src = "5 min" if src == "1 min" else "1 min"
            out["veto"] = (f"KI 1 min und KI 5 min uneins ({src}: {out['decision']} {round(out['confidence'] * 100)} %, "
                           f"{o_src}: {'LONG' if other[1] >= 0.5 else 'SHORT'} {round(max(other[1], 1 - other[1]) * 100)} %)")
        return out

    def _real(self) -> bool:
        return isinstance(self.broker, BitgetBroker)

    def _need_conf(self, sym, h_min, live) -> float | None:
        """Benoetigte Sicherheit: eingestellte Mindest-Sicherheit, mit Kosten-Schutz mindestens so viel, wie
        bei dieser Vorhersagezeit nach Gebuehren noetig ist. Auf dem Konto ist der Kosten-Schutz immer an."""
        p = self.cur
        need = p["min_conf"]
        if not p.get("cost_guard", True) or p.get("use_raw"):
            return need
        if self._has_tb:                        # Ziel-vor-Stop-Modell rechnet die Kosten je Trade selbst (EV-Gate)
            return need
        floor = cost_need_conf(h_min)
        if floor is None:                       # 1-5 min ohne EV-Modell: nur mit Nachweis im Live-Test
            if live.get("n", 0) >= 200 and (live.get("hit") or 0) >= 0.60:
                floor = 0.60
            else:
                return None
        if live.get("n", 0) >= 100 and (live.get("hit") or 0) >= 0.56:
            floor -= 0.01                       # Markt hat sich live bewaehrt -> etwas oefter handeln
        return max(need, floor)

    def _learned_block(self, sym) -> str:
        """Lernen aus den eigenen Abschluessen: Markt pausieren, solange er in den letzten 3 Tagen
        dauerhaft Geld verliert (Gewinnfaktor unter learn_pf bei mind. learn_n Abschluessen)."""
        p = self.cur
        if not p.get("learn_n"):
            return ""
        since = (time.time() - 3 * 86400) * 1000
        rows = [t for t in list(getattr(self, "history", [])) + self.trades
                if t.get("symbol") == sym and (t.get("time") or 0) >= since and t.get("why") != "Teilverkauf"]
        # Not-Aus: stimmt die KI-Sicherheit hier noch? Echter Log-Loss der Trades gegen den erwarteten
        # (bei richtig kalibrierter Sicherheit sind beide gleich); deutlich schlechter -> 12 h Pause
        n_err = p.get("ki_err_n", 20)
        recent = [t for t in rows if (t.get("why_in") or {}).get("Sicherheit")][-n_err:]
        if len(recent) >= n_err:
            exc = 0.0
            for t in recent:
                q = min(0.99, max(0.01, float(t["why_in"]["Sicherheit"])))
                hit = t["net"] > 0
                exc += (-math.log(q if hit else 1 - q)) - (-(q * math.log(q) + (1 - q) * math.log(1 - q)))
            if exc / len(recent) > p.get("ki_err_x", 0.15) and (recent[-1].get("time") or 0) >= (time.time() - 12 * 3600) * 1000:
                return f"gelernt: KI-Sicherheit stimmt hier nicht mehr (Fehler +{exc / len(recent):.2f}) - 12 h Pause"
        if len(rows) < p["learn_n"]:
            return ""
        win = sum(t["net"] for t in rows if t["net"] > 0)
        loss = -sum(t["net"] for t in rows if t["net"] < 0)
        pf = win / loss if loss > 0 else 9.0
        if pf < p["learn_pf"]:
            return f"gelernt: hier zuletzt {len(rows)} Trades mit Verlust (Gewinnfaktor {pf:.2f}) - pausiert"
        return ""

    def _levels(self, df, price, side, fc):
        p = self.cur
        if self.kind != "ki" or not fc:
            return levels(df, price, side, p)
        fees = price * (p["maker"] + p["taker"])
        # Kosten-Schutz: Stop so weit, dass Gebuehren + Schlupf hoechstens ~1/4 des Risikos ausmachen
        floor = p.get("fee_r_x", 4.0) * (fees + price * p["slippage"]) \
            if p.get("cost_guard", True) and not p.get("use_raw") else 0.0
        if p.get("turbo"):         # Turbo: Stop aus der 1-Minuten-Schwankung (x1,5), mind. 2x Gebuehren
            r = max(price * float(fc.get("sd_5m_pct") or 0.05) / 100 * 1.5, price * 0.0005, 2 * fees)
        elif p.get("fast"):        # enge Stops aus der 5-Minuten-Schwankung
            r = max(price * float(fc.get("sd_5m_pct") or 0.1) / 100, price * 0.0008, 2 * fees)
        else:
            r = max(price * fc["band_pct"] / 100 * p["sl_band"], price * 0.0015, 2 * fees)
        r = max(r, floor)
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
        if sl["state"] == "grid":
            self._grid(sym, sl, side, votes, tick, now, fc, df)
            return
        if sl["state"] == "idle":
            if side == 0 or self.session.get("stopping") or len(self.trades) >= p["max_trades"]:
                return
            if time.time() < self.session.get("pause_until", 0):
                sl["signal"]["votes"] = {**votes, "wartet": "Tagesverlust-Limit erreicht - Pause bis morgen (UTC)"}
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
            blocked = self._anomaly(sym, sl, df, tick, now)
            if blocked:
                sl["signal"]["votes"] = {**votes, "wartet": blocked}
                return
            if self.kind == "ki" and p.get("grid"):
                self._grid_open(sym, sl, side, votes, fc, tick, now)
                return
            veto = self._brain_against(sym, side) if self.kind == "ki" else ""
            if veto:
                sl["signal"]["votes"] = {**votes, "wartet": veto}
                return
            price = tick["bid"] if side == 1 else tick["ask"]
            stop, target = self._levels(df, price, side, fc)
            if stop is None:
                sl["signal"]["votes"] = {**votes, "wartet": "Widerstand/Unterstuetzung zu nah - Ziel lohnt nicht"}
                return
            quick = self.kind == "ki" and votes.get("Schnell") == "ja" and not self.session.get("stopping")
            if self.kind == "ki" and p.get("entry_pullback_r") and not quick:
                # Einstieg im kleinen Ruecksetzer (Limit etwas unter/ueber dem Kurs); laeuft der Kurs weg, fasst
                # der Markt-Einstieg nach (chase). Stop/Ziel wandern mit dem tatsaechlichen Einstieg.
                r = abs(price - stop)
                price = price - side * p["entry_pullback_r"] * r
                stop, target = stop - side * p["entry_pullback_r"] * r, target - side * p["entry_pullback_r"] * r
            lev = p["leverage"]
            if p.get("size_mode") == "auto":
                qty, lev = self._auto_size(sym, price, stop, votes.get("Sicherheit") or 0.55, votes.get("Ziel vor Stop"))
                if qty > 0:
                    real = self.broker.set_leverage(sym, lev)
                    if real and real != lev:
                        # Bitget hat einen anderen Hebel: Groesse und Sicherheit damit neu pruefen
                        stop_pct = abs(price - stop) / price
                        if 1 / real < 2 * stop_pct + 0.005:
                            sl["signal"]["votes"] = {**votes, "wartet": f"Hebel auf Bitget x{real} - Liquidation zu nah am Stop"}
                            return
                        if real < lev:
                            qty = self._round(sym, qty * real / lev)
                            if qty * price < p["min_notional"]:
                                qty = 0.0
                        lev = real
            else:
                qty = self._qty(sym, price, stop)
                real = self.broker.set_leverage(sym, lev) if qty > 0 else None
                if real and real != lev:                  # Bitget-Hebel weicht ab -> Einsatz (Margin) beibehalten
                    qty = self._round(sym, qty * real / lev)
                    if qty * price < p["min_notional"]:
                        qty = 0.0
                    lev = real
            if qty <= 0:
                sl["signal"]["votes"] = {**votes, "wartet": "zu wenig freies Guthaben fuer die Mindestgroesse"}
                return
            total = self._balance()[0]
            cap = p.get("max_open_risk_pct") or 0
            if self.kind == "ki" and cap and total > 0 \
                    and self._open_risk() + qty * abs(price - stop) > total * cap / 100:
                sl["signal"]["votes"] = {**votes, "wartet": f"Gesamt-Risiko offener Positionen waere ueber {cap:g} % des Kontos"}
                return
            full = qty
            if self.kind == "ki" and p.get("scale_in"):
                part = self._round(sym, qty * p["first_frac"])
                if part > 0 and part * price >= p["min_notional"]:
                    qty = part                                   # Teilkauf: erst ein Teil, Rest im Gewinn
            if quick:
                # Schnell-Einstieg: zum Marktpreis, Stop/Ziel wandern mit dem tatsaechlichen Einstieg; erst den
                # Zustand merken, dann absichern (schlaegt der Stop fehl, wird nie ein zweites Mal gekauft)
                avg = self.broker.add(sym, side, qty)
                shift = avg - price
                sl.update(state="open", side=side, qty=qty, full=full, price=avg, entry=avg, opened=now, placed=now,
                          sl=stop + shift, tp=target + shift, r0=abs(avg - stop - shift), best=avg, lev=lev, taker_in=True,
                          why_in={k: v for k, v in votes.items() if k != "wartet"}, h_min=(fc or {}).get("decision_min"))
                self._protect(sym, sl)
                self._event(f"{sym.split(':')[0]} {'LONG' if side == 1 else 'SHORT'} Schnell-Einstieg @ {avg:.6g} x{lev} "
                            f"(Ziel {sl['tp']:.6g}, Stop {sl['sl']:.6g})")
                return
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
                sl.update(state="open", entry=avg, qty=filled, opened=now, sl=stop, tp=target, r0=abs(avg - stop), best=avg)
                self._protect(sym, sl)
                self._event(f"{sym.split(':')[0]} ausgefuehrt @ {avg:.6g}")
            elif st == "canceled" or now - sl["placed"] > p["entry_timeout_s"] or side == -sl["side"]:
                self.broker.cancel(sym, sl["oid"])
                st, avg, filled = self.broker.entry_status(sym, sl["oid"])
                if filled > 0:
                    sl.update(state="open", entry=avg, qty=filled, opened=now, r0=abs(avg - sl["sl"]), best=avg)
                    self._protect(sym, sl)
                    self._event(f"{sym.split(':')[0]} teilweise ausgefuehrt ({filled:g}) - abgesichert")
                elif self._chase(sym, sl, side, tick, now):
                    pass
                else:
                    sl.clear()
                    sl["state"] = "idle"
        elif sl["state"] == "open":
            from .hours import close_before_weekend
            if sl.get("unprotected"):
                if not self._protect(sym, sl) and now - sl["unprotected"] > p.get("unprotected_s", 60):
                    px = self.broker.close(sym, sl["side"], sl["qty"], sl["prot"])
                    self._book(sym, sl, px, "ohne Stop nicht haltbar - geschlossen", maker=False)
                    return
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
                self._manage(sym, sl, tick, side, votes, df)

    def _brain_against(self, sym, side) -> str:
        """Supergehirn (alle Blickwinkel: KI 5 min, Turbo, Ziel-vor-Stop, Bot-Strategie, Orderfluss, Muster, Makro,
        Zeitebenen) klar gegen diese Richtung? -> Grund, sonst ''."""
        fn = getattr(self, "brain_fn", None)
        if not fn or not self.cur.get("brain_veto", True) or side == 0:
            return ""
        try:
            b = fn(sym)
        except Exception as e:  # noqa: BLE001 - Supergehirn ist Zusatz
            log.debug("Supergehirn %s: %s", sym, e)
            return ""
        up, dn = float(b.get("up") or 0), float(b.get("down") or 0)
        against = (dn - up) if side == 1 else (up - dn)
        if against >= float(self.cur.get("brain_margin", 1.0)):
            return f"Supergehirn dagegen ({up:g} Long / {dn:g} Short)"
        return ""

    # ------------------------------------------------------------------ Netz-Modus (Mini-Einheiten)
    def _grid_unit_qty(self, sym, price) -> float:
        p = self.cur
        notional = max(float(p["grid_unit_margin"]) * float(p["leverage"]), float(p["min_notional"]) * 1.01)
        qty = self._round(sym, notional / price)
        if qty * price < p["min_notional"]:                     # Rundung nach unten: eine Stufe hoeher
            qty = self._round(sym, notional * 1.1 / price)
        return qty if qty * price >= p["min_notional"] else 0.0

    def _grid_stop_price(self, sl) -> float:
        p = self.cur
        budget_loss = float(p["grid_budget"]) * float(p["grid_max_loss_pct"]) / 100
        return sl["entry"] - sl["side"] * budget_loss / max(sl["qty"], 1e-12)

    def _grid_open(self, sym, sl, side, votes, fc, tick, now) -> None:
        """Erste Einheit (Hauptposition) zum Marktpreis; Netz-Stop auf Bitget fuer die ganze Menge."""
        p = self.cur
        veto = self._brain_against(sym, side)
        if veto:
            sl["signal"]["votes"] = {**votes, "wartet": veto}
            return
        price = tick["ask"] if side == 1 else tick["bid"]
        qty = self._grid_unit_qty(sym, price)
        if qty <= 0:
            sl["signal"]["votes"] = {**votes, "wartet": "Einheit zu klein fuer die Mindestposition"}
            return
        lev = int(p["leverage"])
        real = self.broker.set_leverage(sym, lev)
        if real and real != lev:
            lev = real
        total = self._balance()[0]
        cap = p.get("max_open_risk_pct") or 0
        if cap and total > 0 and self._open_risk() + float(p["grid_budget"]) * float(p["grid_max_loss_pct"]) / 100 > total * cap / 100:
            sl["signal"]["votes"] = {**votes, "wartet": f"Gesamt-Risiko offener Positionen waere ueber {cap:g} % des Kontos"}
            return
        avg = self.broker.add(sym, side, qty)
        sl.update(state="grid", side=side, qty=qty, entry=avg, price=avg, opened=now, lev=lev, taker_in=True,
                  units=[{"entry": avg, "qty": qty, "time": now}], last_add=avg, sl=None, tp=None,
                  why_in={k: v for k, v in votes.items() if k != "wartet"}, h_min=(fc or {}).get("decision_min"))
        sl["sl"] = self._grid_stop_price(sl)
        try:
            sl["prot"] = self.broker.protect_stop(sym, side, qty, sl["sl"])
        except Exception as e:  # noqa: BLE001 - Stop bleibt in Software (jede Runde geprueft)
            log.warning("Netz-Stop %s nicht gesetzt: %s", sym, e)
            sl["prot"] = {"sl": sl["sl"], "tp": None}
        self._event(f"{sym.split(':')[0]} NETZ {'LONG' if side == 1 else 'SHORT'}: Hauptposition @ {avg:.6g} x{lev} "
                    f"({qty:g}, Netz-Stop {sl['sl']:.6g})")
        if p.get("grid_limit", True):
            self._grid_place_add(sym, sl, avg)

    def _grid_place_add(self, sym, sl, ref) -> None:
        """Naechste Einheit als Limit-Order eine Stufe gegen das Netz legen (Maker)."""
        p = self.cur
        d = sl["side"]
        used = sum(u["entry"] * u["qty"] for u in sl["units"]) / max(sl["lev"], 1)
        if used + float(p["grid_unit_margin"]) > float(p["grid_budget"]) * 1.05 or sl.get("add"):
            return
        price = ref * (1 - d * float(p["grid_step_pct"]) / 100)
        qty = self._grid_unit_qty(sym, price)
        if qty <= 0:
            return
        try:
            oid = self.broker.place_entry(sym, d, qty, price)
        except Exception as e:  # noqa: BLE001
            log.warning("Netz-Nachkauf %s nicht gelegt: %s", sym, e)
            return
        sl["add"] = {"oid": oid, "price": price, "qty": qty}

    def _grid_place_take(self, sym, sl, unit) -> None:
        """Mitnahme einer Einheit als reduzierende Limit-Order (Maker)."""
        d = sl["side"]
        price = unit["entry"] * (1 + d * float(self.cur["grid_take_pct"]) / 100)
        try:
            unit["take_oid"], unit["take_px"] = self.broker.place_exit(sym, d, unit["qty"], price), price
        except Exception as e:  # noqa: BLE001
            log.warning("Netz-Mitnahme %s nicht gelegt: %s", sym, e)

    def _grid_cancel_orders(self, sym, sl) -> None:
        for unit in sl.get("units") or []:
            if unit.get("take_oid"):
                try:
                    self.broker.cancel(sym, unit["take_oid"])
                except Exception as e:  # noqa: BLE001
                    log.debug("Netz Storno Mitnahme: %s", e)
                unit["take_oid"] = None
        if sl.get("add"):
            try:
                self.broker.cancel(sym, sl["add"]["oid"])
            except Exception as e:  # noqa: BLE001
                log.debug("Netz Storno Nachkauf: %s", e)
            sl["add"] = None

    def _grid_book(self, sym, sl, unit, px, why, maker_out=False) -> None:
        p = self.cur
        gross = sl["side"] * (px - unit["entry"]) * unit["qty"]
        fees = unit["entry"] * unit["qty"] * (p["maker"] if unit.get("maker_in") else p["taker"]) \
            + px * unit["qty"] * (p["maker"] if maker_out else p["taker"])
        t = {"symbol": sym, "side": "long" if sl["side"] == 1 else "short", "entry": unit["entry"], "exit": px,
             "qty": unit["qty"], "gross": round(gross, 6), "fees": round(fees, 6), "net": round(gross - fees, 6),
             "why": why, "secs": round(time.time() - unit.get("time", time.time())), "time": int(time.time() * 1000),
             "kind": self.kind, "label": self.session.get("label"), "why_in": sl.get("why_in"), "h_min": sl.get("h_min")}
        self.trades.append(t)
        self._save_trade(t)

    def _grid_close(self, sym, sl, why) -> None:
        """Ganzes Netz zum Marktpreis schliessen und jede Einheit einzeln verbuchen."""
        self._grid_cancel_orders(sym, sl)
        px = self.broker.close(sym, sl["side"], sl["qty"], sl.get("prot") or {"sl": sl.get("sl"), "tp": None})
        net = 0.0
        for unit in sl["units"]:
            self._grid_book(sym, sl, unit, px, why)
            net += self.trades[-1]["net"]
        self._event(f"{sym.split(':')[0]} Netz {why}: {len(sl['units'])} Einheiten, netto {net:+.4f} USDT")
        sl.clear()
        sl["state"] = "idle"

    def _grid_limit(self, sym, sl, side, now) -> None:
        """Limit-Netz: Mitnahmen und Nachkauf liegen als Maker-Orders auf der Boerse; hier nur nachsehen,
        was gefuellt wurde, und die naechsten Orders legen."""
        p = self.cur
        d = sl["side"]
        try:
            open_ids = self.broker.open_orders(sym)
        except Exception as e:  # noqa: BLE001
            log.debug("Netz offene Auftraege %s: %s", sym, e)
            return
        changed = False
        keep = 1 if p.get("grid_keep_base", True) else 0
        for unit in list(sl["units"]):
            oid = unit.get("take_oid")
            if oid and oid not in open_ids:
                st, px = self.broker.order_fill(sym, oid)
                if st == "filled":
                    self._grid_book(sym, sl, unit, px or unit["take_px"], "Einheit im Gewinn verkauft", maker_out=True)
                    sl["units"].remove(unit)
                    changed = True
                elif st == "canceled":
                    unit["take_oid"] = None
            elif not oid and sl["units"].index(unit) >= keep:
                self._grid_place_take(sym, sl, unit)
        if not sl["units"]:
            self._grid_cancel_orders(sym, sl)
            self._event(f"{sym.split(':')[0]} Netz: alle Einheiten im Gewinn verkauft")
            sl.clear()
            sl["state"] = "idle"
            return
        add = sl.get("add")
        if add and add["oid"] not in open_ids:
            st, px = self.broker.order_fill(sym, add["oid"])
            sl["add"] = None
            if st == "filled":
                unit = {"entry": px or add["price"], "qty": add["qty"], "time": now, "maker_in": True}
                sl["units"].append(unit)
                sl["last_add"] = unit["entry"]
                changed = True
                self._event(f"{sym.split(':')[0]} Netz: Einheit {len(sl['units'])} @ {unit['entry']:.6g} (Limit)")
                self._grid_place_take(sym, sl, unit)
        if side != d and sl.get("add"):                       # KI nicht mehr dafuer: nicht weiter nachkaufen
            self._grid_cancel_orders_add(sym, sl)
        elif side == d and not sl.get("add") and time.time() >= self.session.get("pause_until", 0) \
                and not self._brain_against(sym, d):
            self._grid_place_add(sym, sl, sl["last_add"])
        if changed:
            sl["qty"] = sum(u["qty"] for u in sl["units"])
            sl["entry"] = sum(u["entry"] * u["qty"] for u in sl["units"]) / sl["qty"]
            self._grid_restop(sym, sl)

    def _grid_cancel_orders_add(self, sym, sl) -> None:
        try:
            self.broker.cancel(sym, sl["add"]["oid"])
        except Exception as e:  # noqa: BLE001
            log.debug("Netz Storno Nachkauf: %s", e)
        sl["add"] = None

    def _grid_restop(self, sym, sl) -> None:
        sl["sl"] = self._grid_stop_price(sl)
        prot = sl.get("prot") or {}
        prot["sl"] = sl["sl"]
        try:
            if hasattr(self.broker, "pos_size"):
                self.broker.resize(sym, sl["side"], sl["qty"], prot)
                if sl["sl"] != prot.get("sl"):
                    self.broker.move_stop(sym, sl["side"], sl["qty"], prot, sl["sl"])
        except Exception as e:  # noqa: BLE001
            log.warning("Netz-Stop %s nicht angepasst: %s", sym, e)
        sl["prot"] = prot

    def _grid_reversal(self, sl, fc, df) -> str:
        """Trendwende gegen das Netz: die KI neigt klar zur Gegenseite (ab grid_flip_conf, unter der
        Einstiegs-Sicherheit), oder der Kurs ist unter/ueber die EMA-Kreuzung gelaufen und die KI neigt dagegen.
        Dann schliesst die KI das ganze Netz samt Hauptposition - Verluste werden nicht ausgesessen."""
        d = sl["side"]
        p_up = (fc or {}).get("p_up")
        if p_up is None:
            return ""
        against = d * (float(p_up) - 0.5) < 0
        conf = max(float(p_up), 1 - float(p_up))
        if against and conf >= float(self.cur.get("grid_flip_conf", 0.53)):
            return f"KI gedreht ({'SHORT' if d == 1 else 'LONG'} {round(conf * 100)} %)"
        if against and df is not None and len(df) >= 60:
            c = df["close"].astype(float)
            e20, e50 = float(ema(c, 20).iloc[-1]), float(ema(c, 50).iloc[-1])
            if d * (e20 - e50) < 0 and d * (float(c.iloc[-1]) - e20) < 0:
                return "Trendwende (EMA 20 unter 50, KI neigt dagegen)" if d == 1 else "Trendwende (EMA 20 ueber 50, KI neigt dagegen)"
        if against:
            veto = self._brain_against(sl.get("_sym", ""), d)
            if veto:
                return f"Trendwende ({veto}, KI neigt dagegen)"
        return ""

    def _grid(self, sym, sl, side, votes, tick, now, fc=None, df=None) -> None:
        """Netz fuehren: Einheiten im Gewinn verkaufen (Hauptposition bleibt), bei Kurs gegen das Netz bis zum
        Budget nachkaufen, Netz-Stop und KI-Drehung schliessen alles."""
        p = self.cur
        d = sl["side"]
        last = tick["last"]
        sell_px = tick["bid"] if d == 1 else tick["ask"]
        buy_px = tick["ask"] if d == 1 else tick["bid"]
        unreal = d * (last - sl["entry"]) * sl["qty"]
        used = sum(u["entry"] * u["qty"] for u in sl["units"]) / max(sl["lev"], 1)
        budget_loss = float(p["grid_budget"]) * float(p["grid_max_loss_pct"]) / 100
        votes = {**votes, "Netz": f"{len(sl['units'])} Einheiten, Margin {used:.2f}/{float(p['grid_budget']):g} USDT, "
                                  f"offen {unreal:+.4f} USDT"}
        sl["signal"]["votes"] = votes
        # 1) Netz-Stop (Software; auf Bitget liegt derselbe Stop) oder von aussen geschlossen
        if hasattr(self.broker, "pos_size"):
            try:
                if self.broker.pos_size(sym, d) < sl["qty"] * 0.5:
                    sl["miss"] = sl.get("miss", 0) + 1
                    if sl["miss"] >= 3:
                        px = sl["sl"] if abs(last - sl["sl"]) <= abs(sl["sl"]) * 0.003 else last
                        for unit in sl["units"]:
                            self._grid_book(sym, sl, unit, px, "Netz-Stop (Bitget)" if px == sl["sl"] else "von aussen geschlossen")
                        self._event(f"{sym.split(':')[0]} Netz auf Bitget geschlossen @ {px:.6g}")
                        sl.clear()
                        sl["state"] = "idle"
                        return
                else:
                    sl["miss"] = 0
            except Exception as e:  # noqa: BLE001
                log.debug("Netz Positionsgroesse %s: %s", sym, e)
        if unreal <= -budget_loss or d * (last - sl["sl"]) <= 0:
            self._grid_close(sym, sl, "Netz-Stop")
            return
        # 2) KI dreht / Trendwende -> alles schliessen, auch die Hauptposition
        sl["_sym"] = sym
        why = "KI gedreht" if side == -d else self._grid_reversal(sl, fc, df)
        # 2b) nicht endlos aussitzen: nach grid_stale_x x Vorhersagezeit ohne Gewinn schliessen
        h = float(sl.get("h_min") or 5)
        if not why and (now - sl.get("opened", now)) / 60 >= float(p.get("grid_stale_x", 4.0)) * h and unreal <= 0:
            why = f"zu lange ohne Gewinn ({round((now - sl['opened']) / 60)} min)"
        if why:
            self._grid_close(sym, sl, why)
            return
        if self.session.get("stopping"):
            return
        if p.get("grid_limit", True):
            self._grid_limit(sym, sl, side, now)
            return
        # 3) Einheiten im Gewinn verkaufen (die Hauptposition bleibt, solange grid_keep_base)
        take = float(p["grid_take_pct"]) / 100
        keep = 1 if p.get("grid_keep_base", True) else 0
        changed = False
        for unit in list(sl["units"]):
            if len(sl["units"]) <= keep:
                break
            if d * (sell_px - unit["entry"]) / unit["entry"] >= take:
                px = self.broker.close_part(sym, d, unit["qty"], sl.get("prot") or {})
                self._grid_book(sym, sl, unit, px, "Einheit im Gewinn verkauft")
                sl["units"].remove(unit)
                changed = True
        if changed and not sl["units"]:
            self._event(f"{sym.split(':')[0]} Netz: alle Einheiten im Gewinn verkauft")
            sl.clear()
            sl["state"] = "idle"
            return
        # 4) Nachkaufen, wenn der Kurs gegen das Netz gelaufen ist und Budget uebrig ist
        step = float(p["grid_step_pct"]) / 100
        unit_margin = float(p["grid_unit_margin"])
        if side == d and d * (sl["last_add"] - buy_px) / sl["last_add"] >= step and used + unit_margin <= float(p["grid_budget"]) * 1.05 \
                and time.time() >= self.session.get("pause_until", 0):
            qty = self._grid_unit_qty(sym, buy_px)
            if qty > 0 and not self._brain_against(sym, d):
                avg = self.broker.add(sym, d, qty)
                sl["units"].append({"entry": avg, "qty": qty, "time": now})
                sl["last_add"] = avg
                changed = True
                self._event(f"{sym.split(':')[0]} Netz: Einheit {len(sl['units'])} @ {avg:.6g}")
        if changed:
            sl["qty"] = sum(u["qty"] for u in sl["units"])
            sl["entry"] = sum(u["entry"] * u["qty"] for u in sl["units"]) / sl["qty"]
            self._grid_restop(sym, sl)

    def _anomaly(self, sym, sl, df, tick, now) -> str:
        """Schutz vor Boersen-Anomalien (belegte Bitget-Vorfaelle): Wartung, zu weiter Spread, Kurs-Sprung in
        einer Minute, Letztkurs weit weg vom Mark-Preis. Dann kein neuer Einstieg."""
        p = self.cur
        status_fn = getattr(self, "status_fn", None)
        if status_fn:
            try:
                st = status_fn(sym)
            except Exception:  # noqa: BLE001
                st = None
            if st and st != "normal":
                return f"Markt bei Bitget nicht normal ({st}) - Wartung?"
        if now < sl.get("calm_until", 0):
            return "nach Kurssprung noch keine neuen Einstiege"
        bid, ask, last = tick.get("bid"), tick.get("ask"), tick.get("last")
        if bid and ask and last:
            spread = (ask - bid) / last
            base = sym.split("/")[0]
            cap = (p.get("spread_max") or {}).get(base, 0.001)
            hist = sl.setdefault("spreads", [])
            hist.append(spread)
            del hist[:-120]
            limit = max(cap, 2.5 * sorted(hist)[len(hist) // 2]) if len(hist) >= 20 else cap
            if spread > limit:
                return f"Spread zu weit ({spread * 100:.3f} %)"
            mark = tick.get("mark")
            if mark and abs(last - mark) / mark > p.get("mark_div_max", 0.003):
                return f"Letztkurs {abs(last - mark) / mark * 100:.2f} % vom Mark-Preis entfernt - Anomalie"
        try:
            c = df["close"].astype(float).to_numpy()
            if len(c) > 30:
                r = np.diff(np.log(c[-61:]))
                sd = float(np.std(r[:-1])) or 1e-9
                if abs(r[-1]) > p.get("move_sigma_x", 4.0) * sd and abs(r[-1]) > 0.002:
                    sl["calm_until"] = now + p.get("calm_s", 600)
                    return f"Kurssprung {r[-1] * 100:+.2f} % in einer Minute - {p.get('calm_s', 600) // 60} min Pause"
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _chase(self, sym, sl, side, tick, now) -> bool:
        """Limit-Einstieg nicht ausgefuehrt, KI aber weiter dafuer und Kurs kaum weggelaufen -> zum Marktpreis
        einsteigen. Sonst fuellen sich Limit-Orders fast nur, wenn der Kurs GEGEN die Richtung laeuft, und die
        guten Trades werden verpasst."""
        p = self.cur
        if self.kind != "ki" or not p.get("chase") or side != sl["side"] or self.session.get("stopping"):
            return False
        secs = int(now) % 900
        if p.get("chase_avoid_marks", True) and (secs < 30 or secs > 870 or (int(now) % 28800) < 120
                                                 or (int(now) % 28800) > 28680):
            return False                              # um :00/:15/:30/:45 und Funding (00/08/16 UTC) springt der Spread
        r0 = abs(sl["price"] - sl["sl"])
        px_now = tick["ask"] if side == 1 else tick["bid"]
        if r0 <= 0 or side * (px_now - sl["price"]) > p.get("chase_max_r", 0.15) * r0:
            return False
        avg = self.broker.add(sym, side, sl["qty"])
        shift = avg - sl["price"]
        stop, target = sl["sl"] + shift, sl["tp"] + shift
        # erst den Zustand merken, dann absichern: schlaegt der Stop fehl, wird NIE ein zweites Mal gekauft
        sl.update(state="open", entry=avg, opened=now, sl=stop, tp=target, r0=abs(avg - stop), best=avg, taker_in=True)
        self._protect(sym, sl)
        self._event(f"{sym.split(':')[0]} Limit nicht gefuellt - zum Marktpreis eingestiegen @ {avg:.6g}")
        return True

    def _protect(self, sym, sl) -> bool:
        """Stop (und Ziel) auf Bitget setzen. Klappt es nicht, bleibt die Position als 'ungeschuetzt' markiert
        und wird in jeder Runde erneut abgesichert - und nach unprotected_s zur Sicherheit geschlossen."""
        try:
            sl["prot"] = self.broker.protect(sym, sl["side"], sl["qty"], sl["sl"], sl["tp"])
            sl.pop("unprotected", None)
            return True
        except Exception as e:  # noqa: BLE001
            sl["prot"] = {"sl": sl["sl"], "tp": sl["tp"], "sl_id": "", "tp_id": "", "t": time.time(), "miss": 0}
            sl.setdefault("unprotected", time.time())
            self._event(f"{sym.split(':')[0]} ACHTUNG: Stop nicht gesetzt ({e}) - neuer Versuch")
            return False

    def _manage(self, sym, sl, tick, side, votes, df=None) -> None:
        """KI-Autopilot fuehrt die Position: Einstand, Teilverkauf, Nachziehen, Ausstieg bei KI-Wende."""
        p = self.cur
        d = sl["side"]
        last = tick["last"]
        r0 = sl.get("r0") or abs(sl["entry"] - sl["sl"]) or sl["entry"] * 0.002
        sl["best"] = max(sl.get("best", last), last) if d == 1 else min(sl.get("best", last), last)
        gain = d * (last - sl["entry"]) / r0
        name = sym.split(":")[0]
        # KI dreht klar in die Gegenrichtung -> ganz verkaufen (nur mit deutlich mehr Sicherheit und nicht
        # sofort nach dem Einstieg - sonst kostet staendiges Rein/Raus nur Gebuehren)
        held = time.time() - sl.get("opened", time.time())
        min_hold = max(60.0, float(sl.get("h_min") or 5) * 60 * p.get("min_hold_frac", 0.5))
        if side == -d and (votes.get("Sicherheit") or 0) >= p["min_conf"] + p.get("flip_extra", 0.0) \
                and held >= min_hold:
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
        # Ziel erreicht, aber keine Ziel-Order auf Bitget (konnte nicht gesetzt werden) -> selbst schliessen
        prot = sl.get("prot") or {}
        if "tp_id" in prot and not prot.get("tp_id") and sl.get("tp") and d * (last - sl["tp"]) >= 0:
            px = self.broker.close(sym, d, sl["qty"], prot)
            self._book(sym, sl, px, "Ziel erreicht", False)
            return
        # Zeit: die Prognose gilt fuer h_min Minuten - danach ist ihr Vorteil weg
        h = float(sl.get("h_min") or 30) * 60
        if held >= p.get("time_take_x", 2.0) * h and gain >= 0.3 and side != d:
            px = self.broker.close(sym, d, sl["qty"], prot)
            self._book(sym, sl, px, f"Prognosezeit vorbei - Gewinn gesichert (+{gain:.1f} R)", False)
            return
        if held >= p.get("time_stale_x", 4.0) * h and gain < 1.0 and side != d:
            px = self.broker.close(sym, d, sl["qty"], prot)
            self._book(sym, sl, px, f"Prognosezeit lange vorbei ({gain:+.1f} R)", False)
            return
        # Gewinn sichern: Einstand, nie mehr als einen Teil des besten Gewinns hergeben, ATR-Nachzieh-Stop,
        # bei Erschoepfung (RSI-Extrem, Umkehrkerze, EMA-Wende) enger
        peak = d * (sl["best"] - sl["entry"]) / r0
        ev = exit_view(df, d) if df is not None and peak >= 0.8 else {}
        tired = bool(ev.get("tired"))
        be = sl["entry"] * (1 + d * (p["maker"] + p["taker"]))
        cands = []
        if peak >= p["be_r"]:
            cands.append(be)
        keep = p.get("keep_2r", 0.7) if peak >= 2.0 else p.get("keep", 0.5)
        if tired:
            keep = max(keep, 0.7)
        if peak >= 1.0:
            cands.append(sl["entry"] + d * keep * peak * r0)
            trending = (ev.get("adx") or 0) >= p.get("trail_adx", 20)
            if ev.get("atr") and (trending or tired or sl.get("partial_done")):
                tight = tired or sl.get("partial_done")
                k = p.get("atr_k_tight", 1.2) if tight else p.get("atr_k", 2.5)
                dist = max(k * ev["atr"], (0.5 if tight else 0.8) * r0)   # nie enger als 0,5 / 0,8 R
                cands.append(sl["best"] - d * dist)
        if sl.get("partial_done"):
            cands.append(sl["best"] - d * p["trail_r"] * r0)
        ok = [c for c in cands if d * (last - c) > 0]                 # nur Stops, die unter (Long) dem Kurs liegen
        # Kurs ist zwischen zwei Pruefungen schon unter die Halte-Linie gerutscht (Stop stand noch tiefer):
        # jetzt zum Marktpreis sichern, statt den Gewinn ganz herzugeben
        if peak >= 1.0 and gain > 0.15 and d * (last - (sl["entry"] + d * keep * peak * r0)) <= 0 \
                and d * (sl["sl"] - sl["entry"]) < keep * peak * r0 * 0.9:
            px = self.broker.close(sym, d, sl["qty"], sl["prot"])
            self._book(sym, sl, px, f"Gewinn gesichert (Rueckfall von +{peak:.1f} R)", False)
            return
        want = (max(ok) if d == 1 else min(ok)) if ok else None
        if want is not None and d * (want - sl["sl"]) >= 0.1 * r0:
            self.broker.move_stop(sym, d, sl["qty"], sl["prot"], want)
            why = ""
            if tired:
                hot = ev.get("rsi") is not None and (ev["rsi"] >= 72 if d == 1 else ev["rsi"] <= 28)
                label = ev["reversal"] or ("RSI " + str(ev["rsi"]) if hot else "EMA-Wende")
                why = f" ({label})"
            self._event(f"{name} Gewinn gesichert: Stop auf {want:.6g}{why}")
            sl["sl"] = want

    def _round(self, sym, qty) -> float:
        try:
            return float(self.broker.c.amount_to_precision(sym, qty)) if hasattr(self.broker, "c") else qty
        except Exception:  # noqa: BLE001
            return 0.0

    def _book_part(self, sym, sl, px, part) -> None:
        p = self.cur
        gross = sl["side"] * (px - sl["entry"]) * part
        fees = sl["entry"] * part * (p["taker"] if sl.get("taker_in") else p["maker"]) + px * part * p["taker"]
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

    def _auto_size(self, sym, price, stop, conf, p_tp=None) -> tuple[float, int]:
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
        if p_tp is not None:                                                # Viertel-Kelly, risk_pct als Deckel
            kelly = max(0.0, float(p_tp) - (1 - float(p_tp)) / float(p["tp_r"])) * p.get("kelly_frac", 0.25)
            risk = min(risk, total * kelly)
            if risk <= 0:
                return 0.0, p["leverage"]
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
        fees = sl["entry"] * sl["qty"] * (p["taker"] if sl.get("taker_in") else p["maker"]) \
            + px * sl["qty"] * (p["maker"] if maker else p["taker"])
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

    def _load_history(self, label: str) -> list[dict]:
        """Fruehere Abschluesse (gleiche Art, Simulation getrennt vom Konto) - daraus lernt der Autopilot."""
        path = self.cur.get("log_path")
        if not path:
            return []
        import json
        from pathlib import Path
        sim = label == "Simulation"
        out = []
        try:
            for line in Path(path).read_text(encoding="utf-8").splitlines()[-2000:]:
                try:
                    t = json.loads(line)
                except ValueError:
                    continue
                if t.get("kind") == self.kind and (t.get("label") == "Simulation") == sim and "net" in t:
                    out.append(t)
        except OSError:
            return []
        return out

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
            "pause_until": self.session.get("pause_until") if self.session.get("pause_until", 0) > time.time() else None,
            "open_risk": round(self._open_risk(), 4), "day_net": round(self._day_net(), 4),
            "resumed": bool(self.session.get("resumed")),
            "trades": len(tr), "wins": len(wins), "hit": round(len(wins) / len(tr), 3) if tr else None,
            "gross": round(sum(t["gross"] for t in tr), 4), "fees": round(sum(t["fees"] for t in tr), 4),
            "net": round(self._net(), 4), "last_trades": tr[-12:][::-1], "events": self.events[-15:][::-1],
            "slots": {s: {"state": v.get("state"), "side": v.get("side"), "entry": v.get("entry") or v.get("price"),
                          "opened": int(v["opened"]) if v.get("opened") else None,
                          "sl": v.get("sl"), "tp": v.get("tp"), "qty": v.get("qty"), "partial": v.get("partial_done"),
                          "lev": v.get("lev"), "added": v.get("added"), "units": len(v.get("units") or []) or None,
                          "signal": (v.get("signal") or {}).get("votes")}
                      for s, v in self.slots.items()},
        }
