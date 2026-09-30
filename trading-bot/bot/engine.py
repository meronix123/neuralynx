"""Hauptschleife des Bots."""
import json
import logging
import time
from datetime import datetime, timezone

from .config import ROOT
from .indicators import atr
from .notify import Notifier
from .risk import RiskGuard, position_size, round_amount, stop_is_safe
from .strategy import compute_signals, last_closed_signal, trail_stop

log = logging.getLogger("bot")
STATE_FILE = ROOT / "state.json"
STOP_FILE = ROOT / "STOP"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


class Bot:
    def __init__(self, cfg: dict, exchange):
        self.cfg = cfg
        self.ex = exchange
        self.s = cfg["strategy"]
        self.fee = cfg["fees"]["taker"]
        self.state = load_state()
        self.state.setdefault("guard", {})
        self.state.setdefault("meta", {})       # Stop-Verwaltung je offener Position
        self.state.setdefault("last_sig", {})   # zuletzt gehandelter Bar je Symbol
        self.guard = RiskGuard(cfg["risk"], self.state["guard"])
        self.notify = Notifier(cfg["telegram"]["token"], cfg["telegram"]["chat_id"])

    def run(self) -> None:
        syms = ", ".join(self.ex.symbols)
        self.notify.send(f"Bot gestartet ({self.cfg['mode']}) - {syms}")
        for sym in self.ex.symbols:
            self.ex.setup(sym)
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 - Bot soll bei Netzfehlern weiterlaufen
                log.exception("Fehler im Durchlauf: %s", e)
            save_state(self.state)
            time.sleep(self.cfg["loop_seconds"])

    def step(self) -> None:
        now = datetime.now(timezone.utc)
        equity = self.ex.equity()
        self.guard.update_day(equity, now)
        positions = self.ex.positions()

        self._handle_closed(positions, now)
        self._manage_open(positions)

        if STOP_FILE.exists():
            log.info("STOP-Datei gefunden - keine neuen Trades")
            return
        for sym in self.ex.symbols:
            if sym in positions:
                continue
            ok, why = self.guard.can_open(equity, len(positions), now)
            if not ok:
                log.debug("Kein neuer Trade: %s", why)
                return
            if self._try_enter(sym, equity):
                positions[sym] = True

    # ------------------------------------------------------------------
    def _handle_closed(self, positions: dict, now: datetime) -> None:
        for sym in list(self.state["meta"]):
            if sym in positions:
                continue
            m = self.state["meta"].pop(sym)
            pnl = self.ex.closed_pnl(sym, m["opened_ms"])
            if pnl is None:  # Schaetzung ueber letzten Kurs
                price = self.ex.last_price(sym)
                sign = 1 if m["side"] == "long" else -1
                pnl = sign * (price - m["entry"]) * m["amount"]
            self.guard.on_close(pnl, now)
            self.notify.send(f"{'GEWINN' if pnl >= 0 else 'VERLUST'} {sym} {m['side']} geschlossen: {pnl:+.2f} USDT")

    def _manage_open(self, positions: dict) -> None:
        for sym, p in positions.items():
            m = self.state["meta"].get(sym)
            if not m:
                continue
            price = self.ex.last_price(sym)
            df = self.ex.candles(sym, self.cfg["timeframe"], 100)
            atr_now = float(atr(df, self.s["atr_period"]).iloc[-1])
            new_sl = trail_stop(p["side"], m["entry"], m["sl_init"], m["sl"], price, atr_now, self.s, self.fee)
            if new_sl is None:
                continue
            try:
                self.ex.set_stop(sym, p["side"], new_sl, m["tp"])
                log.info("%s Stop nachgezogen: %.6g -> %.6g", sym, m["sl"], new_sl)
                m["sl"] = new_sl
            except Exception as e:  # noqa: BLE001
                log.warning("%s Stop nachziehen fehlgeschlagen: %s", sym, e)

    def _try_enter(self, sym: str, equity: float) -> bool:
        df = self.ex.candles(sym, self.cfg["timeframe"], 300)
        tdf = self.ex.candles(sym, self.cfg["trend_timeframe"], 300)
        sig_df = compute_signals(df, tdf, self.s, self.cfg["timeframe"], self.cfg["trend_timeframe"])
        sig = last_closed_signal(sig_df, self.s, self.ex.funding_rate(sym))
        if sig is None or self.state["last_sig"].get(sym) == sig.ts:
            return False
        self.state["last_sig"][sym] = sig.ts

        lev = self.cfg["leverage"]
        if not stop_is_safe(sig.price, sig.sl, lev):
            log.info("%s Signal verworfen: Stop zu nah an Liquidation", sym)
            return False
        r = self.cfg["risk"]
        amount = position_size(equity, sig.price, sig.sl, lev, r["risk_per_trade_pct"],
                               r["max_margin_per_trade_pct"], self.fee)
        step, min_amt = self.ex.amount_rules(sym)
        amount = round_amount(amount, step, min_amt)
        if amount <= 0:
            log.info("%s Signal verworfen: Konto zu klein fuer Mindestmenge", sym)
            return False

        entry = self.ex.open(sym, sig.side, amount, sig.sl, sig.tp)
        self.guard.on_open()
        self.state["meta"][sym] = {
            "side": sig.side, "entry": entry, "amount": amount,
            "sl_init": sig.sl, "sl": sig.sl, "tp": sig.tp,
            "opened_ms": int(time.time() * 1000),
        }
        self.notify.send(
            f"NEU {sym} {sig.side.upper()} {amount:g} @ {entry:.6g} | SL {sig.sl:.6g} | TP {sig.tp:.6g} | {lev}x"
        )
        return True
