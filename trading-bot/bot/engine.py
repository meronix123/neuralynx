"""Hauptschleife des Bots."""
import json
import logging
import time
from datetime import datetime, timezone

from .config import ROOT
from .context import MarketContext
from .exchange import is_metal
from .notify import Notifier
from .risk import RiskGuard, position_size, round_amount, stop_is_safe
from .strategy import TF_MS, compute_signals, last_closed_signal, snapshot, trail_stop

log = logging.getLogger("bot")
STATE_FILE = ROOT / "state.json"
STOP_FILE = ROOT / "STOP"
CHART_BARS = 150


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _series(df, col):
    return [None if v != v else round(float(v), 8) for v in df[col].tail(CHART_BARS)]


class Bot:
    def __init__(self, cfg: dict, exchange, context: MarketContext | None = None):
        self.cfg = cfg
        self.ex = exchange
        self.s = cfg["strategy"]
        self.r = cfg["risk"]
        self.fee = cfg["fees"]["taker"]
        self.state = load_state()
        self.state.setdefault("guard", {})
        self.state.setdefault("meta", {})       # Stop-Verwaltung je offener Position
        self.state.setdefault("last_sig", {})   # zuletzt gehandelter Bar je Symbol
        self.state.setdefault("history", [])    # abgeschlossene Trades
        self.state.setdefault("peak_equity", 0.0)
        self.state.setdefault("pending", {})    # offene Limit-Orders fuer den Einstieg
        self.guard = RiskGuard(self.r, self.state["guard"])
        self.ctx = context or MarketContext(cfg["context"])
        self.notify = Notifier(cfg["telegram"]["token"], cfg["telegram"]["chat_id"])
        self.status: dict = {"symbols": {}}     # fuer die Oberflaeche

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
                self.status["error"] = str(e)
            save_state(self.state)
            time.sleep(self.cfg["loop_seconds"])

    # ------------------------------------------------------------------
    def step(self) -> None:
        now = datetime.now(timezone.utc)
        equity = self.ex.equity()
        prev_day_start = self.guard.s["day_start_equity"]
        if self.guard.update_day(equity, now) and prev_day_start:
            self._daily_report(prev_day_start, equity)
        self.state["peak_equity"] = max(self.state["peak_equity"], equity)

        positions = self.ex.positions()
        self._check_pending(positions, now)
        self._handle_closed(positions, now)

        views = {}
        for sym in self.ex.symbols:
            views[sym] = self._analyze(sym)
        self._manage_open(positions, views)

        block, risk_factor = self._global_block(equity, now)
        reasons = {}
        for sym in self.ex.symbols:
            if sym in positions:
                reasons[sym] = "Position offen"
                continue
            if sym in self.state["pending"]:
                reasons[sym] = "Limit-Order wartet auf Ausfuehrung"
                continue
            if block:
                reasons[sym] = block
                continue
            reasons[sym] = self._try_enter(sym, equity, positions, views[sym], risk_factor, now)
        self._update_status(now, equity, positions, views, reasons, block)

    def _global_block(self, equity: float, now: datetime) -> tuple[str, float]:
        """Gruende, warum gerade GAR KEIN neuer Trade eroeffnet wird ('' = alles frei)."""
        if STOP_FILE.exists():
            return "STOP-Datei vorhanden", 0.0
        peak = self.state["peak_equity"]
        if peak > 0 and equity <= peak * (1 - self.r["max_total_drawdown_pct"] / 100):
            if not self.state.get("dd_alarm"):
                self.state["dd_alarm"] = True
                self.notify.send("ALARM: Maximaler Gesamtverlust erreicht - Bot eroeffnet keine Trades mehr. "
                                 "Zum Fortsetzen state.json loeschen.")
            return "Max. Gesamtverlust erreicht", 0.0
        ok, why, factor = self.ctx.check(now)
        if not ok:
            return why, 0.0
        return "", factor

    # ------------------------------------------------------------------
    def _check_pending(self, positions: dict, now: datetime) -> None:
        """Limit-Einstiege: ausgefuehrt -> Position uebernehmen, abgelaufen -> stornieren."""
        now_ms = int(now.timestamp() * 1000)
        for sym, o in list(self.state["pending"].items()):
            if sym not in positions and now_ms >= o["expires_ms"]:
                try:
                    self.ex.cancel(sym, o["order_id"])
                except Exception as e:  # noqa: BLE001 - evtl. gerade ausgefuehrt
                    log.info("%s Storno: %s", sym, e)
                p = self.ex.positions().get(sym)  # zwischen Abfrage und Storno ausgefuehrt?
                if p:
                    positions[sym] = p
                else:
                    del self.state["pending"][sym]
                    log.info("%s Limit-Order nicht ausgefuehrt - storniert", sym)
                    continue
            if sym in positions:
                del self.state["pending"][sym]
                self._register_open(sym, o["side"], positions[sym]["entry"], positions[sym]["amount"],
                                    o["sl"], o["tp"], o["score"])

    def _register_open(self, sym, side, entry, amount, sl, tp, score) -> None:
        self.guard.on_open()
        self.state["meta"][sym] = {
            "side": side, "entry": entry, "amount": amount, "sl_init": sl, "sl": sl, "tp": tp,
            "r0": abs(entry - sl), "score": score, "partial_done": False,
            "opened_ms": int(time.time() * 1000),
        }
        self.notify.send(
            f"NEU {sym} {side.upper()} {amount:g} @ {entry:.6g} | SL {sl:.6g} | TP {tp:.6g} "
            f"| {self.cfg['leverage']}x | Punkte {score}/6"
        )

    def _analyze(self, sym: str) -> dict:
        df = self.ex.candles(sym, self.cfg["timeframe"], 300)
        tdf = self.ex.candles(sym, self.cfg["trend_timeframe"], 300)
        sig_df = compute_signals(df, tdf, self.s, self.cfg["timeframe"], self.cfg["trend_timeframe"])
        return {"sig_df": sig_df, "snapshot": snapshot(sig_df, self.s)}

    def _handle_closed(self, positions: dict, now: datetime) -> None:
        for sym in list(self.state["meta"]):
            if sym in positions:
                continue
            m = self.state["meta"].pop(sym)
            price = self.ex.last_price(sym)
            pnl = self.ex.closed_pnl(sym, m["opened_ms"])
            if pnl is None:  # Schaetzung ueber letzten Kurs
                sign = 1 if m["side"] == "long" else -1
                pnl = sign * (price - m["entry"]) * m["amount"]
            self.guard.on_close(pnl, now)
            self.state["history"].append({
                "symbol": sym, "side": m["side"], "entry": m["entry"], "exit": price,
                "amount": m["amount"], "sl": m["sl_init"], "tp": m["tp"], "pnl": pnl,
                "score": m.get("score"), "opened_ms": m["opened_ms"], "closed_ms": int(now.timestamp() * 1000),
            })
            self.state["history"] = self.state["history"][-300:]
            self.notify.send(f"{'GEWINN' if pnl >= 0 else 'VERLUST'} {sym} {m['side']} geschlossen: {pnl:+.2f} USDT")

    def _manage_open(self, positions: dict, views: dict) -> None:
        for sym, p in positions.items():
            m = self.state["meta"].get(sym)
            if not m or sym not in views:
                continue
            sig_df = views[sym]["sig_df"]
            price = self.ex.last_price(sym)
            if self._partial_take_profit(sym, p, m, price):
                continue
            atr_now = float(sig_df["atr"].iloc[-1])
            new_sl = trail_stop(p["side"], m["entry"], m["sl_init"], m["sl"], price, atr_now, self.s, self.fee)
            if new_sl is None:
                continue
            try:
                self.ex.set_stop(sym, p["side"], new_sl, m["tp"])
                log.info("%s Stop nachgezogen: %.6g -> %.6g", sym, m["sl"], new_sl)
                m["sl"] = new_sl
            except Exception as e:  # noqa: BLE001
                log.warning("%s Stop nachziehen fehlgeschlagen: %s", sym, e)

    def _partial_take_profit(self, sym: str, p: dict, m: dict, price: float) -> bool:
        """Bei +partial_tp_r R einen Teil verkaufen und den Stop auf Einstand ziehen."""
        part_r = self.s.get("partial_tp_r", 0)
        if not part_r or m.get("partial_done"):
            return False
        sign = 1 if p["side"] == "long" else -1
        r0 = m.get("r0") or abs(m["entry"] - m["sl_init"])
        if r0 <= 0 or sign * (price - m["entry"]) / r0 < part_r:
            return False
        step, min_amt = self.ex.amount_rules(sym)
        qty = round_amount(p["amount"] * self.s.get("partial_tp_frac", 0.5), step, min_amt)
        m["partial_done"] = True
        if not 0 < qty < p["amount"]:
            log.info("%s Teilverkauf uebersprungen (Menge zu klein)", sym)
            return False
        self.ex.close(sym, p["side"], qty)
        be = m["entry"] * (1 + sign * 2 * self.fee)
        try:
            self.ex.set_stop(sym, p["side"], be, m["tp"])
            m["sl"] = be
        except Exception as e:  # noqa: BLE001
            log.warning("%s Stop auf Einstand fehlgeschlagen: %s", sym, e)
        m["amount"] = p["amount"] - qty
        self.notify.send(f"TEILGEWINN {sym}: {qty:g} verkauft @ ~{price:.6g}, Stop auf Einstand {be:.6g}")
        return True

    def _try_enter(self, sym: str, equity: float, positions: dict, view: dict,
                   risk_factor: float, now: datetime) -> str:
        """Versucht einen Einstieg. Rueckgabe: Begruendung fuer die Oberflaeche."""
        sig = last_closed_signal(view["sig_df"], self.s, self.ex.funding_rate(sym))
        if sig is None:
            return "Kein Signal"
        if self.state["last_sig"].get(sym) == sig.ts:
            return "Signal bereits bearbeitet"

        ok, why = self.guard.can_open(equity, len(positions) + len(self.state["pending"]), now)
        if not ok:
            return why
        same_dir = sum(1 for p in positions.values() if p["side"] == sig.side)
        if same_dir >= self.r["max_same_direction"]:
            return f"Schon {same_dir} Positionen {sig.side}"
        if is_metal(sym) and not self.cfg["filters"]["metals_trade_weekend"] and now.weekday() >= 5:
            return "Metalle am Wochenende pausiert"
        spread = self.ex.spread_pct(sym)
        if spread is not None and spread > self.cfg["filters"]["max_spread_pct"]:
            return f"Spread zu hoch ({spread:.3f} %)"

        # ab hier gilt das Signal als bearbeitet (kein zweiter Versuch fuer denselben Bar)
        self.state["last_sig"][sym] = sig.ts
        lev = self.cfg["leverage"]
        if not stop_is_safe(sig.price, sig.sl, lev):
            return "Stop zu nah an Liquidation"
        amount = position_size(equity, sig.price, sig.sl, lev, self.r["risk_per_trade_pct"] * risk_factor,
                               self.r["max_margin_per_trade_pct"], self.fee)
        step, min_amt = self.ex.amount_rules(sym)
        amount = round_amount(amount, step, min_amt)
        if amount <= 0:
            return "Konto zu klein fuer Mindestmenge"

        if self.cfg["fees"].get("entry_order", "market") == "limit":
            order_id = self.ex.place_limit(sym, sig.side, amount, sig.price, sig.sl, sig.tp)
            self.state["pending"][sym] = {
                "order_id": order_id, "side": sig.side, "price": sig.price, "amount": amount,
                "sl": sig.sl, "tp": sig.tp, "score": sig.score,
                # gueltig bis zum Ende des Bars nach dem Signal-Bar (wie im Backtest)
                "expires_ms": sig.ts + 2 * TF_MS[self.cfg["timeframe"]],
            }
            positions_pending = f"Limit-Order {sig.side} @ {sig.price:.6g}"
            log.info("%s %s", sym, positions_pending)
            return positions_pending
        entry = self.ex.open(sym, sig.side, amount, sig.sl, sig.tp)
        positions[sym] = {"side": sig.side, "amount": amount, "entry": entry}
        self._register_open(sym, sig.side, entry, amount, sig.sl, sig.tp, sig.score)
        return "Position eroeffnet"

    # ------------------------------------------------------------------
    def _daily_report(self, start: float, end: float) -> None:
        day = [t for t in self.state["history"] if t["closed_ms"] >= (time.time() - 86_400) * 1000]
        wins = sum(1 for t in day if t["pnl"] > 0)
        self.notify.send(
            f"Tagesbericht: {len(day)} Trades, {wins} Gewinner | Konto {start:.2f} -> {end:.2f} USDT "
            f"({(end / start - 1) * 100:+.1f} %)"
        )

    def _update_status(self, now, equity, positions, views, reasons, block) -> None:
        syms = {}
        for sym, v in views.items():
            df = v["sig_df"]
            m = self.state["meta"].get(sym)
            syms[sym] = {
                "snapshot": v["snapshot"],
                "reason": reasons.get(sym, ""),
                "position": {**positions[sym], **(m or {})} if sym in positions else None,
                "pending": self.state["pending"].get(sym),
                "candles": [
                    {"time": int(r.ts // 1000), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                    for r in df.tail(CHART_BARS).itertuples()
                ],
                "ema_fast": _series(df, "ema_f"),
                "ema_slow": _series(df, "ema_s"),
                "vwap": _series(df, "vwap"),
            }
        fng = self.ctx.fng
        self.status = {
            "mode": self.cfg["mode"],
            "leverage": self.cfg["leverage"],
            "timeframe": self.cfg["timeframe"],
            "tf_seconds": TF_MS[self.cfg["timeframe"]] // 1000,
            "updated": now.isoformat(),
            "equity": equity,
            "day_start_equity": self.guard.s["day_start_equity"],
            "peak_equity": self.state["peak_equity"],
            "trades_today": self.guard.s["trades_today"],
            "max_trades_per_day": self.r["max_trades_per_day"],
            "block": block,
            "fear_greed": fng,
            "calendar_ok": self.ctx.cal_ok,
            "next_events": [
                {**e, "time": e["time"].isoformat()} for e in self.ctx.next_events(now)
            ],
            "symbols": syms,
            "history": self.state["history"][-50:],
        }
