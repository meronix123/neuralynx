"""Hauptschleife des Bots."""
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

from .config import ROOT
from .context import MarketContext
from .derivs import fetch_funding_history, funding_blocks, live_rank
from .exchange import is_metal, spread_from_ticker
from .macro import fetch_macro, latest as macro_latest, macro_blocks
from .filters import health_gate, is_leader, leader_blocks, mtf_blocks, time_stop_due
from .flow import FlowMonitor, fetch_walls, flow_score, flow_verdict, wall_blocks
from .orderblocks import ob_blocks, zones_for_chart
from .ml import features, train_from_examples
from .notify import Notifier
from .risk import RiskGuard, position_size, round_amount, stop_is_safe
from .strategy import (MTF_EMAS, MTF_ORDER, REGIMES, STRATEGY_NAMES, TF_MS, compute_signals, higher_tfs,
                       last_closed_signal, plan_text, snapshot, tf_direction, trail_stop)
from .tfselect import active_tfs, adaptive, multi, profit_factor, recent, shadow_results, tf_allowed, tf_cfg

log = logging.getLogger("bot")
STATE_FILE = ROOT / "state.json"
STOP_FILE = ROOT / "STOP"
CHART_BARS = 150
CANDLES = 1000   # so viele Kerzen je Zeiteinheit laden (lange EMAs rechnen wie im Backtest)


def state_file(mode: str = "paper"):
    """Eigene Datei je Modus - Paper-, Test- und Echtgeld-Trades mischen sich nie."""
    return STATE_FILE if mode == "paper" else STATE_FILE.with_name(f"state_{mode}.json")


def load_state(mode: str = "paper") -> dict:
    f = state_file(mode)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict, mode: str = "paper") -> None:
    state_file(mode).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _series(df, col, n: int = CHART_BARS):
    return [None if v != v else round(float(v), 8) for v in df[col].tail(n)]


class Bot:
    def __init__(self, cfg: dict, exchange, context: MarketContext | None = None):
        self.cfg = cfg
        self.ex = exchange
        self.s = cfg["strategy"]
        self.r = cfg["risk"]
        self.fee = cfg["fees"]["taker"]
        self.state = load_state(cfg["mode"])
        self.state.setdefault("guard", {})
        self.state.setdefault("meta", {})       # Stop-Verwaltung je offener Position
        self.state.setdefault("last_sig", {})   # zuletzt gehandelter Bar je Symbol
        self.state.setdefault("history", [])    # abgeschlossene Trades
        self.state.setdefault("peak_equity", 0.0)
        self.state.setdefault("pending", {})    # offene Limit-Orders fuer den Einstieg
        self.guard = RiskGuard(self.r, self.state["guard"])
        if hasattr(self.ex, "restore") and "paper" in self.state:
            self.ex.restore(self.state["paper"])
        self.ctx = context or MarketContext(cfg["context"])
        self.notify = Notifier(cfg["telegram"]["token"], cfg["telegram"]["chat_id"])
        self.flow_cfg = cfg.get("flow", {})
        self.flow = FlowMonitor(self.flow_cfg)
        from .speed import SpeedTrader
        fee_cfg = {"maker": cfg["fees"].get("maker", cfg["fees"]["taker"]), "taker": cfg["fees"]["taker"],
                   "slippage": cfg["fees"]["slippage"], "min_notional": cfg["fees"].get("min_notional", 5.0)}
        self.speed = SpeedTrader(self.speed_data, {**cfg.get("speed", {}), **fee_cfg})
        self.autopilot = SpeedTrader(self.speed_data, {**cfg.get("autopilot", {}), **fee_cfg},
                                     forecast_fn=self.forecast, kind="ki")
        self.last_flow: dict = {}               # Symbol -> (Zeit, Messwerte) fuer die Oberflaeche
        self.status: dict = {"symbols": {}}     # fuer die Oberflaeche
        self.views: dict = {}
        self._candle_cache: dict = {}
        # Automatische Zeiteinheit: mehrere Einstiegs-Zeiteinheiten gleichzeitig beobachten
        self.multi, self.adaptive = multi(cfg), adaptive(cfg)
        self.tfs = active_tfs(cfg)
        self.tf_conf = {tf: tf_cfg(cfg, tf) for tf in self.tfs}
        self.tf_stats: dict = {}
        self._sig_cache: dict = {}
        self.manual_close: list[str] = []      # Schliessen-Knopf in der Oberflaeche
        self.last_error: dict | None = None
        self.thoughts: list[dict] = []        # Gedanken-Protokoll fuer die Oberflaeche
        self._last_seen: dict = {}            # zuletzt gemeldeter Zustand (nur Aenderungen melden)
        self.model = None
        self._model_trained_on = -1
        self._train_model()

    # ------------------------------------------------------------------
    def _ml_examples(self) -> list[dict]:
        """Lernbeispiele: Backtest (data/ml_examples.json) + eigene abgeschlossene Trades."""
        ex = []
        f = ROOT / "data" / "ml_examples.json"
        if f.exists():
            try:
                ex = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                log.warning("ml_examples.json unlesbar - wird ignoriert")
        ex += [{"x": t["x"], "win": int(t["pnl"] > 0)} for t in self.state["history"] if t.get("x")]
        return ex

    def _train_model(self) -> None:
        if not self.s.get("ml_filter"):
            return
        live = sum(1 for t in self.state["history"] if t.get("x"))
        if live == self._model_trained_on:
            return
        examples = self._ml_examples()
        if len(examples) >= self.s.get("ml_min_trades", 40):
            self.model = train_from_examples(examples)
            log.info("ML-Filter trainiert mit %d Beispielen", len(examples))
        self._model_trained_on = live

    def run(self) -> None:
        syms = ", ".join(self.ex.symbols)
        self.notify.send(f"Bot gestartet ({self.cfg['mode']}) - {syms}")
        for sym in self.ex.symbols:
            self.ex.setup(sym)
        last = time.time()
        while True:
            gap = time.time() - last
            if gap > max(300, 5 * self.cfg["loop_seconds"]):
                msg = (f"{gap / 60:.0f} Minuten Pause erkannt - der PC war vermutlich im Ruhezustand. "
                       "In dieser Zeit hat der Bot nichts gesehen und nichts gehandelt.")
                log.warning(msg)
                self.status["sleep_warning"] = {"at": datetime.now(timezone.utc).isoformat(), "msg": msg}
            last = time.time()
            self._heartbeat()
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 - Bot soll bei Netzfehlern weiterlaufen
                log.exception("Fehler im Durchlauf: %s", e)
                self._set_error(str(e))
            if hasattr(self.ex, "dump"):
                self.state["paper"] = self.ex.dump()
            save_state(self.state, self.cfg["mode"])
            time.sleep(self.cfg["loop_seconds"])

    def think(self, sym: str, text: str, kind: str = "info") -> None:
        """Gedanken-Protokoll: was der Bot gerade beobachtet, entscheidet, tut."""
        self.thoughts.append({"ms": int(time.time() * 1000), "sym": sym, "text": text, "kind": kind})
        del self.thoughts[:-250]

    def _changed(self, key: str, value) -> bool:
        old = self._last_seen.get(key)
        self._last_seen[key] = value
        return old is not None and old != value

    def _set_error(self, msg: str) -> None:
        self.last_error = {"at": datetime.now(timezone.utc).isoformat(), "msg": msg[:300]}
        self.status["last_error"] = self.last_error

    def _heartbeat(self) -> None:
        """Einmal pro Stunde ins Log: Bot lebt, so viele Signale, so viele gehandelt, haeufigste Gruende."""
        now = time.time()
        if now - self.__dict__.get("_hb", now - 3601 + 60) < 3600:
            return
        self._hb = now
        recent = [x for x in self.state.get("signal_log", []) if x["ms"] >= (now - 3600) * 1000]
        traded = sum(1 for x in recent if x["traded"])
        reasons: dict[str, int] = {}
        for x in recent:
            if not x["traded"]:
                k = x["why"].split("(")[0].strip()
                reasons[k] = reasons.get(k, 0) + 1
        top = ", ".join(f"{k}: {n}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
        log.info("LEBENSZEICHEN: laeuft | letzte Stunde %d Signale, %d gehandelt%s", len(recent), traded,
                 f" | nicht gehandelt wegen: {top}" if top else "")

    # ------------------------------------------------------------------
    def step(self) -> None:
        now = datetime.now(timezone.utc)
        equity = self.ex.equity()
        prev_day_start = self.guard.s["day_start_equity"]
        if self.guard.update_day(equity, now) and prev_day_start:
            self._daily_report(prev_day_start, equity)
        self.state["peak_equity"] = max(self.state["peak_equity"], equity)

        positions = self.ex.positions()
        if self.manual_close:
            self._do_manual_close(positions)
            positions = self.ex.positions()
        self._check_pending(positions, now)
        self._handle_closed(positions, now)

        views = {}
        for sym in self.ex.symbols:
            try:
                views[sym] = self._analyze(sym)
            except Exception as e:  # noqa: BLE001 - Markt voruebergehend nicht abrufbar
                log.warning("%s Analyse fehlgeschlagen: %s", sym, e)
                self._set_error(f"{sym}: {e}")
                if sym in self.views:
                    views[sym] = self.views[sym]  # letzte bekannte Daten weiter anzeigen
        self.views = views
        if self.multi:
            self.tf_stats = self._tf_stats(views)
        self._manage_open(positions, views, now)
        self._train_model()

        block, risk_factor = self._global_block(equity, now)
        if self._changed("block", block):
            self.think("-", f"Neue Trades gesperrt: {block}" if block else "Sperre aufgehoben - neue Trades wieder erlaubt",
                       "warn" if block else "info")
        for tf, st in self.tf_stats.items():
            if self._changed(f"gate|{tf}", st.get("ok")):
                self.think("-", f"Zeiteinheit {tf} " + ("wieder FREI" if st.get("ok") else "GESPERRT")
                           + f" (Schatten-Profit-Faktor {st.get('pf')})", "info" if st.get("ok") else "warn")
        for sym, v in views.items():
            for tf, tv in (v.get("tfs") or {self.cfg["timeframe"]: v}).items():
                if TF_MS[tf] >= TF_MS["30m"] and len(tv["sig_df"]) > 2:
                    reg = str(tv["sig_df"]["regime"].iloc[-2])
                    if self._changed(f"reg|{sym}|{tf}", reg):
                        self.think(sym, f"{tf}: Marktlage jetzt {REGIMES.get(reg, reg)}")
        reasons = {}
        for sym in self.ex.symbols:
            if sym not in views:
                reasons[sym] = "Keine Kursdaten (siehe Log)"
                continue
            if sym in positions:
                reasons[sym] = "Position offen"
                continue
            if self.speed.busy(sym) or self.autopilot.busy(sym):
                reasons[sym] = f"{'Speed-Trading' if self.speed.busy(sym) else 'KI-Autopilot'} laeuft in diesem Markt"
                continue
            if sym in self.state["pending"]:
                reasons[sym] = "Limit-Order wartet auf Ausfuehrung"
                continue
            if block:
                reasons[sym] = block
                continue
            try:
                reasons[sym] = self._enter_any(sym, equity, positions, views[sym], risk_factor, now)
            except Exception as e:  # noqa: BLE001 - ein Markt mit Fehler darf die anderen nicht blockieren
                log.exception("%s Einstieg: %s", sym, e)
                reasons[sym] = f"Fehler: {e}"
                self._set_error(f"{sym}: {e}")
        self._update_status(now, equity, positions, views, reasons, block)

    def request_close(self, sym: str) -> str:
        """Von der Oberflaeche: Bot-Position beim naechsten Durchlauf schliessen."""
        if sym not in self.state["meta"] and sym not in self.state["pending"]:
            raise RuntimeError(f"Der Bot hat in {sym} keine Position")
        self.manual_close.append(sym)
        return f"{sym} wird geschlossen (naechster Durchlauf, max. {self.cfg['loop_seconds']} s)"

    def request_close_all(self) -> str:
        syms = sorted(set(self.state["meta"]) | set(self.state["pending"]))
        if not syms:
            return "Der Bot hat keine offene Position"
        self.manual_close.extend(s for s in syms if s not in self.manual_close)
        return f"{len(syms)} Position(en)/Order(s) werden geschlossen (max. {self.cfg['loop_seconds']} s)"

    def _do_manual_close(self, positions: dict) -> None:
        while self.manual_close:
            sym = self.manual_close.pop(0)
            try:
                o = self.state["pending"].pop(sym, None)
                if o:
                    self.ex.cancel(sym, o["order_id"])
                    positions = self.ex.positions()  # evtl. kurz vor dem Storno ausgefuehrt
                p = positions.get(sym)
                if p:
                    m = self.state["meta"].get(sym, {})
                    self._cancel_tp(sym, m)
                    self.ex.close(sym, p["side"], p["amount"])
                    self.notify.send(f"MANUELL {sym} {p['side']} geschlossen (Oberflaeche)")
            except Exception as e:  # noqa: BLE001
                log.warning("%s manuell schliessen: %s", sym, e)
                self.status["error"] = f"{sym} schliessen fehlgeschlagen: {e}"

    def _global_block(self, equity: float, now: datetime) -> tuple[str, float]:
        """Gruende, warum gerade GAR KEIN neuer Trade eroeffnet wird ('' = alles frei)."""
        if STOP_FILE.exists():
            return "STOP-Datei vorhanden", 0.0
        peak = self.state["peak_equity"]
        if peak > 0 and equity <= peak * (1 - self.r["max_total_drawdown_pct"] / 100):
            if not self.state.get("dd_alarm"):
                self.state["dd_alarm"] = True
                self.notify.send("ALARM: Maximaler Gesamtverlust erreicht - Bot eroeffnet keine Trades mehr. "
                                 f"Zum Fortsetzen {state_file(self.cfg['mode']).name} loeschen.")
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
                    # ausgefuehrt UND schon wieder geschlossen (z. B. Stop in wenigen Sekunden)?
                    pnl = self.ex.closed_pnl(sym, o.get("placed_ms", o["expires_ms"] - 86_400_000))
                    if pnl is None:
                        log.info("%s Limit-Order nicht ausgefuehrt - storniert", sym)
                        self.think(sym, f"Limit-Order @ {o['price']:.6g} nicht ausgefuehrt (Kurs kam nicht zurueck) - storniert")
                    else:
                        log.info("%s Limit-Order ausgefuehrt und sofort wieder geschlossen: %+.2f USDT", sym, pnl)
                        self.guard.on_open()
                        m = {"side": o["side"], "entry": o["price"], "amount": o["amount"], "sl_init": o["sl"],
                             "tp": o["tp"], "score": o.get("score"), "opened_ms": o.get("placed_ms", now_ms),
                             **(o.get("info") or {})}
                        self._add_history(sym, m, self.ex.last_price(sym), pnl, now)
                    continue
            if sym in positions:
                del self.state["pending"][sym]
                self._register_open(sym, o["side"], positions[sym]["entry"], positions[sym]["amount"],
                                    o["sl"], o["tp"], o["score"], o.get("info"))

    # --- Take-Profit als liegende Limit-Order (Maker-Gebuehr) ------------------------
    @property
    def limit_tp(self) -> bool:
        return self.cfg["fees"].get("tp_order", "market") == "limit"

    def _exchange_tp(self, m: dict):
        """TP fuer die Boersen-Absicherung: None, wenn er als eigene Limit-Order liegt."""
        return None if m.get("tp_order_id") else m["tp"]

    def _place_tp(self, sym: str, side: str, amount: float, m: dict) -> None:
        try:
            m["tp_order_id"] = self.ex.place_tp_limit(sym, side, amount, m["tp"])
        except Exception as e:  # noqa: BLE001 - z. B. Kurs schon ueber dem Ziel -> normaler TP
            log.warning("%s TP-Limit-Order nicht moeglich (%s) - setze normalen Take-Profit", sym, e)
            m["tp_order_id"] = None
            try:
                self.ex.set_stop(sym, side, m["sl"], m["tp"])
            except Exception as e2:  # noqa: BLE001
                log.warning("%s Take-Profit setzen fehlgeschlagen: %s", sym, e2)

    def _cancel_tp(self, sym: str, m: dict) -> None:
        oid = m.get("tp_order_id")
        if oid:
            try:
                self.ex.cancel(sym, oid)
            except Exception as e:  # noqa: BLE001 - schon ausgefuehrt/erledigt
                log.debug("%s TP-Storno: %s", sym, e)
            m["tp_order_id"] = None

    def _register_open(self, sym, side, entry, amount, sl, tp, score, info=None) -> None:
        info = info or {}
        self.guard.on_open()
        self.state["meta"][sym] = {
            "side": side, "entry": entry, "amount": amount, "sl_init": sl, "sl": sl, "tp": tp,
            "r0": abs(entry - sl), "score": score, "partial_done": False,
            "opened_ms": int(time.time() * 1000), **info,
        }
        if self.limit_tp:
            self._place_tp(sym, side, amount, self.state["meta"][sym])
        self.think(sym, f"Position eroeffnet: {side.upper()} {amount:g} @ {entry:.6g}, Stop {sl:.6g}, Ziel {tp:.6g}", "trade")
        strat = STRATEGY_NAMES.get(info.get("strategy", ""), info.get("strategy", ""))
        self.notify.send(
            f"NEU {sym} {side.upper()} {amount:g} @ {entry:.6g} | SL {sl:.6g} | TP {tp:.6g} "
            f"| {self.cfg['leverage']}x | {strat}"
        )

    def _macro_score(self, sym: str) -> float | None:
        """Makro-Ampel (-1 Risiko aus .. +1 Risiko an), alle 6 Stunden neu geladen."""
        if not (self.s.get("macro_filter") or self.cfg.get("dashboard", {}).get("enabled")):
            return None
        at, scores = self.__dict__.get("_macro_cache", (0, None))
        if time.time() - at > 6 * 3600:
            try:
                new = fetch_macro(400)
                if new is not None and len(new):
                    scores = new
            except Exception as e:  # noqa: BLE001
                log.warning("Makro-Daten: %s", e)
            # ohne Daten in 30 Minuten erneut versuchen statt erst in 6 Stunden
            self._macro_cache = (time.time() if scores is not None else time.time() - 5.5 * 3600, scores)
        return macro_latest(scores, is_metal(sym), int(time.time() * 1000))

    def _funding_rank(self, sym: str, current: float | None) -> float | None:
        """Wie extrem ist das aktuelle Funding im Vergleich zum letzten Monat? (stuendlich neu geladen)"""
        if not self.s.get("funding_filter") or current is None:
            return None
        cache = self.__dict__.setdefault("_funding_cache", {})
        at, hist = cache.get(sym, (0, None))
        if time.time() - at > 3600:
            try:
                hist = fetch_funding_history(self.ex.c, sym, 45)
            except Exception as e:  # noqa: BLE001 - Filter ist optional
                log.debug("Funding-Historie %s: %s", sym, e)
            cache[sym] = (time.time(), hist)
        return live_rank(hist, current)

    def _cached_candles(self, sym: str, tf: str, limit: int = CANDLES):
        """Kerzen erst neu laden, wenn ein Bar abgeschlossen ist (spaetestens alle 5 Min)."""
        key, now = (sym, tf), time.time()
        hit = self._candle_cache.get(key)
        if hit and now - hit[0] < 300 and len(hit[1]):
            if now * 1000 < int(hit[1]["ts"].iloc[-1]) + TF_MS[tf] + 2000:
                return hit[1]
        df = self._candles(sym, tf, limit)
        self._candle_cache[key] = (now, df)
        return df

    def _candles(self, sym: str, tf: str, limit: int = CANDLES):
        """Kerzen laden; lehnt die Boerse die Menge ab, mit 300 erneut versuchen."""
        try:
            return self.ex.candles(sym, tf, limit)
        except Exception as e:  # noqa: BLE001
            if limit <= 300:
                raise
            log.debug("%s %s: %d Kerzen abgelehnt (%s) - nehme 300", sym, tf, limit, e)
            return self.ex.candles(sym, tf, 300)

    def _analyze(self, sym: str) -> dict:
        tf, ttf = self.cfg["timeframe"], self.cfg["trend_timeframe"]
        df = self._candles(sym, tf)              # jede Runde frisch (Chart, offener Bar)
        tdf = self._cached_candles(sym, ttf)     # Trend-Zeiteinheit aendert sich nur bei Bar-Schluss
        mtf = {}
        for h in higher_tfs(tf):
            try:
                mtf[h] = self._cached_candles(sym, h)
            except Exception as e:  # noqa: BLE001 - eine fehlende Zeitebene soll den Bot nicht stoppen
                log.debug("%s %s: %s", sym, h, e)
        sig_df = compute_signals(df, tdf, self.s, tf, ttf, mtf)
        # Richtung auf ALLEN Zeitebenen (1m ... 1w) - kleine nur als Info/Timing
        scan = {}
        for t in MTF_ORDER:
            try:
                frame = df if t == tf else mtf.get(t)
                if frame is None:
                    frame = self._cached_candles(sym, t)
                closed = frame.iloc[:-1]  # letzter Bar ist noch offen
                scan[t] = int(tf_direction(closed, *MTF_EMAS.get(t, (20, 50))).iloc[-1]) if len(closed) else 0
            except Exception as e:  # noqa: BLE001
                log.debug("%s Scan %s: %s", sym, t, e)
                scan[t] = None
        snap = snapshot(sig_df, self.s)
        snap["mtf"] = scan
        snap["macro"] = self._macro_score(sym)
        snap["mtf_score"] = round(float(sig_df["mtf_score"].iloc[-2]), 2) if "mtf_score" in sig_df else None
        view = {"sig_df": sig_df, "snapshot": snap}
        if self.multi:
            view["tfs"] = {}
            for t in self.tfs:
                try:
                    view["tfs"][t] = self._analyze_tf(sym, t, snap["macro"])
                except Exception as e:  # noqa: BLE001 - eine Zeiteinheit darf fehlen
                    log.debug("%s %s: %s", sym, t, e)
        return view

    def _analyze_tf(self, sym: str, tf: str, macro: float | None) -> dict:
        """Signale einer weiteren Einstiegs-Zeiteinheit (nur neu rechnen, wenn ein Bar schliesst)."""
        c = self.tf_conf[tf]
        df = self._cached_candles(sym, tf)
        last = int(df["ts"].iloc[-1])
        hit = self._sig_cache.get((sym, tf))
        if hit and hit[0] == last:
            sig_df = hit[1]
        else:
            mtf = {}
            for h in higher_tfs(tf):
                try:
                    mtf[h] = self._cached_candles(sym, h)
                except Exception as e:  # noqa: BLE001
                    log.debug("%s %s: %s", sym, h, e)
            sig_df = compute_signals(df, self._cached_candles(sym, c["trend_timeframe"]), c["strategy"],
                                     tf, c["trend_timeframe"], mtf)
            self._sig_cache[(sym, tf)] = (last, sig_df)
        ms = round(float(sig_df["mtf_score"].iloc[-2]), 2) if "mtf_score" in sig_df else None
        return {"sig_df": sig_df, "snapshot": {"macro": macro, "mtf_score": ms}}

    def _tf_stats(self, views: dict) -> dict:
        """Schatten-Konto je Zeiteinheit aus den letzten Kerzen aller Maerkte (wie im Backtest)."""
        leader = next((k for k in views if is_leader(k)), None)
        out = {}
        for tf in self.tfs:
            s = self.tf_conf[tf]["strategy"]
            lead = None
            if leader and tf in views[leader].get("tfs", {}):
                ld = views[leader]["tfs"][tf]["sig_df"]
                lead = dict(zip(ld["ts"].astype("int64"), ld["regime"].astype(str)))
            rows = []
            for sym, v in views.items():
                if tf not in v.get("tfs", {}):
                    continue
                sdf = v["tfs"][tf]["sig_df"].iloc[:-1]  # letzter Bar ist noch offen
                d = {k: sdf[k].tolist() for k in ("ts", "high", "low", "close", "signal", "sl_dist", "tp_dist")}
                if "mtf_score" in sdf:
                    d["mtf_score"] = sdf["mtf_score"].tolist()
                lr = None
                if lead is not None and not is_leader(sym) and not is_metal(sym):
                    lr = [lead.get(int(x)) for x in d["ts"]]
                rows += shadow_results(d, tf, s, self.fee, lr)
            rs = [r for _, r in sorted(rows)]
            pf = profit_factor(recent(rs, s))
            out[tf] = {"n": len(rs), "pf": None if pf is None else round(pf, 2),
                       "ok": tf_allowed(rs, s) if self.adaptive else True}
        return out

    def _enter_any(self, sym: str, equity: float, positions: dict, view: dict,
                   risk_factor: float, now: datetime) -> str:
        """Feste Zeiteinheit oder: alle beobachteten Zeiteinheiten pruefen (groesste zuerst)."""
        if not self.multi:
            return self._try_enter(sym, equity, positions, view, risk_factor, now)
        notes = []
        for tf in self.tfs:
            v = view.get("tfs", {}).get(tf)
            if v is None:
                continue
            st = self.tf_stats.get(tf, {})
            if not st.get("ok", True):
                row = v["sig_df"].iloc[-2]
                if int(row["signal"]) != 0:
                    why = f"Zeiteinheit laeuft gerade schlecht (Schatten-PF {st.get('pf')})"
                    notes.append(f"{tf}: Signal ausgelassen - {why}")
                    self._note_signal(sym, tf, "long" if int(row["signal"]) == 1 else "short",
                                      str(row["strategy"]), int(row["ts"]), why)
                continue
            why = self._try_enter(sym, equity, positions, v, risk_factor, now, tf)
            if sym in positions or sym in self.state["pending"]:
                return f"{tf}: {why}"
            if why != "Kein Signal":
                notes.append(f"{tf}: {why}")
        return " | ".join(notes) if notes else f"Kein Signal ({', '.join(self.tfs)})"

    def _handle_closed(self, positions: dict, now: datetime) -> None:
        for sym in list(self.state["meta"]):
            if sym in positions:
                continue
            m = self.state["meta"].pop(sym)
            self._cancel_tp(sym, m)
            price = self.ex.last_price(sym)
            pnl = self.ex.closed_pnl(sym, m["opened_ms"])
            if pnl is None:  # Schaetzung ueber letzten Kurs
                sign = 1 if m["side"] == "long" else -1
                pnl = sign * (price - m["entry"]) * m["amount"]
            self._add_history(sym, m, price, pnl, now)

    def _add_history(self, sym: str, m: dict, price: float, pnl: float, now: datetime) -> None:
        """Abgeschlossenen Trade verbuchen (Risiko-Zaehler, Historie, Nachricht)."""
        self.guard.on_close(pnl, now)
        self.think(sym, f"Position geschlossen: {pnl:+.2f} USDT ({'Gewinn' if pnl >= 0 else 'Verlust'})",
                   "win" if pnl >= 0 else "loss")
        self.state["history"].append({
            "symbol": sym, "side": m["side"], "entry": m["entry"], "exit": price,
            "amount": m["amount"], "sl": m["sl_init"], "tp": m["tp"], "pnl": pnl,
            "score": m.get("score"), "opened_ms": m["opened_ms"], "tf": m.get("tf"),
            "closed_ms": int(now.timestamp() * 1000),
            "strategy": m.get("strategy"), "regime": m.get("regime"), "flow": m.get("flow"),
            "x": m.get("x"), "ml_prob": m.get("ml_prob"),
        })
        self.state["history"] = self.state["history"][-300:]
        self.notify.send(f"{'GEWINN' if pnl >= 0 else 'VERLUST'} {sym} {m['side']} geschlossen: {pnl:+.2f} USDT")

    def _manage_open(self, positions: dict, views: dict, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        for sym, p in list(positions.items()):
            m = self.state["meta"].get(sym)
            if not m or sym not in views:
                continue
            try:
                self._manage_one(sym, p, m, views, now)
            except Exception as e:  # noqa: BLE001 - z. B. Position gerade vom Stop geschlossen
                log.warning("%s Positionsverwaltung: %s", sym, e)

    def _manage_one(self, sym: str, p: dict, m: dict, views: dict, now: datetime) -> None:
        """Zeit-Stop, Teilverkauf und Trailing-Stop fuer eine offene Position."""
        sig_df = views[sym].get("tfs", {}).get(m.get("tf"), views[sym])["sig_df"]
        price = self.ex.last_price(sym)
        if self._time_stop(sym, p, m, price, now) or self._partial_take_profit(sym, p, m, price):
            return
        atr_now = float(sig_df["atr"].iloc[-1])
        new_sl = trail_stop(p["side"], m["entry"], m["sl_init"], m["sl"], price, atr_now, self.s, self.fee)
        if new_sl is None:
            return
        try:
            self.ex.set_stop(sym, p["side"], new_sl, self._exchange_tp(m))
            log.info("%s Stop nachgezogen: %.6g -> %.6g", sym, m["sl"], new_sl)
            self.think(sym, f"Stop nachgezogen {m['sl']:.6g} -> {new_sl:.6g} (Gewinn sichern)")
            m["sl"] = new_sl
        except Exception as e:  # noqa: BLE001
            log.warning("%s Stop nachziehen fehlgeschlagen: %s", sym, e)

    def _time_stop(self, sym: str, p: dict, m: dict, price: float, now: datetime) -> bool:
        """Trade kommt nicht vom Fleck -> schliessen (wie im Backtest)."""
        bars = int((now.timestamp() * 1000 - m["opened_ms"]) // TF_MS[m.get("tf") or self.cfg["timeframe"]])
        sign = 1 if p["side"] == "long" else -1
        r0 = m.get("r0") or abs(m["entry"] - m["sl_init"])
        progress = sign * (price - m["entry"]) / r0 if r0 > 0 else 0.0
        if not time_stop_due(bars, progress, m.get("partial_done", False), self.s):
            return False
        self._cancel_tp(sym, m)
        self.ex.close(sym, p["side"], p["amount"])
        self.notify.send(f"ZEIT-STOP {sym}: nach {bars} Bars ohne Fortschritt geschlossen ({progress:+.2f}R)")
        self.think(sym, f"Zeit-Stop: nach {bars} Bars kaum Fortschritt ({progress:+.2f}R) - Position geschlossen")
        return True

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
        if m.get("tp_order_id"):  # liegende TP-Order auf die Restmenge anpassen
            self._cancel_tp(sym, m)
            self._place_tp(sym, p["side"], p["amount"] - qty, m)
        be = m["entry"] * (1 + sign * 2 * self.fee)
        try:
            self.ex.set_stop(sym, p["side"], be, self._exchange_tp(m))
            m["sl"] = be
        except Exception as e:  # noqa: BLE001
            log.warning("%s Stop auf Einstand fehlgeschlagen: %s", sym, e)
        m["amount"] = p["amount"] - qty
        self.notify.send(f"TEILGEWINN {sym}: {qty:g} verkauft @ ~{price:.6g}, Stop auf Einstand {be:.6g}")
        return True

    def _try_enter(self, sym: str, equity: float, positions: dict, view: dict,
                   risk_factor: float, now: datetime, tf: str | None = None) -> str:
        """Versucht einen Einstieg. Rueckgabe: Begruendung fuer die Oberflaeche."""
        tf = tf or self.cfg["timeframe"]
        s = self.tf_conf[tf]["strategy"] if tf in self.tf_conf else self.s
        sig_key = f"{sym}|{tf}" if self.multi else sym
        if len(view["sig_df"]) < 2 or int(view["sig_df"]["signal"].iloc[-2]) == 0:
            return "Kein Signal"
        funding = self.ex.funding_rate(sym)
        sig = last_closed_signal(view["sig_df"], s, funding)
        if sig is None:
            return "Signal verworfen (Funding-Rate extrem)"
        if self.state["last_sig"].get(sig_key) == sig.ts:
            return "Signal bereits bearbeitet"
        why = self._decide(sym, equity, positions, view, risk_factor, now, tf, s, sig_key, sig, funding)
        self._note_signal(sym, tf, sig.side, sig.strategy, sig.ts, why)
        return why

    def _note_signal(self, sym: str, tf: str, side: str, strategy: str, sig_ts: int, why: str) -> None:
        """Jedes neue Signal einmal protokollieren - mit dem Grund, warum (nicht) gehandelt wurde."""
        key = f"{sym}|{tf}|{sig_ts}"
        seen = self.__dict__.setdefault("_noted", set())
        if key in seen:
            return
        seen.add(key)
        if len(seen) > 2000:
            seen.clear()
            seen.add(key)
        traded = why.startswith(("Position eroeffnet", "Limit-Order"))
        log.info("SIGNAL %s %s %s (%s) -> %s", sym, tf, side.upper(), STRATEGY_NAMES.get(strategy, strategy), why)
        self.think(sym, f"{tf}: Signal {side.upper()} ({STRATEGY_NAMES.get(strategy, strategy)}) -> "
                        + ("gehandelt: " if traded else "nicht gehandelt: ") + why, "trade" if traded else "skip")
        lst = self.state.setdefault("signal_log", [])
        lst.append({"ms": int(time.time() * 1000), "symbol": sym, "tf": tf, "side": side,
                    "strategy": strategy, "why": why, "traded": traded})
        del lst[:-150]

    def _decide(self, sym, equity, positions, view, risk_factor, now, tf, s, sig_key, sig, funding) -> str:
        ok, why = self.guard.can_open(equity, len(positions) + len(self.state["pending"]), now)
        if not ok:
            return why
        same_dir = sum(1 for p in positions.values() if p["side"] == sig.side) + sum(
            1 for k, o in self.state["pending"].items() if o["side"] == sig.side and k not in positions)
        if same_dir >= self.r["max_same_direction"]:
            return f"Schon {same_dir} Positionen {sig.side}"
        if is_metal(sym) and not self.cfg["filters"]["metals_trade_weekend"] and now.weekday() >= 5:
            return "Metalle am Wochenende pausiert"
        spread = self.ex.spread_pct(sym)
        if spread is not None and spread > self.cfg["filters"]["max_spread_pct"]:
            return f"Spread zu hoch ({spread:.3f} %)"
        if self.s.get("leader_filter") and not is_leader(sym) and not is_metal(sym):
            leader = next((k for k in self.views if is_leader(k)), None)
            lv = self.views[leader].get("tfs", {}).get(tf, self.views[leader]) if leader else None
            lr = str(lv["sig_df"].iloc[-2]["regime"]) if lv else None
            if leader_blocks(sig.side, lr):
                return f"BTC im {'Abwaerts' if lr == 'trend_down' else 'Aufwaerts'}trend - kein {sig.side}"
        frank = self._funding_rank(sym, funding)
        if funding_blocks(sig.side, frank, self.s):
            self.state["last_sig"][sig_key] = sig.ts
            crowd = "zu viele Longs" if sig.side == "long" else "zu viele Shorts"
            return f"Markt ueberfuellt ({crowd}, Funding-Rang {frank * 100:.0f} % im Monatsvergleich)"
        mac = view["snapshot"].get("macro")
        if macro_blocks(sig.side, mac, self.s):
            self.state["last_sig"][sig_key] = sig.ts
            return f"Makro-Ampel dagegen ({'Risiko aus' if mac < 0 else 'Risiko an'}, {mac:+.2f})"
        mscore = view["snapshot"].get("mtf_score")
        if mtf_blocks(sig.side, mscore, self.s):
            self.state["last_sig"][sig_key] = sig.ts
            return f"Zeitebenen dagegen (Gesamtrichtung {mscore:+.2f})"
        pnls = [t["pnl"] for t in self.state["history"] if t.get("strategy") == sig.strategy]
        if not health_gate(sig.strategy, pnls, self.s, self.state.setdefault("health_skips", {})):
            self.state["last_sig"][sig_key] = sig.ts
            return f"Strategie '{STRATEGY_NAMES.get(sig.strategy, sig.strategy)}' pausiert (laeuft gerade schlecht)"
        row = view["sig_df"].iloc[-2]
        ob_why = ob_blocks(sig.side, float(row["close"]), float(row["tp_dist"]), row.get("ob_above", np.nan),
                           row.get("ob_below", np.nan), int(row.get("ob_support", 0) or 0), s)
        if ob_why:
            self.state["last_sig"][sig_key] = sig.ts
            return ob_why
        x = features({k: row[k] for k in ("rsi", "adx", "bb_width", "atr_pct", "vol_ratio", "macd_n",
                                          "trend", "regime", "strategy") if k in row}, sig.side)
        prob = self.model.proba(x) if (self.s.get("ml_filter") and self.model) else None
        if prob is not None and prob < self.s.get("ml_threshold", 0.45):
            self.state["last_sig"][sig_key] = sig.ts
            return f"ML-Filter: Gewinnchance nur {prob * 100:.0f} %"
        ki_mode = self.s.get("ki_filter", "auto")
        fcr = getattr(self, "_forecaster", None)
        ki_ready = ki_mode == "on" or (ki_mode == "auto" and fcr is not None
                                       and fcr.log.stats(sym)["n"] >= self.s.get("ki_min_live", 100))
        if ki_ready:
            from .forecast import ki_blocks
            try:
                fc = self.forecast(sym)
            except Exception as e:  # noqa: BLE001 - ohne Prognose weiter wie bisher
                log.debug("%s KI-Prognose: %s", sym, e)
                fc = None
            ki_why = ki_blocks(sig.side, fc, self.s)
            if ki_why:
                self.state["last_sig"][sig_key] = sig.ts
                return ki_why

        # ab hier gilt das Signal als bearbeitet (kein zweiter Versuch fuer denselben Bar)
        self.state["last_sig"][sig_key] = sig.ts
        flow = self.flow.measure(self.ex.c, sym, funding)
        self.last_flow[sym] = (time.time(), flow)
        ok, flow_why, flow_factor = flow_verdict(sig.side, flow, self.flow_cfg)
        if not ok:
            return flow_why
        wall_why = wall_blocks(sig.side, sig.price, sig.tp, self._walls(sym, force=True), self.flow_cfg)
        if wall_why:
            return wall_why
        risk_factor *= flow_factor
        info = {"strategy": sig.strategy, "regime": sig.regime, "x": x, "ml_prob": prob, "tf": tf,
                "mtf": view["snapshot"].get("mtf"), "mtf_score": view["snapshot"].get("mtf_score"),
                "macro": view["snapshot"].get("macro"), "funding_rank": frank,
                "flow": {**flow, "score": round(flow_score(sig.side, flow), 3)}}
        lev = self.cfg["leverage"]
        if not stop_is_safe(sig.price, sig.sl, lev):
            return "Stop zu nah an Liquidation"
        amount = position_size(equity, sig.price, sig.sl, lev, self.r["risk_per_trade_pct"] * risk_factor,
                               self.r["max_margin_per_trade_pct"], self.fee)
        step, min_amt = self.ex.amount_rules(sym)
        amount = round_amount(amount, step, min_amt)
        if amount <= 0:
            return "Konto zu klein fuer Mindestmenge"
        if amount * sig.price < self.cfg["fees"].get("min_notional", 5.0):
            return (f"Position waere nur {amount * sig.price:.2f} USDT wert - Bitget verlangt mind. "
                    f"{self.cfg['fees'].get('min_notional', 5.0):.0f} USDT")

        try:
            return self._place_entry(sym, sig, amount, info, positions, tf)
        except Exception as e:  # noqa: BLE001 - z. B. zu wenig Guthaben, Boerse lehnt ab
            log.warning("%s Order abgelehnt: %s", sym, e)
            return f"Order von der Boerse abgelehnt: {e}"

    def _place_entry(self, sym, sig, amount, info, positions, tf) -> str:
        if self.cfg["fees"].get("entry_order", "market") == "limit":
            order_id = self.ex.place_limit(sym, sig.side, amount, sig.price, sig.sl,
                                           None if self.limit_tp else sig.tp)
            self.state["pending"][sym] = {
                "order_id": order_id, "side": sig.side, "price": sig.price, "amount": amount,
                "placed_ms": int(time.time() * 1000),
                "sl": sig.sl, "tp": sig.tp, "score": sig.score, "info": info,
                # gueltig bis zum Ende des Bars nach dem Signal-Bar (wie im Backtest)
                "expires_ms": sig.ts + 2 * TF_MS[tf],
            }
            positions_pending = f"Limit-Order {sig.side} @ {sig.price:.6g} ({tf})"
            log.info("%s %s", sym, positions_pending)
            return positions_pending
        entry = self.ex.open(sym, sig.side, amount, sig.sl, None if self.limit_tp else sig.tp)
        positions[sym] = {"side": sig.side, "amount": amount, "entry": entry}
        self._register_open(sym, sig.side, entry, amount, sig.sl, sig.tp, sig.score, info)
        return "Position eroeffnet"

    # ------------------------------------------------------------------
    def _daily_report(self, start: float, end: float) -> None:
        day = [t for t in self.state["history"] if t["closed_ms"] >= (time.time() - 86_400) * 1000]
        wins = sum(1 for t in day if t["pnl"] > 0)
        self.notify.send(
            f"Tagesbericht: {len(day)} Trades, {wins} Gewinner | Konto {start:.2f} -> {end:.2f} USDT "
            f"({(end / start - 1) * 100:+.1f} %)"
        )

    def _walls(self, sym: str, force: bool = False) -> dict | None:
        """Grosse Orderbuch-Waende (jede Minute neu, vor einem Einstieg sofort)."""
        cache = self.__dict__.setdefault("_wall_cache", {})
        at, w = cache.get(sym, (0, None))
        if force or time.time() - at > self.flow_cfg.get("wall_refresh_s", 60):
            try:
                w = fetch_walls(self.ex.c, sym, self.flow_cfg)
            except Exception as e:  # noqa: BLE001
                log.debug("Waende %s: %s", sym, e)
            cache[sym] = (time.time(), w)
        return w

    def _flow_for_display(self, sym: str) -> dict | None:
        at, m = self.last_flow.get(sym, (0, None))
        if time.time() - at > self.flow_cfg.get("display_refresh_s", 120):
            try:
                m = self.flow.measure(self.ex.c, sym, None)
                self.last_flow[sym] = (time.time(), m)
            except Exception as e:  # noqa: BLE001
                log.debug("Flow-Anzeige %s: %s", sym, e)
        return m

    def _market(self, sym: str) -> dict:
        """Echte Marktdaten fuer die Uebersicht (Ticker jede Runde, Funding alle 2 Minuten)."""
        out: dict = {}
        try:
            tk = self.ex.c.fetch_ticker(sym)
            out = {"last": tk.get("last"), "change_pct": tk.get("percentage"), "high": tk.get("high"),
                   "low": tk.get("low"), "volume_usd": tk.get("quoteVolume"), "bid": tk.get("bid"),
                   "ask": tk.get("ask"), "spread_pct": spread_from_ticker(tk)}
        except Exception as e:  # noqa: BLE001 - Anzeige darf den Bot nie stoppen
            log.debug("Ticker %s: %s", sym, e)
        cache = self.__dict__.setdefault("_funding_now", {})
        at, rate = cache.get(sym, (0, None))
        if time.time() - at > 120:
            try:
                rate = self.ex.funding_rate(sym)
            except Exception as e:  # noqa: BLE001
                log.debug("Funding %s: %s", sym, e)
            cache[sym] = (time.time(), rate)
        out["funding"] = rate
        return out

    def _position_rows(self, positions: dict, market: dict, now: datetime) -> list[dict]:
        """Offene Positionen mit aktuellem Kurs: Gewinn/Verlust, Abstand zu Stop/Ziel, Liquidation."""
        lev = self.cfg["leverage"]
        rows = []
        for sym, p in positions.items():
            m = self.state["meta"].get(sym, {})
            price = (market.get(sym) or {}).get("last")
            if price is None:
                try:
                    price = self.ex.last_price(sym)
                except Exception:  # noqa: BLE001
                    continue
            sign = 1 if p["side"] == "long" else -1
            entry, amount = float(p["entry"]), float(p["amount"])
            sl, tp = m.get("sl"), m.get("tp")
            r0 = m.get("r0") or (abs(entry - m["sl_init"]) if m.get("sl_init") else None)
            upnl = sign * (price - entry) * amount - price * amount * self.fee  # inkl. Ausstiegsgebuehr
            margin = entry * amount / lev
            rows.append({
                "symbol": sym, "side": p["side"], "tf": m.get("tf") or self.cfg["timeframe"],
                "strategy": m.get("strategy"), "entry": entry, "price": price, "amount": amount,
                "value": price * amount, "margin": margin, "pnl": upnl,
                "pnl_pct": 100 * upnl / margin if margin else None,
                "r": sign * (price - entry) / r0 if r0 else None, "r0": r0,
                "sl": sl, "tp": tp,
                "sl_dist_pct": 100 * abs(price - sl) / price if sl else None,
                "tp_dist_pct": 100 * abs(tp - price) / price if tp else None,
                # isoliert: Liquidation grob bei 1/Hebel minus 0,5 % Wartungsmarge
                "liq": entry * (1 - sign * (1 / lev - 0.005)),
                "at_sl": sign * (sl - entry) * amount if sl else None,
                "at_tp": sign * (tp - entry) * amount if tp else None,
                "opened_ms": m.get("opened_ms"), "partial_done": m.get("partial_done", False),
            })
        return rows

    def live_prices(self) -> dict:
        """Aktuelle Kurse fuer die Oberflaeche - hoechstens einmal pro Sekunde von Bitget geholt."""
        at, prices = self.__dict__.get("_live", (0.0, {}))
        if time.time() - at < 1.0:
            return prices
        self._live = (time.time(), prices)   # parallele Anfragen nicht stapeln
        new = {}
        try:
            for sym, tk in self.ex.c.fetch_tickers(self.ex.symbols).items():
                if tk.get("last") is not None:
                    new[sym] = float(tk["last"])
        except Exception:  # noqa: BLE001 - z. B. nicht unterstuetzt -> einzeln
            for sym in self.ex.symbols:
                try:
                    new[sym] = float(self.ex.c.fetch_ticker(sym)["last"])
                except Exception as e:  # noqa: BLE001
                    log.debug("Kurs %s: %s", sym, e)
        prices = {**prices, **new}
        self._live = (time.time(), prices)
        return prices

    def _min_amount(self, sym: str) -> float:
        """Kleinste handelbare Menge eines Markts (z. B. 0,0001 BTC)."""
        try:
            m = self.ex.c.market(sym)
            return float(((m.get("limits") or {}).get("amount") or {}).get("min")
                         or (m.get("precision") or {}).get("amount") or 0)
        except Exception:  # noqa: BLE001
            return 0.0

    CHART_TFS = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w")

    def speed_data(self, sym: str):
        """Fuer Speed-Trading: 1-Minuten-Kerzen (5 s zwischengespeichert), Orderbuch und Kurs - frisch."""
        from .exchange import to_df
        cache = self.__dict__.setdefault("_speed_c", {})
        hit = cache.get(sym)
        if not hit or time.time() - hit[0] > 5:
            hit = (time.time(), to_df(self.ex.c.fetch_ohlcv(sym, "1m", limit=120)))
            cache[sym] = hit
        book = self.ex.c.fetch_order_book(sym, 50)
        bid, ask = float(book["bids"][0][0]), float(book["asks"][0][0])
        tk = self.ex.c.fetch_ticker(sym)
        last = float(tk.get("last") or (bid + ask) / 2)
        df = hit[1].copy()
        df.loc[df.index[-1], ["close"]] = last                  # offene Kerze mit dem aktuellen Kurs
        df.loc[df.index[-1], "high"] = max(float(df["high"].iloc[-1]), last)
        df.loc[df.index[-1], "low"] = min(float(df["low"].iloc[-1]), last)
        return df, book, {"last": last, "bid": bid, "ask": ask}

    def forecast(self, sym: str) -> dict:
        """KI-Prognose der naechsten 30 Minuten fuer einen Markt (siehe forecast.py)."""
        from .forecast import Forecaster
        known = sym in self.ex.symbols or (sym in (getattr(self.ex.c, "markets", None) or {})
                                           and self.ex.c.markets[sym].get("swap"))
        if not known:
            raise ValueError("Markt unbekannt")
        if getattr(self, "_forecaster", None) is None:
            leader = next((k for k in self.ex.symbols if is_leader(k)), None)

            def price(s):
                return (self.live_prices() or {}).get(s) if s in self.ex.symbols else None

            def macro():
                self._macro_score(self.ex.symbols[0])        # laedt/teilt den Makro-Speicher des Bots
                return self.__dict__.get("_macro_cache", (0, None))[1]
            eth = next((k for k in self.ex.symbols if k.startswith("ETH/")), None)
            self._forecaster = Forecaster(
                self._candles, ROOT / "data", self.cfg["mode"], leader, price,
                ohlcv_fn=lambda s, tf, since, limit: self.ex.c.fetch_ohlcv(s, tf, since=since, limit=limit),
                book_fn=lambda s: self.ex.c.fetch_order_book(s, 100),
                funding_fn=lambda s: fetch_funding_history(self.ex.c, s, 75),
                macro_fn=macro, leaders=[x for x in (leader, eth) if x])
        from .forecast import ki_active
        with self.__dict__.setdefault("_fc_lock", __import__("threading").Lock()):
            fc = self._forecaster.get(sym)
        active, why = ki_active(fc, self.s)
        return {**fc, "bot_uses": active, "bot_note": why}

    def chart_data(self, sym: str, tf: str, bars: int = 300) -> dict:
        """Chart fuer die Oberflaeche in beliebiger Zeiteinheit: Kerzen, EMAs, VWAP, Order Blocks."""
        from .indicators import atr, ema, vwap_daily
        known = sym in self.ex.symbols or (sym in (getattr(self.ex.c, "markets", None) or {})
                                           and self.ex.c.markets[sym].get("swap"))
        if not known or tf not in self.CHART_TFS:
            raise ValueError("Markt oder Zeiteinheit unbekannt")
        key = (sym, tf)
        hit = self.__dict__.setdefault("_chart_cache", {}).get(key)
        if hit and time.time() - hit[0] < 4:      # mehrere Fenster/Handy gleichzeitig: nicht doppelt laden
            return hit[1]
        df = self._candles(sym, tf, 1000).tail(1000).reset_index(drop=True)
        df["ema_f"] = ema(df["close"], self.s["ema_fast"])
        df["ema_s"] = ema(df["close"], self.s["ema_slow"])
        df["vwap"] = vwap_daily(df) if TF_MS[tf] < TF_MS["1d"] else np.nan
        df["atr"] = atr(df, self.s["atr_period"])
        zones = zones_for_chart(df, self.s, max_each=6, disp=min(1.0, self.s.get("ob_disp_atr", 1.5)))
        tail = df.tail(bars)
        out = {
            "symbol": sym, "tf": tf, "tf_seconds": TF_MS[tf] // 1000,
            "candles": [{"time": int(r.ts // 1000), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                        for r in tail.itertuples()],
            "ema_fast": _series(tail, "ema_f", bars), "ema_slow": _series(tail, "ema_s", bars),
            "vwap": _series(tail, "vwap", bars),
            "order_blocks": zones, "walls": self._walls(sym),
            "price": float(df["close"].iloc[-1]),
            "min_amount": self._min_amount(sym),
            "min_notional": self.cfg["fees"].get("min_notional", 5.0),
        }
        self._chart_cache[key] = (time.time(), out)
        return out

    def _update_status(self, now, equity, positions, views, reasons, block) -> None:
        market = {sym: self._market(sym) for sym in views}
        pos_rows = self._position_rows(positions, market, now)
        syms = {}
        for sym, v in views.items():
            df = v["sig_df"]
            m = self.state["meta"].get(sym)
            syms[sym] = {
                "snapshot": v["snapshot"],
                "reason": reasons.get(sym, ""),
                "position": {**positions[sym], **(m or {})} if sym in positions else None,
                "pending": self.state["pending"].get(sym),
                "market": market.get(sym),
                "order_blocks": zones_for_chart(df, self.s),
                "walls": self._walls(sym),
                "flow": self._flow_for_display(sym),
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
            "tf_select": self.cfg.get("tf_select", "fixed") if self.multi else "fixed",
            "entry_tfs": self.tfs,
            "tf_stats": self.tf_stats,
            "tf_seconds": TF_MS[self.cfg["timeframe"]] // 1000,
            "updated": now.isoformat(),
            "equity": equity,
            "day_start_equity": self.guard.s["day_start_equity"],
            "peak_equity": self.state["peak_equity"],
            "trades_today": self.guard.s["trades_today"],
            "max_trades_per_day": self.r["max_trades_per_day"],
            "block": block,
            "paused": STOP_FILE.exists(),
            "modes_ready": self._modes_ready(),
            "speed": self.speed.status(),
            "autopilot": self.autopilot.status(),
            "signal_log": self.state.get("signal_log", [])[-40:],
            "sleep_warning": self.status.get("sleep_warning"),
            "last_error": self.last_error,
            "fear_greed": fng,
            "calendar_ok": self.ctx.cal_ok,
            "next_events": [
                {**e, "time": e["time"].isoformat()} for e in self.ctx.next_events(now)
            ],
            "symbols": syms,
            "positions": pos_rows,
            "unrealized": sum(r["pnl"] for r in pos_rows),
            "max_positions": self.r["max_open_positions"],
            "pending_orders": [{"symbol": k, **{f: o.get(f) for f in ("side", "price", "amount", "sl", "tp", "expires_ms")},
                                "tf": (o.get("info") or {}).get("tf")} for k, o in self.state["pending"].items()],
            "history": self.state["history"][-50:],
            "brain": self._brain(views, positions, block),
        }

    @staticmethod
    def _modes_ready() -> dict:
        """Sind Schluessel fuer Testkonto / echtes Konto hinterlegt? (fuer die Modus-Knoepfe)"""
        import os
        live = all(os.getenv(n) for n in ("BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE"))
        demo = all(os.getenv(n) for n in ("BITGET_DEMO_API_KEY", "BITGET_DEMO_API_SECRET", "BITGET_DEMO_API_PASSPHRASE"))
        demo = demo or (os.getenv("BITGET_DEMO") == "1" and live)
        return {"paper": True, "demo": demo, "live": live}

    def _brain(self, views: dict, positions: dict, block: str) -> dict:
        """Alles fuer das Fenster 'Bot-Gehirn': Plan je Markt und Zeiteinheit, Filter, Gedanken."""
        markets = {}
        for sym, v in views.items():
            cells = {}
            for tf, tv in (v.get("tfs") or {self.cfg["timeframe"]: v}).items():
                try:
                    df = tv["sig_df"]
                    row = df.iloc[-2]
                    c = self.tf_conf.get(tf, {"trend_timeframe": self.cfg["trend_timeframe"], "strategy": self.s})
                    p = plan_text(row, c["strategy"], tf, c["trend_timeframe"])
                    st = self.tf_stats.get(tf, {})
                    if self.multi and not st.get("ok", True):
                        p = {**p, "state": "blocked", "text": f"GESPERRT (laeuft schlecht, Schatten-PF {st.get('pf')}) - "
                                                              + p["text"]}
                    p["mtf"] = round(float(row["mtf_score"]), 2) if "mtf_score" in row else None
                    p["rsi"] = round(float(row["rsi"]), 0)
                    p["regime"] = REGIMES.get(str(row["regime"]), str(row["regime"]))
                    cells[tf] = p
                except Exception as e:  # noqa: BLE001
                    cells[tf] = {"icon": "?", "state": "wait", "text": f"keine Daten ({e})"}
            snap = v["snapshot"]
            markets[sym] = {"cells": cells, "position": sym in positions, "pending": sym in self.state["pending"],
                            "macro": snap.get("macro"), "mtf_score": snap.get("mtf_score")}
        g = self.guard.s
        leader = next((k for k in views if is_leader(k)), None)
        return {
            "markets": markets, "tfs": self.tfs, "block": block,
            "risk": {"trades_today": g["trades_today"], "max_trades": self.r["max_trades_per_day"],
                     "consec_losses": g["consec_losses"], "max_consec": self.r["max_consecutive_losses"],
                     "paused_until": g["pause_until"] if g["pause_until"] > time.time() else None,
                     "day_start": g["day_start_equity"], "daily_limit_pct": self.r["daily_loss_limit_pct"],
                     "positions": len(positions), "max_positions": self.r["max_open_positions"]},
            "filters": {
                "leader_regime": REGIMES.get(str(views[leader]["sig_df"]["regime"].iloc[-2]), "") if leader else None,
                "fear_greed": self.ctx.fng, "calendar_ok": self.ctx.cal_ok,
                "macro_on": bool(self.s.get("macro_filter")), "funding_on": bool(self.s.get("funding_filter")),
                "mtf_on": bool(self.s.get("mtf_filter")), "ob": self.s.get("ob_filter", "off"),
                "leader_on": bool(self.s.get("leader_filter")),
                "strategies": self.s.get("strategies", []), "tf_select": self.cfg.get("tf_select", "fixed"),
                "tf_min_pf": self.s.get("tf_min_pf", 1.0),
            },
            "thoughts": self.thoughts[-120:],
        }
