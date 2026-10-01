"""Bitget-Konto in der Oberflaeche: Guthaben, Positionen, Orders, Trade-Historie und Aktionen.

Die API-Schluessel werden in der Oberflaeche eingegeben, einmal gegen Bitget geprueft und
nur lokal in der Datei .env gespeichert (nie angezeigt, nie hochgeladen).
Aktionen (schliessen, stornieren, Stop/Ziel aendern, neue Order) wirken auf das ECHTE Konto
(bzw. das Demokonto, wenn "Demo" gewaehlt wurde) - unabhaengig davon, ob der Bot selbst
im Paper-Modus laeuft.
"""
import json
import logging
import os
import re
import threading
import time

from .config import ROOT
from .exchange import cancel_plan, is_hedged, make_client, place_pos_tpsl, place_size_tpsl

log = logging.getLogger("bot")
ENV_FILE = ROOT / ".env"
PROFILE_KEYS = {
    "live": ("BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE"),
    "demo": ("BITGET_DEMO_API_KEY", "BITGET_DEMO_API_SECRET", "BITGET_DEMO_API_PASSPHRASE"),
}
PROFILE_NAMES = {"live": "Echtkonto", "demo": "Testkonto"}
HISTORY_DAYS = 30
REFRESH_S = 5
MIN_NOTIONAL = 5.0   # Bitget: jede Order mind. 5 USDT Positionswert


def save_env(values: dict[str, str], path=None) -> None:
    """Eintraege in .env setzen/ersetzen, alle anderen Zeilen bleiben erhalten."""
    path = path or ENV_FILE
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done = set()
    out = []
    for line in lines:
        name = line.split("=", 1)[0].strip()
        if name in values:
            out.append(f"{name}={values[name]}")
            done.add(name)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in values.items() if k not in done]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # nur der eigene Benutzer darf lesen (Linux/Mac)
    except OSError:
        pass
    for k, v in values.items():
        os.environ[k] = v


def explain_error(e: Exception) -> str:
    """Bitget-Fehler verstaendlich machen (statt einer ganzen Cloudflare-Webseite)."""
    msg = str(e)
    if "403" in msg and ("Cloudflare" in msg or "blocked" in msg):
        return ("Bitget hat die Anfrage blockiert (Cloudflare 403). Haeufigste Ursache: falsche Eingabe - "
                "die Passphrase ist das selbst vergebene API-Passwort, nicht die Berechtigungen. "
                "Sonst: VPN aus, kurz warten, erneut versuchen.")
    if "45110" in msg or "minimum amount" in msg:
        return ("Order zu klein: Bitget verlangt mindestens 5 USDT Positionswert (Menge x Kurs). "
                "Mehr Kapital-% oder hoeheren Hebel waehlen.")
    if "40014" in msg:
        return ("Dem API-Schluessel fehlen Futures-Rechte. Bei Bitget unter API-Verwaltung -> Schluessel bearbeiten: "
                "Futures 'Positionen' (Holdings) UND 'Orders' auf Lesen + Bearbeiten stellen. "
                "Auszahlen/Ueberweisen/Unterkonten bitte AUS lassen.")
    if "40037" in msg or "apikey" in msg.lower() and "not exist" in msg.lower():
        return "API-Key unbekannt - bitte genau kopieren (bei Demo-Schluesseln 'Testkonto' waehlen)."
    if "40012" in msg or "passphrase" in msg.lower():
        return "Passphrase falsch - es ist das Passwort, das du beim Anlegen des API-Schluessels vergeben hast."
    if "40018" in msg or "ip" in msg.lower() and "whitelist" in msg.lower():
        return "Deine IP ist fuer diesen API-Schluessel nicht freigegeben (IP-Beschraenkung bei Bitget pruefen)."
    return msg[:300]


def _f(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


class Account:
    def __init__(self, cfg: dict, symbols: list[str] | None = None, factory=make_client, env_path=None):
        self.cfg = cfg
        self.symbols = list(symbols or [])
        self.factory = factory
        self.env_path = env_path
        self.clients: dict = {"live": None, "demo": None}   # Echtkonto / Testkonto (Bitget-Demo)
        self.active = "demo" if cfg.get("mode") == "demo" else "live"
        self.error = ""
        self.data: dict = {}
        self.updated = 0.0
        self.lock = threading.RLock()
        for profile, names in PROFILE_KEYS.items():
            vals = [os.getenv(n, "") for n in names]
            if profile == "live" and os.getenv("BITGET_DEMO") == "1" and not os.getenv(PROFILE_KEYS["demo"][0]):
                continue  # alte Speicherung: Demo-Schluessel standen im Echt-Platz
            if profile == "demo" and not all(vals) and os.getenv("BITGET_DEMO") == "1":
                vals = [os.getenv(n, "") for n in PROFILE_KEYS["live"]]
            if all(vals):
                try:
                    self.connect(*vals, demo=profile == "demo", save=False, activate=False)
                except Exception as e:  # noqa: BLE001 - Oberflaeche zeigt den Fehler an
                    self.error = f"Gespeicherte {PROFILE_NAMES[profile]}-Schluessel funktionieren nicht: {e}"
        if self.clients[self.active] is None:
            other = "demo" if self.active == "live" else "live"
            if self.clients[other] is not None:
                self.active = other
        self.refresh(force=True)

    # --- Verbindung ------------------------------------------------------
    @property
    def client(self):
        return self.clients.get(self.active)

    @property
    def demo(self) -> bool:
        return self.active == "demo"

    @property
    def connected(self) -> bool:
        return self.client is not None

    def connect(self, key: str, secret: str, password: str, demo: bool = False, save: bool = True,
                activate: bool = True) -> None:
        # Key/Secret enthalten nie Leerzeichen -> beim Kopieren mitgerutschte Umbrueche entfernen
        key, secret = (re.sub(r"[\s\u200b-\u200d\ufeff]+", "", v) for v in (key, secret))
        password = re.sub(r"[\u200b-\u200d\ufeff]", "", password).strip()
        if not (key and secret and password):
            raise ValueError("API-Key, Secret und Passphrase eingeben")
        for name, v in (("API-Key", key), ("Secret", secret), ("Passphrase", password)):
            if not v.isascii() or any(ch.isspace() for ch in v):
                raise ValueError(f"{name} enthaelt Leerzeichen oder Umlaute - die Passphrase ist das Passwort, "
                                 "das du beim Anlegen des API-Schluessels vergeben hast (nicht die Berechtigungen)")
        profile = "demo" if demo else "live"
        client = self.factory({"key": key, "secret": secret, "password": password}, demo=demo)
        try:
            client.fetch_balance({"type": "swap"})  # prueft die Schluessel (Fehler -> Exception)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(explain_error(e)) from e
        with self.lock:
            self.clients[profile] = client
            if activate:
                self.active, self.error, self.data = profile, "", {}
        if save:
            save_env(dict(zip(PROFILE_KEYS[profile], (key, secret, password))), self.env_path)
        if activate:
            self.refresh(force=True)

    def switch(self, profile: str) -> str:
        """Zwischen Testkonto (demo) und Echtkonto (live) umschalten."""
        if profile not in self.clients:
            raise ValueError("demo oder live")
        with self.lock:
            self.active, self.data, self.error, self.updated = profile, {}, "", 0.0
        self.refresh(force=True)
        return f"{PROFILE_NAMES[profile]} gewaehlt" + ("" if self.connected else " - bitte Schluessel eingeben")

    def disconnect(self, forget: bool = False) -> None:
        with self.lock:
            self.clients[self.active], self.data, self.error = None, {}, ""
        if forget:
            save_env({v: "" for v in PROFILE_KEYS[self.active]}, self.env_path)

    # --- Daten -----------------------------------------------------------
    def refresh(self, force: bool = False) -> None:
        if not self.connected or (not force and time.time() - self.updated < REFRESH_S):
            return
        c = self.client
        try:
            bal = c.fetch_balance({"type": "swap"}).get("USDT", {}) or {}
            positions = [self._pos_row(p) for p in c.fetch_positions() if (_f(p.get("contracts")) or 0) > 0]
            syms = sorted(set(self.symbols) | {p["symbol"] for p in positions})
            data = {
                "balance": {"total": _f(bal.get("total")), "free": _f(bal.get("free")), "used": _f(bal.get("used"))},
                "positions": positions,
                "unrealized": sum(p["pnl"] or 0 for p in positions),
                "orders": self._orders(c, syms),
                "history": self._history(c),
                "updated_ms": int(time.time() * 1000),
            }
            try:
                self._sync_tickets(positions, data["orders"])
            except Exception as e:  # noqa: BLE001
                log.warning("Einzel-Positionen abgleichen: %s", e)
            data["tickets"] = self.tickets().open()
            data["tickets_closed"] = [x for x in self.tickets().items if x["status"] == "closed"][-30:][::-1]
            with self.lock:
                self.data, self.error, self.updated = data, "", time.time()
        except Exception as e:  # noqa: BLE001
            log.warning("Bitget-Konto: %s", e)
            with self.lock:
                self.error, self.updated = str(e), time.time()

    @staticmethod
    def _pos_row(p: dict) -> dict:
        info = p.get("info") or {}
        return {
            "symbol": p["symbol"], "side": p.get("side"), "amount": _f(p.get("contracts")),
            "entry": _f(p.get("entryPrice")), "mark": _f(p.get("markPrice")),
            "liq": _f(p.get("liquidationPrice")), "pnl": _f(p.get("unrealizedPnl")),
            "pnl_pct": _f(p.get("percentage")), "leverage": _f(p.get("leverage")),
            "margin_mode": p.get("marginMode"), "margin": _f(p.get("initialMargin") or p.get("collateral")),
            "value": _f(p.get("notional")),
            "sl": _f(p.get("stopLossPrice")), "tp": _f(p.get("takeProfitPrice")),
            "opened_ms": p.get("timestamp") or _f(info.get("cTime")),
        }

    @staticmethod
    def _order_row(o: dict, kind: str) -> dict:
        return {
            "id": str(o.get("id")), "symbol": o.get("symbol"), "kind": kind, "type": o.get("type"),
            "side": o.get("side"), "price": _f(o.get("price")), "trigger": _f(o.get("triggerPrice") or o.get("stopPrice")),
            "amount": _f(o.get("amount")), "filled": _f(o.get("filled")), "reduce_only": o.get("reduceOnly"),
            "ts": o.get("timestamp"),
        }

    def _orders(self, c, syms: list[str]) -> list[dict]:
        out, seen = [], set()
        for sym in syms:
            for kind, params in (("normal", {}), ("tpsl", {"trigger": True, "planType": "profit_loss"}),
                                 ("plan", {"trigger": True})):
                try:
                    for o in c.fetch_open_orders(sym, None, None, params):
                        if (o.get("id"), kind) not in seen:
                            seen.add((o.get("id"), kind))
                            out.append(self._order_row(o, kind))
                except Exception as e:  # noqa: BLE001 - einzelne Order-Arten koennen fehlen
                    log.debug("Orders %s %s: %s", sym, kind, e)
        return out

    def _history(self, c) -> list[dict]:
        since = int(time.time() * 1000) - HISTORY_DAYS * 86_400_000
        try:
            rows = c.fetch_positions_history(None, since, 100)
        except Exception as e:  # noqa: BLE001
            log.debug("Positions-Historie: %s", e)
            return []
        out = []
        for h in rows:
            info = h.get("info") or {}
            out.append({
                "symbol": h.get("symbol"), "side": h.get("side"), "entry": _f(h.get("entryPrice")),
                "exit": _f(info.get("closeAvgPrice")), "amount": _f(h.get("contracts") or info.get("closeTotalPos")),
                "pnl": _f(h.get("realizedPnl") if h.get("realizedPnl") is not None else info.get("netProfit")),
                "fees": _f(info.get("totalFee")), "funding": _f(info.get("totalFunding")),
                "opened_ms": h.get("timestamp") or _f(info.get("cTime")),
                "closed_ms": h.get("lastUpdateTimestamp") or _f(info.get("uTime")),
            })
        out.sort(key=lambda r: r["closed_ms"] or 0, reverse=True)
        return out

    def view(self) -> dict:
        with self.lock:
            return {"connected": self.connected, "demo": self.demo, "error": self.error, "active": self.active,
                    "profiles": {k: v is not None for k, v in self.clients.items()},
                    "bot_mode": self.cfg.get("mode"), "symbols": self.symbols, **self.data}

    # --- Aktionen --------------------------------------------------------
    def _need(self):
        if not self.connected:
            raise RuntimeError("Kein Bitget-Konto verbunden")
        return self.client

    def _position(self, symbol: str) -> dict:
        for p in self._need().fetch_positions([symbol]):
            if p["symbol"] == symbol and (_f(p.get("contracts")) or 0) > 0:
                return p
        raise RuntimeError(f"Keine offene Position in {symbol}")

    def close(self, symbol: str, fraction: float = 1.0) -> str:
        c = self._need()
        p = self._position(symbol)
        side = p.get("side")
        if fraction >= 0.999:
            c.close_position(symbol, self._close_side(c, side))
            msg = f"{symbol} {side} komplett geschlossen"
        else:
            qty = float(c.amount_to_precision(symbol, _f(p["contracts"]) * fraction))
            px_now = _f(p.get("markPrice")) or _f(p.get("entryPrice")) or 0
            if qty <= 0 or qty * px_now < MIN_NOTIONAL:
                raise RuntimeError("Haelfte waere unter 5 USDT (Bitget-Minimum) - bitte ganz schliessen")
            c.create_order(symbol, "market", "sell" if side == "long" else "buy", qty, None,
                           {"reduceOnly": True, "marginMode": p.get("marginMode") or self.cfg.get("margin_mode"),
                            "hedged": is_hedged(c)})
            msg = f"{symbol}: {qty:g} von {_f(p['contracts']):g} geschlossen"
        self.refresh(force=True)
        return msg

    # --- Einzel-Positionen (Tickets) ---------------------------------------
    def tickets(self) -> "Tickets":
        cache = self.__dict__.setdefault("_tickets", {})
        if self.active not in cache:
            path = (self.env_path.parent / "data" if self.env_path else ROOT / "data") / f"tickets_{self.active}.json"
            cache[self.active] = Tickets(path)
        return cache[self.active]

    def quick_order(self, symbol: str, side: str, margin_pct: float = 20, leverage: int = 10,
                    sl_pct: float = 10, tp_pct: float = 20) -> str:
        """Schnell-Order: margin_pct % des freien Guthabens als Margin, Stop bei sl_pct % Verlust und
        Ziel bei tp_pct % Gewinn - jeweils bezogen auf die Margin. Jede Order = eigene Einzel-Position."""
        c = self._need()
        if side not in ("long", "short"):
            raise ValueError("Richtung long oder short")
        if not 0 < margin_pct <= 100 or not 1 <= int(leverage) <= 125:
            raise ValueError("Kapital 1-100 %, Hebel 1-125")
        if not tp_pct or tp_pct <= 0:
            raise ValueError("Take-Profit ist Pflicht")
        if not 0 < sl_pct < 80:
            raise ValueError("Stop-Loss 1 bis 79 % der Margin (darueber droht vorher die Liquidation)")
        c.load_markets()
        if symbol not in c.markets:
            raise ValueError(f"Markt {symbol} gibt es auf Bitget nicht")
        # Bitget One-Way-Modus: Gegenrichtung wuerde bestehende Einzel-Positionen aufloesen (Hedge-Modus: erlaubt)
        for p in ([] if is_hedged(c) else c.fetch_positions([symbol])):
            if (_f(p.get("contracts")) or 0) > 0 and p.get("side") != side:
                raise RuntimeError(f"In {symbol} ist eine {p.get('side')}-Position offen - Gegenrichtung erst nach dem Schliessen")
        free = _f(c.fetch_balance({"type": "swap"}).get("USDT", {}).get("free")) or 0.0
        margin = free * margin_pct / 100
        price = float(c.fetch_ticker(symbol)["last"])
        lev = int(leverage)
        qty = float(c.amount_to_precision(symbol, margin * lev / price))
        min_amt = _f(((c.market(symbol).get("limits") or {}).get("amount") or {}).get("min")) or 0
        if qty <= 0 or qty < min_amt or qty * price < MIN_NOTIONAL:
            need = MIN_NOTIONAL / lev / free * 100 if free > 0 else 100
            raise ValueError(f"Order zu klein: {margin:.2f} USDT Margin x {lev} = {margin * lev:.2f} USDT Positionswert. "
                             f"Bitget verlangt mind. {MIN_NOTIONAL:.0f} USDT (bzw. die Mindestmenge) - mind. "
                             f"{min(100, need):.0f} % Kapital bei {lev}x waehlen (freies Guthaben {free:.2f} USDT).")
        sign = 1 if side == "long" else -1
        sl = price * (1 - sign * sl_pct / 100 / lev)
        tp = price * (1 + sign * tp_pct / 100 / lev)
        mm = self.cfg.get("margin_mode", "isolated")
        for fn in (lambda: c.set_margin_mode(mm, symbol, {"marginCoin": "USDT"}),
                   lambda: c.set_leverage(lev, symbol, {"holdSide": "long", "marginMode": mm}),
                   lambda: c.set_leverage(lev, symbol, {"holdSide": "short", "marginMode": mm})):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - "schon gesetzt" / Position offen
                log.debug("Schnell-Order Vorbereitung %s: %s", symbol, e)
        o = c.create_order(symbol, "market", "buy" if side == "long" else "sell", qty, None,
                           {"marginMode": mm, "hedged": is_hedged(c)})
        entry = _f(o.get("average")) or price
        sl_id = tp_id = ""
        err = []
        try:
            sl_id = place_size_tpsl(c, symbol, side, "loss_plan", sl, qty)
        except Exception as e:  # noqa: BLE001
            err.append(str(e))
        try:
            tp_id = place_size_tpsl(c, symbol, side, "profit_plan", tp, qty)
        except Exception as e:  # noqa: BLE001
            err.append(str(e))
        if not sl_id:
            # ohne Stop keine offene Position stehen lassen
            c.create_order(symbol, "market", "sell" if side == "long" else "buy", qty, None,
                           {"reduceOnly": True, "marginMode": mm, "hedged": is_hedged(c)})
            if tp_id:
                try:
                    cancel_plan(c, symbol, tp_id)
                except Exception as e:  # noqa: BLE001
                    log.debug("TP-Storno: %s", e)
            raise RuntimeError("Stop-Loss konnte nicht gesetzt werden - Position sofort wieder geschlossen. " + "; ".join(err))
        self.tickets().add(symbol=symbol, side=side, amount=qty, entry=entry, sl=sl, tp=tp, sl_id=sl_id,
                           tp_id=tp_id, leverage=lev, margin=entry * qty / lev, opened_ms=int(time.time() * 1000))
        self.refresh(force=True)
        msg = f"{side.upper()} {qty:g} {symbol.split(':')[0]} @ ~{entry:.6g} | Stop {sl:.6g} | Ziel {tp:.6g}"
        return msg + (f" (Hinweis: {'; '.join(err)})" if err else "")

    @staticmethod
    def _close_side(c, side: str) -> str:
        """close_position: One-Way erwartet 'buy'/'sell', Hedge 'long'/'short'."""
        if is_hedged(c):
            return side
        return "buy" if side == "long" else "sell"

    def close_ticket(self, ticket_id: int) -> str:
        c = self._need()
        tk = next((x for x in self.tickets().open() if x["id"] == int(ticket_id)), None)
        if tk is None:
            raise RuntimeError("Einzel-Position nicht gefunden (schon geschlossen?)")
        for oid in (tk.get("sl_id"), tk.get("tp_id")):
            if oid:
                try:
                    cancel_plan(c, tk["symbol"], oid)
                except Exception as e:  # noqa: BLE001 - evtl. schon ausgefuehrt
                    log.debug("Plan-Storno %s: %s", oid, e)
        o = c.create_order(tk["symbol"], "market", "sell" if tk["side"] == "long" else "buy", tk["amount"], None,
                           {"reduceOnly": True, "marginMode": self.cfg.get("margin_mode", "isolated"), "hedged": is_hedged(c)})
        price = _f(o.get("average")) or float(c.fetch_ticker(tk["symbol"])["last"])
        self.tickets().close(tk, price, "manuell")
        self.refresh(force=True)
        return f"Einzel-Position #{tk['id']} geschlossen ({tk['pnl']:+.2f} USDT geschaetzt)"

    def _sync_tickets(self, positions: list[dict], orders: list[dict]) -> None:
        """Hat Bitget ein Ziel/einen Stop ausgefuehrt? Dann Ticket schliessen und den Gegen-Auftrag stornieren."""
        tks = self.tickets()
        open_ids = {o["id"] for o in orders if o.get("kind") == "tpsl"}
        size = {}
        for p in positions:
            k = (p["symbol"], p.get("side"))
            size[k] = size.get(k, 0) + (p["amount"] or 0)
        c = self.client
        for tk in tks.open():
            sl_open, tp_open = tk.get("sl_id") in open_ids, tk.get("tp_id") in open_ids
            if size.get((tk["symbol"], tk["side"]), 0) <= 0:
                why = "Ziel erreicht" if not tp_open and sl_open else "Stop ausgeloest" if not sl_open and tp_open else "Position geschlossen"
                price = tk["tp"] if why == "Ziel erreicht" else tk["sl"] if why == "Stop ausgeloest" else None
            elif not tp_open and tk.get("tp_id"):
                why, price = "Ziel erreicht", tk["tp"]
            elif not sl_open and tk.get("sl_id"):
                why, price = "Stop ausgeloest", tk["sl"]
            else:
                continue
            for oid in (tk.get("sl_id"), tk.get("tp_id")):
                if oid in open_ids:
                    try:
                        cancel_plan(c, tk["symbol"], oid)
                    except Exception as e:  # noqa: BLE001
                        log.debug("Plan-Storno %s: %s", oid, e)
            tks.close(tk, price, why)
            log.info("Einzel-Position #%s %s: %s", tk["id"], tk["symbol"], why)

    def close_all(self) -> str:
        """Alle offenen Positionen des gewaehlten Kontos zum Marktpreis schliessen."""
        c = self._need()
        done, failed = [], []
        for p in c.fetch_positions():
            if (_f(p.get("contracts")) or 0) <= 0:
                continue
            try:
                c.close_position(p["symbol"], self._close_side(c, p.get("side")))
                done.append(p["symbol"].split(":")[0])
            except Exception as e:  # noqa: BLE001
                failed.append(f"{p['symbol']}: {e}")
        for tk in self.tickets().open():
            if tk["symbol"].split(":")[0] in done:
                self.tickets().close(tk, None, "alle geschlossen")
        self.refresh(force=True)
        if failed:
            raise RuntimeError(f"Geschlossen: {', '.join(done) or '-'} | Fehler: {'; '.join(failed)}")
        return f"{len(done)} Position(en) geschlossen: {', '.join(done)}" if done else "Keine offene Position"

    def cancel(self, order_id: str, symbol: str, kind: str = "normal") -> str:
        c = self._need()
        params = {} if kind == "normal" else {"trigger": True, **({"planType": "profit_loss"} if kind == "tpsl" else {})}
        c.cancel_order(order_id, symbol, params)
        self.refresh(force=True)
        return f"Order {order_id} storniert"

    def set_tpsl(self, symbol: str, sl: float | None, tp: float | None) -> str:
        c = self._need()
        p = self._position(symbol)
        place_pos_tpsl(c, symbol, p.get("side"), sl, tp)
        self.refresh(force=True)
        return f"{symbol}: Stop {sl if sl is not None else '-'} / Ziel {tp if tp is not None else '-'} gesetzt"

    def order(self, symbol: str, side: str, usdt: float, leverage: int, sl: float,
              tp: float | None = None, order_type: str = "market", price: float | None = None,
              margin_mode: str | None = None) -> str:
        """Neue Position: Positionswert in USDT, Hebel, Stop-Loss ist Pflicht."""
        c = self._need()
        if side not in ("long", "short"):
            raise ValueError("Richtung long oder short")
        if not usdt or usdt <= 0:
            raise ValueError("Positionswert in USDT angeben")
        if not sl:
            raise ValueError("Stop-Loss ist Pflicht")
        if not tp:
            raise ValueError("Take-Profit ist Pflicht")
        if not 1 <= int(leverage) <= 125:
            raise ValueError("Hebel 1 bis 125")
        c.load_markets()
        mm = margin_mode or self.cfg.get("margin_mode", "isolated")
        ref = price if order_type == "limit" and price else float(c.fetch_ticker(symbol)["last"])
        if (side == "long" and sl >= ref) or (side == "short" and sl <= ref):
            raise ValueError("Stop-Loss liegt auf der falschen Seite des Kurses")
        if tp and ((side == "long" and tp <= ref) or (side == "short" and tp >= ref)):
            raise ValueError("Take-Profit liegt auf der falschen Seite des Kurses")
        for fn in (lambda: c.set_margin_mode(mm, symbol, {"marginCoin": "USDT"}),
                   lambda: c.set_leverage(int(leverage), symbol, {"holdSide": "long", "marginMode": mm}),
                   lambda: c.set_leverage(int(leverage), symbol, {"holdSide": "short", "marginMode": mm})):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - "schon gesetzt" / Position offen
                log.debug("Order-Vorbereitung %s: %s", symbol, e)
        qty = float(c.amount_to_precision(symbol, usdt / ref))
        if qty <= 0 or qty * ref < MIN_NOTIONAL:
            raise ValueError(f"Order zu klein: Bitget verlangt mind. {MIN_NOTIONAL:.0f} USDT Positionswert")
        params = {"marginMode": mm, "stopLoss": {"triggerPrice": sl}, "hedged": is_hedged(c)}
        if tp:
            params["takeProfit"] = {"triggerPrice": tp}
        o = c.create_order(symbol, "limit" if order_type == "limit" else "market",
                           "buy" if side == "long" else "sell", qty, price if order_type == "limit" else None, params)
        self.refresh(force=True)
        return f"Order {o.get('id')} {side} {qty:g} {symbol} gesendet"


class Tickets:
    """Einzel-Positionen wie bei MetaTrader: jede Schnell-Order ist ein eigenes 'Ticket' mit eigenem
    Einstieg, Stop-Loss und (Pflicht-)Take-Profit. Bitget fuehrt alles in EINER Position pro Markt -
    die Tickets werden hier verwaltet, ihre Stops/Ziele liegen als Teil-Auftraege (Menge = Ticket) auf Bitget."""

    def __init__(self, path):
        self.path = path
        self.items: list[dict] = []
        if path and path.exists():
            try:
                self.items = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                log.warning("Ticket-Datei unlesbar - starte leer")

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(exist_ok=True)
            self.path.write_text(json.dumps(self.items[-500:], indent=1), encoding="utf-8")

    def open(self) -> list[dict]:
        return [x for x in self.items if x["status"] == "open"]

    def add(self, **kw) -> dict:
        tid = max([x["id"] for x in self.items] or [0]) + 1
        item = {"id": tid, "status": "open", **kw}
        self.items.append(item)
        self.save()
        return item

    def close(self, item: dict, exit_price: float | None, why: str) -> None:
        sign = 1 if item["side"] == "long" else -1
        item.update(status="closed", exit=exit_price, why=why, closed_ms=int(time.time() * 1000),
                    pnl=None if exit_price is None else sign * (exit_price - item["entry"]) * item["amount"])
        self.save()


def refresher(account: Account) -> threading.Thread:
    """Konto im Hintergrund regelmaessig neu laden."""
    def loop():
        while True:
            try:
                account.refresh()
            except Exception as e:  # noqa: BLE001
                log.debug("Konto-Aktualisierung: %s", e)
            time.sleep(3)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th
