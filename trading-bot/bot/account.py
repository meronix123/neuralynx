"""Bitget-Konto in der Oberflaeche: Guthaben, Positionen, Orders, Trade-Historie und Aktionen.

Die API-Schluessel werden in der Oberflaeche eingegeben, einmal gegen Bitget geprueft und
nur lokal in der Datei .env gespeichert (nie angezeigt, nie hochgeladen).
Aktionen (schliessen, stornieren, Stop/Ziel aendern, neue Order) wirken auf das ECHTE Konto
(bzw. das Demokonto, wenn "Demo" gewaehlt wurde) - unabhaengig davon, ob der Bot selbst
im Paper-Modus laeuft.
"""
import logging
import os
import re
import threading
import time

from .config import ROOT
from .exchange import make_client, place_pos_tpsl

log = logging.getLogger("bot")
ENV_FILE = ROOT / ".env"
PROFILE_KEYS = {
    "live": ("BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE"),
    "demo": ("BITGET_DEMO_API_KEY", "BITGET_DEMO_API_SECRET", "BITGET_DEMO_API_PASSPHRASE"),
}
PROFILE_NAMES = {"live": "Echtkonto", "demo": "Testkonto"}
HISTORY_DAYS = 30
REFRESH_S = 15


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
            c.close_position(symbol, "buy" if side == "long" else "sell")
            msg = f"{symbol} {side} komplett geschlossen"
        else:
            qty = float(c.amount_to_precision(symbol, _f(p["contracts"]) * fraction))
            if qty <= 0:
                raise RuntimeError("Menge zu klein zum Teil-Schliessen")
            c.create_order(symbol, "market", "sell" if side == "long" else "buy", qty, None,
                           {"reduceOnly": True, "marginMode": p.get("marginMode") or self.cfg.get("margin_mode")})
            msg = f"{symbol}: {qty:g} von {_f(p['contracts']):g} geschlossen"
        self.refresh(force=True)
        return msg

    def close_all(self) -> str:
        """Alle offenen Positionen des gewaehlten Kontos zum Marktpreis schliessen."""
        c = self._need()
        done, failed = [], []
        for p in c.fetch_positions():
            if (_f(p.get("contracts")) or 0) <= 0:
                continue
            try:
                c.close_position(p["symbol"], "buy" if p.get("side") == "long" else "sell")
                done.append(p["symbol"].split(":")[0])
            except Exception as e:  # noqa: BLE001
                failed.append(f"{p['symbol']}: {e}")
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
        if qty <= 0:
            raise ValueError("Betrag zu klein fuer die Mindestmenge")
        params = {"marginMode": mm, "stopLoss": {"triggerPrice": sl}}
        if tp:
            params["takeProfit"] = {"triggerPrice": tp}
        o = c.create_order(symbol, "limit" if order_type == "limit" else "market",
                           "buy" if side == "long" else "sell", qty, price if order_type == "limit" else None, params)
        self.refresh(force=True)
        return f"Order {o.get('id')} {side} {qty:g} {symbol} gesendet"


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
