"""Bitget-Konto in der Oberflaeche: Guthaben, Positionen, Orders, Trade-Historie und Aktionen.

Die API-Schluessel werden in der Oberflaeche eingegeben, einmal gegen Bitget geprueft und
nur lokal in der Datei .env gespeichert (nie angezeigt, nie hochgeladen).
Aktionen (schliessen, stornieren, Stop/Ziel aendern, neue Order) wirken auf das ECHTE Konto
(bzw. das Demokonto, wenn "Demo" gewaehlt wurde) - unabhaengig davon, ob der Bot selbst
im Paper-Modus laeuft.
"""
import logging
import os
import threading
import time

from .config import ROOT
from .exchange import make_client, place_pos_tpsl

log = logging.getLogger("bot")
ENV_FILE = ROOT / ".env"
ENV_KEYS = {"key": "BITGET_API_KEY", "secret": "BITGET_API_SECRET", "password": "BITGET_API_PASSPHRASE"}
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
        self.client = None
        self.demo = False
        self.error = ""
        self.data: dict = {}
        self.updated = 0.0
        self.lock = threading.RLock()
        api = cfg.get("api") or {}
        if api.get("key") and api.get("secret") and api.get("password"):
            try:
                demo = cfg.get("mode") == "demo" or os.getenv("BITGET_DEMO") == "1"
                self.connect(api["key"], api["secret"], api["password"], demo, save=False)
            except Exception as e:  # noqa: BLE001 - Oberflaeche zeigt den Fehler an
                self.error = f"Gespeicherte Schluessel funktionieren nicht: {e}"

    # --- Verbindung ------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.client is not None

    def connect(self, key: str, secret: str, password: str, demo: bool = False, save: bool = True) -> None:
        key, secret, password = key.strip(), secret.strip(), password.strip()
        if not (key and secret and password):
            raise ValueError("API-Key, Secret und Passphrase eingeben")
        client = self.factory({"key": key, "secret": secret, "password": password}, demo=demo)
        client.fetch_balance({"type": "swap"})  # prueft die Schluessel (Fehler -> Exception)
        with self.lock:
            self.client, self.demo, self.error = client, demo, ""
        if save:
            save_env({ENV_KEYS["key"]: key, ENV_KEYS["secret"]: secret, ENV_KEYS["password"]: password,
                      "BITGET_DEMO": "1" if demo else "0"}, self.env_path)
        self.refresh(force=True)

    def disconnect(self, forget: bool = False) -> None:
        with self.lock:
            self.client, self.data, self.error = None, {}, ""
        if forget:
            save_env({v: "" for v in ENV_KEYS.values()}, self.env_path)

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
            return {"connected": self.connected, "demo": self.demo, "error": self.error,
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
