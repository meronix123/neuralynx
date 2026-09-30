"""Bitget-Anbindung (live/demo) und Papier-Simulation mit gleicher Schnittstelle."""
import logging
import time

import ccxt
import pandas as pd

log = logging.getLogger("bot")

ALIASES = {
    "GOLD": ["XAU/USDT:USDT", "XAUT/USDT:USDT", "PAXG/USDT:USDT"],
    "SILVER": ["XAG/USDT:USDT"],
}


def make_client(api: dict | None = None, demo: bool = False) -> ccxt.bitget:
    params = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    if api:
        params.update(apiKey=api["key"], secret=api["secret"], password=api["password"])
    client = ccxt.bitget(params)
    if demo:
        client.enable_demo_trading(True)
    return client


def resolve_symbols(client: ccxt.Exchange, wanted: list[str]) -> list[str]:
    markets = client.load_markets()
    out = []
    for w in wanted:
        for cand in ALIASES.get(w.upper(), [w]):
            m = markets.get(cand)
            if m and m.get("swap") and m.get("active", True):
                out.append(cand)
                break
        else:
            log.warning("Symbol %s auf Bitget nicht gefunden - wird uebersprungen", w)
    return out


METALS = {"XAU", "XAUT", "PAXG", "XAG"}


def is_metal(symbol: str) -> bool:
    return symbol.split("/")[0] in METALS


def spread_from_ticker(t: dict) -> float | None:
    bid, ask = t.get("bid"), t.get("ask")
    if not bid or not ask:
        return None
    return (ask - bid) / ((ask + bid) / 2) * 100


def to_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


class BitgetExchange:
    """Echter Handel (mode live) oder Bitget-Demokonto (mode demo)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.c = make_client(cfg["api"], demo=cfg["mode"] == "demo")
        self.symbols = resolve_symbols(self.c, cfg["symbols"])

    # --- Daten -------------------------------------------------------
    def candles(self, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
        return to_df(self.c.fetch_ohlcv(symbol, tf, limit=limit))

    def funding_rate(self, symbol: str) -> float | None:
        try:
            return self.c.fetch_funding_rate(symbol).get("fundingRate")
        except Exception as e:  # noqa: BLE001 - Funding ist nur ein Filter
            log.debug("Funding %s: %s", symbol, e)
            return None

    def last_price(self, symbol: str) -> float:
        return float(self.c.fetch_ticker(symbol)["last"])

    def spread_pct(self, symbol: str) -> float | None:
        return spread_from_ticker(self.c.fetch_ticker(symbol))

    def equity(self) -> float:
        bal = self.c.fetch_balance({"type": "swap"})
        return float(bal.get("USDT", {}).get("total") or 0.0)

    def amount_rules(self, symbol: str) -> tuple[float, float]:
        m = self.c.market(symbol)
        return float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0)

    # --- Setup -------------------------------------------------------
    def setup(self, symbol: str) -> None:
        lev, mm = self.cfg["leverage"], self.cfg["margin_mode"]
        for fn, desc in (
            (lambda: self.c.set_position_mode(False, symbol), "One-Way-Modus"),
            (lambda: self.c.set_margin_mode(mm, symbol, {"marginCoin": "USDT"}), "Margin-Modus"),
            (lambda: self.c.set_leverage(lev, symbol, {"holdSide": "long", "marginMode": mm}), "Hebel long"),
            (lambda: self.c.set_leverage(lev, symbol, {"holdSide": "short", "marginMode": mm}), "Hebel short"),
        ):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - "schon gesetzt" ist kein Fehler
                log.debug("%s %s: %s", symbol, desc, e)

    # --- Handel ------------------------------------------------------
    def open(self, symbol: str, side: str, amount: float, sl: float, tp: float) -> float:
        """Market-Order mit Stop-Loss und Take-Profit direkt auf der Boerse."""
        order = self.c.create_order(
            symbol,
            "market",
            "buy" if side == "long" else "sell",
            amount,
            None,
            {
                "marginMode": self.cfg["margin_mode"],
                "stopLoss": {"triggerPrice": sl},
                "takeProfit": {"triggerPrice": tp},
            },
        )
        time.sleep(1)
        pos = self.positions().get(symbol)
        if pos:
            return pos["entry"]
        return float(order.get("average") or order.get("price") or self.last_price(symbol))

    def place_limit(self, symbol: str, side: str, amount: float, price: float, sl: float, tp: float) -> str:
        """Post-Only-Limit-Order (Maker-Gebuehr) mit Stop-Loss/Take-Profit."""
        order = self.c.create_order(
            symbol, "limit", "buy" if side == "long" else "sell", amount, price,
            {
                "marginMode": self.cfg["margin_mode"],
                "postOnly": True,
                "stopLoss": {"triggerPrice": sl},
                "takeProfit": {"triggerPrice": tp},
            },
        )
        return str(order["id"])

    def cancel(self, symbol: str, order_id: str) -> None:
        self.c.cancel_order(order_id, symbol)

    def positions(self) -> dict:
        out = {}
        for p in self.c.fetch_positions(self.symbols):
            if float(p.get("contracts") or 0) > 0:
                out[p["symbol"]] = {
                    "side": p["side"],
                    "amount": float(p["contracts"]),
                    "entry": float(p["entryPrice"]),
                }
        return out

    def set_stop(self, symbol: str, side: str, sl: float, tp: float) -> None:
        """Stop-Loss/Take-Profit der ganzen Position neu setzen (Bitget place-pos-tpsl)."""
        market = self.c.market(symbol)
        product_type, _ = self.c.handle_product_type_and_params(market, {})
        base = {
            "symbol": market["id"],
            "productType": product_type,
            "marginCoin": "USDT",
            "stopLossTriggerPrice": self.c.price_to_precision(symbol, sl),
            "stopLossTriggerType": "mark_price",
            "stopSurplusTriggerPrice": self.c.price_to_precision(symbol, tp),
            "stopSurplusTriggerType": "fill_price",
        }
        # One-Way-Modus erwartet je nach Konto "buy"/"sell" oder "long"/"short"
        hold = ["buy", "long"] if side == "long" else ["sell", "short"]
        last_err = None
        for h in hold:
            try:
                self.c.privateMixPostV2MixOrderPlacePosTpsl({**base, "holdSide": h})
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise RuntimeError(f"Stop konnte nicht gesetzt werden: {last_err}")

    def close(self, symbol: str, side: str, amount: float) -> None:
        self.c.create_order(
            symbol, "market", "sell" if side == "long" else "buy", amount, None,
            {"reduceOnly": True, "marginMode": self.cfg["margin_mode"]},
        )

    def closed_pnl(self, symbol: str, since_ms: int) -> float | None:
        try:
            hist = self.c.fetch_positions_history([symbol], since_ms)
            if hist:
                return float(hist[-1].get("realizedPnl") or 0)
        except Exception as e:  # noqa: BLE001
            log.debug("PnL-Historie %s: %s", symbol, e)
        return None


class PaperExchange:
    """Simulation mit echten Bitget-Kursen. Kein API-Schluessel noetig."""

    def __init__(self, cfg: dict, client: ccxt.Exchange | None = None):
        self.cfg = cfg
        self.c = client or make_client()
        self.symbols = resolve_symbols(self.c, cfg["symbols"])
        self.cash = float(cfg["paper"]["start_equity"])
        self.pos: dict = {}
        self.orders: dict = {}
        self.closed: dict = {}
        self.maker = cfg["fees"].get("maker", cfg["fees"]["taker"])
        self.fee = cfg["fees"]["taker"]
        self.slip = cfg["fees"]["slippage"]

    def candles(self, symbol, tf, limit=300):
        return to_df(self.c.fetch_ohlcv(symbol, tf, limit=limit))

    def funding_rate(self, symbol):
        try:
            return self.c.fetch_funding_rate(symbol).get("fundingRate")
        except Exception:  # noqa: BLE001
            return None

    def last_price(self, symbol):
        return float(self.c.fetch_ticker(symbol)["last"])

    def spread_pct(self, symbol):
        return spread_from_ticker(self.c.fetch_ticker(symbol))

    def amount_rules(self, symbol):
        m = self.c.market(symbol)
        return float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0)

    def setup(self, symbol):
        pass

    def equity(self):
        eq = self.cash
        for sym, p in self.pos.items():
            price = self.last_price(sym)
            sign = 1 if p["side"] == "long" else -1
            eq += sign * (price - p["entry"]) * p["amount"]
        return eq

    def open(self, symbol, side, amount, sl, tp):
        price = self.last_price(symbol)
        entry = price * (1 + self.slip) if side == "long" else price * (1 - self.slip)
        self.cash -= entry * amount * self.fee
        self.pos[symbol] = {"side": side, "amount": amount, "entry": entry, "sl": sl, "tp": tp,
                            "realized": 0.0, "fee_in": entry * amount * self.fee}
        return entry

    def dump(self) -> dict:
        """Spielgeld-Konto sichern (ueberlebt Neustarts)."""
        return {"cash": self.cash, "pos": self.pos, "orders": self.orders, "closed": self.closed}

    def restore(self, d: dict) -> None:
        self.cash = float(d.get("cash", self.cash))
        self.pos = d.get("pos", {})
        self.orders = d.get("orders", {})
        self.closed = d.get("closed", {})

    def place_limit(self, symbol, side, amount, price, sl, tp):
        oid = f"paper-{symbol}-{len(self.closed)}-{price}"
        self.orders[symbol] = {"id": oid, "side": side, "amount": amount, "price": price, "sl": sl, "tp": tp}
        return oid

    def cancel(self, symbol, order_id):
        if self.orders.get(symbol, {}).get("id") == order_id:
            del self.orders[symbol]

    def _check_orders(self):
        for sym in list(self.orders):
            o = self.orders[sym]
            price = self.last_price(sym)
            if (price <= o["price"]) if o["side"] == "long" else (price >= o["price"]):
                del self.orders[sym]
                self.cash -= o["price"] * o["amount"] * self.maker
                self.pos[sym] = {"side": o["side"], "amount": o["amount"], "entry": o["price"],
                                 "sl": o["sl"], "tp": o["tp"], "realized": 0.0,
                                 "fee_in": o["price"] * o["amount"] * self.maker}

    def positions(self):
        self._check_orders()
        self._check_exits()
        return {s: {k: p[k] for k in ("side", "amount", "entry")} for s, p in self.pos.items()}

    def set_stop(self, symbol, side, sl, tp):
        if symbol in self.pos:
            self.pos[symbol].update(sl=sl, tp=tp)

    def close(self, symbol, side, amount):
        p = self.pos.get(symbol)
        if p and amount < p["amount"] - 1e-12:  # Teilverkauf
            sign = 1 if p["side"] == "long" else -1
            fill = self.last_price(symbol) * (1 - sign * self.slip)
            pnl = sign * (fill - p["entry"]) * amount - fill * amount * self.fee
            self.cash += pnl
            p["realized"] += pnl
            p["amount"] -= amount
            return
        self._exit(symbol, self.last_price(symbol))

    def closed_pnl(self, symbol, since_ms):
        return self.closed.pop(symbol, None)

    def _check_exits(self):
        for sym in list(self.pos):
            p = self.pos[sym]
            price = self.last_price(sym)
            hit_sl = price <= p["sl"] if p["side"] == "long" else price >= p["sl"]
            hit_tp = price >= p["tp"] if p["side"] == "long" else price <= p["tp"]
            if hit_sl or hit_tp:
                self._exit(sym, p["sl"] if hit_sl else p["tp"])

    def _exit(self, sym, price):
        p = self.pos.pop(sym)
        sign = 1 if p["side"] == "long" else -1
        fill = price * (1 - sign * self.slip)
        pnl = sign * (fill - p["entry"]) * p["amount"] - fill * p["amount"] * self.fee
        self.cash += pnl
        # Einstiegsgebuehr (schon beim Oeffnen abgezogen) und Teilgewinne mit einrechnen
        self.closed[sym] = pnl + p["realized"] - p["fee_in"]
