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
    if api:
        mode_safe(client)
    return client


MODE_ERR = "40774"   # Bitget: Order passt nicht zum Positionsmodus (One-Way / Hedge)


def _flip_mode(client) -> None:
    client._bot_hedged = not is_hedged(client)
    log.warning("Bitget meldet anderen Positionsmodus - nutze jetzt %s", "Hedge" if client._bot_hedged else "One-Way")


def mode_safe(client) -> None:
    """Alle Order-Aufrufe dieses Clients automatisch an den Positionsmodus des Kontos anpassen.
    Meldet Bitget trotzdem 40774, wird einmal mit dem anderen Modus wiederholt."""
    raw_create, raw_close = client.create_order, client.close_position
    raw_tpsl, raw_pos = client.privateMixPostV2MixOrderPlaceTpslOrder, client.privateMixPostV2MixOrderPlacePosTpsl

    def create_order(symbol, type, side, amount, price=None, params=None):  # noqa: A002 - ccxt-Name
        p = dict(params or {})
        for attempt in (0, 1):
            p["hedged"] = is_hedged(client)
            try:
                return raw_create(symbol, type, side, amount, price, p)
            except Exception as e:  # noqa: BLE001
                if MODE_ERR not in str(e) or attempt:
                    raise
                _flip_mode(client)

    def close_position(symbol, side=None, params=None):
        pos_side = {"buy": "long", "sell": "short"}.get(side, side)   # beide Schreibweisen annehmen
        for attempt in (0, 1):
            s = pos_side if is_hedged(client) else ("buy" if pos_side == "long" else "sell")
            try:
                return raw_close(symbol, s, params or {})
            except Exception as e:  # noqa: BLE001
                if MODE_ERR not in str(e) or attempt:
                    raise
                _flip_mode(client)

    def with_hold(raw):
        def call(req, *a, **k):
            pos_side = {"buy": "long", "sell": "short"}.get(req.get("holdSide"), req.get("holdSide"))
            for attempt in (0, 1):
                h = pos_side if is_hedged(client) else ("buy" if pos_side == "long" else "sell")
                try:
                    return raw({**req, "holdSide": h}, *a, **k)
                except Exception as e:  # noqa: BLE001
                    if MODE_ERR not in str(e) or attempt:
                        raise
                    _flip_mode(client)
        return call

    client.create_order, client.close_position = create_order, close_position
    client.privateMixPostV2MixOrderPlaceTpslOrder = with_hold(raw_tpsl)
    client.privateMixPostV2MixOrderPlacePosTpsl = with_hold(raw_pos)


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


def is_hedged(client) -> bool:
    """Steht das Bitget-Konto im Hedge-Modus (Long und Short getrennt) statt One-Way? (einmal je Verbindung)"""
    cached = getattr(client, "_bot_hedged", None)
    if cached is not None:
        return cached
    hedged = False
    try:
        client.load_markets()
        product_type, _ = client.handle_product_type_and_params(client.market("BTC/USDT:USDT"), {})
        r = client.privateMixGetV2MixAccountAccounts({"productType": product_type})
        hedged = any(a.get("posMode") == "hedge_mode" for a in (r.get("data") or []))
    except Exception as e:  # noqa: BLE001 - im Zweifel One-Way (Standard des Bots)
        log.debug("Positionsmodus: %s", e)
    try:
        client._bot_hedged = hedged
    except AttributeError:
        pass
    log.info("Bitget Positionsmodus: %s", "Hedge (Long/Short getrennt)" if hedged else "One-Way")
    return hedged


def hold_side(client, side: str) -> str:
    """holdSide fuer TP/SL-Auftraege: One-Way 'buy'/'sell', Hedge 'long'/'short'."""
    if is_hedged(client):
        return "long" if side == "long" else "short"
    return "buy" if side == "long" else "sell"


def place_pos_tpsl(client, symbol: str, side: str, sl: float | None, tp: float | None) -> None:
    """Stop-Loss und/oder Take-Profit fuer die ganze Position auf Bitget setzen (place-pos-tpsl)."""
    if sl is None and tp is None:
        return
    market = client.market(symbol)
    product_type, _ = client.handle_product_type_and_params(market, {})
    base = {"symbol": market["id"], "productType": product_type, "marginCoin": "USDT"}
    if sl is not None:
        base.update(stopLossTriggerPrice=client.price_to_precision(symbol, sl), stopLossTriggerType="mark_price")
    if tp is not None:
        base.update(stopSurplusTriggerPrice=client.price_to_precision(symbol, tp),
                    stopSurplusTriggerType="fill_price")
    try:
        client.privateMixPostV2MixOrderPlacePosTpsl({**base, "holdSide": hold_side(client, side)})
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Stop/Ziel konnte nicht gesetzt werden: {e}") from e


def place_size_tpsl(client, symbol: str, side: str, plan: str, trigger: float, size: float) -> str:
    """Take-Profit (plan='profit_plan') oder Stop-Loss ('loss_plan') fuer eine TEILMENGE der Position
    (Bitget place-tpsl-order mit size). So bekommt jede Einzel-Position ihr eigenes Ziel und ihren Stop."""
    market = client.market(symbol)
    product_type, _ = client.handle_product_type_and_params(market, {})
    base = {"symbol": market["id"], "productType": product_type, "marginCoin": "USDT", "planType": plan,
            "triggerPrice": client.price_to_precision(symbol, trigger),
            "triggerType": "mark_price" if plan == "loss_plan" else "fill_price",
            "executePrice": "0", "size": client.amount_to_precision(symbol, size)}
    try:
        r = client.privateMixPostV2MixOrderPlaceTpslOrder({**base, "holdSide": hold_side(client, side)})
        return str((r.get("data") or {}).get("orderId") or "")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{'Ziel' if plan == 'profit_plan' else 'Stop'} konnte nicht gesetzt werden: {e}") from e


def cancel_plan(client, symbol: str, order_id: str) -> None:
    """TP/SL-Auftrag (Plan-Order) stornieren."""
    market = client.market(symbol)
    product_type, _ = client.handle_product_type_and_params(market, {})
    client.privateMixPostV2MixOrderCancelPlanOrder({
        "symbol": market["id"], "productType": product_type, "marginCoin": "USDT",
        "orderIdList": [{"orderId": order_id}], "planType": "profit_loss"})


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
    def _protect(self, sl: float, tp: float | None) -> dict:
        """Stop-Loss immer; Take-Profit nur, wenn er nicht als eigene Limit-Order liegt (tp=None)."""
        p = {"marginMode": self.cfg["margin_mode"], "stopLoss": {"triggerPrice": sl}, "hedged": is_hedged(self.c)}
        if tp is not None:
            p["takeProfit"] = {"triggerPrice": tp}
        return p

    def open(self, symbol: str, side: str, amount: float, sl: float, tp: float | None) -> float:
        """Market-Order mit Stop-Loss (und ggf. Take-Profit) direkt auf der Boerse."""
        order = self.c.create_order(
            symbol, "market", "buy" if side == "long" else "sell", amount, None, self._protect(sl, tp),
        )
        time.sleep(1)
        pos = self.positions().get(symbol)
        if pos:
            return pos["entry"]
        return float(order.get("average") or order.get("price") or self.last_price(symbol))

    def place_limit(self, symbol: str, side: str, amount: float, price: float, sl: float,
                    tp: float | None) -> str:
        """Post-Only-Limit-Order (Maker-Gebuehr) mit Stop-Loss (und ggf. Take-Profit).

        Liegt der Kurs schon auf der anderen Seite (Post-Only wuerde sofort ausfuehren und wird
        abgelehnt), wird eine normale Limit-Order gesetzt: sofort ausgefuehrt zum Limit oder besser."""
        args = (symbol, "limit", "buy" if side == "long" else "sell", amount, price)
        try:
            order = self.c.create_order(*args, {**self._protect(sl, tp), "postOnly": True})
        except ccxt.ExchangeError as e:  # Bitget meldet das je nach Fall unterschiedlich
            log.info("%s Post-Only abgelehnt (%s) - normale Limit-Order", symbol, e)
            order = self.c.create_order(*args, self._protect(sl, tp))
        return str(order["id"])

    def place_tp_limit(self, symbol: str, side: str, amount: float, price: float) -> str:
        """Take-Profit als liegende Limit-Order (nur reduzierend, Maker-Gebuehr)."""
        order = self.c.create_order(
            symbol, "limit", "sell" if side == "long" else "buy", amount, price,
            {"reduceOnly": True, "postOnly": True, "marginMode": self.cfg["margin_mode"], "hedged": is_hedged(self.c)},
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

    def set_stop(self, symbol: str, side: str, sl: float, tp: float | None) -> None:
        """Stop-Loss/Take-Profit der ganzen Position neu setzen (Bitget place-pos-tpsl)."""
        place_pos_tpsl(self.c, symbol, side, sl, tp)

    def close(self, symbol: str, side: str, amount: float) -> None:
        self.c.create_order(
            symbol, "market", "sell" if side == "long" else "buy", amount, None,
            {"reduceOnly": True, "marginMode": self.cfg["margin_mode"], "hedged": is_hedged(self.c)},
        )

    def closed_pnl(self, symbol: str, since_ms: int) -> float | None:
        try:
            # Die Position wurde evtl. schon etwas VOR der Registrierung im Bot eroeffnet
            # (Limit-Order ausgefuehrt) -> mit Vorlauf suchen, aber nur Positionen nehmen,
            # die NACH der Registrierung geschlossen wurden.
            def closed_at(h):
                return h.get("lastUpdateTimestamp") or h.get("timestamp") or 0
            hist = [h for h in self.c.fetch_positions_history([symbol], since_ms - 86_400_000)
                    if h.get("symbol") in (None, symbol) and closed_at(h) >= since_ms]
            if hist:
                return float(max(hist, key=closed_at).get("realizedPnl") or 0)
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

    def place_tp_limit(self, symbol, side, amount, price):
        if symbol in self.pos:
            self.pos[symbol].update(tp=price, tp_maker=True)
        return f"paper-tp-{symbol}-{price}"

    def place_limit(self, symbol, side, amount, price, sl, tp):
        oid = f"paper-{symbol}-{len(self.closed)}-{price}"
        self.orders[symbol] = {"id": oid, "side": side, "amount": amount, "price": price, "sl": sl, "tp": tp}
        return oid

    def cancel(self, symbol, order_id):
        if self.orders.get(symbol, {}).get("id") == order_id:
            del self.orders[symbol]
        elif str(order_id).startswith("paper-tp-") and symbol in self.pos:
            self.pos[symbol].update(tp=None, tp_maker=False)

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
            self.pos[symbol]["sl"] = sl
            if tp is not None:
                self.pos[symbol]["tp"] = tp

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
            tp = p.get("tp")
            hit_tp = tp is not None and (price >= tp if p["side"] == "long" else price <= tp)
            if hit_sl:
                # Stop ist eine Markt-Order: bei einer Luecke wird zum schlechteren Kurs ausgefuehrt
                fill = min(p["sl"], price) if p["side"] == "long" else max(p["sl"], price)
                self._exit(sym, fill)
            elif hit_tp:
                self._exit(sym, tp, maker=p.get("tp_maker", False))

    def _exit(self, sym, price, maker=False):
        p = self.pos.pop(sym)
        sign = 1 if p["side"] == "long" else -1
        fill = price if maker else price * (1 - sign * self.slip)
        pnl = sign * (fill - p["entry"]) * p["amount"] - fill * p["amount"] * (self.maker if maker else self.fee)
        self.cash += pnl
        # Einstiegsgebuehr (schon beim Oeffnen abgezogen) und Teilgewinne mit einrechnen
        self.closed[sym] = pnl + p["realized"] - p["fee_in"]
