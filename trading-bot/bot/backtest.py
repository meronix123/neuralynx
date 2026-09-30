"""Backtest: dieselbe Strategie und dieselben Risiko-Regeln auf historischen Daten.

Annahmen (bewusst vorsichtig):
- Market-Einstieg zum Eroeffnungskurs des naechsten Bars + Slippage
- Limit-Einstieg (entry_order: limit) zum Signalkurs, nur wenn der naechste Bar ihn erreicht
- Werden Stop und Ziel im selben Bar beruehrt, zaehlt der Stop
- Taker-Gebuehr beim Ausstieg, beim Einstieg Taker (Market) bzw. Maker (Limit)
- Funding-Kosten und Funding-Filter werden NICHT simuliert
"""
import json
import time
from datetime import datetime, timezone

import pandas as pd

from .config import ROOT
from .exchange import is_metal, make_client, resolve_symbols, to_df
from .filters import health_gate, is_leader, leader_blocks, time_stop_due
from .ml import SignalModel, features
from .risk import RiskGuard, position_size, round_amount, stop_is_safe
from .strategy import TF_MS, compute_signals, levels_from, trail_stop

DATA = ROOT / "data"


def fetch_history(client, symbol: str, tf: str, days: int) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"{symbol.replace('/', '_').replace(':', '_')}_{tf}_{days}d.csv"
    if cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
        cached = pd.read_csv(cache)
        if len(cached) > 1:  # leere Datei von einem frueheren Fehlversuch ignorieren
            return cached
    since = client.milliseconds() - days * 86_400_000
    rows = []
    step_ms = 200 * TF_MS[tf]
    while since < client.milliseconds():
        batch = client.fetch_ohlcv(symbol, tf, since=since, limit=200)
        if not batch:
            # Markt gab es zu diesem Zeitpunkt noch nicht -> ein Stueck weiter vorne probieren
            since += step_ms
            continue
        rows += batch
        since = batch[-1][0] + TF_MS[tf]
        done = min(100, 100 * (1 - (client.milliseconds() - since) / (days * 86_400_000)))
        print(f"\r  {done:5.1f} %", end="", flush=True)
        if since >= client.milliseconds():
            break
    print()
    df = to_df(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    if len(df):
        df = df.astype({"ts": "int64", "open": float, "high": float, "low": float,
                        "close": float, "volume": float})
        df.to_csv(cache, index=False)
        first = pd.to_datetime(df["ts"].iloc[0], unit="ms").date()
        if (df["ts"].iloc[0] - (client.milliseconds() - days * 86_400_000)) > 7 * 86_400_000:
            print(f"  Hinweis: {symbol} gibt es auf Bitget erst seit {first} - weniger Daten.")
    return df


MIN_BARS = 500


def usable(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Maerkte mit zu wenig Kursdaten weglassen (statt abzustuerzen)."""
    out = {}
    for sym, df in data.items():
        if len(df) >= MIN_BARS:
            out[sym] = df
        else:
            print(f"  {sym}: nur {len(df)} Kerzen - wird im Test uebersprungen.")
    return out


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    x = df.set_index(pd.to_datetime(df["ts"], unit="ms"))
    r = x.resample(pd.Timedelta(milliseconds=TF_MS[tf]), origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    # unabhaengig von der internen Zeitaufloesung (ns/us/ms) in Millisekunden umrechnen
    r["ts"] = (r.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    return r.reset_index(drop=True)[["ts", "open", "high", "low", "close", "volume"]]


PREP_COLS = ("ts", "open", "high", "low", "close", "atr", "signal", "sl_dist", "tp_dist", "strategy", "regime",
             "rsi", "adx", "bb_width", "atr_pct", "vol_ratio", "macd_n", "trend")
FEATURE_COLS = ("rsi", "adx", "bb_width", "atr_pct", "vol_ratio", "macd_n", "trend", "regime", "strategy")


def prepare(cfg: dict, data: dict[str, pd.DataFrame], base_tf: str) -> dict:
    """Signale einmal berechnen (teuer). data: Symbol -> Kerzen in base_tf."""
    s, tf, ttf = cfg["strategy"], cfg["timeframe"], cfg["trend_timeframe"]
    out = {}
    for sym, raw in data.items():
        df = raw if tf == base_tf else resample(raw, tf)
        d = compute_signals(df, resample(raw, ttf), s, tf, ttf)
        out[sym] = {c: d[c].tolist() for c in PREP_COLS}
    # Leitwaehrung: Marktlage von BTC je Zeitpunkt fuer die anderen Krypto-Maerkte
    leader = next((k for k in out if is_leader(k)), None)
    if leader:
        lr = dict(zip(out[leader]["ts"], out[leader]["regime"]))
        for sym, p in out.items():
            p["leader_regime"] = [None if is_leader(sym) or is_metal(sym) else lr.get(t) for t in p["ts"]]
    return out


def simulate(cfg: dict, prep: dict, rules: dict[str, tuple[float, float]],
             start_ts: int | None = None, end_ts: int | None = None,
             seed_examples: list[dict] | None = None) -> dict:
    """Handelt die vorbereiteten Signale im Zeitfenster [start_ts, end_ts).

    Positionsfuehrung (strategy):
      partial_tp_r / partial_tp_frac: bei +X R einen Teil verkaufen, Stop auf Einstand
      pyramid_at_r / pyramid_max_adds / pyramid_size_frac: im GEWINN aufstocken
        (nie im Verlust), Stop danach mindestens auf den neuen Einstand
    """
    s, r, f = cfg["strategy"], cfg["risk"], cfg["fees"]
    taker, slip, lev = f["taker"], f["slippage"], cfg["leverage"]
    limit_entry = f.get("entry_order", "market") == "limit"
    entry_fee = f.get("maker", taker) if limit_entry else taker
    part_r, part_frac = s.get("partial_tp_r", 0), s.get("partial_tp_frac", 0.5)
    pyr_r, pyr_max, pyr_frac = s.get("pyramid_at_r", 0), s.get("pyramid_max_adds", 0), s.get("pyramid_size_frac", 0.5)
    use_leader = s.get("leader_filter", False)
    use_ml, ml_thr = s.get("ml_filter", False), s.get("ml_threshold", 0.45)
    ml_min, ml_every = s.get("ml_min_trades", 40), 10
    model, examples, trained_at = SignalModel(), list(seed_examples or []), 0
    strat_pnls: dict[str, list[float]] = {}
    health_skips: dict[str, int] = {}
    skipped = {"leader": 0, "health": 0, "ml": 0}

    index = {sym: {t: i for i, t in enumerate(p["ts"])} for sym, p in prep.items()}
    timeline = sorted(t for t in set().union(*[set(p["ts"]) for p in prep.values()])
                      if (start_ts is None or t >= start_ts) and (end_ts is None or t < end_ts))

    start_equity = float(cfg["paper"]["start_equity"])
    equity = start_equity
    guard = RiskGuard(r, {})
    pos: dict = {}
    pending: dict = {}
    trades = []
    curve = []

    def be(p):  # Einstand inkl. Gebuehren
        return p["entry"] * (1 + 2 * taker) if p["side"] == "long" else p["entry"] * (1 - 2 * taker)

    def raise_stop(p, level):
        p["sl"] = max(p["sl"], level) if p["side"] == "long" else min(p["sl"], level)

    def close(sym, price, ts, why):
        nonlocal equity
        p = pos.pop(sym)
        sign = 1 if p["side"] == "long" else -1
        fill = price * (1 - sign * slip)
        pnl = p["realized"] + sign * (fill - p["entry"]) * p["amount"] - fill * p["amount"] * taker - p["fees"]
        equity += pnl
        guard.on_close(pnl, datetime.fromtimestamp(ts / 1000, timezone.utc))
        trades.append({"symbol": sym, "side": p["side"], "entry": p["entry0"], "exit": fill,
                       "pnl": pnl, "why": why, "opened": p["ts"], "closed": ts,
                       "partial": p["partial_done"], "adds": p["adds"], "strategy": p["strategy"]})
        strat_pnls.setdefault(p["strategy"], []).append(pnl)
        examples.append({"x": p["x"], "win": int(pnl > 0)})

    for ts in timeline:
        now = datetime.fromtimestamp(ts / 1000, timezone.utc)
        guard.update_day(equity, now)
        for sym, d in prep.items():
            i = index[sym].get(ts)
            if i is None:
                continue
            o, h, lo, c, a, sig = d["open"][i], d["high"][i], d["low"][i], d["close"][i], d["atr"][i], d["signal"][i]
            step, min_amt = rules.get(sym, (0, 0))

            # 1) ausstehenden Einstieg ausfuehren
            if sym in pending:
                side, sig_atr, ref, sl_dist, tp_dist, strat, x = pending.pop(sym)
                if limit_entry:
                    filled = lo <= ref if side == "long" else h >= ref
                    entry = min(ref, o) if side == "long" else max(ref, o)
                else:
                    filled = True
                    entry = o * (1 + slip if side == "long" else 1 - slip)
                ok, _ = guard.can_open(equity, len(pos), now)
                same = sum(1 for p in pos.values() if p["side"] == side)
                if filled and ok and same < r.get("max_same_direction", 99):
                    sl, tp = levels_from(side, entry, sl_dist, tp_dist)
                    if stop_is_safe(entry, sl, lev):
                        amt = position_size(equity, entry, sl, lev, r["risk_per_trade_pct"],
                                            r["max_margin_per_trade_pct"], taker)
                        amt = round_amount(amt, step, min_amt)
                        if amt > 0:
                            pos[sym] = {"side": side, "entry": entry, "entry0": entry, "amount": amt,
                                        "amount0": amt, "fees": entry * amt * entry_fee, "realized": 0.0,
                                        "sl_init": sl, "sl": sl, "tp": tp, "r0": abs(entry - sl), "ts": ts,
                                        "partial_done": False, "adds": 0, "strategy": strat, "x": x,
                                        "bars": 0}
                            guard.on_open()

            # 2) Stop / Teilverkauf / Ziel im Bar (Reihenfolge vorsichtig: Stop zuerst)
            if sym in pos:
                p = pos[sym]
                sign = 1 if p["side"] == "long" else -1
                hit_sl = lo <= p["sl"] if sign == 1 else h >= p["sl"]
                if hit_sl:
                    close(sym, min(p["sl"], o) if sign == 1 else max(p["sl"], o), ts, "stop")
                else:
                    if part_r and not p["partial_done"]:
                        lvl = p["entry0"] + sign * part_r * p["r0"]
                        if (h >= lvl) if sign == 1 else (lo <= lvl):
                            qty = round_amount(p["amount"] * part_frac, step, min_amt)
                            if 0 < qty < p["amount"]:
                                fill = lvl * (1 - sign * slip)
                                p["realized"] += sign * (fill - p["entry"]) * qty - fill * qty * taker
                                p["amount"] -= qty
                                p["partial_done"] = True
                                raise_stop(p, be(p))
                    hit_tp = h >= p["tp"] if sign == 1 else lo <= p["tp"]
                    if hit_tp:
                        close(sym, max(p["tp"], o) if sign == 1 else min(p["tp"], o), ts, "ziel")

            # 3) zum Bar-Schluss: Zeit-Stop, aufstocken (nur im Gewinn), Stop nachziehen
            if sym in pos:
                p = pos[sym]
                sign = 1 if p["side"] == "long" else -1
                progress = sign * (c - p["entry0"]) / p["r0"]
                p["bars"] += 1
                if time_stop_due(p["bars"], progress, p["partial_done"], s):
                    close(sym, c, ts, "zeit")
                    continue
                if pyr_r and p["adds"] < pyr_max and progress >= pyr_r * (p["adds"] + 1):
                    qty = round_amount(p["amount0"] * pyr_frac, step, min_amt)
                    cap = equity * r["max_margin_per_trade_pct"] / 100 * lev
                    if qty > 0 and (p["amount"] + qty) * c <= cap:
                        fill = c * (1 + sign * slip)
                        new_entry = (p["entry"] * p["amount"] + fill * qty) / (p["amount"] + qty)
                        new_be = new_entry * (1 + sign * 2 * taker)
                        if (new_be < c) if sign == 1 else (new_be > c):
                            p["fees"] += fill * qty * taker
                            p["entry"], p["amount"] = new_entry, p["amount"] + qty
                            p["adds"] += 1
                            raise_stop(p, be(p))
                new_sl = trail_stop(p["side"], p["entry0"], p["sl_init"], p["sl"], c, a, s, taker)
                if new_sl is not None:
                    raise_stop(p, new_sl)
            # 4) neues Signal -> Filter -> Einstieg im naechsten Bar
            elif sig != 0:
                side = "long" if sig == 1 else "short"
                strat = d["strategy"][i]
                if use_leader and "leader_regime" in d and leader_blocks(side, d["leader_regime"][i]):
                    skipped["leader"] += 1
                    continue
                if not health_gate(strat, strat_pnls.get(strat, []), s, health_skips):
                    skipped["health"] += 1
                    continue
                x = features({k: d[k][i] for k in FEATURE_COLS if k in d}, side)
                if use_ml:
                    if len(examples) >= ml_min and len(examples) - trained_at >= ml_every:
                        model.fit([e["x"] for e in examples], [e["win"] for e in examples])
                        trained_at = len(examples)
                    prob = model.proba(x)
                    if prob is not None and prob < ml_thr:
                        skipped["ml"] += 1
                        continue
                pending[sym] = (side, a, c, d["sl_dist"][i], d["tp_dist"][i], strat, x)
        curve.append(equity)

    for sym in list(pos):  # offene Positionen am Ende zum letzten Kurs schliessen
        i = index[sym][max(t for t in prep[sym]["ts"] if t <= timeline[-1])]
        close(sym, prep[sym]["close"][i], timeline[-1], "ende")
    curve.append(equity)

    res = summarize(trades, curve, start_equity)
    res["skipped"] = skipped
    res["examples"] = examples
    days = (timeline[-1] - timeline[0]) / 86_400_000 if len(timeline) > 1 else 0
    res["trades_per_day"] = res["trades"] / days if days else 0.0
    return res


def run_backtest(cfg: dict, data: dict[str, pd.DataFrame], rules: dict[str, tuple[float, float]],
                 base_tf: str | None = None) -> dict:
    """data: Symbol -> Kerzen (base_tf, Standard = cfg['timeframe'])."""
    return simulate(cfg, prepare(cfg, data, base_tf or cfg["timeframe"]), rules)


def _per_group(t: pd.DataFrame, col: str) -> pd.DataFrame:
    def pf(x):
        loss = -x[x <= 0].sum()
        return round(x[x > 0].sum() / loss, 2) if loss > 0 else float("inf")
    g = t.groupby(col).pnl
    return pd.DataFrame({"trades": g.count(), "summe": g.sum().round(2), "profit_faktor": g.apply(pf)})


def summarize(trades: list, curve: list, start: float) -> dict:
    t = pd.DataFrame(trades)
    peak, mdd = start, 0.0
    for e in curve:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak if peak > 0 else 0)
    if t.empty:
        return {"trades": 0, "start": start, "end": start, "max_drawdown_pct": 0.0, "table": t}
    wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
    gross_loss = -losses.pnl.sum()
    return {
        "trades": len(t),
        "winrate_pct": 100 * len(wins) / len(t),
        "profit_factor": wins.pnl.sum() / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win": wins.pnl.mean() if len(wins) else 0.0,
        "avg_loss": losses.pnl.mean() if len(losses) else 0.0,
        "start": start,
        "end": curve[-1] if curve else start,
        "return_pct": 100 * ((curve[-1] if curve else start) / start - 1),
        "max_drawdown_pct": 100 * mdd,
        "per_symbol": t.groupby("symbol").pnl.agg(["count", "sum"]).round(2),
        "per_strategy": _per_group(t, "strategy") if "strategy" in t else None,
        "table": t,
    }


def backtest_cli(cfg: dict, days: int) -> None:
    client = make_client()
    symbols = resolve_symbols(client, cfg["symbols"])
    data, rules = {}, {}
    for sym in symbols:
        print(f"Lade {days} Tage {sym} ...")
        data[sym] = fetch_history(client, sym, cfg["timeframe"], days)
        m = client.market(sym)
        rules[sym] = (float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0))
    data = usable(data)
    res = run_backtest(cfg, data, rules)
    print_report(res, days)
    save_ml_examples(res.get("examples", []))


ML_FILE = DATA / "ml_examples.json"


def save_ml_examples(examples: list[dict]) -> None:
    """Trainingsbeispiele fuer den Live-Filter (lernt aus Backtest-Trades)."""
    if examples:
        DATA.mkdir(exist_ok=True)
        ML_FILE.write_text(json.dumps(examples), encoding="utf-8")
        print(f"{len(examples)} Lernbeispiele fuer den ML-Filter gespeichert.")


def print_report(res: dict, days: int) -> None:
    print("\n========== BACKTEST ==========")
    print(f"Zeitraum:        {days} Tage")
    print(f"Trades:          {res['trades']}")
    if res["trades"]:
        print(f"Trefferquote:    {res['winrate_pct']:.1f} %")
        print(f"Profit-Faktor:   {res['profit_factor']:.2f}   (ueber 1.3 = brauchbar)")
        print(f"Schnitt Gewinn:  {res['avg_win']:+.2f} USDT")
        print(f"Schnitt Verlust: {res['avg_loss']:+.2f} USDT")
        print(f"Konto:           {res['start']:.2f} -> {res['end']:.2f} USDT ({res['return_pct']:+.1f} %)")
        print(f"Max. Rueckgang:  {res['max_drawdown_pct']:.1f} %")
        print("\nJe Symbol (Anzahl, Summe USDT):")
        print(res["per_symbol"].to_string())
        if res.get("per_strategy") is not None:
            print("\nJe Strategie:")
            print(res["per_strategy"].to_string())
        sk = res.get("skipped") or {}
        if any(sk.values()):
            print(f"\nAussortierte Signale: BTC-Filter {sk.get('leader', 0)}, "
                  f"Strategie pausiert {sk.get('health', 0)}, ML-Filter {sk.get('ml', 0)}")
    print("==============================")
    print("Hinweis: Vergangene Ergebnisse garantieren keine zukuenftigen Gewinne.")
