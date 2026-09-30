"""Backtest: dieselbe Strategie und dieselben Risiko-Regeln auf historischen Daten.

Annahmen (bewusst vorsichtig):
- Einstieg zum Eroeffnungskurs des naechsten Bars + Slippage
- Werden Stop und Ziel im selben Bar beruehrt, zaehlt der Stop
- Taker-Gebuehr bei Ein- und Ausstieg
- Funding-Kosten und Funding-Filter werden NICHT simuliert
"""
import time
from datetime import datetime, timezone

import pandas as pd

from .config import ROOT
from .exchange import make_client, resolve_symbols, to_df
from .risk import RiskGuard, position_size, round_amount, stop_is_safe
from .strategy import TF_MS, compute_signals, levels, trail_stop

DATA = ROOT / "data"


def fetch_history(client, symbol: str, tf: str, days: int) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"{symbol.replace('/', '_').replace(':', '_')}_{tf}_{days}d.csv"
    if cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
        return pd.read_csv(cache)
    since = client.milliseconds() - days * 86_400_000
    rows = []
    while True:
        batch = client.fetch_ohlcv(symbol, tf, since=since, limit=200)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + TF_MS[tf]
        if since >= client.milliseconds():
            break
    df = to_df(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    x = df.set_index(pd.to_datetime(df["ts"], unit="ms"))
    r = x.resample(pd.Timedelta(milliseconds=TF_MS[tf])).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    # unabhaengig von der internen Zeitaufloesung (ns/us/ms) in Millisekunden umrechnen
    r["ts"] = (r.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    return r.reset_index(drop=True)[["ts", "open", "high", "low", "close", "volume"]]


def run_backtest(cfg: dict, data: dict[str, pd.DataFrame], rules: dict[str, tuple[float, float]]) -> dict:
    """data: Symbol -> 5m-Kerzen. rules: Symbol -> (Mengen-Schritt, Mindestmenge)."""
    s, r = cfg["strategy"], cfg["risk"]
    fee, slip, lev = cfg["fees"]["taker"], cfg["fees"]["slippage"], cfg["leverage"]
    tf, ttf = cfg["timeframe"], cfg["trend_timeframe"]

    sig = {sym: compute_signals(df, resample(df, ttf), s, tf, ttf).set_index("ts") for sym, df in data.items()}
    timeline = sorted(set().union(*[d.index for d in sig.values()]))

    equity = float(cfg["paper"]["start_equity"])
    guard = RiskGuard(r, {})
    pos: dict = {}
    pending: dict = {}
    trades = []
    curve = []

    def close(sym, price, ts, why):
        nonlocal equity
        p = pos.pop(sym)
        sign = 1 if p["side"] == "long" else -1
        fill = price * (1 - sign * slip)
        pnl = sign * (fill - p["entry"]) * p["amount"] - (p["entry"] + fill) * p["amount"] * fee
        equity += pnl
        guard.on_close(pnl, datetime.fromtimestamp(ts / 1000, timezone.utc))
        trades.append({"symbol": sym, "side": p["side"], "entry": p["entry"], "exit": fill,
                       "pnl": pnl, "why": why, "opened": p["ts"], "closed": ts})

    for ts in timeline:
        now = datetime.fromtimestamp(ts / 1000, timezone.utc)
        guard.update_day(equity, now)
        for sym, d in sig.items():
            if ts not in d.index:
                continue
            bar = d.loc[ts]

            # 1) ausstehenden Einstieg zum Eroeffnungskurs ausfuehren
            if sym in pending:
                side, sig_atr = pending.pop(sym)
                entry = bar["open"] * (1 + slip if side == "long" else 1 - slip)
                sl, tp = levels(side, entry, sig_atr, s)
                ok, _ = guard.can_open(equity, len(pos), now)
                if ok and stop_is_safe(entry, sl, lev):
                    amt = position_size(equity, entry, sl, lev, r["risk_per_trade_pct"],
                                        r["max_margin_per_trade_pct"], fee)
                    amt = round_amount(amt, *rules.get(sym, (0, 0)))
                    if amt > 0:
                        pos[sym] = {"side": side, "entry": entry, "amount": amt,
                                    "sl_init": sl, "sl": sl, "tp": tp, "ts": ts}
                        guard.on_open()

            # 2) Stop / Ziel im Bar pruefen
            if sym in pos:
                p = pos[sym]
                if p["side"] == "long":
                    if bar["low"] <= p["sl"]:
                        close(sym, min(p["sl"], bar["open"]), ts, "stop")
                    elif bar["high"] >= p["tp"]:
                        close(sym, max(p["tp"], bar["open"]), ts, "ziel")
                else:
                    if bar["high"] >= p["sl"]:
                        close(sym, max(p["sl"], bar["open"]), ts, "stop")
                    elif bar["low"] <= p["tp"]:
                        close(sym, min(p["tp"], bar["open"]), ts, "ziel")

            # 3) Stop nachziehen (zum Bar-Schluss)
            if sym in pos:
                p = pos[sym]
                new_sl = trail_stop(p["side"], p["entry"], p["sl_init"], p["sl"],
                                    bar["close"], bar["atr"], s, fee)
                if new_sl is not None:
                    p["sl"] = new_sl

            # 4) neues Signal -> Einstieg im naechsten Bar
            elif bar["signal"] != 0:
                pending[sym] = ("long" if bar["signal"] == 1 else "short", bar["atr"])
        curve.append(equity)

    for sym in list(pos):  # offene Positionen am Ende zum letzten Kurs schliessen
        close(sym, sig[sym]["close"].iloc[-1], timeline[-1], "ende")

    return summarize(trades, curve, float(cfg["paper"]["start_equity"]))


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
    res = run_backtest(cfg, data, rules)
    print_report(res, days)


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
    print("==============================")
    print("Hinweis: Vergangene Ergebnisse garantieren keine zukuenftigen Gewinne.")
