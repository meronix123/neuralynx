import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot import engine
from bot.backtest import resample, run_backtest
from bot.config import load_config
from bot.risk import RiskGuard, position_size, round_amount, stop_is_safe
from bot.strategy import compute_signals, last_closed_signal, levels, trail_stop

CFG = load_config()
# Tests laufen auf 5m-Testdaten -> kurze Zeiteinheiten, unabhaengig von der aktuellen config.yaml
CFG["timeframe"], CFG["trend_timeframe"] = "5m", "1h"
CFG["strategy"].update(trend_ema_fast=50, trend_ema_slow=200, min_score=4, partial_tp_r=0,
                       breakeven_at_r=1.0,
                       # im Test vergeht keine echte Zeit -> zwischengespeicherte hoehere Zeitebenen
                       # waeren veraltet; der Zeitebenen-Filter hat einen eigenen Test
                       mtf_filter=False)
CFG["fees"]["entry_order"] = "market"


def synthetic(n=6000, seed=1, drift=0.00004, start=100.0):
    rng = np.random.default_rng(seed)
    # Trendphasen, damit Signale in beide Richtungen entstehen
    regime = np.repeat(rng.choice([-1, 1], size=n // 500 + 1), 500)[:n]
    ret = regime * drift + rng.normal(0, 0.0025, n)
    close = start * np.exp(np.cumsum(ret))
    open_ = np.r_[start, close[:-1]]
    spread = np.abs(rng.normal(0, 0.0015, n)) * close
    ts = 1_700_000_000_000 + np.arange(n) * 300_000
    return pd.DataFrame({
        "ts": ts, "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close, "volume": rng.uniform(50, 150, n),
    })


def test_position_size_respects_risk_and_margin():
    eq, entry, sl = 35.0, 100.0, 99.0
    amt = position_size(eq, entry, sl, 10, 2.0, 45, 0.0006)
    loss_at_sl = amt * (entry - sl) + 2 * 0.0006 * entry * amt
    assert loss_at_sl <= eq * 0.02 + 1e-9
    assert amt * entry <= eq * 0.45 * 10 + 1e-9
    # sehr enger Stop -> Margin-Grenze greift
    tight = position_size(eq, entry, 99.99, 10, 2.0, 45, 0.0006)
    assert tight * entry == pytest.approx(eq * 0.45 * 10)


def test_round_amount_and_min():
    assert round_amount(0.01237, 0.001, 0.001) == pytest.approx(0.012)
    assert round_amount(0.0004, 0.001, 0.001) == 0.0


def test_stop_must_be_far_from_liquidation():
    assert stop_is_safe(100, 98.0, 10)       # 2 % < 5 %
    assert not stop_is_safe(100, 94.0, 10)   # 6 % >= 5 %


def test_trailing_only_moves_in_profit_direction():
    s = CFG["strategy"]
    # long, noch nicht +1R -> nichts
    assert trail_stop("long", 100, 99, 99, 100.5, 0.5, s, 0.0006) is None
    # long, +2R -> Stop mindestens Einstand, unter dem Kurs
    new = trail_stop("long", 100, 99, 99, 102, 0.5, s, 0.0006)
    assert 100 < new < 102
    # nie zurueck
    assert trail_stop("long", 100, 99, 101.9, 102, 0.5, s, 0.0006) is None
    # short spiegelbildlich
    new = trail_stop("short", 100, 101, 101, 98, 0.5, s, 0.0006)
    assert 98 < new < 100


def test_levels():
    sl, tp = levels("long", 100, 1, CFG["strategy"])
    assert sl < 100 < tp
    sl, tp = levels("short", 100, 1, CFG["strategy"])
    assert tp < 100 < sl


def test_risk_guard_limits():
    r = CFG["risk"]
    g = RiskGuard(r, {})
    now = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    assert g.update_day(100, now)
    assert not g.update_day(100, now)
    assert g.can_open(100, 0, now)[0]
    assert not g.can_open(100 * (1 - r["daily_loss_limit_pct"] / 100) - 0.01, 0, now)[0]
    assert not g.can_open(100, r["max_open_positions"], now)[0]
    for _ in range(r["max_consecutive_losses"]):
        g.on_close(-1, now)
    assert not g.can_open(100, 0, now)[0]            # Pause
    later = now + timedelta(minutes=r["pause_after_losses_minutes"] + 1)
    assert g.can_open(100, 0, later)[0]
    for _ in range(r["max_trades_per_day"]):
        g.on_open()
    assert not g.can_open(100, 0, later)[0]           # Trades pro Tag
    assert g.update_day(100, now + timedelta(days=1))
    assert g.can_open(100, 0, now + timedelta(days=1))[0]


def test_trend_has_no_lookahead():
    """Das Ergebnis fuer einen Bar darf sich nicht aendern, wenn spaetere Daten dazukommen."""
    df = synthetic(3000)
    s, tf, ttf = CFG["strategy"], CFG["timeframe"], CFG["trend_timeframe"]
    full = compute_signals(df, resample(df, ttf), s, tf, ttf)
    part_df = df.iloc[:2000]
    part = compute_signals(part_df, resample(part_df, ttf), s, tf, ttf)
    cols = ["trend", "signal"]
    pd.testing.assert_frame_equal(
        full.iloc[:1990][cols].reset_index(drop=True),
        part.iloc[:1990][cols].reset_index(drop=True),
    )


def test_last_closed_signal_ignores_open_bar_and_funding():
    df = synthetic(6000)
    s, tf, ttf = CFG["strategy"], CFG["timeframe"], CFG["trend_timeframe"]
    d = compute_signals(df, resample(df, ttf), s, tf, ttf)
    idx = int(np.flatnonzero(d["signal"].to_numpy() != 0)[0])
    view = d.iloc[: idx + 2]  # Signal-Bar ist der vorletzte (abgeschlossene)
    sig = last_closed_signal(view, s)
    assert sig is not None and sig.ts == int(d.iloc[idx]["ts"])
    extreme = 0.01 if sig.side == "long" else -0.01
    assert last_closed_signal(view, s, funding=extreme) is None


def test_backtest_runs_and_charges_fees():
    cfg = copy.deepcopy(CFG)
    data = {"AAA/USDT:USDT": synthetic(6000, 1), "BBB/USDT:USDT": synthetic(6000, 2, start=50)}
    res = run_backtest(cfg, data, {k: (0.0001, 0.0001) for k in data})
    assert res["trades"] > 0
    t = res["table"]
    assert res["end"] == pytest.approx(res["start"] + t.pnl.sum())
    # nie mehr gleichzeitig offene Positionen als erlaubt
    events = sorted([(o, 1) for o in t.opened] + [(c, -1) for c in t.closed], key=lambda x: (x[0], x[1]))
    open_count = max(np.cumsum([e[1] for e in events]))
    assert open_count <= cfg["risk"]["max_open_positions"]
    # max. Trades pro Tag
    per_day = pd.to_datetime(t.opened, unit="ms").dt.date.value_counts()
    assert per_day.max() <= cfg["risk"]["max_trades_per_day"]


class FakeClient:
    """Ersatz fuer ccxt, damit Paper-Modus und Bot ohne Internet getestet werden."""

    def __init__(self, df):
        self.df = df
        self.i = 400

    def load_markets(self):
        return {"AAA/USDT:USDT": {"swap": True, "active": True}}

    def market(self, sym):
        return {"precision": {"amount": 0.001}, "limits": {"amount": {"min": 0.001}}}

    def fetch_ohlcv(self, sym, tf, limit=300, since=None):
        d = self.df.iloc[: self.i] if tf == "5m" else resample(self.df.iloc[: self.i], tf)
        return d.tail(limit).values.tolist()

    def fetch_ticker(self, sym):
        return {"last": float(self.df.iloc[self.i - 1]["close"])}

    def fetch_funding_rate(self, sym):
        return {"fundingRate": 0.0001}


class FakeContext:
    fng = {"value": 50, "label": "Neutral"}
    cal_ok = True

    def __init__(self, block=""):
        self.block = block

    def check(self, now):
        return (not self.block, self.block, 1.0)

    def next_events(self, now):
        return []


def test_paper_bot_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    client = FakeClient(synthetic(6000, 3))
    ex = PaperExchange(cfg, client)
    bot = engine.Bot(cfg, ex, FakeContext())
    opened = 0
    for i in range(400, 6000):
        client.i = i
        before = len(bot.state["meta"])
        bot.step()
        opened += len(bot.state["meta"]) > before
        # Paper-Konto und Positionen bleiben konsistent
        assert set(bot.state["meta"]) == set(ex.positions())
    assert opened > 0
    assert ex.equity() > 0
    # Oberflaechen-Status ist befuellt und JSON-faehig
    import json
    st = json.loads(json.dumps(bot.status, default=str))
    sym = st["symbols"]["AAA/USDT:USDT"]
    assert len(sym["candles"]) == engine.CHART_BARS and sym["reason"]
    assert len(st["history"]) > 0


def test_news_block_prevents_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    client = FakeClient(synthetic(6000, 3))
    bot = engine.Bot(cfg, PaperExchange(cfg, client), FakeContext("Wirtschaftstermin: USD CPI"))
    for i in range(400, 6000, 3):
        client.i = i
        bot.step()
    assert bot.state["meta"] == {} and bot.state["history"] == []
    assert bot.status["symbols"]["AAA/USDT:USDT"]["reason"].startswith("Wirtschaftstermin")


def test_calendar_parsing_and_window():
    from bot.context import MarketContext, parse_calendar

    rows = [
        {"title": "CPI m/m", "country": "USD", "date": "2026-01-14T08:30:00-05:00", "impact": "High"},
        {"title": "Retail", "country": "USD", "date": "2026-01-14T10:00:00-05:00", "impact": "Medium"},
        {"title": "ECB", "country": "EUR", "date": "2026-01-14T09:00:00-05:00", "impact": "High"},
    ]
    ev = parse_calendar(rows, ["USD"], ["High"])
    assert [e["title"] for e in ev] == ["CPI m/m"]
    assert ev[0]["time"] == datetime(2026, 1, 14, 13, 30, tzinfo=timezone.utc)

    class Http:
        def get(self, url, timeout):
            class R:
                def json(self_inner):
                    return rows if "faireconomy" in url else {"data": [{"value": "90", "value_classification": "Extreme Greed"}]}
            return R()

    ctx = MarketContext(CFG["context"], http=Http())
    ok, why, _ = ctx.check(datetime(2026, 1, 14, 13, 10, tzinfo=timezone.utc))
    assert not ok and "CPI" in why
    ok, _, factor = ctx.check(datetime(2026, 1, 14, 15, 0, tzinfo=timezone.utc))
    assert ok and factor == 0.5   # extreme Gier -> halbes Risiko


def test_dashboard_serves_status():
    import json
    import urllib.request

    from bot.dashboard import start_dashboard

    class B:
        status = {"symbols": {}, "equity": 35.0}

    srv = start_dashboard(B(), 0)
    port = srv.server_address[1]
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status").read()
    assert json.loads(body)["equity"] == 35.0
    html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
    assert "lightweight-charts" in html
    js = urllib.request.urlopen(f"http://127.0.0.1:{port}/lightweight-charts.js").read()
    assert b"Lightweight Charts" in js
    srv.shutdown()


def test_partial_take_profit_and_pyramiding_accounting():
    """Kontrollierter Ablauf: Einstieg 100, Teilverkauf bei +1R, Aufstocken, Stop auf Einstand."""
    from bot.backtest import simulate

    cfg = copy.deepcopy(CFG)
    cfg["fees"].update(slippage=0.0, entry_order="market")
    cfg["strategy"].update(sl_atr=1.0, tp_atr=5.0, breakeven_at_r=1e9, partial_tp_r=1.0, partial_tp_frac=0.5,
                           pyramid_at_r=1.0, pyramid_max_adds=1, pyramid_size_frac=0.5)
    #          Signal    Einstieg   +1R erreicht   Ruecksetzer
    bars = [(100, 100.3, 99.9, 100), (100, 100.5, 99.8, 100.2), (100.2, 101.5, 100.1, 101.4),
            (101.4, 101.5, 100.0, 100.5), (100.5, 100.6, 100.4, 100.5)]
    prep = {"AAA/USDT:USDT": {
        "ts": [1_700_000_000_000 + i * 900_000 for i in range(len(bars))],
        "open": [b[0] for b in bars], "high": [b[1] for b in bars], "low": [b[2] for b in bars],
        "close": [b[3] for b in bars], "atr": [1.0] * len(bars), "signal": [1, 0, 0, 0, 0],
        "sl_dist": [1.0] * len(bars), "tp_dist": [5.0] * len(bars), "strategy": ["trend"] * len(bars),
        "regime": ["trend_up"] * len(bars),
    }}
    res = simulate(cfg, prep, {"AAA/USDT:USDT": (0, 0)})
    t = res["table"].iloc[0]
    assert res["trades"] == 1 and t.partial and t.adds == 1 and t.why == "stop"
    assert t.pnl > 0                        # Teilgewinn gesichert, Rest auf Einstand ausgestoppt
    assert res["end"] == pytest.approx(res["start"] + t.pnl)


def test_never_adds_to_losing_position():
    from bot.backtest import simulate

    cfg = copy.deepcopy(CFG)
    cfg["fees"].update(slippage=0.0)
    cfg["strategy"].update(sl_atr=1.0, tp_atr=3.0, pyramid_at_r=1.0, pyramid_max_adds=2, pyramid_size_frac=0.5)
    bars = [(100, 100.2, 99.9, 100), (100, 100.1, 99.5, 99.6), (99.6, 99.7, 98.5, 98.8)]
    prep = {"AAA/USDT:USDT": {
        "ts": [1_700_000_000_000 + i * 900_000 for i in range(len(bars))],
        "open": [b[0] for b in bars], "high": [b[1] for b in bars], "low": [b[2] for b in bars],
        "close": [b[3] for b in bars], "atr": [1.0] * len(bars), "signal": [1, 0, 0],
        "sl_dist": [1.0] * len(bars), "tp_dist": [3.0] * len(bars), "strategy": ["trend"] * len(bars),
        "regime": ["trend_up"] * len(bars),
    }}
    res = simulate(cfg, prep, {"AAA/USDT:USDT": (0, 0)})
    t = res["table"].iloc[0]
    assert t.adds == 0 and t.why == "stop" and t.pnl < 0
    # Verlust nicht groesser als das geplante Risiko (1 % vom Konto)
    assert -t.pnl <= res["start"] * cfg["risk"]["risk_per_trade_pct"] / 100 + 1e-9


def test_optimizer_small_grid(monkeypatch):
    from bot import optimize

    monkeypatch.setattr(optimize, "TIMEFRAMES", [("15m", "1h", 50, 200)])
    monkeypatch.setattr(optimize, "MIN_SCORES", [4])
    monkeypatch.setattr(optimize, "SL_RR", [(1.5, 2.0)])
    monkeypatch.setattr(optimize, "STRATEGY_SETS", {"nur_trend": ["trend"], "alle_nach_lage": ["trend", "range", "breakout"]})
    data = {}
    for k in range(2):
        df = synthetic(8000, k + 1)
        df["ts"] = 1_700_000_000_000 + df.index * 900_000
        data[f"S{k}/USDT:USDT"] = df
    res = optimize.optimize(CFG, data, {k: (0.0001, 0.0001) for k in data}, base_tf="15m")
    assert len(res) == 2 * len(optimize.ENTRY) * len(optimize.MANAGE) * len(optimize.FILTERS)
    assert set(res["filter"]) == set(optimize.FILTERS)
    assert {"train_pf", "test_pf", "fuehrung", "einstieg", "strategien"} <= set(res.columns)


def test_regimes_and_strategies_produce_signals():
    """Alle drei Strategien feuern auf passenden Daten, und nur in ihrer Marktlage."""
    df = synthetic(12000, 5)
    s = copy.deepcopy(CFG["strategy"])
    s["strategies"] = ["trend", "range", "breakout"]
    d = compute_signals(df, resample(df, "1h"), s, "5m", "1h")
    assert set(d["regime"].unique()) >= {"trend_up", "range", "squeeze"}
    fired = d[d.signal != 0]
    assert set(fired.strategy) == {"trend", "range", "breakout"}
    assert (fired[fired.strategy == "range"].regime == "range").all()
    assert (fired[fired.strategy == "trend"].regime.isin(["trend_up", "trend_down", "unclear"])).all()
    assert (d[d.regime == "chaos"].signal == 0).all()
    # SL/TP liegen immer auf der richtigen Seite
    assert (fired.sl_dist > 0).all() and (fired.tp_dist > 0).all()
    # Strategie abschaltbar
    s["strategies"] = ["trend"]
    only = compute_signals(df, resample(df, "1h"), s, "5m", "1h")
    assert set(only[only.signal != 0].strategy) == {"trend"}


def test_flow_scoring():
    from bot.flow import book_imbalance, flow_verdict, taker_flow

    book = {"bids": [[99.9, 10], [99.8, 10]], "asks": [[100.1, 2], [100.2, 2]]}
    imb = book_imbalance(book)
    assert imb == pytest.approx((20 - 4) / 24)
    tf = taker_flow([{"side": "buy", "amount": 3}, {"side": "sell", "amount": 1}])
    assert tf == pytest.approx(0.5)
    m = {"imbalance": imb, "taker_flow": tf}
    assert flow_verdict("long", m, {})[0]                   # passt zu Long
    ok, why, _ = flow_verdict("short", m, {"flow_block": 0.35})
    assert not ok and "dagegen" in why                      # Short gegen starken Kaufdruck -> nein
    assert flow_verdict("long", {"imbalance": None, "taker_flow": None}, {})[2] == 1.0  # keine Daten -> neutral


def test_paper_bot_limit_entry_and_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["fees"]["entry_order"] = "limit"
    cfg["strategy"].update(partial_tp_r=1.0, partial_tp_frac=0.5, breakeven_at_r=1e9)
    client = FakeClient(synthetic(6000, 3))
    ex = PaperExchange(cfg, client)
    bot = engine.Bot(cfg, ex, FakeContext())
    placed = partial = 0
    for i in range(400, 6000):
        client.i = i
        before = len(bot.state["pending"])
        bot.step()
        placed += len(bot.state["pending"]) > before
        partial += sum(1 for m in bot.state["meta"].values() if m.get("partial_done"))
        # jede Position ist verwaltet oder stammt aus einer gerade ausgefuehrten Limit-Order
        assert set(ex.positions()) <= set(bot.state["meta"]) | set(bot.state["pending"])
        assert not (set(bot.state["pending"]) & set(bot.state["meta"]))
    assert placed > 0
    assert len(bot.state["history"]) > 0
    assert ex.equity() > 0


def test_partial_take_profit_in_live_engine(tmp_path, monkeypatch):
    """Kurs erreicht +1R -> Haelfte wird verkauft, Stop auf Einstand, Rest laeuft weiter."""
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    class PriceClient(FakeClient):
        price = 100.0

        def fetch_ticker(self, sym):
            return {"last": self.price}

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["strategy"].update(partial_tp_r=1.0, partial_tp_frac=0.5, breakeven_at_r=1e9)
    client = PriceClient(synthetic(1000, 1))
    ex = PaperExchange(cfg, client)
    bot = engine.Bot(cfg, ex, FakeContext())
    sym = "AAA/USDT:USDT"
    entry = ex.open(sym, "long", 1.0, 99.0, 104.0)
    bot._register_open(sym, "long", entry, 1.0, 99.0, 104.0, 5)
    cash_before = ex.cash

    client.price = entry + 0.5          # noch nicht +1R
    bot._manage_open(ex.positions(), {sym: {"sig_df": pd.DataFrame({"atr": [1.0]})}})
    assert not bot.state["meta"][sym]["partial_done"]

    client.price = entry + 1.1          # +1,1R
    bot._manage_open(ex.positions(), {sym: {"sig_df": pd.DataFrame({"atr": [1.0]})}})
    m = bot.state["meta"][sym]
    assert m["partial_done"] and m["amount"] == pytest.approx(0.5)
    assert ex.pos[sym]["amount"] == pytest.approx(0.5)
    assert m["sl"] > entry and ex.pos[sym]["sl"] == pytest.approx(m["sl"])   # Stop auf Einstand
    assert ex.cash > cash_before                                              # Teilgewinn gebucht

    client.price = entry                 # zurueck -> Rest wird auf Einstand ausgestoppt
    bot._handle_closed(ex.positions(), datetime.now(timezone.utc))
    assert bot.state["history"][-1]["pnl"] > 0


def test_real_config_is_consistent():
    cfg = load_config()
    from bot.strategy import TF_MS
    assert cfg["timeframe"] in TF_MS and cfg["trend_timeframe"] in TF_MS
    assert cfg["fees"]["entry_order"] in ("market", "limit")
    assert stop_is_safe(100, 100 - cfg["strategy"]["sl_atr"] * 1.5, cfg["leverage"])  # ATR 1,5 %


def test_paper_account_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    client = FakeClient(synthetic(1000, 1))
    ex = PaperExchange(cfg, client)
    price = ex.last_price("AAA/USDT:USDT")
    ex.open("AAA/USDT:USDT", "long", 1.0, price * 0.9, price * 1.2)
    ex.cash -= 5
    bot = engine.Bot(cfg, ex, FakeContext())
    bot.state["paper"] = ex.dump()
    engine.save_state(bot.state)

    ex2 = PaperExchange(cfg, client)
    engine.Bot(cfg, ex2, FakeContext())       # Neustart
    assert ex2.cash == pytest.approx(ex.cash)
    assert "AAA/USDT:USDT" in ex2.positions()


def test_ml_model_learns_and_roundtrips():
    from bot.ml import SignalModel, features, train_from_examples

    rng = np.random.default_rng(0)
    examples = []
    for _ in range(300):
        adx_v = rng.uniform(5, 50)
        row = {"rsi": 50, "adx": adx_v, "bb_width": 0.05, "atr_pct": 0.5, "vol_ratio": 1.0,
               "macd_n": 0.1, "trend": 1, "regime": "trend_up", "strategy": "trend"}
        examples.append({"x": features(row, "long"), "win": int(adx_v > 25)})   # starker Trend gewinnt
    m = train_from_examples(examples)
    strong = features({"adx": 45, "regime": "trend_up", "strategy": "trend", "trend": 1}, "long")
    weak = features({"adx": 8, "regime": "trend_up", "strategy": "trend", "trend": 1}, "long")
    assert m.proba(strong) > 0.7 > 0.3 > m.proba(weak)
    m2 = SignalModel.from_json(m.to_json())
    assert m2.proba(strong) == pytest.approx(m.proba(strong))


def test_leader_health_and_time_stop_rules():
    from bot.filters import leader_blocks, strategy_healthy, time_stop_due

    assert leader_blocks("long", "trend_down") and leader_blocks("short", "trend_up")
    assert not leader_blocks("long", "trend_up") and not leader_blocks("long", None)
    s = {"health_window": 5, "health_min_pf": 0.6}
    assert strategy_healthy([-1, -1, -1], s)                 # zu wenig Daten -> weiter
    assert not strategy_healthy([1, -1, -1, -1, -1], s)      # PF 0,25 -> Pause
    assert strategy_healthy([2, -1, 2, -1, -1], s)           # PF 1,33 -> ok
    t = {"max_hold_bars": 12}
    assert time_stop_due(12, 0.2, False, t) and not time_stop_due(11, 0.2, False, t)
    assert not time_stop_due(20, 0.8, False, t) and not time_stop_due(20, 0.0, True, t)


def test_filters_and_ml_run_in_simulation():
    from bot.backtest import prepare, simulate

    cfg = copy.deepcopy(CFG)
    data = {"BTC/USDT:USDT": synthetic(12000, 5), "ETH/USDT:USDT": synthetic(12000, 6, start=50)}
    rules = {k: (0.0001, 0.0001) for k in data}
    cfg["strategy"].update(leader_filter=False, health_window=0, max_hold_bars=0, ml_filter=False,
                           mtf_filter=False)
    prep = prepare(cfg, data, "5m")
    assert "leader_regime" in prep["ETH/USDT:USDT"]
    base = simulate(cfg, prep, rules)
    cfg["strategy"].update(leader_filter=True, health_window=10, max_hold_bars=3,
                           ml_filter=True, ml_min_trades=30)
    filt = simulate(cfg, prep, rules)
    sk = filt["skipped"]
    assert sk["leader"] > 0 and sk["leader"] + sk["health"] + sk["ml"] > 0
    assert base["skipped"] == {"leader": 0, "mtf": 0, "health": 0, "ml": 0}
    assert filt["end"] == pytest.approx(filt["start"] + filt["table"].pnl.sum())
    assert (filt["table"].why == "zeit").any()


def test_paused_strategy_gets_probe_trade():
    from bot.filters import health_gate

    s = {"health_window": 5, "health_min_pf": 0.6, "health_pause_signals": 3}
    bad = [-1, -1, -1, -1, -1]
    skips = {}
    decisions = [health_gate("range", bad, s, skips) for _ in range(8)]
    assert decisions == [False, False, False, True, False, False, False, True]


def test_fetch_history_skips_time_before_listing(tmp_path, monkeypatch):
    """Markt erst seit kurzem gelistet: leere Antworten ueberspringen statt abbrechen."""
    from bot import backtest

    monkeypatch.setattr(backtest, "DATA", tmp_path)
    now = 1_760_000_000_000
    listed = now - 20 * 86_400_000

    class ListedLate:
        def milliseconds(self):
            return now

        def fetch_ohlcv(self, sym, tf, since=None, limit=200):
            start = max(since, listed)
            rows = [[t, 1.0, 1.1, 0.9, 1.0, 5.0] for t in range(start, min(now, start + limit * 3_600_000), 3_600_000)]
            return [] if since < listed - limit * 3_600_000 else rows

    df = backtest.fetch_history(ListedLate(), "XAU/USDT:USDT", "1h", 365)
    assert len(df) >= 19 * 24 and df["ts"].dtype == "int64"
    assert backtest.usable({"XAU/USDT:USDT": df.head(100), "BTC/USDT:USDT": synthetic(600)}).keys() == {"BTC/USDT:USDT"}


def test_multi_timeframe_scan_and_filter():
    from bot.backtest import prepare, simulate
    from bot.filters import mtf_blocks
    from bot.strategy import higher_tfs, resample as rs

    assert higher_tfs("4h") == ["1d", "1w"] and higher_tfs("1h") == ["2h", "4h", "1d", "1w"]
    s = {"mtf_filter": True, "mtf_min": 0.25}
    assert mtf_blocks("long", 0.0, s) and not mtf_blocks("long", 0.5, s)
    assert mtf_blocks("short", 0.0, s) and not mtf_blocks("short", -0.5, s)
    assert not mtf_blocks("long", -1.0, {"mtf_filter": False})
    # Wochenkerzen beginnen montags (wie bei Bitget)
    df = synthetic(6000)
    w = rs(df, "1w")
    assert all(pd.to_datetime(t, unit="ms").weekday() == 0 for t in w.ts)

    cfg = copy.deepcopy(CFG)
    cfg["strategy"].update(leader_filter=False, health_window=0, max_hold_bars=0, ml_filter=False)
    data = {"AAA/USDT:USDT": synthetic(12000, 5)}
    prep = prepare(cfg, data, "5m")
    scores = [v for v in prep["AAA/USDT:USDT"]["mtf_score"] if v == v]
    assert min(scores) >= -1 and max(scores) <= 1 and len(set(scores)) > 2
    cfg["strategy"]["mtf_filter"] = False
    free = simulate(cfg, prep, {"AAA/USDT:USDT": (0.0001, 0.0001)})
    cfg["strategy"]["mtf_filter"] = True
    gated = simulate(cfg, prep, {"AAA/USDT:USDT": (0.0001, 0.0001)})
    assert gated["skipped"]["mtf"] > 0 and free["skipped"]["mtf"] == 0
