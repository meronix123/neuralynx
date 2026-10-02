import copy
import time
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
                       mtf_filter=False, macro_filter=False, ob_filter="off")
CFG["dashboard"]["enabled"] = False  # keine echten Makro-Abrufe im Test
CFG["fees"]["entry_order"] = "market"
CFG["tf_select"] = "fixed"  # Auto-Zeiteinheit hat eigene Tests


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
    assert base["skipped"] == {"muster": 0, "leader": 0, "mtf": 0, "funding": 0, "macro": 0, "orderblock": 0,
                               "zeiteinheit": 0, "health": 0, "ml": 0}
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


def test_limit_take_profit_is_cheaper_and_needs_trade_through():
    from bot.backtest import simulate

    def run(tp_order, high_on_target):
        cfg = copy.deepcopy(CFG)
        cfg["fees"].update(slippage=0.0003, entry_order="market", tp_order=tp_order)
        cfg["strategy"].update(partial_tp_r=0, breakeven_at_r=1e9, max_hold_bars=0, health_window=0,
                               leader_filter=False, ml_filter=False)
        bars = [(100, 100.2, 99.9, 100), (100, 100.5, 99.8, 100.2), (100.2, high_on_target, 100.1, 101.5),
                (101.5, 101.52, 101.4, 101.5)]
        prep = {"AAA/USDT:USDT": {
            "ts": [1_700_000_000_000 + i * 300_000 for i in range(len(bars))],
            "open": [b[0] for b in bars], "high": [b[1] for b in bars], "low": [b[2] for b in bars],
            "close": [b[3] for b in bars], "atr": [1.0] * len(bars), "signal": [1, 0, 0, 0],
            "sl_dist": [1.0] * 4, "tp_dist": [1.5] * 4, "strategy": ["trend"] * 4, "regime": ["trend_up"] * 4,
        }}
        return simulate(cfg, prep, {"AAA/USDT:USDT": (0, 0)})

    market = run("market", 102.0)
    limit = run("limit", 102.0)
    assert market["table"].iloc[0].why == limit["table"].iloc[0].why == "ziel"
    assert limit["table"].iloc[0].pnl > market["table"].iloc[0].pnl        # Maker-Gebuehr, kein Slippage
    # Kurs beruehrt das Ziel nur genau -> liegende Limit-Order wird (vorsichtig) NICHT gefuellt
    exact = run("limit", 100.0 * (1 + 0.0003) + 1.5)
    assert exact["table"].iloc[0].why != "ziel"


def test_paper_engine_places_limit_take_profit(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    class PriceClient(FakeClient):
        price = 100.0

        def fetch_ticker(self, sym):
            return {"last": self.price}

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["fees"]["tp_order"] = "limit"
    client = PriceClient(synthetic(1000, 1))
    ex = PaperExchange(cfg, client)
    bot = engine.Bot(cfg, ex, FakeContext())
    sym = "AAA/USDT:USDT"
    entry = ex.open(sym, "long", 1.0, 99.0, None)
    bot._register_open(sym, "long", entry, 1.0, 99.0, 101.0, 5)
    assert bot.state["meta"][sym]["tp_order_id"] and ex.pos[sym]["tp_maker"]
    cash = ex.cash
    client.price = 101.2
    assert sym not in ex.positions()                      # Ziel erreicht, Limit-Order gefuellt
    gain = ex.cash - cash
    assert gain == pytest.approx((101.0 - entry) * 1.0 - 101.0 * cfg["fees"]["maker"])


def test_fast_profile_grid_runs_end_to_end():
    from bot import optimize

    saved = {k: getattr(optimize, k) for k in optimize.FAST}
    try:
        optimize.use_fast_profile()
        assert [t[0] for t in optimize.TIMEFRAMES] == ["5m", "15m", "30m"]
        assert optimize.TP_ORDERS == ["market", "limit"] and optimize.ENTRY == ["limit"]
        # wirklich durchrechnen (klein), wie auf dem PC mit --fast
        optimize.TIMEFRAMES = [("5m", "1h", 50, 200)]
        optimize.MIN_SCORES, optimize.SL_RR = [4], [(1.5, 2.0)]
        optimize.STRATEGY_SETS = {"nur_trend": ["trend"]}
        data = {"AAA/USDT:USDT": synthetic(6000, 1)}
        res = optimize.optimize(CFG, data, {"AAA/USDT:USDT": (0.0001, 0.0001)}, base_tf="5m")
        assert len(res) == 1 * 2 * len(optimize.MANAGE) * len(optimize.FILTERS)
        assert set(res["tp"]) == {"market", "limit"}
        assert set(res["filter"]) == {"filter", "filter+zeitebenen", "filter+zeitebenen+funding"}
    finally:
        for k, v in saved.items():
            setattr(optimize, k, v)
        optimize.MIN_SCORES, optimize.SL_RR = [4, 5], [(1.0, 2.0), (1.5, 2.0), (2.0, 3.0)]


def test_funding_crowding_filter():
    from bot.backtest import prepare, simulate
    from bot.derivs import fetch_funding_history, funding_blocks, live_rank, merge_funding

    s = {"funding_filter": True, "funding_block_rank": 0.9}
    assert funding_blocks("long", 0.95, s) and not funding_blocks("short", 0.95, s)
    assert funding_blocks("short", 0.05, s) and not funding_blocks("long", 0.05, s)
    assert not funding_blocks("long", float("nan"), s) and not funding_blocks("long", 0.99, {})

    # Seitenweises Laden (neueste zuerst) bis zum Startdatum
    now = 1_760_000_000_000
    all_rates = [(now - k * 8 * 3_600_000, 0.0001 * (k % 7)) for k in range(400)]

    class C:
        def milliseconds(self):
            return now

        def fetch_funding_rate_history(self, sym, since, limit, params):
            p = params["pageNo"]
            chunk = all_rates[(p - 1) * limit: p * limit]
            return [{"timestamp": t, "fundingRate": r} for t, r in chunk]

    f = fetch_funding_history(C(), "AAA/USDT:USDT", 60)
    assert f.ts.is_monotonic_increasing and f.ts.min() >= now - 60 * 86_400_000 and len(f) == 60 * 3 + 1
    assert live_rank(f, 1.0) == 1.0 and live_rank(f, -1.0) == 0.0

    # kein Blick in die Zukunft: ein Bar sieht nur Funding, das vor seinem Schluss feststand
    bars = pd.DataFrame({"avail": [now - 10 * 3_600_000, now]})
    fut = pd.DataFrame({"ts": [now - 5 * 3_600_000], "rate": [0.01]})
    assert merge_funding(bars, fut).isna().iloc[0]

    df = synthetic(6000, 2)
    fund = pd.DataFrame({"ts": df.ts.iloc[::96].to_numpy(), "rate": np.linspace(-0.001, 0.003, len(df.ts.iloc[::96]))})
    cfg = copy.deepcopy(CFG)
    cfg["strategy"].update(leader_filter=False, health_window=0, max_hold_bars=0, ml_filter=False, mtf_filter=False)
    prep = prepare(cfg, {"AAA/USDT:USDT": df}, "5m", {"AAA/USDT:USDT": fund})
    ranks = [r for r in prep["AAA/USDT:USDT"]["funding_rank"] if r == r]
    assert ranks and max(ranks) <= 1
    cfg["strategy"]["funding_filter"] = True
    res = simulate(cfg, prep, {"AAA/USDT:USDT": (0.0001, 0.0001)})
    assert res["skipped"]["funding"] > 0      # steigendes Funding = ueberfuellte Longs werden aussortiert


def test_macro_traffic_light():
    from bot.backtest import prepare, simulate
    from bot.macro import compute_scores, fetch_macro, latest, macro_blocks

    days = pd.date_range("2025-01-01", periods=300, freq="D")
    up = pd.Series(np.linspace(100, 200, 300), index=days)      # Aktien steigen
    down = pd.Series(np.linspace(120, 100, 300), index=days)    # Dollar faellt
    sc = compute_scores({"spx": up, "ndx": up, "dxy": down, "us10y": down})
    assert sc["crypto"].iloc[-1] == pytest.approx(1.0) and sc["gold"].iloc[-1] == pytest.approx(1.0)
    risk_off = compute_scores({"spx": down, "dxy": up, "us10y": up})
    assert risk_off["crypto"].iloc[-1] == pytest.approx(-1.0)

    s = {"macro_filter": True, "macro_min": 0.3}
    assert macro_blocks("long", -0.5, s) and not macro_blocks("short", -0.5, s)
    assert macro_blocks("short", 0.5, s) and not macro_blocks("long", 0.5, s)
    assert not macro_blocks("long", -1, {}) and not macro_blocks("long", float("nan"), s)

    # Versatz: ein Tageswert ist erst 2 Tage spaeter nutzbar
    first_avail = int(sc["avail"].iloc[60])
    assert latest(sc, False, first_avail - 1) == sc["crypto"].iloc[59]

    class R:
        def __init__(self, text=None, js=None):
            self.text, self._js = text, js

        def json(self):
            return self._js

    class Http:
        def get(self, url, params=None, timeout=None):
            if "fred" in url:
                return R(text="observation_date,X\n" + "\n".join(
                    f"{d.date()},{v}" for d, v in zip(days, np.linspace(1, 2, 300))))
            if "llama" in url:
                return R(js=[{"date": str(int(d.timestamp())), "totalCirculatingUSD": {"peggedUSD": 1e9 + i}}
                             for i, d in enumerate(days)])
            return R(js={"result": {"data": [[int(d.timestamp() * 1000), 1, 1, 1, 50.0] for d in days],
                                    "continuation": None}})

    m = fetch_macro(300, http=Http())
    assert m is not None and m["crypto"].notna().sum() > 100

    # im Backtest: Risiko-aus-Phase sortiert Longs aus
    df = synthetic(6000, 2)
    start = pd.to_datetime(df.ts.iloc[0], unit="ms").normalize() - pd.Timedelta(days=100)
    d2 = pd.date_range(start, periods=150, freq="D")
    off = compute_scores({"spx": pd.Series(np.linspace(200, 100, 150), index=d2),
                          "dxy": pd.Series(np.linspace(100, 200, 150), index=d2)})
    cfg = copy.deepcopy(CFG)
    cfg["strategy"].update(leader_filter=False, health_window=0, max_hold_bars=0, ml_filter=False, mtf_filter=False)
    prep = prepare(cfg, {"AAA/USDT:USDT": df}, "5m", None, off)
    assert np.nanmax(prep["AAA/USDT:USDT"]["macro_score"]) < 0
    cfg["strategy"]["macro_filter"] = True
    res = simulate(cfg, prep, {"AAA/USDT:USDT": (0.0001, 0.0001)})
    assert res["skipped"]["macro"] > 0
    if res["trades"]:
        assert (res["table"].side == "short").all()


def test_shadow_trades_and_timeframe_gate():
    from bot.tfselect import ShadowBook, profit_factor, shadow_results, tf_allowed

    s = {"max_hold_bars": 3, "tf_window": 4, "tf_min_pf": 1.0, "tf_warmup": 3}
    # Long bei 100, Stop 99, Ziel 102 -> Bar 2 erreicht das Ziel; Short bei 100 -> Bar 5 trifft den Stop
    d = {"ts": [0, 300_000, 600_000, 900_000, 1_200_000, 1_500_000],
         "high": [100, 100.5, 102.5, 100, 100.2, 101.5], "low": [100, 99.5, 101, 99.8, 99.5, 100],
         "close": [100, 100, 102, 100, 100, 101], "signal": [1, 0, 0, -1, 0, 0],
         "sl_dist": [1, 1, 1, 1, 1, 1], "tp_dist": [2, 2, 2, 2, 2, 2]}
    res = shadow_results(d, "5m", s, fee=0.0)
    assert res == [(900_000, 2.0), (1_800_000, -1.0)]  # bekannt erst zum Bar-Schluss
    fee_res = shadow_results(d, "5m", s, fee=0.0006)
    assert fee_res[0][1] == pytest.approx(2.0 - 2 * 0.0006 * 100)
    # Zeitebenen-Filter gilt auch fuer Schatten-Trades
    blocked = shadow_results({**d, "mtf_score": [-1.0] * 6}, "5m", {**s, "mtf_filter": True}, 0.0)
    assert blocked == [(1_800_000, -1.0)]
    # Anlaufphase erlaubt alles, danach entscheidet der Profit-Faktor der letzten Trades
    assert tf_allowed([-1, -1], s)
    assert not tf_allowed([2, -1, -1, -1, -1], s)
    assert tf_allowed([-1, -1, 2, 2, -1], s)
    assert profit_factor([2, -1]) == 2.0
    book = ShadowBook({"5m": res})
    assert book.known("5m", 899_999) == [] and book.known("5m", 900_000) == [2.0]


def test_multi_timeframe_backtest_one_position_per_market():
    from bot.backtest import prepare_multi, simulate

    data = {"AAA/USDT:USDT": synthetic(6000, 1), "BBB/USDT:USDT": synthetic(6000, 2, start=50)}
    rules = {k: (0.0001, 0.0001) for k in data}
    out = {}
    for mode in ("all", "adaptive"):
        cfg = copy.deepcopy(CFG)
        cfg["tf_select"], cfg["timeframes"] = mode, ["5m", "15m", "30m"]
        prep = prepare_multi(cfg, data, "5m")
        assert set(prep) == {f"{s}|{t}" for s in data for t in ("5m", "15m", "30m")}
        res = simulate(cfg, prep, rules)
        t = res["table"]
        assert res["trades"] > 0 and set(t.tf) <= {"5m", "15m", "30m"}
        assert res["end"] == pytest.approx(res["start"] + t.pnl.sum())
        for sym, g in t.groupby("symbol"):  # nie zwei Positionen im selben Markt
            g = g.sort_values("opened")
            assert (g.opened.to_numpy()[1:] >= g.closed.to_numpy()[:-1]).all()
        out[mode] = res
    assert out["all"]["skipped"]["zeiteinheit"] == 0
    assert out["adaptive"]["skipped"]["zeiteinheit"] > 0
    assert out["all"]["per_tf"] is not None


def test_paper_bot_chooses_timeframe(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["tf_select"], cfg["timeframes"] = "adaptive", ["5m", "15m"]
    # unabhaengig von config.yaml: genug Signale, Anlaufphase laesst alle Zeiteinheiten zu
    cfg["strategy"].update(strategies=["trend", "range", "breakout"], tf_warmup=10_000)
    client = FakeClient(synthetic(1800, 3))
    ex = PaperExchange(cfg, client)
    bot = engine.Bot(cfg, ex, FakeContext())
    assert bot.tfs == ["15m", "5m"]
    tfs_used = set()
    for i in range(400, 1800):
        client.i = i
        bot.step()
        assert set(bot.state["meta"]) == set(ex.positions())
        tfs_used |= {m.get("tf") for m in bot.state["meta"].values()}
    tfs_used |= {h.get("tf") for h in bot.state["history"]}
    assert tfs_used - {None} <= {"5m", "15m"} and tfs_used - {None}
    assert set(bot.status["tf_stats"]) == {"5m", "15m"}
    assert all(t.get("tf") in ("5m", "15m") for t in bot.state["history"])
    assert any(k.endswith("|5m") or k.endswith("|15m") for k in bot.state["last_sig"])


def test_timeframe_comparison_runs():
    from bot import optimize

    saved = (optimize.ZEIT_TFS, optimize.ZEIT_MODES, optimize.ZEIT_SETS, optimize.ZEIT_FILTERS)
    try:
        optimize.ZEIT_TFS = ["5m", "15m"]
        optimize.ZEIT_MODES = {"fest_15m": (["15m"], "fixed", 20, 1.0),
                               "auto": (["5m", "15m"], "adaptive", 20, 1.0)}
        optimize.ZEIT_SETS, optimize.ZEIT_FILTERS = ["nur_trend"], ["filter+zeitebenen"]
        data = {"AAA/USDT:USDT": synthetic(6000, 1)}
        res = optimize.compare_timeframes(CFG, data, {"AAA/USDT:USDT": (0.0001, 0.0001)}, base_tf="5m")
        assert list(res.modus) == ["fest_15m", "auto"]
        assert all(x.strip().startswith("15m") for x in res.iloc[0]["test_je_zeiteinheit"].split(";") if x)
    finally:
        optimize.ZEIT_TFS, optimize.ZEIT_MODES, optimize.ZEIT_SETS, optimize.ZEIT_FILTERS = saved


def test_real_config_timeframes_are_supported():
    from bot.tfselect import TREND_FOR, active_tfs

    cfg = load_config()
    for tf in active_tfs(cfg):
        assert tf in TREND_FOR and tf in cfg["strategy"]["atr_limits"]


class FakeBitget:
    """Simuliertes Bitget-Konto fuer die Konto-Ansicht (ohne Internet)."""

    def __init__(self, api=None, demo=False):
        if api and api.get("key") == "falsch":
            raise_on = True
        else:
            raise_on = False
        self.bad = raise_on
        self.calls = []
        self.pos = [{"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.004, "entryPrice": 60000,
                     "markPrice": 61000, "liquidationPrice": 54500, "unrealizedPnl": 4.0, "percentage": 16.4,
                     "leverage": 10, "marginMode": "isolated", "initialMargin": 24.0, "notional": 244.0,
                     "stopLossPrice": 59000, "takeProfitPrice": 63000, "timestamp": 1_790_000_000_000}]

    def fetch_balance(self, params=None):
        if self.bad:
            raise RuntimeError("apikey does not exist")
        return {"USDT": {"total": 35.5, "free": 11.5, "used": 24.0}}

    def fetch_positions(self, symbols=None):
        return [p for p in self.pos if not symbols or p["symbol"] in symbols]

    # --- Ausloese-Auftraege (Stop/Ziel der Einzel-Positionen: reduceOnly + triggerPrice) ---
    def _plan_create(self, sym, side, amount, params, price=None):
        if params and params.get("reduceOnly") and price is not None and "triggerPrice" not in params:
            params = {**params, "triggerPrice": price, "limit": True}      # Ziel = Limit-Order (nur reduzierend)
        if not params or "triggerPrice" not in params:
            return None
        self.plans = getattr(self, "plans", {})
        self.n_plans = getattr(self, "n_plans", 0) + 1
        oid = f"p{self.n_plans}"
        kind = "loss_plan" if params.get("triggerType") == "mark_price" else "profit_plan"
        self.plans[oid] = {"symbol": sym.split("/")[0] + "USDT", "sym": sym, "planType": kind, "side": side,
                           "kind": "normal" if params.get("limit") else "plan",
                           "triggerPrice": params["triggerPrice"], "size": amount, "reduceOnly": params.get("reduceOnly"),
                           "marginMode": params.get("marginMode")}
        self.calls.append(("plan", kind, params["triggerPrice"], amount))
        return {"id": oid}

    def privateMixPostV2MixOrderPlaceTpslOrder(self, req):  # noqa: N802 - Bitget-Teil-TP/SL
        assert "executePrice" not in req                       # executePrice 0 -> Bitget-Fehler 43011
        self.plans = getattr(self, "plans", {})
        self.n_plans = getattr(self, "n_plans", 0) + 1
        oid = f"p{self.n_plans}"
        sym = req["symbol"][:-4] + "/USDT:USDT"
        self.plans[oid] = {**req, "sym": sym, "kind": "tpsl", "side": req["holdSide"]}
        self.calls.append(("plan", req["planType"], req["triggerPrice"], req["size"]))
        return {"data": {"orderId": oid}}

    def privateMixPostV2MixOrderCancelPlanOrder(self, req):  # noqa: N802
        oid = req["orderIdList"][0]["orderId"]
        if oid not in getattr(self, "plans", {}):
            raise RuntimeError('bitget {"code":"43025","msg":"Plan order does not exist"}')
        self.plans.pop(oid)
        self.calls.append(("cancel_plan", oid))

    def _plan_cancel(self, oid, params):
        if not (params or {}).get("trigger") or oid not in getattr(self, "plans", {}):
            return False
        self.plans.pop(oid)
        self.calls.append(("cancel_plan", oid))
        return True

    def _plan_list(self, sym, params):
        p = params or {}
        if p.get("planType") not in (None, "profit_loss"):
            return None
        want = "tpsl" if p.get("planType") else "plan" if p.get("trigger") else "normal"
        return [{"id": k, "symbol": v["sym"], "type": "market", "side": v["side"], "triggerPrice": float(v["triggerPrice"]),
                 "amount": float(v["size"]), "reduceOnly": True}
                for k, v in getattr(self, "plans", {}).items()
                if v["kind"] == want and (sym is None or v["sym"] == sym)]

    def fetch_open_orders(self, sym, since=None, limit=None, params=None):
        lst = self._plan_list(sym, params) or []
        if (params or {}).get("planType") == "profit_loss" and sym in (None, "BTC/USDT:USDT") \
                and "77" not in getattr(self, "gone", ()):
            lst = lst + [{"id": "77", "symbol": "BTC/USDT:USDT", "type": "market", "side": "sell", "triggerPrice": 59000,
                          "amount": 0.004, "filled": 0, "reduceOnly": True, "timestamp": 1_790_000_000_000}]
        return lst

    def fetch_positions_history(self, symbols=None, since=None, limit=None):
        return [{"symbol": "ETH/USDT:USDT", "side": "short", "entryPrice": 3000, "contracts": 0.1,
                 "realizedPnl": -1.2, "timestamp": 1_789_000_000_000, "lastUpdateTimestamp": 1_789_100_000_000,
                 "info": {"closeAvgPrice": "3012", "totalFee": "-0.36", "totalFunding": "0.01"}}]

    def load_markets(self):
        return {}

    def fetch_ticker(self, sym):
        return {"last": 61000.0}

    def amount_to_precision(self, sym, v):
        return f"{int(v * 1000) / 1000:.3f}"

    def price_to_precision(self, sym, v):
        return str(v)

    def market(self, sym):
        return {"id": sym.split("/")[0] + "USDT"}

    def handle_product_type_and_params(self, market, params):
        return "USDT-FUTURES", params

    def close_position(self, sym, side=None, params=None):
        self.calls.append(("close_position", sym, side))
        self.pos = [p for p in self.pos if p["symbol"] != sym]

    def create_order(self, *a):
        plan = self._plan_create(a[0], a[2], a[3], a[5] if len(a) > 5 else None, a[4] if len(a) > 4 else None)
        if plan:
            return plan
        self.calls.append(("create_order",) + a)
        return {"id": "1"}

    def cancel_order(self, oid, sym, params=None):
        if self._plan_cancel(oid, params):
            return
        self.calls.append(("cancel_order", oid, sym, params))

    def set_margin_mode(self, *a):
        pass

    def set_leverage(self, *a):
        self.calls.append(("set_leverage",) + a)

    def privateMixPostV2MixOrderPlacePosTpsl(self, req):  # noqa: N802 - Name von ccxt
        self.calls.append(("tpsl", req))


def test_account_view_and_actions(tmp_path, monkeypatch):
    from bot.account import PROFILE_KEYS, Account

    for n in (*PROFILE_KEYS["live"], *PROFILE_KEYS["demo"], "BITGET_DEMO"):
        monkeypatch.delenv(n, raising=False)

    env = tmp_path / ".env"
    env.write_text("TELEGRAM_TOKEN=abc\n", encoding="utf-8")
    fake = {}

    def factory(api, demo=False):
        c = FakeBitget(api, demo)
        if demo:
            fake["c"] = c
        return c

    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, ["BTC/USDT:USDT"], factory=factory, env_path=env)
    assert not acc.connected
    with pytest.raises(RuntimeError):
        acc.connect("falsch", "s", "p")
    assert not acc.connected and "BITGET" not in env.read_text()  # falsche Schluessel nie speichern
    acc.connect(" key1 ", "sec", "pass", demo=True)
    text = env.read_text()
    assert "BITGET_DEMO_API_KEY=key1" in text and "TELEGRAM_TOKEN=abc" in text and "BITGET_API_KEY" not in text
    v = acc.view()
    assert v["connected"] and v["demo"] and v["balance"]["free"] == 11.5
    assert v["profiles"] == {"live": False, "demo": True}
    # Echtkonto ist getrennt: umschalten -> noch nicht verbunden, eigene Schluessel
    acc.switch("live")
    assert not acc.view()["connected"] and acc.view()["active"] == "live"
    acc.connect("livekey", "s2", "p2", demo=False)
    assert "BITGET_API_KEY=livekey" in env.read_text() and acc.view()["profiles"] == {"live": True, "demo": True}
    acc.switch("demo")
    v = acc.view()
    assert v["positions"][0]["liq"] == 54500 and v["unrealized"] == 4.0
    assert v["orders"][0]["kind"] == "tpsl" and v["orders"][0]["trigger"] == 59000
    assert v["history"][0]["exit"] == 3012 and v["history"][0]["pnl"] == -1.2
    c = fake["c"]
    acc.close("BTC/USDT:USDT", 0.5)
    assert c.calls[-1][0] == "create_order" and c.calls[-1][3] == "sell" and c.calls[-1][4] == 0.002
    acc.set_tpsl("BTC/USDT:USDT", 59500, None)
    req = [x for x in c.calls if x[0] == "tpsl"][-1][1]
    assert req["stopLossTriggerPrice"] == "59500" and "stopSurplusTriggerPrice" not in req
    acc.cancel("77", "BTC/USDT:USDT", "tpsl")
    assert c.calls[-1] == ("cancel_order", "77", "BTC/USDT:USDT", {"trigger": True, "planType": "profit_loss"})
    with pytest.raises(ValueError):  # Stop auf der falschen Seite
        acc.order("BTC/USDT:USDT", "long", 100, 10, sl=62000)
    with pytest.raises(ValueError):  # ohne Stop keine Order
        acc.order("BTC/USDT:USDT", "long", 100, 10, sl=None)
    acc.order("BTC/USDT:USDT", "short", 122, 5, sl=62000, tp=58000)
    o = c.calls[-1]
    assert o[1:5] == ("BTC/USDT:USDT", "market", "sell", 0.002) and o[6]["stopLoss"]["triggerPrice"] == 62000
    acc.close("BTC/USDT:USDT")
    assert ("close_position", "BTC/USDT:USDT", None) in c.calls   # One-Way: ohne holdSide
    assert acc.close_all() == "Keine offene Position"
    acc.disconnect(forget=True)
    text = env.read_text()
    assert "BITGET_DEMO_API_KEY=\n" in text and "BITGET_API_KEY=livekey" in text and not acc.connected


def test_dashboard_actions_need_token(tmp_path, monkeypatch):
    import json
    import re
    import urllib.error
    import urllib.request

    from bot.account import Account
    from bot.dashboard import start_dashboard

    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    client = FakeClient(synthetic(1000, 3))
    bot = engine.Bot(cfg, PaperExchange(cfg, client), FakeContext())
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, [], factory=lambda api, demo=False: FakeBitget(api, demo), env_path=tmp_path / ".env")
    stop = tmp_path / "STOP"
    srv = start_dashboard(bot, 8093, acc, stop_file=stop)
    try:
        html = urllib.request.urlopen("http://127.0.0.1:8093/").read().decode()
        token = re.search(r'const TOKEN = "([0-9a-f]{32})"', html).group(1)

        def post(path, body, tok):
            req = urllib.request.Request("http://127.0.0.1:8093" + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "X-Token": tok})
            try:
                return json.loads(urllib.request.urlopen(req).read())
            except urllib.error.HTTPError as e:
                return {"code": e.code, **json.loads(e.read())}

        assert post("/api/bot/pause", {"on": True}, "falsch")["code"] == 403
        assert not stop.exists()
        assert post("/api/bot/pause", {"on": True}, token)["ok"] and stop.exists()
        assert post("/api/bot/pause", {"on": False}, token)["ok"] and not stop.exists()
        r = post("/api/account/connect", {"key": "k", "secret": "s", "password": "p"}, token)
        assert r["ok"]
        view = json.loads(urllib.request.urlopen("http://127.0.0.1:8093/api/account").read())
        assert view["connected"] and view["balance"]["total"] == 35.5 and "k" not in json.dumps(view.get("error"))
        assert "secret" not in json.dumps(view)
        r = post("/api/bot/close", {"symbol": "AAA/USDT:USDT"}, token)
        assert not r["ok"] and "keine Position" in r["msg"]
        assert "keine offene" in post("/api/bot/close_all", {}, token)["msg"]
        r = post("/api/account/close_all", {}, token)
        assert r["ok"] and "1 Position" in r["msg"]
        # Modus: Echtgeld nur mit JA und nur mit Echtkonto-Schluesseln
        from bot.dashboard import switch_mode
        for n in ("BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE", "BITGET_DEMO", "BOT_MODE"):
            monkeypatch.delenv(n, raising=False)
        with pytest.raises(RuntimeError):
            switch_mode(bot, {"mode": "live", "confirm": "JA"}, restart=False, env_path=tmp_path / ".env")
        monkeypatch.setenv("BITGET_API_KEY", "k")
        monkeypatch.setenv("BITGET_API_SECRET", "s")
        monkeypatch.setenv("BITGET_API_PASSPHRASE", "p")
        with pytest.raises(RuntimeError):
            switch_mode(bot, {"mode": "live"}, restart=False, env_path=tmp_path / ".env")
        msg = switch_mode(bot, {"mode": "live", "confirm": "JA"}, restart=False, env_path=tmp_path / ".env")
        assert "gespeichert" in msg and "BOT_MODE=live" in (tmp_path / ".env").read_text()
    finally:
        srv.shutdown()


def test_order_blocks_detection_and_filter():
    import numpy as np
    from bot.orderblocks import detect, ob_blocks

    # rote Kerze (Bar 2) -> kraeftiger Anstieg -> Ruecklauf in die Zone -> haelt
    o = np.array([100, 100, 100.5, 99.8, 101.0, 103.0, 102.0, 100.4, 100.6, 101.5])
    c = np.array([100, 100.5, 99.8, 101.0, 103.0, 102.5, 100.6, 100.5, 101.4, 102.0])
    h = np.maximum(o, c) + 0.2
    lo = np.minimum(o, c) - 0.2
    atr = np.full(10, 1.0)
    above, below, support, zones = detect(o, h, lo, c, atr, disp_atr=1.5, look=3)
    z = [z for z in zones if z["kind"] == "bull"][0]
    assert z["i"] == 2 and z["top"] == pytest.approx(100.7) and z["bottom"] == pytest.approx(99.6)
    assert z["confirm"] == 4                     # erst mit Bar 4 bekannt (kein Blick in die Zukunft)
    assert np.isnan(below[3]) and below[4] == pytest.approx(100.7)
    assert support[7] == 1                        # Antest in Bar 7 gehalten
    # Ergebnis bis Bar t haengt nie von spaeteren Bars ab
    a2, b2, s2, _ = detect(o[:6], h[:6], lo[:6], c[:6], atr[:6], 1.5, 3)
    assert np.array_equal(np.nan_to_num(b2), np.nan_to_num(below[:6])) and np.array_equal(s2, support[:6])
    # Zone bricht, wenn ein Bar darunter schliesst
    c2 = c.copy(); c2[8] = 99.0; lo2 = np.minimum(o, c2) - 0.2
    *_, zones2 = detect(o, h, lo2, c2, atr, 1.5, 3)
    assert not [z for z in zones2 if z["kind"] == "bull"]
    s = {"ob_filter": "avoid", "ob_room": 0.6}
    assert ob_blocks("long", 100, 2.0, 100.8, np.nan, 0, s)       # Widerstand nach 0,8 < 1,2
    assert not ob_blocks("long", 100, 2.0, 101.5, np.nan, 0, s)
    assert not ob_blocks("long", 100, 2.0, np.nan, np.nan, 0, s)
    assert ob_blocks("short", 100, 2.0, np.nan, 99.5, 0, s)
    assert ob_blocks("long", 100, 2.0, np.nan, np.nan, 0, {"ob_filter": "confirm"})
    assert not ob_blocks("long", 100, 2.0, np.nan, np.nan, 1, {"ob_filter": "confirm"})
    assert not ob_blocks("long", 100, 2.0, 100.1, np.nan, 0, {"ob_filter": "off"})
    assert not ob_blocks("long", 100, 2.0, 100.1, np.nan, 0, {"ob_filter": False})  # YAML: off -> False


def test_order_book_walls():
    from bot.flow import book_walls, wall_blocks

    bids = [[100 - i * 0.01, 1.0] for i in range(1, 200)] + [[99.0, 80.0]]
    asks = [[100 + i * 0.01, 1.0] for i in range(1, 200)] + [[101.2, 60.0]]
    w = book_walls({"bids": sorted(bids, reverse=True), "asks": sorted(asks)}, 2.0, 0.1, 3.0)
    assert w["bids"][0]["price"] == pytest.approx(99.0, abs=0.05) and w["bids"][0]["usdt"] > 7000
    assert w["asks"][0]["price"] == pytest.approx(101.2, abs=0.05)
    assert w["range_pct"] > 1.5
    cfg = {"wall_filter": True, "wall_block_x": 5.0}
    assert wall_blocks("long", 100.0, 102.0, w, cfg)          # Verkaufs-Wand vor dem Ziel
    assert not wall_blocks("long", 100.0, 101.0, w, cfg)      # Ziel vor der Wand
    assert not wall_blocks("long", 100.0, 102.0, w, {"wall_filter": False})
    assert book_walls({"bids": [], "asks": []})["bids"] == []


def test_orderblock_comparison_runs():
    from bot import optimize

    saved = (optimize.OB_DISP, optimize.OB_MODES)
    try:
        optimize.OB_DISP = [1.5]
        optimize.OB_MODES = {k: optimize.OB_MODES[k] for k in ("ohne", "meiden_60%", "nur_nach_antest")}
        cfg = copy.deepcopy(CFG)
        cfg["strategy"]["strategies"] = ["trend", "range", "breakout"]
        data = {"AAA/USDT:USDT": synthetic(6000, 1)}
        res = optimize.compare_orderblocks(cfg, data, {"AAA/USDT:USDT": (0.0001, 0.0001)}, base_tf="5m")
        assert list(res.order_blocks) == ["ohne", "meiden_60%", "nur_nach_antest"]
        assert res.iloc[0]["aussortiert_test"] == 0 and res.iloc[2]["aussortiert_test"] > 0
        assert res.iloc[2]["test_trades"] < res.iloc[0]["test_trades"]
    finally:
        optimize.OB_DISP, optimize.OB_MODES = saved


def test_signal_log_explains_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["strategy"]["strategies"] = ["trend", "range", "breakout"]  # unabhaengig von config.yaml
    client = FakeClient(synthetic(2000, 3))
    bot = engine.Bot(cfg, PaperExchange(cfg, client), FakeContext())
    for i in range(400, 2000):
        client.i = i
        bot.step()
    log_ = bot.state["signal_log"]
    assert log_ and any(x["traded"] for x in log_)
    keys = [(x["symbol"], x["tf"], x["why"], x["ms"]) for x in log_]
    assert len(keys) == len(set(keys))                 # jedes Signal nur einmal
    assert all(x["why"] for x in log_)
    assert bot.status["signal_log"]
    bot._hb = 0
    bot._heartbeat()                                    # darf nicht abstuerzen


class FakeBitgetExchange(FakeClient):
    """Simulierte Bitget-Boerse fuer den Test- und Echtkonto-Weg (Orders, Stops, Ausfuehrung)."""

    def __init__(self, df):
        super().__init__(df)
        self.cash = 35.0
        self.pos = None          # {"side", "contracts", "entryPrice", "sl", "tp"}
        self.orders = {}         # id -> order
        self.hist = []
        self.n = 0
        self.log = []

    def milliseconds(self):
        import time
        return int(time.time() * 1000)   # echte Uhr wie bei Bitget (Ausfuehrung kurz VOR der Registrierung)

    def _px(self):
        return float(self.df.iloc[self.i - 1]["close"])

    def _settle(self):
        px = self._px()
        for oid, o in list(self.orders.items()):
            hit = px <= o["price"] if o["side"] == "buy" else px >= o["price"]
            if hit:
                del self.orders[oid]
                self._fill(o["side"], o["amount"], o["price"], o, reduce=o.get("reduceOnly"))
        if self.pos:
            p = self.pos
            long = p["side"] == "long"
            if p.get("sl") and (px <= p["sl"] if long else px >= p["sl"]):
                self._close(px)
            elif p.get("tp") and (px >= p["tp"] if long else px <= p["tp"]):
                self._close(p["tp"])

    def _fill(self, side, amount, price, params, reduce=False):
        if reduce or (self.pos and ((side == "sell") == (self.pos["side"] == "long"))):
            if self.pos:
                if amount >= self.pos["contracts"] - 1e-12:
                    self._close(price)
                else:
                    self.pos["contracts"] -= amount
            return
        self.pos = {"side": "long" if side == "buy" else "short", "contracts": amount, "entryPrice": price,
                    "sl": (params.get("stopLoss") or {}).get("triggerPrice"),
                    "tp": (params.get("takeProfit") or {}).get("triggerPrice"), "ts": self.milliseconds()}

    def _close(self, price):
        p = self.pos
        sign = 1 if p["side"] == "long" else -1
        pnl = sign * (price - p["entryPrice"]) * p["contracts"]
        self.cash += pnl
        self.hist.append({"symbol": "AAA/USDT:USDT", "realizedPnl": pnl, "timestamp": p["ts"],
                          "lastUpdateTimestamp": self.milliseconds()})
        self.pos = None
        for oid in [k for k, o in self.orders.items() if o.get("reduceOnly")]:
            del self.orders[oid]

    # --- ccxt-Schnittstelle ---
    def fetch_balance(self, params=None):
        return {"USDT": {"total": self.cash, "free": self.cash, "used": 0.0}}

    def fetch_positions(self, symbols=None):
        self._settle()
        if not self.pos:
            return []
        return [{"symbol": "AAA/USDT:USDT", "side": self.pos["side"], "contracts": self.pos["contracts"],
                 "entryPrice": self.pos["entryPrice"]}]

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        params = params or {}
        self.n += 1
        oid = str(self.n)
        self.log.append((type_, side, amount, price, dict(params)))
        px = self._px()
        if type_ == "market":
            self._fill(side, amount, px, params, reduce=params.get("reduceOnly"))
            return {"id": oid, "average": px}
        crosses = px <= price if side == "buy" else px >= price
        if params.get("postOnly") and crosses:
            import ccxt
            raise ccxt.InvalidOrder("bitget post only order would be filled immediately")
        if crosses:
            self._fill(side, amount, price, params, reduce=params.get("reduceOnly"))
        else:
            self.orders[oid] = {"side": side, "amount": amount, "price": price, **params}
        return {"id": oid}

    def cancel_order(self, oid, symbol=None, params=None):
        if oid not in self.orders:
            raise RuntimeError("order not found")
        del self.orders[oid]

    def fetch_positions_history(self, symbols=None, since=None, limit=None):
        return [h for h in self.hist if since is None or h["timestamp"] >= since]

    def set_position_mode(self, *a, **k):
        pass

    def set_margin_mode(self, *a, **k):
        pass

    def set_leverage(self, *a, **k):
        pass

    def handle_product_type_and_params(self, market, params):
        return "USDT-FUTURES", params

    def price_to_precision(self, sym, v):
        return str(v)

    def privateMixPostV2MixOrderPlacePosTpsl(self, req):  # noqa: N802 - Name von ccxt
        if self.pos:
            if req.get("stopLossTriggerPrice"):
                self.pos["sl"] = float(req["stopLossTriggerPrice"])
            if req.get("stopSurplusTriggerPrice"):
                self.pos["tp"] = float(req["stopSurplusTriggerPrice"])


def test_demo_mode_against_fake_bitget(tmp_path, monkeypatch):
    """Kompletter Test-/Echtkonto-Weg: Limit-Einstieg (Post-Only, sonst normal), Stop auf der Boerse,
    TP als Limit-Order, Schliessen, Ergebnis aus der Boersen-Historie - ohne einen Fehler im Log."""
    import logging

    from bot import exchange as exmod

    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    fake = FakeBitgetExchange(synthetic(2600, 3))
    monkeypatch.setattr(exmod, "make_client", lambda api=None, demo=False: fake)
    cfg = copy.deepcopy(CFG)
    cfg["mode"] = "demo"
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["fees"].update(entry_order="limit", tp_order="limit")
    # kurze Trend-EMAs: die Testdaten reichen sonst nicht fuer den 200er-Trend
    cfg["strategy"].update(strategies=["trend", "range", "breakout"], trend_ema_fast=20, trend_ema_slow=50)
    ex = exmod.BitgetExchange(cfg)
    bot = engine.Bot(cfg, ex, FakeContext())
    problems = []

    class Catch(logging.Handler):
        def emit(self, r):
            if r.levelno >= logging.WARNING:
                problems.append(r.getMessage())

    h = Catch()
    logging.getLogger("bot").addHandler(h)
    try:
        for i in range(400, 2600):
            fake.i = i
            bot.step()
            assert set(bot.state["meta"]) <= {"AAA/USDT:USDT"}
            if fake.pos and "AAA/USDT:USDT" in bot.state["meta"]:
                assert fake.pos["sl"] is not None          # jede Position hat einen Stop auf der Boerse
    finally:
        logging.getLogger("bot").removeHandler(h)
    hist = bot.state["history"]
    assert len(hist) >= 3, (len(fake.log), fake.log[:3], bot.state.get("signal_log", [])[:5])
    assert any(o[0] == "limit" and o[4].get("postOnly") for o in fake.log)      # Maker-Einstieg versucht
    assert any(o[0] == "limit" and o[4].get("reduceOnly") for o in fake.log)    # TP als Limit-Order
    assert sum(t["pnl"] for t in hist) == pytest.approx(fake.cash - 35.0, abs=1e-6) or fake.pos
    assert not [p for p in problems if "Fehler" in p or "Traceback" in p], problems[:5]


def test_live_macro_score_loads(tmp_path, monkeypatch):
    """Makro-Ampel live: Tabelle wird geladen und der aktuelle Wert gelesen (Fehler vom 01.10. abgesichert)."""
    import time as _time

    import pandas as pd
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    from bot.exchange import PaperExchange

    now_ms = int(_time.time() * 1000)
    scores = pd.DataFrame({"crypto": [0.5, -0.6], "gold": [0.2, 0.4], "avail": [now_ms - 3 * 86_400_000, now_ms - 1000]})
    calls = []
    monkeypatch.setattr(engine, "fetch_macro", lambda days: calls.append(days) or scores)
    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["strategy"]["macro_filter"] = True
    bot = engine.Bot(cfg, PaperExchange(cfg, FakeClient(synthetic(500, 1))), FakeContext())
    assert bot._macro_score("BTC/USDT:USDT") == -0.6
    assert bot._macro_score("XAU/USDT:USDT") == 0.4
    assert len(calls) == 1                                  # 6 Stunden zwischengespeichert
    monkeypatch.setattr(engine, "fetch_macro", lambda days: None)   # Quelle faellt aus -> alter Wert bleibt
    bot._macro_cache = (0, scores)
    assert bot._macro_score("BTC/USDT:USDT") == -0.6


def test_remote_access_requires_password(tmp_path, monkeypatch):
    import http.client
    import json
    import re

    from bot.dashboard import check_password, hash_password, start_dashboard

    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    from bot.exchange import PaperExchange

    h = hash_password("richtig-geheim-123")
    assert "richtig" not in h and check_password("richtig-geheim-123", h) and not check_password("falsch", h)
    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    bot = engine.Bot(cfg, PaperExchange(cfg, FakeClient(synthetic(500, 1))), FakeContext())
    bot.status = {"symbols": {}, "x": 1}
    srv = start_dashboard(bot, 8094, None, stop_file=tmp_path / "STOP", remote_pw_hash=h)
    remote_host = "100.101.102.103:8094"   # wie ueber Tailscale vom Handy

    def req(method, path, body=None, headers=None, host=remote_host):
        c = http.client.HTTPConnection("127.0.0.1", 8094, timeout=10)
        c.request(method, path, body=body, headers={"Host": host, **(headers or {})})
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read().decode()

    try:
        assert req("GET", "/api/status", host="127.0.0.1:8094")[0] == 200     # am PC selbst ohne Anmeldung
        st, _, body = req("GET", "/")
        assert st == 200 and "Passwort" in body and "TOKEN" not in body       # vom Handy: erst Anmeldung
        assert req("GET", "/api/status")[0] == 401
        form = {"Content-Type": "application/x-www-form-urlencoded"}
        assert req("POST", "/login", "password=falsch", form)[0] == 401
        st, hd, _ = req("POST", "/login", "password=richtig-geheim-123", form)
        assert st == 303 and "HttpOnly" in hd["Set-Cookie"] and "SameSite=Strict" in hd["Set-Cookie"]
        cookie = {"Cookie": hd["Set-Cookie"].split(";")[0]}
        st, _, body = req("GET", "/api/status", headers=cookie)
        assert st == 200 and json.loads(body)["x"] == 1
        page = req("GET", "/", headers=cookie)[2]
        token = re.search(r'const TOKEN = "([0-9a-f]{32})"', page).group(1)
        act = {**cookie, "X-Token": token, "Content-Type": "application/json"}
        assert req("POST", "/api/bot/pause", '{"on": true}', act)[0] == 200 and (tmp_path / "STOP").exists()
        assert req("POST", "/api/bot/pause", '{"on": false}', {"X-Token": token})[0] == 403    # ohne Anmeldung
        assert req("POST", "/api/bot/pause", '{"on": false}', {**act, "Origin": "http://boese.example"})[0] == 403
        for _ in range(5):                                                      # Sperre nach 5 Fehlversuchen
            req("POST", "/login", "password=falsch", form)
        assert req("POST", "/login", "password=richtig-geheim-123", form)[0] == 429
    finally:
        srv.shutdown()
        srv.server_close()


def test_bitget_errors_are_explained(tmp_path):
    from bot.account import Account, explain_error

    assert "Passphrase" in explain_error(RuntimeError("bitget GET ... 403 Forbidden Sorry, you have been blocked Cloudflare"))
    assert "Passphrase" in explain_error(RuntimeError('{"code":"40012","msg":"apikey/password is incorrect"}'))
    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, [], factory=lambda api, demo=False: FakeBitget(api, demo))
    with pytest.raises(ValueError, match="Berechtigungen"):
        acc.connect("bg_123", "abc", "Lesen Schreiben Futures")      # Berechtigungen statt Passphrase
    seen = {}
    acc2 = Account(cfg, [], factory=lambda api, demo=False: seen.update(api) or FakeBitget(api, demo),
                   env_path=tmp_path / ".env")
    acc2.connect(" bg_123 ", "d989a9be3382\ne0cd 024c\u200b", "MeinPass123")   # zweizeilig kopiertes Secret
    assert seen == {"key": "bg_123", "secret": "d989a9be3382e0cd024c", "password": "MeinPass123"}


def test_chart_data_any_timeframe_and_brain(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(engine, "STOP_FILE", tmp_path / "STOP")
    from bot.exchange import PaperExchange

    cfg = copy.deepcopy(CFG)
    cfg["symbols"] = ["AAA/USDT:USDT"]
    cfg["tf_select"], cfg["timeframes"] = "adaptive", ["5m", "15m"]
    client = FakeClient(synthetic(1500, 2))
    client.i = 1500
    bot = engine.Bot(cfg, PaperExchange(cfg, client), FakeContext())
    bot.step()
    for tf in ("5m", "1h", "4h"):
        cd = bot.chart_data("AAA/USDT:USDT", tf)
        assert cd["tf"] == tf and len(cd["candles"]) > 5 and len(cd["ema_fast"]) == len(cd["candles"])
        assert all(z["kind"] in ("bull", "bear") and z["top"] >= z["bottom"] for z in cd["order_blocks"])
    with pytest.raises(ValueError):
        bot.chart_data("AAA/USDT:USDT", "7m")
    brain = bot.status["brain"]
    assert set(brain["markets"]["AAA/USDT:USDT"]["cells"]) == {"5m", "15m"}
    assert all(c["text"] for c in brain["markets"]["AAA/USDT:USDT"]["cells"].values())
    assert "risk" in brain and "filters" in brain


def test_quick_orders_are_separate_positions(tmp_path, monkeypatch):
    """MetaTrader-Stil: jede Schnell-Order = eigenes Ticket mit eigenem Stop/Ziel (Teilmenge auf Bitget)."""
    from bot.account import PROFILE_KEYS, Account

    for n in (*PROFILE_KEYS["live"], *PROFILE_KEYS["demo"], "BITGET_DEMO"):
        monkeypatch.delenv(n, raising=False)

    class FB(FakeBitget):
        markets = {"BTC/USDT:USDT": {"swap": True}, "ETH/USDT:USDT": {"swap": True}}
        plans: dict = {}

        def market(self, sym):
            return {"id": sym.split("/")[0] + "USDT", "limits": {"amount": {"min": 0.001}}}

    fake = {}
    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, ["BTC/USDT:USDT"], factory=lambda api, demo=False: fake.setdefault("c", FB(api, demo)),
                  env_path=tmp_path / ".env")
    acc.connect("k", "s", "p", demo=True)
    c = fake["c"]
    # 2,30 USDT Einsatz x 10 = 23 USDT Wert -> 0,000377 BTC -> Mindestmenge 0,001 nicht erreicht
    with pytest.raises(ValueError, match="zu klein"):
        acc.quick_order("BTC/USDT:USDT", "long", 2.3, 10, 10, 20)
    with pytest.raises(ValueError, match="mehr als verfuegbar"):
        acc.quick_order("BTC/USDT:USDT", "long", 50, 10, 10, 20)
    c.fetch_ticker = lambda sym: {"last": 2000.0}
    with pytest.raises(RuntimeError, match="Gegenrichtung"):       # BTC-Long offen -> kein Short
        acc.quick_order("BTC/USDT:USDT", "short", 5, 10, 10, 20)
    with pytest.raises(ValueError, match="Take-Profit"):
        acc.quick_order("BTC/USDT:USDT", "long", 5, 10, 10, 0)
    acc.quick_order("BTC/USDT:USDT", "long", 5.75, 10, 10, 20)
    acc.quick_order("BTC/USDT:USDT", "long", 5.75, 10, 10, 20)
    tks = acc.tickets().open()
    assert len(tks) == 2 and tks[0]["id"] != tks[1]["id"]                    # zwei getrennte Positionen
    t1 = tks[0]
    assert t1["sl"] == pytest.approx(2000 * (1 - 0.10 / 10)) and t1["tp"] == pytest.approx(2000 * (1 + 0.20 / 10))
    plans = [x for x in c.calls if x[0] == "plan"]
    assert {p[1] for p in plans} == {"loss_plan", "profit_plan"} and all(float(p[3]) == t1["amount"] for p in plans)
    # Ticket 1 von Hand schliessen: nur seine Menge, seine Auftraege storniert
    acc.close_ticket(t1["id"])
    last = [x for x in c.calls if x[0] == "create_order"][-1]
    assert last[3] == "sell" and last[4] == t1["amount"] and last[6]["reduceOnly"]
    assert ("cancel_plan", t1["sl_id"]) in c.calls and ("cancel_plan", t1["tp_id"]) in c.calls
    # Ticket 2: Ziel wurde auf Bitget ausgefuehrt -> Stop stornieren, Ticket geschlossen
    t2 = acc.tickets().open()[0]
    c.plans.pop(t2["tp_id"])
    for x in acc.tickets().items:
        x["opened_ms"] -= 60_000        # Schonfrist fuer frisch eroeffnete Positionen vorbei
    acc.refresh(force=True)
    assert not acc.tickets().open()
    closed = [x for x in acc.tickets().items if x["id"] == t2["id"]][0]
    assert closed["why"] == "Ziel erreicht" and closed["pnl"] > 0 and t2["sl_id"] not in c.plans
    assert acc.view()["tickets_closed"][0]["id"] == t2["id"]
    # Tickets ueberleben einen Neustart (Datei)
    acc2 = Account(cfg, [], factory=lambda api, demo=False: FB(api, demo), env_path=tmp_path / ".env")
    acc2.connect("k", "s", "p", demo=True)
    assert len(acc2.tickets().items) == 2


def test_hedge_mode_account_orders(tmp_path, monkeypatch):
    """Bitget-Konto im Hedge-Modus: Orders mit tradeSide (hedged), TP/SL mit holdSide long/short."""
    from bot.account import PROFILE_KEYS, Account
    from bot.exchange import hold_side, is_hedged

    for n in (*PROFILE_KEYS["live"], *PROFILE_KEYS["demo"], "BITGET_DEMO"):
        monkeypatch.delenv(n, raising=False)

    class FH(FakeBitget):
        markets = {"BTC/USDT:USDT": {"swap": True}}

        def market(self, sym):
            return {"id": "BTCUSDT", "limits": {"amount": {"min": 0.001}}}

        def privateMixGetV2MixAccountAccounts(self, req):  # noqa: N802
            return {"data": [{"marginCoin": "USDT", "posMode": "hedge_mode"}]}

        def fetch_ticker(self, sym):
            return {"last": 2000.0}

    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    box = {}
    acc = Account(cfg, [], factory=lambda api, demo=False: box.setdefault("c", FH(api, demo)), env_path=tmp_path / ".env")
    acc.connect("k", "s", "p")
    c = box["c"]
    assert is_hedged(c) and hold_side(c, "short") == "short"
    acc.quick_order("BTC/USDT:USDT", "short", 5.75, 10, 10, 20)   # Gegenrichtung zur offenen Long-Position erlaubt
    order = [x for x in c.calls if x[0] == "create_order"][-1]
    assert order[3] == "sell" and order[6]["hedged"] is True
    assert {p["holdSide"] for p in c.plans.values()} == {"short"}          # Bitget-TP/SL im Hedge-Modus


def test_limit_quick_order_breakeven_and_close_all(tmp_path, monkeypatch):
    from bot.account import PROFILE_KEYS, Account

    for n in (*PROFILE_KEYS["live"], *PROFILE_KEYS["demo"], "BITGET_DEMO"):
        monkeypatch.delenv(n, raising=False)

    class FL(FakeBitget):
        markets = {"ETH/USDT:USDT": {"swap": True}}
        last = 2000.0

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.pos = []
            self.plans, self.orders_ = {}, {}

        def market(self, sym):
            return {"id": "ETHUSDT", "limits": {"amount": {"min": 0.001}}}

        def fetch_ticker(self, sym):
            return {"last": self.last}

        def create_order(self, sym, typ, side, amount, price=None, params=None):
            plan = self._plan_create(sym, side, amount, params, price)
            if plan:
                return plan
            self.calls.append(("create_order", sym, typ, side, amount, price, params))
            oid = f"o{len(self.calls)}"
            self.orders_[oid] = {"status": "open" if typ == "limit" else "closed", "filled": 0 if typ == "limit" else amount,
                                 "average": price or self.last, "price": price}
            return {"id": oid, "average": None if typ == "limit" else self.last}

        def fetch_order(self, oid, sym):
            return self.orders_[oid]

        def cancel_order(self, oid, sym, params=None):
            if self._plan_cancel(oid, params):
                return
            self.calls.append(("cancel_order", oid))
            self.orders_[oid]["status"] = "canceled"

    box = {}
    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, ["ETH/USDT:USDT"], factory=lambda api, demo=False: box.setdefault("c", FL(api, demo)),
                  env_path=tmp_path / ".env")
    acc.connect("k", "s", "p")
    c = box["c"]
    with pytest.raises(ValueError, match="Limit-Preis"):
        acc.quick_order("ETH/USDT:USDT", "long", 5.75, 10, 10, 20, "limit", None)
    msg = acc.quick_order("ETH/USDT:USDT", "long", 5.75, 10, 10, 20, "limit", 1950.0)
    assert "liegt" in msg
    tk = acc.tickets().active()[0]
    assert tk["status"] == "pending" and not c.plans                          # noch kein Stop/Ziel
    acc.refresh(force=True)
    assert acc.tickets().active()[0]["status"] == "pending"
    c.orders_[tk["order_id"]].update(status="closed", filled=tk["amount"], average=1950.0)   # ausgefuehrt
    c.pos = [{"symbol": "ETH/USDT:USDT", "side": "long", "contracts": tk["amount"], "entryPrice": 1950.0}]
    acc.refresh(force=True)
    tk = acc.tickets().active()[0]
    assert tk["status"] == "open" and tk["entry"] == 1950.0 and len(c.plans) == 2
    assert tk["sl"] == pytest.approx(1950 * 0.99) and tk["tp"] == pytest.approx(1950 * 1.02)
    # Stop auf Einstand: erst wenn der Kurs darueber ist
    c.last = 1940.0
    with pytest.raises(RuntimeError, match="Einstand"):
        acc.breakeven(tk["id"])
    c.last = 1970.0
    acc.breakeven(tk["id"])
    tk = acc.tickets().active()[0]
    assert tk["be"] and tk["sl"] > 1950 and len(c.plans) == 2 and c.plans[tk["sl_id"]]["planType"] == "loss_plan"
    # Stop/Ziel im Chart ziehen: neuer Auftrag, alter weg; falsche Seite wird abgelehnt
    with pytest.raises(ValueError, match="sofort"):
        acc.move_level(tk["id"], "sl", 1980.0)
    acc.move_level(tk["id"], "tp", 2050.0)
    acc.move_level(tk["id"], "sl", 1930.0)
    tk = acc.tickets().active()[0]
    assert tk["tp"] == 2050.0 and tk["sl"] == 1930.0 and not tk["be"] and len(c.plans) == 2
    assert float(c.plans[tk["tp_id"]]["triggerPrice"]) == 2050.0
    # Stop/Ziel-Auftraege nicht lesbar -> Einzel-Position bleibt offen (kein Schliessen ohne Beweis)
    real = c.fetch_open_orders
    c.fetch_open_orders = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    acc.refresh(force=True)
    assert acc.tickets().active()[0]["status"] == "open"
    c.fetch_open_orders = real
    # zweite Limit-Order wartet, dann alles schliessen: Position zu, Limit storniert
    acc.quick_order("ETH/USDT:USDT", "long", 5.75, 10, 10, 20, "limit", 1900.0)
    assert len(acc.tickets().active()) == 2
    assert "2 Einzel-Position" in acc.close_all_tickets()
    assert not acc.tickets().active() and not c.plans
    whys = sorted(x["why"] for x in acc.tickets().items)
    assert whys == ["Limit-Order storniert", "manuell"]


def test_position_mode_auto_retry_on_40774():
    """Erkennt der Bot den Positionsmodus falsch, korrigiert er sich bei Bitget-Fehler 40774 selbst."""
    from bot.exchange import is_hedged, mode_safe

    class Raw:
        markets = {}
        mode_hedge = True          # wirklicher Modus des Kontos

        def __init__(self):
            self.sent = []

        def load_markets(self):
            raise RuntimeError("offline")     # Modus-Abfrage schlaegt fehl -> Bot nimmt One-Way an

        def _check(self, hedged_req):
            if hedged_req != self.mode_hedge:
                raise RuntimeError('bitget {"code":"40774","msg":"The order type for unilateral position ..."}')

        def create_order(self, symbol, type, side, amount, price=None, params=None):  # noqa: A002
            self.sent.append(("order", params.get("hedged")))
            self._check(bool(params.get("hedged")))
            return {"id": "1"}

        def close_position(self, symbol, side=None, params=None):
            self.sent.append(("close", side))
            if not self.mode_hedge and side is not None:   # echtes Bitget: One-Way ohne holdSide
                raise RuntimeError('bitget {"code":"40017","msg":"Parameter verification failed holdSide"}')
            self._check(side in ("long", "short"))
            return {"id": "2"}

        def privateMixPostV2MixOrderPlaceTpslOrder(self, req):  # noqa: N802
            self.sent.append(("tpsl", req["holdSide"]))
            self._check(req["holdSide"] in ("long", "short"))
            return {"data": {"orderId": "3"}}

        def privateMixPostV2MixOrderPlacePosTpsl(self, req):  # noqa: N802
            self._check(req["holdSide"] in ("long", "short"))
            return {}

    c = Raw()
    mode_safe(c)
    assert not is_hedged(c)                                   # falsch erkannt
    c.create_order("ETH/USDT:USDT", "market", "buy", 1, None, {})
    assert c.sent == [("order", False), ("order", True)] and is_hedged(c)   # selbst korrigiert
    c.close_position("ETH/USDT:USDT", "buy")
    assert c.sent[-1] == ("close", "long")
    c.privateMixPostV2MixOrderPlaceTpslOrder({"holdSide": "sell", "planType": "loss_plan"})
    assert c.sent[-1] == ("tpsl", "short")
    c.mode_hedge = False                                      # Konto auf One-Way umgestellt
    c.create_order("ETH/USDT:USDT", "market", "buy", 1, None, {})
    assert c.sent[-1] == ("order", False) and not is_hedged(c)
    c.close_position("ETH/USDT:USDT", "long")                 # One-Way: Schliessen ohne holdSide (kein 40017)
    assert c.sent[-1] == ("close", None)
    c._bot_hedged = True                                      # falsch erkannt -> 40017 -> ohne holdSide wiederholt
    c.close_position("ETH/USDT:USDT", "long")
    assert c.sent[-2:] == [("close", "long"), ("close", None)]


def test_quick_order_too_small_message_for_btc(tmp_path, monkeypatch):
    from bot.account import PROFILE_KEYS, Account

    for n in (*PROFILE_KEYS["live"], *PROFILE_KEYS["demo"], "BITGET_DEMO"):
        monkeypatch.delenv(n, raising=False)

    class FBtc(FakeBitget):
        markets = {"BTC/USDT:USDT": {"swap": True}}

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.pos = []

        def market(self, sym):
            return {"id": "BTCUSDT", "limits": {"amount": {"min": 0.0001}}, "precision": {"amount": 0.0001}}

        def fetch_ticker(self, sym):
            return {"last": 85000.0}

        def amount_to_precision(self, sym, v):     # wie ccxt: Fehler, wenn auf 0 gerundet wird
            if v < 0.0001:
                raise RuntimeError("bitget amount of BTC/USDT:USDT must be greater than minimum amount precision of 0.0001")
            return f"{int(v * 10000) / 10000:.4f}"

    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, [], factory=lambda api, demo=False: FBtc(api, demo), env_path=tmp_path / ".env")
    acc.connect("k", "s", "p")
    with pytest.raises(ValueError) as e:
        acc.quick_order("BTC/USDT:USDT", "long", 0.5, 10, 10, 20)
    assert "BTC" in str(e.value) and "8.50" in str(e.value) and "Einsatz" in str(e.value)


def test_position_margin_from_bitget_margin_size():
    from bot.account import Account

    row = Account._pos_row({"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.004, "entryPrice": 60000,
                            "unrealizedPnl": 3.0, "initialMargin": 24.0, "leverage": 10, "notional": 244,
                            "info": {"marginSize": "12.2", "leverage": "20"}})
    assert row["margin"] == 12.2 and row["leverage"] == 20 and row["pnl_pct"] == pytest.approx(3 / 12.2 * 100)
    row = Account._pos_row({"symbol": "X", "contracts": 1, "initialMargin": 5.0, "unrealizedPnl": 1.0, "leverage": 10})
    assert row["margin"] == 5.0 and row["pnl_pct"] == pytest.approx(20.0)


def test_ticket_stops_are_bitget_tpsl_and_leftovers_are_cancelled(tmp_path, monkeypatch):
    """Stop/Ziel = Bitget-Teil-TP/SL (an die Position gebunden, kann nie nachkaufen). Ausloese-/Limit-Orders
    aelterer Versionen werden ersetzt, uebrig gebliebene Auftraege geschlossener Tickets storniert."""
    from bot.account import PROFILE_KEYS, Account

    for n in (*PROFILE_KEYS["live"], *PROFILE_KEYS["demo"], "BITGET_DEMO"):
        monkeypatch.delenv(n, raising=False)
    box = {}
    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, ["BTC/USDT:USDT"], factory=lambda api, demo=False: box.setdefault("c", FakeBitget(api, demo)),
                  env_path=tmp_path / ".env")
    acc.connect("k", "s", "p")
    c = box["c"]
    c.gone = {"77"}
    # alte Version: Stop als Ausloese-Auftrag (konnte nachkaufen)
    old = c.create_order("BTC/USDT:USDT", "market", "sell", 0.001, None,
                         {"triggerPrice": 59000.0, "reduceOnly": True, "triggerType": "mark_price"})["id"]
    acc.tickets().add(symbol="BTC/USDT:USDT", side="long", amount=0.001, entry=60000.0, leverage=10,
                      sl=59000.0, tp=63000.0, sl_id=old, tp_id="", opened_ms=0, prot_v=3)
    # geschlossenes Ticket, dessen Stop noch liegt
    left = c.create_order("BTC/USDT:USDT", "market", "sell", 0.001, None,
                          {"triggerPrice": 58000.0, "reduceOnly": True, "triggerType": "mark_price"})["id"]
    acc.tickets().add(symbol="BTC/USDT:USDT", side="long", amount=0.001, entry=60000.0, leverage=10,
                      sl=58000.0, tp=64000.0, sl_id=left, tp_id="", status="closed", why="Ziel erreicht")
    acc.refresh(force=True)
    tk = acc.tickets().open()[0]
    assert old not in c.plans and left not in c.plans                  # alter Stop ersetzt, Rest storniert
    sl, tp = c.plans[tk["sl_id"]], c.plans[tk["tp_id"]]
    assert sl["kind"] == tp["kind"] == "tpsl" and sl["planType"] == "loss_plan" and tp["planType"] == "profit_plan"
    assert sl["holdSide"] == "buy" and sl["size"] == "0.001" and float(sl["triggerPrice"]) == 59000.0  # One-Way Long
    assert tk["prot_v"] == 4
    acc.refresh(force=True)                                            # nichts doppelt
    assert len(c.plans) == 2


def test_quick_order_refused_where_bot_trades(tmp_path):
    from bot.dashboard import handle_action

    class B:
        cfg = {"mode": "live"}
        state = {"meta": {"BTC/USDT:USDT": {}}, "pending": {}}

    class A:
        active = "live"

        def quick_order(self, *a):
            return "ok"
    with pytest.raises(RuntimeError, match="zusammenlegen"):
        handle_action(B(), A(), "/api/account/quick", {"symbol": "BTC/USDT:USDT", "side": "long"}, tmp_path / "S")
    assert handle_action(B(), A(), "/api/account/quick", {"symbol": "ETH/USDT:USDT", "side": "long"}, tmp_path / "S") == "ok"


def _fc_frame(n, phi, seed):
    """5-Minuten-Kerzen; phi > 0 = Schwung (Rendite folgt der vorigen) -> vorhersagbar."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.002)
    close = 100 * np.exp(np.cumsum(r))
    ts = 1_700_000_000_000 + np.arange(n) * 300_000
    return pd.DataFrame({"ts": ts, "open": np.r_[close[0], close[:-1]], "high": close * 1.001,
                         "low": close * 0.999, "close": close, "volume": 1000 + rng.random(n) * 100})


def test_forecast_learns_real_patterns_and_admits_randomness():
    from bot.forecast import build, path

    fc = build(_fc_frame(6000, 0.6, 1))
    assert fc["useful"] and fc["hit_30m"] > 0.55 and fc["decision"] in ("LONG", "SHORT")
    assert fc["confidence"] > 0.5 and len(fc["nodes"]) == 7 and fc["nodes"][-1]["sec"] == 30 * 60
    noise = build(_fc_frame(6000, 0.0, 9))
    assert not noise["useful"] and noise["confidence"] == 0.5     # Zufall: keine vorgetaeuschte Sicherheit
    assert noise["decision"] in ("LONG", "SHORT")                  # entscheidet sich trotzdem
    assert build(_fc_frame(200, 0.6, 3)) is None                   # zu wenig Daten
    p = path(fc, 100.0, fc["time"] + 60, seed=1)
    assert p[0]["price"] == pytest.approx(100.0) and p[-1]["time"] == fc["time"] + 30 * 60
    assert all(x["band"] >= 0 for x in p)


def test_forecast_follows_btc_for_altcoins():
    """Altcoin folgt BTC mit Verzoegerung -> die KI nutzt die BTC-Merkmale."""
    from bot.forecast import build

    btc = _fc_frame(3000, 0.0, 11)
    r_btc = np.diff(np.log(btc["close"].to_numpy()), prepend=np.log(btc["close"].iloc[0]))
    rng = np.random.default_rng(12)
    r_alt = np.r_[0.0, 0.8 * r_btc[:-1]] + rng.normal(0, 0.001, len(r_btc))
    alt = btc.copy()
    alt["close"] = 50 * np.exp(np.cumsum(r_alt))
    alt["open"], alt["high"], alt["low"] = np.r_[alt["close"].iloc[0], alt["close"].to_numpy()[:-1]], \
        alt["close"] * 1.001, alt["close"] * 0.999
    z5 = lambda fc: fc["quality"][0]["z"]           # Wirkung zeigt sich in den naechsten 5 Minuten
    assert z5(build(alt)) < 2                        # ohne BTC: nicht vorhersagbar
    assert z5(build(alt, btc)) > 4                   # mit BTC: Muster deutlich gefunden


def test_forecast_memory_log_and_bot_filter(tmp_path):
    from bot.forecast import CandleMemory, ForecastLog, ki_active, ki_blocks

    mem = CandleMemory(tmp_path)
    df = _fc_frame(400, 0.0, 4)
    out = mem.merge("BTC/USDT:USDT", df.iloc[:300])
    out = CandleMemory(tmp_path).merge("BTC/USDT:USDT", df.iloc[200:])   # neu gestartet: Gedaechtnis + neue Kerzen
    assert len(out) == 400 and out["ts"].is_monotonic_increasing

    lg = ForecastLog(tmp_path / "log.json")
    t0 = 1_700_000_000
    fc = {"time": t0, "decision": "LONG", "confidence": 0.6, "band_pct": 0.5}
    lg.add("X", fc, 100.0)
    lg.add("X", fc, 100.0)                           # eine Entscheidung je Kerze
    lg.resolve("X", pd.DataFrame({"ts": [(t0 + 25 * 60) * 1000], "close": [100.8]}))
    st = ForecastLog(tmp_path / "log.json").stats("X")
    assert st["n"] == 1 and st["hit"] == 1.0 and st["hit_sure"] == 1.0

    s = {"ki_filter": "auto", "ki_min_live": 100, "ki_min_hit": 0.53, "ki_min_conf": 0.55}
    good = {"ok": True, "useful": True, "decision": "SHORT", "confidence": 0.6, "live": {"n": 150, "hit": 0.56}}
    assert ki_active(good, s)[0] and ki_blocks("long", good, s) and not ki_blocks("short", good, s)
    young = {**good, "live": {"n": 20, "hit": 0.7}}
    assert not ki_active(young, s)[0] and not ki_blocks("long", young, s)     # noch nicht bewiesen
    bad = {**good, "live": {"n": 150, "hit": 0.49}}
    assert not ki_blocks("long", bad, s)
    unsure = {**good, "confidence": 0.52}
    assert not ki_blocks("long", unsure, s)                                     # nur bei klarer Gegen-Meinung
    assert ki_blocks("long", young, {**s, "ki_filter": "on"})
    assert not ki_blocks("long", good, {**s, "ki_filter": "off"})


def test_diagnose_compares_each_filter():
    from bot import optimize

    saved = optimize.DIAG_VARIANTS
    try:
        optimize.DIAG_VARIANTS = {k: saved[k] for k in ("aktuell (config.yaml)", "ohne Order-Block-Filter",
                                                        "alle Zusatzfilter aus")}
        cfg = copy.deepcopy(CFG)
        cfg["strategy"].update(strategies=["trend", "range", "breakout"], ob_filter="confirm")
        data = {"AAA/USDT:USDT": synthetic(6000, 1)}
        res, now = optimize.diagnose(cfg, data, {"AAA/USDT:USDT": (0.0001, 0.0001)}, base_tf="5m", recent_days=7)
        assert list(res.variante) == list(optimize.DIAG_VARIANTS)
        assert now["signale"] >= now["trades"] and now["aussortiert"]["orderblock"] > 0
        assert res.iloc[1]["test_trades"] > res.iloc[0]["test_trades"]   # Filter aus -> mehr Trades
    finally:
        optimize.DIAG_VARIANTS = saved


def test_account_stats_and_history_paging(tmp_path, monkeypatch):
    import time as _t
    from bot.account import PROFILE_KEYS, Account, account_stats

    now = 1_790_000_000_000
    hist = [{"closed_ms": now - 3_600_000, "pnl": 2.0, "fees": -0.1},
            {"closed_ms": now - 3 * 86_400_000, "pnl": -1.0, "fees": -0.1},
            {"closed_ms": now - 20 * 86_400_000, "pnl": 0.5, "fees": -0.05}]
    pos = [{"side": "long", "margin": 5.0, "pnl": 0.5, "value": 50.0}, {"side": "short", "margin": 5.0, "pnl": -0.2, "value": 49.0}]
    st = account_stats({"total": 40.0}, pos, [{"kind": "normal"}, {"kind": "tpsl"}], hist, now)
    assert st["open_positions"] == 2 and st["longs"] == 1 and st["margin"] == 10.0 and st["margin_pct"] == 25.0
    assert st["unrealized"] == pytest.approx(0.3) and st["unrealized_pct"] == pytest.approx(3.0) and st["open_orders"] == 1
    assert st["d1"]["trades"] == 1 and st["d1"]["pnl"] == 2.0
    assert st["d7"]["trades"] == 2 and st["d7"]["pnl"] == 1.0 and st["d7"]["pf"] == 2.0
    assert st["d30"]["trades"] == 3 and st["d30"]["wins"] == 2 and st["d30"]["fees"] == pytest.approx(0.25)

    for n in (*PROFILE_KEYS["live"], *PROFILE_KEYS["demo"], "BITGET_DEMO"):
        monkeypatch.delenv(n, raising=False)

    class FP(FakeBitget):
        def fetch_positions_history(self, symbols=None, since=None, limit=None, params=None):
            until = (params or {}).get("until") or int(_t.time() * 1000)
            rows = [{"symbol": "BTC/USDT:USDT", "realizedPnl": 1.0, "lastUpdateTimestamp": until - i * 600_000,
                     "info": {"positionId": str(until - i * 600_000)}} for i in range(100)]
            return [r for r in rows if r["lastUpdateTimestamp"] >= since][:limit]

    cfg = copy.deepcopy(CFG)
    cfg["api"] = {"key": "", "secret": "", "password": ""}
    acc = Account(cfg, ["BTC/USDT:USDT"], factory=lambda api, demo=False: FP(api, demo), env_path=tmp_path / ".env")
    acc.connect("k", "s", "p")
    v = acc.view()
    assert len(v["history"]) > 100 and v["stats"]["d1"]["trades"] > 100       # mehr als eine Seite


def _speed_df(n, drift, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(drift + rng.normal(0, 0.0002, n)))
    ts = 1_700_000_000_000 + np.arange(n) * 60_000
    return pd.DataFrame({"ts": ts, "open": np.r_[close[0], close[:-1]], "high": close * 1.0005,
                         "low": close * 0.9995, "close": close, "volume": np.full(n, 1000.0)})


def test_speed_signal_follows_all_timeframes_and_book():
    from bot.speed import levels, micro_signal

    book_up = {"bids": [[99.9, 50]] * 20, "asks": [[100.1, 10]] * 20}
    book_dn = {"bids": [[99.9, 10]] * 20, "asks": [[100.1, 50]] * 20}
    up, dn = _speed_df(120, 0.0008), _speed_df(120, -0.0008)
    assert micro_signal(up, book_up, 3)[0] == 1
    assert micro_signal(dn, book_dn, 3)[0] == -1
    assert micro_signal(up, book_dn, 1)[0] == 0          # vorsichtig: Orderbuch dagegen -> nichts
    assert micro_signal(up.head(20), book_up)[0] == 0    # zu wenig Daten
    p = {"maker": 0.0002, "tp_atr": 1.0, "sl_atr": 1.0, "min_tp_fee_x": 3.0}
    sl, tp = levels(up, 100.0, 1, p)
    assert sl < 100.0 < tp and (tp - 100.0) >= 3 * 100.0 * 0.0004 - 1e-9   # Ziel deckt Gebuehren mehrfach


def test_speed_session_in_simulation():
    from bot.speed import PaperBroker, SpeedTrader

    feed = {"df": _speed_df(120, 0.0008), "last": 100.0}
    book = {"bids": [[99.99, 50]] * 20, "asks": [[100.01, 10]] * 20}

    def data(sym):
        last = feed["last"]
        return feed["df"], book, {"last": last, "bid": last - 0.01, "ask": last + 0.01}

    sp = SpeedTrader(data, {"loop_s": 0.0})
    broker = PaperBroker(lambda s: sp.market[s], sp.p)
    sp.cur, sp.broker = {**sp.p, "margin_usdt": 5, "leverage": 10}, broker
    sp.session = {"symbols": ["BTC/USDT:USDT"], "end": time.time() + 300, "equity0": 35.0}
    sp.slots = {"BTC/USDT:USDT": {"state": "idle"}}
    sp.step("BTC/USDT:USDT")
    s = sp.slots["BTC/USDT:USDT"]
    assert s["state"] == "pending" and s["side"] == 1 and s["price"] == pytest.approx(99.99)
    sp.step("BTC/USDT:USDT")
    assert s["state"] == "pending"                       # Kurs noch nicht durch die Limit-Order gelaufen
    feed["last"] = 99.98
    sp.step("BTC/USDT:USDT")
    assert s["state"] == "open" and s["entry"] == pytest.approx(99.99)
    feed["last"] = s["tp"] + 0.01                        # Ziel durchlaufen
    sp.step("BTC/USDT:USDT")
    t = sp.trades[-1]
    assert t["why"] == "ziel" and t["gross"] > 0 and t["net"] == pytest.approx(t["gross"] - t["fees"], abs=1e-5)
    assert t["fees"] == pytest.approx((t["entry"] + t["exit"]) * t["qty"] * 0.0002, abs=1e-5)   # beide Seiten Maker
    assert t["net"] > 0                                  # Ziel ist groesser als die Gebuehren
    # zweiter Trade: Stop
    feed["last"] = 100.5
    sp.step("BTC/USDT:USDT")
    feed["last"] = s["price"] - 0.01
    sp.step("BTC/USDT:USDT")
    feed["last"] = s["sl"] - 0.05
    sp.step("BTC/USDT:USDT")
    assert sp.trades[-1]["why"] == "stop" and sp.trades[-1]["net"] < 0
    st = sp.status()
    assert st["trades"] == 2 and st["wins"] == 1 and st["net"] == pytest.approx(sum(x["net"] for x in sp.trades), abs=1e-4)


def test_speed_start_stop_and_wind_down():
    """Ein/Aus: Aus schliesst nichts (Position laeuft mit Stop/Ziel weiter), "jetzt schliessen" schliesst alles."""
    from bot.speed import PaperBroker, SpeedTrader

    feed = {"last": 100.0}
    df = _speed_df(120, 0.0008)
    book = {"bids": [[99.99, 50]] * 20, "asks": [[100.01, 10]] * 20}

    def make():
        sp = SpeedTrader(lambda s: (df, book, {"last": feed["last"], "bid": feed["last"] - 0.01,
                                               "ask": feed["last"] + 0.01}), {"loop_s": 0.05})
        return sp, PaperBroker(lambda s: sp.market[s], sp.p)

    sp, broker = make()
    with pytest.raises(ValueError, match="Bitget-Minimum"):
        sp.start(["BTC/USDT:USDT"], broker, 35.0, "Simulation", margin_usdt=0.2, leverage=10)
    with pytest.raises(ValueError, match="mehr als verfuegbar"):
        sp.start(["BTC/USDT:USDT"], broker, 3.0, "Simulation", margin_usdt=5, leverage=10)
    assert "bis du Aus" in sp.start(["BTC/USDT:USDT"], broker, 35.0, "Simulation", margin_usdt=5, leverage=10)
    assert not sp.busy("BTC/USDT:USDT")                  # Simulation sperrt nichts auf dem Konto
    with pytest.raises(RuntimeError, match="laeuft bereits"):
        sp.start(["ETH/USDT:USDT"], broker, 35.0, "Simulation")
    time.sleep(0.3)
    feed["last"] = 99.9                                  # Einstieg ausgefuehrt
    time.sleep(0.3)
    assert sp.slots["BTC/USDT:USDT"]["state"] == "open"
    assert "laufen mit Stop und Ziel weiter" in sp.stop()  # Aus: nichts schliessen
    time.sleep(0.3)
    st = sp.status()
    assert st["active"] and st["stopping"] and st["trades"] == 0          # Position laeuft weiter
    assert sp.slots["BTC/USDT:USDT"]["state"] == "open"
    feed["last"] = sp.slots["BTC/USDT:USDT"]["tp"] + 0.05               # Ziel erreicht -> Sitzung endet
    sp.thread.join(5)
    st = sp.status()
    assert not st["active"] and st["trades"] == 1 and sp.trades[0]["why"] == "ziel"
    # "Aus + jetzt schliessen"
    feed["last"] = 100.0
    sp, broker = make()
    sp.start(["BTC/USDT:USDT"], broker, 35.0, "Simulation", margin_usdt=5, leverage=10)
    time.sleep(0.3)
    feed["last"] = 99.9
    time.sleep(0.3)
    sp.stop(close=True)
    sp.thread.join(5)
    assert not sp.active and sp.trades[-1]["why"] == "von Hand geschlossen"
    assert sp.slots["BTC/USDT:USDT"]["state"] == "idle"

def test_speed_bitget_broker_orders():
    from bot.speed import DEFAULTS, BitgetBroker

    class FS(FakeBitget):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.orders_ = {}

        def create_order(self, sym, typ, side, amount, price=None, params=None):
            oid = f"o{len(self.orders_) + 1}"
            self.orders_[oid] = {"status": "open", "filled": 0, "price": price, "average": None}
            self.calls.append(("create_order", sym, typ, side, amount, price, params))
            return {"id": oid}

        def fetch_order(self, oid, sym):
            return self.orders_[oid]

        def cancel_order(self, oid, sym, params=None):
            if self._plan_cancel(oid, params):
                return
            self.orders_[oid]["status"] = "canceled"

    c = FS()
    b = BitgetBroker(c, DEFAULTS, "isolated")
    oid = b.place_entry("BTC/USDT:USDT", 1, 0.001, 60000.0)
    entry = c.calls[-1]
    assert entry[2] == "limit" and entry[3] == "buy" and entry[6] == {"postOnly": True, "marginMode": "isolated"}
    assert b.entry_status("BTC/USDT:USDT", oid)[0] == "open"
    c.orders_[oid].update(status="closed", filled=0.001, average=60000.0)
    assert b.entry_status("BTC/USDT:USDT", oid) == ("filled", 60000.0, 0.001)
    prot = b.protect("BTC/USDT:USDT", 1, 0.001, 59900.0, 60200.0)
    sl = c.plans[prot["sl_id"]]
    assert sl["planType"] == "loss_plan" and sl["holdSide"] == "buy" and sl["size"] == "0.001"   # an die Position gebunden
    tp = c.calls[-1]
    assert tp[2] == "limit" and tp[3] == "sell" and tp[6] == {"reduceOnly": True, "marginMode": "isolated"}
    assert b.exit_status("BTC/USDT:USDT", 1, 0.001, prot) is None     # Position (0,004) noch da, Ziel offen
    c.orders_[prot["tp_id"]].update(status="closed", average=60200.0)
    assert b.exit_status("BTC/USDT:USDT", 1, 0.001, prot) == ("ziel", 60200.0, True)
    assert prot["sl_id"] not in c.plans                               # Stop nach dem Ziel storniert


def test_boost_learns_interactions_linear_cannot():
    from bot.forecast import Boost, _logit_fit, _logit_p

    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)          # Wechselwirkung (XOR)
    flip = rng.random(6000) < 0.15
    y[flip] = 1 - y[flip]
    w = np.ones(6000)
    tr, te = slice(0, 4500), slice(4500, None)
    lin = (_logit_p(_logit_fit(X[tr], y[tr], w[tr]), X[te]) >= 0.5) == (y[te] == 1)
    bst = (Boost().fit(X[tr], y[tr], w[tr]).proba(X[te]) >= 0.5) == (y[te] == 1)
    assert lin.mean() < 0.6 and bst.mean() > 0.75


def test_forecaster_learns_backfills_and_records_book(tmp_path):
    from bot.forecast import CandleMemory, Forecaster

    full = _fc_frame(4000, 0.6, 21)
    state = {"n": 3000}

    def candles(sym, tf, limit):
        return full.iloc[state["n"] - min(limit, 1000):state["n"]].reset_index(drop=True)

    def ohlcv(sym, tf, since, limit):                     # Bitget-Vergangenheit seitenweise
        part = full[(full["ts"] >= since) & (full["ts"] < full["ts"].iloc[state["n"] - 1000])].head(limit)
        return part.values.tolist()

    book = {"bids": [[99.95, 5.0], [99.9, 5.0]], "asks": [[100.05, 1.0], [100.1, 1.0]]}
    fx = Forecaster(candles, tmp_path, "paper", ohlcv_fn=ohlcv, book_fn=lambda s: book,
                    background=False)
    fx.memory.backfill = lambda sym, fn, days=70, ref_ms=None, _b=fx.memory.backfill: _b(sym, fn, 11, ref_ms)
    out = fx.get("BTC/USDT:USDT")
    assert out["ok"] and out["learning"]["bars"] > 2900               # 1000 frisch + Vergangenheit nachgeladen
    assert out["learning"]["retrains"] == 1 and out["learning"]["since"]
    assert out["model"]["kind"] in ("linear", "Baeume", "linear + Baeume")
    state["n"] += 1                                                     # neue Kerze -> Orderbuch der alten gespeichert
    fx.fresh.clear()
    out2 = fx.get("BTC/USDT:USDT")
    assert out2["learning"]["retrains"] == 2 and out2["learning"]["book_bars"] == 1
    mem = CandleMemory(tmp_path / "ki")                                 # Neustart: Gedaechtnis + Orderbuch bleiben
    df = mem.merge("BTC/USDT:USDT", full.iloc[state["n"] - 1000:state["n"] + 1].reset_index(drop=True))
    assert df["book_imb"].notna().sum() == 1 and len(df) > 2900
    assert mem.meta["BTC/USDT:USDT"]["retrains"] == 2


def test_ki_autopilot_manages_position():
    """KI-Autopilot: Einstieg nach KI, Stop auf Einstand, Teilverkauf, Nachziehen, Ausstieg wenn die KI dreht."""
    from bot.speed import PaperBroker, SpeedTrader

    feed = {"last": 100.0}
    fc = {"ok": True, "p_up": 0.62, "p_up_raw": 0.62, "band_pct": 0.5, "decision": "LONG"}
    df = _speed_df(120, 0.0)
    book = {"bids": [[99.99, 10]] * 20, "asks": [[100.01, 10]] * 20}
    ap = SpeedTrader(lambda s: (df, book, {"last": feed["last"], "bid": feed["last"] - 0.01, "ask": feed["last"] + 0.01}),
                     {"loop_s": 0.0}, forecast_fn=lambda s: fc, kind="ki")
    broker = PaperBroker(lambda s: ap.market[s], ap.p)
    ap.cur, ap.broker = {**ap.p, "margin_usdt": 5, "leverage": 10}, broker
    ap.session = {"symbols": ["BTC/USDT:USDT"], "end": time.time() + 3600, "equity0": 35.0}
    ap.slots = {"BTC/USDT:USDT": {"state": "idle"}}
    sym = "BTC/USDT:USDT"
    s = ap.slots[sym]
    fc["p_up"] = 0.53                                     # KI zu unsicher -> nichts
    ap.step(sym)
    assert s["state"] == "idle"
    fc["p_up"] = 0.62
    ap.step(sym)
    assert s["state"] == "pending" and s["side"] == 1
    r = 100.0 - s["sl"]
    assert r == pytest.approx(0.5, rel=0.05) and s["tp"] - 99.99 == pytest.approx(2 * r, rel=0.05)   # Ziel 2R
    feed["last"] = 99.98
    ap.step(sym)                                          # ausgefuehrt
    assert s["state"] == "open" and s["qty"] == pytest.approx(0.5, rel=0.01)
    feed["last"] = s["entry"] + 0.75 * s["r0"]
    ap.step(sym)                                          # +0,75 R -> Stop auf Einstand
    assert s["sl"] > s["entry"] and not s.get("partial_done")
    feed["last"] = s["entry"] + 1.1 * s["r0"]
    ap.step(sym)                                          # +1,1 R -> Haelfte verkauft
    assert s["partial_done"] and s["qty"] == pytest.approx(0.25, rel=0.01)
    assert ap.trades[-1]["why"] == "Teilverkauf" and ap.trades[-1]["net"] > 0
    feed["last"] = s["entry"] + 1.8 * s["r0"]
    ap.step(sym)                                          # Stop zieht nach (1 R hinter dem besten Kurs)
    assert s["sl"] == pytest.approx(s["entry"] + 0.8 * s["r0"], rel=1e-4)
    fc.update(p_up=0.35)                                  # KI dreht klar auf SHORT -> ganz verkaufen
    ap.step(sym)
    assert s["state"] == "idle" and ap.trades[-1]["why"].startswith("KI dreht") and ap.trades[-1]["net"] > 0
    assert sum(t["net"] for t in ap.trades) > 0


def test_session_blocks_only_markets_it_holds_and_waits_for_free_markets():
    """Schnell-Orders nur gesperrt, wo die Sitzung auf dem Konto selbst etwas haelt; die Sitzung wartet,
    wo im Konto schon etwas anderes offen ist."""
    from bot.speed import PaperBroker, SpeedTrader

    df = _speed_df(120, 0.0008)
    book = {"bids": [[99.99, 50]] * 20, "asks": [[100.01, 10]] * 20}
    sp = SpeedTrader(lambda s: (df, book, {"last": 100.0, "bid": 99.99, "ask": 100.01}), {"loop_s": 0.0})
    occupied = {"ETH/USDT:USDT"}
    sp.cur, sp.broker, sp.free_fn = {**sp.p, "margin_usdt": 5, "leverage": 10}, PaperBroker(lambda s: sp.market[s], sp.p), \
        (lambda s: s not in occupied)
    sp.active = True
    sp.session = {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "end": time.time() + 300, "equity0": 35.0,
                  "label": "Testkonto", "stopping": False}
    sp.slots = {s: {"state": "idle"} for s in sp.session["symbols"]}
    assert not sp.busy("BTC/USDT:USDT")                  # nur suchen -> Markt frei fuer Schnell-Orders
    sp.step("BTC/USDT:USDT")
    sp.step("ETH/USDT:USDT")
    assert sp.slots["BTC/USDT:USDT"]["state"] == "pending" and sp.busy("BTC/USDT:USDT")
    assert sp.slots["ETH/USDT:USDT"]["state"] == "idle"   # dort ist schon etwas offen -> warten
    assert "wartet" in sp.slots["ETH/USDT:USDT"]["signal"]["votes"]
    occupied.clear()
    sp.step("ETH/USDT:USDT")
    assert sp.slots["ETH/USDT:USDT"]["state"] == "pending"


def test_ki_autopilot_safety_rules():
    from bot.speed import PaperBroker, SpeedTrader

    fc = {"ok": True, "p_up": 0.5, "p_up_raw": 0.7, "band_pct": 0.5}
    df = _speed_df(120, 0.0)
    book = {"bids": [[99.99, 10]] * 20, "asks": [[100.01, 10]] * 20}
    ap = SpeedTrader(lambda s: (df, book, {"last": 100.0, "bid": 99.99, "ask": 100.01}), {"loop_s": 0.05},
                     forecast_fn=lambda s: fc, kind="ki")
    broker = PaperBroker(lambda s: ap.market[s], ap.p)
    with pytest.raises(ValueError, match="nur in der Simulation"):
        ap.start(["BTC/USDT:USDT"], broker, 35.0, "Testkonto", use_raw=True)
    # unbewaehrte KI (Sicherheit 50 %) handelt nicht - mit Rohsignal (nur Simulation) schon
    ap.cur, ap.broker = {**ap.p, "margin_usdt": 5, "leverage": 10}, broker
    ap.session = {"symbols": ["BTC/USDT:USDT"], "end": time.time() + 3600, "equity0": 35.0}
    ap.slots = {"BTC/USDT:USDT": {"state": "idle"}}
    ap.step("BTC/USDT:USDT")
    assert ap.slots["BTC/USDT:USDT"]["state"] == "idle"
    ap.cur["use_raw"] = True
    ap.step("BTC/USDT:USDT")
    assert ap.slots["BTC/USDT:USDT"]["state"] == "pending"
    assert ap.status()["name"] == "KI-Autopilot"


def test_pattern_detection_is_causal_and_filters():
    """Muster: richtig erkannt, nur aus Vergangenheit (Zukunft aendert nichts), Filter-Logik."""
    from bot.patterns import analyze, candles, pattern_blocks, recent

    # Hammer nach Fall, Bearish Engulfing nach Anstieg
    o = np.array([10, 9.8, 9.6, 9.4, 9.2, 9.0, 8.95]); c = np.array([9.8, 9.6, 9.4, 9.2, 9.0, 8.9, 9.0])
    h = np.maximum(o, c) + 0.01; lo = np.minimum(o, c) - 0.01; lo[-1] = 8.6
    assert candles(o, h, lo, c)["hammer"][-1]
    # Anstieg, dann grosse rote Kerze, die die vorige gruene ganz umschliesst
    o2 = np.array([9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.5])
    c2 = np.array([9.2, 9.4, 9.6, 9.8, 10.0, 10.4, 9.7])
    assert candles(o2, np.maximum(o2, c2) + 0.02, np.minimum(o2, c2) - 0.02, c2)["engulf_dn"][-1]
    # Doppel-Tief mit Bruch der Nackenlinie
    path = [10, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 6.05, 7, 8, 9, 10, 10.6, 10.8, 11]
    close = np.repeat(path, 1).astype(float)
    df = pd.DataFrame({"ts": np.arange(len(close)) * 300_000, "open": np.r_[close[0], close[:-1]], "close": close,
                       "high": np.maximum(close, np.r_[close[0], close[:-1]]) + 0.05,
                       "low": np.minimum(close, np.r_[close[0], close[:-1]]) - 0.05, "volume": 1000.0})
    pa = analyze(df)
    assert pa["pat_dbl_bottom"].iloc[-4:].max() == 1 and pa["pat_dbl_top"].sum() == 0
    # kausal: was bis Kerze i erkannt wurde, aendert sich nicht durch spaetere Kerzen
    big = _fc_frame(1500, 0.3, 5)
    full, part = analyze(big), analyze(big.iloc[:1000])
    pd.testing.assert_frame_equal(full.iloc[:1000].reset_index(drop=True), part.reset_index(drop=True))
    r = recent(big, 300)
    assert "score" in r[0] and all(x["time"] > 0 for x in r[1:])
    from bot.patterns import explain
    ex = explain(big)
    assert len(ex["candles"]) == 10 and ex["verdict"] in ("bullisch", "baerisch", "neutral")
    assert ex["candles"][0]["time"] == int(big["ts"].iloc[-2]) // 1000          # neueste ABGESCHLOSSENE zuerst
    assert all("Kerze" in x["candle"] for x in ex["candles"])
    s = {"pattern_filter": "avoid", "pattern_avoid": 0.5, "pattern_confirm": 0.3}
    assert pattern_blocks("long", -0.6, s) and not pattern_blocks("long", -0.4, s) and not pattern_blocks("short", -0.6, s)
    assert pattern_blocks("long", 0.1, {**s, "pattern_filter": "confirm"}) and not pattern_blocks("long", 0.4, {**s, "pattern_filter": "confirm"})
    assert not pattern_blocks("long", -1.0, {"pattern_filter": False})       # YAML: off = False


def test_forecast_with_all_data_sources():
    """KI lernt auch mit Funding, Makro-Ampel, Leitwaehrungen und Orderbuch-Spalten (wie live)."""
    from bot.forecast import build

    df = _fc_frame(3000, 0.6, 31)
    df["book_imb"] = np.where(np.arange(len(df)) > 1500, 0.2, np.nan)
    df["wall_bid"], df["wall_ask"] = 0.5, 0.8
    funding = pd.DataFrame({"ts": df["ts"].iloc[::96].to_numpy(), "rate": 0.0001})
    days = pd.date_range(pd.to_datetime(df["ts"].iloc[0], unit="ms") - pd.Timedelta(days=5), periods=30, freq="D")
    macro = pd.DataFrame({"crypto": np.linspace(-0.5, 0.5, 30), "gold": 0.1}, index=days)
    macro["avail"] = (macro.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    fc = build(df, {"btc": df, "eth": df}, funding=funding, macro=macro)
    assert fc is not None and fc["decision"] in ("LONG", "SHORT")
    assert {"macro", "funding", "book_imb", "wall_bid", "btc_r1", "eth_r1", "m5_pattern_score"} <= set(fc["feature_names"])


def test_forecast_scenario_moves_like_a_chart():
    from bot.forecast import path

    fc = {"time": 1_700_000_100, "decision": "LONG", "typical_30_pct": 0.4, "sd_5m_pct": 0.15,
          "nodes": [{"sec": 0, "ret": 0.0, "band": 0.0}, {"sec": 1800, "ret": 0.01, "band": 0.6}]}
    p = path(fc, 100.0, fc["time"], seed=7)
    prices = [x["price"] for x in p]
    assert prices[0] == pytest.approx(100.0) and prices[-1] == pytest.approx(100.4, rel=1e-3)   # Richtung LONG
    moves = np.diff(prices)
    assert (moves > 0).any() and (moves < 0).any()                                             # Auf und Ab
    assert path(fc, 100.0, fc["time"], seed=7) == p                                            # stabil
    down = path({**fc, "decision": "SHORT"}, 100.0, fc["time"], seed=7)
    assert down[-1]["price"] == pytest.approx(99.6, rel=1e-3)
