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
