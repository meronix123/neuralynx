"""Netz-Simulation: synthetische 1-Minuten-Kurse mit Trend-/Seitwaerts-Phasen, KI-Stellvertreter mit
einstellbarer Trefferquote; vergleicht Netz-Einstellungen und den normalen KI-Autopilot nach Gebuehren."""
import sys, time as _time, itertools, numpy as np, pandas as pd
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import bot.speed as speed
from bot.speed import SpeedTrader, PaperBroker

class FakeTime:
    now = 1_800_000_000.0
    def time(self): return FakeTime.now
    def __getattr__(self, k): return getattr(_time, k)
speed.time = FakeTime()

def series(n, seed, regime):
    rng = np.random.default_rng(seed)
    vol = np.empty(n); v = 0.0004
    for i in range(n):                                   # Vol-Clustering
        v = 0.9 * v + 0.1 * (0.0004 + 0.0008 * rng.random())
        vol[i] = v
    drift = np.zeros(n)
    if regime == "trend":                                # abwechselnd 300-Bar-Trends
        for k in range(0, n, 300):
            drift[k:k + 300] = rng.choice([-1, 1]) * 0.00025
    elif regime == "mix":
        for k in range(0, n, 300):
            drift[k:k + 300] = rng.choice([-1, 0, 0, 1]) * 0.00025
    r = drift + rng.normal(0, 1, n) * vol
    close = 100 * np.exp(np.cumsum(r))
    ts = 1_700_000_000_000 + np.arange(n) * 60_000
    return pd.DataFrame({"ts": ts, "open": np.r_[close[0], close[:-1]], "high": close * (1 + vol * 0.8),
                         "low": close * (1 - vol * 0.8), "close": close, "volume": np.full(n, 1000.0)})

def run(df, hit, cfg, seed, horizon=5):
    rng = np.random.default_rng(seed + 99)
    n = len(df); close = df["close"].to_numpy()
    fut = np.sign(np.r_[close[horizon:] - close[:-horizon], np.zeros(horizon)])
    # KI-Stellvertreter: kennt die Richtung in `hit` der Faelle, sonst zufaellig; nur jede 5. Minute neu
    know = rng.random(n) < (2 * hit - 1)
    guess = np.where(know, fut, rng.choice([-1, 1], n))
    state = {"i": 0}
    def data_fn(sym):
        i = state["i"]; w = df.iloc[max(0, i - 120):i + 1]
        last = close[i]
        return w, {"bids": [[last * 0.9999, 10]] * 20, "asks": [[last * 1.0001, 10]] * 20}, \
               {"last": last, "bid": last * 0.9999, "ask": last * 1.0001}
    def fc_fn(sym):
        i = state["i"]; g = guess[(i // 10) * 10]
        sd = float(np.std(np.diff(np.log(close[max(0, i - 60):i + 1]))) * 100) or 0.05
        p = 0.62 if g > 0 else 0.38
        return {"ok": True, "p_up": p, "p_up_raw": p, "band_pct": sd * 2.5, "sd_5m_pct": sd * 2.2,
                "decision": "LONG" if g > 0 else "SHORT", "decision_min": horizon, "live": {"n": 300, "hit": 0.58}}
    ap = SpeedTrader(data_fn, {"loop_s": 0.0}, forecast_fn=fc_fn, kind="ki")
    ap.broker = PaperBroker(lambda s: ap.market[s], ap.p)
    ap.cur = {**ap.p, "leverage": 50, "chase_avoid_marks": False, "entry_pullback_r": 0, "cost_guard": False,
              "min_conf": 0.56, "size_mode": "usdt", "margin_usdt": 0.1, "max_open_risk_pct": 100, "learn_n": 10**9, "ki_err_n": 10**9, **cfg}
    if cfg.get("turbo"):
        ap.cur.update(speed.TURBO_AUTO)
    ap.active, ap.session = True, {"symbols": ["BTC/USDT:USDT"], "end": 9e12, "equity0": 87.0, "label": "Simulation", "stopping": False}
    ap.slots = {"BTC/USDT:USDT": {"state": "idle"}}
    curve = []
    for i in range(130, n):
        state["i"] = i; FakeTime.now += 60
        ap.step("BTC/USDT:USDT")
        curve.append(ap._net())
    sl = ap.slots["BTC/USDT:USDT"]
    if sl.get("state") == "grid": ap._grid_close("BTC/USDT:USDT", sl, "Ende")
    elif sl.get("state") == "open": ap._book("BTC/USDT:USDT", sl, close[-1], "Ende", maker=False)
    c = np.array(curve + [ap._net()]); dd = float((np.maximum.accumulate(c) - c).max())
    return ap._net(), len(ap.trades), dd, sum(t["fees"] for t in ap.trades)

if __name__ == "__main__":
    hit = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
    n = 6000
    base_grid = {"grid": True, "grid_unit_margin": 0.1, "grid_budget": 1.0, "turbo": True, "fast": True}
    variants = {"normal (KI-Autopilot, Turbo)": {"turbo": True, "fast": True}}
    for step, take, loss, flip in itertools.product((0.15, 0.3), (0.3, 0.6), (25, 50), (0.53,)):
        variants[f"Netz step {step} take {take} stop {loss} flip {flip}"] = {**base_grid, "grid_step_pct": step, "grid_take_pct": take,
                                                                             "grid_max_loss_pct": loss, "grid_flip_conf": flip}
    print(f"KI-Trefferquote {hit:.0%}, {n} Minuten je Lauf, 3 Regime x 2 Seeds; Netto in USDT (Summe), Trades, max. Drawdown, Gebuehren")
    rows = []
    for name, cfg in variants.items():
        tot = np.zeros(4)
        for regime in ("trend", "range", "mix"):
            for seed in (1, 2):
                tot += np.array(run(series(n, seed, regime), hit, cfg, seed))
        rows.append((tot[0], name, tot))
    for net, name, t in sorted(rows, reverse=True):
        print(f"{net:+8.3f}  {name:45s} Trades {int(t[1]):4d}  DD {t[2]:6.3f}  Gebuehren {t[3]:6.3f}")
