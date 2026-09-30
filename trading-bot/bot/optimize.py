"""Strategie-Tester: viele Einstellungen durchrechnen und ehrlich pruefen.

Walk-forward: Die Varianten werden nach ihrem Ergebnis im TRAININGS-Zeitraum
(erste 2/3) sortiert. Entscheidend ist aber die Spalte TEST (letztes 1/3):
Diese Daten hat die Auswahl nie gesehen - so waere es live gelaufen.
"""
import copy
import itertools
import time

import pandas as pd

from .backtest import DATA, fetch_history, prepare, simulate
from .exchange import make_client, resolve_symbols

# Einstiegs-Zeiteinheit, Trend-Zeiteinheit, Trend-EMAs (schnell, langsam)
TIMEFRAMES = [
    ("15m", "1h", 50, 200),
    ("1h", "4h", 50, 200),
    ("4h", "1d", 20, 50),
]
MIN_SCORES = [3, 4, 5]
SL_ATR = [1.0, 1.5, 2.0]
RR = [1.5, 2.0, 3.0]            # Take-Profit = SL-Abstand x RR
TRAIL = [True, False]
ENTRY = ["market", "limit"]
# Positionsfuehrung: (Name, Teilverkauf bei R, Aufstocken bei R)
MANAGE = [
    ("normal", 0, 0),
    ("teilverkauf", 1.0, 0),          # 50 % bei +1R verkaufen, Stop auf Einstand
    ("aufstocken", 0, 1.0),           # bei +1R und +2R je 50 % nachlegen (nur im Gewinn)
    ("beides", 1.0, 1.0),
]
MIN_TRAIN_TRADES = 20


def variants():
    for (tf, ttf, tf_fast, tf_slow), score in itertools.product(TIMEFRAMES, MIN_SCORES):
        combos = itertools.product(SL_ATR, RR, TRAIL, ENTRY, MANAGE)
        yield (tf, ttf, tf_fast, tf_slow, score), list(combos)


def make_cfg(base: dict, tf, ttf, tf_fast, tf_slow, score, sl=None, rr=None, trail=None, entry=None,
             manage=("normal", 0, 0)) -> dict:
    cfg = copy.deepcopy(base)
    cfg["timeframe"], cfg["trend_timeframe"] = tf, ttf
    st = cfg["strategy"]
    st.update(trend_ema_fast=tf_fast, trend_ema_slow=tf_slow, min_score=score)
    if sl is not None:
        st.update(sl_atr=sl, tp_atr=round(sl * rr, 3), breakeven_at_r=1.0 if trail else 1e9)
        cfg["fees"]["entry_order"] = entry
        _, part_r, pyr_r = manage
        st.update(partial_tp_r=part_r, partial_tp_frac=0.5,
                  pyramid_at_r=pyr_r, pyramid_max_adds=2 if pyr_r else 0, pyramid_size_frac=0.5)
    return cfg


def optimize(base: dict, data: dict, rules: dict, base_tf: str = "15m") -> pd.DataFrame:
    all_ts = sorted(set().union(*[set(df["ts"]) for df in data.values()]))
    split = all_ts[int(len(all_ts) * 2 / 3)]
    rows = []
    total = sum(len(c) for _, c in variants())
    done, t0 = 0, time.time()
    for key, combos in variants():
        prep = prepare(make_cfg(base, *key), data, base_tf)  # Signale nur einmal je Zeiteinheit/Punktzahl
        for sl, rr, trail, entry, manage in combos:
            cfg = make_cfg(base, *key, sl, rr, trail, entry, manage)
            train = simulate(cfg, prep, rules, end_ts=split)
            test = simulate(cfg, prep, rules, start_ts=split)
            rows.append({
                "zeit": key[0], "punkte": key[4], "sl_atr": sl, "rr": rr,
                "trailing": "ja" if trail else "nein", "einstieg": entry, "fuehrung": manage[0],
                "train_trades": train["trades"], "train_pf": round(train.get("profit_factor", 0), 2),
                "train_rendite_%": round(train.get("return_pct", 0), 1),
                "test_trades": test["trades"], "test_pf": round(test.get("profit_factor", 0), 2),
                "test_rendite_%": round(test.get("return_pct", 0), 1),
                "test_trefferquote_%": round(test.get("winrate_pct", 0), 1),
                "test_max_rueckgang_%": round(test.get("max_drawdown_pct", 0), 1),
                "test_trades_pro_tag": round(test.get("trades_per_day", 0), 2),
            })
            done += 1
            print(f"\r  Variante {done}/{total}  ({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def optimize_cli(base: dict, days: int) -> None:
    client = make_client()
    symbols = resolve_symbols(client, base["symbols"])
    data, rules = {}, {}
    for sym in symbols:
        print(f"Lade {days} Tage {sym} (15m) ...")
        data[sym] = fetch_history(client, sym, "15m", days)
        m = client.market(sym)
        rules[sym] = (float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0))

    res = optimize(base, data, rules)
    DATA.mkdir(exist_ok=True)
    out = DATA / "optimierung.csv"
    res.to_csv(out, index=False, sep=";", decimal=",")

    ok = res[res.train_trades >= MIN_TRAIN_TRADES].sort_values("train_pf", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n===== TOP 15 (sortiert nach Training, entscheidend ist TEST) =====")
    print(ok.head(15).to_string(index=False))
    robust = ok[(ok.train_pf > 1.2) & (ok.test_pf > 1.2) & (ok.test_trades >= 10)]
    print(f"\nVarianten, die in Training UND Test profitabel waren (PF > 1,2): {len(robust)} von {len(res)}")
    if len(robust):
        print(robust.sort_values("test_pf", ascending=False).head(10).to_string(index=False))
    print(f"\nAlle Ergebnisse: {out}  (mit Excel oeffnen)")
    print("Hinweis: Auch eine gute Variante kann kuenftig verlieren. Erst Paper/Demo, dann Echtgeld.")
