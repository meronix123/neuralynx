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
# (15m/5m sind rausgeflogen: dort fressen die Gebuehren jeden Vorteil auf - siehe fruehere Tests)
TIMEFRAMES = [
    ("1h", "4h", 50, 200),
    ("4h", "1d", 20, 50),
]
MIN_SCORES = [4, 5]
SL_RR = [(1.0, 2.0), (1.5, 2.0), (2.0, 3.0)]   # Trend-Strategie: SL in ATR, Chance/Risiko
STRATEGY_SETS = {
    "nur_trend": ["trend"],
    "nur_seitwaerts": ["range"],
    "nur_ausbruch": ["breakout"],
    "alle_nach_lage": ["trend", "range", "breakout"],
}
ENTRY = ["market", "limit"]
# Positionsfuehrung: (Name, Teilverkauf bei R, Aufstocken bei R)
MANAGE = [
    ("normal", 0, 0),
    ("teilverkauf", 1.0, 0),          # 50 % bei +1R verkaufen, Stop auf Einstand
]
MIN_TRAIN_TRADES = 20


def variants():
    """Signal-relevante Einstellungen (teuer, einmal rechnen) -> guenstige Varianten."""
    for tf_row, score, (sl, rr), sset in itertools.product(TIMEFRAMES, MIN_SCORES, SL_RR, STRATEGY_SETS):
        yield (*tf_row, score, sl, rr, sset), list(itertools.product(ENTRY, MANAGE))


def make_cfg(base: dict, tf, ttf, tf_fast, tf_slow, score, sl, rr, sset,
             entry="market", manage=("normal", 0, 0)) -> dict:
    cfg = copy.deepcopy(base)
    cfg["timeframe"], cfg["trend_timeframe"] = tf, ttf
    st = cfg["strategy"]
    st.update(trend_ema_fast=tf_fast, trend_ema_slow=tf_slow, min_score=score,
              sl_atr=sl, tp_atr=round(sl * rr, 3), breakeven_at_r=1e9, strategies=STRATEGY_SETS[sset])
    cfg["fees"]["entry_order"] = entry
    _, part_r, pyr_r = manage
    st.update(partial_tp_r=part_r, partial_tp_frac=0.5,
              pyramid_at_r=pyr_r, pyramid_max_adds=2 if pyr_r else 0, pyramid_size_frac=0.5)
    return cfg


def optimize(base: dict, data: dict, rules: dict, base_tf: str = "1h") -> pd.DataFrame:
    all_ts = sorted(set().union(*[set(df["ts"]) for df in data.values()]))
    split = all_ts[int(len(all_ts) * 2 / 3)]
    rows = []
    total = sum(len(c) for _, c in variants())
    done, t0 = 0, time.time()
    for key, combos in variants():
        prep = prepare(make_cfg(base, *key), data, base_tf)  # Signale nur einmal je Einstellung
        for entry, manage in combos:
            cfg = make_cfg(base, *key, entry, manage)
            train = simulate(cfg, prep, rules, end_ts=split)
            test = simulate(cfg, prep, rules, start_ts=split)
            per = test.get("per_strategy")
            rows.append({
                "zeit": key[0], "strategien": key[7], "punkte": key[4], "sl_atr": key[5], "rr": key[6],
                "einstieg": entry, "fuehrung": manage[0],
                "train_trades": train["trades"], "train_pf": round(train.get("profit_factor", 0), 2),
                "train_rendite_%": round(train.get("return_pct", 0), 1),
                "test_trades": test["trades"], "test_pf": round(test.get("profit_factor", 0), 2),
                "test_rendite_%": round(test.get("return_pct", 0), 1),
                "test_trefferquote_%": round(test.get("winrate_pct", 0), 1),
                "test_max_rueckgang_%": round(test.get("max_drawdown_pct", 0), 1),
                "test_trades_pro_tag": round(test.get("trades_per_day", 0), 2),
                "test_je_strategie": "; ".join(f"{k}: {r.trades} Tr. PF {r.profit_faktor}"
                                               for k, r in per.iterrows()) if per is not None else "",
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
        print(f"Lade {days} Tage {sym} (1h) ...")
        data[sym] = fetch_history(client, sym, "1h", days)
        m = client.market(sym)
        rules[sym] = (float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0))

    res = optimize(base, data, rules, base_tf="1h")
    DATA.mkdir(exist_ok=True)
    out = DATA / "optimierung.csv"
    res.to_csv(out, index=False, sep=";", decimal=",")

    ok = res[res.train_trades >= MIN_TRAIN_TRADES].sort_values("train_pf", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = [c for c in ok.columns if c != "test_je_strategie"]
    print("\n===== TOP 15 (sortiert nach Training, entscheidend ist TEST) =====")
    print(ok.head(15)[show].to_string(index=False))
    robust = ok[(ok.train_pf > 1.2) & (ok.test_pf > 1.2) & (ok.test_trades >= 10)]
    print(f"\nVarianten, die in Training UND Test profitabel waren (PF > 1,2): {len(robust)} von {len(res)}")
    if len(robust):
        best = robust.sort_values("test_pf", ascending=False).head(10)
        print(best[show].to_string(index=False))
        print("\nAufteilung der besten Variante im TEST nach Strategie:")
        print("  " + best.iloc[0]["test_je_strategie"])
    print(f"\nAlle Ergebnisse: {out}  (mit Excel oeffnen)")
    print("Hinweis: Auch eine gute Variante kann kuenftig verlieren. Erst Paper/Demo, dann Echtgeld.")
