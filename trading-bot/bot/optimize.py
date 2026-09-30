"""Strategie-Tester: viele Einstellungen durchrechnen und ehrlich pruefen.

Walk-forward: Die Varianten werden nach ihrem Ergebnis im TRAININGS-Zeitraum
(erste 2/3) sortiert. Entscheidend ist aber die Spalte TEST (letztes 1/3):
Diese Daten hat die Auswahl nie gesehen - so waere es live gelaufen.
"""
import copy
import itertools
import time

import pandas as pd

from .backtest import DATA, fetch_history, prepare, simulate, usable
from .exchange import make_client, resolve_symbols

# Einstiegs-Zeiteinheit, Trend-Zeiteinheit, Trend-EMAs (schnell, langsam)
TIMEFRAMES = [
    ("1h", "4h", 50, 200),
    ("2h", "1d", 20, 50),
    ("4h", "1d", 20, 50),
]
MIN_SCORES = [4, 5]
SL_RR = [(1.0, 2.0), (1.5, 2.0), (2.0, 3.0)]   # Trend-Strategie: SL in ATR, Chance/Risiko
STRATEGY_SETS = {
    "nur_trend": ["trend"],
    "nur_seitwaerts": ["range"],
    "nur_ausbruch": ["breakout"],
    "trend+ausbruch": ["trend", "breakout"],
    "alle_nach_lage": ["trend", "range", "breakout"],
}
ENTRY = ["market", "limit"]
TP_ORDERS = ["market"]          # Take-Profit als Markt-Order (Taker) oder liegende Limit-Order (Maker)
# Positionsfuehrung: (Name, Teilverkauf bei R, Aufstocken bei R)
MANAGE = [
    ("normal", 0, 0),
    ("teilverkauf", 1.0, 0),          # 50 % bei +1R verkaufen, Stop auf Einstand
]
# Zusatz-Filter: BTC-Leitwaehrung, Strategie-Gesundheit, Zeit-Stop, selbstlernender ML-Filter
FILTERS = {
    "ohne": dict(leader_filter=False, health_window=0, max_hold_bars=0, ml_filter=False, mtf_filter=False),
    "filter": dict(leader_filter=True, health_window=10, max_hold_bars=12, ml_filter=False, mtf_filter=False),
    "filter+zeitebenen": dict(leader_filter=True, health_window=10, max_hold_bars=12, ml_filter=False,
                              mtf_filter=True),
    "filter+ml": dict(leader_filter=True, health_window=10, max_hold_bars=12, ml_filter=True, mtf_filter=False),
}

# Schnell-Modus: kurze Einstiegs-Zeiteinheiten, hoehere nur zur Orientierung, Gebuehren so niedrig wie moeglich
FAST = dict(
    TIMEFRAMES=[("5m", "1h", 50, 200), ("15m", "1h", 50, 200), ("30m", "4h", 50, 200)],
    STRATEGY_SETS={k: STRATEGY_SETS[k] for k in ("nur_trend", "nur_ausbruch", "trend+ausbruch", "alle_nach_lage")},
    ENTRY=["limit"],
    TP_ORDERS=["market", "limit"],
    FILTERS={k: FILTERS[k] for k in ("filter", "filter+zeitebenen")},
)
MIN_TRAIN_TRADES = 20


def variants():
    """Signal-relevante Einstellungen (teuer, einmal rechnen) -> guenstige Varianten."""
    for tf_row, score, (sl, rr), sset in itertools.product(TIMEFRAMES, MIN_SCORES, SL_RR, STRATEGY_SETS):
        yield (*tf_row, score, sl, rr, sset), list(itertools.product(ENTRY, TP_ORDERS, MANAGE, FILTERS))


def make_cfg(base: dict, tf, ttf, tf_fast, tf_slow, score, sl, rr, sset,
             entry="market", tp_order="market", manage=("normal", 0, 0), filt="ohne") -> dict:
    cfg = copy.deepcopy(base)
    cfg["timeframe"], cfg["trend_timeframe"] = tf, ttf
    st = cfg["strategy"]
    st.update(trend_ema_fast=tf_fast, trend_ema_slow=tf_slow, min_score=score,
              sl_atr=sl, tp_atr=round(sl * rr, 3), breakeven_at_r=1e9, strategies=STRATEGY_SETS[sset])
    cfg["fees"]["entry_order"] = entry
    cfg["fees"]["tp_order"] = tp_order
    _, part_r, pyr_r = manage
    st.update(partial_tp_r=part_r, partial_tp_frac=0.5,
              pyramid_at_r=pyr_r, pyramid_max_adds=2 if pyr_r else 0, pyramid_size_frac=0.5)
    st.update(FILTERS[filt])
    return cfg


def optimize(base: dict, data: dict, rules: dict, base_tf: str = "1h") -> pd.DataFrame:
    all_ts = sorted(set().union(*[set(df["ts"]) for df in data.values()]))
    split = all_ts[int(len(all_ts) * 2 / 3)]
    rows = []
    total = sum(len(c) for _, c in variants())
    done, t0 = 0, time.time()
    for key, combos in variants():
        prep = prepare(make_cfg(base, *key), data, base_tf)  # Signale nur einmal je Einstellung
        for entry, tp_order, manage, filt in combos:
            cfg = make_cfg(base, *key, entry, tp_order, manage, filt)
            train = simulate(cfg, prep, rules, end_ts=split)
            # ML-Filter startet im Test mit dem Wissen aus dem Training (nur Vergangenheit)
            test = simulate(cfg, prep, rules, start_ts=split, seed_examples=train.get("examples"))
            per = test.get("per_strategy")
            rows.append({
                "zeit": key[0], "strategien": key[7], "punkte": key[4], "sl_atr": key[5], "rr": key[6],
                "einstieg": entry, "tp": tp_order, "fuehrung": manage[0], "filter": filt,
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


def use_fast_profile() -> None:
    globals().update(FAST)


def optimize_cli(base: dict, days: int, fast: bool = False) -> None:
    base_tf = "5m" if fast else "1h"
    if fast:
        use_fast_profile()
        print("Schnell-Modus: Einstieg auf 5m/15m/30m, 1h bis 1 Woche nur zur Orientierung.")
    client = make_client()
    symbols = resolve_symbols(client, base["symbols"])
    data, rules = {}, {}
    for sym in symbols:
        print(f"Lade {days} Tage {sym} ({base_tf}) ...")
        data[sym] = fetch_history(client, sym, base_tf, days)
        m = client.market(sym)
        rules[sym] = (float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0))

    data = usable(data)
    res = optimize(base, data, rules, base_tf=base_tf)
    DATA.mkdir(exist_ok=True)
    out = DATA / "optimierung.csv"
    res.to_csv(out, index=False, sep=";", decimal=",")

    # Varianten mit identischem Ergebnis (Einstellung wirkt dort gar nicht) nur einmal zeigen
    result_cols = ["zeit", "strategien", "einstieg", "tp", "fuehrung", "filter", "train_trades", "train_pf",
                   "test_trades", "test_pf", "test_rendite_%"]
    uniq = res.drop_duplicates(subset=result_cols)
    ok = uniq[uniq.train_trades >= MIN_TRAIN_TRADES].sort_values("train_pf", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = [c for c in ok.columns if c != "test_je_strategie"]
    print(f"\n{len(res)} Varianten, davon {len(uniq)} mit unterschiedlichem Ergebnis.")
    print("\n===== TOP 15 (sortiert nach Training, entscheidend ist TEST) =====")
    print(ok.head(15)[show].to_string(index=False))
    robust = ok[(ok.train_pf > 1.2) & (ok.test_pf > 1.2) & (ok.test_trades >= 10)]
    print(f"\nIn Training UND Test profitabel (PF > 1,2): {len(robust)} von {len(uniq)}")
    if len(robust):
        best = robust.sort_values("test_pf", ascending=False)
        print(best[show].to_string(index=False))
        print("\nAufteilung im TEST nach Strategie (je robuste Variante):")
        for _, r in best.head(8).iterrows():
            print(f"  {r['strategien']:<15} {r['fuehrung']:<12} {r['filter']:<10} -> {r['test_je_strategie']}")
    cols = ["test_trades", "test_pf", "test_rendite_%"]
    print("\nIm Schnitt im TEST je Zeiteinheit:")
    print(uniq.groupby("zeit")[cols].median().round(2).to_string())
    for col, title in (("strategien", "Strategie-Kombination"), ("filter", "Zusatz-Filter"), ("tp", "Take-Profit-Art")):
        if uniq[col].nunique() > 1:
            print(f"\nIm Schnitt im TEST je {title} und Zeiteinheit:")
            print(uniq.groupby(["zeit", col])[cols].median().round(2).to_string())
    print(f"\nAlle Ergebnisse: {out}  (mit Excel oeffnen)")
    print("Hinweis: Auch eine gute Variante kann kuenftig verlieren. Erst Paper/Demo, dann Echtgeld.")
