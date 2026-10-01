"""Strategie-Tester: viele Einstellungen durchrechnen und ehrlich pruefen.

Walk-forward: Die Varianten werden nach ihrem Ergebnis im TRAININGS-Zeitraum
(erste 2/3) sortiert. Entscheidend ist aber die Spalte TEST (letztes 1/3):
Diese Daten hat die Auswahl nie gesehen - so waere es live gelaufen.
"""
import copy
import itertools
import time

import pandas as pd

from .backtest import (DATA, fetch_history, load_funding, load_macro, prepare, prepare_multi, simulate, streams,
                       usable)
from .exchange import make_client, resolve_symbols
from .strategy import TF_MS
from .tfselect import tf_cfg

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
    "filter+zeitebenen+funding": dict(leader_filter=True, health_window=10, max_hold_bars=12, ml_filter=False,
                                      mtf_filter=True, funding_filter=True),
    "filter+zeitebenen+makro": dict(leader_filter=True, health_window=10, max_hold_bars=12, ml_filter=False,
                                    mtf_filter=True, macro_filter=True),
    "alles": dict(leader_filter=True, health_window=10, max_hold_bars=12, ml_filter=False,
                  mtf_filter=True, funding_filter=True, macro_filter=True),
}
for _k in FILTERS:
    FILTERS[_k].setdefault("funding_filter", False)
    FILTERS[_k].setdefault("macro_filter", False)

# Schnell-Modus: kurze Einstiegs-Zeiteinheiten, hoehere nur zur Orientierung, Gebuehren so niedrig wie moeglich
FAST = dict(
    TIMEFRAMES=[("5m", "1h", 50, 200), ("15m", "1h", 50, 200), ("30m", "4h", 50, 200)],
    STRATEGY_SETS={k: STRATEGY_SETS[k] for k in ("nur_trend", "nur_ausbruch", "trend+ausbruch", "alle_nach_lage")},
    ENTRY=["limit"],
    TP_ORDERS=["market", "limit"],
    FILTERS={k: FILTERS[k] for k in ("filter", "filter+zeitebenen", "filter+zeitebenen+funding")},
)
MIN_TRAIN_TRADES = 20


def variants():
    """Signal-relevante Einstellungen (teuer, einmal rechnen) -> guenstige Varianten."""
    for tf_row, score, (sl, rr), sset in itertools.product(TIMEFRAMES, MIN_SCORES, SL_RR, STRATEGY_SETS):
        yield (*tf_row, score, sl, rr, sset), list(itertools.product(ENTRY, TP_ORDERS, MANAGE, FILTERS))


def make_cfg(base: dict, tf, ttf, tf_fast, tf_slow, score, sl, rr, sset,
             entry="market", tp_order="market", manage=("normal", 0, 0), filt=None) -> dict:
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
    if filt is not None:  # beim Vorbereiten der Signale spielen die Filter noch keine Rolle
        st.update(FILTERS[filt])
    return cfg


def optimize(base: dict, data: dict, rules: dict, base_tf: str = "1h",
             funding: dict | None = None, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    all_ts = sorted(set().union(*[set(df["ts"]) for df in data.values()]))
    split = all_ts[int(len(all_ts) * 2 / 3)]
    rows = []
    total = sum(len(c) for _, c in variants())
    done, t0 = 0, time.time()
    for key, combos in variants():
        prep = prepare(make_cfg(base, *key), data, base_tf, funding, macro)  # Signale nur einmal je Einstellung
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
    funding = load_funding(client, list(data), days)
    macro = load_macro(days)
    res = optimize(base, data, rules, base_tf=base_tf, funding=funding, macro=macro)
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


# ---------------------------------------------------------------------------
# Zeiteinheiten-Vergleich: feste Zeiteinheit gegen automatische Wahl
ZEIT_TFS = ["15m", "30m", "1h", "2h", "4h"]
# Name -> (Zeiteinheiten, tf_select, Fenster, Mindest-Profit-Faktor)
ZEIT_MODES = {
    **{f"fest_{tf}": ([tf], "fixed", 20, 1.0) for tf in ZEIT_TFS},
    "alle_gleichzeitig": (ZEIT_TFS, "all", 20, 1.0),
    "auto": (ZEIT_TFS, "adaptive", 20, 1.0),
    "auto_streng": (ZEIT_TFS, "adaptive", 20, 1.2),
    "auto_lang": (ZEIT_TFS, "adaptive", 40, 1.0),
    "auto_ab_1h": (["1h", "2h", "4h"], "adaptive", 20, 1.0),
}
ZEIT_SETS = ["trend+ausbruch", "nur_trend"]
ZEIT_FILTERS = ["filter+zeitebenen", "alles"]


def compare_timeframes(base: dict, data: dict, rules: dict, base_tf: str = "15m",
                       funding: dict | None = None, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    all_ts = sorted(set().union(*[set(df["ts"]) for df in data.values()]))
    split = all_ts[int(len(all_ts) * 2 / 3)]
    rows, t0 = [], time.time()
    total, done = len(ZEIT_SETS) * len(ZEIT_FILTERS) * len(ZEIT_MODES), 0
    for sset in ZEIT_SETS:
        base_s = copy.deepcopy(base)
        base_s["strategy"]["strategies"] = STRATEGY_SETS[sset]
        base_s["tf_select"], base_s["timeframes"] = "all", ZEIT_TFS
        # Signale je Zeiteinheit nur einmal berechnen
        preps = {tf: prepare(tf_cfg(base_s, tf), data, base_tf, funding, macro) for tf in ZEIT_TFS}
        for filt in ZEIT_FILTERS:
            for name, (tfs, mode, window, min_pf) in ZEIT_MODES.items():
                cfg = copy.deepcopy(base_s)
                cfg["strategy"].update(FILTERS[filt], tf_window=window, tf_min_pf=min_pf)
                cfg["tf_select"], cfg["timeframes"] = mode, tfs
                if mode == "fixed":
                    cfg["timeframe"] = tfs[0]
                prep = streams({tf: preps[tf] for tf in tfs})
                train = simulate(cfg, prep, rules, end_ts=split)
                test = simulate(cfg, prep, rules, start_ts=split)
                per = test.get("per_tf")
                rows.append({
                    "modus": name, "strategien": sset, "filter": filt,
                    "train_trades": train["trades"], "train_pf": round(train.get("profit_factor", 0), 2),
                    "train_rendite_%": round(train.get("return_pct", 0), 1),
                    "test_trades": test["trades"], "test_pf": round(test.get("profit_factor", 0), 2),
                    "test_rendite_%": round(test.get("return_pct", 0), 1),
                    "test_max_rueckgang_%": round(test.get("max_drawdown_pct", 0), 1),
                    "test_trades_pro_tag": round(test.get("trades_per_day", 0), 2),
                    "test_je_zeiteinheit": "; ".join(f"{k}: {int(r.trades)} Tr. PF {r.profit_faktor}"
                                                     for k, r in per.iterrows()) if per is not None else "",
                })
                done += 1
                print(f"\r  Variante {done}/{total}  ({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def compare_cli(base: dict, days: int) -> None:
    base_tf = "15m"
    print("Zeiteinheiten-Vergleich: feste Zeiteinheit gegen automatische Wahl (15m bis 4h).")
    client = make_client()
    symbols = resolve_symbols(client, base["symbols"])
    data, rules = {}, {}
    for sym in symbols:
        print(f"Lade {days} Tage {sym} ({base_tf}) ...")
        data[sym] = fetch_history(client, sym, base_tf, days)
        m = client.market(sym)
        rules[sym] = (float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0))
    data = usable(data)
    funding = load_funding(client, list(data), days)
    macro = load_macro(days)
    res = compare_timeframes(base, data, rules, base_tf, funding, macro)
    DATA.mkdir(exist_ok=True)
    out = DATA / "zeiteinheiten.csv"
    res.to_csv(out, index=False, sep=";", decimal=",")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = [c for c in res.columns if c != "test_je_zeiteinheit"]
    print("\n===== ERGEBNIS (sortiert nach TEST-Profit-Faktor) =====")
    print(res.sort_values("test_pf", ascending=False)[show].to_string(index=False))
    print("\nIm Schnitt je Modus (TEST):")
    print(res.groupby("modus")[["test_trades", "test_trades_pro_tag", "test_pf", "test_rendite_%"]]
          .median().round(2).sort_values("test_pf", ascending=False).to_string())
    print("\nAuto-Modus - welche Zeiteinheit hat er im TEST gehandelt:")
    for _, r in res[res.modus.str.startswith("auto")].iterrows():
        print(f"  {r['modus']:<12} {r['strategien']:<15} {r['filter']:<18} -> {r['test_je_zeiteinheit']}")
    print(f"\nAlle Ergebnisse: {out}  (mit Excel oeffnen)")


# ---------------------------------------------------------------------------
# Order-Block-Test: aktuelle Einstellungen (config.yaml) mit und ohne Order-Block-Filter
OB_DISP = [1.0, 1.5, 2.0]          # wie kraeftig die Bewegung nach der Zone sein muss (x ATR)
OB_MODES = {
    "ohne": dict(ob_filter="off"),
    "meiden_60%": dict(ob_filter="avoid", ob_room=0.6),
    "meiden_100%": dict(ob_filter="avoid", ob_room=1.0),
    "nur_nach_antest": dict(ob_filter="confirm"),
    "beides": dict(ob_filter="both", ob_room=0.6),
}


def compare_orderblocks(base: dict, data: dict, rules: dict, base_tf: str = "15m",
                        funding: dict | None = None, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    all_ts = sorted(set().union(*[set(df["ts"]) for df in data.values()]))
    split = all_ts[int(len(all_ts) * 2 / 3)]
    rows, t0, done = [], time.time(), 0
    total = len(OB_DISP) * (len(OB_MODES) - ("ohne" in OB_MODES)) + ("ohne" in OB_MODES)
    for disp in OB_DISP:
        cfg0 = copy.deepcopy(base)
        cfg0["strategy"]["ob_disp_atr"] = disp
        prep = prepare_multi(cfg0, data, base_tf, funding, macro)
        for name, upd in OB_MODES.items():
            if name == "ohne" and disp != OB_DISP[0]:
                continue  # ohne Filter spielt die Staerke keine Rolle
            cfg = copy.deepcopy(cfg0)
            cfg["strategy"].update(upd)
            train = simulate(cfg, prep, rules, end_ts=split)
            test = simulate(cfg, prep, rules, start_ts=split)
            rows.append({
                "order_blocks": name, "staerke_atr": disp if name != "ohne" else "-",
                "train_trades": train["trades"], "train_pf": round(train.get("profit_factor", 0), 2),
                "train_rendite_%": round(train.get("return_pct", 0), 1),
                "test_trades": test["trades"], "test_pf": round(test.get("profit_factor", 0), 2),
                "test_rendite_%": round(test.get("return_pct", 0), 1),
                "test_max_rueckgang_%": round(test.get("max_drawdown_pct", 0), 1),
                "test_trades_pro_tag": round(test.get("trades_per_day", 0), 2),
                "aussortiert_test": test["skipped"].get("orderblock", 0),
            })
            done += 1
            print(f"\r  Variante {done}/{total}  ({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def orderblocks_cli(base: dict, days: int) -> None:
    from .tfselect import active_tfs
    from .strategy import TF_MS
    base_tf = min(active_tfs(base), key=lambda t: TF_MS[t])
    print(f"Order-Block-Test mit den aktuellen Einstellungen (Zeiteinheiten: {', '.join(active_tfs(base))}).")
    client = make_client()
    symbols = resolve_symbols(client, base["symbols"])
    data, rules = {}, {}
    for sym in symbols:
        print(f"Lade {days} Tage {sym} ({base_tf}) ...")
        data[sym] = fetch_history(client, sym, base_tf, days)
        m = client.market(sym)
        rules[sym] = (float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0))
    data = usable(data)
    res = compare_orderblocks(base, data, rules, base_tf, load_funding(client, list(data), days), load_macro(days))
    DATA.mkdir(exist_ok=True)
    out = DATA / "orderblocks.csv"
    res.to_csv(out, index=False, sep=";", decimal=",")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n===== ERGEBNIS (sortiert nach TEST-Profit-Faktor) =====")
    print(res.sort_values("test_pf", ascending=False).to_string(index=False))
    print(f"\nAlle Ergebnisse: {out}")
    print("Hinweis: Die Orderbuch-Waende lassen sich nicht rueckwirkend testen (Bitget speichert kein altes Orderbuch).")


# ---------------------------------------------------------------------------
# Alle Zeiteinheiten: aktuelle Einstellungen, Auto-Wahl ueber verschieden viele Zeiteinheiten
ALLE_MODES = {
    "bisher_15m-4h": ["15m", "30m", "1h", "2h", "4h"],
    "15m-1d": ["15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
    "5m-4h": ["5m", "15m", "30m", "1h", "2h", "4h"],
    "5m-1d_alle": ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
    "1h-1d_langsam": ["1h", "2h", "4h", "6h", "12h", "1d"],
}


def compare_all_timeframes(base: dict, data: dict, rules: dict, base_tf: str = "5m",
                           funding: dict | None = None, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    all_ts = sorted(set().union(*[set(df["ts"]) for df in data.values()]))
    split = all_ts[int(len(all_ts) * 2 / 3)]
    cfg0 = copy.deepcopy(base)
    cfg0["tf_select"] = "adaptive"
    tfs = sorted({t for v in ALLE_MODES.values() for t in v}, key=lambda t: TF_MS[t])
    cfg0["timeframes"] = tfs
    preps = {}
    for tf in tfs:
        print(f"  Signale {tf} ...", flush=True)
        preps[tf] = prepare(tf_cfg(cfg0, tf), data, base_tf, funding, macro)
    rows = []
    for name, sel in ALLE_MODES.items():
        cfg = copy.deepcopy(cfg0)
        cfg["timeframes"] = sel
        prep = streams({tf: preps[tf] for tf in sel})
        train = simulate(cfg, prep, rules, end_ts=split)
        test = simulate(cfg, prep, rules, start_ts=split)
        per = test.get("per_tf")
        rows.append({
            "zeiteinheiten": name,
            "train_trades": train["trades"], "train_pf": round(train.get("profit_factor", 0), 2),
            "train_rendite_%": round(train.get("return_pct", 0), 1),
            "test_trades": test["trades"], "test_pf": round(test.get("profit_factor", 0), 2),
            "test_rendite_%": round(test.get("return_pct", 0), 1),
            "test_max_rueckgang_%": round(test.get("max_drawdown_pct", 0), 1),
            "test_trades_pro_tag": round(test.get("trades_per_day", 0), 2),
            "test_je_zeiteinheit": "; ".join(f"{k}: {int(r.trades)} Tr. PF {r.profit_faktor}"
                                             for k, r in per.iterrows()) if per is not None else "",
        })
        print(f"  {name}: fertig", flush=True)
    return pd.DataFrame(rows)


def all_timeframes_cli(base: dict, days: int) -> None:
    base_tf = "5m"
    print(f"Test: Auto-Wahl ueber verschieden viele Zeiteinheiten (5 Minuten bis 1 Tag), {days} Tage.")
    client = make_client()
    symbols = resolve_symbols(client, base["symbols"])
    data, rules = {}, {}
    for sym in symbols:
        print(f"Lade {days} Tage {sym} ({base_tf}) ...")
        data[sym] = fetch_history(client, sym, base_tf, days)
        m = client.market(sym)
        rules[sym] = (float(m["precision"]["amount"] or 0), float(m["limits"]["amount"]["min"] or 0))
    data = usable(data)
    res = compare_all_timeframes(base, data, rules, base_tf, load_funding(client, list(data), days),
                                 load_macro(days))
    DATA.mkdir(exist_ok=True)
    out = DATA / "alle_zeiteinheiten.csv"
    res.to_csv(out, index=False, sep=";", decimal=",")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = [c for c in res.columns if c != "test_je_zeiteinheit"]
    print("\n===== ERGEBNIS =====")
    print(res[show].to_string(index=False))
    print("\nIm TEST gehandelt je Zeiteinheit:")
    for _, r in res.iterrows():
        print(f"  {r['zeiteinheiten']:<15} -> {r['test_je_zeiteinheit']}")
    print(f"\nAlle Ergebnisse: {out}")
