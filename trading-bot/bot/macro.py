"""Makro-Ampel: Finanzmarkt-Daten, die kaum ein Retail-Bot beachtet.

Quellen (kostenlos, ohne Anmeldung, rueckwirkend verfuegbar):
  FRED (US-Notenbank St. Louis): S&P 500, Nasdaq, US-Dollar-Index, 10-jaehrige US-Rendite
  DefiLlama: Gesamtmenge der Stablecoins (frisches Geld im Kryptomarkt)
  Deribit: DVOL - Volatilitaetsindex des BTC-Optionsmarkts (Nervositaet der Profis)

Daraus je Tag ein Wert von -1 (Risiko aus) bis +1 (Risiko an), getrennt fuer Krypto und
Gold. Tagesdaten sind erst am Folgetag sicher bekannt -> sie werden mit 2 Tagen Versatz
verwendet (kein Blick in die Zukunft).
"""
import io
import logging
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("bot")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_SERIES = {"spx": "SP500", "ndx": "NASDAQCOM", "dxy": "DTWEXBGS", "us10y": "DGS10"}
STABLE_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
DAY_MS = 86_400_000
DELAY_MS = 2 * DAY_MS


def _fred(series: str, http) -> pd.Series:
    text = http.get(FRED_URL.format(series), timeout=30).text
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["date"]))


def _stablecoins(http) -> pd.Series:
    rows = http.get(STABLE_URL, timeout=30).json()
    idx = pd.to_datetime([int(r["date"]) for r in rows], unit="s")
    vals = [float((r.get("totalCirculatingUSD") or {}).get("peggedUSD") or 0) for r in rows]
    return pd.Series(vals, index=idx).replace(0, np.nan).dropna()


def _dvol(http, days: int) -> pd.Series:
    end = int(time.time() * 1000)
    start = end - days * DAY_MS
    out = {}
    while True:
        r = http.get(DVOL_URL, params={"currency": "BTC", "start_timestamp": start, "end_timestamp": end,
                                       "resolution": "1D"}, timeout=30).json()["result"]
        for ts, _o, _h, _l, c in r.get("data", []):
            out[int(ts)] = float(c)
        cont = r.get("continuation")
        if not cont or cont <= start:
            break
        end = int(cont)
    s = pd.Series(out).sort_index()
    s.index = pd.to_datetime(s.index, unit="ms")
    return s


def trend_sign(s: pd.Series, n: int = 50) -> pd.Series:
    """+1 ueber dem n-Tage-Durchschnitt, -1 darunter."""
    return np.sign(s - s.rolling(n, min_periods=n // 2).mean())


def compute_scores(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """Tages-Werte -> Ampel je Tag. Fehlende Quellen werden einfach ausgelassen."""
    parts_crypto, parts_gold = [], []
    daily = lambda s: s.resample("1D").last().ffill()  # noqa: E731
    if "spx" in raw or "ndx" in raw:
        eq = [trend_sign(daily(raw[k])) for k in ("spx", "ndx") if k in raw]
        parts_crypto.append(pd.concat(eq, axis=1).mean(axis=1))               # Aktien steigen = Risiko an
    if "dxy" in raw:
        d = -trend_sign(daily(raw["dxy"]))                                     # Dollar steigt = Risiko aus
        parts_crypto.append(d)
        parts_gold.append(d)
    if "us10y" in raw:
        y = daily(raw["us10y"])
        z = -np.sign(y - y.shift(20))                                          # Zinsen steigen = Risiko aus
        parts_crypto.append(z)
        parts_gold.append(z)
    if "stable" in raw:
        st = daily(raw["stable"])
        parts_crypto.append(np.sign(st.pct_change(30)))                        # Stablecoins wachsen = Geld kommt
    if "dvol" in raw:
        dv = daily(raw["dvol"])
        rank = dv.rolling(180, min_periods=30).rank(pct=True)
        parts_crypto.append(pd.Series(np.where(rank > 0.8, -1.0, 0.0), index=dv.index))  # Panik am Optionsmarkt
    out = pd.DataFrame({
        "crypto": pd.concat(parts_crypto, axis=1).mean(axis=1) if parts_crypto else pd.Series(dtype=float),
        "gold": pd.concat(parts_gold, axis=1).mean(axis=1) if parts_gold else pd.Series(dtype=float),
    }).sort_index()
    out["avail"] = (out.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1) + DELAY_MS
    return out.dropna(how="all", subset=["crypto", "gold"])


def fetch_macro(days: int = 400, http=requests) -> pd.DataFrame | None:
    """Alle Quellen laden; einzelne Ausfaelle werden uebersprungen."""
    raw = {}
    for key, series in FRED_SERIES.items():
        try:
            raw[key] = _fred(series, http)
        except Exception as e:  # noqa: BLE001
            log.warning("Makro %s (FRED %s) nicht abrufbar: %s", key, series, e)
    try:
        raw["stable"] = _stablecoins(http)
    except Exception as e:  # noqa: BLE001
        log.warning("Stablecoin-Daten nicht abrufbar: %s", e)
    try:
        raw["dvol"] = _dvol(http, days)
    except Exception as e:  # noqa: BLE001
        log.warning("DVOL nicht abrufbar: %s", e)
    return compute_scores(raw) if raw else None


def merge_macro(bars: pd.DataFrame, scores: pd.DataFrame | None, gold: bool) -> pd.Series:
    """Makro-Wert je Bar (nur was zum Bar-Schluss schon bekannt war)."""
    col = "gold" if gold else "crypto"
    if scores is None or len(scores) == 0 or scores[col].isna().all():
        return pd.Series(np.nan, index=bars.index)
    right = scores[["avail", col]].dropna().sort_values("avail")
    left = pd.DataFrame({"avail": bars["avail"].to_numpy(), "_i": np.arange(len(bars))})
    m = pd.merge_asof(left.sort_values("avail"), right, on="avail", direction="backward").sort_values("_i")
    return pd.Series(m[col].to_numpy(), index=bars.index)


def macro_blocks(side: str, score: float | None, s: dict) -> bool:
    """Makro-Ampel klar gegen die Richtung? (Long bei Risiko aus, Short bei Risiko an)"""
    if not s.get("macro_filter") or score is None or score != score:
        return False
    lim = s.get("macro_min", 0.3)
    return score <= -lim if side == "long" else score >= lim


def latest(scores: pd.DataFrame | None, gold: bool, now_ms: int) -> float | None:
    if scores is None or len(scores) == 0:
        return None
    col = "gold" if gold else "crypto"
    ok = scores[scores["avail"] <= now_ms][col].dropna()
    return float(ok.iloc[-1]) if len(ok) else None
