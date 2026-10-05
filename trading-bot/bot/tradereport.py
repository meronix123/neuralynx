"""Handelsbericht: Wo wurde Geld verdient, wo verloren? (python run.py handelsbericht)

Quellen: Bitget-Historie (alle geschlossenen Positionen der letzten 30 Tage, echte Ergebnisse inkl.
Gebuehren/Funding) und die gespeicherten Speed-/Autopilot-Abschluesse (mit Grund und KI-Sicherheit).
"""
import json
import time
from collections import defaultdict
from pathlib import Path


def _stats(rows: list[dict]) -> dict:
    net = [r["net"] for r in rows]
    win = [x for x in net if x > 0]
    loss = -sum(x for x in net if x < 0)
    return {"n": len(rows), "net": sum(net), "fees": sum(r.get("fees") or 0 for r in rows),
            "gross": sum(r.get("gross", r["net"] + (r.get("fees") or 0)) for r in rows),
            "hit": len(win) / len(rows) if rows else 0.0,
            "pf": (sum(win) / loss) if loss > 0 else (99.0 if win else 0.0),
            "avg_win": sum(win) / len(win) if win else 0.0,
            "avg_loss": -loss / (len(net) - len(win)) if len(net) > len(win) else 0.0}


def _line(name: str, st: dict) -> str:
    return (f"   {name:<26} {st['n']:>4} Trades  {st['hit'] * 100:5.1f} % im Plus  netto {st['net']:+8.3f}  "
            f"Gebuehren {st['fees']:7.3f}  PF {st['pf']:4.2f}")


def _group(rows, key) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def from_bitget(history: list[dict]) -> list[dict]:
    """Bitget-Positionshistorie -> einheitliche Zeilen."""
    out = []
    for h in history:
        if h.get("pnl") is None:
            continue
        fees = abs(h.get("fees") or 0)
        net = float(h["pnl"])
        out.append({"source": "bitget", "symbol": h.get("symbol") or "?", "side": h.get("side") or "?",
                    "net": net, "fees": fees, "gross": net + fees - (h.get("funding") or 0),
                    "time": h.get("closed_ms") or h.get("opened_ms") or 0,
                    "secs": ((h.get("closed_ms") or 0) - (h.get("opened_ms") or 0)) / 1000 if h.get("opened_ms") else None})
    return out


def from_sessions(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            t = json.loads(line)
        except ValueError:
            continue
        t["source"] = t.get("kind") or "speed"
        out.append(t)
    return out


def analyze(rows: list[dict], title: str) -> list[str]:
    w = []
    if not rows:
        return [f"{title}: keine Daten"]
    st = _stats(rows)
    days = max(1.0, (max(r["time"] for r in rows) - min(r["time"] for r in rows)) / 86_400_000)
    w.append(f"===== {title} =====")
    w.append(f"{st['n']} Abschluesse in {days:.0f} Tagen ({st['n'] / days:.1f}/Tag): netto {st['net']:+.3f} USDT, "
             f"davon Gebuehren {st['fees']:.3f}, vor Gebuehren {st['gross']:+.3f}")
    w.append(f"Im Plus: {st['hit'] * 100:.1f} %, Durchschnitt Gewinn {st['avg_win']:+.4f} / Verlust {st['avg_loss']:+.4f}, "
             f"Profit-Faktor {st['pf']:.2f}")
    if st["gross"] > 0 > st["net"]:
        w.append("-> VOR Gebuehren im Plus, die GEBUEHREN machen den Verlust: weniger, groessere Trades / Limit-Orders.")
    elif st["gross"] <= 0:
        w.append("-> Schon VOR Gebuehren im Minus: die Einstiege selbst waren schlecht (Signal ohne Vorsprung).")
    sections = [
        ("Je Markt", lambda r: str(r.get("symbol", "?")).split(":")[0]),
        ("Long / Short", lambda r: r.get("side", "?")),
        ("Je Uhrzeit (UTC)", lambda r: (lambda b: f"{b:02d}-{b + 4:02d} Uhr")(time.gmtime(r["time"] / 1000).tm_hour // 4 * 4)
                                       if r.get("time") else "?"),
        ("Haltedauer", lambda r: "unter 1 min" if (r.get("secs") or 0) < 60 else "1-5 min" if r["secs"] < 300
         else "5-30 min" if r["secs"] < 1800 else "30 min - 4 Std" if r["secs"] < 14400 else "ueber 4 Std"),
    ]
    if any("why" in r for r in rows):
        sections.append(("Ausstiegsgrund", lambda r: str(r.get("why", "?"))[:24]))
        sections.append(("KI-Sicherheit beim Einstieg", lambda r: (
            "-" if not (r.get("why_in") or {}).get("Sicherheit") else
            "unter 56 %" if r["why_in"]["Sicherheit"] < 0.56 else "56-60 %" if r["why_in"]["Sicherheit"] < 0.6
            else "ueber 60 %")))
    bad = []
    for name, key in sections:
        w.append(f"{name}:")
        for k, g in sorted(_group(rows, key).items(), key=lambda kv: _stats(kv[1])["net"]):
            s_ = _stats(g)
            w.append(_line(str(k), s_))
            if name in ("Je Markt", "Je Uhrzeit (UTC)") and s_["n"] >= 10 and s_["net"] < 0 and s_["pf"] < 0.8:
                bad.append(f"{name}: {k} ({s_['n']} Trades, netto {s_['net']:+.2f}, PF {s_['pf']:.2f})")
    if bad:
        w.append("Klar verlustreich (mind. 10 Trades, PF < 0,8) - hier besser nicht handeln:")
        w += [f"   - {b}" for b in bad]
    return w


def report(cfg: dict, data_dir: Path, account=None) -> str:
    lines = []
    if account is not None:
        try:
            account.refresh(force=True)
            hist = account.view().get("history") or []
            lines += analyze(from_bitget(hist), "BITGET-KONTO (letzte 30 Tage, alle geschlossenen Positionen)")
        except Exception as e:  # noqa: BLE001
            lines.append(f"Bitget-Historie nicht abrufbar: {e}")
        lines.append("")
    rows = from_sessions(data_dir / f"sitzungen_{cfg['mode']}.jsonl")
    lines += analyze(rows, f"SPEED-TRADING / KI-AUTOPILOT (gespeicherte Abschluesse, Modus {cfg['mode']})")
    if not rows:
        lines.append("(Abschluesse werden ab diesem Update gespeichert - nach ein paar Tagen erneut ausfuehren.)")
    return "\n".join(lines)
