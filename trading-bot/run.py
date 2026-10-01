"""Start:
    python run.py backtest --days 60   # Strategie an alten Daten pruefen
    python run.py optimize --days 365  # viele Einstellungen testen (dauert)
    python run.py optimize --fast      # Schnell-Modus: 5m/15m/30m-Einstiege (120 Tage)
    python run.py optimize --zeiten    # feste Zeiteinheit gegen automatische Wahl (365 Tage)
    python run.py optimize --orderblocks  # Order-Block-Filter testen (365 Tage)
    python run.py report               # Auswertung der bisherigen Trades (Paper/Demo/Live)
    python run.py signale              # welche Signale es gab und warum (nicht) gehandelt wurde
    python run.py check                # Verbindung + API-Schluessel pruefen
    python run.py bot                  # Bot starten (Modus aus config.yaml)
"""
import argparse
import os
import logging
import sys

from bot.config import ROOT, load_config


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "bot.log", encoding="utf-8"),
        ],
    )


def make_exchange(cfg):
    from bot.exchange import BitgetExchange, PaperExchange

    if cfg["mode"] == "paper":
        return PaperExchange(cfg)
    if not (cfg["api"]["key"] and cfg["api"]["secret"] and cfg["api"]["password"]):
        sys.exit("API-Schluessel fehlen - siehe .env.example")
    return BitgetExchange(cfg)


def report(mode: str = "paper") -> None:
    """Ist der Bot reif fuer Echtgeld? Auswertung der Trades im aktuellen Modus."""
    from bot.engine import load_state

    print(f"Auswertung Modus: {mode}")
    hist = load_state(mode).get("history", [])
    if not hist:
        print("Noch keine abgeschlossenen Trades.")
        return
    pnl = [t["pnl"] for t in hist]
    wins = [p for p in pnl if p > 0]
    losses = [-p for p in pnl if p <= 0]
    pf = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float("inf")
    eq, peak, mdd = 0.0, 0.0, 0.0
    for p in pnl:
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    print(f"Trades:         {len(pnl)}")
    print(f"Trefferquote:   {100 * len(wins) / len(pnl):.1f} %")
    print(f"Profit-Faktor:  {pf:.2f}")
    print(f"Summe:          {sum(pnl):+.2f} USDT")
    print(f"Max. Rueckgang: {mdd:.2f} USDT")
    ready = len(pnl) >= 30 and pf >= 1.3
    print("\nBereit fuer Echtgeld:", "JA (Kriterien erfuellt)" if ready else
          "NEIN - noetig: mind. 30 Trades und Profit-Faktor >= 1,3")


def signals(mode: str = "paper") -> None:
    """Letzte Signale mit Grund - zum Nachsehen, warum der Bot (nicht) gehandelt hat."""
    from collections import Counter
    from datetime import datetime

    from bot.engine import load_state

    log_ = load_state(mode).get("signal_log", [])
    if not log_:
        print("Noch keine Signale protokolliert (gibt es ab dieser Version).")
        return
    for x in log_[-40:]:
        t = datetime.fromtimestamp(x["ms"] / 1000).strftime("%d.%m. %H:%M")
        mark = "GEHANDELT" if x["traded"] else "nein"
        print(f"{t}  {x['symbol'].split(':')[0]:<10} {x['tf']:<4} {x['side']:<5} {mark:<9} {x['why']}")
    c = Counter(x["why"].split("(")[0].strip() for x in log_ if not x["traded"])
    print(f"\n{len(log_)} Signale, {sum(x['traded'] for x in log_)} gehandelt. Haeufigste Gruende dagegen:")
    for why, n in c.most_common(6):
        print(f"  {n:>4}x  {why}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["backtest", "optimize", "report", "signale", "check", "bot"])
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="Schnell-Modus (5m/15m/30m)")
    ap.add_argument("--zeiten", action="store_true", help="feste Zeiteinheit gegen automatische Wahl")
    ap.add_argument("--orderblocks", action="store_true", help="Order-Block-Filter testen")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging()

    if args.command == "backtest":
        from bot.backtest import backtest_cli
        backtest_cli(cfg, args.days or 60)
        return
    if args.command == "optimize":
        from bot.optimize import compare_cli, optimize_cli, orderblocks_cli
        if args.orderblocks:
            orderblocks_cli(cfg, args.days or 365)
            return
        if args.zeiten:
            compare_cli(cfg, args.days or 365)
            return
        optimize_cli(cfg, args.days or (120 if args.fast else 365), fast=args.fast)
        return

    if args.command == "signale":
        signals(cfg["mode"])
        return
    if args.command == "report":
        report(cfg["mode"])
        return

    ex = make_exchange(cfg)
    print(f"Modus: {cfg['mode']} | Symbole: {', '.join(ex.symbols)}")
    print(f"Kontostand: {ex.equity():.2f} USDT")
    if args.command == "check":
        print("Verbindung OK.")
        return

    if cfg["mode"] == "live":
        print("\n!!! ACHTUNG: ECHTES GELD, Hebel", cfg["leverage"], "!!!")
        if os.environ.pop("BOT_LIVE_OK", "") == "1":
            print("Echtgeld in der Oberflaeche mit JA bestaetigt.")
        elif input("Zum Starten JA eingeben: ").strip() != "JA":
            sys.exit("Abgebrochen.")

    from bot.engine import Bot
    bot = Bot(cfg, ex)
    if cfg["dashboard"]["enabled"]:
        from bot.account import Account, refresher
        from bot.dashboard import start_dashboard
        account = Account(cfg, ex.symbols)   # echtes Bitget-Konto (Schluessel in der Oberflaeche eingeben)
        refresher(account)
        start_dashboard(bot, cfg["dashboard"]["port"], account)
        print(f"Oberflaeche im Browser oeffnen: http://localhost:{cfg['dashboard']['port']}")
    try:
        bot.run()
    except KeyboardInterrupt:
        if cfg["mode"] == "paper":
            print("\nBot gestoppt. Simulierte Positionen bleiben gespeichert und laufen beim naechsten Start weiter.")
        else:
            print("\nBot gestoppt. Offene Positionen behalten ihren Stop-Loss/Take-Profit auf Bitget.")


if __name__ == "__main__":
    main()
