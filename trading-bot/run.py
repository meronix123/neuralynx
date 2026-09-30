"""Start:
    python run.py backtest --days 60   # Strategie an alten Daten pruefen
    python run.py optimize --days 365  # viele Einstellungen testen (dauert)
    python run.py report               # Auswertung der bisherigen Trades (Paper/Demo/Live)
    python run.py check                # Verbindung + API-Schluessel pruefen
    python run.py bot                  # Bot starten (Modus aus config.yaml)
"""
import argparse
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


def report() -> None:
    """Ist der Bot reif fuer Echtgeld? Auswertung aus state.json."""
    from bot.engine import load_state

    hist = load_state().get("history", [])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["backtest", "optimize", "report", "check", "bot"])
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    setup_logging()

    if args.command == "backtest":
        from bot.backtest import backtest_cli
        backtest_cli(cfg, args.days or 60)
        return
    if args.command == "optimize":
        from bot.optimize import optimize_cli
        optimize_cli(cfg, args.days or 365)
        return

    if args.command == "report":
        report()
        return

    ex = make_exchange(cfg)
    print(f"Modus: {cfg['mode']} | Symbole: {', '.join(ex.symbols)}")
    print(f"Kontostand: {ex.equity():.2f} USDT")
    if args.command == "check":
        print("Verbindung OK.")
        return

    if cfg["mode"] == "live":
        print("\n!!! ACHTUNG: ECHTES GELD, Hebel", cfg["leverage"], "!!!")
        if input("Zum Starten JA eingeben: ").strip() != "JA":
            sys.exit("Abgebrochen.")

    from bot.engine import Bot
    bot = Bot(cfg, ex)
    if cfg["dashboard"]["enabled"]:
        from bot.dashboard import start_dashboard
        start_dashboard(bot, cfg["dashboard"]["port"])
        print(f"Oberflaeche im Browser oeffnen: http://localhost:{cfg['dashboard']['port']}")
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nBot gestoppt. Offene Positionen behalten ihren Stop-Loss/Take-Profit auf Bitget.")


if __name__ == "__main__":
    main()
