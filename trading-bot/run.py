"""Start:
    python run.py backtest --days 60   # Strategie an alten Daten pruefen
    python run.py optimize --days 180  # viele Einstellungen testen (dauert)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["backtest", "optimize", "check", "bot"])
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
        optimize_cli(cfg, args.days or 180)
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
