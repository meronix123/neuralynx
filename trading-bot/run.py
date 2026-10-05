"""Start:
    python run.py backtest --days 60   # Strategie an alten Daten pruefen
    python run.py optimize --days 365  # viele Einstellungen testen (dauert)
    python run.py optimize --fast      # Schnell-Modus: 5m/15m/30m-Einstiege (120 Tage)
    python run.py optimize --zeiten    # feste Zeiteinheit gegen automatische Wahl (365 Tage)
    python run.py optimize --orderblocks  # Order-Block-Filter testen (365 Tage)
    python run.py optimize --alle-zeiten  # Auto-Wahl mit 5 Minuten bis 1 Tag testen (180 Tage)
    python run.py report               # Auswertung der bisherigen Trades (Paper/Demo/Live)
    python run.py signale              # welche Signale es gab und warum (nicht) gehandelt wurde
    python run.py fernzugriff          # Oberflaeche auch vom Handy (mit Passwort) / "fernzugriff aus"
    python run.py modus live           # womit der Bot handelt: paper (Simulation) | demo (Testkonto) | live
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


def _tailscale_ip() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5)
        ip = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
        return ip
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def set_mode(cfg: dict, option: str) -> None:
    """Modus des Bots im Terminal umstellen (gilt ab dem naechsten Start)."""
    from bot.account import save_env

    names = {"paper": "Simulation", "demo": "Testkonto (Bitget-Demo)", "live": "ECHTES KONTO"}
    alias = {"simulation": "paper", "test": "demo", "testkonto": "demo", "echt": "live", "real": "live"}
    mode = alias.get(option.lower(), option.lower())
    print(f"Aktueller Modus: {names.get(cfg['mode'], cfg['mode'])}")
    if mode not in names:
        print("Bitte angeben: python run.py modus paper | demo | live")
        return
    keys = {"demo": ("BITGET_DEMO_API_KEY", "BITGET_DEMO_API_SECRET", "BITGET_DEMO_API_PASSPHRASE"),
            "live": ("BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE")}.get(mode, ())
    legacy_demo = mode == "demo" and os.getenv("BITGET_DEMO") == "1" and os.getenv("BITGET_API_KEY")
    if keys and not all(os.getenv(k) for k in keys) and not legacy_demo:
        print(f"Fuer {names[mode]} sind keine API-Schluessel gespeichert.")
        print("Oberflaeche -> Reiter 'Bitget-Konto' -> Schluessel eintragen und bei 'Diese Schluessel sind fuer'")
        print(f"'{'Echtes Konto' if mode == 'live' else 'Testkonto'}' waehlen -> Verbinden. Danach diesen Befehl nochmal.")
        return
    if mode == "live":
        print(f"\n!!! Der Bot handelt dann SELBSTSTAENDIG mit ECHTEM GELD und {cfg['leverage']}x Hebel !!!")
        if input("Zum Bestaetigen JA eingeben: ").strip().upper() != "JA":
            print("Abgebrochen - nichts geaendert.")
            return
    save_env({"BOT_MODE": mode})
    print(f"Gespeichert: {names[mode]}. Jetzt den Bot neu starten (Strg+C, dann python run.py bot).")


def remote_setup(cfg: dict, option: str) -> None:
    """Fernzugriff ein-/ausschalten. Passwort wird nur als Hash in .env gespeichert."""
    from getpass import getpass

    from bot.account import save_env
    from bot.dashboard import hash_password

    if option.lower() in ("aus", "off"):
        save_env({"DASHBOARD_REMOTE": "0"})
        print("Fernzugriff AUS. Die Oberflaeche ist nur noch auf diesem PC erreichbar (nach Neustart des Bots).")
        return
    print("Fernzugriff: Die Oberflaeche wird im Netzwerk erreichbar - NUR mit Passwort.")
    print("Wichtig: KEINE Portfreigabe im Router! Fuer unterwegs Tailscale nutzen (siehe Anleitung).")
    ask = input if option.lower() == "sichtbar" else getpass
    if ask is getpass:
        print("(Beim Tippen erscheint nichts - das ist normal. Einfach tippen und Enter druecken.")
        print(" Lieber sichtbar tippen? Abbrechen mit Strg+C und: python run.py fernzugriff sichtbar)")
    while True:
        pw = ask("Neues Passwort (mind. 10 Zeichen): ").strip()
        if len(pw) < 10:
            print("Zu kurz.")
            continue
        if ask("Passwort wiederholen: ").strip() != pw:
            print("Stimmt nicht ueberein.")
            continue
        break
    save_env({"DASHBOARD_REMOTE": "1", "DASHBOARD_PASSWORD_HASH": hash_password(pw)})
    port = cfg["dashboard"]["port"]
    ip = _tailscale_ip()
    print("\nGespeichert. Bot neu starten, dann auf dem Handy oeffnen:")
    if ip:
        print(f"  unterwegs (Tailscale): http://{ip}:{port}")
    else:
        print("  Tailscale ist noch nicht eingerichtet - siehe Anleitung. Danach zeigt dieser Befehl die Adresse.")
    print(f"  im selben WLAN: http://<IP dieses PCs>:{port}   (IP anzeigen: hostname -I)")


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
    ap.add_argument("command", choices=["backtest", "optimize", "diagnose", "ki-bericht", "handelsbericht", "report", "signale", "fernzugriff", "modus", "check", "bot"])
    ap.add_argument("option", nargs="?", default="")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="Schnell-Modus (5m/15m/30m)")
    ap.add_argument("--zeiten", action="store_true", help="feste Zeiteinheit gegen automatische Wahl")
    ap.add_argument("--orderblocks", action="store_true", help="Order-Block-Filter testen")
    ap.add_argument("--alle-zeiten", dest="alle_zeiten", action="store_true", help="5m bis 1d testen")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging()

    if args.command == "backtest":
        from bot.backtest import backtest_cli
        backtest_cli(cfg, args.days or 60)
        return
    if args.command == "optimize":
        from bot.optimize import compare_cli, optimize_cli, orderblocks_cli
        if args.alle_zeiten:
            from bot.optimize import all_timeframes_cli
            all_timeframes_cli(cfg, args.days or 180)
            return
        if args.orderblocks:
            orderblocks_cli(cfg, args.days or 365)
            return
        if args.zeiten:
            compare_cli(cfg, args.days or 365)
            return
        optimize_cli(cfg, args.days or (120 if args.fast else 365), fast=args.fast)
        return

    if args.command == "handelsbericht":
        from bot.account import Account
        from bot.config import ROOT
        from bot.tradereport import report as trade_report
        acc = Account(cfg, cfg["symbols"])
        print(trade_report(cfg, ROOT / "data", acc if acc.connected else None))
        return
    if args.command == "ki-bericht":
        from bot.config import ROOT
        from bot.forecast import report as ki_report
        print("Werte die KI aus (frischer Test je Markt dauert etwas) ...")
        print(ki_report(ROOT / "data", cfg["mode"], None))
        if (ROOT / "data" / "ki1m").exists():
            print()
            print(ki_report(ROOT / "data", cfg["mode"], None, name="ki1m", step_min=1, horizons=[1, 2, 3, 5],
                            build_opts={"boost": False, "move_min": 0.05}, title="TURBO-KI (1-Minuten-Modell)"))
        return
    if args.command == "diagnose":
        from bot.optimize import diagnose_cli
        diagnose_cli(cfg, args.days or 90)
        return

    if args.command == "modus":
        set_mode(cfg, args.option)
        return
    if args.command == "fernzugriff":
        remote_setup(cfg, args.option)
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
        elif os.getenv("BOT_LIVE_CONFIRMED") == "1":
            print("Echtgeld per BOT_LIVE_CONFIRMED=1 in der .env freigegeben (Dienst-Betrieb).")
        elif not sys.stdin.isatty():
            sys.exit("Echtgeld ohne Tastatur (Dienst): zum Freigeben in die .env die Zeile BOT_LIVE_CONFIRMED=1 eintragen.")
        elif input("Zum Starten JA eingeben: ").strip() != "JA":
            sys.exit("Abgebrochen.")

    from bot.engine import Bot
    bot = Bot(cfg, ex)
    if cfg["dashboard"]["enabled"]:
        from bot.account import Account, refresher
        from bot.dashboard import start_dashboard
        account = Account(cfg, ex.symbols, async_refresh=True)   # echtes Bitget-Konto (Schluessel in der Oberflaeche)
        refresher(account)
        from bot.dashboard import resume_sessions
        for msg in resume_sessions(bot, account):
            print(msg)
        pw_hash = os.getenv("DASHBOARD_PASSWORD_HASH") if os.getenv("DASHBOARD_REMOTE") == "1" else None
        try:
            start_dashboard(bot, cfg["dashboard"]["port"], account, remote_pw_hash=pw_hash or None)
        except OSError as e:
            if getattr(e, "errno", None) in (98, 48, 10048):     # Port belegt (Linux / Mac / Windows)
                sys.exit(f"\nDer Bot laeuft schon (Port {cfg['dashboard']['port']} ist belegt) - zwei Bots wuerden doppelt "
                         "handeln.\nAlten Bot beenden:  pkill -f \"run.py bot\"   (oder im anderen Fenster Strg+C)\n"
                         "Danach neu starten.")
            raise
        print(f"Oberflaeche im Browser oeffnen: http://localhost:{cfg['dashboard']['port']}")
        if pw_hash:
            ip = _tailscale_ip()
            print("Fernzugriff (mit Passwort) aktiv" + (f": vom Handy http://{ip}:{cfg['dashboard']['port']}" if ip else ""))
    try:
        bot.run()
    except KeyboardInterrupt:
        if cfg["mode"] == "paper":
            print("\nBot gestoppt. Simulierte Positionen bleiben gespeichert und laufen beim naechsten Start weiter.")
        else:
            print("\nBot gestoppt. Offene Positionen behalten ihren Stop-Loss/Take-Profit auf Bitget.")


if __name__ == "__main__":
    main()
