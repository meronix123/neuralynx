"""Lokale Oberflaeche: http://localhost:<port> (nur auf diesem PC erreichbar).

Aktionen (Schluessel speichern, Positionen schliessen, Orders ...) nur mit einem Zufalls-Token,
das nur die eigene Seite kennt - fremde Webseiten im selben Browser koennen nichts ausloesen.
"""
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("bot")
HTML = Path(__file__).with_name("dashboard.html")
CHART_JS = Path(__file__).with_name("lightweight-charts.js")  # TradingView, Apache-2.0
MAX_BODY = 20_000
SESSION_DAYS = 30
MAX_FAILS, LOCK_S = 5, 15 * 60

LOGIN_HTML = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Trading-Bot - Anmelden</title>
<style>body{margin:0;background:#0f1115;color:#e6e8ee;font:16px system-ui,sans-serif;display:flex;
min-height:100vh;align-items:center;justify-content:center}form{background:#171a21;border:1px solid #262b36;
border-radius:12px;padding:24px;width:min(340px,90vw)}input,button{width:100%;box-sizing:border-box;
padding:10px;margin-top:10px;border-radius:8px;border:1px solid #262b36;background:#11141a;color:#e6e8ee;
font:inherit}button{background:#1a2233;border-color:#3b82f6;cursor:pointer}.e{color:#ef4444;font-size:14px}</style>
</head><body><form method="post" action="/login"><b>Trading-Bot</b><div class="e">__MSG__</div>
<input type="password" name="password" placeholder="Passwort" autofocus autocomplete="current-password">
<button>Anmelden</button></form></body></html>"""


def hash_password(pw: str, salt: bytes | None = None, iters: int = 200_000) -> str:
    """Passwort nur als Hash speichern (PBKDF2-SHA256), nie im Klartext."""
    salt = salt or secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iters)
    return f"pbkdf2${iters}${salt.hex()}${h.hex()}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, iters, salt, h = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(calc.hex(), h)
    except (ValueError, TypeError):
        return False


def _num(v):
    if v is None or v == "":
        return None
    return float(str(v).replace(",", "."))


MODE_NAMES = {"paper": "Simulation (Paper)", "demo": "Testkonto (Bitget-Demo)", "live": "ECHTES KONTO"}


def restart_bot(bot, live_ok: bool) -> None:
    """Zustand speichern und den Bot mit dem neuen Modus neu starten (gleiches Terminal)."""
    from .engine import save_state
    time.sleep(1.5)  # Antwort an die Oberflaeche geht noch raus
    try:
        if hasattr(bot.ex, "dump"):
            bot.state["paper"] = bot.ex.dump()
        save_state(bot.state, bot.cfg["mode"])
    except Exception as e:  # noqa: BLE001
        log.warning("Zustand vor Neustart: %s", e)
    if live_ok:
        os.environ["BOT_LIVE_OK"] = "1"
    run_py = str(Path(__file__).resolve().parent.parent / "run.py")
    log.info("Neustart im neuen Modus ...")
    os.execv(sys.executable, [sys.executable, run_py, "bot"])


def switch_mode(bot, body: dict, restart=True, env_path=None) -> str:
    from .account import PROFILE_KEYS, save_env
    mode = body.get("mode")
    if mode not in MODE_NAMES:
        raise ValueError("Modus paper, demo oder live")
    if mode != "paper":
        profile = PROFILE_KEYS["demo" if mode == "demo" else "live"]
        legacy = mode == "demo" and os.getenv("BITGET_DEMO") == "1" and os.getenv("BITGET_API_KEY")
        if not all(os.getenv(n) for n in profile) and not legacy:
            raise RuntimeError(f"Erst im Reiter 'Bitget-Konto' die Schluessel fuer {MODE_NAMES[mode]} verbinden")
    if mode == "live" and body.get("confirm") != "JA":
        raise RuntimeError("Echtgeld nur mit Bestaetigung JA")
    if mode == bot.cfg["mode"]:
        return f"Bot laeuft bereits im Modus {MODE_NAMES[mode]}"
    save_env({"BOT_MODE": mode}, env_path)
    if restart:
        threading.Thread(target=restart_bot, args=(bot, mode == "live"), daemon=True).start()
    return f"Modus {MODE_NAMES[mode]} gespeichert - Bot startet neu (ca. 5 s), Seite laedt dann neu"


def speed_start(bot, account, body: dict, sp=None) -> str:
    """Speed-Trading / KI-Autopilot starten: Simulation oder verbundenes Bitget-Konto (Echtgeld nur mit JA)."""
    from .speed import BitgetBroker, PaperBroker
    sp = sp or bot.speed
    other = bot.autopilot if sp is bot.speed else bot.speed
    syms = [s for s in body.get("symbols") or [] if s]
    scan_n = _num(body.get("scan_top"))
    scan_fn = None
    if scan_n and int(scan_n) > 0 and sp is getattr(bot, "autopilot", None):
        scan_fn = (lambda n=int(scan_n): bot.scan_markets(n)["symbols"])
        opts_scan = int(scan_n)
        found = scan_fn()
        log.info("Markt-Scanner (Top %d): %s", int(scan_n), ", ".join(found) or "keine passenden Maerkte")
        for s in found:                                              # Markt-Scanner: Top-Maerkte nach Umsatz dazu
            if s not in syms:
                syms.append(s)
    else:
        opts_scan = None
    opts = {k: _num(body.get(k)) for k in ("minutes", "margin_usdt", "leverage", "aggressiveness", "min_conf")}
    opts = {k: (int(v) if k in ("minutes", "leverage", "aggressiveness") else float(v)) for k, v in opts.items() if v}
    if "min_conf" in opts and opts["min_conf"] > 1:
        opts["min_conf"] /= 100                     # 56 -> 0,56
    opts["use_raw"] = bool(body.get("use_raw"))
    opts["scan_top"] = opts_scan
    opts["fast"] = bool(body.get("fast"))
    opts["turbo"] = bool(body.get("turbo"))
    if body.get("size_mode") in ("auto", "usdt", "pct", "risk"):
        opts["size_mode"] = body["size_mode"]
    for k in ("size_pct", "risk_pct", "partial_frac"):
        v = _num(body.get(k))
        if v is not None:
            opts[k] = float(v) / 100 if k == "partial_frac" and float(v) > 1 else float(v)
    if body.get("cost_guard") is not None:
        opts["cost_guard"] = bool(body.get("cost_guard"))
        if not opts["cost_guard"] and body.get("target") == "account" and body.get("confirm_nocost") != "OHNE SCHUTZ":
            raise RuntimeError("Ohne Kosten-Schutz auf dem Konto nur mit Bestaetigung OHNE SCHUTZ")
    if body.get("scale_in") is not None:
        opts["scale_in"] = bool(body.get("scale_in"))
    if body.get("target") == "account":
        if not account or not account.connected:
            raise RuntimeError("Erst im Reiter Bitget-Konto ein Konto verbinden")
        if not account.demo and body.get("confirm") != "JA":
            raise RuntimeError(f"{sp.name} mit ECHTEM Geld nur mit Bestaetigung JA")
        v = account.view()

        def free(sym):
            """Im Konto nichts anderes offen (Einzel-Positionen, Bot, andere Sitzung)? Sonst dort warten,
            denn Bitget wuerde die Positionen zusammenlegen."""
            av = account.view()
            if any(p["symbol"] == sym for p in av.get("positions") or []):
                return False
            if any(t["symbol"] == sym for t in account.tickets().active()):
                return False
            if sym in (getattr(bot, "state", None) or {}).get("meta", {}) or other.busy(sym):
                return False
            return True
        broker = BitgetBroker(account.client, sp.p, bot.cfg.get("margin_mode", "isolated"))
        from .exchange import apply_leverage
        for s in syms:          # Hebel/Margin-Modus fuer die Speed-Maerkte setzen
            try:
                account.client.set_margin_mode(broker.mm, s, {"marginCoin": "USDT"})
            except Exception:  # noqa: BLE001 - schon gesetzt
                pass
            if opts.get("size_mode") != "auto":        # bei "auto" setzt die KI den Hebel je Trade
                apply_leverage(account.client, s, int(opts.get("leverage", sp.p["leverage"])), broker.mm)
        bal = v.get("balance") or {}
        equity = float(bal.get("total") or bal.get("free") or 0)     # Limits auf das ganze Konto beziehen
        opts["free"] = float(bal.get("free") or 0)
        label = "Testkonto" if account.demo else "ECHTES KONTO"
    else:
        broker = PaperBroker(lambda s: sp.market.get(s) or bot.speed_data(s)[2], sp.p)
        equity = float(bot.cfg.get("paper", {}).get("start_equity", 35))
        label = "Simulation"
    return sp.start(syms, broker, equity, label, free_fn=free if body.get("target") == "account" else None,
                    scan_fn=scan_fn, **opts)


def resume_sessions(bot, account) -> list[str]:
    """Beim Start des Bots: auf Platte gespeicherte Konto-Sitzungen (KI-Autopilot, Speed) fortsetzen,
    damit offene Positionen nach einem Neustart weiter gefuehrt werden."""
    from .speed import BitgetBroker
    out = []
    for sp in (getattr(bot, "autopilot", None), getattr(bot, "speed", None)):
        if sp is None:
            continue
        snap = sp.saved_session(sp.p.get("log_path"))
        if not snap:
            continue
        label = (snap.get("session") or {}).get("label")
        if label == "Simulation":
            continue                      # Spielgeld-Stand ist nach dem Neustart weg
        if not account or not account.connected or (label == "Testkonto") != account.demo:
            log.warning("%s: gespeicherte Sitzung (%s) kann ohne passendes Konto nicht fortgesetzt werden", sp.name, label)
            continue
        other = bot.autopilot if sp is bot.speed else bot.speed

        def free(sym, other=other):
            if any(t["symbol"] == sym for t in account.tickets().active()):
                return False
            if sym in (getattr(bot, "state", None) or {}).get("meta", {}) or other.busy(sym):
                return False
            return True
        try:
            broker = BitgetBroker(account.client, sp.p, bot.cfg.get("margin_mode", "isolated"))
            snap["session"]["resumed"] = True
            n_scan = (snap.get("cur") or {}).get("scan_top")
            sp.scan_fn = (lambda n=int(n_scan): bot.scan_markets(n)["symbols"]) if n_scan else None
            out.append(sp.resume(snap, broker, free_fn=free))
        except Exception as e:  # noqa: BLE001
            log.warning("%s: Sitzung nicht fortgesetzt: %s", sp.name, e)
    return out


def handle_action(bot, account, path: str, body: dict, stop_file: Path) -> str:
    """Alle Knoepfe der Oberflaeche. Rueckgabe: Meldung fuer den Nutzer (Fehler -> Exception)."""
    if path == "/api/bot/reset_peak":
        return bot.reset_peak()
    if path == "/api/bot/scan":
        r = bot.scan_markets(int(_num(body.get("n")) or 10))
        top = ", ".join(f"{x['symbol'].split('/')[0]} ({x['volume'] / 1e6:.0f} Mio)" for x in r["top"][:int(_num(body.get("n")) or 10)])
        new = ", ".join(f"{x['symbol'].split('/')[0]} ({x['age_days']} T)" for x in r["new"]) or "keine"
        return f"Scanner: {top or 'keine passenden Maerkte'} | frische Listings (nur Anzeige): {new}"
    if path == "/api/bot/pause":
        if body.get("on"):
            stop_file.write_text("Pause ueber die Oberflaeche\n", encoding="utf-8")
            return "Bot pausiert - keine neuen Trades (offene Positionen laufen mit Stop/Ziel weiter)"
        stop_file.unlink(missing_ok=True)
        return "Bot handelt wieder"
    if path == "/api/bot/close":
        return bot.request_close(body["symbol"])
    if path == "/api/bot/close_all":
        return bot.request_close_all()
    if path == "/api/bot/mode":
        return switch_mode(bot, body, env_path=getattr(account, "env_path", None))
    if account is None:
        raise RuntimeError("Konto-Funktion nicht verfuegbar")
    if path == "/api/account/connect":
        account.connect(body.get("key", ""), body.get("secret", ""), body.get("password", ""),
                        bool(body.get("demo")))
        return "Verbunden - Schluessel lokal in .env gespeichert"
    if path == "/api/account/disconnect":
        account.disconnect(forget=bool(body.get("forget")))
        return "Schluessel geloescht" if body.get("forget") else "Getrennt"
    if path == "/api/account/switch":
        return account.switch(body["profile"])
    if path == "/api/account/close_all":
        return account.close_all()
    if path == "/api/account/refresh":
        account.refresh(force=True)
        return "Aktualisiert"
    if path == "/api/account/close":
        return account.close(body["symbol"], float(body.get("fraction") or 1.0))
    if path == "/api/account/cancel":
        return account.cancel(str(body["id"]), body["symbol"], body.get("kind", "normal"))
    if path == "/api/account/tpsl":
        return account.set_tpsl(body["symbol"], _num(body.get("sl")), _num(body.get("tp")))
    if path == "/api/speed/start":
        return speed_start(bot, account, body)
    if path == "/api/speed/stop":
        return bot.speed.stop(close=bool(body.get("close")))
    if path == "/api/auto/start":
        return speed_start(bot, account, body, bot.autopilot)
    if path == "/api/auto/stop":
        return bot.autopilot.stop(close=bool(body.get("close")))
    if path == "/api/account/quick":
        for tr in (getattr(bot, "speed", None), getattr(bot, "autopilot", None)):
            if tr is not None and tr.busy(body.get("symbol")):
                raise RuntimeError(f"{tr.name} haelt in diesem Markt gerade selbst eine Position - Bitget wuerde "
                                   "deine Order damit zusammenlegen. Anderen Markt waehlen oder warten, bis sie zu ist.")
        st = getattr(bot, "state", None) or {}
        if (bot.cfg.get("mode") == account.active and
                (body.get("symbol") in st.get("meta", {}) or body.get("symbol") in st.get("pending", {}))):
            raise RuntimeError(f"Der Bot hat in {body['symbol'].split(':')[0]} gerade selbst eine Position/Order auf "
                               "diesem Konto - Bitget wuerde deine Einzel-Position damit zusammenlegen und der Stop des "
                               "Bots gilt fuer alles. Bitte einen anderen Markt waehlen oder warten, bis der Bot fertig ist.")
        return account.quick_order(body["symbol"], body["side"], float(_num(body.get("margin_usdt")) or 0),
                                   int(_num(body.get("leverage")) or 10), float(_num(body.get("sl_pct")) or 10),
                                   float(_num(body.get("tp_pct")) or 0), body.get("type", "market"),
                                   _num(body.get("price")))
    if path == "/api/account/ticket_close":
        return account.close_ticket(int(body["id"]))
    if path == "/api/account/tickets_close_all":
        return account.close_all_tickets()
    if path == "/api/account/ticket_move":
        return account.move_level(int(body["id"]), body.get("kind", ""), _num(body.get("price")))
    if path == "/api/account/ticket_be":
        return account.breakeven(int(body["id"]))
    if path == "/api/account/order":
        return account.order(body["symbol"], body["side"], _num(body.get("usdt")), int(_num(body.get("leverage")) or 0),
                             _num(body.get("sl")), _num(body.get("tp")), body.get("type", "market"),
                             _num(body.get("price")))
    raise KeyError(path)


def start_dashboard(bot, port: int, account=None, stop_file: Path | None = None,
                    remote_pw_hash: str | None = None) -> ThreadingHTTPServer:
    """remote_pw_hash gesetzt = Fernzugriff (Handy) erlaubt: lauscht im Netzwerk, aber nur mit Passwort.
    Vom PC selbst (localhost) geht es weiterhin ohne Anmeldung."""
    token = secrets.token_hex(16)
    sessions: dict[str, float] = {}          # Sitzungs-Cookie -> gueltig bis
    fails: dict[str, list] = {}              # IP -> [Fehlversuche, gesperrt bis]
    if stop_file is None:
        from .engine import STOP_FILE
        stop_file = STOP_FILE

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass   # Browser hat die Anfrage abgebrochen (z. B. schnell umgeschaltet) - kein Fehler

        def _local(self) -> bool:
            p = self.server.server_address[1]  # echter Port (auch wenn 0 = "frei waehlen" angegeben war)
            return (self.client_address[0] in ("127.0.0.1", "::1")
                    and self.headers.get("Host") in {f"127.0.0.1:{p}", f"localhost:{p}"})

        def _session(self) -> bool:
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == "botsession" and sessions.get(v, 0) > time.time():
                    return True
            return False

        def _authorized(self) -> bool:
            return self._local() or (remote_pw_hash is not None and self._session())

        def _login_page(self, msg: str = "", code: int = 200) -> None:
            self._send(code, LOGIN_HTML.replace("__MSG__", msg).encode(), "text/html; charset=utf-8")

        def _login(self) -> None:
            ip = self.client_address[0]
            f = fails.setdefault(ip, [0, 0.0])
            if f[1] > time.time():
                self._login_page("Zu viele Fehlversuche - bitte 15 Minuten warten.", 429)
                return
            n = min(int(self.headers.get("Content-Length") or 0), 2000)
            raw = self.rfile.read(n).decode(errors="replace")
            from urllib.parse import parse_qs
            pw = (parse_qs(raw).get("password") or [""])[0]
            if remote_pw_hash and check_password(pw, remote_pw_hash):
                fails.pop(ip, None)
                sid = secrets.token_hex(24)
                sessions[sid] = time.time() + SESSION_DAYS * 86_400
                self._send(303, b"", "text/plain", {
                    "Location": "/",
                    "Set-Cookie": f"botsession={sid}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_DAYS * 86_400}",
                })
                log.info("Oberflaeche: Anmeldung von %s", ip)
                return
            f[0] += 1
            if f[0] >= MAX_FAILS:
                f[:] = [0, time.time() + LOCK_S]
                log.warning("Oberflaeche: %d Fehlversuche von %s - 15 Minuten gesperrt", MAX_FAILS, ip)
            self._login_page("Falsches Passwort.", 401)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):  # noqa: N802 - Name vorgegeben
            if not self._authorized():
                if self.path.startswith("/api/"):
                    self._json(401, {"ok": False, "msg": "Bitte anmelden"})
                else:
                    self._login_page()
                return
            if self.path.startswith("/api/status"):
                st = bot.status
                if isinstance(st, dict) and hasattr(bot, "_modes_ready"):   # Schluessel gerade verbunden? sofort zeigen
                    st = {**st, "modes_ready": bot._modes_ready(), "mode": bot.cfg.get("mode", st.get("mode"))}
                self._json(200, st)
            elif self.path.startswith("/api/live"):
                try:
                    self._json(200, {"prices": bot.live_prices(), "ms": int(time.time() * 1000)})
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"ok": False, "msg": str(e)})
            elif self.path.startswith("/api/chart"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                try:
                    self._json(200, bot.chart_data(q.get("symbol", [""])[0], q.get("tf", [""])[0]))
                except Exception as e:  # noqa: BLE001 - z. B. Netz kurz weg
                    self._json(400, {"ok": False, "msg": str(e)})
            elif self.path.startswith("/api/speed"):
                self._json(200, bot.speed.status())
            elif self.path.startswith("/api/auto"):
                self._json(200, bot.autopilot.status())
            elif self.path.startswith("/api/forecast"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                try:
                    self._json(200, bot.forecast(q.get("symbol", [""])[0]))
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"ok": False, "msg": str(e)})
            elif self.path.startswith("/api/brain"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                try:
                    self._json(200, bot.brain(q.get("symbol", [""])[0]))
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"ok": False, "msg": str(e)})
            elif self.path.startswith("/api/account"):
                self._json(200, account.view() if account else {"connected": False, "error": "nicht verfuegbar"})
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/lightweight-charts.js":
                self._send(200, CHART_JS.read_bytes(), "application/javascript")
            elif self.path in ("/", "/index.html"):
                html = HTML.read_text(encoding="utf-8").replace("__TOKEN__", token)
                self._send(200, html.encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if self.path == "/login":
                self._login()
                return
            origin = self.headers.get("Origin")
            host = self.headers.get("Host") or ""
            same_origin = not origin or origin.split("//")[-1] == host
            if not self._authorized() or self.headers.get("X-Token") != token or not same_origin:
                self._json(403, {"ok": False, "msg": "Nicht erlaubt"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self._json(413, {"ok": False, "msg": "Zu gross"})
                return
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                msg = handle_action(bot, account, self.path, body, stop_file)
                self._json(200, {"ok": True, "msg": msg})
            except KeyError as e:
                self._json(404 if str(e).strip("'").startswith("/") else 400, {"ok": False, "msg": f"Fehlt: {e}"})
            except Exception as e:  # noqa: BLE001 - Fehler von Bitget an die Oberflaeche geben
                log.warning("Aktion %s: %s", self.path, e)
                self._json(400, {"ok": False, "msg": str(e)})

        def log_message(self, *args):
            pass

    class QuietServer(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            import sys as _sys
            if isinstance(_sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
                return   # abgebrochene Verbindung - kein Grund fuer eine Fehlermeldung
            super().handle_error(request, client_address)

    server = QuietServer(("0.0.0.0" if remote_pw_hash else "127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Oberflaeche: http://localhost:%d%s", port, " (Fernzugriff mit Passwort aktiv)" if remote_pw_hash else "")
    return server
