"""Lokale Oberflaeche: http://localhost:<port> (nur auf diesem PC erreichbar).

Aktionen (Schluessel speichern, Positionen schliessen, Orders ...) nur mit einem Zufalls-Token,
das nur die eigene Seite kennt - fremde Webseiten im selben Browser koennen nichts ausloesen.
"""
import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("bot")
HTML = Path(__file__).with_name("dashboard.html")
CHART_JS = Path(__file__).with_name("lightweight-charts.js")  # TradingView, Apache-2.0
MAX_BODY = 20_000


def _num(v):
    if v is None or v == "":
        return None
    return float(str(v).replace(",", "."))


def handle_action(bot, account, path: str, body: dict, stop_file: Path) -> str:
    """Alle Knoepfe der Oberflaeche. Rueckgabe: Meldung fuer den Nutzer (Fehler -> Exception)."""
    if path == "/api/bot/pause":
        if body.get("on"):
            stop_file.write_text("Pause ueber die Oberflaeche\n", encoding="utf-8")
            return "Bot pausiert - keine neuen Trades (offene Positionen laufen mit Stop/Ziel weiter)"
        stop_file.unlink(missing_ok=True)
        return "Bot handelt wieder"
    if path == "/api/bot/close":
        return bot.request_close(body["symbol"])
    if account is None:
        raise RuntimeError("Konto-Funktion nicht verfuegbar")
    if path == "/api/account/connect":
        account.connect(body.get("key", ""), body.get("secret", ""), body.get("password", ""),
                        bool(body.get("demo")))
        return "Verbunden - Schluessel lokal in .env gespeichert"
    if path == "/api/account/disconnect":
        account.disconnect(forget=bool(body.get("forget")))
        return "Schluessel geloescht" if body.get("forget") else "Getrennt"
    if path == "/api/account/refresh":
        account.refresh(force=True)
        return "Aktualisiert"
    if path == "/api/account/close":
        return account.close(body["symbol"], float(body.get("fraction") or 1.0))
    if path == "/api/account/cancel":
        return account.cancel(str(body["id"]), body["symbol"], body.get("kind", "normal"))
    if path == "/api/account/tpsl":
        return account.set_tpsl(body["symbol"], _num(body.get("sl")), _num(body.get("tp")))
    if path == "/api/account/order":
        return account.order(body["symbol"], body["side"], _num(body.get("usdt")), int(_num(body.get("leverage")) or 0),
                             _num(body.get("sl")), _num(body.get("tp")), body.get("type", "market"),
                             _num(body.get("price")))
    raise KeyError(path)


def start_dashboard(bot, port: int, account=None, stop_file: Path | None = None) -> ThreadingHTTPServer:
    token = secrets.token_hex(16)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if stop_file is None:
        from .engine import STOP_FILE
        stop_file = STOP_FILE

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):  # noqa: N802 - Name vorgegeben
            if self.path.startswith("/api/status"):
                self._json(200, bot.status)
            elif self.path.startswith("/api/account"):
                self._json(200, account.view() if account else {"connected": False, "error": "nicht verfuegbar"})
            elif self.path == "/lightweight-charts.js":
                self._send(200, CHART_JS.read_bytes(), "application/javascript")
            elif self.path in ("/", "/index.html"):
                html = HTML.read_text(encoding="utf-8").replace("__TOKEN__", token)
                self._send(200, html.encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            origin = self.headers.get("Origin")
            if (self.headers.get("Host") not in allowed_hosts or self.headers.get("X-Token") != token
                    or (origin and origin.split("//")[-1] not in allowed_hosts)):
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

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Oberflaeche: http://localhost:%d", port)
    return server
