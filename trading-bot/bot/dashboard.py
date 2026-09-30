"""Lokale Oberflaeche: http://localhost:<port> (nur auf diesem PC erreichbar)."""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("bot")
HTML = Path(__file__).with_name("dashboard.html")
CHART_JS = Path(__file__).with_name("lightweight-charts.js")  # TradingView, Apache-2.0


def start_dashboard(bot, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - Name vorgegeben
            if self.path.startswith("/api/status"):
                body = json.dumps(bot.status, default=str).encode()
                ctype = "application/json"
            elif self.path == "/lightweight-charts.js":
                body = CHART_JS.read_bytes()
                ctype = "application/javascript"
            elif self.path in ("/", "/index.html"):
                body = HTML.read_bytes()
                ctype = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Oberflaeche: http://localhost:%d", port)
    return server
