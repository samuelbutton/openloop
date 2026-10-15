"""Loopback-only, read-only UI adapter. Routes cannot select files or run jobs.

The host selects a ledger at startup. Static assets are allowlisted. Ledger
text is untrusted display data; the browser inserts it with textContent only.
"""

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from openloop.ledger import InvalidInputError, Ledger, LedgerError, NotFoundError
from openloop.toolbelt.reports import compare_runs

from .reports import snapshot

_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/dom.js": ("dom.js", "text/javascript; charset=utf-8"),
    "/curves.js": ("curves.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def create_server(ledger_path: str | Path, *, port: int = 8765) -> ThreadingHTTPServer:
    """Bind only IPv4 loopback and open the existing ledger read-only per request."""
    path = Path(ledger_path).resolve(strict=True)
    with Ledger(path, read_only=True):
        pass
    assets = Path(__file__).with_name("static")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            # Never log request paths or run IDs supplied by a browser.
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self'; base-uri 'none'; "
                "frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: object) -> None:
            body = json.dumps(data, allow_nan=False, ensure_ascii=True).encode()
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            authority = self.headers.get("Host", "")
            allowed = {
                f"127.0.0.1:{server.server_port}",
                f"localhost:{server.server_port}",
            }
            origin = self.headers.get("Origin")
            if (
                authority not in allowed
                or (origin is not None and origin != f"http://{authority}")
                or self.headers.get("Sec-Fetch-Site") == "cross-site"
            ):
                self._json(403, {"error": "Use the local UI address"})
                return
            request = urlsplit(self.path)
            try:
                if request.path in _ASSETS and not request.query:
                    filename, content_type = _ASSETS[request.path]
                    self._send(200, (assets / filename).read_bytes(), content_type)
                elif request.path == "/api/snapshot" and not request.query:
                    with Ledger(path, read_only=True) as ledger:
                        data = snapshot(ledger.query_all(include_reused=True))
                    self._json(200, data | {"ledger": path.name})
                elif request.path == "/api/diff":
                    query = parse_qs(request.query, keep_blank_values=True)
                    if set(query) != {"baseline", "candidate"} or any(
                        len(values) != 1 or not values[0] for values in query.values()
                    ):
                        raise InvalidInputError("Choose one baseline and one candidate")
                    with Ledger(path, read_only=True) as ledger:
                        report = compare_runs(
                            ledger.get(query["baseline"][0]),
                            ledger.get(query["candidate"][0]),
                        )
                    self._json(200, report)
                else:
                    self._json(404, {"error": "Unknown UI route"})
            except NotFoundError as error:
                self._json(404, {"error": str(error)})
            except LedgerError as error:
                self._json(400, {"error": str(error)})
            except sqlite3.Error:
                # Busy, moved, or corrupt ledgers still get a JSON reply.
                self._json(503, {"error": "Cannot read the ledger"})

        def do_POST(self) -> None:
            self._json(405, {"error": "This UI is read-only"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server
