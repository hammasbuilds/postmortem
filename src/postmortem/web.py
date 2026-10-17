"""Minimal web layer: stdlib only, no framework, no build step.

One dependency-free server per product keeps the three apps installable with
nothing but Python, which matters while everything runs CPU-only on one laptop.
Routes are explicit, JSON in and JSON out, and the static UI is served from
``ui/`` beside the package.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

Handler = Callable[[dict], object]


class Raw:
    """A non-JSON response: a file the browser should download.

    Clause returns review reports as .md, .html and .pdf, so the web layer needs
    one way to send bytes with a filename instead of a JSON body.
    """

    def __init__(self, payload: bytes, content_type: str, filename: str = ""):
        self.payload = payload
        self.content_type = content_type
        self.filename = filename


class App:
    def __init__(self, static_dir: Path, *, title: str = "app"):
        self.static_dir = Path(static_dir)
        self.title = title
        self.routes: dict[str, Handler] = {}
        self.lock = threading.Lock()

    def route(self, path: str):
        def deco(fn: Handler) -> Handler:
            self.routes[path] = fn
            return fn

        return deco

    def serve(self, host: str = "127.0.0.1", port: int = 8800) -> None:
        app = self

        class RequestHandler(BaseHTTPRequestHandler):
            server_version = f"{app.title}/1.0"

            def log_message(self, fmt: str, *args) -> None:  # quieter console
                if "500" in (fmt % args):
                    print(f"[{self.address_string()}] {fmt % args}")

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, payload: object) -> None:
                self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

            def _static(self, path: str) -> None:
                rel = path.lstrip("/") or "index.html"
                target = (app.static_dir / rel).resolve()
                if not str(target).startswith(str(app.static_dir.resolve())) or not target.is_file():
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype == "application/javascript":
                    ctype += "; charset=utf-8"
                self._send(200, target.read_bytes(), ctype)

            def _handle(self, payload: dict) -> None:
                parsed = urlparse(self.path)
                fn = app.routes.get(parsed.path)
                if fn is None:
                    self._static(parsed.path)
                    return
                payload |= {k: v[0] for k, v in parse_qs(parsed.query).items()}
                try:
                    result = fn(payload)
                    if isinstance(result, Raw):
                        self.send_response(200)
                        self.send_header("Content-Type", result.content_type)
                        self.send_header("Content-Length", str(len(result.payload)))
                        if result.filename:
                            self.send_header(
                                "Content-Disposition", f'attachment; filename="{result.filename}"'
                            )
                        self.end_headers()
                        self.wfile.write(result.payload)
                        return
                    self._json(200, result)
                except Exception as exc:  # noqa: BLE001 - report, never crash the server
                    traceback.print_exc()
                    self._json(500, {"error": type(exc).__name__, "detail": str(exc)})

            def do_GET(self) -> None:
                self._handle({})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self._json(400, {"error": "invalid JSON body"})
                    return
                self._handle(body if isinstance(body, dict) else {"value": body})

        httpd = ThreadingHTTPServer((host, port), RequestHandler)
        print(f"{self.title} on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
        httpd.serve_forever()
