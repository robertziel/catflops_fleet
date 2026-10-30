"""The gateway panel, UNCHANGED, for the browser test: vllm-lb's own static files (mounted
read-only from lb/vllm_lb/static) and a /lb/status captured from the live LB at test start.
Nothing here talks to vllm-lb; the page's control calls (POST) are answered 403 and counted."""
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATIC = pathlib.Path(sys.argv[1])
STATUS = pathlib.Path(sys.argv[2]).read_bytes()
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "application/javascript; charset=utf-8"}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p == "/lb/status":
            return self._send(200, STATUS, "application/json")
        if p.startswith("/lb/"):
            return self._send(200, b"{}", "application/json")
        name = {"/": "index.html"}.get(p, p.lstrip("/"))
        f = (STATIC / name)
        if "/" in name or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        print("PANEL-CONTROL-POST", self.path, flush=True)
        self._send(403, b'{"error":"fixture"}', "application/json")

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
