"""DEMO ALERT RECEIVER: the webhook channel's target, which refuses every alert with HTTP 500.

The gateway's alert sender (vllm_lb/alerts.py) never retries a failed send and never reroutes it to another channel; it
records `alert.failed` and the findings band shows the last failure. This receiver makes that failure happen on demand.
Each POST is logged (one line: the path and the body's length), so a reader of `docker logs demo-alerts` sees what the
gateway tried to send.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102 — one line per POST below instead
        return

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        print(f"[alerts] POST {self.path} {len(body)} B -> 500", flush=True)
        payload = b'{"error":"demo receiver refuses every alert"}'
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
