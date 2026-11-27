"""DEMO ENGINES: two stand-in inference servers on one container, for the gateway to probe.

  :8100  the `node` cluster's head on box-a (fleet.demo.yml) — answers for the whole run.
  :8001  box-c's classic server — answers for DEMO_ENGINE_DOWN_AFTER_S seconds, then closes its listener, so the
         gateway's `backend_down` finding goes PENDING (fleet.demo.yml holds it pending for an hour) and the Check
         dialog shows a failed engine row with its hint.

Each answers what the gateway's prober reads: GET /v1/models, /metrics, /health. Nothing here is a model; a chat
request gets a canned reply. Stdlib only; run as `python -I engine.py` in the gateway's own image.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVED = {8100: "Qwen/Qwen3.8-27B", 8001: "Qwen/Qwen3.8-27B"}


def handler(served: str):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: D102 — quiet
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/v1/models"):
                data = [{"id": served, "object": "model", "max_model_len": 65536}]
                return self._send(200, json.dumps({"object": "list", "data": data}).encode(), "application/json")
            if self.path.startswith("/metrics"):
                text = (f'vllm:num_requests_waiting{{model_name="{served}"}} 0\n'
                        f'vllm:num_requests_running{{model_name="{served}"}} 1\n')
                return self._send(200, text.encode(), "text/plain")
            if self.path.startswith("/health"):
                return self._send(200, b"", "text/plain")
            return self._send(404, b'{"error":"not found"}', "application/json")

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            reply = {"id": "demo", "object": "chat.completion", "model": served,
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}
            return self._send(200, json.dumps(reply).encode(), "application/json")
    return H


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("0.0.0.0", port), handler(SERVED[port]))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    down_after = float(os.environ["DEMO_ENGINE_DOWN_AFTER_S"])
    serve(8100)
    classic = serve(8001)
    print(f"[engine] :8100 and :8001 up; :8001 stops in {down_after:g} s", flush=True)
    time.sleep(down_after)
    classic.shutdown()
    classic.server_close()
    print("[engine] :8001 down", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
