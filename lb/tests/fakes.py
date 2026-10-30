"""In-process fake backends for the proxy tests — a real HTTP server on 127.0.0.1.

`FakeBackend` answers like a vLLM/ollama would: `/v1/models`, `/metrics`, `/api/tags`, and
any inference path with a canned JSON body or an SSE stream (with per-chunk delays, or a
mid-stream drop). It records every request it saw so a test can assert on what was
forwarded. `down()` closes the listener so connections are refused.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeBackend:
    def __init__(self, *, kind: str = "vllm", served=("qwen3.8:latest",), max_model_len=4096,
                 port: int = 0):
        self.kind = kind
        self.served = list(served)
        self.max_model_len = max_model_len
        self.requests: list[dict] = []
        self.reply_json: dict = {"id": "msg_1", "type": "message", "content": [{"type": "text", "text": "OK"}]}
        self.reply_status = 200
        self.stream_chunks: list[bytes] | None = None
        self.stream_delay_s = 0.0
        self.drop_after_chunks: int | None = None
        self.drop_before_headers = False   # accept the request, then hang up: the risky case
        self.stall_after_chunks: int | None = None   # send N chunks then go silent, forever
        self.first_byte_delay_s = 0.0                # queued / loading / prefilling
        self.waiting = 0
        self.running = 0
        self.tags_hang_s = 0.0
        self.ps: dict = {"models": []}               # ollama's GET /api/ps: what is resident
        self.extra_headers: dict[str, str] = {}
        self.tokenize_count: int | None = None       # POST /tokenize's count; None answers 404
        self.tokenized: list[bytes] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # noqa: D102
                return

            def _record(self, body: bytes):
                self.server.fake.requests.append({
                    "method": self.command, "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                })

            def do_GET(self):
                self._record(b"")
                if self.path == "/v1/models":
                    data = [{"id": m, "object": "model", "max_model_len": fake.max_model_len}
                            for m in fake.served]
                    return self._json({"object": "list", "data": data})
                if self.path == "/metrics":
                    text = (f"vllm:num_requests_waiting{{model_name=\"x\"}} {fake.waiting}\n"
                            f"vllm:num_requests_running{{model_name=\"x\"}} {fake.running}\n")
                    return self._raw(200, text.encode(), "text/plain")
                if self.path == "/api/tags":
                    time.sleep(fake.tags_hang_s)
                    return self._json({"models": [{"name": m} for m in fake.served]})
                if self.path == "/api/ps":
                    return self._json(fake.ps)
                if self.path == "/health":
                    return self._raw(200, b"", "text/plain")
                return self._json({"error": "not found"}, 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                if self.path == "/tokenize":         # the LB's prompt count: kept apart from the relayed requests
                    fake.tokenized.append(body)
                    if fake.tokenize_count is None:
                        return self._json({"object": "error", "message": "Not Found"}, 404)
                    return self._json({"tokens": [1] * fake.tokenize_count, "count": fake.tokenize_count})
                self._record(body)
                if fake.first_byte_delay_s:
                    time.sleep(fake.first_byte_delay_s)
                if fake.drop_before_headers:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                if fake.stream_chunks is not None:
                    return self._stream()
                return self._json(fake.reply_json, fake.reply_status)

            def _json(self, obj, status=200):
                self._raw(status, json.dumps(obj).encode(), "application/json")

            def _raw(self, status, payload: bytes, ctype: str):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                for k, v in fake.extra_headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i, chunk in enumerate(fake.stream_chunks):
                    if fake.stall_after_chunks is not None and i >= fake.stall_after_chunks:
                        time.sleep(120)      # silence, without closing: the "stuck" case
                        return
                    if fake.drop_after_chunks is not None and i >= fake.drop_after_chunks:
                        self.close_connection = True
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        return
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                    time.sleep(fake.stream_delay_s)
                if fake.stall_after_chunks is not None and len(fake.stream_chunks) >= fake.stall_after_chunks:
                    time.sleep(120)
                    return
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        ThreadingHTTPServer.allow_reuse_address = True
        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._server.fake = self
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def down(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    close = down


def free_port() -> int:
    """A port nothing listens on yet — so a backend can be configured before it exists."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeProber:
    """Answers `probe_backend`'s contract from a table; a url mapped to an Exception raises."""

    def __init__(self, table: dict | None = None):
        self.table = dict(table or {})
        self.calls: list[str] = []

    def __call__(self, backend, cfg, *, now):
        self.calls.append(backend.url)
        val = self.table.get(backend.url, {"live": False, "last_error": "unprobed"})
        if isinstance(val, Exception):
            raise val
        return {"last_probe_at": now, **val}


def claim_gpu(host: str, *, pool_modules=(), serving_context: int | None = 262144,
              current_model=None, require_model=False, known_models=None,
              served_types=("ollama", "vllm"), box=None):
    """Claim as a GPU lane would. On a broker_parrot with migration 0028 the lane states the
    window its LLM server serves (`serving_context`); on an older engine that argument does
    not exist and every row is unconstrained. One helper so the tests read the same on both.

    `box` is the claiming worker's PHYSICAL box (migration 0020). Without it the engine claims
    only UNCONSTRAINED rows — so a fallback lease, which pins its box with `force_box`, is
    invisible to a claim that does not name one."""
    import inspect

    from queue_workflows import node_queue

    kw = {}
    if "serving_context" in inspect.signature(node_queue.claim_next_gpu_job).parameters:
        kw["serving_context"] = serving_context
    if known_models is not None:
        kw["known_models"] = known_models
    if box is not None and "box" in inspect.signature(node_queue.claim_next_gpu_job).parameters:
        kw["box"] = box
    return node_queue.claim_next_gpu_job(0, current_model, host=host,
                                         served_types=list(served_types),
                                         require_model=require_model,
                                         pool_modules=pool_modules, **kw)


class FakeClusterExecutor:
    """Stands in for `cluster_exec.SshExecutor`: one rank per GB10 box, the broker's vLLM and free
    memory per box, what each box HAS, reachability per box, and a log of every call — no SSH, no
    Docker."""

    def __init__(self, boxes=("box-a", "box-b")):
        # None | {"running": bool, "gen": int, "exit_code": int | None, "model_choice": str | None}
        self.ranks = {b: None for b in boxes}
        self.broker_vllm = {b: False for b in boxes}
        self.mem = {b: 115.0 for b in boxes}
        #: box -> what its OWN ollama says is resident, as tp-rank hands it over
        #: (`{"date": <the server's Date>, "ps": <the /api/ps body>}`); None = it could not be asked.
        self.ollama: dict[str, dict | None] = {b: None for b in boxes}
        #: box -> its `gpu` object as tp-rank reads it off the driver; None = no nvidia-smi there, or
        #: the driver did not answer. A fake box observes nothing until a test says it does.
        self.gpu: dict[str, dict | None] = {b: None for b in boxes}
        self.reachable = {b: True for b in boxes}
        self.log_text = {b: "" for b in boxes}
        #: box -> its rank's docker `StartedAt` as tp-rank reports it (RFC 3339, UTC, the box's own clock); None =
        #: not reported, which is what a fake box says until a test gives it one.
        self.started_at: dict[str, str | None] = {b: None for b in boxes}
        #: box -> when the box BOOTED (its /proc/stat `btime`, UTC epoch seconds) as tp-rank reports it; None = it
        #: could not say. A rank whose `started_at` is before this did not exit — the box rebooted under it.
        self.boot_at: dict[str, float | None] = {b: None for b in boxes}
        #: box -> the box's OWN clock (a callable returning its `date +%s.%N`), sent as tp-rank sends it: as the
        #: `t` of its RDMA reading (`net`). None = no `net` at all, as from a box with no RDMA device.
        self.box_clock: dict = {b: None for b in boxes}
        #: box -> what `tp-rank stacks` prints for its RUNNING rank: py-spy's dump of every engine process.
        self.stacks_text = {b: "" for b in boxes}
        #: box -> {spec id: True present | False missing | None the box could not say | a str: the box LOOKED and
        #: could not see, in tp-rank's own words ("… is not mounted: NFS … is not up on this box yet")}. A spec
        #: left out is present; the whole table set to None is a box that answers nothing at all.
        self.have: dict[str, dict | None] = {b: {} for b in boxes}
        self.calls: list[tuple] = []
        self.last_error: dict[str, str] = {}
        #: What `heal` answers from a reachable box.
        self.heal_reply: tuple[bool, str] = (True, "remounted /srv/hdd; worker backstop restarted")
        #: The settings every call carries, as `SshExecutor.env` holds them (`cluster_run.tp_rank_env`): the fleet's
        #: prefill budget among them — the test fleet's 8192, as lb/fleet.example.yml states it.
        self.env: dict[str, str] = {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"}
        #: box -> the operator's engine knobs its last `up` carried: {max_num_batched_tokens,
        #: num_speculative_tokens}, None where the default was left to the launch.
        self.knobs: dict[str, dict] = {}

    def _gone(self, host) -> bool:
        if self.reachable[host.box]:
            self.last_error[host.box] = ""
            return False
        self.last_error[host.box] = "ssh: connect to host timed out"
        return True

    def status(self, host):
        self.calls.append(("status", host.box))
        if self._gone(host):
            return None
        r = self.ranks[host.box]
        return {"rank": {"present": r is not None, "running": bool(r and r["running"]),
                         "gen": r["gen"] if r else None, "exit_code": r.get("exit_code") if r else None,
                         "started_at": self.started_at[host.box] if r else None,
                         "model_choice": r.get("model_choice") if r else None},
                "broker_vllm_running": self.broker_vllm[host.box],
                "mem_available_gib": self.mem[host.box], "mem_total_gib": 121.7, "port": 8100, "boot_at": self.boot_at[host.box],
                "ollama": self.ollama[host.box], "gpu": self.gpu[host.box],
                "net": (None if self.box_clock[host.box] is None else
                        {"tx_bytes": 0, "rx_bytes": 0, "devices": 1, "pcie_gt_lanes": 0,
                         "t": self.box_clock[host.box]()})}

    def up(self, host, *, rank, gen, max_num_seqs, max_model_len, spec=None, max_num_batched_tokens=None,
           num_speculative_tokens=None, ple_replicate=None, kv_cache_memory=None, kv_offload_gib=None,
           long_prefill_token_threshold=None, uncensored=None, kv_cache_dtype=None):
        """The ModelChoice travels as its ModelSpec, and the rank remembers the id it was launched for. The
        operator's engine knobs are kept per box in `knobs`, so the call tuple stays what it always was."""
        choice = spec.id if spec is not None else None
        self.calls.append(("up", host.box, rank, gen, max_num_seqs, max_model_len, choice))
        self.knobs[host.box] = {"max_num_batched_tokens": max_num_batched_tokens,
                                "num_speculative_tokens": num_speculative_tokens,
                                "ple_replicate": ple_replicate, "kv_cache_memory": kv_cache_memory,
                                "kv_offload_gib": kv_offload_gib,
                                "long_prefill_token_threshold": long_prefill_token_threshold,
                                "uncensored": uncensored,
                                # only when the run names one, so a call from before the knob reads as it did
                                **({"kv_cache_dtype": kv_cache_dtype} if kv_cache_dtype is not None else {})}
        if self._gone(host):
            return False, self.last_error[host.box]
        self.ranks[host.box] = {"running": True, "gen": gen, "exit_code": None, "model_choice": choice}
        return True, "started"

    def availability(self, host, specs, *, observed_at=None):
        """One `Availability` per ModelSpec, as the box would answer — None when it could not be asked."""
        from vllm_lb.catalog import CLUSTER_MODE, Availability

        specs = list(specs)
        self.calls.append(("availability", host.box, tuple(s.availability_key for s in specs)))
        if self._gone(host) or self.have[host.box] is None:
            return None
        out = {}
        for s in specs:
            # `have[box]` names a CHECKPOINT by its availability key; a model id alone covers the model and its
            # twin, so a test from before twins existed still hides both.
            present = self.have[host.box].get(s.availability_key, self.have[host.box].get(s.id, True))
            if isinstance(present, str):             # it looked and could not see: no folder, its reason
                out[s.availability_key] = Availability(
                    name=s.name, searched=tuple(s.cluster_dirs), dir=None,
                    image=s.profile(CLUSTER_MODE).image, image_present=True,
                    observed_at=observed_at, error=present)
                continue
            out[s.availability_key] = Availability(
                name=s.name, searched=tuple(s.cluster_dirs),
                dir=s.cluster_dirs[0] if present else None,
                image=s.profile(CLUSTER_MODE).image, image_present=present,
                observed_at=observed_at,
                error="" if present is not None else f"{host.box} could not look")
        return out

    def down(self, host):
        self.calls.append(("down", host.box))
        if self._gone(host):
            return False, self.last_error[host.box]
        self.ranks[host.box] = None
        return True, "removed"

    def halt(self, host):
        """`docker stop`: the rank stays, exited (137), and keeps its log."""
        self.calls.append(("halt", host.box))
        if self._gone(host):
            return False, self.last_error[host.box]
        r = self.ranks[host.box]
        if r is None or not r["running"]:
            return True, "nothing to halt"
        self.ranks[host.box] = {**r, "running": False, "exit_code": 137}
        return True, "halted"

    def heal(self, host):
        """tp-rank `heal`: remount the box's NFS and restart its worker backstop. `heal_reply` is what the
        box says; a box that cannot be reached fails like every other verb."""
        self.calls.append(("heal", host.box))
        if self._gone(host):
            return False, self.last_error[host.box]
        return self.heal_reply

    def stacks(self, host):
        """`tp-rank stacks`: py-spy's dump of a RUNNING rank's engine processes — "" with no rank running,
        or from a box that cannot be reached, as the real executor answers."""
        self.calls.append(("stacks", host.box))
        if self._gone(host):
            return ""
        r = self.ranks[host.box]
        return self.stacks_text[host.box] if r is not None and r["running"] else ""

    def logs(self, host, n=160):
        """`docker logs --tail n` of `log_text`: its last n lines. Split on \\n only — a tqdm \\r is
        not a line end, and a read that ignored n would hide a tail too short to hold a start."""
        self.calls.append(("logs", host.box, n))
        if self._gone(host):
            return ""
        lines = self.log_text[host.box].split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return "".join(f"{line}\n" for line in lines[-n:])
