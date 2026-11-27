"""THE FIDELITY BENCH: how well the LOADED model predicts a frozen text, so a quant, a KV dtype
or an engine change can be told apart from "same quality" by a number and not by feel.

WHAT IS MEASURED. Every document of the frozen corpus (library/fidelity_corpus.txt — public-domain text, its sources in
its header, its sha256 pinned as CORPUS_SHA256) is sent ONCE to `/v1/completions` with `echo: true, logprobs: 1,
max_tokens: 1`. With echo, the OpenAI legacy completions shape returns `choices[0].logprobs.token_logprobs` for every
PROMPT position followed by the one generated token: vLLM fills it from its prompt logprobs (serving_completion turns
echo + logprobs into `prompt_logprobs`), SGLang from its input-token logprobs (`logprob_start_len 0` under echo). The
first `usage.prompt_tokens` entries are the prompt; position 0 has no context and is dropped (it is null, or a score
against nothing), the generated token after the prompt is dropped. Per document the bench keeps the token count, the
summed negative log-likelihood and a hash of the token strings; per run, the token-weighted mean NLL, the perplexity
exp(NLL) and bits per byte. Nothing is generated that matters, nothing is sampled.

NO SUBSTITUTE MEASURE. An engine whose reply carries no token logprobs (no `logprobs`, no `token_logprobs`, a null
inside the prompt, no `usage.prompt_tokens` to say where the prompt ends) FAILS the run with an error naming the engine
and the document — never a perplexity estimated from something else. A reply from another model than the loaded one is
refused the same way: the request went through the gateway by served name, and a gateway that answered with a
different model would make the number a different model's.

THE LOADED MODEL ONLY, THROUGH THE GATEWAY. The request is sent to this LB's own listener (loopback, the trusted no-role
caller), with the served name of the model already serving — so it queues and routes like any client's request, and
the bench never loads, unloads or restarts anything. Nothing is serving → refused (409), nothing written.

NEVER ON ITS OWN. A run puts the whole corpus through the pair, so it starts only on the operator's POST
(`/lb/fidelity/run`, admin-only like every write the member allowlist does not name), one at a time: a second POST while
one runs is a 409. Nothing in the LB schedules it; a run left half-done by a restart is shown as interrupted, never
resumed.

WHAT A RUN RECORDS. The loaded model as the cluster states it (`describe_cluster`): served name, catalog model, its
engine and the target it resolved to (accelerator × TP — the key measurements use, library/measurements.json), its
quant format, and the KV dtype the run LAUNCHED (the effective one, else its profile's own, which is what a run that
chose none launches). A figure the LB cannot read is null with a `why` beside it — never a default.

THE COMPARISON. Two finished runs on one corpus pair document by document; only documents tokenised identically in both
(same token-string hash and count) pair, the rest are counted as mismatched. ΔNLL = NLL(B) − NLL(A), token-weighted
over the paired documents; its 95 % interval is a bootstrap over documents (BOOTSTRAP_REPS resamples, a fixed seed so
the same two runs always show the same interval). "Same quality" = the interval spans 0.

WHERE RUNS LIVE. One JSON file per run in `<perf_dir>/fidelity/` — the LB's state volume, beside the perf log and the
event feed, so no new setting: written atomically (atomic.py), read back like bench.py reads saved runs (an unreadable
file is listed with its reason, never dropped).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import pathlib
import random
import re
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Callable

from vllm_lb import atomic

log = logging.getLogger(__name__)

CORPUS_PATH = pathlib.Path(__file__).resolve().parent / "library" / "fidelity_corpus.txt"
#: The corpus IS these bytes. Changing the file without changing this is refused at load: a run's figures are only
#: comparable with another run's on the same text, so a new corpus must be a deliberate act, never an edit that slips.
CORPUS_SHA256 = "830241b5b7b2db342012a74dff68ff22fd628ed3e780e13c670370a5c66b10fa"

#: Resamples of the bootstrap, and its seed: fixed, so the interval of two given runs never moves between page loads.
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261105
#: How long one document may take through the gateway: a queue behind real traffic plus one prefill of a short text.
DOC_TIMEOUT_S = 900.0
#: The request every document is sent as (the prompt and the model are added per document).
REQUEST = {"echo": True, "logprobs": 1, "max_tokens": 1, "temperature": 0, "stream": False}

_ID_RE = re.compile(r"^fid-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4}$")
_DOC_RE = re.compile(r"^@@ ([a-z0-9][a-z0-9_-]*) \| (.+)$")


class FidelityError(Exception):
    """One document could not be scored: the words say which engine, which document and what was missing."""


class Busy(Exception):
    """A run is already running."""


class NothingLoaded(Exception):
    """No model is serving that a run could be sent to."""


class ServerRequired(Exception):
    """Several servers are serving: the run must name one."""


# ── the corpus ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Doc:
    id: str
    source: str
    text: str

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8"))


@dataclass(frozen=True)
class Corpus:
    docs: tuple[Doc, ...]
    sha256: str
    path: str

    def view(self) -> dict:
        return {"sha256": self.sha256, "docs": len(self.docs), "bytes": sum(d.bytes for d in self.docs),
                "file": pathlib.Path(self.path).name}


def load_corpus(path=CORPUS_PATH, expect_sha: str = CORPUS_SHA256) -> Corpus:
    """The frozen corpus, or ValueError: a file whose bytes are not the pinned ones, or one with no document."""
    raw = pathlib.Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expect_sha:
        raise ValueError(f"{path}: sha256 {sha[:12]}… is not the pinned corpus {expect_sha[:12]}… "
                         f"(fidelity.CORPUS_SHA256): a changed corpus is a new one, pinned deliberately")
    docs: list[Doc] = []
    cur: tuple[str, str] | None = None
    lines: list[str] = []

    def close():
        if cur is not None:
            text = "\n".join(lines).strip("\n")
            if not text.strip():
                raise ValueError(f"{path}: document {cur[0]!r} has no text")
            docs.append(Doc(cur[0], cur[1].strip(), text))

    for line in raw.decode("utf-8").splitlines():
        m = _DOC_RE.match(line)
        if m:
            close()
            cur, lines = (m.group(1), m.group(2)), []
        elif line.startswith("@@"):
            raise ValueError(f"{path}: {line!r} is not `@@ <id> | <source>`")
        elif cur is not None:
            lines.append(line)
    close()
    if not docs:
        raise ValueError(f"{path}: no document (`@@ <id> | <source>`)")
    if len({d.id for d in docs}) != len(docs):
        raise ValueError(f"{path}: two documents share an id")
    return Corpus(tuple(docs), sha, str(path))


# ── one document's score ──────────────────────────────────────────────────────


def score_reply(reply, *, engine, served_name: str, doc_id: str) -> dict:
    """{tokens, sum_nll, tokens_sha} for one document's reply — or FidelityError naming `engine` and the document."""
    who = f"{engine or 'the engine (unnamed)'} ({served_name}), document {doc_id}"
    if not isinstance(reply, dict):
        raise FidelityError(f"{who}: the reply is not a JSON object")
    model = reply.get("model")
    if isinstance(model, str) and model and model != served_name:
        raise FidelityError(f"{who}: the gateway answered with model {model!r}, not the loaded {served_name!r}")
    choices = reply.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise FidelityError(f"{who}: the reply has no choices")
    lp = choices[0].get("logprobs")
    values = lp.get("token_logprobs") if isinstance(lp, dict) else None
    tokens = lp.get("tokens") if isinstance(lp, dict) else None
    if not isinstance(values, list) or not isinstance(tokens, list):
        raise FidelityError(f"{who}: returned no token logprobs (choices[0].logprobs.token_logprobs is absent) — "
                            f"this engine does not score the prompt under echo + logprobs")
    n = (reply.get("usage") or {}).get("prompt_tokens") if isinstance(reply.get("usage"), dict) else None
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise FidelityError(f"{who}: no usage.prompt_tokens, so where the prompt ends among the "
                            f"{len(values)} positions cannot be told")
    if n < 2:
        raise FidelityError(f"{who}: fewer than two prompt tokens ({n}): nothing to score")
    if len(values) < n or len(tokens) < n:
        raise FidelityError(f"{who}: {len(values)} logprobs for {n} prompt tokens — the prompt was not echoed whole")
    total = 0.0
    for i in range(1, n):
        v = values[i]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise FidelityError(f"{who}: no logprob at prompt position {i} ({v!r})")
        total -= float(v)
    sha = hashlib.sha256(json.dumps(tokens[:n], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return {"tokens": n - 1, "sum_nll": total, "tokens_sha": sha}


def _totals(docs: list[dict], corpus_bytes: int) -> dict:
    tokens = sum(d["tokens"] for d in docs)
    nll = sum(d["sum_nll"] for d in docs)
    if not tokens:
        return {"tokens": 0, "mean_nll": None, "ppl": None, "bits_per_byte": None}
    return {"tokens": tokens, "mean_nll": nll / tokens, "ppl": math.exp(nll / tokens),
            "bits_per_byte": nll / math.log(2) / corpus_bytes if corpus_bytes else None}


# ── what a run records about the loaded model ─────────────────────────────────


def describe_cluster(name: str, view: dict, spec, mode: str = "cluster") -> dict:
    """The loaded model of cluster `name` as its view and its catalog row state it. Unreadable → null + `why`."""
    loaded = (view or {}).get("loaded_model") or {}
    eff = (view or {}).get("effective") or {}
    why: dict[str, str] = {}
    out = {"server": name, "served_name": loaded.get("served_name"), "model": loaded.get("spec"),
           "model_label": loaded.get("label")}
    if spec is None:
        reason = f"the catalog has no row {loaded.get('spec')!r} for the loaded model"
        for key in ("engine", "accelerator", "tp", "quant", "kv_cache_dtype"):
            out[key] = None
            why[key] = reason
        if eff.get("kv_cache_dtype"):
            out["kv_cache_dtype"] = eff["kv_cache_dtype"]
            why.pop("kv_cache_dtype")
        if eff.get("engine"):
            out["engine"] = eff["engine"]
            why.pop("engine")
        out["why"] = why
        return out
    out["model"] = spec.id
    out["model_label"] = spec.label
    out["served_name"] = out["served_name"] or spec.served_name
    profile = spec.profile(mode)
    res = getattr(profile, "resolution", None) if profile is not None else None
    target = getattr(res, "target", None) if res is not None else None
    out["engine"] = (res.engine if res is not None else getattr(profile, "engine", None))
    if out["engine"] is None:
        why["engine"] = f"{spec.label} has no {mode} launch profile"
    out["accelerator"] = target.accelerator.key if target is not None else None
    out["tp"] = target.tp if target is not None else None
    if target is None:
        why["accelerator"] = why["tp"] = f"{spec.label}'s {mode} launch did not resolve to a target on this fleet"
    out["quant"] = getattr(spec, "quant_format", None)
    if out["quant"] is None:
        why["quant"] = f"{spec.label}'s catalog row states no quant format"
    # The dtype the run LAUNCHED: the effective one, else the profile's own — which is what a run that chose none
    # launches (cluster._runs_kv), not a default standing in for an unknown.
    kv = eff.get("kv_cache_dtype") or (getattr(profile, "kv_cache_dtype", None) if profile is not None else None)
    out["kv_cache_dtype"] = kv
    if kv is None:
        why["kv_cache_dtype"] = "neither the running cluster nor its launch profile states a KV dtype"
    out["why"] = why
    return out


def _one_served(b) -> str:
    served = sorted(b.served_models)
    if len(served) != 1:
        raise NothingLoaded(f"{getattr(b, 'cluster', None) or b.host_label or b.url} serves "
                            f"{', '.join(served) or 'no model it names'}: not one loaded model to score")
    return served[0]


def loaded_target(snap, catalog, tasks: dict, nothing_why: str, server: str | None = None) -> dict:
    """The meta of the ONE model serving now (cluster first), or NothingLoaded / ServerRequired."""
    from vllm_lb.state import CLUSTER, VLLM
    live = [b for b in snap.backends if b.live and (b.kind == VLLM or (
        b.kind == CLUSTER and tasks.get(getattr(b, "cluster", None), "chat") == "chat"))]

    def name(b):
        return (b.cluster or "cluster") if b.kind == CLUSTER else (b.host_label or b.url)

    if server is not None:
        live = [b for b in live if name(b) == server]
        if not live:
            raise NothingLoaded(f"{server} is serving no chat model on this gateway")
    if not live:
        raise NothingLoaded(nothing_why)
    clusters = [b for b in live if b.kind == CLUSTER]
    names = sorted({name(b) for b in (clusters or live)})
    if len(names) > 1:
        raise ServerRequired(f"{len(names)} servers are serving ({', '.join(names)}): name one in `server`")
    b = (clusters or live)[0]
    if b.kind == CLUSTER:
        view = snap.cluster_views().get(b.cluster) or {}
        spec = catalog.get(((view.get("loaded_model") or {}).get("spec")))
        meta = describe_cluster(name(b), view, spec)
        if not meta["served_name"]:
            meta["served_name"] = _one_served(b)
        return meta
    why = "a classic vLLM is broker_parrot's to launch: this gateway does not know its launch"
    return {"server": name(b), "served_name": _one_served(b), "model": None, "model_label": None, "engine": "vllm",
            "accelerator": None, "tp": None, "quant": None, "kv_cache_dtype": None,
            "why": {k: why for k in ("model", "accelerator", "tp", "quant", "kv_cache_dtype")}}


# ── the request, through the gateway ──────────────────────────────────────────


def gateway_post(host: str, port_of: Callable[[], int | None]) -> Callable[[dict, float], tuple[int, dict]]:
    """POST /v1/completions to this LB's own listener: the request routes like a client's, by served name. The Host
    header is the loopback name the host allowlist always admits (host_guard.LOOPBACK_NAMES)."""
    connect = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host

    def post(body: dict, timeout: float) -> tuple[int, dict]:
        port = port_of()
        c = http.client.HTTPConnection(connect, port, timeout=timeout)
        try:
            c.request("POST", "/v1/completions", json.dumps(body).encode("utf-8"),
                      {"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
            r = c.getresponse()
            raw = r.read()
        finally:
            c.close()
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"error": {"message": raw[:300].decode("utf-8", "replace")}}
        return r.status, parsed if isinstance(parsed, dict) else {"error": {"message": "not a JSON object"}}
    return post


def _error_words(body) -> str:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or err)[:300]
    return str(err or body.get("message") or "no body")[:300] if isinstance(body, dict) else "no body"


# ── storage ───────────────────────────────────────────────────────────────────


class FidelityStore:
    """One JSON file per run in `directory`. Unreadable files are listed with their reason."""

    def __init__(self, directory: str) -> None:
        self.dir = pathlib.Path(directory)

    def _path(self, run_id) -> pathlib.Path | None:
        return self.dir / f"{run_id}.json" if isinstance(run_id, str) and _ID_RE.match(run_id) else None

    def save(self, run: dict) -> None:
        path = self._path(run.get("id"))
        if path is None:
            raise ValueError(f"{run.get('id')!r} is not a fidelity run id")
        atomic.write_json(str(path), run)

    def _read(self, path: pathlib.Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"id": path.stem, "status": "unreadable", "error": f"{path.name}: {exc}", "started_at": None}
        if not isinstance(data, dict):
            return {"id": path.stem, "status": "unreadable", "error": f"{path.name}: not a JSON object",
                    "started_at": None}
        return data

    def get(self, run_id) -> dict | None:
        path = self._path(run_id)
        if path is None or not path.is_file():
            return None
        return self._read(path)

    def runs(self) -> list[dict]:
        try:
            paths = [p for p in self.dir.iterdir() if p.suffix == ".json" and _ID_RE.match(p.stem)]
        except FileNotFoundError:
            return []
        out = [self._read(p) for p in paths]
        out.sort(key=lambda r: (r.get("started_at") or 0, r.get("id") or ""), reverse=True)
        return out


# ── the bench ─────────────────────────────────────────────────────────────────


class FidelityBench:
    """Starts one run at a time in a background thread, on the operator's call only."""

    def __init__(self, store: FidelityStore, *, post: Callable[[dict, float], tuple[int, dict]],
                 describe: Callable[..., dict], corpus: Callable[[], Corpus] = load_corpus,
                 clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.post = post
        self.describe = describe
        self.corpus = corpus
        self.clock = clock
        self._lock = threading.Lock()
        self._active: dict | None = None

    def start(self, server: str | None = None, asked_by: str = "") -> dict:
        """Begin a run against the loaded model; Busy / NothingLoaded / ServerRequired / ValueError (corpus)."""
        with self._lock:
            if self._active is not None:
                raise Busy(f"run {self._active['id']} is running ({self._active['done_docs']}/"
                           f"{self._active['total_docs']} documents)")
            meta = self.describe(server)
            if not meta.get("served_name"):
                raise NothingLoaded("the loaded model names no served name to send the corpus to")
            corpus = self.corpus()
            now = self.clock()
            run = {"id": f"fid-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(now))}-{secrets.token_hex(2)}",
                   "status": "running", "error": None, "started_at": now, "finished_at": None,
                   "meta": meta, "corpus": corpus.view(), "total_docs": len(corpus.docs), "done_docs": 0,
                   "docs": [], "tokens": 0, "mean_nll": None, "ppl": None, "bits_per_byte": None}
            self.store.save(run)
            self._active = run
        log.info("[fidelity] run %s started on %s (%s) — %s", run["id"], meta.get("served_name"),
                 meta.get("engine"), asked_by or "no caller named")
        started = {k: v for k, v in run.items() if k != "docs"}      # as it stands now, before the thread moves it
        threading.Thread(target=self._run, args=(run, corpus), name=f"fidelity-{run['id']}", daemon=True).start()
        return started

    def _run(self, run: dict, corpus: Corpus) -> None:
        meta = run["meta"]
        try:
            for d in corpus.docs:
                status, body = self.post({"model": meta["served_name"], "prompt": d.text, **REQUEST}, DOC_TIMEOUT_S)
                if status != 200:
                    raise FidelityError(f"HTTP {status} from the gateway for document {d.id}: {_error_words(body)}")
                score = score_reply(body, engine=meta.get("engine"), served_name=meta["served_name"], doc_id=d.id)
                with self._lock:
                    run["docs"].append({"id": d.id, "bytes": d.bytes, **score})
                    run["done_docs"] = len(run["docs"])
            run.update(_totals(run["docs"], sum(d.bytes for d in corpus.docs)), status="done")
        except Exception as exc:                              # every failure ends the run, in its own words
            run.update(status="failed", error=str(exc) if isinstance(exc, FidelityError)
                       else f"{type(exc).__name__}: {exc}", mean_nll=None, ppl=None, bits_per_byte=None)
            log.warning("[fidelity] run %s failed: %s", run["id"], run["error"])
        finally:
            run["finished_at"] = self.clock()
            try:
                self.store.save(run)
            except OSError as exc:
                log.error("[fidelity] run %s could not be saved: %s", run["id"], exc)
            with self._lock:
                self._active = None
        if run["status"] == "done":
            log.info("[fidelity] run %s done: NLL %.4f · ppl %.3f · %d tokens", run["id"], run["mean_nll"],
                     run["ppl"], run["tokens"])

    def view(self) -> dict:
        with self._lock:
            active = None if self._active is None else {k: v for k, v in self._active.items() if k != "docs"}
            active_id = active["id"] if active else None
        runs = []
        for r in self.store.runs():
            if r.get("status") == "running" and r.get("id") != active_id:
                r = {**r, "status": "interrupted", "error": "the gateway restarted mid-run: not resumed"}
            runs.append(r)
        try:
            corpus, corpus_error = self.corpus().view(), ""
        except (OSError, ValueError) as exc:
            corpus, corpus_error = None, str(exc)
        return {"active": active, "runs": runs, "corpus": corpus, "corpus_error": corpus_error,
                "dir": str(self.store.dir)}


# ── the paired comparison ─────────────────────────────────────────────────────


def compare(a: dict, b: dict) -> dict:
    """ΔNLL = NLL(B) − NLL(A) over the documents both runs tokenised identically, with its 95 % bootstrap interval."""
    def no(reason):
        return {"comparable": False, "reason": reason}

    for name, r in (("A", a), ("B", b)):
        if r.get("status") != "done":
            return no(f"{name} ({r.get('id')}) is not a finished run ({r.get('status')})")
    if (a.get("corpus") or {}).get("sha256") != (b.get("corpus") or {}).get("sha256"):
        return no("the runs scored a different corpus")
    by_id = {d["id"]: d for d in b.get("docs") or []}
    pairs, mismatched = [], 0
    for da in a.get("docs") or []:
        db = by_id.get(da["id"])
        if db is None:
            continue
        if db["tokens_sha"] != da["tokens_sha"] or db["tokens"] != da["tokens"]:
            mismatched += 1
            continue
        pairs.append((da["tokens"], da["sum_nll"], db["sum_nll"]))
    if not pairs:
        return {**no("no document was tokenised identically in both runs (another tokenizer?)"),
                "mismatched_docs": mismatched}
    tokens = sum(p[0] for p in pairs)
    sa, sb = sum(p[1] for p in pairs), sum(p[2] for p in pairs)
    rng = random.Random(BOOTSTRAP_SEED)
    k = len(pairs)
    samples = []
    for _ in range(BOOTSTRAP_REPS):
        t = da_ = db_ = 0.0
        for _ in range(k):
            p = pairs[rng.randrange(k)]
            t += p[0]
            da_ += p[1]
            db_ += p[2]
        samples.append((db_ - da_) / t)
    samples.sort()

    def at(q):
        return samples[min(len(samples) - 1, max(0, math.floor(q * (len(samples) - 1))))]

    lo, hi = round(at(0.025), 4), round(at(0.975), 4)
    d = round((sb - sa) / tokens, 4)
    return {"comparable": True, "reason": None, "a": a.get("id"), "b": b.get("id"), "tokens": tokens,
            "matched_docs": k, "mismatched_docs": mismatched,
            "nll_a": round(sa / tokens, 4), "nll_b": round(sb / tokens, 4),
            "ppl_a": round(math.exp(sa / tokens), 3), "ppl_b": round(math.exp(sb / tokens), 3),
            "d_nll": d, "ci95": [lo, hi], "same_quality": lo <= 0 <= hi, "reps": BOOTSTRAP_REPS}


# ── the HTTP surface (proxy.py routes here, one block) ────────────────────────


def answer_start(bench: FidelityBench, raw: bytes, content_type: str | None, asked_by: str) -> tuple[int, dict]:
    """POST /lb/fidelity/run `{server?}` → 202 {run}; 415 not JSON; 409 busy / nothing loaded; 400 server needed."""
    if (content_type or "").split(";", 1)[0].strip().lower() != "application/json":
        return 415, {"error": {"type": "unsupported_media_type",
                               "message": "a fidelity run is a POST with `Content-Type: application/json` ({} or "
                                          "{\"server\": <name>}); any other type is one a cross-site page could send"}}
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = None
    if not isinstance(body, dict) or ("server" in body and not isinstance(body["server"], (str, type(None)))):
        return 400, {"error": {"type": "invalid_request", "message": "the body is {} or {\"server\": <name>}"}}
    try:
        run = bench.start(body.get("server"), asked_by)
    except Busy as exc:
        return 409, {"error": {"type": "fidelity_busy", "message": str(exc)}}
    except NothingLoaded as exc:
        return 409, {"error": {"type": "fidelity_nothing_loaded", "message": str(exc)}}
    except ServerRequired as exc:
        return 400, {"error": {"type": "fidelity_server_required", "message": str(exc)}}
    except (OSError, ValueError) as exc:
        return 500, {"error": {"type": "fidelity_unavailable", "message": str(exc)}}
    return 202, {"run": run}


def answer_compare(bench: FidelityBench, query: str) -> tuple[int, dict]:
    """GET /lb/fidelity/compare?a=<run>&b=<run> → the paired comparison."""
    q = urllib.parse.parse_qs(query or "")
    a_id, b_id = (q.get("a") or [""])[0], (q.get("b") or [""])[0]
    if not a_id or not b_id:
        return 400, {"error": {"type": "invalid_request", "message": "name two runs: ?a=<run id>&b=<run id>"}}
    runs = {}
    for rid in (a_id, b_id):
        r = bench.store.get(rid)
        if r is None:
            return 404, {"error": {"type": "not_found", "message": f"no fidelity run {rid!r}"}}
        runs[rid] = r
    return 200, compare(runs[a_id], runs[b_id])
