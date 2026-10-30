"""A real two-box start, as `tp-rank logs` would have returned it at any moment of it.

`fixtures/qwen38-start-2026-09-15-rank{0,1}.log` are the ranks' own logs of the start pressed at
2026-09-15T19:55:44Z (max_num_seqs=1), captured with `docker logs -f -t`: the head from its first line
to the LB's first successful probe (20:08:43), rank 1 to the end of its engine init (20:07:42).
tp-rank runs `docker logs --tail N` WITHOUT `-t`, so a line reaches the LB as everything after its
docker timestamp — carriage returns included (tqdm writes one before every counter it prints).
"""
from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).with_name("fixtures")
HEAD = "qwen38-start-2026-09-15-rank0.log"
RANK1 = "qwen38-start-2026-09-15-rank1.log"
DAY = "2026-09-15T"


def stamped(name: str) -> list[tuple[str, str]]:
    """(docker timestamp, the line as tp-rank returns it) for every line of a fixture, in order."""
    out = []
    for raw in (FIXTURES / name).read_bytes().decode("utf-8", "replace").split("\n"):
        if raw:
            stamp, _, body = raw.partition(" ")
            out.append((stamp, body))
    return out


def log_at(name: str, when: str, day: str = DAY) -> str:
    """What `docker logs` held at `when` (UTC `HH:MM:SS[.fraction]` on `day`, 2026-09-15 unless said): every
    line docker had stamped by then, without its stamp. Plain string order is time order for these stamps.
    A `\\r`-drawn bar is ONE line here, stamped when its last frame ended — exactly as docker holds it."""
    key = day + when
    return "".join(f"{body}\n" for stamp, body in stamped(name) if stamp <= key)


def tail(text: str, n: int) -> str:
    """`docker logs --tail n`: the last n lines. Split on \\n only — a tqdm \\r is not a line end."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return "".join(f"{line}\n" for line in lines[-n:])


#: The head's own log of the 2026-10-02 generation-109 start — Qwen3.8-27B-DFlash2, a `dflash` drafter at
#: K=7, vLLM 0.29.0, image myllmbox/qwen38-flash-next-cluster-vllm:v5 — read off box-a with `docker logs`
#: and cut after the head's "Application startup complete". UNSTAMPED, which is exactly what reaches the
#: LB: tp-rank runs `docker logs --tail N` WITHOUT `-t`, so a line arrives as its body, tqdm's carriage
#: returns intact. The capture bars carry no time of their own; the engine times the step itself
#: ("Graph capturing finished in 39 secs, took 1.56 GiB").
DFLASH2_HEAD = "qwen38-dflash2-start-2026-10-02-rank0.log"


def body_lines(name: str) -> list[str]:
    """Every `\\n`-line of an UNSTAMPED fixture, in order, as `tp-rank logs` hands it to the LB."""
    text = (FIXTURES / name).read_bytes().decode("utf-8", "replace")
    return [line for line in text.split("\n") if line]


def grown_to(lines: list[str], last: int) -> str:
    """What `docker logs` held once the head had printed `lines[:last + 1]`: one of the growing prefixes
    the LB reads, one per `Config.cluster_logs_every_s`."""
    return "".join(f"{line}\n" for line in lines[: last + 1])
