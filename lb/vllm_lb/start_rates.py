"""THE START SUCCESS RATE of every model chooser row: of the starts the LB LAUNCHED for one checkpoint over the last
seven days, how many reached serving — per day, and over the seven.

WHY. The operator picks a model and holds Start; a start that dies costs a cold start (5-15 minutes) and the boxes
it reserved, taken from their fleet meanwhile. Before this, whether a row "fails to start often" was knowledge in
someone's head (2026-10-10/11: the vLLM Flash-Next twin killed by tp-rank's memory guard four starts out of five).
The chooser row now carries it as figures: `/lb/status.start_rates.rows[<row>]`, drawn by gateway.js beside each row.

THE DATA is the cluster boot log the LB already keeps (`perflog.BootRecorder`): `boots-YYYY-MM-DD.jsonl` in
`Config.perf_dir` (LB_PERF_DIR, on the state volume), one JSON line per start, one file per UTC day, read back
through the one reader `/lb/perf` and `/lb/sessions` use (`perflog.PerfLog`). The fields read here:
  t_start            when the start was asked (epoch seconds) — its DAY
  generation         the Start it belongs to (one per press of Start; a relaunch keeps it)
  attempt            1 = the operator's own launch, 2.. = the LB's relaunches of a hung start (absent before
                     2026-10-07: 1)
  config             what was launched: `model_choice` (the catalog id) and `uncensored` (the twin; null before the
                     twins existed = the original)
  states             {state: seconds since asked} for every planner state the start passed, first entry each
  outcome            serving | failed | stopped — how the line closed (`perflog.BOOT_ENDS`; a hang closes `failed`)

ONE ATTEMPT IS ONE LAUNCH — a line during which the LB sent the model's ranks up: the operator's Start, and each
relaunch the LB makes on its own (a hung start torn down and launched again: `attempt` 2, 3; a pair whose box
rebooted under it, launched again through every gate). The question is "does this model start", and each launch
is one answer: a model that hangs once and serves on the retry failed one start of two, which is what happened.
  * Launched = the line reached `starting`: the planner says it on the tick that sends the ranks up and while they
    come up, and never before (cluster.plan: every gate — the checkpoint, the reservation, the RAM — lies between).
  * SUCCESS: the outcome is `serving` — the head answered /health serving the chosen model (planner `running`).
  * FAILURE: the outcome is `failed` after the launch — a rank exited (the memory guard's kill among them), the
    start hung (relaunched or halted: `stalled`), a box ran out of memory under it, the head served another model.
  * `stopped` after the launch — Stop, or a newer Start, before it served — has NO VERDICT: the model neither
    started nor failed. Counted apart, never in the rate.
  * `refused`: a line that ended `failed` WITHOUT reaching `starting` — the LB refused to launch it (its own build
    check, a RAM need the boxes can never meet) or broker_parrot ended a reservation first. Nothing was launched,
    so it is not an attempt; it is counted apart. (A refusal on a Start's very first tick from rest writes no line
    at all — BootRecorder opens a line only in a state a start passes — so `refused` counts what the log holds.)
  * A line that ended `stopped` without reaching `starting` (the operator stopped it while it waited for the boxes)
    says nothing about the model and is not counted at all.
  * READ AGAIN, NOT A START (`perflog.read_again` — one rule, which /lb/sessions keeps too): a line that OPENED in
    `starting` (its first state) while an EARLIER line of the same start exists — same generation, attempt and
    checkpoint. That is the LB finding a pair already up: a redeployed LB beside a serving pair (`starting` →
    `running` 7 s later), or a serving head that missed a /health probe and was read again (generation 173,
    2026-10-09, then killed while serving — a failure of a pair that had started). Without an earlier line it IS
    the start: an LB restarted in the middle of a start forgets its open line, and the line the new LB writes is
    the only record of how that start ended (generation 138, 2026-10-05). The earlier line is looked for in every
    line the log keeps, not only this week's: a pair serving for a month is not started today by today's redeploy.
    And a line that SAYS it is one (`read_again: true`): a node the LB requests, which that LB had seen serve under
    the same intent before a probe went unanswered (2026-10-12: site-b's intent #14 started before any line was
    written, so no earlier line could ever mark its re-reads).

A NODE THE LB REQUESTS (control local) writes the same lines (cluster_loop `_tick_local`): its generation is the
released intent's id, its row the fleet choice the intent selected — the same `<id>` / `<id>@uncensored` key.

THE MODEL is the chooser row: `catalog.availability_key(model_choice, uncensored)` — `<id>` or `<id>@uncensored`,
the key the rows, `cluster.setups` and `setup_targets` already share. The ENGINE is the id's own (catalog
`launch.cluster.engine`): a model on SGLang and on vLLM is two catalog ids, so two rows, never one rate.

THE DAYS are UTC — the boot log's own boundary (`perflog.day_of`; a day's lines are one `boots-<day>.jsonl`, so a
point can be checked with `wc -l`). The window is today and the six days before it. A day on which the row had no
attempt has NO POINT (null) — never 0 %, never a value carried from a neighbour; a row with no attempt in the
window has no rate at all, and the page says "no starts in 7 d".

Pure: no clock, no disk. The caller hands in every line the log keeps, oldest first, and `now`.
"""
from __future__ import annotations

from typing import Iterable

from vllm_lb.catalog import availability_key
from vllm_lb.perflog import LAUNCHED, boot_start, day_of, read_again

#: The window: today and the six days before it, UTC.
WINDOW_DAYS = 7
DAY_S = 86400.0
#: What a line says about its row's starts.
OK, FAILED, STOPPED, REFUSED = "ok", "failed", "stopped", "refused"
#: A launched line's outcome (perflog.BOOT_ENDS: what the recorder closes a line with) → its verdict.
_AFTER_LAUNCH = {"serving": OK, "failed": FAILED, "stopped": STOPPED}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _day(t) -> str | None:
    """The UTC day of a time, or None for one no calendar holds."""
    try:
        return day_of(t) if _number(t) else None
    except (OverflowError, OSError, ValueError):
        return None


def _read(line) -> tuple | None:
    """(row key, the start it belongs to, its day) — or None for a line the rules cannot read: no model, no
    generation, no start time, no states, an outcome the recorder never writes."""
    if not isinstance(line, dict):
        return None
    config = line.get("config")
    model = config.get("model_choice") if isinstance(config, dict) else None
    generation, states, day = line.get("generation"), line.get("states"), _day(line.get("t_start"))
    if (not isinstance(model, str) or not model or not isinstance(generation, int) or isinstance(generation, bool)
            or day is None or not isinstance(states, dict) or not states
            or not all(_number(v) for v in states.values()) or line.get("outcome") not in _AFTER_LAUNCH):
        return None
    start = boot_start(line)                              # (generation, attempt, model, twin)
    return availability_key(model, start[3]), start, day


def _verdict(line: dict, earlier: set) -> str | None:
    """OK, FAILED, STOPPED or REFUSED — or None: not a start of its row (an engine already up read again, or a start
    stopped before anything was launched). `earlier`: the starts of every line before this one."""
    if LAUNCHED not in line["states"]:
        return REFUSED if line["outcome"] == "failed" else None
    if read_again(line, earlier):
        return None
    return _AFTER_LAUNCH[line["outcome"]]


def start_rates(lines: Iterable[dict], *, now: float, days: int = WINDOW_DAYS) -> dict:
    """`/lb/status.start_rates`, less what only the reader knows (malformed lines, an unreadable directory):

        {"tz": "UTC", "days": [<oldest day>, …, <today>],
         "rows": {<row>: {"ok": served, "n": attempts, "stopped": n, "refused": n,
                          "days": [null | [served, attempts], … one per day]}},
         "unread": lines the rules could not read}

    `lines`: every line the boot log keeps, OLDEST FIRST (`PerfLog.index(days=None)`) — the earlier lines are what
    tells a pair read again from a start. A row appears once a line of the window counts for it. `unread` counts
    the lines of the window, and those with no day at all, that the rules could not read — never a guess at them."""
    window = [day_of(now - k * DAY_S) for k in range(days - 1, -1, -1)]
    slot = {d: i for i, d in enumerate(window)}
    rows: dict[str, dict] = {}
    earlier: set = set()
    unread = 0
    for line in lines:
        read = _read(line)
        if read is None:
            day = _day(line.get("t_start")) if isinstance(line, dict) else None
            unread += 1 if day is None or day in slot else 0
            continue
        key, start, day = read
        said = _verdict(line, earlier)
        earlier.add(start)
        if said is None or day not in slot:
            continue
        r = rows.setdefault(key, {"ok": 0, "n": 0, "stopped": 0, "refused": 0, "days": [None] * days})
        if said in (STOPPED, REFUSED):
            r[said] += 1
            continue
        cell = r["days"][slot[day]] or [0, 0]
        cell[1] += 1
        r["n"] += 1
        if said == OK:
            cell[0] += 1
            r["ok"] += 1
        r["days"][slot[day]] = cell
    return {"tz": "UTC", "days": window, "rows": rows, "unread": unread}
