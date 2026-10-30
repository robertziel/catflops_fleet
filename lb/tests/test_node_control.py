"""THE FILES BETWEEN THE LB AND A ONE-HOST NODE'S HOST (fleet control `local`; vllm_lb/node_control.py).

The LB writes ONE intent — {id, at, state up|down, force, launch: the figures it moves} — whole or not at all, and
reads it back as the node's desired state; the host's applier reports back (heartbeat, what it applied and how).
`force` is the operator's word that a relaunch may cut the requests the node is serving — without it the applier
refuses one while any is in flight. Pinned here: the envelope is exact (a missing, extra, malformed or out-of-domain
key is refused, naming it — the launch keys themselves are fleet.validate_launch's), ids only grow, and a report says
whether anyone is listening, in seconds.
"""
from __future__ import annotations

import json
import os

import pytest

from vllm_lb import node_control as nc

AT = 1_791_742_000.0


def intent(**over) -> dict:
    return {"id": 1, "at": AT, "state": "up", "force": False, "launch": {"max_num_seqs": 2}, **over}


# ── the envelope ────────────────────────────────────────────────────────────────────────────────


def test_an_intent_is_exactly_its_five_keys():
    assert nc.parse_intent(intent()) == {"id": 1, "at": AT, "state": "up", "force": False,
                                         "launch": {"max_num_seqs": 2}}
    assert nc.parse_intent(intent(state="down", launch={}, force=True))["force"] is True


@pytest.mark.parametrize("raw, says", [
    ("not an object", "one JSON object"),
    ({k: v for k, v in intent().items() if k != "state"}, "missing state"),
    (intent(image="vllm/vllm-openai@sha256:" + "0" * 64), "unknown image"),
    (intent(gpu=0), "unknown gpu"),
    (intent(id=0), "id"),
    (intent(id=True), "id"),
    (intent(id="3"), "id"),
    (intent(at="now"), "at"),
    (intent(at=float("nan")), "at"),
    (intent(state="restart"), "state"),
    (intent(force="yes"), "force"),
    (intent(force=None), "force"),
    (intent(launch=["max_num_seqs", 2]), "launch"),
    (intent(launch={"max_num_seqs; rm -rf /": 2}), "launch key"),
])
def test_anything_else_is_refused_naming_it(raw, says):
    with pytest.raises(ValueError, match=says):
        nc.parse_intent(raw)


# ── the LB's store: written whole, read back, ids only grow ─────────────────────────────────────


def test_the_store_reads_nothing_until_an_intent_is_written_then_reads_it_back(tmp_path):
    store = nc.IntentStore(str(tmp_path / nc.INTENT))
    assert store.read() is None, "no intent: the fleet's boot default stands"
    first = store.write("up", {"max_num_seqs": 2}, now=AT)
    assert first == {"id": 1, "at": AT, "state": "up", "force": False, "launch": {"max_num_seqs": 2}}
    assert store.read() == first
    assert json.loads((tmp_path / nc.INTENT).read_text()) == first
    second = store.write("down", {}, now=AT + 5, force=True)
    assert (second["id"], store.read()["state"], store.read()["force"]) == (2, "down", True)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")], "no half-written file left behind"


def test_an_unreadable_intent_is_said_never_read_as_none_and_a_write_replaces_it(tmp_path):
    (tmp_path / nc.INTENT).write_text("{not json")
    store = nc.IntentStore(str(tmp_path / nc.INTENT))
    with pytest.raises(ValueError, match="intent.json"):
        store.read()
    assert store.write("up", {}, now=AT)["id"] == 1
    wish = nc.IntentStore(str(tmp_path / "cluster.json"))
    (tmp_path / "cluster.json").write_text('{"mode": "classic", "generation": 3}')
    with pytest.raises(ValueError, match="cluster.json takes id"):
        wish.read()          # a pair's switch left behind is said, never read as a wish


def test_the_directory_is_the_lbs_state_dir_beside_its_cluster_switch():
    assert nc.node_dir("/data/cluster.json") == "/data/node"


# ── the applier's report, as the LB reads it ────────────────────────────────────────────────────


def applied(**over) -> dict:
    return {"id": 3, "at": AT, "state": "up", "force": False, "done_at": AT + 4, "exit": 0, "stderr": "",
            "refused": "", **over}


def test_a_fresh_heartbeat_is_listening_and_the_applied_intent_is_carried(tmp_path):
    nc.write_report(str(tmp_path), nc.report_doc(heartbeat=AT, host="node-a-host", applied=applied(),
                                                 container={"status": "running", "exit_code": 0, "restarts": 0},
                                                 log="INFO started"))
    got = nc.read_report(str(tmp_path), now=AT + 20)
    assert (got["listening"], got["heartbeat_age_s"], got["host"]) == (True, 20.0, "node-a-host")
    assert got["applied"] == applied() and got["container"]["status"] == "running" and got["log"] == "INFO started"
    assert got["error"] == ""


def test_a_stale_heartbeat_is_no_applier_listening_with_its_age(tmp_path):
    nc.write_report(str(tmp_path), nc.report_doc(heartbeat=AT, host="h", applied=None, container=None, log=None))
    got = nc.read_report(str(tmp_path), now=AT + nc.APPLIER_STALE_S + 1)
    assert got["listening"] is False and got["heartbeat_age_s"] == nc.APPLIER_STALE_S + 1
    assert "no applier listening" in got["error"]


def test_an_applier_running_a_long_bring_up_is_still_listening_until_its_own_limit(tmp_path):
    """A pass writes `applying` (and a heartbeat) BEFORE it runs up.sh, which may pull an image for minutes: the panel
    then says what it is doing, not "no applier listening" — until the command's own limit has passed."""
    busy = {"id": 4, "at": AT, "state": "up", "force": False, "since": AT}
    nc.write_report(str(tmp_path), nc.report_doc(heartbeat=AT, host="h", applied=None, container=None, log=None,
                                                 applying=busy))
    got = nc.read_report(str(tmp_path), now=AT + 600)
    assert (got["listening"], got["applying"], got["error"]) == (True, busy, "")
    late = nc.read_report(str(tmp_path), now=AT + nc.UP_TIMEOUT_S + nc.APPLIER_STALE_S + 1)
    assert late["listening"] is False and "no applier listening" in late["error"]


def test_no_report_at_all_says_no_applier_has_run(tmp_path):
    got = nc.read_report(str(tmp_path), now=AT)
    assert (got["listening"], got["applied"]) == (False, None)
    assert "no applier has reported" in got["error"]


def test_an_unreadable_report_is_said(tmp_path):
    (tmp_path / nc.REPORT).write_text("[1, 2")
    got = nc.read_report(str(tmp_path), now=AT)
    assert got["listening"] is False and "report.json" in got["error"]


def test_the_intent_and_its_report_are_one_identity():
    """The applier applies an intent once: by its (id, at), so an id the LB restarted at 1 after an unreadable file
    is still new."""
    assert nc.same_intent(intent(), applied(id=1, at=AT)) is True
    assert nc.same_intent(intent(), applied(id=1, at=AT + 1)) is False
    assert nc.same_intent(intent(), None) is False
