"""The benchmark swarm's sessions, against the fleet's real capture directories."""
from __future__ import annotations

from usage_mcp import swarm
from usage_mcp.readonly import Sources


def test_the_real_captures_are_found_and_bracketed(src: Sources):
    found = swarm.sessions(src)
    assert len(found) >= 5, [s["label"] for s in found]
    assert all(s["label"].startswith("S-") for s in found)
    bracketed = [s for s in found if s.get("started_at") and s.get("ended_at")]
    assert bracketed, "no session has both capture-file mtimes"
    for s in bracketed:
        assert s["ended_at"] >= s["started_at"], s["label"]


def test_the_newest_session_matches_benchq_session_json(src: Sources):
    """The one session the LB still holds state for must agree with the mtime bracket."""
    found = swarm.sessions(src)
    from_json = [s for s in found if "session.json" in (s.get("bracket_source") or "")]
    assert len(from_json) == 1, [s["bracket_source"] for s in found]
    latest = from_json[0]
    assert latest is found[-1], (latest["label"], found[-1]["label"])
    assert latest.get("served_model"), latest
    assert latest.get("guards"), latest
    # the label encodes the local start time, so it must agree with the recorded one
    assert abs(swarm._from_label(latest["label"]) - latest["started_at"]) < 2.0, latest


def test_sessions_are_oldest_first(src: Sources):
    starts = [s["started_at"] for s in swarm.sessions(src) if s.get("started_at")]
    assert starts == sorted(starts)


def test_a_session_is_measured_from_the_lbs_own_windows(src: Sources):
    """Not from anything the session says about itself."""
    found = swarm.sessions(src)
    measured = [(s, swarm.measure(src, s, days=99)) for s in found]
    worked = [(s, m) for s, m in measured if m.get("measured")]
    assert worked, [m.get("why") for _, m in measured]
    for s, m in worked:
        assert m["windows"] >= 1 and m["minutes"] > 0, s["label"]
        assert m["gen_tokens"] >= 0
        if m["gen_tokens"] > 0:
            assert m["gen_tok_s"] and m["gen_tok_s"] > 0, s["label"]
            assert m["mean_concurrency"] and m["mean_concurrency"] >= 1.0, s["label"]
        assert m["model_choices"], s["label"]


def test_a_session_with_no_windows_says_why_instead_of_raising(src: Sources):
    out = swarm.measure(src, {"label": "S-19990101-000000",
                              "started_at": 915148800.0, "ended_at": 915152400.0}, days=99)
    assert out["measured"] is False and out["why"]


def test_a_session_with_no_start_is_reported_not_raised(src: Sources):
    assert swarm.measure(src, {"label": "S-x"}, days=1)["measured"] is False


def test_the_between_session_spread_caveat_names_the_real_range():
    assert "2.36" in swarm.SPREAD and "11.64" in swarm.SPREAD
    assert "configuration" in swarm.SPREAD


def test_client_accounting_is_read_but_its_dollars_are_labelled_fiction(src: Sources):
    """The client counted tokens at Anthropic LIST prices for traffic the GB10 boxes served."""
    found = swarm.sessions(src)
    accounted = [(s["label"], swarm.client_accounting(src, s["label"])) for s in found]
    recorded = [(label, a) for label, a in accounted if a["recorded"]]
    assert recorded, [a.get("why") for _, a in accounted]
    for label, a in recorded:
        assert a["input_tokens"] is not None, label
        assert "fiction" in a["note"].lower() or "FICTION" in a["note"], label
        assert "total_cost_usd_IS_FICTION" in a, label
        assert a["client_models"], label
    # every recorded session ran against the LOCAL gateway, so no client model is an Anthropic id
    for label, a in recorded:
        for model in a["client_models"]:
            assert not model.startswith("claude-"), (label, model)


def test_the_arithmetic_behind_the_fiction_label_holds(src: Sources):
    """total_cost_usd is in+out at $4/$20 per MTok - list price for a model that did not serve it."""
    for s in swarm.sessions(src):
        a = swarm.client_accounting(src, s["label"])
        cost = a.get("total_cost_usd_IS_FICTION")
        if not a["recorded"] or not cost or a["cache_read_input_tokens"]:
            continue
        expected = a["input_tokens"] * 4e-6 + a["output_tokens"] * 20e-6
        if abs(expected - cost) < 1e-6:
            assert "unknown" in a["cost_basis"] or a["cost_basis"] == [], a
            return
    import pytest
    pytest.skip("no cache-free session with a recorded cost to check the arithmetic on")
