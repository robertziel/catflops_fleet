"""The window reader, checked against the fleet's own recorded minutes.

Every number asserted here was recorded by the LB, not written by this suite. The strongest check
is `test_this_reader_agrees_with_the_lbs_own_reader`: the LB counted the same files independently
and published the counts at `GET /lb/perf`, so the two readers must agree file by file - including
on the one malformed line in perf-2026-10-02.jsonl (line 470, a run of NUL bytes from a crash
mid-append).
"""
from __future__ import annotations

import json

import pytest

from pathlib import Path

from usage_mcp import caveats, perf
from usage_mcp.readonly import Sources, SourceUnavailable

DATA = Path(__file__).resolve().parent / "data"
LB_COUNTS = json.loads((DATA / "lb-perf-filecounts-2026-10-03.json").read_text())


def test_the_recorder_has_written_perf_days(src: Sources):
    windows, meta = perf.load_windows(src, days=99)
    assert meta["days"], "no recorded perf days under the state directory"
    assert windows, "recorded days but no windows in them"


def test_this_reader_agrees_with_the_lbs_own_reader(src: Sources):
    """The LB parsed the same files and published its counts; this reader must match them."""
    checked = 0
    for entry in LB_COUNTS["files"]:
        day = entry["file"].removeprefix("perf-").removesuffix(".jsonl")
        try:
            windows, meta = perf.load_windows(src, day=day)
        except SourceUnavailable:
            continue                                  # that day has since rotated out: not a bug
        if entry["file"] == LB_COUNTS["files"][-1]["file"]:
            # the newest file was still being appended to when the LB counted it
            assert len(windows) >= entry["windows"], (entry, len(windows))
            continue
        assert len(windows) == entry["windows"], (entry["file"], len(windows), entry["windows"])
        assert meta["malformed_lines"] == entry["malformed"], (entry["file"], meta["malformed"])
        checked += 1
    assert checked >= 1, "no closed day to cross-check against the LB's counts"


def test_the_one_known_malformed_line_is_counted_not_fatal(src: Sources):
    """perf-2026-10-02.jsonl line 470 is NUL bytes. The day's other 596 windows must survive."""
    bad = next((m for m in LB_COUNTS["malformed"]), None)
    if bad is None:
        pytest.skip("the captured /lb/perf reported no malformed line to check")
    day = bad["file"].removeprefix("perf-").removesuffix(".jsonl")
    try:
        windows, meta = perf.load_windows(src, day=day)
    except SourceUnavailable:
        pytest.skip(f"{bad['file']} has rotated out")
    assert meta["malformed_lines"] >= 1
    assert meta["malformed"][0]["line"] == bad["line"], meta["malformed"]
    assert len(windows) > 500, len(windows)


def test_totals_for_a_closed_day_are_the_sum_of_its_windows(src: Sources, closed_perf_day: str):
    windows, _ = perf.load_windows(src, day=closed_perf_day)
    folded = perf.fold(windows)
    for key in ("gen_tokens", "prompt_tokens", "prompt_computed_tokens", "requests_finished"):
        assert getattr(folded, key) == pytest.approx(sum(w.get(key) or 0 for w in windows))
    assert folded.windows == len(windows)
    assert folded.totals()["gen_tokens"] > 0


def test_the_prefix_cache_shows_up_as_prompt_tokens_above_what_was_computed(
        src: Sources, closed_perf_day: str):
    """Not a tautology: it is the fleet's measured prefix-cache behaviour on a real day."""
    windows, _ = perf.load_windows(src, day=closed_perf_day)
    f = perf.fold(windows)
    assert f.prompt_tokens > f.prompt_computed_tokens, (f.prompt_tokens, f.prompt_computed_tokens)


def test_an_impossible_prefix_hit_rate_is_refused_not_returned(src: Sources):
    """Real defect, real data: 39 windows recorded on 2026-10-03 have hits ABOVE the queried total.

    Until lb 9c32807 the probe recorded SGLang's `mode="input"` series as the queried total and
    that series excludes the hits, so those windows imply shares up to 627%. The reader must hand
    back no rate and say why - a caller reading 627% is worse than a caller reading nothing.
    """
    windows, _ = perf.load_windows(src, days=99)
    broken = [w for w in windows
              if (w.get("prefix_query_tokens") or 0) > 0
              and (w.get("prefix_hit_tokens") or 0) > (w.get("prefix_query_tokens") or 0)]
    if not broken:
        pytest.skip("no window with hits above the queried total is still on disk")
    f = perf.fold(broken)
    assert f.prefix_hit_tokens > f.prefix_query_tokens
    assert f.prefix_hit_rate() is None, "an impossible share was returned as a number"
    note = f.prefix_hit_note()
    assert note and "9c32807" in note and "EXCLUDES" in note
    totals = f.totals()
    assert totals["prefix_hit_rate"] is None
    assert totals["prefix_hit_rate_unusable"] == note
    # the raw counts are still handed back, so a caller can see what happened
    assert totals["prefix_hit_tokens"] > 0 and totals["prefix_query_tokens"] > 0


def test_a_sound_day_still_yields_a_share(src: Sources, closed_perf_day: str):
    windows, _ = perf.load_windows(src, day=closed_perf_day)
    f = perf.fold(windows)
    rate = f.prefix_hit_rate()
    assert rate is not None and 0.0 < rate <= 1.0, rate
    assert f.prefix_hit_note() is None
    assert "prefix_hit_rate_unusable" not in f.totals()


def test_a_stated_itl_basis_wins_and_a_disagreement_is_reported():
    """The LB publishes `itl_basis` on the scrape since 9c32807; it is authoritative when present."""
    agreeing = caveats.itl_semantics(1000.0, 998.0, basis=caveats.ITL_PER_TOKEN)
    assert agreeing["means"] == caveats.ITL_PER_TOKEN
    assert agreeing["stated_by_the_engine"] == caveats.ITL_PER_TOKEN
    assert agreeing["disagreement"] is None
    assert agreeing["per_token_divisor"] == 1.0

    stepwise = caveats.itl_semantics(3000.0, 1000.0, basis=caveats.ITL_PER_STEP)
    assert stepwise["means"] == caveats.ITL_PER_STEP and stepwise["per_token_divisor"] == 3.0

    conflicting = caveats.itl_semantics(3000.0, 1000.0, basis=caveats.ITL_PER_TOKEN)
    assert conflicting["means"] == caveats.ITL_PER_TOKEN, "the stated basis must win"
    assert conflicting["disagreement"] and "3.000" in conflicting["disagreement"]

    # measured-only, which is the case for every window on disk today
    measured = caveats.itl_semantics(3000.0, 1000.0)
    assert measured["means"] == caveats.ITL_PER_STEP
    assert measured["stated_by_the_engine"] is None and measured["disagreement"] is None


def test_the_basis_vocabulary_matches_the_lbs_own(src: Sources):
    """probe.py reads `ITL_PER_TOKEN, ITL_PER_STEP = "token", "step"`; drifting apart would confuse."""
    assert caveats.ITL_PER_TOKEN == "token" and caveats.ITL_PER_STEP == "step"
    probe = (Path(__file__).resolve().parents[2] / "lb" / "vllm_lb" / "probe.py")
    if not probe.exists():
        pytest.skip("the LB source is not mounted in this run")
    line = next((ln for ln in probe.read_text().splitlines()
                 if ln.startswith("ITL_PER_TOKEN, ITL_PER_STEP")), None)
    assert line, "probe.py no longer declares the pair on one line - re-check by hand"
    assert '"token", "step"' in line, line


def test_a_window_carrying_a_basis_is_believed(src: Sources, closed_perf_day: str):
    """Recorded windows have no `itl_basis` yet; when the LB starts persisting one, it is used."""
    windows, _ = perf.load_windows(src, day=closed_perf_day)
    assert perf.fold(windows).itl_bases == set(), "a window already carries itl_basis - good"
    tagged = [dict(w, itl_basis="token") for w in windows]
    assert perf.fold(tagged).itl_bases == {"token"}
    mixed = perf.fold([dict(windows[0], itl_basis="token"), dict(windows[1], itl_basis="step")])
    assert mixed.latency()["itl_semantics"]["disagreement"], "a mixed basis must be flagged"


def test_histogram_buckets_sum_to_the_sample_count(src: Sources, closed_perf_day: str):
    """The LB writes de-cumulated `le` buckets, which is what `percentiles` assumes."""
    windows, _ = perf.load_windows(src, day=closed_perf_day)
    checked = 0
    for w in windows:
        itl = w.get("itl") or {}
        if not itl.get("n") or not itl.get("buckets"):
            continue
        assert sum(b[1] for b in itl["buckets"]) == pytest.approx(itl["n"]), w["t_start"]
        checked += 1
    assert checked > 100, checked


def test_percentiles_are_monotonic_and_inside_the_bucket_range(src: Sources, closed_perf_day: str):
    windows, _ = perf.load_windows(src, day=closed_perf_day)
    f = perf.fold(windows)
    got = perf.percentiles(f.itl_buckets, f.itl_n)
    values = [got[k] for k in ("p50", "p90", "p99") if got[k] is not None]
    assert values, got
    assert values == sorted(values), got
    assert max(values) <= max(f.itl_buckets), (got, max(f.itl_buckets))


def test_itl_semantics_are_measured_per_engine_not_assumed(src: Sources):
    """SGLang's ITL samples are tokens; vLLM's are speculative steps. Measured, over real windows.

    This is the defect the caveat exists for: quoting a vLLM figure raw overstated latency 3.2x.
    """
    windows, _ = perf.load_windows(src, days=99)
    by_model = perf.fold_by(windows, "model")
    seen = {}
    for choice, f in by_model.items():
        if f.itl_n < 1000:
            continue
        sem = caveats.itl_semantics(f.gen_tokens, f.itl_n)
        seen[choice] = sem["means"]
        if "sglang" in choice:
            assert sem["means"] == caveats.ITL_PER_TOKEN, (choice, sem)
            assert sem["tokens_per_sample"] == pytest.approx(1.0, abs=0.05), (choice, sem)
            assert sem["per_token_divisor"] == 1.0
        else:
            assert sem["means"] == caveats.ITL_PER_STEP, (choice, sem)
            assert sem["tokens_per_sample"] > 2.0, (choice, sem)
    assert caveats.ITL_PER_TOKEN in seen.values(), seen
    assert caveats.ITL_PER_STEP in seen.values(), seen


def test_folding_by_each_group_partitions_the_windows(src: Sources, closed_perf_day: str):
    windows, _ = perf.load_windows(src, day=closed_perf_day)
    whole = perf.fold(windows)
    for by in perf.GROUPS:
        parts = perf.fold_by(windows, by)
        assert sum(f.windows for f in parts.values()) == whole.windows, by
        assert sum(f.gen_tokens for f in parts.values()) == pytest.approx(whole.gen_tokens), by


def test_an_unknown_group_is_refused():
    with pytest.raises(ValueError):
        perf.group_key({}, "colour")


def test_filtering_to_a_model_keeps_only_that_model(src: Sources):
    windows, _ = perf.load_windows(src, days=99)
    choices = {(w.get("config") or {}).get("model_choice") for w in windows}
    choices.discard(None)
    assert len(choices) > 1, "need more than one model recorded to test the filter"
    pick = sorted(choices)[0]
    only, meta = perf.load_windows(src, days=99, model=pick)
    assert only and len(only) < len(windows)
    assert {(w["config"]["model_choice"]) for w in only} == {pick}
    assert meta["filtered_to_model"] == pick


def test_a_day_with_no_file_says_which_days_exist(src: Sources):
    with pytest.raises(SourceUnavailable) as exc:
        perf.load_windows(src, day="1999-01-01")
    assert "recorded days" in str(exc.value)


def test_boots_carry_the_configuration_and_the_outcome(src: Sources):
    boots, meta = perf.load_boots(src, days=99)
    assert boots, meta
    assert all(b.get("kind") == "boot" for b in boots)
    outcomes = {b.get("outcome") for b in boots}
    assert "serving" in outcomes, outcomes
    serving = [b for b in boots if b.get("outcome") == "serving"]
    assert all((b.get("config") or {}).get("model_choice") for b in serving)
    assert sorted(b["t_start"] for b in boots) == [b["t_start"] for b in boots], "not sorted"


def test_boots_can_be_filtered_to_one_outcome(src: Sources):
    everything, _ = perf.load_boots(src, days=99)
    kinds = {b.get("outcome") for b in everything}
    assert len(kinds) > 1, kinds
    only, meta = perf.load_boots(src, days=99, outcome="serving")
    assert only and {b["outcome"] for b in only} == {"serving"}
    assert meta["filtered_to_outcome"] == "serving"


def test_a_window_cannot_be_mistaken_for_a_boot(src: Sources):
    """Both file kinds live in the same directory; neither reader may pick up the other's rows."""
    windows, _ = perf.load_windows(src, days=99)
    assert not any(w.get("kind") == "boot" for w in windows)
    boots, _ = perf.load_boots(src, days=99)
    assert not any("itl" in b for b in boots)
