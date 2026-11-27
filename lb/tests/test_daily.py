"""THE DAILY ROLLUP (vllm_lb/daily.py): the last 30 UTC days of generated tok/s and tokens per model,
and each model's last 24 h, folded from the perf log's own minute windows — no new collector, no second file.

Pinned here:
  * a day's tok/s is its tokens over the seconds its minutes covered (pooled, never a mean of minute rates), and its
    peak is the best minute; a day without a minute has 0 tokens and NO rate (a rate needs a minute);
  * a minute whose engine exposed no token counter is not 0 tokens: counted under `unmeasured`, and a day of only such
    minutes has null tokens;
  * the 24 h total is the minutes that ENDED inside the last 86 400 s;
  * the window is read past perflog's MAX_WINDOWS cut (30 days of minutes is more than /lb/perf hands over);
  * the endpoint is /lb/perf/daily, scoped by `?server=` like /lb/perf, and an admin's (a member is refused).
"""
from __future__ import annotations

import http.client
import json

from vllm_lb import perflog, roles
from vllm_lb.config import Config, VllmBox
from vllm_lb.daily import DAYS, fold, scan
from vllm_lb.hub import Hub
from vllm_lb.perflog import PerfLog, day_of
from vllm_lb.proxy import Proxy

DAY = 86400.0
#: 2026-11-20 12:00:00 UTC.
NOW = 1_795_176_000.0


def w(t_end, model="m-a", gen=600, prompt=1200, span=60.0, cluster=None):
    rec = {"v": 1, "t_start": t_end - span, "t_end": t_end, "span_s": span, "config": {"model_choice": model},
           "gen_tokens": gen, "prompt_tokens": prompt}
    if cluster is not None:
        rec["cluster"] = cluster
    return rec


def test_a_day_pools_its_minutes_into_one_rate_and_keeps_the_best_minute():
    out = fold([w(NOW - 100, gen=600), w(NOW - 50, gen=1800)], now=NOW)
    (m,) = out["models"]
    today = m["series"][-1]
    assert today["day"] == day_of(NOW)
    assert today["gen_tokens"] == 2400 and today["prompt_tokens"] == 2400
    assert today["gen_tok_s"] == 20.0, "2400 tokens over 120 s, not the mean of 10 and 30"
    assert today["peak_gen_tok_s"] == 30.0
    assert today["minutes"] == 2


def test_the_peak_is_the_best_single_minute_not_the_last_the_biggest_or_the_pooled_rate():
    # 900 tokens in a 30 s minute (30 tok/s) FIRST, then 1200 in a full minute (20 tok/s) and 300 in one (5 tok/s):
    # the last minute, the minute with most tokens, tokens/60 and the pooled 2400/150 = 16 are all something else.
    out = fold([w(NOW - 300, gen=900, span=30.0), w(NOW - 200, gen=1200), w(NOW - 100, gen=300)], now=NOW)
    today = out["models"][0]["series"][-1]
    assert today["peak_gen_tok_s"] == 30.0
    assert today["gen_tok_s"] == 16.0


def test_thirty_utc_days_oldest_first_and_an_empty_day_has_tokens_but_no_rate():
    out = fold([w(NOW - 3 * DAY)], now=NOW)
    assert len(out["days"]) == DAYS == 30
    assert out["days"][-1] == day_of(NOW) and out["days"][0] == day_of(NOW - 29 * DAY)
    series = out["models"][0]["series"]
    assert [d["day"] for d in series] == out["days"]
    empty = series[-1]
    assert empty["gen_tokens"] == 0 and empty["gen_tok_s"] is None and empty["minutes"] == 0
    assert series[-4]["gen_tokens"] == 600


def test_a_minute_older_than_the_range_is_not_folded():
    out = fold([w(NOW - 31 * DAY), w(NOW - 10)], now=NOW)
    assert sum(d["gen_tokens"] or 0 for d in out["models"][0]["series"]) == 600


def test_an_unmeasured_minute_is_not_zero_tokens():
    out = fold([w(NOW - 10, gen=None, prompt=None)], now=NOW)
    today = out["models"][0]["series"][-1]
    assert today["gen_tokens"] is None and today["gen_tok_s"] is None
    assert today["unmeasured"] == 1 and today["minutes"] == 1
    assert out["models"][0]["last_24h"]["gen_tokens"] is None


def test_the_24h_total_is_the_minutes_that_ended_in_the_last_86400_s_per_model():
    out = fold([w(NOW - DAY - 5, gen=999), w(NOW - DAY + 5, gen=100), w(NOW - 1, gen=200),
                w(NOW - 2, model="m-b", gen=7, prompt=3)], now=NOW)
    by = {m["id"]: m for m in out["models"]}
    assert by["m-a"]["last_24h"] == {"gen_tokens": 300, "prompt_tokens": 2400, "minutes": 2, "unmeasured": 0}
    assert by["m-b"]["last_24h"]["gen_tokens"] == 7
    assert out["total_24h"]["gen_tokens"] == 307


def test_models_are_ordered_by_their_30_day_tokens():
    out = fold([w(NOW - 10, model="small", gen=1), w(NOW - 20, model="big", gen=10_000)], now=NOW)
    assert [m["id"] for m in out["models"]] == ["big", "small"]


def test_scan_reads_past_perflogs_window_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(perflog, "MAX_WINDOWS", 5)
    for i in range(12):
        rec = w(NOW - i * 3600)
        with open(tmp_path / f"perf-{day_of(rec['t_end'])}.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    with open(tmp_path / f"perf-{day_of(NOW)}.jsonl", "a") as fh:
        fh.write("{not json\n")
    log = PerfLog(str(tmp_path))
    got = scan(log, now=NOW)
    assert len(got["windows"]) == 12
    assert got["malformed_total"] == 1 and got["error"] == ""


def test_scan_of_no_log_says_so(tmp_path):
    got = scan(PerfLog(str(tmp_path / "nowhere")), now=NOW)
    assert got["windows"] == [] and "no performance log" in got["error"]


# ── the endpoint ───────────────────────────────────────────────────────────────────────────────

def _proxy(tmp_path):
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),), listen_host="127.0.0.1",
                 listen_port=0, perf_dir=str(tmp_path))
    p = Proxy(cfg, Hub(cfg))
    p.start()
    return p


def _get(port, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    return r.status, json.loads(r.read())


def test_the_endpoint_folds_the_log_and_names_each_model(tmp_path):
    import time
    now = time.time()
    with open(tmp_path / f"perf-{day_of(now)}.jsonl", "w") as fh:
        fh.write(json.dumps(w(now - 5, gen=1200)) + "\n")
    p = _proxy(tmp_path)
    try:
        status, body = _get(p.port, "/lb/perf/daily")
        assert status == 200
        assert body["models"][0]["id"] == "m-a" and body["models"][0]["label"] == "m-a"
        assert body["models"][0]["last_24h"]["gen_tokens"] == 1200
        assert body["server"] is None and body["error"] == ""
        status, body = _get(p.port, "/lb/perf/daily?server=nope")
        assert status == 404
        status, _ = _get(p.port, "/lb/perf/daily", {roles.ROLE_HEADER: roles.MEMBER, roles.MODELS_HEADER: "m-a"})
        assert status == 403
    finally:
        p.stop()
