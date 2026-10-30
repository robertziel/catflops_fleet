REAL captures, not fixtures.

lb-status-2026-10-03.json  —  `GET http://127.0.0.1:8010/lb/status` taken on 2026-10-03 while the
                              pair was serving generation 123 (qwen3.8-27b-sglang-dflash2, SGLang,
                              16 seats). 58 KB; the edits are `json.dump(indent=1)` and the
                              scrub of the operator's own values: LAN addresses → RFC 5737
                              (192.0.2.2 → 192.0.2.2, 192.0.2.12 → 198.51.100.2), model
                              directories → /srv/models, the repo → /srv/repo. The shape is unchanged.
lb-bench-2026-10-03.json   —  `GET /lb/bench` from the same minute: zero saved runs, which is the
                              real state of lb/state/bench and the case the tools must answer
                              honestly rather than treat as an error.

The perf windows, boot logs and swarm captures the suite reads are NOT copied here: the tests read
the fleet's own lb/state/ directly (mounted read-only), so they cannot pass against invented data.

lb-perf-filecounts-2026-10-03.json  —  the `files` / `malformed` / counts sections of
                              `GET /lb/perf` from the same minute (the windows themselves are 3.5
                              MB and are NOT copied: the suite reads the real jsonl instead). This
                              is a cross-check fixture: the LB's OWN reader counted 596 windows and
                              1 malformed line in perf-2026-10-02.jsonl, so test_perf.py asserts
                              this package's reader agrees with it file by file.
