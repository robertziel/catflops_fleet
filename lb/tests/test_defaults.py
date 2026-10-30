"""The shipped defaults must match what the servers MEASURABLY do (worklog/vllm-lb-context.md).

Two numbers travel together and are easy to get wrong apart:

  * the window we ADVERTISE on /v1/models — clients size their context from it
    (`fullclaudevllm` feeds it into CLAUDE_CODE_MAX_CONTEXT_TOKENS and compacts at 80 % of it);
  * how long we WAIT for a backend — a full-window prompt is prefill-bound, and on box-c's
    iGPU (216 tok/s measured) 262144 tokens is ~20 min. Advertising a window the proxy will
    not wait for just moves the failure from "compacted early" into a 502.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import RETIRED_KEYS, Config

FALLBACK_KEYS = ("LB_FALLBACK_URLS", "LB_FALLBACK_MODEL", "LB_FALLBACK_MAX_MODEL_LEN", "LB_FALLBACK_BOXES",
                 "LB_FALLBACK_BOX_LABEL", "LB_FALLBACK_IDLE_RELEASE_S", "LB_FALLBACK_LEASE_MAX_HOLD_S")


def test_the_stall_budget_is_the_short_one():
    """The FIRST-BYTE budget covers a full-window prefill plus queueing; the stall budget (silence mid-stream) is the
    short one."""
    cfg = Config()
    assert cfg.stall_timeout_s < cfg.first_byte_timeout_s


def test_env_still_overrides_the_first_byte_budget():
    c = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_BACKEND_TIMEOUT_S": "600"})
    assert c.first_byte_timeout_s == 600.0, "the old env name still sets the first-byte budget"


@pytest.mark.parametrize("key", FALLBACK_KEYS)
@pytest.mark.parametrize("value", ["", "x"])
def test_every_fallback_key_is_refused_at_boot_naming_itself(key, value):
    """The fallback tier is retired (operator, 2026-10-17: "If can't run then can't run, no fallback box"): a
    LB_FALLBACK_* key left in the environment — even blanked — refuses the start naming itself, never ignored."""
    with pytest.raises(ValueError, match=rf"^{key} is retired: .*remove {key} from the environment"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", key: value})


def test_no_fallback_key_is_left_unretired():
    assert {k for k in RETIRED_KEYS if k.startswith("LB_FALLBACK_")} == set(FALLBACK_KEYS)
    assert not [f for f in Config.__dataclass_fields__ if f.startswith("fallback_")], "no fallback setting remains"


@pytest.mark.parametrize("value", ["", "x"])
def test_the_keep_loaded_state_key_is_refused_at_boot_naming_itself(value):
    """Keep loaded is retired (operator, 2026-10-17): a box's server is held by demand alone. LB_KEEP_STATE left in
    the environment — even blanked — refuses the start naming itself, never ignored."""
    with pytest.raises(ValueError, match=r"^LB_KEEP_STATE is retired: .*remove LB_KEEP_STATE from the environment"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_KEEP_STATE": value})
    assert not [f for f in Config.__dataclass_fields__ if f.startswith("keep_")], "no keep-loaded setting remains"


def test_the_alias_migration_boots_when_the_old_line_is_blanked():
    """The migration an operator actually performs: add the new var, blank the old line. A
    nested eager parse of the old name made that a boot-time ValueError."""
    c = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FIRST_BYTE_TIMEOUT_S": "900", "LB_BACKEND_TIMEOUT_S": ""})
    assert c.first_byte_timeout_s == 900.0


def test_a_blank_value_means_unset_not_zero():
    c = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_STALL_TIMEOUT_S": "   "})
    assert c.stall_timeout_s == Config.stall_timeout_s


def test_a_non_positive_timeout_is_refused_at_boot_not_at_request_time():
    """`0` is the usual 'disable this' convention, but socket.settimeout(0) means NON-BLOCKING:
    every read would raise BlockingIOError. `-1` parses, then raises inside forward(). Both must
    fail loudly at boot, where an operator can see it."""
    import pytest
    for bad in ("0", "-1"):
        with pytest.raises(ValueError) as e:
            Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_STALL_TIMEOUT_S": bad})
        assert "LB_STALL_TIMEOUT_S" in str(e.value), str(e.value)
    with pytest.raises(ValueError):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FIRST_BYTE_TIMEOUT_S": "0"})
