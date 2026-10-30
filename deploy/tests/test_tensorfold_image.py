"""The fleet's TensorFold image (deploy/images/tensorfold, built by deploy/tensorfold/image/build.sh): the mia-ai image with tensorfold-native rebuilt from the same
tree plus the fleet's patches, gated by TensorFold's own server unit tests, built on the boxes the operator names.

Claude Code (beta mid-conversation-system-2026-04-07) sends its reminders as a trailing `role: system` message; the
mia-ai image's native server answered every such turn 400 "the model's chat template does not support mid-conversation
system messages" (2026-10-29), because Qwen3.8's template renders no later system message.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMAGE = REPO / "deploy" / "images" / "tensorfold"
PIN = IMAGE / "native.nvidia-sm121.env"
ENTRY = REPO / "deploy" / "tensorfold" / "image" / "build.sh"


class TensorFoldImage(unittest.TestCase):
    def test_the_fleet_patch_series_is_the_late_system_merge_then_the_work_counters(self):
        self.assertEqual(sorted(p.name for p in IMAGE.glob("*.patch")),
                         ["0013-anthropic-merge-late-system.patch", "0014-work-counters.patch"])
        patch = (IMAGE / "0013-anthropic-merge-late-system.patch").read_text()
        # the refusal leaves the route, and the translator takes the template's late-system role
        self.assertIn('-            return sendError(conn, a, "the model\'s chat template does not support mid-conversation '
                      'system messages", 400);', patch)
        self.assertIn("+pub fn translate(cx: *Cx, body: Value, count: bool, late_system: []const u8)", patch)

    def test_the_work_counters_reach_metrics_from_the_lane_core_and_the_engine(self):
        """0014: the #pair COMPUTE / PREFILL tiles had no source — TensorFold published no pass or computed-prompt
        counter (2026-10-29). The lane core's backend is wrapped in a counter (decode passes, verified rows, the last
        pass's context, busy seconds) and the Flash-Next engine counts its own prompt chunks; /metrics renders both."""
        patch = (IMAGE / "0014-work-counters.patch").read_text()
        self.assertIn("+pub const Counting = struct {", patch)                       # core/lanes/counting.zig
        self.assertIn("+        prompt_work: ?*const fn (ptr: *anyopaque) PromptWork = null,", patch)
        self.assertIn("+    h.core = lanes.Engine.init(gpa, &h.cfg, h.counting.backend(), h.clock.clock());", patch)
        self.assertIn("+            e.notePromptChunk(R);", patch)                    # cuda_engine: each committed chunk
        self.assertIn("+            .prompt_work = if (v.prompt_work != null) promptWorkFn else null,", patch)  # mirror
        for family in ("decode_passes_total", "prompt_passes_total", "prompt_tokens_computed_total",
                       "prompt_tokens_cached_total", "decode_context_tokens", "verify_width", "busy_seconds_total",
                       "request_prefill_computed_tokens_total"):
            self.assertIn(f'"{family}"', patch)

    def test_the_lane_core_tests_gate_the_build_too(self):
        dockerfile = (IMAGE / "Dockerfile").read_text()
        gate = dockerfile.index("zig/src/core/lanes/lanes.zig")
        self.assertLess(dockerfile.index("/opt/fleet-patches/*.patch"), gate)
        self.assertLess(gate, dockerfile.index("zig build"))

    def test_the_server_unit_tests_gate_the_build_after_every_patch(self):
        dockerfile = (IMAGE / "Dockerfile").read_text()
        gate = dockerfile.index("zig/tests/server/zig_test.sh")
        self.assertLess(dockerfile.index("/opt/fleet-patches/*.patch"), gate)
        self.assertLess(gate, dockerfile.index("zig build"))

    def test_the_build_pins_the_tree_the_base_was_built_from(self):
        build = PIN.read_text()
        for pin in ("TF_REF=db281878ddb836fd0df510d8771ecb7e0fe47d26",
                    "RECIPE_REF=be3ed857ab1055ca47b36134d22691f00f2bb94a",   # image_hash f3e1300f4f62 = BASE's tf.patches
                    "BASE_ID=sha256:5eb5d4d384773543109baa26783fc6ea38ad619dff330f01672e05c01721cfb0",
                    "ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8"):
            self.assertIn(pin, build)

    def test_the_tag_is_the_image_matrixs_native_build(self):
        self.assertIn('images/build.sh" tensorfold native nvidia-sm121', ENTRY.read_text())
        self.assertIn("PLATFORM=linux/arm64", PIN.read_text().splitlines())
        r = subprocess.run(["bash", "deploy/image-tag.sh", "tensorfold", "native", "nvidia-sm121", "linux/arm64"],
                           cwd=REPO, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "tensorfold-qwen38fn:zig-db28187-f3e1300f4f62-fleet.2"))

    def test_it_takes_no_default_box_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "ssh.log"
            fake = Path(tmp) / "ssh"
            fake.write_text(f'#!/bin/sh\necho "$@" >> {log}\n')
            fake.chmod(0o755)
            r = subprocess.run(["bash", str(ENTRY)], capture_output=True, text=True,
                               env={"PATH": f"{tmp}:{os.environ['PATH']}"})
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("usage:", r.stderr)
            self.assertFalse(log.exists(), "no box dialled")


if __name__ == "__main__":
    unittest.main()
