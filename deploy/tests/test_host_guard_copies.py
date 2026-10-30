"""THE HOST ALLOWLIST is ONE module (lb/vllm_lb/host_guard.py) shipped twice: the gateway and the power panel are two
services, two images, stdlib only, with no package both could import — so power/fleet_power/host_guard.py is a copy,
and this guard keeps it one: a fix to the rule lands in both or the suite fails.

    docker run --rm -v <checkout>:/repo:ro -w /repo python:3.12-slim python -m unittest discover -s deploy/tests
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "lb" / "vllm_lb" / "host_guard.py"
COPIES = (REPO / "power" / "fleet_power" / "host_guard.py",)


class HostGuardCopies(unittest.TestCase):
    def test_every_copy_is_the_gateways_byte_for_byte(self):
        source = SOURCE.read_bytes()
        self.assertIn(b"def parse_allowed", source)
        for copy in COPIES:
            with self.subTest(copy=str(copy.relative_to(REPO))):
                self.assertEqual(copy.read_bytes(), source, f"{copy.relative_to(REPO)} differs from "
                                 f"{SOURCE.relative_to(REPO)}: change one, copy it to the other")

    def test_the_guard_bites_a_diverged_copy(self):
        source = SOURCE.read_bytes()
        self.assertNotEqual(source.replace(b"DEFAULT_PORTS = (80, 443)", b"DEFAULT_PORTS = (80,)"), source)


if __name__ == "__main__":
    unittest.main()
