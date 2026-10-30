"""THE ONE CLAUDE CODE the sealed runner runs: an exact version, its npm tarball and its binary, by hash.

The host's `claude` updates itself, so a benchmark that ran on it ran on whatever had landed overnight:
the host's swarm sessions of 2026-10-05..11 ran SIX versions (each session's `init` event names its
`claude_code_version`: 2.1.289, .291, .292, .294, .295, .296). The sealed runner runs exactly this build:
install_claude.py refuses a tarball or a binary whose sha256 is not below, and entrypoint.py refuses to
start the runner on a binary that differs.

2.1.296 is the version the last three host sessions ran (2026-10-10 20:02 to 2026-10-11 10:11), and
BINARY_SHA256 is the hash of the host's own ~/.local/share/claude/versions/2.1.296 — npm's linux-x64 build
of the same version is byte-for-byte that binary. Bumping it is a deliberate commit: all three values
from `npm pack @anthropic-ai/claude-code-linux-x64@<version>` and `sha256sum`, and a fresh seal proof.
"""
VERSION = "2.1.296"
#: npm's per-platform package: one file, `package/claude`, a native x86_64 executable (glibc).
PACKAGE = "@anthropic-ai/claude-code-linux-x64"
TARBALL_URL = ("https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/"
               f"claude-code-linux-x64-{VERSION}.tgz")
TARBALL_SHA256 = "eef5a2e2b09a5e7d2ce784d1b3ba335ae7fc6c7f6256c802a0ebf2c21e4fd0e1"
BINARY_MEMBER = "package/claude"
BINARY_SHA256 = "24972e3bc859fab2b46ed4c1e51f7d6130f06d3bd550811a114640de3370d0de"
#: Root-owned, outside every HOME: the session can neither replace it nor shadow it on its PATH.
INSTALL_PATH = "/opt/claude/bin/claude"
#: The only machine both deployments run the runner on today; anything else is refused at build.
ARCH = "x86_64"
