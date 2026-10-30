"""Install the pinned Claude Code into the image — at BUILD time, and only if both hashes match.

    python -m bench_runner.install_claude          (the Dockerfile's one network step)

Downloads claude_pin.TARBALL_URL, checks the tarball's sha256, extracts claude_pin.BINARY_MEMBER, checks the
binary's sha256, and writes it 0755 to claude_pin.INSTALL_PATH. Any mismatch fails the build: an image
either carries exactly the pinned client or does not exist.
"""
from __future__ import annotations

import hashlib
import io
import os
import pathlib
import platform
import sys
import tarfile
import urllib.request

from bench_runner import claude_pin


class PinMismatch(Exception):
    """What was downloaded is not what is pinned."""


def verify_and_extract(tgz: bytes, tgz_sha256: str, member: str, bin_sha256: str,
                       dest: pathlib.Path) -> None:
    got = hashlib.sha256(tgz).hexdigest()
    if got != tgz_sha256:
        raise PinMismatch(f"the tarball's sha256 is {got}, not the pinned {tgz_sha256}")
    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tar:
        try:
            info = tar.getmember(member)
        except KeyError:
            raise PinMismatch(f"the tarball has no {member}") from None
        if not info.isfile():
            raise PinMismatch(f"{member} is not a regular file")
        payload = tar.extractfile(info).read()
    got = hashlib.sha256(payload).hexdigest()
    if got != bin_sha256:
        raise PinMismatch(f"{member}'s sha256 is {got}, not the pinned {bin_sha256}")
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(payload)
    os.chmod(tmp, 0o755)
    os.replace(tmp, dest)


def main() -> int:
    if platform.machine() != claude_pin.ARCH:
        print(f"install_claude: no pinned Claude Code build for {platform.machine()} "
              f"(pinned: {claude_pin.ARCH})", file=sys.stderr)
        return 2
    with urllib.request.urlopen(claude_pin.TARBALL_URL, timeout=600) as r:
        tgz = r.read()
    try:
        verify_and_extract(tgz, claude_pin.TARBALL_SHA256, claude_pin.BINARY_MEMBER,
                           claude_pin.BINARY_SHA256, pathlib.Path(claude_pin.INSTALL_PATH))
    except PinMismatch as exc:
        print(f"install_claude: REFUSED — {exc}", file=sys.stderr)
        return 1
    print(f"install_claude: Claude Code {claude_pin.VERSION} at {claude_pin.INSTALL_PATH} "
          f"(sha256 {claude_pin.BINARY_SHA256[:12]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
