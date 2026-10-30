"""One small JSON file, written so that no reader and no restart ever sees half of it.

The LB's switches live in files under `/data` (each cluster's, e.g. `cluster.json`), so a restarted LB
reads what the operator last chose. A write that dies part-way must leave the file it was replacing
whole: the new content goes to a temporary file in the same directory, is flushed and fsynced, and only then takes the old file's name in one `os.replace`.
"""
from __future__ import annotations

import json
import os

import pytest

from vllm_lb.atomic import write_json


def test_a_reader_gets_back_exactly_what_was_written(tmp_path):
    path = tmp_path / "state" / "switch.json"              # its directory does not exist yet
    write_json(str(path), {"boxes": ["box-a"], "updated_at": 1.5})
    assert json.loads(path.read_text()) == {"boxes": ["box-a"], "updated_at": 1.5}
    write_json(str(path), {"boxes": []})
    assert json.loads(path.read_text()) == {"boxes": []}
    assert [p.name for p in path.parent.iterdir()] == ["switch.json"], "no temporary file is left"


def test_the_new_content_is_synced_beside_the_file_before_it_takes_the_name(tmp_path, monkeypatch):
    path = tmp_path / "switch.json"
    path.write_text('{"old": true}')
    seen: list[tuple] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        seen.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        return real_fsync(fd)

    def replace(src, dst):
        seen.append(("replace", str(src), str(dst)))
        assert json.loads(path.read_text()) == {"old": True}, "a reader saw new content before the rename"
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    write_json(str(path), {"new": True})
    monkeypatch.undo()

    assert [step[0] for step in seen] == ["fsync", "replace"], seen
    synced, (_, src, dst) = seen[0][1], seen[1]
    assert synced == src != dst == str(path), "the file synced is the one renamed onto the target"
    assert os.path.dirname(src) == str(tmp_path), "a rename is atomic only inside one filesystem"
    assert json.loads(path.read_text()) == {"new": True}


@pytest.mark.parametrize("failure", ["unserialisable", "rename refused"])
def test_a_write_that_fails_leaves_the_old_file_whole_and_no_temporary_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "switch.json"
    path.write_text('{"old": true}')
    new: dict = {"new": True}
    if failure == "unserialisable":
        new = {"new": True, "then": object()}          # json.dump has written part of it when it raises
    else:
        def disk_full(src, dst):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(os, "replace", disk_full)
    with pytest.raises((TypeError, OSError)):
        write_json(str(path), new)
    monkeypatch.undo()
    assert json.loads(path.read_text()) == {"old": True}
    assert [p.name for p in tmp_path.iterdir()] == ["switch.json"]
