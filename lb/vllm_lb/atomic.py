"""One small JSON file, written so that no reader and no restart ever sees half of it.

The LB keeps the operator's switches in files under `/data`: each cluster's (`cluster.DesiredStore`).
A restarted LB reads what was last chosen, so a write that dies
part-way (a crash, a full disk, a value that will not serialise) must leave the previous file whole.
The new content goes to a temporary file beside the target, is flushed and fsynced, and only then
takes the target's name in one `os.replace`, which is atomic within a filesystem.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("vllm_lb.atomic")


def write_json(path: str, obj) -> None:
    """Replace the file at `path` with `obj` as JSON: whole, or not at all. Creates its directory."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)            # a half-written temp file is nobody's state
        except OSError:
            pass
        raise


#: THE PROVENANCE FIELD OF THE BLUEPRINT ERA (2026-10-30, a few hours): a pair's switch file and a node's wish named the
#: start blueprint a selection came from. Blueprints left the runtime (operator, 2026-10-30: "Blueprints meant to be
#: only for agent to build their own fleet settings, not something user chooses"; docs/design/engines.md
#: "Blueprints"), and the field never launched anything — every figure it stood for is stored beside it.
RETIRED_STATE_FIELD = "blueprint"


def retire_blueprint_field(path: str, raw):
    """`raw` (a state file's document, as read from `path`) without the retired `blueprint` field — MIGRATED ONCE: the
    file is rewritten whole without it, and the line says so, naming the file and the id it carried. Neither refused
    (the pair or node would stop over a field that changed nothing it runs) nor read past (an ignored key is a value
    nobody can see is dead): removed where it is stored, with nothing else touched — the selection, its generation and
    its start id are what they were. A document without it is returned as it is, unwritten."""
    if not isinstance(raw, dict) or RETIRED_STATE_FIELD not in raw:
        return raw
    out = {k: v for k, v in raw.items() if k != RETIRED_STATE_FIELD}
    write_json(path, out)
    log.info("state: migrated %s — removed the retired `%s` field (%r): blueprints are the setup wizard's reference "
             "setups now, never a runtime object; the stored selection is unchanged", path, RETIRED_STATE_FIELD,
             raw[RETIRED_STATE_FIELD])
    return out
