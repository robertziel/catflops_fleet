"""The change between two /lb/status documents, as a small patch: what /lb/stream sends after its first event.

WHY A PATCH. The page used to fetch the whole status every 3 s: 299,618 bytes, uncompressed, most of it the catalog
and the history rings, which change by one sample a probe. At one probe a second, sending the document whole
would be ~300 KB/s per open tab. The stream sends it whole ONCE, then only what moved: the scalars that changed
and, for each history ring, the sample it took.

THE FORMAT — a node per value, the wire contract with the page's applier (static/status_delta.js), pinned by
tests/fixtures/status_delta_cases.json, which both sides' tests read:

    {"=": v}                 the value is now v (any JSON)
    {"o": {k: node}, "x": [k, …]}   an object: these keys changed (each by its own node), these are gone
    {"i": {"<index>": node}} a list of the same length: these items changed
    {"r": [drop, tail]}      a RING: the old list without its first `drop` items, then `tail`
                             (a history ring that took a sample: drop 1, push 1 — or drop 0 while it fills)
    {"u": [drop, head]}      a NEWEST-FIRST list: `head`, then the old list without its last `drop` items
                             (the requests card's finished rows, newest on top)
    null                     (the whole patch) nothing changed

A list that changed any other way is sent whole. No patch is ever guessed: `apply(diff(a, b), a)` is `b`, and a
page that missed an event cannot apply the next one to the wrong base — it reconnects and is sent the document
whole again (the stream never resumes a patch sequence).

EQUALITY IS JSON's. `True == 1` in Python, but `true` and `1` are different values on the page, so a bool that
became a number is a change. An int and a float of the same value are the same number (JSON has one).
"""
from __future__ import annotations

#: How many samples a ring may have taken between two documents and still be sent as a shift. One a probe is
#: the rule; a few cover a stream that was late. Past this the list is compared item by item or sent whole —
#: and the search stays short, because it runs for every ring of every backend, every second, per open tab.
MAX_SHIFT = 8


def _same(x, y) -> bool:
    """JSON equality: bools are not numbers, ints and floats of one value are one number."""
    if x is y:
        return True
    tx, ty = type(x), type(y)
    if tx is bool or ty is bool:
        return tx is ty and x == y
    if tx in (int, float) and ty in (int, float):
        return x == y
    if tx is not ty:
        return False
    if tx is dict:
        return len(x) == len(y) and all(k in y and _same(v, y[k]) for k, v in x.items())
    if tx is list:
        return _same_items(x, 0, y, 0, len(x)) if len(x) == len(y) else False
    return x == y


def _same_items(a: list, i: int, b: list, j: int, n: int) -> bool:
    """a[i:i+n] and b[j:j+n] are the same items — without copying either."""
    for k in range(n):
        if not _same(a[i + k], b[j + k]):
            return False
    return True


def _ring(a: list, b: list):
    """`[drop, tail]` when b is a with its first `drop` items gone and `tail` pushed — else None. The overlap
    must be at least one item (`drop` stops one short of a), or it is not a ring that moved, it is a new list."""
    for drop in range(0, min(len(a) - 1, MAX_SHIFT) + 1):
        keep = len(a) - drop
        if keep > len(b) or len(b) - keep > MAX_SHIFT:
            continue
        if _same_items(a, drop, b, 0, keep):
            return [drop, b[keep:]]
    return None


def _unshift(a: list, b: list):
    """`[drop, head]` when b is `head` followed by a without its last `drop` items — the ring's mirror, for a list
    kept newest first. The same rules: at least one item kept, at most MAX_SHIFT moved."""
    for drop in range(0, min(len(a) - 1, MAX_SHIFT) + 1):
        keep = len(a) - drop
        if keep > len(b) or len(b) - keep > MAX_SHIFT:
            continue
        if _same_items(a, 0, b, len(b) - keep, keep):
            return [drop, b[:len(b) - keep]]
    return None


def diff(a, b):
    """The patch that turns `a` into `b`, or None when they are the same document."""
    if type(a) is dict and type(b) is dict:
        changed, gone = {}, []
        for k, v in b.items():
            if k not in a:
                changed[k] = {"=": v}
                continue
            node = diff(a[k], v)
            if node is not None:
                changed[k] = node
        for k in a:
            if k not in b:
                gone.append(k)
        if not changed and not gone:
            return None
        out = {"o": changed}
        if gone:
            out["x"] = gone
        return out
    if type(a) is list and type(b) is list:
        if len(a) == len(b) and _same_items(a, 0, b, 0, len(a)):
            return None
        ring = _ring(a, b) if a else None
        if ring is not None:
            return {"r": ring}
        front = _unshift(a, b) if a else None
        if front is not None:
            return {"u": front}
        if len(a) == len(b):
            items = {}
            for idx, (x, y) in enumerate(zip(a, b)):
                node = diff(x, y)
                if node is not None:
                    items[str(idx)] = node
            return {"i": items}
        return {"=": b}
    return None if _same(a, b) else {"=": b}


def apply(patch, doc):
    """`doc` with `patch` applied — a new document; `doc` is not changed. The page's static/status_delta.js is
    this function in JavaScript; this one is what the tests hold the pair to."""
    if patch is None:
        return doc
    if "=" in patch:
        return patch["="]
    if "o" in patch:
        out = dict(doc)
        for k, node in patch["o"].items():
            out[k] = apply(node, doc.get(k))
        for k in patch.get("x", ()):
            out.pop(k, None)
        return out
    if "i" in patch:
        out = list(doc)
        for idx, node in patch["i"].items():
            out[int(idx)] = apply(node, doc[int(idx)])
        return out
    if "r" in patch:
        drop, tail = patch["r"]
        return list(doc[drop:]) + list(tail)
    if "u" in patch:
        drop, head = patch["u"]
        return list(head) + list(doc[:len(doc) - drop])
    raise ValueError(f"not a status patch node: {sorted(patch)}")
