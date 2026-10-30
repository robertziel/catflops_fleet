"""ONE FIELD OF A multipart/form-data BODY, read or set — every other byte kept (RFC 7578).

A transcription upload (POST /v1/audio/transcriptions, router.TRANSCRIPTION_PATHS) is a form: the audio file and a few
text fields, one of them `model`. Clients name the model loosely (OpenAI clients send `whisper-1`), and the engine refuses any
name but its own (vLLM: 404 "The model `x` does not exist"), so the LB sets the field to what the chosen backend serves
— as `router.rewrite_model` does for a JSON body — after routing by the name it gave (`get_field`). Only that part's VALUE changes: the boundary, the other parts (the
audio's bytes above all) and their order are the client's. Stdlib only, no parser that re-encodes the body.
"""
from __future__ import annotations

import re

_BOUNDARY = re.compile(r'boundary=(?:"([^"]+)"|([^\s;]+))', re.IGNORECASE)
_NAME = re.compile(rb'(?im)^content-disposition:\s*form-data\s*;(?:.*?;)?\s*name="([^"]*)"')
#: A value this module writes into a part: one line, so it cannot open a header or a part of its own.
_VALUE = re.compile(r"[^\r\n]+")


def boundary(content_type: str | None) -> bytes:
    """The boundary a multipart/form-data Content-Type names — or ValueError: not a form, or no boundary."""
    ctype = (content_type or "").strip()
    if ctype.split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise ValueError(f"the body is {ctype or 'of no type'}, not multipart/form-data")
    m = _BOUNDARY.search(ctype)
    if m is None:
        raise ValueError("multipart/form-data names no boundary")
    return (m.group(1) or m.group(2)).encode()


def _parts(body: bytes, content_type: str | None) -> tuple[bytes, list[bytes]]:
    delim = b"--" + boundary(content_type)
    parts = body.split(delim)
    if len(parts) < 2 or not parts[-1].startswith(b"--"):
        raise ValueError("the multipart/form-data body has no closing boundary")
    return delim, parts


def _field(part: bytes, name: str) -> tuple[bytes, bytes, bytes] | None:
    """(head, separator, value) of `part` when it is the TEXT field `name` (a file part is never one), else None."""
    head, sep, rest = part.partition(b"\r\n\r\n")
    if not sep:
        raise ValueError("a part of the multipart/form-data body has no header end")
    m = _NAME.search(head)
    if m is None or m.group(1).decode("utf-8", "replace") != name or b"filename=" in head.lower():
        return None
    return head, sep, rest


def get_field(body: bytes, content_type: str | None, name: str) -> str | None:
    """The value of text field `name` (its first part), or None where the form has none. ValueError for a body that is
    not a complete form."""
    _, parts = _parts(body, content_type)
    for part in parts[1:-1]:
        f = _field(part, name)
        if f is not None:
            return f[2].removesuffix(b"\r\n").decode("utf-8", "replace")
    return None


def set_field(body: bytes, content_type: str | None, name: str, value: str) -> bytes:
    """`body` with form field `name` set to `value`: its first part of that name has its value replaced, or — none — one
    is added as the last part. ValueError for a body that is not a complete form, or a value that is not one line."""
    if not _VALUE.fullmatch(value):
        raise ValueError(f"a form value is one line, got {value!r}")
    delim, parts = _parts(body, content_type)
    for i in range(1, len(parts) - 1):
        f = _field(parts[i], name)
        if f is not None:
            parts[i] = f[0] + f[1] + value.encode() + b"\r\n"
            break
    else:
        parts.insert(-1, f'\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    return delim.join(parts)
