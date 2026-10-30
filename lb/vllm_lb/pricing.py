"""WHAT THE TOKENS WOULD HAVE COST — a vendor's published price list applied to what the fleet generated.

Operator, 2026-10-05: "include price estimate based on model official pricing ... to know how much saved".

TWO KINDS OF COMPARISON, side by side, because they answer different questions:

  own        the served model's OWN vendor's list (Qwen3.8-27B on Alibaba Model Studio): what the SAME
             model costs from the people who make it — what the hardware saved.
  reference  a model you would otherwise have used (Claude Opus 5.5): what the work would have cost
             through the API you actually drive this fleet instead of — what choosing it saved.

THE DATA IS HARDCODED, DATED AND SOURCED (`pricing.json`). Neither vendor publishes a machine-readable
price API — Anthropic's list is a docs page, Alibaba's a console table — so each list carries its
`as_of`, its `source` URL and a `basis` saying exactly where each number came from, and the page prints
the date. A list that cannot say all three is refused at load.

NO FALLBACKS (operator rule, 2026-10-05). A model with no published price is UNPRICED — its cost is
None, never 0 and never another model's figure — and says why. Qwen3.8-Flash-Next is exactly that case:
it is not on Alibaba's list, and the nearest listed model is a different one with an unpublished cache
price. A total that leaves unpriced sessions out says so (`priced` of `of`), because a sum over part of
the history presented as the whole would be the silent substitution this rule exists to prevent.

THE ARITHMETIC. A session's tokens are three populations: prompt tokens the engine COMPUTED (`prompt_
computed_tokens` — billed at the list's input price), prompt tokens it served from its prefix cache
(billed − computed, at the list's cache-read price), and tokens it generated (output price). A list with
no cache-read rate cannot price the cached population, so it prices NOTHING rather than invent a rate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

PRICING_PATH = Path(__file__).resolve().with_name("pricing.json")
#: The rates every list must state: a list that cannot price all three populations is not a list.
RATES = ("input", "output", "cache_read")
_TEXT = ("vendor", "label", "as_of", "source", "basis")


class PricingError(ValueError):
    """A price document that breaks a rule — refused whole, with the reason."""


@dataclass(frozen=True)
class PriceList:
    id: str
    vendor: str
    label: str
    usd_per_mtok: dict
    as_of: str
    source: str
    basis: str
    excludes: str = ""
    tokenizer: str = ""

    def view(self) -> dict:
        """What the page shows: the figures AND where they came from — a price with no date and no
        source is a guess wearing a number."""
        return {"id": self.id, "vendor": self.vendor, "label": self.label, "usd_per_mtok": dict(self.usd_per_mtok),
                "as_of": self.as_of, "source": self.source, "basis": self.basis, "excludes": self.excludes,
                "tokenizer": self.tokenizer}

    def cost(self, *, computed: float, cached: float, generated: float) -> float:
        """Dollars for these three populations of tokens."""
        r = self.usd_per_mtok
        return (computed * r["input"] + cached * r["cache_read"] + generated * r["output"]) / 1e6


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Pricing:
    def __init__(self, lists=(), references=(), own=None, error: str = "") -> None:
        self.lists: dict[str, PriceList] = {p.id: p for p in lists}
        #: the lists every model is compared against, in the order the page draws them
        self.references: tuple[str, ...] = tuple(references)
        #: model id -> {"list": <list id>} or {"unpriced": <the reason it has none>} — two shapes that
        #: cannot be mistaken for one another, which a bare string for either could be
        self.own: dict[str, dict] = dict(own or {})
        #: "" when the document loaded. Otherwise why not — and then NOTHING is priced.
        self.error = error

    @classmethod
    def load(cls, path: str | Path = PRICING_PATH) -> "Pricing":
        """The price document, never raising: a pricing file the LB cannot read is a status line on the
        page, and the sessions are still listed — with no cost on them, never a made-up one."""
        path = Path(path)
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            return cls(error=f"{path}: {type(exc).__name__}: {exc}"[:300])

    @classmethod
    def from_dict(cls, raw) -> "Pricing":
        if not isinstance(raw, dict) or set(raw) != {"lists", "references", "own"}:
            raise PricingError("the price document must be exactly {lists, references, own}")
        lists = []
        for pid, item in (raw["lists"] or {}).items():
            where = f"price list {pid}"
            if not isinstance(item, dict):
                raise PricingError(f"{where} must be an object")
            unknown = set(item) - {"usd_per_mtok", "excludes", "tokenizer", *_TEXT}
            if unknown:
                raise PricingError(f"{where}: unknown {sorted(unknown)}")
            for key in _TEXT:
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise PricingError(f"{where}: {key} is required — a price with no date and no source is a guess")
            rates = item.get("usd_per_mtok")
            if not isinstance(rates, dict) or set(rates) != set(RATES):
                raise PricingError(f"{where}: usd_per_mtok must state exactly {', '.join(RATES)}")
            for k, v in rates.items():
                if not (_number(v) and v >= 0):
                    raise PricingError(f"{where}: {k} must be a non-negative number of USD per million tokens")
            if rates["cache_read"] > rates["input"]:
                raise PricingError(f"{where}: a cache read dearer than the input it replaces is a typo")
            lists.append(PriceList(id=pid, usd_per_mtok={k: float(rates[k]) for k in RATES},
                                   **{k: item[k] for k in _TEXT}, excludes=str(item.get("excludes", "")),
                                   tokenizer=str(item.get("tokenizer", ""))))
        ids = {p.id for p in lists}
        refs = raw["references"]
        if not isinstance(refs, list) or not all(isinstance(r, str) and r in ids for r in refs):
            raise PricingError("references must name price lists that exist")
        own = raw["own"]
        if not isinstance(own, dict):
            raise PricingError("own must map a model id to a price list id or to a reason it has none")
        for model, v in own.items():
            if isinstance(v, str):
                if v not in ids:
                    raise PricingError(f"model {model} names price list {v!r}, which does not exist")
            elif not (isinstance(v, dict) and set(v) == {"unpriced"} and isinstance(v["unpriced"], str)
                      and v["unpriced"].strip()):
                raise PricingError(f"model {model}: a list id, or {{\"unpriced\": <the reason>}}")
        return cls(lists, refs, {m: ({"list": v} if isinstance(v, str) else dict(v)) for m, v in own.items()},
                   error="")

    # ── reads ────────────────────────────────────────────────────────────

    def own_of(self, model_choice: str) -> tuple[PriceList | None, str]:
        """`(its own vendor's list, "")` or `(None, why there is none)` — every model answers one or the other."""
        if self.error:
            return None, f"the price document could not be read: {self.error}"
        v = self.own.get(model_choice)
        if v is None:
            return None, "this model has no entry in pricing.json — nobody has decided what it would cost"
        return (self.lists[v["list"]], "") if "list" in v else (None, v["unpriced"])

    def reference_lists(self) -> list[PriceList]:
        return [] if self.error else [self.lists[r] for r in self.references]


def _populations(row: dict) -> tuple[float, float, float] | None:
    """(computed, cached, generated) tokens of a session, or None when the log did not carry them. The
    cached population is the prompt minus what the engine computed, clamped at 0: a window's hit counter
    can run ahead of its prompt counter (generation 123 recorded a 949% hit rate), and a negative
    population must never become a negative price."""
    prompt, computed, gen = row.get("prompt_tokens"), row.get("prompt_computed_tokens"), row.get("gen_tokens")
    if not all(_number(v) for v in (prompt, computed, gen)):
        return None
    return float(computed), max(0.0, float(prompt) - float(computed)), float(gen)


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def apply(out: dict, pricing: Pricing) -> dict:
    """Put a cost on every session, every model's rollup and the totals of `perflog.sessions`' output.

    A cost is `{list id: dollars}` and is None for a list that cannot price that session (no published
    price, or tokens the log did not carry) — never 0. Totals are sums over the PRICED sessions only and
    carry `{"usd", "priced", "of"}` so the page can say how much of the history a figure covers."""
    refs = pricing.reference_lists()
    rows = out.get("sessions") or []

    def price(row: dict) -> dict:
        pop = _populations(row)
        cost: dict[str, float | None] = {}
        own, _ = pricing.own_of(row["model_choice"])
        for lst in ([own] if own is not None else []) + refs:
            cost[lst.id] = None if pop is None else lst.cost(computed=pop[0], cached=pop[1], generated=pop[2])
        return cost

    for row in rows:
        row["cost"] = {k: _round(v) for k, v in price(row).items()}
        own, why = pricing.own_of(row["model_choice"])
        row["own_price"] = {"list": own.id if own is not None else None, "why": why}

    def total(subset: list[dict], list_of) -> dict:
        usd, priced = 0.0, 0
        for row in subset:
            lid = list_of(row)
            v = (row["cost"] or {}).get(lid) if lid else None
            if v is not None:
                usd += v
                priced += 1
        return {"usd": _round(usd) if priced else None, "priced": priced, "of": len(subset)}

    own_id = lambda r: r["own_price"]["list"]            # noqa: E731 — each row's own list
    totals = {"own": total(rows, own_id)}
    for lst in refs:
        totals[lst.id] = total(rows, lambda r, i=lst.id: i)
    out["totals"]["cost"] = totals
    for model in out.get("by_model") or []:
        mine = [r for r in rows if r["model_choice"] == model["model_choice"]]
        model["cost"] = {"own": total(mine, own_id),
                         **{lst.id: total(mine, lambda r, i=lst.id: i) for lst in refs}}
    out["price_lists"] = {"own": sorted({r["own_price"]["list"] for r in rows if r["own_price"]["list"]}),
                          "references": [lst.id for lst in refs],
                          "lists": {p.id: p.view() for p in pricing.lists.values()},
                          "unpriced": {r["model_choice"]: r["own_price"]["why"]
                                       for r in rows if not r["own_price"]["list"]},
                          "error": pricing.error}
    return out
