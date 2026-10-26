"""The repair call: one LLM request per failing signature group (or per rejected template).

The request carries the spec, the failing fields with what they returned, the values the
analysis expected on the failing page, and that page compressed like a union page. The answer
is ``{"fields": {name: field object}}``; each returned field is validated with the spec model and
must keep its type. Verification then scores the repaired field against every page and keeps it
only when it fails on fewer pages than before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import orjson
from lxml import etree, html
from pydantic import TypeAdapter, ValidationError

from ditto.fingerprint.dom import canonical_classes, clean_html
from ditto.specgen.models import (
    KeyValueField,
    ListField,
    RecordListField,
    ScalarField,
    Spec,
    SpecField,
)
from ditto.specgen.prompts import load_prompt

if TYPE_CHECKING:
    from ditto.analyze.models import CandidateTable
    from ditto.llm.base import LLM

REPAIR_PROMPT = "repair"
REPAIR_PROMPT_VERSION = 1
REPAIR_PAGE_TOKENS = 12000
CHARS_PER_TOKEN = 4
KEEP_ATTRS = ("id", "class", "href", "src", "itemprop", "data-testid", "role", "aria-label", "type")
DROPPED_KEYS = ("stats", "selection_reason", "degraded", "optional", "source_candidates")

_FIELD_ADAPTER: TypeAdapter[SpecField] = TypeAdapter(SpecField)


def repair_prompt_id() -> str:
    return load_prompt(REPAIR_PROMPT, REPAIR_PROMPT_VERSION).id


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 1) // 2
    return text[:half] + "…" + text[-half:]


def compact_page(page_html: str, *, max_tokens: int = REPAIR_PAGE_TOKENS) -> str:
    """Cleaned page with attributes pruned and long text cut (used when no candidate table
    exists to build a union-style page)."""
    tree = clean_html(page_html)
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        for name in list(el.attrib):
            if name not in KEEP_ATTRS:
                del el.attrib[name]
        if "class" in el.attrib:
            kept = canonical_classes(el.get("class"))
            if kept:
                el.set("class", " ".join(kept))
            else:
                del el.attrib["class"]
        for name in ("href", "src"):
            value = el.get(name)
            if value and len(value) > 80:
                el.set(name, value[:80])
        if el.text and len(el.text) > 120:
            el.text = _truncate_middle(el.text, 120)
        if el.tail and len(el.tail) > 120:
            el.tail = _truncate_middle(el.tail, 120)
    text = str(html.tostring(tree, encoding="unicode"))
    return text[: max_tokens * CHARS_PER_TOKEN]


def compress_page(
    page_html: str,
    page_id: int,
    table: CandidateTable | None,
    *,
    max_tokens: int = REPAIR_PAGE_TOKENS,
) -> str:
    """The failing page as the LLM should see it: annotated and compressed exactly like a
    union page (one page, no donors) when the analysis table exists."""
    if table is None:
        return compact_page(page_html, max_tokens=max_tokens)
    from ditto.union import build_union_page

    return build_union_page(table, {page_id: page_html}, min_tokens=0, max_tokens=max_tokens).html


def spec_for_prompt(spec: Spec, *, only: Sequence[str] | None = None) -> str:
    """The spec as JSON without verification bookkeeping (stats, reasons)."""
    fields: dict[str, Any] = {}
    for name, f in spec.fields.items():
        data = f.model_dump(mode="json", exclude_none=True)
        for key in DROPPED_KEYS:
            data.pop(key, None)
        if only is None or name in only:
            fields[name] = data
        else:
            fields[name] = {"type": data["type"], "note": "working; not shown"}
    payload = {"template_id": spec.template_id, "site": spec.site, "fields": fields}
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()


def make_validator(spec: Spec, names: Sequence[str]) -> Callable[[Any], dict[str, SpecField]]:
    """Validate the repair answer: only the failing fields, same types, valid spec fields."""
    wanted = set(names)

    def validate(obj: Any) -> dict[str, SpecField]:
        if not isinstance(obj, dict) or not isinstance(obj.get("fields"), dict):
            raise ValueError('expected a JSON object {"fields": {"<field name>": {...}}}')
        out: dict[str, SpecField] = {}
        errors: list[str] = []
        for name, raw in obj["fields"].items():
            if name not in wanted:
                errors.append(f"{name}: not one of the failing fields {sorted(wanted)}")
                continue
            if not isinstance(raw, dict):
                errors.append(f"{name}: field definition must be an object")
                continue
            old = spec.fields[name]
            data = {k: v for k, v in raw.items() if k not in DROPPED_KEYS}
            data.setdefault("type", old.type)
            if data["type"] != old.type:
                errors.append(f"{name}: must keep type {old.type!r}, got {data['type']!r}")
                continue
            data["description"] = old.description
            data["source_candidates"] = list(old.source_candidates)
            try:
                out[name] = _FIELD_ADAPTER.validate_python(data)
            except ValidationError as exc:
                msgs = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                errors.append(f"{name}: {msgs}")
        if errors:
            raise ValueError("; ".join(errors))
        if not out:
            raise ValueError("no field definition returned")
        return out

    return validate


def request_repair(
    llm: LLM,
    spec: Spec,
    names: Sequence[str],
    *,
    situation: str,
    failures: str,
    expected: str,
    page: str,
    page_id: int,
) -> dict[str, SpecField]:
    """One repair request (``complete_json`` retries once on invalid JSON / fields)."""
    from ditto.llm import complete_json

    prompt = load_prompt(REPAIR_PROMPT, REPAIR_PROMPT_VERSION)
    messages = prompt.messages(
        template_id=spec.template_id,
        situation=situation,
        spec=spec_for_prompt(spec),
        failures=failures,
        expected=expected,
        page_id=str(page_id),
        page=page,
    )
    return complete_json(
        llm,
        messages,
        validate=make_validator(spec, names),
        prompt_version=prompt.id,
        template_id=spec.template_id,
    )


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def repair_variants(old: SpecField, new: SpecField, winner_xpath: str) -> list[SpecField]:
    """Field variants worth scoring after a repair: the answer as given, and the answer
    combined with the old winning selector (which still works on the other groups)."""
    variants: list[SpecField] = [new]
    if isinstance(new, (ScalarField, ListField)):
        merged = _dedupe([*new.candidates[:2], winner_xpath])
        if len(merged) >= 2 and merged != list(new.candidates):
            variants.append(
                new.model_copy(update={"candidates": merged, "selected": 0, "fallbacks": []})
            )
    elif isinstance(new, RecordListField) and isinstance(old, RecordListField):
        if new.columns == old.columns and new.container != old.container:
            alternates = _dedupe([new.container, *old.container_alternates])
            alternates = [x for x in alternates if x != old.container][:2]
            variants.append(old.model_copy(update={"container_alternates": alternates}))
    elif (
        isinstance(new, KeyValueField)
        and isinstance(old, KeyValueField)
        and (new.key, new.value) == (old.key, old.value)
        and new.pairs != old.pairs
    ):
        alternates = [x for x in _dedupe([new.pairs, *old.pairs_alternates]) if x != old.pairs]
        variants.append(old.model_copy(update={"pairs_alternates": alternates[:2]}))
    return variants


def describe_nodes(items: Sequence[Any], limit: int = 3) -> str:
    """Short description of what a selector returned on a page."""
    if not items:
        return "0 nodes"
    shown = []
    for it in items[:limit]:
        if isinstance(it, etree._Element):
            text = " ".join("".join(str(t) for t in it.itertext()).split())
        else:
            text = str(it)
        shown.append(repr(_truncate_middle(text, 80)))
    more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return f"{len(items)} node(s): {', '.join(shown)}{more}"
