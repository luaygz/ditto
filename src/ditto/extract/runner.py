"""Run one spec over one cleaned page. No LLM, no I/O: this is what ``Spec.apply`` calls."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from lxml import etree

from ditto.fingerprint.dom import clean_html
from ditto.specgen.transforms import apply_transforms

if TYPE_CHECKING:
    from ditto.specgen.models import (
        KeyValueField,
        ListField,
        RecordListField,
        ScalarField,
        Spec,
        SpecField,
    )


@lru_cache(maxsize=16384)
def compile_xpath(expr: str) -> etree.XPath:
    return etree.XPath(expr)


def evaluate(expr: str, node: etree._Element) -> list[Any]:
    """Evaluate an XPath and always return a list of result items (elements or strings)."""
    try:
        result = compile_xpath(expr)(node)
    except etree.XPathError:
        return []
    if isinstance(result, list):
        return [r for r in result if not isinstance(r, etree._Comment)]
    if isinstance(result, bool):
        return [result] if result else []
    if isinstance(result, float):
        return [] if result != result else [result]  # NaN -> nothing
    if isinstance(result, str):
        return [result] if result.strip() else []
    return [result]


def _first_value(items: list[Any], transforms: list[str]) -> Any:
    for item in items:
        value = apply_transforms(item, transforms)
        if value is not None:
            return value
    return None


def scalar_value(field: ScalarField, root: etree._Element) -> Any:
    order = [field.selected, *[i for i in field.fallbacks if i != field.selected]]
    for idx in order:
        value = _first_value(evaluate(field.candidates[idx], root), field.transforms)
        if value is not None:
            return value
    return None


def list_value(field: ListField, root: etree._Element) -> list[Any]:
    order = [field.selected, *[i for i in field.fallbacks if i != field.selected]]
    for idx in order:
        values = [
            apply_transforms(item, field.transforms)
            for item in evaluate(field.candidates[idx], root)
        ]
        kept = [v for v in values if v is not None]
        if kept:
            return kept
    return []


def record_rows(field: RecordListField, root: etree._Element) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in evaluate(field.container, root):
        if not isinstance(row, etree._Element):
            continue
        record = {
            name: _first_value(evaluate(xpath, row), field.transforms.get(name, []))
            for name, xpath in field.columns.items()
        }
        if any(v is not None for v in record.values()):
            rows.append(record)
    return rows


def key_values(field: KeyValueField, root: etree._Element) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key_t = field.transforms.get("key", [])
    value_t = field.transforms.get("value", [])
    for pair in evaluate(field.pairs, root):
        if not isinstance(pair, etree._Element):
            continue
        key = _first_value(evaluate(field.key, pair), key_t)
        if key is None:
            continue
        key = str(key).rstrip(":").strip()
        if key and key not in out:
            out[key] = _first_value(evaluate(field.value, pair), value_t)
    return out


def field_value(field: SpecField, root: etree._Element) -> Any:
    if field.type == "scalar":
        return scalar_value(field, root)
    if field.type == "list":
        return list_value(field, root)
    if field.type == "record_list":
        return record_rows(field, root)
    return key_values(field, root)


def is_missing(value: Any) -> bool:
    return value is None or value == [] or value == {}


def apply_spec(spec: Spec, root: etree._Element) -> dict[str, Any]:
    """Apply every field of ``spec`` to an already-cleaned tree."""
    out: dict[str, Any] = {}
    missing: list[str] = []
    for name, field in spec.fields.items():
        value = field_value(field, root)
        out[name] = value
        if is_missing(value):
            missing.append(name)
    out["_meta"] = {"missing": missing, "spec_version": spec.spec_version}
    return out


def apply_spec_html(spec: Spec, html: str | bytes) -> dict[str, Any]:
    return apply_spec(spec, clean_html(html))
