"""Generate ``docs/spec.schema.json`` and ``docs/spec-format.md`` from the Pydantic spec models.

Both files are build artifacts of :mod:`ditto.specgen.models` (field tables come from model
fields and their descriptions; rules come from the lint, transform, runner and typecheck module
docstrings). Regenerate after changing any of them::

    uv run python -m ditto.specgen.schema_export
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ditto.extract import runner
from ditto.specgen import lint, models, transforms, typecheck
from ditto.specgen.models import (
    FieldStats,
    GeneratedBy,
    KeyValueField,
    ListField,
    RecordListField,
    ScalarField,
    Spec,
    _BaseField,
    _SelectorField,
    spec_json_schema,
)

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: The example from the product brief, verbatim. It must validate (tested).
EXAMPLE_SPEC: dict[str, Any] = {
    "spec_version": "1",
    "template_id": "t_3f9a",
    "site": "example.org",
    "generated_by": {
        "model": "<model id>",
        "prompt_version": "specgen.v1",
        "pages_seen": ["/staff/a-khoury", "/staff/l-haddad"],
    },
    "fields": {
        "name": {
            "type": "scalar",
            "candidates": ["//h1[@itemprop='name']", "//header//h1"],
            "selected": 0,
            "transforms": ["normalize_ws"],
            "stats": {"presence_rate": 1.0, "multiplicity": 1.0, "type_check_pass": 1.0},
        },
        "email": {
            "type": "scalar",
            "candidates": [
                "//dt[normalize-space()='Email']/following-sibling::dd[1]//a",
                "//a[starts-with(@href,'mailto:')]",
            ],
            "selected": 0,
            "transforms": ["normalize_ws"],
            "type_check": "email",
            "optional": True,
            "stats": {"presence_rate": 0.41, "multiplicity": 1.0, "type_check_pass": 0.998},
        },
        "publications": {
            "type": "record_list",
            "container": "//table[contains(@class,'pubs')]//tr[td]",
            "columns": {
                "title": "./td[count(ancestor::table//th[normalize-space()='Title']"
                "/preceding-sibling::th)+1]",
                "year": "./td[count(ancestor::table//th[normalize-space()='Year']"
                "/preceding-sibling::th)+1]",
            },
            "column_source": "header_row",
            "transforms": {"year": ["cast(int)"]},
            "stats": {"presence_rate": 0.77, "rows_min": 1, "rows_max": 212},
        },
        "details": {
            "type": "key_value",
            "pairs": "//dl[@class='profile']/dt",
            "key": ".",
            "value": "./following-sibling::dd[1]",
            "stats": {"presence_rate": 0.99, "keys_seen": 14},
        },
    },
}

#: A page of the example's template; ``Spec.apply`` on it is shown in the format doc.
EXAMPLE_HTML = """\
<html><head><title>Dr. Amal Khoury | Example University</title>
<script>trackPageView();</script></head>
<body>
<header class="site"><nav><a href="/">Home</a> <a href="/staff">Staff</a></nav></header>
<main>
  <header class="profile-head"><h1 itemprop="name">Dr. Amal
      Khoury</h1></header>
  <dl class="profile">
    <dt>Email</dt><dd><a href="mailto:a.khoury@example.org">a.khoury@example.org</a></dd>
    <dt>Office:</dt><dd>Building B, Room 214</dd>
    <dt>Phone</dt><dd>+961 1 350 000 ext. 4321</dd>
  </dl>
  <table class="pubs">
    <tr><th>Year</th><th>Title</th></tr>
    <tr><td>2021</td><td>Wrapper induction with one call per template</td></tr>
    <tr><td>2019</td><td>Template clustering at web scale</td></tr>
  </table>
</main>
</body></html>
"""


def spec_schema() -> dict[str, Any]:
    """The JSON Schema written to ``docs/spec.schema.json``."""
    return {"$schema": JSON_SCHEMA_DIALECT, **spec_json_schema()}


def _write(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def schema_text() -> str:
    return json.dumps(spec_schema(), indent=2, ensure_ascii=False) + "\n"


def write_schema(path: str | Path = "docs/spec.schema.json") -> Path:
    """Write the spec JSON Schema; returns the path written."""
    return _write(path, schema_text())


# ---------------------------------------------------------------------------- markdown rendering


def _code(text: str) -> str:
    """reST ``code`` and :func:`name` -> Markdown `code`."""
    return re.sub(r":\w+:`([^`]+)`", r"`\1`", re.sub(r"``(.+?)``", r"`\1`", text))


def _md(text: str) -> str:
    """Inline reST -> Markdown for a table cell (pipes escaped)."""
    return _code(text).replace("|", "\\|")


def _rst_to_md(doc: str) -> str:
    """Docstring -> Markdown: inline code, and ``::`` literal blocks become fenced blocks."""
    lines = doc.strip("\n").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::"):
            out.append(_code(line.rstrip()[:-2].rstrip()) + ":")
            i += 1
            block: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                block.append(lines[i][4:])
                i += 1
            while block and not block[0].strip():
                block.pop(0)
            while block and not block[-1].strip():
                block.pop()
            out += ["", "```text", *block, "```", ""]
            continue
        out.append(_code(line))
        i += 1
    return "\n".join(out).strip() + "\n"


def _type_str(prop: dict[str, Any]) -> str:
    if "const" in prop:
        return f'`"{prop["const"]}"`'
    if "enum" in prop:
        return " or ".join(f'`"{v}"`' for v in prop["enum"])
    if "$ref" in prop:
        return f"object (`{str(prop['$ref']).rsplit('/', 1)[-1]}`)"
    if "anyOf" in prop:
        kinds = [_type_str(p) for p in prop["anyOf"] if p.get("type") != "null"]
        nullable = any(p.get("type") == "null" for p in prop["anyOf"])
        return " or ".join(kinds) + (" or null" if nullable else "")
    if "oneOf" in prop:
        return "one of " + ", ".join(_type_str(p) for p in prop["oneOf"])
    kind = prop.get("type", "any")
    if kind == "array":
        out = f"array of {_type_str(prop.get('items', {}))}"
        lo, hi = prop.get("minItems"), prop.get("maxItems")
        if lo is not None and hi is not None:
            out += f" ({lo}–{hi} items)"
        elif hi is not None:
            out += f" (at most {hi})"
        elif lo is not None:
            out += f" (at least {lo})"
        return out
    if kind == "object":
        keys = prop.get("propertyNames", {}).get("enum")
        values = prop.get("additionalProperties")
        if isinstance(values, dict):
            key_part = " or ".join(f"`{k}`" for k in keys) if keys else "name"
            out = f"object: {key_part} → {_type_str(values)}"
            if prop.get("minProperties"):
                out += f" (at least {prop['minProperties']})"
            return out
        return "object"
    bounds = ""
    if "minimum" in prop and "maximum" in prop:
        bounds = f" in [{prop['minimum']}, {prop['maximum']}]"
    elif "minimum" in prop:
        bounds = f" ≥ {prop['minimum']}"
    elif "minLength" in prop:
        bounds = ", non-empty"
    return f"{kind}{bounds}"


def _default_str(model: type[BaseModel], name: str) -> str:
    info = model.model_fields[name]
    if info.is_required():
        return "**required**"
    value = info.get_default(call_default_factory=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    return "`" + json.dumps(value, ensure_ascii=False) + "`"


def _field_table(model: type[BaseModel], names: list[str] | None = None) -> str:
    props = model.model_json_schema()["properties"]
    rows = ["| Key | Type | Default | Meaning |", "| --- | --- | --- | --- |"]
    for name in names if names is not None else list(model.model_fields):
        info = model.model_fields[name]
        rows.append(
            f"| `{name}` | {_type_str(props[name])} | "
            f"{_default_str(model, name)} | {_md(info.description or '')} |"
        )
    return "\n".join(rows) + "\n"


def _own_fields(model: type[BaseModel], base: type[BaseModel]) -> list[str]:
    return [n for n in model.model_fields if n not in base.model_fields]


def _doc(obj: Any) -> str:
    return " ".join((obj.__doc__ or "").split())


def _example_output() -> dict[str, Any]:
    return Spec.model_validate(EXAMPLE_SPEC).apply(EXAMPLE_HTML)


def _json_block(data: Any) -> str:
    return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```\n"


def format_doc() -> str:
    """The full text of ``docs/spec-format.md``."""
    selector_keys = _own_fields(_SelectorField, _BaseField)
    transform_rows = "".join(
        f"| {_md('``' + sig + '``')} | {_md(text)} |\n"
        for sig, text in transforms.TRANSFORM_DOCS.values()
    )
    parts = [
        "# Spec format\n",
        "<!-- Generated by `uv run python -m ditto.specgen.schema_export` from "
        "src/ditto/specgen/models.py and the lint, transform, runner and typecheck modules. "
        "Do not edit by hand. -->\n",
        _rst_to_md(_after_first_line(models.__doc__)),
        "The JSON Schema is [`spec.schema.json`](spec.schema.json). In Python, "
        "`Spec.load(path)` validates a spec file and `spec.apply(html)` runs it on one page "
        "with no other machinery.\n",
        "## Field types\n",
        "| `type` | Output | Shape |\n| --- | --- | --- |\n"
        f"| `scalar` | string, number or null | {_md(_doc(ScalarField))} |\n"
        f"| `list` | array | {_md(_doc(ListField))} |\n"
        f"| `record_list` | array of objects | {_md(_doc(RecordListField))} |\n"
        f"| `key_value` | object | {_md(_doc(KeyValueField))} |\n",
        "## Top-level keys\n",
        _field_table(Spec),
        "### `generated_by`\n",
        _md(_doc(GeneratedBy)) + "\n",
        _field_table(GeneratedBy),
        "## Field definitions\n",
        "Keys every field type accepts:\n",
        _field_table(ScalarField, list(_BaseField.model_fields)),
        "### `scalar` and `list`\n",
        _field_table(ScalarField, ["type", *selector_keys]),
        '`type` is `"list"` for a list field; the other keys are identical.\n',
        "### `record_list`\n",
        _field_table(RecordListField, _own_fields(RecordListField, _BaseField)),
        "### `key_value`\n",
        _field_table(KeyValueField, _own_fields(KeyValueField, _BaseField)),
        "### `stats`\n",
        _md(_doc(FieldStats)) + "\n",
        _field_table(FieldStats),
        "## Selector rules\n",
        _rst_to_md(_after_first_line(lint.__doc__)),
        "## Transforms\n",
        "| Transform | Meaning |\n| --- | --- |\n" + transform_rows,
        _rst_to_md(_after_first_line(transforms.__doc__)),
        "## Extraction semantics\n",
        _rst_to_md(_after_first_line(runner.__doc__)),
        "## Value types (`type_check`)\n",
        _rst_to_md(typecheck.__doc__ or ""),
        "## Example\n",
        "The example from the product brief:\n",
        _json_block(EXAMPLE_SPEC),
        "A page of that template (scripts and `<head>` except `<title>` are stripped before any "
        "selector runs):\n",
        "```html\n" + EXAMPLE_HTML + "```\n",
        "`Spec.load(path).apply(html)` returns:\n",
        _json_block(_example_output()),
        "`publications` finds its columns by header text even though `Year` comes first on "
        "the page; `details` drops the trailing colon of `Office:`.\n",
    ]
    return "\n".join(p.rstrip("\n") + "\n" for p in parts)


def _after_first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n", 1)[-1]


def write_format_doc(path: str | Path = "docs/spec-format.md") -> Path:
    """Write the spec format reference; returns the path written."""
    return _write(path, format_doc())


def main() -> None:
    for written in (write_schema(), write_format_doc()):
        print(f"wrote {written}")


if __name__ == "__main__":  # pragma: no cover
    main()
