from __future__ import annotations

import json
from pathlib import Path

import pytest

from ditto.specgen.models import Spec
from ditto.specgen.schema_export import (
    EXAMPLE_SPEC,
    JSON_SCHEMA_DIALECT,
    _type_str,
    format_doc,
    main,
    schema_text,
    spec_schema,
    write_format_doc,
    write_schema,
)

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_example_spec_validates_and_round_trips() -> None:
    spec = Spec.model_validate(EXAMPLE_SPEC)
    assert json.loads(spec.model_dump_json(exclude_unset=True)) == EXAMPLE_SPEC


@pytest.mark.parametrize(
    ("prop", "text"),
    [
        (
            {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "array of string (at least 1)",
        ),
        ({"type": "object"}, "object"),
        ({"type": "integer", "minimum": 0, "maximum": 5}, "integer in [0, 5]"),
        ({"oneOf": [{"type": "string"}, {"type": "null"}]}, "one of string, null"),
    ],
)
def test_type_rendering(prop: dict[str, object], text: str) -> None:
    assert _type_str(prop) == text


def test_schema_content() -> None:
    schema = spec_schema()
    assert schema["$schema"] == JSON_SCHEMA_DIALECT
    assert schema["title"] == "Spec"
    assert set(schema["$defs"]) >= {"ScalarField", "RecordListField", "FieldStats"}
    assert json.loads(schema_text()) == schema


def test_write_functions(tmp_path: Path) -> None:
    schema_path = write_schema(tmp_path / "a" / "spec.schema.json")
    doc_path = write_format_doc(tmp_path / "b" / "spec-format.md")
    assert schema_path.read_text(encoding="utf-8") == schema_text()
    doc = doc_path.read_text(encoding="utf-8")
    assert doc == format_doc()
    for heading in (
        "## Field types",
        "## Top-level keys",
        "### `record_list`",
        "### `stats`",
        "## Selector rules",
        "## Transforms",
        "## Extraction semantics",
        "## Example",
    ):
        assert heading in doc
    # every model field appears in a table with its description
    assert "| `column_source` |" in doc and "header text, never by position" in doc
    assert "| `cast(int\\|float\\|date)` |" in doc
    assert '"year": 2021' in doc  # the example's extraction output is rendered


def test_main_writes_default_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    main()
    assert (tmp_path / "docs" / "spec.schema.json").read_text(encoding="utf-8") == schema_text()
    assert (tmp_path / "docs" / "spec-format.md").read_text(encoding="utf-8") == format_doc()


@pytest.mark.skipif(not (DOCS / "spec.schema.json").exists(), reason="no source checkout")
def test_committed_docs_are_current() -> None:
    """Regenerate with ``uv run python -m ditto.specgen.schema_export`` when this fails."""
    assert (DOCS / "spec.schema.json").read_text(encoding="utf-8") == schema_text()
    assert (DOCS / "spec-format.md").read_text(encoding="utf-8") == format_doc()
