from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ditto.specgen.models import (
    RESERVED_FIELD_NAMES,
    FieldStats,
    KeyValueField,
    ListField,
    RecordListField,
    ScalarField,
    Spec,
    SpecField,
    spec_json_schema,
)
from ditto.specgen.schema_export import EXAMPLE_SPEC

FIELD = TypeAdapter(SpecField)
GEN = {"model": "openai/gpt-6.1-sol", "prompt_version": "specgen.v1"}


def _spec(fields: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "template_id": "t_1",
        "site": "example.org",
        "generated_by": GEN,
        "fields": fields,
        **extra,
    }


def _errors(data: dict[str, Any]) -> str:
    with pytest.raises(ValidationError) as info:
        FIELD.validate_python(data)
    return str(info.value)


def test_brief_example_validates() -> None:
    spec = Spec.model_validate(EXAMPLE_SPEC)
    assert isinstance(spec.fields["name"], ScalarField)
    assert isinstance(spec.fields["publications"], RecordListField)
    assert isinstance(spec.fields["details"], KeyValueField)
    assert spec.fields["email"].optional
    assert spec.fields["publications"].stats.rows_max == 212
    assert spec.status == "draft" and spec.spec_version == "1"


@pytest.mark.parametrize(
    ("data", "cls"),
    [
        ({"type": "scalar", "candidates": ["//h1[@id='t']", "//header//h1"]}, ScalarField),
        (
            {
                "type": "list",
                "candidates": ["//ul[@class='tags']/li", "//a[@rel='tag']"],
                "selected": 1,
                "fallbacks": [0],
                "transforms": ["normalize_ws"],
                "type_check": "text",
            },
            ListField,
        ),
        (
            {
                "type": "record_list",
                "container": "//table[@id='specs']//tr[td]",
                "container_alternates": ["//table[contains(@class,'specs')]//tr[td]"],
                "columns": {"name": "./td[1]", "value": "./td[2]"},
                "transforms": {"value": ["strip"]},
            },
            RecordListField,
        ),
        (
            {
                "type": "key_value",
                "pairs": "//dl[@class='profile']/dt",
                "pairs_alternates": ["//dl[@id='profile']/dt"],
                "value": "./following-sibling::dd[1]",
                "transforms": {"key": ["strip"], "value": ["normalize_ws"]},
            },
            KeyValueField,
        ),
    ],
)
def test_discriminated_union_round_trip(data: dict[str, Any], cls: type) -> None:
    field = FIELD.validate_python(data)
    assert type(field) is cls
    dumped = json.loads(FIELD.dump_json(field, exclude_none=True))
    assert type(FIELD.validate_python(dumped)) is cls
    assert FIELD.validate_python(dumped) == field


def test_type_is_required_to_pick_a_field_class() -> None:
    assert "Unable to extract tag using discriminator" in _errors({"candidates": ["//h1", "//h2"]})
    assert "does not match any of the expected tags" in _errors(
        {"type": "table", "candidates": ["//h1", "//h2"]}
    )


def test_load_save_round_trip(tmp_path: Path) -> None:
    spec = Spec.model_validate(EXAMPLE_SPEC)
    path = tmp_path / "specs" / "t_3f9a.json"
    spec.save(path)
    text = path.read_text()
    assert text.endswith("\n") and '"description"' not in text  # None values are omitted
    loaded = Spec.load(path)
    assert loaded == spec
    assert loaded.model_dump() == spec.model_dump()


def test_extra_keys_rejected_but_allowed_in_stats_and_generated_by() -> None:
    data = _spec(
        {
            "name": {
                "type": "scalar",
                "candidates": ["//h1[@id='t']", "//header//h1"],
                "stats": {"presence_rate": 1.0, "agreement": 0.97},
            }
        },
        generated_by={**GEN, "temperature": 0},
    )
    spec = Spec.model_validate(data)
    assert spec.fields["name"].stats.model_extra == {"agreement": 0.97}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Spec.model_validate({**data, "notes": "x"})
    bad = copy.deepcopy(data)
    bad["fields"]["name"]["selector"] = "//h1"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Spec.model_validate(bad)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"candidates": ["//h1[@id='t']"]}, "at least 2 items"),
        (
            {"candidates": ["//h1[@id='a']", "//h1[@id='b']", "//h1[@id='c']", "//h1[@id='d']"]},
            "at most 3",
        ),
        ({"candidates": ["//h1[@id='t']", " //h1[@id='t'] "]}, "distinct"),
        ({"selected": 2}, "selected=2 out of range"),
        ({"selected": -1}, "greater than or equal to 0"),
        ({"fallbacks": [5]}, "fallbacks index out of range"),
        ({"transforms": ["upper"]}, "unknown transform 'upper'"),
        ({"transforms": ["regex('(')"]}, "invalid regex"),
        ({"type_check": "phone"}, "Input should be 'email'"),
        ({"candidates": ["//div", "//h1[@id='t']"]}, "candidates[0]: purely positional"),
        (
            {"candidates": ["//h1[@id='t']", "//a[matches(., 'x')]"]},
            "candidates[1]: uses matches()",
        ),
        ({"candidates": ["//h1[@id='t']", "//div[@data-x-cand='c1']"]}, "data-x-"),
        ({"stats": {"presence_rate": 1.5}}, "less than or equal to 1"),
        ({"stats": {"rows_min": -1}}, "greater than or equal to 0"),
    ],
)
def test_selector_field_validation(change: dict[str, Any], message: str) -> None:
    data = {"type": "scalar", "candidates": ["//h1[@id='t']", "//header//h1"], **change}
    assert message in _errors(data)


def test_all_selector_errors_reported_together() -> None:
    msg = _errors(
        {
            "type": "list",
            "candidates": ["//div", "//a[lower-case(.)='x']"],
            "transforms": ["cast(bool)"],
        }
    )
    assert "candidates[0]: purely positional" in msg
    assert "candidates[1]: uses lower-case()" in msg
    assert "transform 'cast(bool)'" in msg


def test_record_list_validation() -> None:
    base = {
        "type": "record_list",
        "container": "//table[@class='pubs']//tr[td]",
        "columns": {"title": "./td[2]"},
    }
    # headerless (column_source llm) tables may address cells by position...
    assert isinstance(FIELD.validate_python(base), RecordListField)
    # ...but header-row tables must use header text
    assert "columns.title: purely positional" in _errors({**base, "column_source": "header_row"})
    assert "relative selector must start with '.'" in _errors(
        {**base, "columns": {"title": "//td[@class='t']"}}
    )
    assert "container: purely positional" in _errors({**base, "container": "//table//tr"})
    assert "transforms for unknown columns: ['year']" in _errors(
        {**base, "transforms": {"year": ["cast(int)"]}}
    )
    assert "column names must not be empty" in _errors({**base, "columns": {" ": "./td[1]"}})
    assert "at least 1 item" in _errors({**base, "columns": {}})
    assert "at most 2 items" in _errors(
        {**base, "container_alternates": ["//tr[@a='1']", "//tr[@a='2']", "//tr[@a='3']"]}
    )
    assert "container_alternates[0]: purely positional" in _errors(
        {**base, "container_alternates": ["//tr"]}
    )


def test_key_value_validation() -> None:
    base = {
        "type": "key_value",
        "pairs": "//dl[@class='p']/dt",
        "value": "./following-sibling::dd[1]",
    }
    field = FIELD.validate_python(base)
    assert isinstance(field, KeyValueField) and field.key == "."
    assert "pairs: purely positional" in _errors({**base, "pairs": "//dl/dt"})
    assert "pairs_alternates[0]: uses ends-with()" in _errors(
        {**base, "pairs_alternates": ["//dt[ends-with(., ':')]"]}
    )
    assert "key: relative selector must start with '.'" in _errors({**base, "key": "//dt"})
    assert "Input should be 'key' or 'value'" in _errors({**base, "transforms": {"label": []}})
    assert "transform 'split(:, x)'" in _errors({**base, "transforms": {"key": ["split(:, x)"]}})


@pytest.mark.parametrize("name", [*RESERVED_FIELD_NAMES, "_meta", "_private", " "])
def test_reserved_field_names(name: str) -> None:
    data = _spec({name: {"type": "scalar", "candidates": ["//h1[@id='t']", "//header//h1"]}})
    with pytest.raises(ValidationError, match="reserved or empty field names"):
        Spec.model_validate(data)


def test_spec_level_validation() -> None:
    field = {"type": "scalar", "candidates": ["//h1[@id='t']", "//header//h1"]}
    with pytest.raises(ValidationError, match="both in fields and not_present"):
        Spec.model_validate(_spec({"name": field}, not_present={"name": "not on page"}))
    with pytest.raises(ValidationError, match="at least 1 character"):
        Spec.model_validate({**_spec({"name": field}), "template_id": ""})
    with pytest.raises(ValidationError, match="Input should be '1'"):
        Spec.model_validate({**_spec({"name": field}), "spec_version": "2"})
    with pytest.raises(ValidationError, match="Input should be 'draft' or 'final'"):
        Spec.model_validate({**_spec({"name": field}), "status": "done"})
    empty = Spec.model_validate(_spec({}, not_present={"price": "no price on these pages"}))
    assert empty.fields == {} and empty.not_present == {"price": "no price on these pages"}


def test_constructing_models_in_python_defaults_type() -> None:
    field = ScalarField(candidates=["//h1[@id='t']", "//header//h1"])
    assert field.type == "scalar" and field.stats == FieldStats()
    spec = Spec.model_validate(_spec({"title": field.model_dump()}))
    assert spec.fields["title"] == field


def test_json_schema_marks_type_required() -> None:
    schema = spec_json_schema()
    for name in ("ScalarField", "ListField", "RecordListField", "KeyValueField"):
        assert schema["$defs"][name]["required"][0] == "type"
    assert schema["$defs"]["ScalarField"]["properties"]["candidates"]["minItems"] == 2
    assert set(schema["required"]) == {"template_id", "site", "generated_by", "fields"}
