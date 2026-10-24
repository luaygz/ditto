from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from llm_fakes import PRICES, FakeAdapter, make_settings
from pydantic import BaseModel

from ditto.llm.base import Completion, InvalidResponse, Message, Usage
from ditto.llm.client import CachedLLM
from ditto.llm.structured import (
    JSON_ONLY_INSTRUCTION,
    complete_json,
    parse_json,
    strip_code_fences,
    validation_errors,
    with_json_instruction,
)


class Answer(BaseModel):
    name: str
    count: int


MSGS: list[Message] = [
    {"role": "system", "content": "Extract the record."},
    {"role": "user", "content": "<html>...</html>"},
]


def _llm(tmp_path: Path, adapter: FakeAdapter) -> CachedLLM:
    return CachedLLM(
        model="openai/gpt-test",
        role="generation",
        adapter=adapter,
        settings=make_settings(tmp_path),
        prices=PRICES,
        sleep=lambda _s: None,
        rng=random.Random(0),
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('  ```json\n{"a": 1}\n```  ', '{"a": 1}'),
        ('```JSON\n{"a": 1}\n```', '{"a": 1}'),
        ("```\n[1, 2]\n```", "[1, 2]"),
        ('Here is the spec:\n```json\n{"a": 1}\n```\nLet me know!', '{"a": 1}'),
        ('```json\n{"a": 1}', '{"a": 1}'),  # truncated: no closing fence
        ("```", ""),
        ('\ufeff{"a": 1}', '{"a": 1}'),
    ],
)
def test_strip_code_fences(raw: str, expected: str) -> None:
    assert strip_code_fences(raw) == expected


def test_parse_json_is_defensive() -> None:
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('Sure. {"a": 1} Hope this helps.') == {"a": 1}
    assert parse_json("The list: [1, 2, 3].") == [1, 2, 3]
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_json('{"a": 1,,}')
    with pytest.raises(ValueError, match="empty"):
        parse_json("   ")


def test_with_json_instruction() -> None:
    out = with_json_instruction(MSGS)
    assert out[0]["content"].endswith(JSON_ONLY_INSTRUCTION)
    assert MSGS[0]["content"] == "Extract the record."  # input untouched
    assert with_json_instruction(out) == out  # idempotent
    bare = with_json_instruction([MSGS[1]])
    assert bare[0] == {"role": "system", "content": JSON_ONLY_INSTRUCTION}


def test_first_answer_valid(tmp_path: Path) -> None:
    adapter = FakeAdapter(['```json\n{"name": "x", "count": 3}\n```'])
    value = complete_json(
        _llm(tmp_path, adapter),
        MSGS,
        validate=Answer.model_validate,
        prompt_version="t.v1",
        template_id="t_1",
        max_tokens=123,
    )
    assert value == Answer(name="x", count=3)
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["max_tokens"] == 123
    assert JSON_ONLY_INSTRUCTION in adapter.calls[0]["messages"][0]["content"]


def test_retry_appends_exact_errors(tmp_path: Path) -> None:
    bad = '{"name": "x", "count": "many"}'
    adapter = FakeAdapter([bad, '{"name": "x", "count": 2}'])
    value = complete_json(
        _llm(tmp_path, adapter), MSGS, validate=Answer.model_validate, prompt_version="t.v1"
    )
    assert value.count == 2
    assert len(adapter.calls) == 2
    retry = adapter.calls[1]["messages"]
    assert retry[:2] == adapter.calls[0]["messages"]
    assert retry[2] == {"role": "assistant", "content": bad}
    assert retry[3]["role"] == "user" and len(retry) == 4
    assert (
        "- count: Input should be a valid integer, unable to parse string as an integer"
        in retry[3]["content"]
    )


def test_invalid_twice_raises_typed_error(tmp_path: Path) -> None:
    adapter = FakeAdapter(["not json at all", '{"name": "x"}'])
    with pytest.raises(InvalidResponse) as err:
        complete_json(
            _llm(tmp_path, adapter),
            MSGS,
            validate=Answer.model_validate,
            prompt_version="t.v1",
            template_id="t_9",
        )
    exc = err.value
    assert exc.raw == '{"name": "x"}'
    assert exc.errors == ["count: Field required"]
    assert "t_9" in str(exc) and "openai/gpt-test" in str(exc)
    assert "not valid JSON" in adapter.calls[1]["messages"][-1]["content"]


def test_retry_is_replayed_from_cache(tmp_path: Path) -> None:
    script = ['{"name": 1}', '{"name": "x", "count": 1}']
    complete_json(
        _llm(tmp_path, FakeAdapter(list(script))),
        MSGS,
        validate=Answer.model_validate,
        prompt_version="t.v1",
    )
    replay = FakeAdapter()
    value = complete_json(
        _llm(tmp_path, replay), MSGS, validate=Answer.model_validate, prompt_version="t.v1"
    )
    assert value.count == 1 and replay.calls == []


def test_plain_validators_and_truncation_note(tmp_path: Path) -> None:
    def needs_items(data: Any) -> list[str]:
        if not isinstance(data, dict):
            raise TypeError("expected an object")
        items = data["items"]
        if not items:
            raise ValueError("items is empty\nadd at least one item")
        return [str(i) for i in items]

    truncated = Completion(
        text='{"items": [',
        usage=Usage(10, 10),
        model="openai/gpt-test",
        meta={"finish_reason": "length"},
    )
    adapter = FakeAdapter([truncated, '{"other": 1}'])
    with pytest.raises(InvalidResponse) as err:
        complete_json(_llm(tmp_path, adapter), MSGS, validate=needs_items, prompt_version="v")
    first_feedback = adapter.calls[1]["messages"][-1]["content"]
    assert "cut off at the output token limit" in first_feedback
    assert err.value.errors == ["missing key: 'items'"]

    adapter = FakeAdapter(['{"items": []}', '{"items": ["a"]}'])
    assert complete_json(
        _llm(tmp_path, adapter), MSGS, validate=needs_items, prompt_version="v2"
    ) == ["a"]
    assert (
        "- items is empty\n- add at least one item" in adapter.calls[1]["messages"][-1]["content"]
    )


def test_validation_errors_formatting() -> None:
    try:
        Answer.model_validate({"name": None})
    except ValueError as exc:
        assert validation_errors(exc) == [
            "name: Input should be a valid string",
            "count: Field required",
        ]
    assert validation_errors(TypeError("")) == ["TypeError"]
