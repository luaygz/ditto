from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest
from baseline_helpers import FakeLLM, load_module

from ditto.evaluation.baseline_api import PageInput
from ditto.llm.base import BudgetExceeded, Message
from ditto.schema import FieldsSchema

SCHEMA = FieldsSchema.from_obj(
    {
        "title": "the title of the book",
        "author": {"description": "an author", "examples": ["Jane Austen"]},
        "formats": {"type": "list", "description": "available formats"},
    }
)

HTML = """<html><head><title>Emma</title><script>x()</script></head><body>
<div class="product"><h1 class="title">Emma</h1><p>by <a class="author" href="/a/1">Jane Austen</a></p>
<ul class="formats"><li>Paperback</li><li>Hardcover</li></ul></div></body></html>"""


@pytest.fixture(scope="module")
def naive() -> ModuleType:
    return load_module("naive_llm")


def _pages(n: int) -> list[PageInput]:
    return [PageInput(f"{i:04d}", f"http://x/{i}", HTML, "t1") for i in range(n)]


def _answer(messages: list[Message]) -> str:
    assert messages[0]["role"] == "system"
    user = messages[-1]["content"]
    assert "Jane Austen" in user and "x()" not in user
    return '```json\n{"title": "Emma", "author": "Jane Austen", "formats": ["Paperback", "Hardcover"], "isbn": "9"}\n```'


def test_build_messages_shape(naive: ModuleType) -> None:
    msgs = naive.build_messages(_pages(1)[0], SCHEMA)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert "Page URL: http://x/0" in user
    fields = json.loads(user.split("this page's values:\n", 1)[1].split("\n\nPage HTML", 1)[0])
    assert [f["name"] for f in fields] == ["title", "author", "formats"]
    assert fields[1]["examples"] == ["Jane Austen"]
    assert '<h1 class="title">Emma</h1>' in user
    assert "{{" not in user and "<script" not in user
    assert "JSON only" in msgs[0]["content"]


def test_validator(naive: ModuleType) -> None:
    validate = naive.make_validator(SCHEMA)
    assert validate([{"title": "A"}]) == {"title": "A", "author": None, "formats": []}
    with pytest.raises(ValueError, match="expected one JSON object"):
        validate(["a", "b"])


def test_run_baseline_with_fake_llm(naive: ModuleType, tmp_path: Path) -> None:
    llm = FakeLLM(_answer)
    out = naive.run_baseline(_pages(5), SCHEMA, out_dir=tmp_path, max_pages=3, seed=1, llm=llm)
    assert out.pages_run == 3 and len(out.predictions) == 3 and out.calls == 3
    pred = next(iter(out.predictions.values()))
    assert pred == {"title": "Emma", "author": "Jane Austen", "formats": ["Paperback", "Hardcover"]}
    assert out.model == "openai/gpt-6.1-sol"
    assert out.input_tokens > 0 and out.output_tokens > 0
    assert out.cost_usd == pytest.approx((out.input_tokens * 2 + out.output_tokens * 10) / 1e6)
    assert out.spend_usd == pytest.approx(0.003)
    assert all(c["prompt_version"] == "naive_baseline.v1" for c in llm.calls)
    assert {c["template_id"] for c in llm.calls} == {"t1"}
    assert any("per 1,000 pages" in n for n in out.notes)
    rows = [
        json.loads(line) for line in (tmp_path / "naive_llm.pages.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3 and all(r["calls"] == 1 and r["error"] is None for r in rows)
    assert sum(r["input_tokens"] for r in rows) == out.input_tokens
    summary = json.loads((tmp_path / "naive_llm.summary.json").read_text())
    assert summary["prompt"] == "naive_baseline.v1" and summary["pages_run"] == 3
    assert (tmp_path / "naive_llm.predictions.jsonl").is_file()


def test_cached_calls_cost_nominal_but_spend_nothing(naive: ModuleType, tmp_path: Path) -> None:
    llm = FakeLLM(_answer, cached=True)
    out = naive.run_baseline(_pages(2), SCHEMA, out_dir=tmp_path, llm=llm)
    assert out.cost_usd > 0 and out.spend_usd == 0.0
    assert any("2/2 calls served from cache" in n for n in out.notes)


def test_invalid_answers_score_empty(naive: ModuleType, tmp_path: Path) -> None:
    llm = FakeLLM(lambda _m: "I cannot help with that")
    out = naive.run_baseline(_pages(2), SCHEMA, out_dir=tmp_path, llm=llm, concurrency=1)
    assert out.calls == 4  # one retry per page
    assert all(
        p == {"title": None, "author": None, "formats": []} for p in out.predictions.values()
    )
    assert any("got no valid answer" in n for n in out.notes)


def test_budget_exceeded_propagates(naive: ModuleType, tmp_path: Path) -> None:
    def broke(_m: list[Message]) -> str:
        raise BudgetExceeded("over budget")

    with pytest.raises(BudgetExceeded):
        naive.run_baseline(_pages(1), SCHEMA, out_dir=tmp_path, llm=FakeLLM(broke))
