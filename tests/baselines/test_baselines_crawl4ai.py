from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from baseline_helpers import load_module

from ditto.evaluation.baseline_api import NotRun, PageInput
from ditto.llm import Settings
from ditto.schema import FieldsSchema

SCHEMA = FieldsSchema.from_obj(
    {
        "title": {"description": "the job title", "examples": ["QA Lead"]},
        "company": "the hiring company",
        "skills": {"type": "list", "description": "required skills"},
    }
)

CSS_SCHEMA = {
    "name": "Job",
    "baseSelector": "div.job",
    "fields": [
        {"name": "title", "selector": "h1.title", "type": "text"},
        {"name": "Company", "selector": "span.company", "type": "text"},
        {
            "name": "skills",
            "selector": "ul.skills li",
            "type": "list",
            "fields": [{"name": "value", "type": "text"}],
        },
    ],
}
# The same schema with exactly the requested names (passes crawl4ai's strict validation).
EXACT = {
    **CSS_SCHEMA,
    "fields": [
        {**CSS_SCHEMA["fields"][0]},
        {**CSS_SCHEMA["fields"][1], "name": "company"},
        CSS_SCHEMA["fields"][2],
    ],
}


def page_html(i: int) -> str:
    return f"""<html><head><title>Job {i}</title></head><body>
<div class="job related"><h1 class="other">Related job</h1></div>
<div class="job"><h1 class="title">Engineer {i}</h1><span class="company">Acme {i}</span>
<ul class="skills"><li>Python</li><li>SQL</li></ul></div></body></html>"""


@pytest.fixture(scope="module")
def c4a() -> ModuleType:
    pytest.importorskip("crawl4ai")
    return load_module("crawl4ai_schema")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(openai_model="gpt-6.1-sol", cache_dir=tmp_path / "cache")


def _pages() -> list[PageInput]:
    pages = [PageInput(f"{i:04d}", f"http://x/{i}", page_html(i), "t1") for i in range(3)]
    pages.append(PageInput("tail", "http://x/tail", page_html(9), None))
    return pages


def test_query_and_example(c4a: ModuleType) -> None:
    query = c4a.build_query(SCHEMA)
    assert "- title (scalar): the job title (e.g. QA Lead)" in query
    assert "- skills (list): required skills" in query
    example = c4a.target_example(SCHEMA)
    assert list(example) == ["title", "company", "skills"]
    assert example["title"] == "QA Lead"
    assert example["company"] == "<the hiring company>"
    assert example["skills"][0] == {"value": "<required skills>"}
    kv = FieldsSchema.from_obj({"specs": {"type": "key_value"}, "rows": {"type": "record_list"}})
    ex = c4a.target_example(kv)
    assert ex["specs"][0].keys() == {"key", "value"} and isinstance(ex["rows"], list)


def test_map_item(c4a: ModuleType) -> None:
    item = {"Title": "A", "details": {"company": "B"}, "skills": [{"value": "x"}]}
    assert c4a.map_item(item, SCHEMA) == {"title": "A", "company": "B", "skills": [{"value": "x"}]}


def test_extract_page_picks_fullest_item(c4a: ModuleType) -> None:
    pred = c4a.extract_page(CSS_SCHEMA, "http://x/1", page_html(1), SCHEMA)
    assert pred == {"title": "Engineer 1", "company": "Acme 1", "skills": ["Python", "SQL"]}
    broken = {"baseSelector": "div[[[", "fields": []}
    assert c4a.extract_page(broken, "u", page_html(1), SCHEMA) == {
        "title": None,
        "company": None,
        "skills": [],
    }


def test_model_target(c4a: ModuleType, settings: Settings) -> None:
    target = c4a.model_target(settings)
    assert target.model == "openai/gpt-6.1-sol" and target.settings is settings
    with pytest.raises(NotRun, match="OpenAI adapter only"):
        c4a.model_target(settings.with_models(baseline="anthropic/claude-x"))
    assert c4a.projected_usd("<p>hi</p>", "openai/gpt-6.1-sol") > 0


def _fake_completion(content: str, prompt_tokens: int = 1000, completion_tokens: int = 50) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def test_generate_schema_miss_records_usage_and_caches(
    c4a: ModuleType, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crawl4ai.utils as c4u

    seen: list[dict[str, Any]] = []

    async def fake(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return _fake_completion(json.dumps(EXACT))

    monkeypatch.setattr(c4u, "aperform_completion_with_backoff", fake)
    target = c4a.model_target(settings)
    cache = tmp_path / "c4a"
    gen = c4a.generate_schema(
        page_html(1), SCHEMA, target=target, template_id="t1", cache_dir=cache
    )
    assert gen["schema"] == EXACT and gen["error"] is None and not gen["cached"]
    assert gen["calls"] == 1 and gen["input_tokens"] == 1000 and gen["output_tokens"] == 50
    assert seen[0]["provider"] == "openai/gpt-6.1-sol"
    assert seen[0]["extra_args"]["temperature"] == 1.0
    ledger = (settings.cache_dir / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 1 and json.loads(ledger[0])["role"] == "baseline"
    # second time: served from the schema cache, no model call
    again = c4a.generate_schema(page_html(1), SCHEMA, target=target, cache_dir=cache)
    assert again["cached"] and again["schema"] == EXACT and len(seen) == 1
    assert c4u.aperform_completion_with_backoff is fake  # the counting wrapper was removed


def test_generate_schema_failure_is_reported(
    c4a: ModuleType, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crawl4ai.utils as c4u

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    monkeypatch.setattr(c4u, "aperform_completion_with_backoff", boom)
    out = c4a.run_baseline(
        _pages(), SCHEMA, out_dir=tmp_path / "out", settings=settings, cache_dir=tmp_path / "c"
    )
    assert out.calls == 0
    assert all(
        p == {"title": None, "company": None, "skills": []} for p in out.predictions.values()
    )
    assert any("schema generation failed for 1 template" in n for n in out.notes)
    row = json.loads((tmp_path / "out" / "crawl4ai_schema.templates.jsonl").read_text())
    assert "provider down" in row["error"]


def test_run_baseline_from_cache(c4a: ModuleType, settings: Settings, tmp_path: Path) -> None:
    cache = tmp_path / "c4a"
    cache.mkdir()
    query, example = c4a.build_query(SCHEMA), c4a.target_example(SCHEMA)
    for i in range(3):  # whichever page is sampled, its schema is cached
        key = c4a._cache_key("openai/gpt-6.1-sol", page_html(i), query, example)
        entry = {
            "schema": CSS_SCHEMA,
            "error": None,
            "calls": 2,
            "input_tokens": 3000,
            "output_tokens": 100,
            "call_log": [
                {"input_tokens": 1000, "output_tokens": 40},
                {"input_tokens": 2000, "output_tokens": 60},
            ],
            "model": "openai/gpt-6.1-sol",
        }
        (cache / f"{key}.json").write_text(json.dumps(entry))
    out = c4a.run_baseline(
        _pages(), SCHEMA, out_dir=tmp_path / "out", settings=settings, cache_dir=cache, workers=1
    )
    assert out.calls == 2 and out.input_tokens == 3000 and out.output_tokens == 100
    assert out.cost_usd == pytest.approx((3000 * 2 + 100 * 10) / 1e6)
    assert out.spend_usd == 0.0 and out.pages_run == 4 and out.model == "openai/gpt-6.1-sol"
    assert out.predictions["0002"] == {
        "title": "Engineer 2",
        "company": "Acme 2",
        "skills": ["Python", "SQL"],
    }
    assert out.predictions["tail"] == {"title": None, "company": None, "skills": []}
    assert any("1 tail page" in n for n in out.notes)
    summary = json.loads((tmp_path / "out" / "crawl4ai_schema.summary.json").read_text())
    assert summary["usage"]["cached_calls"] == 2
