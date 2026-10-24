from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from baseline_helpers import FakeLLM, load_module

from ditto.evaluation.baseline_api import NotRun, PageInput
from ditto.llm.base import Message
from ditto.schema import FieldsSchema

# The upstream vendor code uses bs4's deprecated ``text=`` argument.
pytestmark = pytest.mark.filterwarnings("ignore:The 'text' argument:DeprecationWarning")
SCHEMA = FieldsSchema.from_obj(
    {
        "title": "the job title",
        "company": "the hiring company",
        "skills": {"type": "list", "description": "required skills"},
        "facts": {"type": "key_value", "description": "job facts"},
    }
)
XPATHS = {
    "title": "//h1[@class='title']/text()",
    "company": "//span[@class='company']/text()",
    "skills": "//ul[@class='skills']/li/text()",
}


def page_html(i: int) -> str:
    return f"""<html><head><title>Job {i}</title></head><body>
<div class="job"><h1 class="title">Engineer {i}</h1><span class="company">Acme {i}</span>
<ul class="skills"><li>Python</li><li>SQL {i}</li></ul></div></body></html>"""


@pytest.fixture(scope="module")
def autoscraper() -> ModuleType:
    return load_module("autoscraper")


@pytest.fixture(scope="module")
def vendor(autoscraper: ModuleType) -> ModuleType:
    if not (autoscraper.VENDOR_DIR / "module" / "stepback_crawler.py").is_file():
        pytest.skip("AutoScraper vendor checkout not present (gitignored; cloned on first run)")
    pytest.importorskip("bs4")
    return autoscraper.load_vendor(autoscraper.VENDOR_DIR)


def answer(messages: list[Message]) -> str:
    query = messages[-1]["content"]
    if "judge whether the extracted value is consistent" in query:
        return '{"thought": "matches", "judgement": "yes"}'
    if "choose one that can be best" in query:
        return '{"thought": "first", "number": "0"}'
    m = re.search(r"Please extract the (\w+):", query)
    assert m, query[:200]
    return json.dumps({"thought": "t", "value": "v", "xpath": XPATHS[m.group(1)]})


def test_safe_index_never_evaluates(autoscraper: ModuleType) -> None:
    assert autoscraper._safe_index(" 2 ") == 2
    for bad in ("1+1", "__import__('os')", "", "-1"):
        with pytest.raises(ValueError):
            autoscraper._safe_index(bad)


def test_instruction_and_prediction(autoscraper: ModuleType) -> None:
    text = autoscraper.instruction_for("title", "the job title")
    assert text.startswith("Here's a webpage with detail information. Please extract the title:")
    assert autoscraper.instruction_for("x", "").count("extract the x.") == 1
    pred = autoscraper.to_prediction({"title": ["", " A "], "skills": ["a", "b"]}, SCHEMA)
    assert pred == {"title": "A", "company": None, "skills": ["a", "b"], "facts": {}}


def test_api_hook_caps_calls(autoscraper: ModuleType) -> None:
    hook = autoscraper.ApiHook(FakeLLM(lambda _m: "{}"), max_calls=2)
    hook.start_field("t1")
    assert hook("q1") == "{}" and hook("q2") == "{}"
    with pytest.raises(autoscraper.CallBudgetExhausted):
        hook("q3")
    hook.start_field("t2")
    hook("q4")
    calls = hook.llm.drain()
    assert len(calls) == 3


def test_synthesize_rules_records_cap(autoscraper: ModuleType) -> None:
    class Crawler:
        def rule_synthesis(
            self, name: str, seeds: list[str], instruction: str, max_token: int
        ) -> list[str]:
            assert max_token == autoscraper.MAX_TOKEN
            if "company" in instruction:
                raise autoscraper.CallBudgetExhausted("cap")
            return ["//h1"]

    hook = autoscraper.ApiHook(FakeLLM(lambda _m: "{}"))
    rules, errors = autoscraper.synthesize_rules(Crawler(), hook, "t1", ["<p/>"], SCHEMA)
    assert rules == {"title": ["//h1"], "company": [], "skills": ["//h1"]}  # facts not attempted
    assert errors == {"company": "cap"}


def test_missing_vendor_is_not_run(autoscraper: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(NotRun, match="vendor checkout missing"):
        autoscraper.ensure_vendor(tmp_path / "nowhere", clone=False)
    with pytest.raises(NotRun):
        autoscraper.run_baseline(
            [], SCHEMA, out_dir=tmp_path, vendor_dir=tmp_path / "nowhere", clone=False
        )


def test_load_vendor_is_sandboxed(vendor: ModuleType) -> None:
    assert vendor.eval("3") == 3  # module-level eval is the safe index parser
    with pytest.raises(ValueError):
        vendor.eval("__import__('os')")
    assert not any(n == "utils" or n.startswith(("utils.", "module.")) for n in sys.modules)


def test_run_baseline_end_to_end_offline(
    autoscraper: ModuleType, vendor: ModuleType, tmp_path: Path
) -> None:
    pages = [PageInput(f"{i:04d}", f"http://x/{i}", page_html(i), "t1") for i in range(4)]
    pages.append(PageInput("tail", "http://x/tail", page_html(9), None))
    llm = FakeLLM(answer)
    out = autoscraper.run_baseline(
        pages, SCHEMA, out_dir=tmp_path, llm=llm, n_seeds=2, workers=1, clone=False
    )
    assert out.predictions["0003"] == {
        "title": "Engineer 3",
        "company": "Acme 3",
        "skills": ["Python", "SQL 3"],
        "facts": {},
    }
    assert out.predictions["tail"]["title"] is None
    # 3 fields x 2 seeds x (xpath + judgement) + 3 synthesis calls
    assert out.calls == 3 * 2 * 2 + 3 == len(llm.calls)
    assert {c["prompt_version"] for c in llm.calls} == {autoscraper.PROMPT_VERSION}
    assert out.pages_run == 5 and out.spend_usd == pytest.approx(0.015)
    assert any("not attempted" in n for n in out.notes)
    row = json.loads((tmp_path / "autoscraper.templates.jsonl").read_text())
    assert row["rules"]["title"] == [XPATHS["title"]] and len(row["seed_keys"]) == 2
    assert (tmp_path / "autoscraper.log").read_text()  # vendor prints went to the log


def test_presimplified_extraction_matches_upstream(vendor: ModuleType) -> None:
    upstream = vendor.StepbackCrawler(api=lambda q: "", simplify=True)
    fast = vendor.StepbackCrawler(api=lambda q: "", simplify=False)
    html = page_html(5)
    simplified = vendor.simplify_html(html)
    for rule in ([XPATHS["skills"]], ["//div[@class='job']", "./span/text()"], []):
        assert upstream.extract_with_sequence(html, rule) == fast.extract_with_sequence(
            simplified, rule
        )
