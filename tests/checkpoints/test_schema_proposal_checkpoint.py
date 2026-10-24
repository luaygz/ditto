"""Unprompted checkpoint: show schema_proposal.json, drop/rename fields, update the drafts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from ditto.checkpoints import show_schema_proposal
from ditto.run import RunDir, write_json
from ditto.specgen.models import GeneratedBy, ListField, ScalarField, Spec


def draft(tid: str, names: list[str]) -> Spec:
    fields: dict[str, Any] = {}
    for name in names:
        sel = [f'//span[@class="{name}"]', f'//div[@class="{name}"]/span']
        fields[name] = ListField(candidates=sel) if name == "tags" else ScalarField(candidates=sel)
    return Spec(
        template_id=tid,
        site="example.org",
        generated_by=GeneratedBy(model="fake/m", prompt_version="specgen.v1", mode="unprompted"),
        fields=fields,
        not_present={"summary": "not found", "isbn": "no isbn"} if "price" in names else {},
    )


def proposal_field(name: str) -> dict[str, Any]:
    return {
        "type": "list" if name == "tags" else "scalar",
        "type_check": None,
        "description": f"the {name}",
        "optional": name == "tags",
        "source_candidates": ["c1"],
        "examples": [["a", "b"]] if name == "tags" else [f"{name} 1", f"{name} 2"],
    }


TEMPLATES = {"t_a": ["title", "price", "tags"], "t_b": ["title", "author"]}


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    run = RunDir(tmp_path / "run").ensure()
    write_json(
        run.schema_proposal,
        {
            "version": 1,
            "site": "example.org",
            "mode": "unprompted",
            "model": "fake/m",
            "prompt_version": "specgen.v1",
            "created_at": "2026-10-24T00:00:00+00:00",
            "templates": {
                tid: {
                    "size": 40,
                    "url_examples": [],
                    "pages_seen": [],
                    "sample_page_ids": [],
                    "fields": {n: proposal_field(n) for n in names},
                }
                for tid, names in TEMPLATES.items()
            },
        },
    )
    for tid, names in TEMPLATES.items():
        draft(tid, names).save(run.draft_spec(tid))
    return run


def scripted(*answers: str) -> Callable[[str], str]:
    queue = list(answers)

    def input_fn(prompt: str) -> str:
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    return input_fn


def console() -> Console:
    return Console(record=True, width=160, file=open("/dev/null", "w"))


def test_yes_only_shows_the_proposal(run: RunDir) -> None:
    before = run.schema_proposal.read_bytes()
    con = console()
    result = show_schema_proposal(run, yes=True, console=con)
    assert result.confirmed_by == "yes"
    assert result.kept == TEMPLATES
    assert result.dropped == {} and result.renamed == {} and result.drafts_updated == []
    assert run.schema_proposal.read_bytes() == before
    shown = con.export_text()
    assert "t_a · 40 pages · proposed fields" in shown
    assert "price 1 | price 2" in shown


def test_drop_and_rename_update_proposal_and_drafts(run: RunDir) -> None:
    con = console()
    result = show_schema_proposal(
        run,
        input_fn=scripted(
            "drop t_a:tags",
            "rename title headline",
            "rename t_b:author byline",
            "drop nonexistent",
            "rename price _meta",
            "rename t_a:price headline",
            "rename price summary",
            "show",
            "what",
            "",
        ),
        console=con,
    )
    assert result.confirmed_by == "user"
    assert result.kept == {"t_a": ["headline", "summary"], "t_b": ["headline", "byline"]}
    assert result.dropped == {"t_a": ["tags"]}
    assert result.renamed == {
        "t_a": {"title": "headline", "price": "summary"},
        "t_b": {"title": "headline", "author": "byline"},
    }
    assert sorted(result.drafts_updated) == ["t_a", "t_b"]
    shown = con.export_text()
    assert "no template has a field 'nonexistent'" in shown
    assert "'_meta' is reserved" in shown
    assert "t_a already has a field 'headline'" in shown

    data = json.loads(run.schema_proposal.read_text())
    assert list(data["templates"]["t_a"]["fields"]) == ["headline", "summary"]
    assert data["templates"]["t_a"]["fields"]["headline"]["description"] == "the title"
    assert data["checkpoint"]["dropped"] == {"t_a": ["tags"]}
    assert data["checkpoint"]["confirmed_by"] == "user"

    spec_a = Spec.load(run.draft_spec("t_a"))
    assert list(spec_a.fields) == ["headline", "summary"]
    assert spec_a.fields["headline"].candidates[0] == '//span[@class="title"]'  # type: ignore[union-attr]
    assert spec_a.not_present == {"isbn": "no isbn"}  # "summary" now exists as a field
    spec_b = Spec.load(run.draft_spec("t_b"))
    assert list(spec_b.fields) == ["headline", "byline"]


def test_missing_draft_is_reported(run: RunDir) -> None:
    run.draft_spec("t_b").unlink()
    result = show_schema_proposal(run, input_fn=scripted("drop author", ""), console=console())
    assert result.dropped == {"t_b": ["author"]}
    assert result.drafts_missing == ["t_b"]
    assert result.drafts_updated == []


def test_unknown_template_target(run: RunDir) -> None:
    con = console()
    result = show_schema_proposal(run, input_fn=scripted("drop t_zz:title", "y"), console=con)
    assert "unknown template t_zz" in con.export_text()
    assert result.dropped == {}


def test_no_proposal(tmp_path: Path) -> None:
    result = show_schema_proposal(RunDir(tmp_path), yes=False, console=console())
    assert result.kept == {}
