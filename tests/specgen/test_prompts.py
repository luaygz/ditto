"""Prompt files: lookup by name and version, placeholder filling, system/user split."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ditto.specgen import prompts
from ditto.specgen.prompts import Prompt, available_prompts, load_prompt

REPO_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


@pytest.fixture
def prompt_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(prompts, "_search_dirs", lambda: [tmp_path])
    load_prompt.cache_clear()
    yield tmp_path
    load_prompt.cache_clear()


def test_latest_and_pinned_versions(prompt_dir: Path) -> None:
    (prompt_dir / "demo.v1.md").write_text("one {{x}}")
    (prompt_dir / "demo.v2.md").write_text("two {{x}}")
    (prompt_dir / "demo.v10.md").write_text("ten {{x}}")
    (prompt_dir / "other.v3.md").write_text("other")
    (prompt_dir / "notes.md").write_text("ignored")
    latest = load_prompt("demo")
    assert (latest.version, latest.id, latest.text) == (10, "demo.v10", "ten {{x}}")
    assert load_prompt("demo", 2).text == "two {{x}}"
    assert available_prompts() == {"demo": [1, 2, 10], "other": [3]}
    with pytest.raises(FileNotFoundError, match=r"demo\.v7\.md"):
        load_prompt("demo", 7)
    with pytest.raises(FileNotFoundError, match=r"missing\.v\*\.md"):
        load_prompt("missing")


def test_render_and_messages() -> None:
    p = Prompt(
        name="demo",
        version=1,
        text="# system\n\nYou are {{who}}.\n\n# user\n\nPage:\n{{page}}\n",
    )
    assert p.placeholders == {"who", "page"}
    msgs = p.messages(who="a tester", page="# system\n<p>{{who}}</p>")
    assert msgs == [
        {"role": "system", "content": "You are a tester."},
        {"role": "user", "content": "Page:\n# system\n<p>{{who}}</p>"},  # values are not re-split
    ]
    assert p.render(who="w", page="x").startswith("# system\n\nYou are w.")
    with pytest.raises(KeyError, match="page"):
        p.messages(who="w")
    with pytest.raises(ValueError, match="unknown placeholder"):
        p.messages(who="w", page="x", extra="y")


def test_prompt_without_sections_is_one_user_message() -> None:
    p = Prompt(name="x", version=1, text="Hello {{name}}")
    assert p.messages(name="you") == [{"role": "user", "content": "Hello you"}]
    stray = Prompt(name="x", version=1, text="preamble\n# user\nHi")
    with pytest.raises(ValueError, match="before the first"):
        stray.messages()


def test_shipped_prompts_have_the_expected_placeholders() -> None:
    load_prompt.cache_clear()
    spec = load_prompt("specgen", 1)
    assert spec.id == "specgen.v1"
    assert spec.placeholders == {
        "mode",
        "output_schema",
        "template_context",
        "fields_section",
        "candidate_table",
        "tables_section",
        "union_page",
    }
    tail = load_prompt("tail_direct", 1)
    assert tail.placeholders == {"fields", "url", "page"}
    for p in (spec, tail):
        assert [m["role"] for m in p.messages(**dict.fromkeys(p.placeholders, "v"))] == [
            "system",
            "user",
        ]
    assert (REPO_PROMPTS / "specgen.v1.md").is_file()
