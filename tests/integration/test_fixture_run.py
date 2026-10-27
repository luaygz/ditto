"""End to end on the 200-page fixture site with recorded LLM responses (no key, no network).

``tests/fixtures/llm/*.json`` and ``tests/fixtures/models.json`` were recorded with::

    DITTO_CACHE_DIR=tests/fixtures uv run ditto run tests/fixtures/site/site --out /tmp/fx \
        --schema tests/fixtures/site/fields.yaml --yes

Any change to what the model sees (prompts, union pages, candidate tables) changes the cache
keys; this test then fails with a cache miss and the responses must be recorded again.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ditto.cli import app
from ditto.textutil import normalize_ws

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
SITE = FIXTURES / "site"
# requested field -> the names the synthetic templates give it
ALIASES = {
    "title": ("title", "headline"),
    "published": ("published", "release_date"),
    "tags": ("tags", "keywords", "topics"),
}
MIN_RECALL = 0.95


def _no_network(*_: Any, **__: Any) -> None:
    raise AssertionError("the offline fixture run tried to open a network connection")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_MODEL_SMALL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name in ("generation", "repair", "probe", "tail_direct", "baseline"):
        monkeypatch.delenv(f"DITTO_MODEL_{name.upper()}", raising=False)
    monkeypatch.setenv("DITTO_LLM_OFFLINE", "1")
    monkeypatch.setenv("DITTO_CACHE_DIR", str(FIXTURES))
    monkeypatch.setenv("LLM_BUDGET_USD", "1000")  # the fixture ledger is not committed
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.chdir(tmp_path)  # no .env is found from here
    yield tmp_path
    assert not (FIXTURES / "ledger.jsonl").exists(), "a cached run must not spend anything"


def _truth() -> dict[str, dict[str, Any]]:
    """Sidecars of the pages of non-tail templates, by URL."""
    templates = json.loads((SITE / "truth" / "templates.json").read_text())["templates"]
    big = {t["template_id"] for t in templates if not t["is_tail"]}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((SITE / "truth").glob("p*.json")):
        car = json.loads(path.read_text())
        if car["template_id"] in big:
            out[car["url"]] = car
    return out


def _true_value(car: dict[str, Any], field: str) -> Any:
    for name in ALIASES[field]:
        if car["fields"].get(name) is not None:
            return car["fields"][name]
        if car["lists"].get(name):
            return car["lists"][name]
    return None


def _norm(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(normalize_ws(str(v)).casefold() for v in value)
    return None if value is None else normalize_ws(str(value)).casefold()


def test_fixture_run_offline(offline: Path) -> None:
    out = offline / "run"
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(SITE / "site"),
            "--out",
            str(out),
            "--schema",
            str(SITE / "fields.yaml"),
            "--yes",
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)

    clusters = json.loads((out / "clusters.json").read_text())
    big = [c for c in clusters["clusters"] if not c["is_tail"] and c["size"] >= 20]
    # four synthetic templates of >= 20 pages; verification may merge a near-duplicate one
    assert sum(1 + len(c["merged_from"]) for c in big) == 4
    tids = {c["template_id"] for c in big}
    for tid in tids:
        assert (out / "specs" / f"{tid}.json").is_file()
    jsonl = sorted((out / "out").glob("*/*.jsonl"))
    assert {p.stem for p in jsonl} >= tids
    assert (out / "report.json").is_file() and (out / "report.md").is_file()
    generation = json.loads((out / "generation.json").read_text())
    assert generation["prompt_version"] == "specgen.v2"
    assert all(t["cached_calls"] == t["calls"] for t in generation["templates"].values())

    rows = {}
    for path in jsonl:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row["url"]] = row
    truth = _truth()
    for field in ALIASES:
        hits = total = false_values = 0
        for url, car in truth.items():
            want = _true_value(car, field)
            got = rows.get(url, {}).get(field)
            if want is None:
                false_values += got not in (None, [], "")
                continue
            total += 1
            hits += _norm(got) == _norm(want)
        assert total and hits / total >= MIN_RECALL, f"{field}: recall {hits}/{total}"
        # e.g. courses have no publication date: nothing may be invented for them
        assert false_values <= 0.05 * len(truth), f"{field}: {false_values} invented values"
