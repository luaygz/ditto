from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

from ditto.evaluation.baseline_api import BaselineOutput, PageInput
from ditto.llm.base import Completion, Usage
from ditto.schema import FieldsSchema

PAGE = """<!DOCTYPE html>
<html><head><title>Ray Allen - Celtics</title><meta name="x" content="drop me">
<script>var tracking = 1;</script><style>.a{color:red}</style></head>
<body class="page mt-4 css-1x2y3z">
  <!-- a comment -->
  <div id="main" data-track="abc" style="color: blue" class="player-info">
    <h1 itemprop="name"   onclick="go()">Ray
       Allen</h1>
    <a href="https://example.com/players/ray-allen/this/is/a/very/long/path/that/keeps/going/on/and/on/forever"
       rel="nofollow">profile</a>
    <img src="/i.png" alt="Ray Allen headshot" width="10">
    <span aria-label="height" role="note" data-testid="h">6-5</span>
    <noscript>enable js</noscript>
  </div>
</body></html>"""


def _pages(n: int) -> list[PageInput]:
    return [PageInput(key=f"{i:04d}", url=f"u{i}", html="<p/>", template_id="t") for i in range(n)]


def test_sample_pages_is_seeded_ordered_and_capped(common: ModuleType) -> None:
    pages = _pages(50)
    a = common.sample_pages(pages, 10, seed=3)
    b = common.sample_pages(pages, 10, seed=3)
    c = common.sample_pages(pages, 10, seed=4)
    assert [p.key for p in a] == [p.key for p in b]
    assert [p.key for p in a] != [p.key for p in c]
    assert len(a) == 10
    assert [p.key for p in a] == sorted(p.key for p in a)  # input order kept
    assert common.sample_pages(pages, None, seed=0) == pages
    assert common.sample_pages(pages, 500, seed=0) == pages
    assert common.sample_pages(pages, 0, seed=0) == []


def test_group_by_template_separates_tail(common: ModuleType) -> None:
    pages = [
        PageInput("a", "", "", "t1"),
        PageInput("b", "", "", None),
        PageInput("c", "", "", "t2"),
        PageInput("d", "", "", "t1"),
    ]
    groups, tail = common.group_by_template(pages)
    assert {k: [p.key for p in v] for k, v in groups.items()} == {"t1": ["a", "d"], "t2": ["c"]}
    assert [p.key for p in tail] == ["b"]


def test_page_for_llm_cleans_and_prunes(common: ModuleType) -> None:
    out = common.page_for_llm(PAGE)
    assert "tracking" not in out and "color:red" not in out and "comment" not in out
    assert "enable js" not in out and "drop me" not in out
    assert "<title>Ray Allen - Celtics</title>" in out
    # pruned attributes
    for gone in ("data-track", "style=", "onclick", "rel=", "width=", "mt-4", "css-1x2y3z"):
        assert gone not in out
    # kept attributes
    for kept in (
        'id="main"',
        'class="player-info"',
        'itemprop="name"',
        'alt="Ray Allen headshot"',
        'aria-label="height"',
        'role="note"',
        'data-testid="h"',
    ):
        assert kept in out
    href = out.split('href="', 1)[1].split('"', 1)[0]
    assert len(href) == common.LINK_ATTR_MAX
    assert ">Ray Allen</h1>" in out  # whitespace collapsed, text kept


def test_page_for_llm_accepts_bytes(common: ModuleType) -> None:
    assert "Ray Allen" in common.page_for_llm(PAGE.encode("utf-8"))


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (" Ray   Allen ", "scalar", "Ray Allen"),
        ("", "scalar", None),
        (None, "scalar", None),
        (12.0, "scalar", "12"),
        (3.5, "scalar", "3.5"),
        (True, "scalar", "true"),
        (["", "a", "b"], "scalar", "a"),
        ([{"v": "x"}], "scalar", "x"),
        ({"only": "y"}, "scalar", "y"),
        ({"a": "1", "b": "2"}, "scalar", None),
        (None, "list", []),
        ("one", "list", ["one"]),
        (["a", "", None, {"name": "b"}, 3], "list", ["a", "b", "3"]),
        ([{"x": "1", "y": ""}, {"x": None}, "junk"], "record_list", [{"x": "1", "y": None}]),
        ({"x": "1"}, "record_list", [{"x": "1"}]),
        ("nope", "record_list", []),
        ({"Color:": "red", "": "skip", "Size": None}, "key_value", {"Color": "red", "Size": None}),
        (
            [{"key": "A", "value": "1"}, ["B", "2"], {"key": "A", "value": "dup"}],
            "key_value",
            {"A": "1", "B": "2"},
        ),
        ("nope", "key_value", {}),
    ],
)
def test_coerce_value(common: ModuleType, value: object, kind: str, expected: object) -> None:
    assert common.coerce_value(value, kind) == expected


def test_coerce_prediction_and_payload(common: ModuleType) -> None:
    schema = FieldsSchema.from_obj(
        {
            "name": "player name",
            "tags": {"type": "list", "examples": ["guard"]},
            "stats": {"type": "record_list"},
            "facts": {"type": "key_value"},
        }
    )
    pred = common.coerce_prediction({"name": ["Ray"], "extra": 1}, schema)
    assert pred == {"name": "Ray", "tags": [], "stats": [], "facts": {}}
    payload = common.fields_payload(schema)
    assert payload[0] == {"name": "name", "type": "scalar", "description": "player name"}
    assert payload[1] == {"name": "tags", "type": "list", "examples": ["guard"]}
    assert [common.empty_value(k) for k in ("scalar", "list", "record_list", "key_value")] == [
        None,
        [],
        [],
        {},
    ]


def test_usage_tally_and_recording_llm(common: ModuleType, fake_llm_factory: type) -> None:
    llm = fake_llm_factory(lambda _m: '{"a": 1}')
    rec = common.RecordingLLM(llm)
    assert rec.model == llm.model and rec.role == "baseline"
    rec.complete([{"role": "user", "content": "x" * 400}], prompt_version="p")
    rec.complete([{"role": "user", "content": "y" * 40}])
    done = rec.drain()
    assert len(done) == 2 and rec.drain() == []
    tally = common.UsageTally()
    for c in done:
        common.tally_completion(tally, c)
    cached = Completion("{}", Usage(1000, 100), "openai/gpt-6.1-sol", cached=True)
    common.tally_completion(tally, cached)
    d = tally.as_dict()
    assert d["calls"] == 3 and d["cached_calls"] == 1
    assert d["input_tokens"] == 100 + 10 + 1000
    # nominal prices every token at $2 / $10 per million (config/prices.yaml)
    expected = (d["input_tokens"] * 2 + d["output_tokens"] * 10) / 1e6
    assert d["nominal_usd"] == pytest.approx(expected)
    assert d["spent_usd"] == pytest.approx(0.002)  # two uncached fake calls at $0.001
    assert "_lock" not in d


def test_writers(common: ModuleType, tmp_path: Path) -> None:
    pages = _pages(2)
    preds = {"0000": {"f": "x"}}
    rows = common.predictions_rows(pages, preds)
    assert rows == [{"key": "0000", "url": "u0", "template_id": "t", "f": "x"}]
    common.write_jsonl(tmp_path / "sub" / "p.jsonl", rows)
    assert json.loads((tmp_path / "sub" / "p.jsonl").read_text()) == rows[0]
    out = BaselineOutput(predictions=preds, calls=1, notes=["n"], spend_usd=0.5, pages_run=2)
    path = common.write_summary(tmp_path, "demo", out, {"k": 1})
    data = json.loads(path.read_text())
    assert data["baseline"] == "demo" and data["spend_usd"] == 0.5 and data["k"] == 1


def test_tiered_client_adds_service_tier(common: ModuleType) -> None:
    seen: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> str:
            seen.update(kwargs)
            return "resp"

    fake = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})(), "models": "models"},
    )()
    client = common.TieredClient(fake, "flex")
    assert client.chat.completions.create(model="m", messages=[]) == "resp"
    assert seen == {"service_tier": "flex", "model": "m", "messages": []}
    assert client.models == "models"


def test_tiered_client_falls_back_when_flex_has_no_capacity(common: ModuleType) -> None:
    import httpx
    import openai

    calls: list[dict[str, object]] = []
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    unavailable = openai.RateLimitError(
        "Flex does not have sufficient resources",
        response=httpx.Response(429, request=request),
        body={"code": "flex_unavailable", "type": "resource_unavailable"},
    )

    class Completions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            if kwargs.get("service_tier") == "flex":
                raise unavailable
            return "default-tier response"

    fake = type(
        "Client", (), {"chat": type("Chat", (), {"completions": Completions()})(), "models": None}
    )()
    client = common.TieredClient(fake, "flex")
    assert client.chat.completions.create(model="m", messages=[]) == "default-tier response"
    assert [c.get("service_tier") for c in calls] == ["flex", None]
    assert client.chat.completions.fallbacks == 1


def test_tiered_client_reraises_other_rate_limits(common: ModuleType) -> None:
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    limited = openai.RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=request),
        body={"code": "rate_limit_exceeded"},
    )

    class Completions:
        def create(self, **kwargs: object) -> object:
            raise limited

    fake = type(
        "Client", (), {"chat": type("Chat", (), {"completions": Completions()})(), "models": None}
    )()
    with pytest.raises(openai.RateLimitError):
        common.TieredClient(fake, "flex").chat.completions.create(model="m", messages=[])


def test_baseline_llm_tiers(common: ModuleType, tmp_path: Path) -> None:
    from ditto.llm import Settings

    s = Settings(openai_api_key="sk-test", openai_model="gpt-6.1-sol", cache_dir=tmp_path)
    flex = common.baseline_llm(s, service_tier="flex")
    assert flex.model == "openai/gpt-6.1-sol" and flex.role == "baseline"
    assert isinstance(flex.adapter._client, common.TieredClient)
    plain = common.baseline_llm(s)
    assert not isinstance(plain.adapter._client, common.TieredClient)
    keyless = common.baseline_llm(
        Settings(openai_model="gpt-6.1-sol", cache_dir=tmp_path, offline=True), service_tier="flex"
    )
    assert keyless.adapter is None  # offline: recorded responses only, no tier adapter
