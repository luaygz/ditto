"""``--tail direct``: one small-model call per tail page, rows in the extraction JSONL shape."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.ingest import PageStore, ingest
from ditto.llm.base import BudgetExceeded, CacheMiss, Completion, Message, Usage
from ditto.run import RunDir, read_json
from ditto.schema import FieldRequest, FieldsSchema
from ditto.specgen.models import GeneratedBy, Spec
from ditto.specgen.tail import compact_page, tail_direct_run
from ditto.union import count_tokens

SMALL = "openai/gpt-5.4-mini"

EVENT = """<!doctype html><html><head><title>{title}</title><meta name="x" content="y">
<script>window.track = {{"id": 1}};</script><style>.a{{color:red}}</style></head>
<body class="mt-4 page flex"><div id="event" data-v-1a2b3c="" onclick="go()" style="color:red">
<h1 class="title text-lg">{title}</h1><p class="when">{date}</p>
<ul class="speakers">{speakers}</ul>
<a class="tickets" href="https://tickets.example.org/{slug}/checkout/standard-admission-and-workshop-materials-included?utm_source=newsletter&amp;utm_medium=email&amp;utm_campaign=autumn-2026-launch">Tickets</a>
</div><noscript>enable js</noscript></body></html>"""

PAGES = [
    ("ws", "Wrapper Workshop", "2026-11-03", ["A. Khoury", "L. Haddad"]),
    ("talk", "Template Talk", "2026-12-01", ["R. Nasr"]),
    ("meet", "Meetup", "2027-01-15", []),
]

SCHEMA = FieldsSchema(
    fields=[
        FieldRequest(name="title", type="scalar", description="event title"),
        FieldRequest(name="date", description="event date"),
        FieldRequest(name="speakers", type="list"),
        FieldRequest(name="sessions", type="record_list"),
        FieldRequest(name="facts", type="key_value"),
    ]
)


class FakeLLM:
    def __init__(self, respond: Callable[[list[Message]], str | Exception]) -> None:
        self.model = SMALL
        self.role = "tail_direct"
        self.respond = respond
        self.calls: list[list[Message]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        with self._lock:
            self.calls.append(messages)
        out = self.respond(messages)
        if isinstance(out, Exception):
            raise out
        return Completion(text=out, usage=Usage(800, 60), model=SMALL, cost_usd=0.0007)


def page_of(messages: list[Message]) -> str:
    user = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
    for m in messages:
        if "URL: " in m["content"]:
            user = m["content"]
    return user.split("URL: ", 1)[1].split("\n", 1)[0]


def good_answer(messages: list[Message]) -> str:
    url = page_of(messages)
    slug = url.rsplit("/", 1)[1]
    title, date, speakers = next((t, d, s) for sl, t, d, s in PAGES if sl == slug)
    return json.dumps(
        {
            "title": f"  {title} ",
            "Date": date,  # names match case-insensitively
            "speakers": speakers[0] if len(speakers) == 1 else speakers,  # str -> [str]
            "sessions": None,
            "facts": {"Venue:": "Hall B", "": "dropped"},
        }
    )


@pytest.fixture(autouse=True)
def isolated_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DITTO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_BUDGET_USD", "40")
    monkeypatch.delenv("DITTO_RUN_BUDGET_USD", raising=False)


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    site = tmp_path / "site"
    site.mkdir()
    lines = []
    for slug, title, date, speakers in PAGES:
        items = "".join(f"<li>{s}</li>" for s in speakers)
        html = EVENT.format(title=title, date=date, speakers=items, slug=slug)
        (site / f"{slug}.html").write_text(html)
        lines.append(f"{slug}.html\thttps://events.example.org/e/{slug}")
    (site / "urls.tsv").write_text("\n".join(lines) + "\n")
    ingest(site, tmp_path / "run", manifest=site / "urls.tsv", site="events.example.org", workers=1)
    rd = RunDir(tmp_path / "run")
    store = PageStore.open(rd)
    ids = {slug: store.lookup(f"https://events.example.org/e/{slug}") for slug, *_ in PAGES}
    big = [ids["ws"], ids["talk"]]
    small = [ids["meet"]]
    Clusters(
        n_pages=3,
        min_cluster_size=2,
        clusters=[
            Cluster(
                template_id="t_big",
                size=2,
                is_tail=False,
                signature_groups=[0],
                page_ids=[i for i in big if i is not None],
            ),
            Cluster(
                template_id="t_one",
                size=1,
                is_tail=True,
                signature_groups=[1],
                page_ids=[i for i in small if i is not None],
            ),
            Cluster(
                template_id="t_two",
                size=1,
                is_tail=True,
                signature_groups=[2],
                page_ids=[i for i in big if i is not None][:1],
            ),
        ],
        signature_groups=[
            SignatureGroup(id=0, sig_hash="0", size=2, template_id="t_big", page_ids=[]),
            SignatureGroup(id=1, sig_hash="1", size=1, template_id="t_one", page_ids=[]),
            SignatureGroup(id=2, sig_hash="2", size=1, template_id="t_two", page_ids=[]),
        ],
    ).save(rd.clusters_json)
    return rd


def test_compact_page_prunes_and_caps() -> None:
    html = EVENT.format(title="T", date="D", speakers="<li>S</li>", slug="s")
    out = compact_page(html)
    assert "<script" not in out and "<style" not in out and "noscript" not in out
    assert "onclick" not in out and "style=" not in out and "data-v-" not in out
    assert 'id="event"' in out and 'class="title"' in out and 'class="page"' in out
    assert "<title>T</title>" in out
    href = out.split('href="', 1)[1].split('"', 1)[0].replace("&amp;", "&")
    assert len(href) == 120 and href.startswith("https://tickets.example.org/s/checkout/")
    long_page = "<html><body>" + "".join(f"<p>paragraph {i} text</p>" for i in range(3000))
    capped = compact_page(long_page + "</body></html>", max_tokens=500)
    assert count_tokens(capped) <= 500


def test_tail_direct_rows_have_the_extraction_shape(run: RunDir) -> None:
    llm = FakeLLM(good_answer)
    result = tail_direct_run(run, SCHEMA, llm=llm, concurrency=2)
    assert result.pages == 2 and result.rows == 2 and len(llm.calls) == 2
    assert result.failed == [] and result.unprocessed == [] and result.aborted is None
    assert result.calls == 2 and result.cost_usd == pytest.approx(0.0014)
    path = run.out_jsonl("events.example.org", None)
    assert result.path == path and path.name == "_tail.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["page_id"] for r in rows] == sorted(r["page_id"] for r in rows)
    by_url = {r["url"]: r for r in rows}
    meet = by_url["https://events.example.org/e/meet"]
    assert list(meet) == [
        "url",
        "page_id",
        "template_id",
        "title",
        "date",
        "speakers",
        "sessions",
        "facts",
        "_meta",
    ]
    assert meet["template_id"] is None
    assert meet["title"] == "Meetup" and meet["date"] == "2027-01-15"
    assert meet["speakers"] == [] and meet["sessions"] == []
    assert meet["facts"] == {"Venue": "Hall B"}
    assert meet["_meta"]["missing"] == ["speakers", "sessions"]
    assert meet["_meta"]["spec_version"] is None
    assert meet["_meta"]["method"] == "tail_direct"
    assert meet["_meta"]["prompt_version"] == "tail_direct.v1"
    assert meet["_meta"]["cluster_id"] == "t_one"
    ws = by_url["https://events.example.org/e/ws"]
    assert ws["speakers"] == ["A. Khoury", "L. Haddad"] and ws["_meta"]["cluster_id"] == "t_two"
    # The model saw the cleaned page, not the scripts.
    assert "window.track" not in llm.calls[0][-1]["content"]
    assert "tail_direct" in read_json(run.timings_json)


def test_tail_clusters_with_a_final_spec_are_left_to_extraction(run: RunDir) -> None:
    def spec(tid: str, status: str) -> Spec:
        by = GeneratedBy(model="openai/gpt-6.1-sol", prompt_version="specgen.v2")
        return Spec.model_validate(
            {
                "template_id": tid,
                "site": "events.example.org",
                "status": status,
                "generated_by": by,
                "fields": {},
                "not_present": {"title": "-"},
            }
        )

    spec("t_one", "final").save(run.spec("t_one"))  # e.g. made by --tail spec
    spec("t_two", "draft").save(run.spec("t_two"))  # not verified: still sent
    llm = FakeLLM(good_answer)
    result = tail_direct_run(run, SCHEMA, llm=llm)
    assert result.pages == 1 and len(llm.calls) == 1
    rows = [
        json.loads(x) for x in run.out_jsonl("events.example.org", None).read_text().splitlines()
    ]
    assert [r["_meta"]["cluster_id"] for r in rows] == ["t_two"]


def test_invalid_answer_is_retried_then_recorded(run: RunDir) -> None:
    def respond(messages: list[Message]) -> str:
        if "could not be used" in messages[-1]["content"]:
            return good_answer(messages)
        return json.dumps({"title": ["a", "b"], "sessions": "not a list"})

    llm = FakeLLM(respond)
    result = tail_direct_run(run, SCHEMA, llm=llm)
    assert result.rows == 2 and result.failed == [] and len(llm.calls) == 4
    retry = llm.calls[1][-1]["content"]
    assert "title: scalar field expects one string" in retry
    assert "sessions: record_list expects an array" in retry

    always_bad = FakeLLM(lambda m: "nope")
    result = tail_direct_run(run, SCHEMA, llm=always_bad)
    assert len(result.failed) == 2
    rows = [
        json.loads(x) for x in run.out_jsonl("events.example.org", None).read_text().splitlines()
    ]
    assert all("InvalidResponse" in r["_meta"]["error"] for r in rows)
    assert all(r["_meta"]["missing"] == SCHEMA.names() for r in rows)


def test_llm_errors_and_budget(run: RunDir, monkeypatch: pytest.MonkeyPatch) -> None:
    result = tail_direct_run(run, SCHEMA, llm=FakeLLM(lambda m: CacheMiss("offline")))
    assert len(result.failed) == 2

    calls = {"n": 0}

    def budget_on_second(messages: list[Message]) -> str | Exception:
        calls["n"] += 1
        return good_answer(messages) if calls["n"] == 1 else BudgetExceeded("spent")

    result = tail_direct_run(run, SCHEMA, llm=FakeLLM(budget_on_second), concurrency=1)
    assert result.rows == 1 and len(result.unprocessed) == 1
    assert result.aborted is not None and result.to_json()["status"] == "aborted"

    monkeypatch.setenv("LLM_BUDGET_USD", "0.0000001")
    llm = FakeLLM(good_answer)
    with pytest.raises(BudgetExceeded):
        tail_direct_run(run, SCHEMA, llm=llm)
    assert llm.calls == []
