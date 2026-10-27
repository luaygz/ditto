"""Run-level tests for ``generate_run``: real ingested pages, hand-built analysis, fake LLM."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ditto.analyze.models import Candidate, CandidateTable, Table, TableColumn
from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.ingest import PageStore, ingest
from ditto.llm.base import BudgetExceeded, Completion, Message, Usage
from ditto.run import RunDir, read_json
from ditto.schema import Alternative, FieldRequest, FieldsSchema, Mapping, TemplateMapping
from ditto.specgen import generate as generate_mod
from ditto.specgen import generate_run
from ditto.specgen.models import Spec

MODEL = "openai/gpt-6.1-sol"
SMALL = "openai/gpt-5.4-mini"

PEOPLE = [
    ("a-khoury", "Amal Khoury", "a.khoury@example.org", [("Wrappers at scale", 2021)]),
    ("l-haddad", "Lina Haddad", None, [("Template drift", 2023), ("Union pages", 2024)]),
    ("r-nasr", "Rami Nasr", "r.nasr@example.org", []),
    ("s-azar", "Sara Azar", "s.azar@example.org", [("XPath anchors", 2019)]),
    ("m-saad", "Maya Saad", None, [("Clustering sites", 2020)]),
    ("k-fares", "Karim Fares", "k.fares@example.org", [("Long tails", 2022)]),
]


def profile_html(slug: str, name: str, email: str | None, pubs: list[tuple[str, int]]) -> str:
    contact = (
        f'<dt>Email</dt><dd><a href="mailto:{email}">{email}</a></dd>' if email else ""
    ) + f"<dt>Office</dt><dd>B-{len(slug)}0{len(name)}</dd>"
    rows = "".join(f"<tr><td>{t}</td><td>{y}</td></tr>" for t, y in pubs)
    table = f'<table id="pubs"><tr><th>Title</th><th>Year</th></tr>{rows}</table>' if pubs else ""
    return f"""<!doctype html><html><head><title>{name} - Staff</title>
<script>var tracking = 1;</script><style>h1{{color:red}}</style></head>
<body class="page"><nav><a href="/">Home</a><a href="/staff">Staff</a></nav>
<div id="profile"><h1 itemprop="name">{name}</h1>
<dl class="contact">{contact}</dl>{table}</div>
<div class="related"><h2>Colleagues</h2><a href="/staff/x">X</a></div>
<footer>© Example University</footer></body></html>"""


def listing_html(page: int) -> str:
    items = "".join(
        f'<li class="person"><a href="/staff/{slug}">{name}</a></li>' for slug, name, *_ in PEOPLE
    )
    return f"""<!doctype html><html><head><title>Staff directory {page}</title></head>
<body><main><h1>Staff directory</h1><ul class="people">{items}</ul></main></body></html>"""


UNION_HTML = """<html><body class="page"><div id="profile">
<!-- c1 · mandatory · on 100% of pages · e.g. Amal Khoury -->
<h1 itemprop="name">Amal Khoury</h1>
<dl class="contact"><dt>Email</dt>
<!-- c2 · optional · on 67% of pages · e.g. a.khoury@example.org -->
<dd><a href="mailto:a.khoury@example.org">a.khoury@example.org</a></dd>
<dt>Office</dt><!-- c3 · mandatory · on 100% of pages · e.g. B-80011 --><dd>B-80011</dd></dl>
<!-- spliced from another page: c4 -->
<table id="pubs"><tr><th>Title</th><th>Year</th></tr>
<tr><!-- c4 · list · on 83% of pages · e.g. Wrappers at scale --><td>Wrappers at scale</td>
<!-- c5 · list · on 83% of pages · e.g. 2021 --><td>2021</td></tr></table>
</div></body></html>"""


def candidates() -> CandidateTable:
    def cand(cid: str, path: str, cls: str, p: float, **kw: Any) -> Candidate:
        return Candidate.model_validate(
            {"id": cid, "path": path, "class": cls, "p": p, "m": 0.2, **kw}
        )

    prof = "html > body.page > div#profile"
    return CandidateTable(
        template_id="t_prof",
        cluster_size=6,
        n_sampled=6,
        candidates=[
            cand("c1", f"{prof} > h1[itemprop=name]", "mandatory", 1.0, examples=["Amal Khoury"]),
            cand("c2", f"{prof} > dl.contact > dd > a", "optional", 0.67, label_hint="Email"),
            cand("c3", f"{prof} > dl.contact > dd", "mandatory", 1.0, label_hint="Office"),
            cand(
                "c4",
                f"{prof} > table#pubs > tr > td",
                "list",
                0.83,
                table_id="T1",
                table_role="record_column",
                column="Title",
                repeat_max=2,
            ),
            cand(
                "c5",
                f"{prof} > table#pubs > tr > td",
                "list",
                0.83,
                table_id="T1",
                table_role="record_column",
                column="Year",
                repeat_max=2,
            ),
        ],
        tables=[
            Table(
                id="T1",
                kind="record_list",
                container_path=f"{prof} > table#pubs",
                row_path=f"{prof} > table#pubs > tr",
                p=0.83,
                rows_min=1,
                rows_max=2,
                columns=[
                    TableColumn(name="Title", candidate_id="c4"),
                    TableColumn(name="Year", candidate_id="c5"),
                ],
            )
        ],
    )


NAME = {
    "type": "scalar",
    "candidates": ["//h1[@itemprop='name']", "//div[@id='profile']//h1"],
    "source_candidates": ["c1"],
}
EMAIL = {
    "type": "scalar",
    "candidates": [
        "//dt[normalize-space()='Email']/following-sibling::dd[1]//a",
        "//a[starts-with(@href,'mailto:')]",
    ],
    "type_check": "email",
    "source_candidates": ["c2"],
}
PUBS = {
    "type": "record_list",
    "container": "//table[@id='pubs']//tr[td]",
    "columns": {
        "title": "./td[count(ancestor::table//th[normalize-space()='Title']/preceding-sibling::th)+1]",
        "year": "./td[count(ancestor::table//th[normalize-space()='Year']/preceding-sibling::th)+1]",
    },
    "column_source": "header_row",
    "transforms": {"year": ["cast(int)"]},
    "source_candidates": ["c4", "c5"],
}
BAD = {
    "type": "scalar",
    "candidates": ["/html/body/div/dl/dd[1]/a", "//dl/dd/a"],
    "source_candidates": ["c2"],
}

SCHEMA = FieldsSchema(
    fields=[
        FieldRequest(name="name", description="full name"),
        FieldRequest(name="email", description="contact email", type_check="email"),
        FieldRequest(name="publications", type="record_list"),
        FieldRequest(name="phone", description="phone number"),
    ]
)


def answer(fields: dict[str, Any], not_present: dict[str, str] | None = None) -> str:
    return json.dumps({"fields": fields, "not_present": not_present or {}})


GOOD = answer({"name": NAME, "email": EMAIL, "publications": PUBS}, {"phone": "no phone shown"})


class FakeLLM:
    def __init__(
        self,
        responses: list[str | Exception] | Callable[[list[Message], str | None], str | Exception],
        model: str = MODEL,
        role: str = "generation",
    ) -> None:
        self.model = model
        self.role = role
        self._responses = responses
        self.calls: list[dict[str, Any]] = []
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
            self.calls.append({"messages": messages, "template_id": template_id})
            out = (
                self._responses(messages, template_id)
                if callable(self._responses)
                else self._responses.pop(0)
            )
        if isinstance(out, Exception):
            raise out
        return Completion(text=out, usage=Usage(1200, 300), model=self.model, cost_usd=0.005)


@pytest.fixture(autouse=True)
def isolated_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DITTO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_BUDGET_USD", "40")
    monkeypatch.delenv("DITTO_RUN_BUDGET_USD", raising=False)


def write_union(run: RunDir, tid: str, html: str, *, focus: list[str] | None = None) -> None:
    run.union_html(tid).parent.mkdir(parents=True, exist_ok=True)
    run.union_html(tid).write_text(html)
    meta = {
        "template_id": tid,
        "tokens": 300,
        "base_page_id": 0,
        "base_url": "https://example.org/staff/a-khoury",
        "donor_page_ids": [1],
        "donor_urls": ["https://example.org/staff/l-haddad"],
        "shown": ["c1", "c2", "c3", "c4", "c5"],
        "focus": focus,
    }
    run.union_meta(tid).write_text(json.dumps(meta))


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    site = tmp_path / "site"
    site.mkdir()
    manifest: list[str] = []
    for slug, name, email, pubs in PEOPLE:
        (site / f"{slug}.html").write_text(profile_html(slug, name, email, pubs))
        manifest.append(f"{slug}.html\thttps://example.org/staff/{slug}")
    for i in (1, 2):
        (site / f"list{i}.html").write_text(listing_html(i))
        manifest.append(f"list{i}.html\thttps://example.org/staff?page={i}")
    (site / "urls.tsv").write_text("\n".join(manifest) + "\n")
    ingest(site, tmp_path / "run", manifest=site / "urls.tsv", site="example.org", workers=1)
    rd = RunDir(tmp_path / "run")
    store = PageStore.open(rd)
    prof = sorted(store.lookup(f"https://example.org/staff/{p[0]}") or 0 for p in PEOPLE)
    lst = sorted(store.lookup(f"https://example.org/staff?page={i}") or 0 for i in (1, 2))
    Clusters(
        n_pages=8,
        min_cluster_size=3,
        clusters=[
            Cluster(
                template_id="t_prof",
                size=6,
                is_tail=False,
                signature_groups=[0],
                url_patterns=["/staff/{slug}"],
                url_examples=[f"https://example.org/staff/{p[0]}" for p in PEOPLE[:3]],
                page_ids=prof,
            ),
            Cluster(
                template_id="t_list",
                size=2,
                is_tail=True,
                signature_groups=[1],
                url_patterns=["/staff?page={n}"],
                page_ids=lst,
            ),
        ],
        signature_groups=[
            SignatureGroup(id=0, sig_hash="a0", size=6, template_id="t_prof", page_ids=prof),
            SignatureGroup(id=1, sig_hash="b1", size=2, template_id="t_list", page_ids=lst),
        ],
    ).save(rd.clusters_json)
    candidates().save(rd.candidates("t_prof"))
    write_union(rd, "t_prof", UNION_HTML)
    return rd


def test_prompted_run_writes_draft_and_generation_json(run: RunDir) -> None:
    llm = FakeLLM([GOOD])
    result = generate_run(run, mode="prompted", schema=SCHEMA, llm=llm, min_cluster_size=3)
    assert [c["template_id"] for c in llm.calls] == ["t_prof"]  # tail cluster skipped
    assert result.by_status("ok") == ["t_prof"]
    spec = Spec.load(run.draft_spec("t_prof"))
    assert spec.status == "draft" and spec.site == "example.org"
    assert spec.generated_by.pages_seen == [
        "https://example.org/staff/a-khoury",
        "https://example.org/staff/l-haddad",
    ]
    assert spec.not_present == {"phone": "no phone shown"}
    gen = read_json(run.generation_json)
    entry = gen["templates"]["t_prof"]
    assert entry["status"] == "ok" and entry["draft"] == "specs/draft/t_prof.json"
    assert entry["calls"] == 1 and entry["prompt_tokens"] > 500
    assert entry["not_present"] == {"phone": "no phone shown"}
    assert gen["mode"] == "prompted" and gen["prompt_version"] == "specgen.v2"
    assert gen["model"] == MODEL and gen["unprocessed"] == [] and gen["aborted"] is None
    assert gen["projected_cost_usd"] > 0
    assert "t_list" not in gen["templates"]
    assert "generate" in read_json(run.timings_json)
    assert not run.schema_proposal.exists()
    # The draft extracts the right values from every real page of the cluster.
    store = PageStore.open(run)
    rows = {
        store.url(pid): spec.apply(store.read_html(pid))
        for pid in Clusters.load(run.clusters_json).get("t_prof").page_ids
    }
    assert rows["https://example.org/staff/l-haddad"]["email"] is None
    assert rows["https://example.org/staff/s-azar"]["email"] == "s.azar@example.org"
    assert rows["https://example.org/staff/l-haddad"]["publications"][1] == {
        "title": "Union pages",
        "year": 2024,
    }


def test_requires_schema_in_prompted_mode(run: RunDir) -> None:
    with pytest.raises(ValueError, match="needs a schema"):
        generate_run(run, mode="prompted", llm=FakeLLM([]))
    with pytest.raises(ValueError, match="tail direct"):
        generate_run(run, mode="unprompted", tail="direct", llm=FakeLLM([]))


def test_double_failure_records_failed_fields(run: RunDir) -> None:
    bad = answer({"name": NAME, "email": BAD, "publications": PUBS}, {"phone": "none"})
    llm = FakeLLM([bad, bad])
    result = generate_run(run, mode="prompted", schema=SCHEMA, llm=llm, min_cluster_size=3)
    assert result.templates["t_prof"].status == "partial"
    entry = read_json(run.generation_json)["templates"]["t_prof"]
    assert entry["status"] == "partial"
    failed = entry["failed_fields"]["email"]
    assert failed["source_candidates"] == ["c2"] and failed["type"] == "scalar"
    assert failed["type_check"] == "email" and failed["errors"]
    spec = Spec.load(run.draft_spec("t_prof"))
    assert set(spec.fields) == {"name", "publications"}


def test_mapping_skips_excluded_and_focuses_union(
    run: RunDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates().model_copy(update={"template_id": "t_other"}).save(run.candidates("t_list"))
    built: list[dict[str, Any]] = []

    class FakeUnion:
        def __init__(self, tid: str, focus: set[str] | None) -> None:
            self.html = UNION_HTML.replace(
                "<!-- c3 · mandatory · on 100% of pages · e.g. B-80011 -->", ""
            )
            self.focus = focus
            self.tid = tid

        def meta(self) -> dict[str, Any]:
            return {
                "base_url": "https://example.org/staff/r-nasr",
                "donor_urls": [],
                "shown": ["c1", "c2", "c4", "c5"],
                "focus": sorted(self.focus or []),
            }

    def fake_build(
        rd: RunDir,
        *,
        template_ids: list[str] | None = None,
        focus: dict[str, set[str]] | None = None,
        **_: Any,
    ) -> dict[str, FakeUnion]:
        built.append({"template_ids": template_ids, "focus": focus})
        return {tid: FakeUnion(tid, (focus or {}).get(tid)) for tid in template_ids or []}

    monkeypatch.setattr(generate_mod, "build_union_run", fake_build)
    mapping = Mapping(
        schema_fields=SCHEMA.names(),
        templates={
            "t_prof": TemplateMapping(
                fields={
                    "name": "c1",
                    "email": "c2",
                    "publications": ["c4", "c5"],
                    "phone": "absent",
                }
            ),
            "t_list": TemplateMapping(included=False, note="directory pages"),
        },
    )
    mapping.save(run.mapping_json)  # picked up automatically in prompted mode
    llm = FakeLLM([answer({"name": NAME, "email": EMAIL, "publications": PUBS})])
    result = generate_run(run, mode="prompted", schema=SCHEMA, llm=llm, min_cluster_size=1)
    assert [c["template_id"] for c in llm.calls] == ["t_prof"]
    assert built == [{"template_ids": ["t_prof"], "focus": {"t_prof": {"c1", "c2", "c4", "c5"}}}]
    assert result.templates["t_list"].status == "skipped"
    assert "excluded in mapping.json" in (result.templates["t_list"].reason or "")
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "Confirmed mapping: c1" in prompt and "Confirmed mapping: c4, c5" in prompt
    assert "`phone`" not in prompt  # absent fields are not asked for
    assert "not on union page; path: html > body.page > div#profile > dl.contact > dd" in prompt
    t = result.templates["t_prof"]
    assert t.status == "ok" and t.focused
    assert t.spec is not None and t.spec.not_present["phone"].startswith("marked absent")
    assert t.spec.generated_by.pages_seen == ["https://example.org/staff/r-nasr"]
    gen = read_json(run.generation_json)
    assert gen["templates"]["t_list"]["status"] == "skipped"
    assert not run.draft_spec("t_list").exists()


def test_unmapped_template_is_skipped(run: RunDir) -> None:
    mapping = Mapping(templates={"t_other": TemplateMapping(fields={"name": "c1"})})
    llm = FakeLLM([])
    result = generate_run(
        run, mode="prompted", schema=SCHEMA, mapping=mapping, llm=llm, min_cluster_size=3
    )
    assert llm.calls == []
    assert result.templates["t_prof"].status == "skipped"
    assert result.templates["t_prof"].reason == "not in mapping.json"


def test_tail_spec_generates_tail_clusters_the_probe_never_mapped(run: RunDir) -> None:
    candidates().model_copy(update={"template_id": "t_list"}).save(run.candidates("t_list"))
    write_union(run, "t_list", UNION_HTML)
    # mapping.json covers non-tail templates only (the probe skips the tail)
    mapping = Mapping(templates={"t_other": TemplateMapping(fields={"name": "c1"})})
    llm = FakeLLM([GOOD])
    result = generate_run(
        run, mode="prompted", schema=SCHEMA, mapping=mapping, llm=llm, tail="spec"
    )
    assert result.templates["t_prof"].reason == "not in mapping.json"  # non-tail: still skipped
    assert [c["template_id"] for c in llm.calls] == ["t_list"]
    assert result.templates["t_list"].status == "ok" and run.draft_spec("t_list").exists()
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "`phone`" in prompt and "Confirmed mapping" not in prompt  # every field, unmapped


def _fake_union(monkeypatch: pytest.MonkeyPatch, built: list[dict[str, Any]]) -> None:
    class Page:
        html = UNION_HTML

        def __init__(self, focus: set[str] | None) -> None:
            self.focus = focus

        def meta(self) -> dict[str, Any]:
            return {"base_url": "u0", "donor_urls": [], "focus": sorted(self.focus or []) or None}

    def fake_build(rd: RunDir, **kw: Any) -> dict[str, Page]:
        built.append(kw)
        focus = kw.get("focus") or {}
        return {tid: Page(focus.get(tid)) for tid in kw["template_ids"]}

    monkeypatch.setattr(generate_mod, "build_union_run", fake_build)


def test_auto_entries_are_suggestions_and_keep_alternatives_in_focus(
    run: RunDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []
    _fake_union(monkeypatch, built)
    mapping = Mapping(
        schema_fields=SCHEMA.names(),
        templates={
            "t_prof": TemplateMapping(
                fields={"name": "c1", "email": "c3", "publications": ["c4", "c5"], "phone": "c2"},
                provenance={"name": "user", "email": "auto", "publications": "auto"},
                alternatives={
                    "email": [
                        Alternative(id="c3", use="c3", kind="mandatory", label="Office"),
                        Alternative(id="c2", use="c2", kind="optional", label="Email"),
                        Alternative(id="c99", use="c99"),  # unknown id: dropped
                    ],
                },
            ),
            "t_list": TemplateMapping(included=False),
        },
    )
    llm = FakeLLM([answer({"name": NAME, "email": EMAIL, "publications": PUBS, "phone": NAME})])
    result = generate_run(
        run, mode="prompted", schema=SCHEMA, mapping=mapping, llm=llm, min_cluster_size=1
    )
    assert result.templates["t_prof"].status in ("ok", "partial")
    prompt = llm.calls[0]["messages"][-1]["content"]
    # phone has no provenance entry: an old-style (binding) user mapping
    assert "`name`: full name. Confirmed mapping: c1" in prompt
    assert "Confirmed mapping: c2" in prompt
    assert (
        'Probe suggestions, best first: c3 (mandatory, label "Office"); '
        'c2 (optional, label "Email")'
    ) in prompt
    assert "c99" not in prompt.split("## Candidate table")[0]
    # the stored alternatives lack the group proposal, so it is added as the only suggestion
    assert "Probe suggestions, best first: c4 = [c4, c5]" in prompt
    assert built[0]["focus"] == {"t_prof": {"c1", "c2", "c3", "c4", "c5"}}


def test_auto_absent_is_still_asked_for_on_the_full_union_page(
    run: RunDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []
    _fake_union(monkeypatch, built)
    mapping = Mapping(
        schema_fields=SCHEMA.names(),
        templates={
            "t_prof": TemplateMapping(
                fields={"name": "c1", "email": "c2", "publications": "absent", "phone": "absent"},
                provenance={"publications": "auto", "phone": "user"},
            ),
            "t_list": TemplateMapping(included=False),
        },
    )
    llm = FakeLLM([answer({"name": NAME, "email": EMAIL, "publications": PUBS})])
    result = generate_run(
        run, mode="prompted", schema=SCHEMA, mapping=mapping, llm=llm, min_cluster_size=1
    )
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "`publications` (record_list). Probe found no candidate" in prompt
    assert "`phone`" not in prompt  # a user's absent stays binding
    assert all(b.get("focus") is None for b in built)  # publications needs the whole page
    t = result.templates["t_prof"]
    assert not t.focused
    assert t.spec is not None and "publications" in t.spec.fields
    assert t.spec.not_present["phone"].startswith("marked absent")


def test_unprompted_writes_schema_proposal(run: RunDir) -> None:
    write_union(run, "t_prof", UNION_HTML)
    llm = FakeLLM([answer({"person_name": NAME, "email": EMAIL, "publications": PUBS})])
    result = generate_run(run, mode="unprompted", llm=llm, min_cluster_size=3)
    assert result.templates["t_prof"].status == "ok"
    proposal = read_json(run.schema_proposal)
    assert proposal == result.proposal
    assert proposal["mode"] == "unprompted" and proposal["site"] == "example.org"
    tmpl = proposal["templates"]["t_prof"]
    assert tmpl["size"] == 6 and len(tmpl["sample_page_ids"]) == 5
    fields = tmpl["fields"]
    assert set(fields) == {"person_name", "email", "publications"}
    assert fields["person_name"]["type"] == "scalar"
    assert len(fields["person_name"]["examples"]) == 3
    assert all(isinstance(e, str) for e in fields["person_name"]["examples"])
    assert fields["email"]["type_check"] == "email" and fields["email"]["optional"] is True
    assert fields["publications"]["columns"] == ["title", "year"]
    first_rows = fields["publications"]["examples"][0]
    assert isinstance(first_rows, list) and set(first_rows[0]) == {"title", "year"}
    assert "Mode: unprompted" in llm.calls[0]["messages"][-1]["content"]


def test_unprompted_rebuilds_a_focused_union_page(
    run: RunDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_union(run, "t_prof", UNION_HTML, focus=["c1"])
    calls: list[Any] = []

    class Page:
        html = UNION_HTML

        def meta(self) -> dict[str, Any]:
            return {"base_url": "u0", "donor_urls": ["u1"], "focus": None}

    def fake_build(rd: RunDir, **kw: Any) -> dict[str, Page]:
        calls.append(kw)
        return {"t_prof": Page()}

    monkeypatch.setattr(generate_mod, "build_union_run", fake_build)
    llm = FakeLLM([answer({"person_name": NAME})])
    result = generate_run(run, mode="unprompted", llm=llm, min_cluster_size=3)
    assert calls == [{"template_ids": ["t_prof"], "focus": None}]
    spec = result.templates["t_prof"].spec
    assert spec is not None and spec.generated_by.pages_seen == ["u0", "u1"]


def test_budget_refusal_records_and_raises(run: RunDir, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BUDGET_USD", "0.000001")
    llm = FakeLLM([GOOD])
    with pytest.raises(BudgetExceeded):
        generate_run(run, mode="prompted", schema=SCHEMA, llm=llm, min_cluster_size=3)
    assert llm.calls == []
    gen = read_json(run.generation_json)
    assert gen["unprocessed"] == ["t_prof"]
    assert gen["aborted"].startswith("refused")
    assert gen["templates"]["t_prof"]["status"] == "unprocessed"
    assert not run.draft_spec("t_prof").exists()


def test_budget_abort_mid_run_lists_unprocessed(run: RunDir) -> None:
    candidates().model_copy(update={"template_id": "t_list"}).save(run.candidates("t_list"))
    write_union(run, "t_list", UNION_HTML)

    def respond(messages: list[Message], tid: str | None) -> str | Exception:
        return GOOD if tid == "t_prof" else BudgetExceeded("spent 40.00 of 40.00 USD")

    llm = FakeLLM(respond)
    result = generate_run(run, mode="prompted", schema=SCHEMA, llm=llm, tail="spec", concurrency=1)
    assert result.templates["t_prof"].status == "ok"  # largest template first
    assert result.unprocessed == ["t_list"]
    assert result.aborted is not None and "budget" in result.aborted
    gen = read_json(run.generation_json)
    assert gen["unprocessed"] == ["t_list"]
    assert gen["templates"]["t_list"]["status"] == "unprocessed"
    assert run.draft_spec("t_prof").exists() and not run.draft_spec("t_list").exists()


def test_tail_spec_forces_small_clusters_and_missing_inputs_are_skipped(run: RunDir) -> None:
    llm = FakeLLM([GOOD])
    result = generate_run(run, mode="prompted", schema=SCHEMA, llm=llm, tail="spec")
    assert result.templates["t_prof"].status == "ok"
    assert result.templates["t_list"].status == "skipped"
    assert "no candidate table" in (result.templates["t_list"].reason or "")


def test_subset_run_merges_generation_json_and_clears_stale_drafts(run: RunDir) -> None:
    generate_run(run, mode="prompted", schema=SCHEMA, llm=FakeLLM([GOOD]), min_cluster_size=3)
    assert run.draft_spec("t_prof").exists()
    gen = read_json(run.generation_json)
    gen["templates"]["t_keep"] = {"status": "ok", "failed_fields": {}}
    run.generation_json.write_text(json.dumps(gen))
    # A failing rerun of t_prof only: the old draft must go, other entries survive.
    llm = FakeLLM(["not json", "still not json"])
    result = generate_run(
        run, mode="prompted", schema=SCHEMA, llm=llm, min_cluster_size=3, template_ids=["t_prof"]
    )
    assert result.templates["t_prof"].status == "failed"
    assert not run.draft_spec("t_prof").exists()
    gen = read_json(run.generation_json)
    assert "t_keep" in gen["templates"]
    assert gen["templates"]["t_prof"]["status"] == "failed"
    assert set(gen["templates"]["t_prof"]["failed_fields"]) == {
        "name",
        "email",
        "publications",
        "phone",
    }
    with pytest.raises(KeyError):
        generate_run(run, mode="prompted", schema=SCHEMA, llm=FakeLLM([]), template_ids=["t_nope"])


def test_tail_direct_runs_after_generation(run: RunDir) -> None:
    def tail_answer(messages: list[Message], tid: str | None) -> str:
        return json.dumps(
            {"name": "Staff directory", "email": None, "publications": [], "phone": None}
        )

    tail_llm = FakeLLM(tail_answer, model=SMALL, role="tail_direct")
    result = generate_run(
        run,
        mode="prompted",
        schema=SCHEMA,
        llm=FakeLLM([GOOD]),
        min_cluster_size=3,
        tail="direct",
        tail_llm=tail_llm,
    )
    assert result.tail is not None and result.tail.rows == 2
    assert len(tail_llm.calls) == 2
    gen = read_json(run.generation_json)
    assert gen["tail_direct"]["rows"] == 2 and gen["tail"] == "direct"
    lines = run.out_jsonl("example.org", None).read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["template_id"] is None
