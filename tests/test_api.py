"""The Python API (``ditto.Site``) end to end on the fixture site with a fake LLM, ``Site.from_warc``
on a WARC written in the test, and ``Spec.load(path).apply(html)`` on its own."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from pipeline_fakes import FIELDS, SITE, FakeLLM, FakeLLMs, copy_run
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

import ditto
import ditto.llm
import ditto.llm.client
from ditto import FieldsSchema, Mapping, Probe, Site, Spec
from ditto.checkpoints import ConfirmError
from ditto.cluster.models import Clusters
from ditto.ingest import PageStore
from ditto.schema import ABSENT


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeLLMs:
    """Isolated ledger; ``get_llm`` hands out fakes and real adapters fail the test."""
    monkeypatch.setenv("DITTO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_BUDGET_USD", "40")
    monkeypatch.delenv("DITTO_RUN_BUDGET_USD", raising=False)

    def no_network(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a real LLM adapter was requested")

    monkeypatch.setattr(ditto.llm.client, "make_adapter", no_network)
    fakes = FakeLLMs()
    monkeypatch.setattr(ditto.llm, "get_llm", fakes)
    return fakes


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("api") / "run"
    site = Site.from_dir(SITE, run_dir=root, workers=1)
    clusters = site.cluster(workers=1)
    assert len(clusters.templates()) == 4
    tables = site.analyze(workers=1)
    assert sorted(tables) == sorted(c.template_id for c in clusters.templates())
    return root


@pytest.fixture
def site(analyzed: Path, tmp_path: Path) -> Site:
    return Site.open(copy_run(analyzed, tmp_path / "run").root)


def template_ids(site: Site) -> set[str]:
    return {c.template_id for c in Clusters.load(site.run.clusters_json).templates()}


def test_public_names() -> None:
    assert set(ditto.__all__) >= {"FieldsSchema", "Mapping", "Probe", "Site", "Spec", "__version__"}
    assert isinstance(ditto.__version__, str)


def test_full_flow_and_results(site: Site, offline: FakeLLMs) -> None:
    probe = site.probe(FIELDS)
    assert isinstance(probe, Probe) and set(probe.result.templates) == template_ids(site)
    mapping = probe.confirm()
    assert isinstance(mapping, Mapping) and site.run.mapping_json.exists()
    assert mapping.schema_fields == ["title", "published", "tags"]

    llm = FakeLLM(site.run)
    gen = site.generate("prompted", schema=FieldsSchema.load(FIELDS), llm=llm)
    assert set(gen.by_status("ok")) == template_ids(site)
    assert len(llm.calls) == 4 and offline.made == []  # the passed LLM is used

    reports = site.verify(max_repair_calls=0)
    assert reports and offline.made == []  # no repair LLM without repair calls
    review = site.review(n=3)
    assert review.signed_off_by == "yes"
    assert all(t.status == "approved" for t in review.templates.values())
    summary = site.extract(workers=1)
    assert summary.pages > 0 and site.run.report_md.exists()

    records = list(site.records())
    assert len(records) == summary.pages
    assert all(r["title"] for r in records)
    specs = site.specs()
    assert specs and all(isinstance(s, Spec) for s in specs.values())
    assert {r["template_id"] for r in records} <= set(specs) | {
        tid for tid in summary.templates if not tid.startswith("t_")
    }


def test_confirm_with_edits(site: Site) -> None:
    probe = site.probe(FieldsSchema.load(FIELDS))
    tid = next(iter(probe.result.templates))
    mapping = probe.confirm(edits={tid: {"tags": ABSENT}})
    assert mapping.templates[tid].fields["tags"] == ABSENT
    assert mapping.templates[tid].provenance_of("tags") == "user"
    assert mapping.templates[tid].provenance_of("title") == "auto"  # accepted unseen
    assert Mapping.load(site.run.mapping_json).templates[tid].fields["tags"] == ABSENT
    assert "use: absent" in site.run.probe_yaml.read_text()  # edits live in probe.yaml too

    before = site.run.probe_yaml.read_text()
    with pytest.raises(ConfirmError, match="unknown candidate id 'c999'"):
        probe.confirm(edits={tid: {"title": "c999"}})
    with pytest.raises(ConfirmError, match="t_nope: unknown template"):
        probe.confirm(edits={"t_nope": {"title": "c1"}})
    assert site.run.probe_yaml.read_text() == before  # rejected edits leave probe.yaml alone


def test_confirm_edits_without_probe_yaml(site: Site) -> None:
    probe = site.probe(FIELDS)
    site.run.probe_yaml.unlink()
    tid = next(iter(probe.result.templates))
    assert probe.confirm(edits={tid: {"title": ABSENT}}).templates[tid].fields["title"] == ABSENT


def test_confirm_edits_on_broken_probe_yaml(site: Site) -> None:
    probe = site.probe(FIELDS)
    site.run.probe_yaml.write_text("- just\n- a list\n")
    with pytest.raises(ConfirmError, match="templates"):
        probe.confirm(edits={"t_x": {"title": ABSENT}})


def test_defaults_use_configured_llms(site: Site, offline: FakeLLMs) -> None:
    probe = site.probe(FIELDS, use_llm=True)
    assert probe.result.llm.used and [m.role for m in offline.made] == ["probe"]
    probe.confirm()
    site.generate("prompted", schema=FIELDS)  # mapping.json picked up, llm from get_llm
    assert offline.made[-1].role == "generation" and len(offline.calls("generation")) == 4
    site.verify()
    assert offline.made[-1].role == "repair"


def test_unprompted_generate(site: Site) -> None:
    gen = site.generate("unprompted", llm=FakeLLM(site.run))
    assert gen.by_status("ok") and gen.proposal is not None
    assert site.run.schema_proposal.exists()


def test_spec_apply_standalone(site: Site) -> None:
    site.probe(FIELDS).confirm()
    site.generate("prompted", schema=FIELDS, llm=FakeLLM(site.run))
    site.verify(max_repair_calls=0)
    site.extract(workers=1)
    row = next(site.records())
    path = site.run.spec(row["template_id"])
    html = PageStore.open(site.run).read_html(row["page_id"])

    spec = Spec.load(path)  # nothing else: no run directory, no store
    got = spec.apply(html)
    want = {k: v for k, v in row.items() if k not in ("url", "page_id", "template_id")}
    assert got == want
    assert spec.apply(html.encode()) == got


def write_warc(path: Path, pages: list[tuple[str, str]]) -> None:
    with path.open("wb") as fh:
        writer = WARCWriter(fh, gzip=True)
        for url, html in pages:
            http = StatusAndHeaders(
                "200 OK", [("Content-Type", "text/html; charset=utf-8")], protocol="HTTP/1.1"
            )
            record = writer.create_warc_record(
                url, "response", payload=io.BytesIO(html.encode()), http_headers=http
            )
            writer.write_record(record)


def _fixture_pages(n: int) -> list[tuple[str, str]]:
    lines = (SITE / "urls.tsv").read_text().splitlines()[:n]
    return [(url, (SITE / name).read_text()) for name, url in (ln.split("\t") for ln in lines)]


def test_from_warc(tmp_path: Path) -> None:
    pages = _fixture_pages(6)
    write_warc(tmp_path / "a.warc.gz", pages[:3])
    write_warc(tmp_path / "b.warc.gz", pages[3:])

    one = Site.from_warc(tmp_path / "a.warc.gz", run_dir=tmp_path / "one", workers=1)
    assert len(PageStore.open(one.run)) == 3

    both = Site.from_warc(
        [tmp_path / "a.warc.gz", str(tmp_path / "b.warc.gz")], run_dir=tmp_path / "both"
    )
    store = PageStore.open(both.run)
    assert sorted(store.url(pid) for pid in store.ids()) == sorted(u for u, _ in pages)
    pid = store.lookup(pages[4][0])
    assert pid is not None and store.read_html(pid) == pages[4][1]


def test_default_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_warc(tmp_path / "crawl.warc.gz", _fixture_pages(2))
    site = Site.from_warc(tmp_path / "crawl.warc.gz")
    assert site.run.root == tmp_path / "ditto-runs" / "crawl.warc"
    assert site.run.read_meta()["n_pages"] == 2
