"""review_run: samples beside page text, sign-off, rejections, golden files."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from checkpoint_builders import clusters
from rich.console import Console

from ditto.checkpoints import review_run
from ditto.checkpoints.review import _window, source_context
from ditto.cluster.models import Cluster, Clusters
from ditto.fingerprint.dom import clean_html
from ditto.run import RunDir
from ditto.specgen.models import (
    GeneratedBy,
    KeyValueField,
    ListField,
    RecordListField,
    ScalarField,
    Spec,
)

TITLES = ["Line Cook", "Barista", "Welder", "Nurse", "Clerk", "Driver", "Baker", "Tutor"]


def page_html(i: int) -> str:
    title = TITLES[i % len(TITLES)]
    return f"""<html><head><title>{title} - Jobs</title></head><body>
    <div class="head"><h1 class="title">{title}</h1></div>
    <p class="pay">Pay: <span class="price">${10 + i}.00</span> per hour</p>
    <a class="apply" href="/apply/{i}">Apply for {title}</a>
    <ul class="tags"><li>food</li><li>shift {i}</li></ul>
    <dl class="facts"><div><dt>Schedule</dt><dd>Days</dd></div><div><dt>Type</dt><dd>Full</dd></div></dl>
    <table class="reviews"><tr><th>Author</th><th>Stars</th></tr>
      <tr><td>Ann</td><td>{i % 5}</td></tr></table>
    </body></html>"""


def job_spec(tid: str = "t_job1") -> Spec:
    return Spec(
        template_id=tid,
        site="example.org",
        status="final",
        generated_by=GeneratedBy(model="fake/m", prompt_version="specgen.v1"),
        fields={
            "title": ScalarField(candidates=['//h1[@class="title"]', '//div[@class="head"]/h1']),
            "price": ScalarField(
                candidates=['//span[@class="price"]', '//p[@class="pay"]/span'],
                type_check="price",
            ),
            "link": ScalarField(
                candidates=['//a[@class="apply"]/@href', '//div[@class="cta"]/a/@href']
            ),
            "tags": ListField(candidates=['//ul[@class="tags"]/li', '//div[@class="tags"]//li']),
            "facts": KeyValueField(pairs='//dl[@class="facts"]/div', key="./dt", value="./dd"),
            "reviews": RecordListField(
                container='//table[@class="reviews"]//tr[td]',
                columns={"author": "./td[1]", "stars": "./td[2]"},
            ),
        },
    )


class FakePages:
    def __init__(self, n: int) -> None:
        self.html = {i: page_html(i) for i in range(n)}
        self.reads: list[int] = []

    def read_html(self, page_id: int) -> str:
        self.reads.append(page_id)
        return self.html[page_id]

    def url(self, page_id: int) -> str:
        return f"https://example.org/jobs/{page_id}"


def scripted(*answers: str) -> Callable[[str], str]:
    queue = list(answers)

    def input_fn(prompt: str) -> str:
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    return input_fn


def recording_console() -> Console:
    return Console(record=True, width=200, file=open("/dev/null", "w"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    run = RunDir(tmp_path / "run").ensure()
    clusters({"t_job1": 8}).save(run.clusters_json)
    job_spec().save(run.spec("t_job1"))
    job_spec("t_draft_only").save(run.draft_spec("t_draft_only"))  # drafts are not reviewed
    return run


def test_yes_approves_samples_and_writes_golden(run: RunDir) -> None:
    pages = FakePages(8)
    result = review_run(run, yes=True, n=5, seed=3, pages=pages, console=recording_console())
    tr = result.templates["t_job1"]
    assert tr.status == "approved"
    assert len(tr.page_ids) == 5 and len(set(tr.page_ids)) == 5
    assert sorted(pages.reads) == tr.page_ids
    assert set(result.templates) == {"t_job1"}
    rows = read_jsonl(run.golden("t_job1"))
    assert [r["page_id"] for r in rows] == tr.page_ids
    first = rows[0]
    i = first["page_id"]
    assert first["url"] == f"https://example.org/jobs/{i}"
    assert first["template_id"] == "t_job1"
    assert first["title"] == TITLES[i]
    assert first["price"] == f"${10 + i}.00"
    assert first["link"] == f"/apply/{i}"
    assert first["tags"] == ["food", f"shift {i}"]
    assert first["facts"] == {"Schedule": "Days", "Type": "Full"}
    assert first["reviews"] == [{"author": "Ann", "stars": str(i % 5)}]
    assert first["_review"] == {"approved_by": "yes", "rejected": [], "spec_version": "1"}
    assert result.records("t_job1")[0]["title"] == TITLES[i]
    on_disk = json.loads(run.review_json.read_text())
    assert on_disk["signed_off_by"] == "yes"
    assert on_disk["templates"]["t_job1"]["golden"] == "golden/t_job1.jsonl"
    assert "samples" not in on_disk["templates"]["t_job1"]
    assert on_disk["repair"] is None
    # same seed, same pages
    again = review_run(run, yes=True, n=5, seed=3, pages=FakePages(8), console=recording_console())
    assert again.templates["t_job1"].page_ids == tr.page_ids


def test_interactive_approval_shows_values_beside_page_text(run: RunDir) -> None:
    con = recording_console()
    result = review_run(run, n=2, pages=FakePages(8), input_fn=scripted(""), console=con)
    assert result.templates["t_job1"].status == "approved"
    assert result.signed_off_by == "user"
    shown = con.export_text()
    i = result.templates["t_job1"].page_ids[0]
    assert f"#{i} https://example.org/jobs/{i}" in shown
    assert f"Pay: ${10 + i}.00 per hour" in shown  # page text around the price node
    assert f'<a href="/apply/{i}"> Apply for {TITLES[i]}' in shown  # attribute context
    assert "2 pairs: Schedule Days Type Full" in shown
    assert "1 rows: Author Stars Ann" in shown
    rows = read_jsonl(run.golden("t_job1"))
    assert rows[0]["_review"]["approved_by"] == "user"


def test_rejection_is_recorded_and_sent_to_repair(
    run: RunDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ditto.verify

    calls: list[tuple[Any, ...]] = []

    def fake_repair(
        run_: RunDir, rejections: dict[str, dict[str, str]], *, llm: Any = None, workers: Any = None
    ) -> dict[str, Any]:
        calls.append((run_, rejections, llm, workers))
        return {"t_job1": object()}

    monkeypatch.setattr(ditto.verify, "repair_rejected", fake_repair)
    sentinel_llm: Any = object()
    answers = scripted("price, nope", "price link", "includes per hour", "")
    result = review_run(
        run,
        n=3,
        pages=FakePages(8),
        input_fn=answers,
        console=recording_console(),
        llm=sentinel_llm,
    )
    tr = result.templates["t_job1"]
    assert tr.status == "rejected"
    assert tr.rejected == {"price": "includes per hour", "link": "rejected at review"}
    assert tr.approved_fields == ["title", "tags", "facts", "reviews"]
    assert calls == [(run, {"t_job1": tr.rejected}, sentinel_llm, None)]
    assert result.repair == {"called": True, "templates": ["t_job1"]}
    review = json.loads(run.review_json.read_text())
    assert review["templates"]["t_job1"]["rejected"] == tr.rejected
    rows = read_jsonl(run.golden("t_job1"))
    assert len(rows) == 3
    assert "price" not in rows[0] and "link" not in rows[0] and "title" in rows[0]
    assert rows[0]["_review"]["rejected"] == ["link", "price"]


def test_rejection_without_repair_is_only_recorded(run: RunDir) -> None:
    result = review_run(
        run,
        n=1,
        pages=FakePages(8),
        input_fn=scripted("title", "wrong node"),
        console=recording_console(),
        repair=False,
    )
    assert result.rejections == {"t_job1": {"title": "wrong node"}}
    assert result.repair == {"called": False, "reason": "repair disabled"}


def test_skip_and_quit(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "run").ensure()
    clusters({"t_a": 4, "t_b": 4, "t_c": 4}).save(run.clusters_json)
    for tid in ("t_a", "t_b", "t_c"):
        job_spec(tid).save(run.spec(tid))
    pages = FakePages(12)
    result = review_run(
        run, n=2, pages=pages, input_fn=scripted("s", "q"), console=recording_console()
    )
    assert [result.templates[t].status for t in ("t_a", "t_b", "t_c")] == ["skipped"] * 3
    assert result.templates["t_a"].reason == "skipped by user"
    assert result.templates["t_b"].reason == "stopped by user"
    assert result.templates["t_c"].reason == "review stopped"
    assert not run.golden_dir.exists()


def test_spec_without_cluster_is_skipped_and_empty_run(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "run").ensure()
    Clusters(n_pages=0, min_cluster_size=20, clusters=[], signature_groups=[]).save(
        run.clusters_json
    )
    job_spec("t_gone").save(run.spec("t_gone"))
    result = review_run(run, pages=FakePages(0), console=recording_console())
    assert result.templates["t_gone"].status == "skipped"
    assert result.templates["t_gone"].reason == "template not in clusters.json"
    empty = review_run(RunDir(tmp_path / "empty"), yes=True, console=recording_console())
    assert empty.templates == {}


def test_review_reads_pages_from_the_ingested_store(tmp_path: Path) -> None:
    from ditto.ingest import PageStore, ingest

    src = tmp_path / "site"
    src.mkdir()
    lines = []
    for i in range(4):
        (src / f"p{i}.html").write_text(page_html(i))
        lines.append(f"p{i}.html\thttps://example.org/jobs/{i}")
    (src / "urls.tsv").write_text("\n".join(lines) + "\n")
    run = RunDir(tmp_path / "run")
    ingest(src, run.root, workers=1)
    store = PageStore.open(run)
    pids = store.ids()
    Clusters(
        n_pages=len(pids),
        min_cluster_size=1,
        clusters=[
            Cluster(
                template_id="t_job1",
                size=len(pids),
                is_tail=False,
                signature_groups=[],
                page_ids=pids,
            )
        ],
        signature_groups=[],
    ).save(run.clusters_json)
    job_spec().save(run.spec("t_job1"))
    result = review_run(run, yes=True, n=2, console=recording_console())
    rows = read_jsonl(run.golden("t_job1"))
    assert len(rows) == 2
    for row in rows:
        i = int(row["url"].rsplit("/", 1)[1])
        assert row["title"] == TITLES[i]
    assert result.templates["t_job1"].status == "approved"


def test_source_context_helpers() -> None:
    root = clean_html(page_html(1))
    spec = job_spec()
    assert source_context(spec.fields["title"], root) == "Barista"
    missing = ScalarField(candidates=['//h2[@class="nope"]', '//h3[@class="nope"]'])
    assert source_context(missing, root) == "(no match)"
    text_node = ScalarField(candidates=['//h1[@class="title"]/text()', '//h1[@id="x"]'])
    assert source_context(text_node, root) == "Barista"
    tags = source_context(spec.fields["tags"], root)
    assert tags.endswith("(+1 more)")
    long = "word " * 60 + "NEEDLE " + "word " * 60
    cut = _window(long, "NEEDLE", width=40)
    assert "NEEDLE" in cut and cut.startswith("…") and cut.endswith("…")
    assert _window("short", "x") == "short"
    assert _window("a" * 200, "zzz", width=10) == "a" * 9 + "…"
