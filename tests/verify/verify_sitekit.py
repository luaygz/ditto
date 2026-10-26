"""Helpers for verification tests: a small staff-directory site ingested into a run directory,
hand-built clusters and candidate tables (paths taken from the real cleaned pages), and a fake
LLM implementing the ``LLM`` protocol."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ditto.analyze.models import Candidate, CandidateTable, Table, TableColumn
from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.fingerprint.dom import clean_html, iter_text_items
from ditto.ingest import PageStore, ingest
from ditto.llm.base import Completion, Message, Usage
from ditto.run import RunDir, write_json
from ditto.specgen.models import GeneratedBy, Spec

BASE = "https://example.org/staff/"
TID = "t_staff"


def cls(name: str) -> str:
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {name} ')"


def staff_page(
    i: int, *, variant: str = "a", email: bool = True, tags: int = 2, pubs: int = 2
) -> str:
    name = f"Person {i:03d}"
    if variant == "a":
        heading = f'<h1 itemprop="name">{name}</h1>'
    elif variant == "b":
        heading = f'<h1 class="title">{name}</h1>'
    else:
        heading = f'<h2 class="fullname">{name}</h2>'
    mail = (
        f'<dt>Email</dt><dd><a href="mailto:p{i}@example.org">p{i}@example.org</a></dd>'
        if email
        else ""
    )
    tag_items = "".join(f"<li>topic{i}-{k}</li>" for k in range(tags))
    rows = "".join(
        f"<tr><td>Paper {i}.{k}</td><td>{2000 + (i + k) % 20}</td></tr>" for k in range(pubs)
    )
    return f"""<html><head><title>{name} — Staff</title><script>var x = {i};</script></head>
<body class="page">
<nav class="menu"><a href="/">Home</a> <a href="/staff">Staff</a></nav>
<main>
{heading}
<dl class="profile">{mail}<dt>Phone</dt><dd>+961 1 {i:03d} {i * 7 % 1000:03d}</dd>
<dt>Office</dt><dd>Room {i}</dd></dl>
<ul class="tags">{tag_items}</ul>
<table class="pubs"><tr><th>Title</th><th>Year</th></tr>{rows}</table>
</main>
<footer>Example University</footer>
</body></html>"""


def build_run(tmp_path: Path, pages: Sequence[tuple[str, str]]) -> RunDir:
    """Ingest ``(url, html)`` pages into ``tmp_path/run``."""
    src = tmp_path / "site"
    src.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, (url, html) in enumerate(pages):
        (src / f"page{n:04d}.html").write_text(html, encoding="utf-8")
        lines.append(f"page{n:04d}.html\t{url}")
    (src / "urls.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ingest(src, tmp_path / "run", workers=1)
    return RunDir(tmp_path / "run")


def ids_of(run: RunDir, urls: Sequence[str]) -> list[int]:
    store = PageStore.open(run)
    out = []
    for url in urls:
        pid = store.lookup(url)
        assert pid is not None, url
        out.append(pid)
    return out


def text_path(html: str, text: str, attr: str | None = None) -> str:
    """Structural path of the first text item whose text equals ``text``."""
    for item in iter_text_items(clean_html(html)):
        if item.text == text and item.attr == attr:
            return item.path
    raise AssertionError(f"no text item {text!r}")


def save_clusters(
    run: RunDir,
    groups: Sequence[Sequence[int]],
    *,
    tid: str = TID,
    min_cluster_size: int = 5,
    extra: Sequence[tuple[str, Sequence[Sequence[int]]]] = (),
) -> Clusters:
    """One cluster per ``(tid, groups)``; every inner list is one signature group."""
    clusters: list[Cluster] = []
    sig_groups: list[SignatureGroup] = []
    for template_id, members in [(tid, groups), *extra]:
        gids = []
        for pids in members:
            gid = len(sig_groups)
            gids.append(gid)
            sig_groups.append(
                SignatureGroup(
                    id=gid,
                    sig_hash=f"{gid + 1:016x}",
                    size=len(pids),
                    template_id=template_id,
                    page_ids=sorted(pids),
                )
            )
        pages = sorted(p for pids in members for p in pids)
        clusters.append(
            Cluster(
                template_id=template_id,
                size=len(pages),
                is_tail=len(pages) < min_cluster_size,
                signature_groups=gids,
                url_patterns=["/staff/{slug}"],
                url_examples=[BASE + "x"],
                page_ids=pages,
            )
        )
    out = Clusters(
        n_pages=sum(c.size for c in clusters),
        min_cluster_size=min_cluster_size,
        clusters=clusters,
        signature_groups=sig_groups,
    )
    out.save(run.clusters_json)
    return out


def cand(cid: str, path: str, p: float, cls_: str = "mandatory", **kw: Any) -> Candidate:
    return Candidate.model_validate(
        {"id": cid, "path": path, "class": cls_, "p": p, "m": 0.1, **kw}
    )


def staff_table(
    run: RunDir,
    sample_a: str,
    *,
    sample_b: str | None = None,
    sample_c: str | None = None,
    p_a: float = 1.0,
    p_email: float = 0.5,
    tid: str = TID,
    size: int = 20,
) -> CandidateTable:
    """Candidate table with paths read off real pages: c0 name (variant a), c1 name (b),
    c2 email (link text), c3 phone/office dd, c4 tags li, c5 pub cell, c6 dt keys,
    c7 name (variant c); t0 pubs record table, t1 profile key/value table."""
    name_a = sample_a.split('itemprop="name">')[1].split("<")[0]
    email_text = sample_a.split("mailto:")[1].split('"')[0]
    office = sample_a.split("<dt>Office</dt><dd>")[1].split("<")[0]
    cands = [
        cand("c0", text_path(sample_a, name_a), p_a),
        cand(
            "c2",
            text_path(sample_a, email_text),
            p_email,
            "optional",
            value_type="email",
            examples=[email_text],
        ),
        cand("c3", text_path(sample_a, office), 1.0, repeat_max=2),
        cand(
            "c4",
            text_path(sample_a, sample_a.split("<li>")[1].split("<")[0]),
            1.0,
            "list",
            repeat_max=3,
        ),
        cand(
            "c5",
            text_path(sample_a, sample_a.split("<td>")[1].split("<")[0]),
            1.0,
            "list",
            table_id="t0",
            table_role="record_column",
        ),
        cand("c6", text_path(sample_a, "Phone"), 1.0, "label", table_id="t1", table_role="kv_key"),
    ]
    if sample_b is not None:
        name_b = sample_b.split('class="title">')[1].split("<")[0]
        cands.append(cand("c1", text_path(sample_b, name_b), 1.0 - p_a))
    if sample_c is not None:
        name_c = sample_c.split('class="fullname">')[1].split("<")[0]
        cands.append(cand("c7", text_path(sample_c, name_c), 0.2))
    row = cands[4].path.rsplit(" > ", 1)[0]
    dt_path = cands[5].path
    tables = [
        Table(
            id="t0",
            kind="record_list",
            container_path=row.rsplit(" > ", 1)[0],
            row_path=row,
            p=1.0,
            rows_min=1,
            rows_max=3,
            columns=[
                TableColumn(name="Title", candidate_id="c5"),
                TableColumn(name="Year", candidate_id="c5"),
            ],
        ),
        Table(
            id="t1",
            kind="key_value",
            container_path=dt_path.rsplit(" > ", 1)[0],
            row_path=dt_path,
            p=1.0,
            key_candidate="c6",
            value_candidate="c3",
            keys_seen=["Email", "Phone", "Office"],
        ),
    ]
    table = CandidateTable(
        template_id=tid, cluster_size=size, n_sampled=size, candidates=cands, tables=tables
    )
    table.save(run.candidates(tid))
    return table


def save_draft(run: RunDir, fields: dict[str, Any], *, tid: str = TID) -> Spec:
    spec = Spec.model_validate(
        {
            "template_id": tid,
            "site": run.site,
            "generated_by": GeneratedBy(
                model="fake/model", prompt_version="specgen.v1"
            ).model_dump(),
            "fields": fields,
        }
    )
    spec.save(run.draft_spec(tid))
    return spec


def write_generation(run: RunDir, templates: dict[str, Any]) -> None:
    write_json(run.generation_json, {"version": 1, "templates": templates})


class FakeLLM:
    """Scripted ``LLM``: returns the given answers in order (the last one repeats)."""

    def __init__(self, answers: Sequence[str | dict[str, Any]], role: str = "repair") -> None:
        self.model = "fake/model"
        self.role = role
        self.answers = list(answers)
        self.calls: list[list[Message]] = []
        self.kwargs: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        self.calls.append(messages)
        self.kwargs.append({"prompt_version": prompt_version, "template_id": template_id})
        answer = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return Completion(text=text, usage=Usage(100, 20), model=self.model)

    def user_text(self, call: int = 0) -> str:
        return "\n".join(m["content"] for m in self.calls[call] if m["role"] == "user")
