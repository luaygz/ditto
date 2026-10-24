"""Hand-built candidate tables, clusters and run directories for the checkpoint tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ditto.analyze.models import Candidate, CandidateTable, Table, TableColumn
from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.run import RunDir
from ditto.schema import FieldsSchema


def cand(
    cid: str,
    path: str,
    cls: str = "mandatory",
    *,
    p: float = 1.0,
    label: str | None = None,
    examples: Sequence[str] = (),
    value_type: str | None = None,
    repeat: tuple[int, int] = (1, 1),
    attr: str | None = None,
    table_id: str | None = None,
    table_role: str | None = None,
    column: str | None = None,
) -> Candidate:
    return Candidate.model_validate(
        {
            "id": cid,
            "path": path,
            "attr": attr,
            "class": cls,
            "p": p,
            "m": 0.1 if cls not in ("label", "boilerplate") else 1.0,
            "pages_present": int(p * 100),
            "repeat_min": repeat[0],
            "repeat_max": repeat[1],
            "label_hint": label,
            "examples": list(examples),
            "value_type": value_type,
            "table_id": table_id,
            "table_role": table_role,
            "column": column,
        }
    )


def table(
    tid: str, candidates: list[Candidate], tables: list[Table] | None = None
) -> CandidateTable:
    return CandidateTable(
        template_id=tid,
        cluster_size=100,
        n_sampled=100,
        candidates=candidates,
        tables=tables or [],
    )


BODY = "html > body.job"


def job_table(tid: str = "t_job1") -> CandidateTable:
    """A job posting detail template."""
    return table(
        tid,
        [
            cand("c0", f"{BODY} > div.header > a.logo", "boilerplate", examples=["Jobs.example"]),
            cand("c1", "html > title", examples=["Line Cook - Jobs.example"]),
            cand(
                "c2",
                f"{BODY} > h1",
                examples=["Senior Data Engineer", "Registered Nurse", "Line Cook"],
            ),
            cand("c3", f"{BODY} > dl > dt", "label", examples=["Company:"]),
            cand(
                "c4",
                f"{BODY} > dl > dd.employer",
                label="Company:",
                examples=["Acme Corp", "Globex Inc", "Initech"],
            ),
            cand(
                "c5",
                f"{BODY} > div.meta > span.where",
                label="Address",
                examples=["Austin, TX", "Denver, CO", "Remote"],
            ),
            cand(
                "c6",
                f"{BODY} > div.meta > span.when",
                label="Posted on",
                examples=["2026-10-01", "2026-09-12", "2026-08-30"],
                value_type="date",
            ),
            cand(
                "c7",
                f"{BODY} > div.pay > span",
                "optional",
                p=0.6,
                label="Cost to company",
                examples=["$120,000", "$95,000", "$40,000"],
                value_type="price",
            ),
            cand(
                "c8",
                f"{BODY} > a.apply @href",
                attr="href",
                label="Apply now",
                examples=["/apply/1", "/apply/2", "/apply/3"],
                value_type="url",
            ),
            cand(
                "c9",
                f"{BODY} > div.body > p",
                "list",
                repeat=(2, 9),
                examples=["We are hiring a ...", "You will ...", "Benefits include ..."],
            ),
        ],
    )


def listing_table(tid: str = "t_list1") -> CandidateTable:
    """A search-result listing template: holds nothing the job schema asks for."""
    return table(
        tid,
        [
            cand("c0", "html > body.search > div.pager > a", "boilerplate", examples=["Next"]),
            cand(
                "c1",
                "html > body.search > ul > li > span.count",
                examples=["12 results", "40 results"],
            ),
        ],
    )


JOB_FIELDS: dict[str, Any] = {
    "title": "the job title",
    "company": "the hiring company",
    "location": "where the job is",
    "date_posted": {"description": "when the job was posted", "type_check": "date"},
    "salary": "the pay",
    "remote_policy": "remote, hybrid or on-site",
}


def job_schema() -> FieldsSchema:
    return FieldsSchema.from_obj(JOB_FIELDS)


def clusters(specs: dict[str, int], *, tail: dict[str, int] | None = None) -> Clusters:
    """Clusters with consecutive page ids: ``{template_id: size}``."""
    out: list[Cluster] = []
    groups: list[SignatureGroup] = []
    next_id = 0
    for i, (tid, size) in enumerate({**specs, **(tail or {})}.items()):
        pids = list(range(next_id, next_id + size))
        next_id += size
        out.append(
            Cluster(
                template_id=tid,
                size=size,
                is_tail=tid in (tail or {}),
                signature_groups=[i],
                url_patterns=[f"https://example.org/{tid}/*"],
                url_examples=[f"https://example.org/{tid}/{p}" for p in pids[:3]],
                page_ids=pids,
            )
        )
        groups.append(
            SignatureGroup(id=i, sig_hash=f"{i:016x}", size=size, template_id=tid, page_ids=pids)
        )
    return Clusters(n_pages=next_id, min_cluster_size=20, clusters=out, signature_groups=groups)


def make_run(root: Path, tables: list[CandidateTable], cl: Clusters) -> RunDir:
    run = RunDir(root).ensure()
    cl.save(run.clusters_json)
    for t in tables:
        t.save(run.candidates(t.template_id))
    return run


def kv_table(tid: str = "t_kv1") -> CandidateTable:
    """Job page whose facts sit in a key/value list (dl) plus a record-list table."""
    kv = Table(
        id="tb0",
        kind="key_value",
        container_path="html > body > dl.facts",
        row_path="html > body > dl.facts > dt",
        p=1.0,
        rows_min=3,
        rows_max=5,
        key_candidate="c1",
        value_candidate="c2",
        keys_seen=["Salary", "Job type", "Schedule"],
    )
    rec = Table(
        id="tb1",
        kind="record_list",
        container_path="html > body > table.reviews",
        row_path="html > body > table.reviews > tr",
        p=0.8,
        rows_min=1,
        rows_max=12,
        columns=[
            TableColumn(name="Author", candidate_id="c4"),
            TableColumn(name="Stars", candidate_id="c5"),
        ],
    )
    return table(
        tid,
        [
            cand("c0", "html > body > h1", examples=["Barista", "Welder", "Clerk"]),
            cand(
                "c1",
                "html > body > dl.facts > dt",
                "list",
                repeat=(3, 5),
                examples=["Salary", "Job type"],
                table_id="tb0",
                table_role="kv_key",
            ),
            cand(
                "c2",
                "html > body > dl.facts > dd",
                "list",
                repeat=(3, 5),
                examples=["$15 an hour", "Full-time", "Day shift"],
                table_id="tb0",
                table_role="kv_value",
            ),
            cand(
                "c3",
                "html > body > table.reviews > tr > th",
                "boilerplate",
                examples=["Author"],
                table_id="tb1",
                table_role="record_header",
            ),
            cand(
                "c4",
                "html > body > table.reviews > tr > td.author",
                "list",
                p=0.8,
                repeat=(1, 12),
                examples=["Ann", "Bo", "Cy"],
                table_id="tb1",
                table_role="record_column",
                column="Author",
            ),
            cand(
                "c5",
                "html > body > table.reviews > tr > td.stars",
                "list",
                p=0.8,
                repeat=(1, 12),
                examples=["4", "5", "2"],
                table_id="tb1",
                table_role="record_column",
                column="Stars",
            ),
        ],
        [kv, rec],
    )
