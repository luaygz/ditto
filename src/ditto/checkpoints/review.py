"""Stage 9c: review. Show extracted samples beside the page and record the sign-off.

After verification, ``review_run`` applies each final spec (``specs/<tid>.json``) to ``n``
random pages of its template and shows every field's extracted value next to the page text
around the node it came from. The user approves, skips, or names fields to reject with a note.
Rejections are written to ``review.json`` and handed to ``ditto.verify.repair_rejected``;
approved samples become ``golden/<tid>.jsonl``.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import orjson
from lxml import etree
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table as RichTable

from ditto.checkpoints._common import InputFn, ask, make_console, short, split_names
from ditto.cluster.models import Clusters
from ditto.extract.runner import apply_spec, evaluate
from ditto.fingerprint.dom import clean_html
from ditto.run import RunDir, write_json
from ditto.specgen.models import ListField, RecordListField, ScalarField, Spec, SpecField
from ditto.textutil import normalize_ws

if TYPE_CHECKING:
    from ditto.llm.base import LLM

CONTEXT_WIDTH = 120
DEFAULT_NOTE = "rejected at review"


class PageSource(Protocol):
    """What review needs from the page store (``ditto.ingest.PageStore`` satisfies it)."""

    def read_html(self, page_id: int) -> str: ...

    def url(self, page_id: int) -> str: ...


class Sample(BaseModel):
    page_id: int
    url: str
    record: dict[str, Any] = Field(description="Extracted fields plus _meta, as in out/*.jsonl.")
    sources: dict[str, str] = Field(description="Field -> page text around the matched node.")


class TemplateReview(BaseModel):
    status: Literal["approved", "rejected", "skipped"]
    reason: str | None = None
    page_ids: list[int] = Field(default_factory=list)
    approved_fields: list[str] = Field(default_factory=list)
    rejected: dict[str, str] = Field(default_factory=dict, description="Field -> user's note.")
    golden: str | None = Field(None, description="Golden file, relative to the run directory.")
    samples: list[Sample] = Field(default_factory=list, exclude=True)


class ReviewResult(BaseModel):
    """``review.json`` (samples are returned but not written; golden files hold them)."""

    version: Literal[1] = 1
    n: int
    seed: int
    signed_off_by: Literal["user", "yes"]
    templates: dict[str, TemplateReview] = Field(default_factory=dict)
    repair: dict[str, Any] | None = None

    @property
    def rejections(self) -> dict[str, dict[str, str]]:
        return {tid: dict(tr.rejected) for tid, tr in self.templates.items() if tr.rejected}

    def records(self, template_id: str) -> list[dict[str, Any]]:
        return [s.record for s in self.templates[template_id].samples]


# -- page text around a node ---------------------------------------------------------------------


def _text(el: etree._Element) -> str:
    """Readable text of a subtree: text pieces joined by spaces (``<dt>A</dt><dd>B</dd>`` reads
    ``A B``, not ``AB``)."""
    return normalize_ws(" ".join(str(t) for t in el.itertext()))


def _window(text: str, needle: str, width: int = CONTEXT_WIDTH) -> str:
    """``text`` cut to ``width`` characters centred on ``needle`` when it is longer."""
    if len(text) <= width:
        return text
    i = text.find(needle) if needle else -1
    if i < 0:
        return text[: width - 1] + "…"
    start = max(0, min(i - (width - len(needle)) // 2, len(text) - width))
    out = text[start : start + width]
    return ("…" if start > 0 else "") + out + ("…" if start + width < len(text) else "")


def _item_context(item: Any) -> str:
    getparent = getattr(item, "getparent", None)
    if isinstance(item, etree._Element):
        el: etree._Element = item
        parent = el.getparent()
        own = _text(el)
        around = _text(parent) if parent is not None else own
        return _window(around or own, own)
    if isinstance(item, str) and getparent is not None:
        owner = getparent()
        if owner is None:
            return short(str(item), CONTEXT_WIDTH)
        if getattr(item, "is_attribute", False):
            label = _text(owner)
            attr = getattr(item, "attrname", "attr")
            text = f'<{owner.tag} {attr}="{item}">' + (f" {label}" if label else "")
            return short(text, CONTEXT_WIDTH)
        return _item_context(owner)
    return short(item, CONTEXT_WIDTH)


def source_context(field: SpecField, root: etree._Element) -> str:
    """The page text around what ``field`` matches on this page (``(no match)`` otherwise)."""
    if isinstance(field, ScalarField | ListField):
        order = [field.selected, *[i for i in field.fallbacks if i != field.selected]]
        for idx in order:
            items = evaluate(field.candidates[idx], root)
            if items:
                more = (
                    f" (+{len(items) - 1} more)" if field.type == "list" and len(items) > 1 else ""
                )
                return _item_context(items[0]) + more
        return "(no match)"
    if isinstance(field, RecordListField):
        containers = [field.container, *field.container_alternates]
        what = "rows"
    else:
        containers = [field.pairs, *field.pairs_alternates]
        what = "pairs"
    for expr in containers:
        items = [i for i in evaluate(expr, root) if isinstance(i, etree._Element)]
        if items:
            parent = items[0].getparent()
            text = _text(parent if parent is not None else items[0])
            return f"{len(items)} {what}: " + short(text, CONTEXT_WIDTH)
    return "(no match)"


def sample_page(spec: Spec, pages: PageSource, page_id: int) -> Sample:
    root = clean_html(pages.read_html(page_id))
    record = apply_spec(spec, root)
    sources = {name: source_context(f, root) for name, f in spec.fields.items()}
    return Sample(page_id=page_id, url=pages.url(page_id), record=record, sources=sources)


# -- rendering and sign-off ------------------------------------------------------------------------


def render_samples(tid: str, size: int, spec: Spec, samples: list[Sample], con: Console) -> None:
    table = RichTable(
        title=f"{tid} · {size} pages · {len(samples)} random pages", title_justify="left"
    )
    table.add_column("page", overflow="fold", max_width=40)
    table.add_column("field", style="bold")
    table.add_column("extracted", overflow="fold", max_width=60)
    table.add_column("on the page", overflow="fold", max_width=70, style="dim")
    for i, s in enumerate(samples):
        if i:
            table.add_section()
        for j, name in enumerate(spec.fields):
            page = f"#{s.page_id} {s.url}" if j == 0 else ""
            table.add_row(page, name, short(s.record.get(name), 120), s.sources.get(name, ""))
    con.print(table)


def ask_signoff(
    tid: str, field_names: list[str], input_fn: InputFn, con: Console
) -> tuple[str, dict[str, str]]:
    """Returns ``(decision, rejected)``: decision is approve | skip | quit."""
    prompt = f"{tid}: Enter/y = approve · s = skip · q = stop reviewing · or fields to reject: "
    while True:
        ans = ask(input_fn, prompt)
        low = ans.lower()
        if low in ("", "y", "yes"):
            return "approve", {}
        if low in ("s", "skip"):
            return "skip", {}
        if low in ("q", "quit"):
            return "quit", {}
        names = split_names(ans)
        unknown = [n for n in names if n not in field_names]
        if unknown:
            con.print(
                f"  [red]unknown field(s): {', '.join(unknown)}[/red] "
                f"(fields: {', '.join(field_names)})"
            )
            continue
        rejected: dict[str, str] = {}
        for name in names:
            note = ask(input_fn, f"  what is wrong with {name}? ")
            rejected[name] = note or DEFAULT_NOTE
        return "approve", rejected


def golden_rows(
    tid: str, spec: Spec, samples: list[Sample], approved: list[str], by: str, rejected: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        row: dict[str, Any] = {"url": s.url, "page_id": s.page_id, "template_id": tid}
        row.update({name: s.record.get(name) for name in approved})
        row["_review"] = {
            "approved_by": by,
            "rejected": rejected,
            "spec_version": spec.spec_version,
        }
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))


def _open_pages(run: RunDir) -> PageSource:
    from ditto.ingest import PageStore

    store: PageSource = PageStore.open(run)
    return store


def _repair(
    run: RunDir,
    rejections: dict[str, dict[str, str]],
    llm: LLM | None,
    workers: int | None,
) -> dict[str, Any]:
    """Send the rejected fields with the user's notes through repair (role ``repair``)."""
    from ditto.verify import repair_rejected

    if llm is None:
        from ditto.llm import get_llm

        llm = get_llm("repair", run=run)
    reports = repair_rejected(run, rejections, llm=llm, workers=workers)
    return {"called": True, "templates": sorted(reports)}


def review_run(
    run: RunDir,
    *,
    yes: bool = False,
    n: int = 5,
    seed: int = 0,
    input_fn: InputFn = input,
    pages: PageSource | None = None,
    console: Console | None = None,
    llm: LLM | None = None,
    repair: bool = True,
    workers: int | None = None,
) -> ReviewResult:
    """Review ``n`` extracted records per verified template and record the sign-off.

    ``yes`` approves every sample without showing it. Rejected fields (with the user's note)
    go to ``review.json`` and, when ``repair`` is on, through ``ditto.verify.repair_rejected``
    (role ``repair``; ``llm`` defaults to the configured one). Approved samples, minus rejected
    fields, are written to ``golden/<tid>.jsonl``.
    """
    t0 = time.perf_counter()
    con = make_console(console)
    result = ReviewResult(n=n, seed=seed, signed_off_by="yes" if yes else "user")
    spec_paths = sorted(run.specs_dir.glob("*.json"))
    clusters = Clusters.load(run.clusters_json) if spec_paths else None
    store = pages if pages is not None or not spec_paths else _open_pages(run)
    stop = False
    n_pages = 0
    for path in spec_paths:
        tid = path.stem
        if stop:
            result.templates[tid] = TemplateReview(status="skipped", reason="review stopped")
            continue
        assert clusters is not None and store is not None
        try:
            cluster = clusters.get(tid)
        except KeyError:
            result.templates[tid] = TemplateReview(
                status="skipped", reason="template not in clusters.json"
            )
            continue
        spec = Spec.load(path)
        rng = random.Random(f"{seed}:{tid}")
        pids = sorted(rng.sample(cluster.page_ids, min(n, len(cluster.page_ids))))
        samples = [sample_page(spec, store, pid) for pid in pids]
        n_pages += len(samples)
        names = list(spec.fields)
        if yes:
            decision, rejected = "approve", dict[str, str]()
        else:
            render_samples(tid, cluster.size, spec, samples, con)
            decision, rejected = ask_signoff(tid, names, input_fn, con)
        if decision != "approve":
            stop = decision == "quit"
            result.templates[tid] = TemplateReview(
                status="skipped",
                reason="stopped by user" if stop else "skipped by user",
                page_ids=pids,
                samples=samples,
            )
            continue
        approved = [f for f in names if f not in rejected]
        golden = run.golden(tid)
        _write_jsonl(
            golden,
            golden_rows(tid, spec, samples, approved, result.signed_off_by, sorted(rejected)),
        )
        result.templates[tid] = TemplateReview(
            status="rejected" if rejected else "approved",
            page_ids=pids,
            approved_fields=approved,
            rejected=rejected,
            golden=str(golden.relative_to(run.root)),
            samples=samples,
        )
    rejections = result.rejections
    if rejections:
        result.repair = (
            _repair(run, rejections, llm, workers)
            if repair
            else {"called": False, "reason": "repair disabled"}
        )
    write_json(run.review_json, result.model_dump(mode="json"))
    run.record_timing("review", time.perf_counter() - t0, n_pages, 1)
    counts = [tr.status for tr in result.templates.values()]
    con.print(
        f"review: {counts.count('approved')} approved, {counts.count('rejected')} with "
        f"rejected fields, {counts.count('skipped')} skipped"
    )
    return result
