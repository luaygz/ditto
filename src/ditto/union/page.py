"""Stage 5: one annotated, compressed union page per template.

``build_union_page`` is pure (raw HTML in, ``UnionPage`` out):

1. survey the sampled pages and pick as base the page holding the most field candidates;
2. splice every field candidate the base lacks from donor pages (``splice``);
3. prune attributes, mark candidates and their labels, trim to ``focus``, truncate text and
   remove candidate-free boilerplate (``prepare``);
4. collapse repeated siblings by variant, and while over ``max_tokens`` drop candidate-free
   subtrees, then raise collapse aggressiveness, then shorten candidate text, and as a last
   resort drop the least frequent candidates (``compress``);
5. render with annotation comments only; no ``data-x-*`` attribute survives.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, field
from typing import Any

from ditto.analyze.models import CandidateTable
from ditto.fingerprint.dom import clean_html
from ditto.union.compress import (
    COLLAPSE_LEVELS,
    candidate_note,
    collapse,
    drop_to_fit,
    render,
    shorten_candidate_text,
    unmark_candidates,
)
from ditto.union.prepare import (
    X_C,
    mark_tree,
    marks,
    normalize_text,
    prune_attrs,
    remove_boilerplate,
    trim_to_focus,
)
from ditto.union.splice import choose_base, cid_key, splice_missing, survey
from ditto.union.tokens import count_tokens
from ditto.union.tree import CandIndex, Element, Tree, iter_elements, locate

LAST_RESORT_TEXT_LIMIT = 120


@dataclass
class UnionPage:
    """One template's union page plus how it was built (``meta()`` is ``union/<tid>.json``)."""

    html: str
    tokens: int
    base_page_id: int
    donor_page_ids: list[int]
    spliced: list[str]
    collapsed_groups: int
    dropped: list[str]
    log: list[str]
    template_id: str = ""
    shown: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    focus: list[str] | None = None
    collapse_level: int = 0
    sample_page_ids: list[int] = field(default_factory=list)
    min_tokens: int = 8000
    max_tokens: int = 30000
    seconds: float = 0.0
    base_url: str | None = None
    donor_urls: list[str] = field(default_factory=list)

    @property
    def under_min(self) -> bool:
        return self.tokens < self.min_tokens

    def meta(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "tokens": self.tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "under_min": self.under_min,
            "base_page_id": self.base_page_id,
            "base_url": self.base_url,
            "donor_page_ids": self.donor_page_ids,
            "donor_urls": self.donor_urls,
            "sample_page_ids": self.sample_page_ids,
            "spliced": self.spliced,
            "shown": self.shown,
            "missing": self.missing,
            "collapsed_groups": self.collapsed_groups,
            "collapse_level": self.collapse_level,
            "dropped": self.dropped,
            "focus": self.focus,
            "seconds": round(self.seconds, 4),
            "log": self.log,
        }


def build_union_page(
    table: CandidateTable,
    pages: Mapping[int, str],
    *,
    min_tokens: int = 8000,
    max_tokens: int = 30000,
    focus: Set[str] | None = None,
) -> UnionPage:
    """Build the union page of one template from sampled raw pages (``page_id -> html``).

    ``focus`` (candidate ids from ``mapping.json``) trims the page to those candidates'
    neighborhoods plus their labels. A page under ``min_tokens`` is left as is (never padded);
    ``max_tokens`` is a hard cap."""
    start = time.perf_counter()
    if not pages:
        raise ValueError("build_union_page needs at least one page")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    log: list[str] = []
    index = CandIndex.build(table)

    focus_ids: set[str] | None = None
    if focus is not None:
        unknown = sorted(set(focus) - set(index.fields), key=cid_key)
        if unknown:
            log.append(f"focus ids that are not field candidates, ignored: {', '.join(unknown)}")
        focus_ids = set(focus) & set(index.fields) or None
        if focus_ids is None:
            log.append("focus has no field candidate: building the full union page")
    targets = focus_ids if focus_ids is not None else set(index.fields)

    surveys = survey(pages, index)
    base_survey = choose_base(surveys, focus_ids)
    base_id = base_survey.page_id
    log.append(
        f"base page {base_id} holds {len(base_survey.present & targets)}/{len(targets)} "
        f"field candidates ({len(surveys)} sampled pages)"
    )
    base = Tree(clean_html(pages[base_id]))
    prune_attrs(base.root)
    missing = sorted(targets - base_survey.present, key=cid_key)
    spliced, donors = splice_missing(base, base_id, pages, surveys, index, missing, log.append)
    if spliced:
        log.append(f"spliced {len(spliced)} candidates from {len(donors)} donor pages")

    located = locate(base, index)
    labels_of = mark_tree(base, index, located)
    if focus_ids is not None:
        removed = trim_to_focus(base, index, located, labels_of, focus_ids)
        log.append(f"focus on {len(focus_ids)} candidates: trimmed {removed} subtrees")
    normalize_text(base.root)
    boilerplate = remove_boilerplate(base.root)
    log.append(f"removed {len(boilerplate)} candidate-free boilerplate subtrees")

    notes = {cid: candidate_note(c) for cid, c in index.fields.items()}
    fit = _fit(base.root, notes, max_tokens, index, targets, log.append)
    shown = sorted({cid for el in iter_elements(fit.root) for cid in marks(el, X_C)}, key=cid_key)
    lost = sorted(targets - set(shown), key=cid_key)
    if fit.tokens < min_tokens:
        log.append(
            f"final page: {fit.tokens} tokens, under min_tokens={min_tokens} "
            "(the template is small; not padded)"
        )
    else:
        log.append(f"final page: {fit.tokens} tokens (budget {min_tokens}-{max_tokens})")
    return UnionPage(
        html=fit.html,
        tokens=fit.tokens,
        base_page_id=base_id,
        donor_page_ids=donors,
        spliced=spliced,
        collapsed_groups=fit.collapsed,
        dropped=fit.dropped,
        log=log,
        template_id=table.template_id,
        shown=shown,
        missing=lost,
        focus=sorted(focus_ids, key=cid_key) if focus_ids is not None else None,
        collapse_level=fit.level,
        sample_page_ids=sorted(pages),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        seconds=time.perf_counter() - start,
    )


@dataclass
class _Fit:
    root: Element
    html: str
    tokens: int
    level: int
    collapsed: int
    dropped: list[str]


def _fit(
    prepared: Element,
    notes: Mapping[str, str],
    max_tokens: int,
    index: CandIndex,
    targets: set[str],
    log: Callable[[str], None],
) -> _Fit:
    """Compress a fresh copy of ``prepared`` at increasing collapse levels; at each level drop
    candidate-free subtrees while over budget. Past the last level: shorten candidate text,
    then forget the least frequent candidates (non-targets first) until the page fits."""

    def render_fn(root: Element) -> str:
        return render(root, notes)

    work = prepared
    html = ""
    tokens = 0
    groups = 0
    dropped: list[str] = []
    level = 0
    for level, params in enumerate(COLLAPSE_LEVELS):
        work = copy.deepcopy(prepared)
        groups = collapse(work, params)
        html = render_fn(work)
        tokens = count_tokens(html)
        log(f"collapse level {level}: {groups} groups collapsed, {tokens} tokens")
        dropped = []
        if tokens > max_tokens:
            html, tokens, dropped = drop_to_fit(work, max_tokens, render_fn, html, tokens)
            if dropped:
                log(f"dropped {len(dropped)} candidate-free subtrees: {tokens} tokens")
        if tokens <= max_tokens:
            return _Fit(work, html, tokens, level, groups, dropped)

    shorten_candidate_text(work, LAST_RESORT_TEXT_LIMIT)
    html = render_fn(work)
    tokens = count_tokens(html)
    log(f"candidate text cut to {LAST_RESORT_TEXT_LIMIT} chars: {tokens} tokens")
    present = {cid for el in iter_elements(work) for cid in marks(el, X_C)}
    order = sorted(
        present,
        key=lambda cid: (cid in targets, index.fields[cid].p, cid_key(cid)),
    )
    while tokens > max_tokens and order:
        batch = order[: max(1, len(order) // 10)]
        order = order[len(batch) :]
        unmark_candidates(work, set(batch))
        dropped.extend(f"candidate {cid} (p={index.fields[cid].p:.2f})" for cid in batch)
        html = render_fn(work)
        tokens = count_tokens(html)
        html, tokens, more = drop_to_fit(work, max_tokens, render_fn, html, tokens)
        dropped.extend(more)
    log(f"over budget after collapse level {level}: dropped candidates, {tokens} tokens")
    return _Fit(work, html, tokens, level, groups, dropped)
