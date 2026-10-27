"""Stage 5: one annotated, compressed union page per template.

``build_union_page`` is pure (raw HTML in, ``UnionPage`` out):

1. survey the sampled pages and pick as base the page holding the most field candidates;
2. splice the field candidates the base lacks from donor pages, most frequent first, while the
   spliced units' estimated tokens stay within ``max_tokens``; a splice whose candidate does
   not locate on the union page, or that hides a candidate at least as frequent, is undone
   (``splice``); candidates no sampled page has are looked for in more pages only while budget
   is left;
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
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Any

from ditto.analyze.models import CandidateTable
from ditto.fingerprint.dom import clean_html
from ditto.union.compress import (
    COLLAPSE_LEVELS,
    candidate_note,
    collapse,
    drop_to_fit,
    estimate_tokens,
    render,
    shorten_candidate_text,
    unmark_candidates,
    unmark_labels,
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
from ditto.union.splice import PageSurvey, choose_base, cid_key, splice_missing, survey
from ditto.union.tokens import count_tokens
from ditto.union.tree import CandIndex, Element, Tree, iter_elements, locate

LAST_RESORT_TEXT_LIMIT = 120
SPLICE_NOTE = "<!-- spliced from another page: c00 -->"


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
    searched_pages: int = 0
    """Pages beyond the sample surveyed for donors of candidates no sampled page has."""
    budget_skipped: list[str] = field(default_factory=list)
    """Field candidates some page has that were not spliced: the token budget was spent."""
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
            "searched_pages": self.searched_pages,
            "spliced": self.spliced,
            "budget_skipped": self.budget_skipped,
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
    more_pages: Iterable[tuple[int, str]] = (),
) -> UnionPage:
    """Build the union page of one template from sampled raw pages (``page_id -> html``).

    ``focus`` (candidate ids from ``mapping.json``) trims the page to those candidates'
    neighborhoods plus their labels. ``more_pages`` (``(page_id, html)``, read lazily, e.g. the
    analysis sample) are searched in order for donors of candidates no sampled page has, if
    splice budget is left. A page under ``min_tokens`` is left as is (never padded);
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
    sample_ids = sorted(pages)
    pages = dict(pages)
    base = Tree(clean_html(pages[base_id]))
    prune_attrs(base.root)
    notes = {cid: candidate_note(c) for cid, c in index.fields.items()}
    budget = max_tokens

    def cost(inserted: Sequence[Element], carried: Collection[str]) -> int:
        return _unit_cost(inserted, carried, notes)

    absent = set(targets).difference(*(s.present for s in surveys))
    missing = targets - base_survey.present - absent
    done = splice_missing(
        base, base_id, pages, surveys, index, missing, log.append, budget=budget, cost=cost
    )
    searched = 0
    if absent and done.spent < budget:
        pages, searched = _search_donors(pages, surveys, index, targets, more_pages, log.append)
        more = splice_missing(
            base,
            base_id,
            pages,
            surveys,
            index,
            absent,
            log.append,
            budget=budget - done.spent,
            cost=cost,
        )
        done.spliced += more.spliced
        done.donors += [d for d in more.donors if d not in done.donors]
        done.over_budget += more.over_budget
        done.spent += more.spent
    elif absent:
        log.append(
            f"{len(absent)} field candidates no sampled page has were not searched for: "
            "the splice budget is spent"
        )
    if done.spliced:
        log.append(
            f"spliced {len(done.spliced)} candidates from {len(done.donors)} donor pages "
            f"(~{done.spent} tokens)"
        )

    located = locate(base, index)
    labels_of = mark_tree(base, index, located)
    if focus_ids is not None:
        removed = trim_to_focus(base, index, located, labels_of, focus_ids)
        log.append(f"focus on {len(focus_ids)} candidates: trimmed {removed} subtrees")
    normalize_text(base.root)
    boilerplate = remove_boilerplate(base.root)
    log.append(f"removed {len(boilerplate)} candidate-free boilerplate subtrees")
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
        donor_page_ids=done.donors,
        spliced=done.spliced,
        collapsed_groups=fit.collapsed,
        dropped=fit.dropped,
        log=log,
        template_id=table.template_id,
        shown=shown,
        missing=lost,
        focus=sorted(focus_ids, key=cid_key) if focus_ids is not None else None,
        collapse_level=fit.level,
        sample_page_ids=sample_ids,
        searched_pages=searched,
        budget_skipped=sorted(done.over_budget, key=cid_key),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        seconds=time.perf_counter() - start,
    )


def _unit_cost(
    inserted: Sequence[Element], carried: Collection[str], notes: Mapping[str, str]
) -> int:
    """Estimated tokens one splice adds: its elements with text truncated as on the page, the
    splice comment and the annotations of the candidates it brings."""
    total = count_tokens(SPLICE_NOTE)
    for el in inserted:
        dup = copy.deepcopy(el)
        dup.tail = None
        normalize_text(dup)
        total += estimate_tokens(dup)
    return total + sum(count_tokens(f"<!-- {notes[c]} -->") for c in carried if c in notes)


def _search_donors(
    pages: Mapping[int, str],
    surveys: list[PageSurvey],
    index: CandIndex,
    targets: Set[str],
    more_pages: Iterable[tuple[int, str]],
    log: Callable[[str], None],
) -> tuple[dict[int, str], int]:
    """Survey ``more_pages`` in order until every target some page has is found; pages holding
    a target no surveyed page has join ``pages`` and ``surveys`` (as donors). Returns the pages
    and how many more were surveyed."""
    out = dict(pages)
    absent = set(targets).difference(*(s.present for s in surveys))
    if not absent:
        return out, 0
    wanted = len(absent)
    searched = 0
    for pid, html in more_pages:
        if pid in out:
            continue
        searched += 1
        (found,) = survey({pid: html}, index)
        if found.present & absent:
            out[pid] = html
            surveys.append(found)
            absent -= found.present
            if not absent:
                break
    log(
        f"searched {searched} more pages for {wanted} field candidates no sampled page has: "
        f"found {wanted - len(absent)}"
    )
    return out, searched


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
    then forget the least frequent candidates (non-targets first) until the page fits, and if
    it still does not, the remaining labels (of tables whose values are not candidates), so
    ``max_tokens`` always holds."""

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
    if tokens > max_tokens:
        unmark_labels(work)
        dropped.append("every remaining label")
        html, tokens, more = drop_to_fit(work, max_tokens, render_fn, html, tokens)
        dropped.extend(more)
    log(f"over budget after collapse level {level}: dropped candidates, {tokens} tokens")
    return _Fit(work, html, tokens, level, groups, dropped)
