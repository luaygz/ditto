"""Does a union page show a known field? The v0.2 gate metric, over ground-truth values.

A known field (a scalar, a list, a record-table column, a key-value block) with known values per
page is **covered** by its template's union page when the page holds an *annotated candidate at
the field's structural path whose node shows one of the field's values*:

1. **Field path.** On (up to ``max_pages``) pages where the field is present, find the elements
   of the cleaned page (``clean_html``) whose own text equals one of the field's values on that
   page; the field's path is the structural path found on the most pages (ties: the
   lexicographically first). No match on any page: the field is ``unlocated``.
2. **Shown.** The union page carries ``<!-- cN · ... -->`` immediately before candidate
   ``cN``'s node; the candidate's nodes there are that element and its following siblings with
   the same tag and class up to the next annotation (repeats, and a label and value that share
   one path, e.g. ``<span>Area:</span><span>90 m²</span>``). The field is covered by such an
   annotation when candidate ``cN``'s path (from the candidate table) is the field's path and
   one of its nodes' own text is one of the field's values on any page of the template (a value
   cut in the middle with `` … `` matches by head and tail). When several candidates share the
   field's path (cells of one table, keys of one block, fields in identical wrappers), a
   candidate whose label (record column header or key-value key, else its label hint) is the
   label of another field of the template does not count.

Reasons a field is not covered: ``unlocated``; ``no_candidate`` (no candidate at the path);
``not_shown`` (no candidate of the field at the path is annotated on the union page: classified
noise, not spliced, or dropped to fit the budget); ``value_mismatch`` (its annotated
candidates show none of its values).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lxml import etree

from ditto.analyze.models import Candidate, CandidateTable
from ditto.fingerprint.dom import clean_html, iter_text_items, own_text
from ditto.textutil import normalize_ws
from ditto.union.tree import label_key

ANNOTATION = re.compile(r"^\s*(c\d+)(?:\s+@(\S+))?\s+·")
ELLIPSIS = " … "
DEFAULT_MAX_PAGES = 50

TextItems = Sequence[tuple[str, str]]
"""``(structural path, own text)`` of every text-bearing element of one cleaned page."""


def page_text_items(html: str) -> list[tuple[str, str]]:
    """``(path, own text)`` of every element with own text on the cleaned page."""
    return [(it.path, it.text) for it in iter_text_items(clean_html(html)) if it.attr is None]


@dataclass(frozen=True)
class Annotated:
    """One candidate annotation on a union page and the own texts of the nodes it marks."""

    cid: str
    attr: str | None
    texts: tuple[str, ...]


def annotated_nodes(union_html: str) -> list[Annotated]:
    """Every candidate annotation (``<!-- c17 · ... -->``, ``<!-- c18 @href · ... -->``) and the
    own texts of the element it precedes plus that element's following siblings with the same
    tag and class up to the next annotation, in document order."""
    if not union_html.strip():
        return []
    root = etree.fromstring(union_html, etree.HTMLParser())
    if root is None:
        return []
    out: list[Annotated] = []
    for comment in root.iter(etree.Comment):
        m = ANNOTATION.match(comment.text or "")
        if not m:
            continue
        node = comment.getnext()
        while node is not None and not isinstance(node.tag, str):
            node = node.getnext()
        texts: list[str] = []
        if node is not None:
            texts.append(own_text(node))
            for sib in node.itersiblings():
                if not isinstance(sib.tag, str):
                    if sib.tag is etree.Comment and ANNOTATION.match(sib.text or ""):
                        break
                    continue
                if sib.tag != node.tag or sib.get("class") != node.get("class"):
                    break
                texts.append(own_text(sib))
        out.append(Annotated(m.group(1), m.group(2), tuple(texts)))
    return out


def text_matches(shown: str, values: Collection[str]) -> bool:
    """``shown`` is one of ``values`` (normalized), or a middle-cut `` … `` form of one."""
    text = normalize_ws(shown)
    if not text:
        return False
    if text in values:
        return True
    if ELLIPSIS not in text:
        return False
    head, tail = text.split(ELLIPSIS, 1)
    return any(
        len(v) > len(head) + len(tail) and v.startswith(head) and v.endswith(tail) for v in values
    )


@dataclass(frozen=True)
class FieldPath:
    """Where a field sits: its modal structural path and the evidence for it."""

    path: str | None
    support: int
    """Pages on which a value of the field sits at ``path``."""
    examined: int
    """Pages examined (the field present on each)."""
    paths: int
    """Distinct paths at which a value of the field was found."""


def field_path(observations: Iterable[tuple[TextItems, Collection[str]]]) -> FieldPath:
    """Modal path of the elements whose own text equals one of the field's values, over
    ``(page text items, the field's values on that page)`` for pages where it is present."""
    counts: Counter[str] = Counter()
    examined = 0
    for items, raw in observations:
        values = {v for v in (normalize_ws(x) for x in raw) if v}
        if not values:
            continue
        examined += 1
        counts.update({path for path, text in items if text in values})
    if not counts:
        return FieldPath(None, 0, examined, 0)
    path, support = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return FieldPath(path, support, examined, len(counts))


@dataclass
class FieldCoverage:
    """Whether one field is covered by the union page, and why not."""

    name: str
    located: FieldPath
    covered: bool
    reason: str | None = None
    detail: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    shown: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.name,
            "covered": self.covered,
            "reason": self.reason,
            "detail": self.detail,
            "path": self.located.path,
            "path_support": f"{self.located.support}/{self.located.examined}",
            "candidates": self.candidates,
            "shown": self.shown,
        }


def candidate_label(c: Candidate) -> str | None:
    """What a candidate is labeled with: its column header / key, else its label hint."""
    return c.column or c.label_hint


def _cand_info(c: Candidate) -> dict[str, Any]:
    return {
        "id": c.id,
        "class": c.cls,
        "p": c.p,
        "pages": c.pages_present,
        "label": candidate_label(c),
    }


def field_coverage(
    name: str,
    located: FieldPath,
    table: CandidateTable,
    annotations: Sequence[Annotated],
    values: Collection[str],
    *,
    other_labels: Collection[str] = (),
) -> FieldCoverage:
    """Coverage of field ``name`` (found at ``located``) by a union page's ``annotations``.

    ``values``: the field's values over the template's pages; ``other_labels``: labels of the
    template's other fields (where several candidates share the path, one labeled as another
    field is not this field)."""
    if located.path is None:
        return FieldCoverage(
            name,
            located,
            covered=False,
            reason="unlocated",
            detail=f"no element's own text equals the field's value on the {located.examined} "
            "pages examined",
        )
    at_path = [c for c in table.candidates if c.path == located.path]
    if not at_path:
        return FieldCoverage(
            name,
            located,
            covered=False,
            reason="no_candidate",
            detail="the candidate table has no candidate at the field's path",
        )
    others = {label_key(x) for x in other_labels} if len(at_path) > 1 else set()
    mine = [c for c in at_path if not ((lab := candidate_label(c)) and label_key(lab) in others)]
    eligible = {c.id for c in mine}
    wanted = {v for v in (normalize_ws(x) for x in values) if v}
    annotated = [a for a in annotations if a.attr is None and a.cid in eligible]
    shown = sorted({a.cid for a in annotated if any(text_matches(t, wanted) for t in a.texts)})
    infos = [_cand_info(c) for c in at_path]
    if shown:
        return FieldCoverage(name, located, covered=True, candidates=infos, shown=shown)
    if not annotated:
        listed = ", ".join(
            f"{c.id} {c.cls} p={c.p:.3f} on {c.pages_present} pages label {candidate_label(c)!r}"
            for c in mine
        )
        return FieldCoverage(
            name,
            located,
            covered=False,
            reason="not_shown",
            detail="no candidate of the field at its path is annotated on the union page"
            + (f" ({listed})" if listed else "; every candidate there is labeled as another field"),
            candidates=infos,
        )
    seen = "; ".join(f"{a.cid}: {' | '.join(a.texts)[:60]!r}" for a in annotated[:3])
    return FieldCoverage(
        name,
        located,
        covered=False,
        reason="value_mismatch",
        detail=f"annotated candidates at the path show no value of the field ({seen})",
        candidates=infos,
    )


@dataclass(frozen=True)
class KnownField:
    """A field whose values are known per page (from ground truth)."""

    name: str
    values: Mapping[str, Sequence[str]]
    """Page key -> the field's values on that page (pages where it is present)."""
    other_labels: tuple[str, ...] = ()
    """Labels of the template's other fields."""


def template_coverage(
    fields: Sequence[KnownField],
    table: CandidateTable,
    union_html: str,
    page_items: Callable[[str], TextItems],
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> list[FieldCoverage]:
    """Coverage of each known field by one template's union page. ``page_items(key)`` returns a
    page's text items; each field is located on its first ``max_pages`` present pages (sorted
    keys)."""
    annotations = annotated_nodes(union_html)
    out: list[FieldCoverage] = []
    for known in fields:
        keys = sorted(k for k, v in known.values.items() if v)[:max_pages]
        located = field_path((page_items(k), known.values[k]) for k in keys)
        everywhere = {v for vs in known.values.values() for v in vs}
        out.append(
            field_coverage(
                known.name,
                located,
                table,
                annotations,
                everywhere,
                other_labels=known.other_labels,
            )
        )
    return out
