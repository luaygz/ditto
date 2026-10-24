"""Union page compression and rendering: collapse repeated siblings by variant, drop
candidate-free subtrees to fit a token budget, and render the annotated HTML."""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from lxml import etree

from ditto.analyze.models import Candidate
from ditto.fingerprint.dom import node_text
from ditto.textutil import normalize_ws
from ditto.union.prepare import (
    X_C,
    X_COLLAPSED,
    X_KV,
    X_LABEL,
    X_PREFIX,
    X_SPLICED,
    X_T,
    describe,
    is_keep_root,
    keep_flags,
    marks,
    remove_element,
    truncate_middle,
)
from ditto.union.tokens import count_tokens
from ditto.union.tree import Element, child_elements, iter_elements, label_of

EXAMPLE_LIMIT = 80
BLOCK_TAGS = frozenset(
    [
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hr",
        "html",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "title",
        "tr",
        "ul",
    ]
)
_MARK_ATTR = re.compile(r'\s(?:data-x-[\w-]+)="[^"]*"')


@dataclass(frozen=True)
class CollapseLevel:
    """How hard repeated sibling groups are collapsed."""

    min_group: int  # collapse groups of at least this many same-label siblings
    coarse: bool  # variant = direct-child labels (True) or descendant structure (False)
    max_variants: int  # variants kept; members of others join the most similar kept variant
    keep_last: bool  # keep the last sibling as an example too


COLLAPSE_LEVELS: tuple[CollapseLevel, ...] = (
    CollapseLevel(4, False, 8, True),
    CollapseLevel(3, True, 4, True),
    CollapseLevel(3, True, 2, False),
    CollapseLevel(2, True, 1, False),
)


# -- collapse ----------------------------------------------------------------------------------


def collapse(root: Element, level: CollapseLevel) -> int:
    """Collapse repeated siblings in place; returns the number of groups collapsed.

    Siblings are grouped by structural step label. Inside a group, members are split into
    variants by child signature; the first member, the last (when ``keep_last``), the first
    member of each kept variant, members holding spliced content and members that show a
    candidate no kept member shows are kept. Key-value containers (``data-x-kv``) and parents
    whose groups carry distinct label texts are never collapsed: each repetition there is a
    different field."""
    collapsed = 0
    stack = [root]
    while stack:
        parent = stack.pop()
        kids = child_elements(parent)
        if len(kids) >= level.min_group and parent.get(X_KV) is None:
            groups: dict[str, list[Element]] = {}
            for kid in kids:
                groups.setdefault(label_of(kid), []).append(kid)
            if not _kv_like(groups.values()):
                for members in groups.values():
                    if len(members) >= level.min_group and _collapse_group(members, level):
                        collapsed += 1
                kids = child_elements(parent)
        stack.extend(kids)
    return collapsed


def _label_text(el: Element) -> str:
    texts = [node_text(n) for n in iter_elements(el) if n.get(X_LABEL) is not None]
    return " | ".join(t for t in texts[:5] if t)


def _kv_like(groups: Iterable[list[Element]]) -> bool:
    """A sibling group whose members carry different label texts is a key-value list."""
    for members in groups:
        if len(members) < 2:
            continue
        texts = [t for t in (_label_text(m) for m in members) if t]
        if len(texts) >= 2 and len(set(texts)) >= max(2, (len(texts) + 1) // 2):
            return True
    return False


def _fine_signature(el: Element, depth: int = 4) -> frozenset[str]:
    out: set[str] = set()
    stack: list[tuple[Element, str, int]] = [(el, "", 0)]
    while stack:
        node, rel, d = stack.pop()
        if d >= depth:
            continue
        for child in child_elements(node):
            path = rel + "/" + label_of(child)
            out.add(path)
            stack.append((child, path, d + 1))
    return frozenset(out)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _ids_in(el: Element) -> set[str]:
    out: set[str] = set()
    for n in iter_elements(el):
        out.update(marks(n, X_C))
    return out


def _has_spliced(el: Element) -> bool:
    return any(n.get(X_SPLICED) is not None for n in iter_elements(el))


def _collapse_group(members: list[Element], level: CollapseLevel) -> bool:
    fine = [_fine_signature(m) for m in members]
    if level.coarse:
        sigs = [frozenset(label_of(c) for c in child_elements(m)) for m in members]
    else:
        sigs = fine
    first_of: dict[frozenset[str], int] = {}
    size: Counter[frozenset[str]] = Counter()
    for i, sig in enumerate(sigs):
        first_of.setdefault(sig, i)
        size[sig] += 1
    ranked = sorted(first_of, key=lambda s: (-size[s], first_of[s]))
    kept_variants = ranked[: max(1, level.max_variants)]
    kept_set = set(kept_variants)
    assign: list[frozenset[str]] = []
    for i, sig in enumerate(sigs):
        if sig in kept_set:
            assign.append(sig)
        else:
            assign.append(
                max(
                    kept_variants,
                    key=lambda v: (_jaccard(fine[i], fine[first_of[v]]), -first_of[v]),
                )
            )
    keep = {0, *(first_of[v] for v in kept_variants)}
    if level.keep_last:
        keep.add(len(members) - 1)
    covered: set[str] = set()
    for i in keep:
        covered |= _ids_in(members[i])
    for i, m in enumerate(members):
        if i in keep:
            continue
        ids = _ids_in(m)
        if _has_spliced(m) or ids - covered:
            keep.add(i)
            covered |= ids
    if len(keep) == len(members):
        return False
    numbering = {v: n + 1 for n, v in enumerate(sorted(kept_variants, key=lambda v: first_of[v]))}
    removed: Counter[frozenset[str]] = Counter(
        assign[i] for i in range(len(members)) if i not in keep
    )
    for variant, count in removed.items():
        members[first_of[variant]].set(
            X_COLLAPSED, f"{count} {numbering[variant]} {len(kept_variants)}"
        )
    for i, m in enumerate(members):
        if i not in keep:
            remove_element(m)
    return True


# -- dropping to budget ------------------------------------------------------------------------


def droppables(root: Element) -> list[Element]:
    """Maximal subtrees holding no candidate and no kept label."""
    flags = keep_flags(root)
    out: list[Element] = []
    stack = [root]
    while stack:
        el = stack.pop()
        for child in child_elements(el):
            if is_keep_root(child):
                continue
            if flags.get(child, False):
                stack.append(child)
            else:
                out.append(child)
    return out


def estimate_tokens(el: Element) -> int:
    raw = etree.tostring(el, method="html", encoding="unicode", with_tail=False)
    return count_tokens(_MARK_ATTR.sub("", raw))


def drop_to_fit(
    root: Element,
    max_tokens: int,
    render_fn: Callable[[Element], str],
    html: str,
    tokens: int,
) -> tuple[str, int, list[str]]:
    """Remove candidate-free subtrees until the rendered page fits ``max_tokens``: the
    smallest subtree that covers the remaining excess, else the largest. Returns the rendered
    page, its token count and a description of each dropped subtree."""
    dropped: list[str] = []
    while tokens > max_tokens:
        pool = [(estimate_tokens(el), n, el) for n, el in enumerate(droppables(root))]
        if not pool:
            break
        need = tokens - max_tokens
        while need > 0 and pool:
            covering = [p for p in pool if p[0] >= need]
            if covering:
                pick = min(covering, key=lambda p: (p[0], p[1]))
            else:
                pick = max(pool, key=lambda p: (p[0], -p[1]))
            pool.remove(pick)
            dropped.append(f"{describe(pick[2])} (~{pick[0]} tokens)")
            remove_element(pick[2])
            need -= pick[0]
        html = render_fn(root)
        tokens = count_tokens(html)
    return html, tokens, dropped


def unmark_candidates(root: Element, ids: set[str]) -> None:
    """Forget candidates (last-resort budget step): their nodes become droppable."""
    for el in iter_elements(root):
        for name in (X_C, X_LABEL):
            current = marks(el, name)
            if current and set(current) & ids:
                rest = [c for c in current if c not in ids]
                if rest:
                    el.set(name, " ".join(rest))
                else:
                    del el.attrib[name]
                    if name == X_C and el.get(X_T) is not None:
                        del el.attrib[X_T]


def shorten_candidate_text(root: Element, limit: int) -> None:
    """Cut candidate text to ``limit`` characters too (budget step before dropping candidates)."""
    for el in iter_elements(root):
        if el.get(X_T) is None:
            continue
        if el.text:
            el.text = truncate_middle(el.text, limit)
        for child in el:
            if child.tail:
                child.tail = truncate_middle(child.tail, limit)


# -- rendering ---------------------------------------------------------------------------------


def candidate_note(cand: Candidate) -> str:
    """``c17 · optional · on 41% of pages · e.g. a.khoury@example.org``."""
    head = cand.id + (f" @{cand.attr}" if cand.attr else "")
    pct = round(cand.p * 100)
    share = "<1" if pct == 0 and cand.p > 0 else str(pct)
    parts = [head, cand.cls, f"on {share}% of pages"]
    example = next((normalize_ws(e) for e in cand.examples if normalize_ws(e)), "")
    if example:
        parts.append("e.g. " + truncate_middle(example, EXAMPLE_LIMIT))
    return " · ".join(parts)


def collapse_note(count: int, variant: int, total: int) -> str:
    noun = "sibling" if count == 1 else "siblings"
    return f"collapsed: {count} more {noun} like the one above (variant {variant} of {total})"


def _comment(text: str) -> etree._Comment:
    safe = re.sub(r"-{2,}", lambda m: " ".join(m.group(0)), text.replace("\n", " "))
    return etree.Comment(f" {safe} ")


def _region(el: Element) -> Element | None:
    """The spliced unit holding ``el`` (``None`` for the base page itself)."""
    node: Element | None = el
    while node is not None:
        if node.get(X_SPLICED) is not None:
            return node
        node = node.getparent()
    return None


def render(root: Element, notes: Mapping[str, str]) -> str:
    """Serialize a copy of ``root`` with annotation comments and without internal marks.

    Each candidate is annotated before its first node in the page; a spliced candidate also
    before its first node inside its spliced unit. Spliced units and collapsed groups get
    their marker comments."""
    out = copy.deepcopy(root)
    seen: set[tuple[Element | None, str]] = set()
    plan: list[tuple[Element, list[str], str | None]] = []
    for el in iter_elements(out):
        before: list[str] = []
        spliced = marks(el, X_SPLICED)
        if spliced:
            before.append("spliced from another page: " + ", ".join(spliced))
        cids = marks(el, X_C)
        if cids:
            region = _region(el)
            spliced_here = set(marks(region, X_SPLICED)) if region is not None else set()
            for cid in cids:
                key = (region, cid) if cid in spliced_here else (None, cid)
                if key not in seen and cid in notes:
                    seen.add(key)
                    before.append(notes[cid])
        collapsed = el.get(X_COLLAPSED)
        after = None
        if collapsed:
            count, variant, total = (int(x) for x in collapsed.split())
            after = collapse_note(count, variant, total)
        plan.append((el, before, after))
    for el, before, after in plan:
        if el.getparent() is not None:
            for text in before:
                el.addprevious(_comment(text))
            if after:
                el.addnext(_comment(after))
        for name in [a for a in el.attrib if str(a).startswith(X_PREFIX)]:
            del el.attrib[name]
        if str(el.tag) in BLOCK_TAGS and not (el.tail or "").strip():
            el.tail = "\n"
    return etree.tostring(out, method="html", encoding="unicode")
