"""Union page preparation: attribute pruning, text truncation, internal marks, focus trimming
and boilerplate removal.

Internal marks are ``data-x-*`` attributes; they survive ``copy.deepcopy`` (the budget loop
compresses fresh copies of the prepared tree) and ``render`` strips every one of them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ditto.fingerprint.dom import (
    ATTR_SEP,
    LINK_ATTRS,
    canonical_classes,
    keep_identifier,
    node_text,
)
from ditto.fingerprint.dom import own_text as dom_own_text
from ditto.textutil import normalize_ws
from ditto.union.tree import (
    CandIndex,
    Element,
    Tree,
    ancestor_at,
    child_elements,
    find_label,
    iter_elements,
    label_of,
    preceding_elements,
)

X_PREFIX = "data-x-"
X_C = "data-x-c"  # field candidate ids carried by the node (space-separated)
X_T = "data-x-t"  # the node carries a text (not attribute) candidate: its text keeps 300 chars
X_LABEL = "data-x-label"  # label kept for the candidate ids listed
X_SPLICED = "data-x-spliced"  # top of a unit spliced from a donor page, with the candidate ids
X_KV = "data-x-kv"  # key-value container: its children never collapse
X_VAR = "data-x-var"  # carries page-specific (non-boilerplate) text
X_COLLAPSED = "data-x-collapsed"  # "<removed count> <variant number> <variant total>"

TEXT_LIMIT = 120
CANDIDATE_TEXT_LIMIT = 300
LINK_LIMIT = 80
OTHER_ATTRS = ("itemprop", "data-testid", "role", "aria-label", "type")
# Table geometry: column indexes (``col5``, header-by-position) count colspan, as the analysis does
SPAN_ATTRS = ("colspan", "rowspan")
INVARIANT_CLASSES = frozenset({"boilerplate", "label"})
NEIGHBOR_MAX = 10  # focus: an adjacent sibling joins the neighborhood when this small (nodes)
SHORT_LABEL_MAX = 80  # adjacent invariant heading/label kept with a candidate: max characters

LANDMARK_TAGS = frozenset({"nav", "footer", "header", "aside", "form", "dialog", "menu"})
LANDMARK_ROLES = frozenset(
    {
        "navigation",
        "banner",
        "contentinfo",
        "complementary",
        "search",
        "dialog",
        "alertdialog",
        "menu",
        "menubar",
    }
)
_BOILER_WORD = re.compile(
    r"(?:^|[-_])(?:nav|navbar|navigation|menu|menubar|footer|header|masthead|cookie|cookies|"
    r"consent|gdpr|banner|breadcrumb|breadcrumbs|sidebar|social|share|sharing|newsletter|"
    r"subscribe|advert|advertisement|ads|promo|popup|modal|overlay|skip)(?:$|[-_])",
    re.IGNORECASE,
)


def truncate_middle(text: str, limit: int) -> str:
    """Cut ``text`` to about ``limit`` characters by removing its middle (`` … `` marks it)."""
    if len(text) <= limit:
        return text
    keep = max(limit - 3, 2)
    head = text[: (keep + 1) // 2].rstrip()
    tail = text[len(text) - keep // 2 :].lstrip()
    return f"{head} … {tail}"


def add_mark(el: Element, name: str, values: Iterable[str]) -> None:
    current = (el.get(name) or "").split()
    for v in values:
        if v not in current:
            current.append(v)
    el.set(name, " ".join(current))


def marks(el: Element, name: str) -> list[str]:
    return (el.get(name) or "").split()


def remove_element(el: Element) -> None:
    """Detach ``el``; its tail text (part of the parent's own text) stays in the document."""
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail
    if tail and tail.strip():
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(el)


# -- attributes and text ---------------------------------------------------------------------


def prune_attrs(root: Element) -> None:
    """Keep only stable ``id``, canonical ``class``, ``href``/``src`` (80 chars), ``itemprop``,
    stable ``data-testid``, ``role``, ``aria-label``, ``type`` and table spans other than 1.
    Drops any ``data-x-*``."""
    for el in iter_elements(root):
        attrib = el.attrib
        if not attrib:
            continue
        kept: list[tuple[str, str]] = []
        ident = keep_identifier(attrib.get("id"))
        if ident:
            kept.append(("id", ident))
        classes = canonical_classes(attrib.get("class"))
        if classes:
            kept.append(("class", " ".join(classes)))
        for name in LINK_ATTRS:
            value = attrib.get(name)
            if value is not None:
                value = value.strip()
                if len(value) > LINK_LIMIT:
                    # ASCII marker: the HTML serializer percent-escapes non-ASCII in URLs
                    value = value[: LINK_LIMIT - 3] + "..."
                kept.append((name, value))
        for name in OTHER_ATTRS:
            value = attrib.get(name)
            if name == "data-testid":
                value = keep_identifier(value)
            if value:
                kept.append((name, truncate_middle(normalize_ws(value), TEXT_LIMIT)))
        for name in SPAN_ATTRS:
            value = (attrib.get(name) or "").strip()
            if value and value != "1":
                kept.append((name, value))
        attrib.clear()
        for name, value in kept:
            attrib[name] = value


def _norm_text(text: str | None, limit: int) -> str | None:
    if not text:
        return None
    core = normalize_ws(text)
    if not core:
        return None
    core = truncate_middle(core, limit)
    lead = " " if text[0].isspace() else ""
    trail = " " if text[-1].isspace() else ""
    return lead + core + trail


def normalize_text(root: Element) -> None:
    """Collapse whitespace; truncate text over 120 chars from the middle (candidate text: 300).
    A node's text and its children's tails are its own text, so they share its limit."""
    for el in iter_elements(root):
        limit = CANDIDATE_TEXT_LIMIT if el.get(X_T) is not None else TEXT_LIMIT
        el.text = _norm_text(el.text, limit)
        for child in el:
            child.tail = _norm_text(child.tail, limit)


# -- marks -------------------------------------------------------------------------------------


def mark_tree(
    tree: Tree, index: CandIndex, located: dict[str, list[Element]]
) -> dict[Element, list[Element]]:
    """Set the internal marks on a (pruned) base tree. Returns candidate node -> its labels."""
    root = tree.root
    for cid, els in located.items():
        text_cand = index.fields[cid].attr is None and ATTR_SEP not in index.fields[cid].path
        for el in els:
            add_mark(el, X_C, [cid])
            if text_cand:
                el.set(X_T, "1")

    # page-specific text vs boilerplate/label text, by the analysis' path classes
    invariant: set[Element] = set()
    for el in iter_elements(root):
        path = tree.path(el)
        if dom_own_text(el):
            if index.path_class.get(path) in INVARIANT_CLASSES:
                invariant.add(el)
            else:
                el.set(X_VAR, "1")
        for attr in LINK_ATTRS:
            value = el.get(attr)
            if value and value.strip():
                cls = index.path_class.get(path + ATTR_SEP + attr)
                if cls not in INVARIANT_CLASSES:
                    el.set(X_VAR, "1")

    # labels named by label_hint: preceding sibling or parent's first child, up to 2 levels up
    labels_of: dict[Element, list[Element]] = {}
    for cid, els in located.items():
        hint = index.fields[cid].label_hint
        if not hint:
            continue
        for el in els:
            for node in up_from(el, 3):
                label = find_label(node, hint)
                if label is not None:
                    add_mark(label, X_LABEL, [cid])
                    labels_of.setdefault(el, []).append(label)
                    break

    by_path = tree.by_path()
    for t in index.table.tables:
        ids = [c.id for c in index.fields.values() if c.table_id == t.id] or [t.id]
        if t.kind == "key_value":
            key_path = index.key_paths.get(t.id)
            for el in by_path.get(key_path or "", []):
                add_mark(el, X_LABEL, ids)
            for el in by_path.get(index.container_paths[t.id], []):
                el.set(X_KV, "1")
            row_path = t.row_path.split(ATTR_SEP)[0]
            for el in by_path.get(row_path, []):
                parent = el.getparent()
                if parent is not None:
                    parent.set(X_KV, "1")

    # header cells of tables that hold candidates (columns are selected by header text)
    tables: dict[Element, set[str]] = {}
    for cid, els in located.items():
        for el in els:
            table_el = next((n for n in up_from(el, 64) if str(n.tag) == "table"), None)
            if table_el is not None:
                tables.setdefault(table_el, set()).add(cid)
    for table_el, ids_in in tables.items():
        for th in table_el.iter("th"):
            add_mark(th, X_LABEL, sorted(ids_in))

    # a short invariant sibling right before a candidate or its parent/grandparent: a heading
    # or label for it (never landmark chrome such as a nav bar)
    near: dict[Element, set[str]] = {}
    for cid, els in located.items():
        for el in els:
            for node in up_from(el, 3):
                near.setdefault(node, set()).add(cid)
    for el, ids_near in near.items():
        for n, sib in enumerate(preceding_elements(el)):
            if n >= 2:
                break
            if sib not in near and _short_invariant(sib, invariant):
                add_mark(sib, X_LABEL, sorted(ids_near))
    return labels_of


def _short_invariant(el: Element, invariant: set[Element]) -> bool:
    """One or two elements of invariant text, at most 80 characters, not landmark chrome."""
    nodes = list(iter_elements(el))
    if len(nodes) > 2 or any(n.get(X_VAR) is not None or is_landmark(n) for n in nodes):
        return False
    if not any(n in invariant for n in nodes):
        return False
    return len(node_text(el)) <= SHORT_LABEL_MAX


# -- trimming ----------------------------------------------------------------------------------


def is_keep_root(el: Element) -> bool:
    return bool(el.get(X_C)) or bool(el.get(X_LABEL))


def keep_flags(root: Element) -> dict[Element, bool]:
    """Element -> whether its subtree holds a candidate or a kept label."""
    flags: dict[Element, bool] = {}
    for el in reversed(list(iter_elements(root))):
        flags[el] = is_keep_root(el) or any(flags.get(c, False) for c in child_elements(el))
    return flags


def is_landmark(el: Element) -> bool:
    if str(el.tag) in LANDMARK_TAGS:
        return True
    if (el.get("role") or "").strip().lower() in LANDMARK_ROLES:
        return True
    tokens = (el.get("id") or "").split() + (el.get("class") or "").split()
    return any(_BOILER_WORD.search(tok) for tok in tokens)


def is_boilerplate(el: Element) -> bool:
    """Landmark chrome (nav, footer, cookie banner, ...) or a subtree with no page-specific
    text at all."""
    if is_landmark(el):
        return True
    return not any(n.get(X_VAR) is not None for n in iter_elements(el))


def remove_boilerplate(root: Element) -> list[str]:
    """Remove candidate-free boilerplate subtrees; returns a description of each removal."""
    flags = keep_flags(root)
    removed: list[str] = []
    stack = [root]
    while stack:
        el = stack.pop()
        for child in child_elements(el):
            if is_keep_root(child):
                continue
            if not flags.get(child, False) and is_boilerplate(child):
                removed.append(describe(child))
                remove_element(child)
            else:
                stack.append(child)
    return removed


def trim_to_focus(
    tree: Tree,
    index: CandIndex,
    located: dict[str, list[Element]],
    labels_of: dict[Element, list[Element]],
    focus: set[str],
) -> int:
    """Keep only the focused candidates' neighborhoods plus labels and ancestors; returns the
    number of subtrees removed. A neighborhood is the candidate's table container when it has
    one, else the node with its nearest preceding and following sibling (when small)."""
    roots: list[Element] = []
    for cid in sorted(focus):
        cand = index.fields.get(cid)
        if cand is None:
            continue
        table = index.tables.get(cand.table_id) if cand.table_id else None
        for el in located.get(cid, []):
            if table is not None:
                container = ancestor_at(tree, el, index.container_paths[table.id])
                if container is not None:
                    roots.append(container)
                    roots.extend(
                        s for s in list(preceding_elements(container))[:2] if s.get(X_LABEL)
                    )
                    continue
            roots.append(el)
            roots.extend(labels_of.get(el, []))
            before = next(preceding_elements(el), None)
            after = next((s for s in el.itersiblings() if isinstance(s.tag, str)), None)
            roots.extend(s for s in (before, after) if s is not None and _at_most(s, NEIGHBOR_MAX))
    inside: set[Element] = set()
    chain: set[Element] = set()
    for r in roots:
        inside.update(iter_elements(r))
        node = r.getparent()
        while node is not None and node not in chain:
            chain.add(node)
            node = node.getparent()
    if tree.root in inside:
        return 0
    removed = 0
    stack = [tree.root]
    while stack:
        el = stack.pop()
        for child in child_elements(el):
            if child in inside:
                continue
            if child in chain:
                stack.append(child)
            else:
                remove_element(child)
                removed += 1
    tree.invalidate()
    return removed


def _at_most(el: Element, limit: int) -> bool:
    return all(n < limit for n, _ in enumerate(iter_elements(el)))


def describe(el: Element) -> str:
    """Short structural description of a node for logs: its last three path steps."""
    steps = [label_of(n) for n in up_from(el, 3)]
    top = up_from(el, 4)
    prefix = "… > " if len(top) > 3 else ""
    return prefix + " > ".join(reversed(steps))


def up_from(el: Element, n: int) -> list[Element]:
    """``el`` and its ancestors, at most ``n`` nodes."""
    out: list[Element] = []
    node: Element | None = el
    while node is not None and len(out) < n:
        out.append(node)
        node = node.getparent()
    return out
