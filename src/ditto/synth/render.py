"""HTML for synthetic pages: a tiny node tree, a serializer with whitespace modes and entity
noise, the site shell (header/nav/footer variants) and per-template content renderers.

Nodes are ``(tag, attribute_string, children)`` tuples; children are nodes or raw text. Text is
escaped at serialization time so noise (entities, NBSP) never changes the visible text.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape
from html.entities import codepoint2name
from typing import Literal, Union

from ditto.synth.blueprint import KVDef, ScalarDef, TableDef, TemplateDef, hashed_class
from ditto.synth.page import PageData
from ditto.synth.params import MarkupStyle
from ditto.synth.vocab import FOOTER_HEADINGS, FOOTER_LINKS, NAV_WORDS

Child = Union["Node", str]
Node = tuple[str, str, list[Child]]
WsMode = Literal["pretty", "minified", "loose"]
WS_MODES: tuple[WsMode, ...] = ("pretty", "minified", "loose")
Escaper = Callable[[str], str]

VOID = frozenset({"meta", "link", "input", "br", "img", "hr"})
RAW = frozenset({"script", "style"})
CONTENT = "\x00content"
_MARK = "\x00MARK\x00"
_INDENT: dict[str, str] = {"pretty": "  ", "loose": "    ", "minified": ""}
SHELL_FEATURES = ("search", "account", "announcement", "newsletter", "cookie")


def attr(name: str, value: str) -> str:
    return f' {name}="{escape(value, quote=True)}"'


def el(tag: str, cls: str = "", children: list[Child] | None = None, attrs: str = "") -> Node:
    # class values are generated here and never need escaping
    return (
        tag,
        (f' class="{cls}"' if cls else "") + attrs,
        children if children is not None else [],
    )


# --- escaping ---------------------------------------------------------------------------------


def esc_plain(text: str) -> str:
    if "&" in text or "<" in text or ">" in text:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if "\xa0" in text:
        text = text.replace("\xa0", "&nbsp;")
    return text


def entity_escaper(rng: random.Random) -> Escaper:
    """Escape with randomly mixed entity spellings (named, decimal, hex, raw). The decoded text
    is identical to the input."""

    def esc(text: str) -> str:
        out: list[str] = []
        for ch in text:
            o = ord(ch)
            if ch == "&":
                out.append(rng.choice(("&amp;", "&#38;", "&#x26;")))
            elif ch == "<":
                out.append(rng.choice(("&lt;", "&#60;")))
            elif ch == ">":
                out.append(rng.choice(("&gt;", "&#62;")))
            elif ch == "\xa0":
                out.append(rng.choice(("&nbsp;", "&#160;", "&#xa0;", "\xa0")))
            elif ch == "'":
                out.append(rng.choice(("'", "&#39;", "&#x27;")))
            elif ch == '"':
                out.append(rng.choice(('"', "&quot;", "&#34;")))
            elif o > 127:
                name = codepoint2name.get(o)
                options = [ch, f"&#{o};", f"&#x{o:x};"] + ([f"&{name};"] if name else [])
                out.append(rng.choice(options))
            elif ch.isalpha() and rng.random() < 0.03:
                out.append(f"&#{o};")
            else:
                out.append(ch)
        return "".join(out)

    return esc


# --- serializer -------------------------------------------------------------------------------


class Writer:
    """Serialize nodes. ``pretty`` indents element-only children; ``loose`` also puts text on
    its own indented lines; ``minified`` emits no inter-tag whitespace. Elements with any text
    child are always written inline so the visible text of every field stays exact."""

    __slots__ = ("count", "esc", "indent", "mark_depth", "mode", "out")

    def __init__(self, mode: WsMode, esc: Escaper = esc_plain) -> None:
        self.mode = mode
        self.esc = esc
        self.indent = _INDENT[mode]
        self.out: list[str] = []
        self.count = 0  # elements written (script/style excluded)
        self.mark_depth = 0

    def write(self, node: Node, depth: int) -> None:
        tag, attrs, children = node
        out = self.out
        if tag == CONTENT:
            self.mark_depth = depth
            out.append(_MARK)
            return
        if tag in RAW:
            out.append(
                f"<{tag}{attrs}>{''.join(c for c in children if isinstance(c, str))}</{tag}>"
            )
            return
        self.count += 1
        if tag in VOID:
            out.append(f"<{tag}{attrs}>")
            return
        if not children:
            out.append(f"<{tag}{attrs}></{tag}>")
            return
        first = children[0]
        if len(children) == 1 and isinstance(first, str):  # text leaf: the common case
            if self.mode == "loose":
                pad = self.indent * depth
                out.append(f"<{tag}{attrs}>\n{pad}{self.indent}{self.esc(first)}\n{pad}</{tag}>")
            else:
                out.append(f"<{tag}{attrs}>{self.esc(first)}</{tag}>")
            return
        inline = self.mode == "minified"
        if not inline:
            for c in children:
                if isinstance(c, str):
                    inline = True
                    break
        if inline:
            out.append(f"<{tag}{attrs}>")
            for c in children:
                if isinstance(c, str):
                    out.append(self.esc(c))
                else:
                    self.write_inline(c)
            out.append(f"</{tag}>")
            return
        out.append(f"<{tag}{attrs}>")
        nl = "\n" + self.indent * (depth + 1)
        for c in children:
            out.append(nl)
            if not isinstance(c, str):  # always true here: no text children on this branch
                self.write(c, depth + 1)
        out.append("\n" + self.indent * depth + f"</{tag}>")

    def write_inline(self, node: Node) -> None:
        tag, attrs, children = node
        if tag == CONTENT:
            self.out.append(_MARK)
            return
        if tag in RAW:
            self.out.append(
                f"<{tag}{attrs}>{''.join(c for c in children if isinstance(c, str))}</{tag}>"
            )
            return
        self.count += 1
        if tag in VOID:
            self.out.append(f"<{tag}{attrs}>")
            return
        self.out.append(f"<{tag}{attrs}>")
        for c in children:
            if isinstance(c, str):
                self.out.append(self.esc(c))
            else:
                self.write_inline(c)
        self.out.append(f"</{tag}>")

    def text(self) -> str:
        return "".join(self.out)


def serialize(
    node: Node, mode: WsMode = "pretty", depth: int = 0, esc: Escaper = esc_plain
) -> tuple[str, int]:
    w = Writer(mode, esc)
    w.write(node, depth)
    return w.text(), w.count


# --- site shell -------------------------------------------------------------------------------


@dataclass
class Chrome:
    """Site-wide boilerplate: name, navigation and footer sizes, class names."""

    style: MarkupStyle
    site_name: str
    seed: int
    hashed_share: float
    n_nav: int = 4
    footer: list[tuple[str, list[str]]] = field(default_factory=list)
    _classes: dict[str, str] = field(default_factory=dict)

    def cls(self, slot: str) -> str:
        c = self._classes.get(slot)
        if c is None:
            rng = random.Random(f"{self.seed}|site-class|{slot}")
            if rng.random() < self.hashed_share:
                c = hashed_class(rng, self.style, slot)
            else:
                c = "site-" + slot.replace("_", "-")
            self._classes[slot] = c
        return c


def _nav_words(n: int) -> list[str]:
    words = list(NAV_WORDS)
    extra = [f"{a} {b.lower()}" for a in NAV_WORDS for b in ("deals", "guides", "new", "top picks")]
    return (words + extra)[:n]


def _footer_links(n: int, start: int) -> list[str]:
    pool = list(FOOTER_LINKS) + [f"{a} {b}" for a in FOOTER_LINKS for b in ("help", "info")]
    return [pool[(start + i) % len(pool)] for i in range(n)]


def size_chrome(ch: Chrome, target_elements: int) -> None:
    """Choose nav and footer link counts so the variant-0 shell has about ``target_elements``
    body elements (at least one nav link and one footer column so the path set is stable)."""
    ch.n_nav, ch.footer = 2, [(FOOTER_HEADINGS[0], _footer_links(2, 0))]
    need = max(0, target_elements - shell_elements(ch, 0))
    nav_cost = 1 if ch.style == "react" else 2  # bare <a>, or <li><a>
    link_cost = 2 if ch.style == "server" else 1  # footer <li><a>, or bare <a>
    col_cost = 2 if ch.style == "table" else 3  # column wrapper + heading (+ list)
    nav = min(10, need // nav_cost // 3)
    ch.n_nav += nav
    need -= nav * nav_cost
    cols = [2]
    while need > 0:
        if cols[-1] >= 8:
            cols.append(1)  # every column carries at least one link
            need -= col_cost + link_cost
        else:
            cols[-1] += 1
            need -= link_cost
    ch.footer = []
    used = 0
    for i, n in enumerate(cols):
        heading = FOOTER_HEADINGS[i % len(FOOTER_HEADINGS)]
        if i >= len(FOOTER_HEADINGS):
            heading = f"{heading} {i // len(FOOTER_HEADINGS) + 1}"
        ch.footer.append((heading, _footer_links(n, used)))
        used += n


def shell_elements(ch: Chrome, variant: int) -> int:
    _, n = serialize(shell_tree(ch, variant))
    return n


def shell_tree(ch: Chrome, variant: int) -> Node:
    """``<body>`` with the content placeholder; variant ``k >= 1`` adds shell feature ``k-1``."""
    feats = {SHELL_FEATURES[variant - 1]} if variant else set()
    C = ch.cls
    content: Node = (CONTENT, "", [])
    nav_words = _nav_words(ch.n_nav)
    script = el("script", "", [], attr("src", "/assets/app.js") + " defer")
    if ch.style == "table":
        return _table_shell(ch, feats, nav_words, content, script)
    react = ch.style == "react"
    d = "div" if react else None

    def link(text: str, href: str, cls: str = "") -> Node:
        return el("a", cls, [text], attr("href", href))

    def slug(text: str) -> str:
        return "/" + "-".join(text.lower().replace("&", "and").split())

    header_kids: list[Child] = [link(ch.site_name, "/", C("logo"))]
    if react:
        header_kids.append(el("div", C("nav"), [link(w, slug(w)) for w in nav_words]))
    else:
        items: list[Child] = [el("li", C("nav_item"), [link(w, slug(w))]) for w in nav_words]
        header_kids.append(
            el("nav", C("nav"), [el("ul", C("nav_list"), items)], attr("aria-label", "Main"))
        )
    if "search" in feats:
        header_kids.append(
            el(
                d or "form",
                C("search"),
                [
                    el(
                        "input",
                        "",
                        None,
                        attr("type", "search") + attr("name", "q") + attr("placeholder", "Search"),
                    ),
                    el("button", "", ["Search"], attr("type", "submit")),
                ],
                "" if react else attr("action", "/search") + attr("role", "search"),
            )
        )
    if "account" in feats:
        header_kids.append(
            el(
                d or "div",
                C("account"),
                [link("Sign in", "/login"), " ", link("Create account", "/register")],
            )
        )
    body: list[Child] = []
    if "announcement" in feats:
        body.append(
            el(
                "div",
                C("announcement"),
                [
                    el(
                        "p",
                        "",
                        ["Free shipping on orders over $50. ", link("Learn more", "/shipping")],
                    )
                ],
            )
        )
    body.append(el(d or "header", C("header"), [el("div", C("header_inner"), header_kids)]))
    body.append(el(d or "main", C("main"), [content], "" if react else attr("id", "main")))
    if "cookie" in feats:
        body.append(
            el(
                "div",
                C("cookie"),
                [
                    el("p", "", ["We use cookies to improve your experience."]),
                    el("button", "", ["Accept"], attr("type", "button")),
                ],
                attr("role", "dialog"),
            )
        )
    cols: list[Child] = []
    for heading, links in ch.footer:
        if react:
            cols.append(
                el(
                    "div",
                    C("footer_col"),
                    [
                        el("div", C("footer_h"), [heading]),
                        el("div", C("footer_list"), [link(x, slug(x)) for x in links]),
                    ],
                )
            )
        else:
            cols.append(
                el(
                    "div",
                    C("footer_col"),
                    [
                        el("h4", C("footer_h"), [heading]),
                        el(
                            "ul",
                            C("footer_list"),
                            [el("li", "", [link(x, slug(x))]) for x in links],
                        ),
                    ],
                )
            )
    foot: list[Child] = [el("div", C("footer_cols"), cols)]
    if "newsletter" in feats:
        foot.append(
            el(
                d or "form",
                C("newsletter"),
                [
                    el("label", "", ["Subscribe to our newsletter"]),
                    el("input", "", None, attr("type", "email") + attr("name", "email")),
                    el("button", "", ["Sign up"], attr("type", "submit")),
                ],
            )
        )
    foot.append(el("p", C("copyright"), [f"© 2026 {ch.site_name}. All rights reserved."]))
    body.append(el(d or "footer", C("footer"), foot))
    body.append(script)
    if react:
        return el(
            "body",
            "",
            [el("div", "", [el("div", C("app"), body[:-1])], attr("id", "root")), script],
        )
    return el("body", C("body"), body)


def _table_shell(
    ch: Chrome, feats: set[str], nav_words: list[str], content: Node, script: Node
) -> Node:
    C = ch.cls

    def link(text: str, href: str) -> Node:
        return el("a", "", [text], attr("href", href))

    top: list[Child] = [el("a", C("logo"), [ch.site_name], attr("href", "/"))]
    if "account" in feats:
        top.append(
            el(
                "span",
                C("account"),
                [link("Sign in", "/login"), " | ", link("Register", "/register")],
            )
        )
    if "announcement" in feats:
        top.append(
            el(
                "div",
                C("announcement"),
                [el("b", "", ["Sale!"]), " ", link("Up to 50% off", "/sale")],
            )
        )
    side: list[Child] = [
        el(
            "ul",
            C("nav_list"),
            [el("li", "", [link(w, "/" + w.lower().replace(" ", "-"))]) for w in nav_words],
        )
    ]
    if "search" in feats:
        side.append(
            el(
                "form",
                C("search"),
                [
                    el("input", "", None, attr("type", "text") + attr("name", "q")),
                    el("input", "", None, attr("type", "submit") + attr("value", "Go")),
                ],
                attr("action", "/search.php"),
            )
        )
    foot: list[Child] = []
    for heading, links in ch.footer:
        kids: list[Child] = [el("b", "", [heading]), " "]
        for i, x in enumerate(links):
            if i:
                kids.append(" | ")
            kids.append(link(x, "/" + x.lower().replace(" ", "-") + ".html"))
        foot.append(el("p", C("footer_col"), kids))
    if "newsletter" in feats:
        foot.append(
            el(
                "form",
                C("newsletter"),
                [
                    el("input", "", None, attr("type", "text") + attr("name", "email")),
                    el("input", "", None, attr("type", "submit") + attr("value", "Subscribe")),
                ],
            )
        )
    if "cookie" in feats:
        foot.append(el("div", C("cookie"), [el("p", "", ["This site uses cookies."])]))
    foot.append(el("p", C("copyright"), [f"Copyright © 2026 {ch.site_name}"]))
    table = el(
        "table",
        C("layout"),
        [
            el("tr", "", [el("td", C("masthead"), top, attr("colspan", "2"))]),
            el(
                "tr",
                "",
                [
                    el("td", C("navcol"), side, attr("valign", "top")),
                    el("td", C("maincol"), [content], attr("valign", "top")),
                ],
            ),
            el("tr", "", [el("td", C("footer"), foot, attr("colspan", "2"))]),
        ],
        attr("width", "100%"),
    )
    return el("body", C("body"), [table, script])


@dataclass
class ShellText:
    pre: str
    post: str
    depth: int
    elements: int


def shell_text(ch: Chrome, variant: int, mode: WsMode) -> ShellText:
    w = Writer(mode)
    w.write(shell_tree(ch, variant), 1)
    text = w.text()
    pre, post = text.split(_MARK)
    return ShellText(pre, post, w.mark_depth, w.count)


def head_text(title: str, url: str, charset: str, mode: WsMode, esc: Escaper) -> str:
    meta = (
        el("meta", "", None, attr("charset", "utf-8"))
        if charset == "utf-8"
        else el(
            "meta",
            "",
            None,
            attr("http-equiv", "Content-Type") + attr("content", f"text/html; charset={charset}"),
        )
    )
    head = el(
        "head",
        "",
        [
            meta,
            el("title", "", [title]),
            el(
                "meta",
                "",
                None,
                attr("name", "viewport") + attr("content", "width=device-width, initial-scale=1"),
            ),
            el("link", "", None, attr("rel", "canonical") + attr("href", url)),
            el("link", "", None, attr("rel", "stylesheet") + attr("href", "/assets/site.css")),
            el("script", "", ["window.dataLayer = window.dataLayer || [];"]),
        ],
    )
    w = Writer(mode, esc)
    w.write(head, 1)
    return w.text()


def assemble(head: str, shell: ShellText, content: str, mode: WsMode) -> str:
    nl = "" if mode == "minified" else "\n"
    ind = _INDENT[mode]
    return f'<!DOCTYPE html>{nl}<html lang="en">{nl}{ind}{head}{nl}{ind}{shell.pre}{content}{shell.post}{nl}</html>{nl}'


# --- content ----------------------------------------------------------------------------------

Nb = Callable[[str], str]


def _ident(text: str) -> str:
    return text


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


def _label(t: TemplateDef, s: ScalarDef) -> str:
    return s.label + (":" if t.label_colon else "")


def _value_node(t: TemplateDef, s: ScalarDef, value: str | None, tag: str, nb: Nb) -> Node:
    """The element holding a scalar's value (absent renderings: empty, ``N/A``, hidden)."""
    attrs = attr("itemprop", _camel(s.name)) if t.field_scheme == "itemprop" else ""
    is_link = s.vtype in ("email", "url")
    if value is None:
        # the template still emits its markup; only the text (and href) is missing
        kids: list[Child] = ["N/A"] if t.absent == "na" else []
        if is_link and tag == "td":
            return el("td", t.value_cls(s.name), [el("a", "", kids)], attrs)
        return el("a" if is_link else tag, t.value_cls(s.name), kids, attrs)
    if is_link:
        href = f"mailto:{value}" if s.vtype == "email" else value
        if tag == "td":
            link = el("a", "", [nb(value)], attr("href", href))
            return el("td", t.value_cls(s.name), [link], attrs)
        return el("a", t.value_cls(s.name), [nb(value)], attr("href", href) + attrs)
    return el(tag, t.value_cls(s.name), [nb(value)], attrs)


def _wrap_attrs(t: TemplateDef, s: ScalarDef, value: str | None) -> tuple[str, str]:
    cls = t.wrap_cls(s.name)
    if value is None and t.absent == "hidden":
        return (cls + " hidden").strip(), attr("aria-hidden", "true")
    return cls, ""


def _scalar_block(t: TemplateDef, s: ScalarDef, value: str | None, nb: Nb) -> Node | None:
    if value is None and t.absent == "omitted":
        return None
    react = t.markup == "react"
    wcls, wattrs = _wrap_attrs(t, s, value)
    if s.group == "para":
        return el(
            "div" if react else "section",
            wcls,
            [
                el("div" if react else "h2", t.label_cls(s.name), [s.label]),
                _value_node(t, s, value, "div" if react else "p", nb),
            ],
            wattrs,
        )
    layout = t.field_layout
    wrap_tag = {"div": "div", "p": "p", "li": "li"}.get(layout, "div")
    label_tag = "strong" if layout == "p" else "span"
    value_tag = "span" if react else ("strong" if s.tag == "strong" else "span")
    return el(
        wrap_tag,
        wcls,
        [
            el(label_tag, t.label_cls(s.name), [_label(t, s)]),
            " ",
            _value_node(t, s, value, value_tag, nb),
        ],
        wattrs,
    )


def _list_block(t: TemplateDef, name: str, d: PageData, nb: Nb) -> Node:
    lst = next(x for x in t.lists if x.name == name)
    react = t.markup == "react"
    items: list[Child] = []
    for text in d.lists[name]:
        inner: Child = (
            el("a", "", [nb(text)], attr("href", f"/{name}/{_slug(text)}"))
            if lst.link
            else nb(text)
        )
        items.append(el("div" if react else "li", t.cls(f"item:{name}"), [inner]))
    list_tag = "div" if react else ("ol" if lst.ordered else "ul")
    return el(
        "div" if react else "section",
        t.cls(f"section:{name}"),
        [
            el("div" if react else "h2", t.cls("h2"), [lst.heading]),
            el(list_tag, t.cls(f"list:{name}"), items),
        ],
    )


def _slug(text: str) -> str:
    return "-".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def _kv_node(t: TemplateDef, block: KVDef, pairs: list[tuple[str, str]], nb: Nb) -> Node:
    C = t.cls
    if block.style == "dl":
        kids: list[Child] = []
        for k, v in pairs:
            kids += [el("dt", C("kv_key"), [k]), el("dd", C("kv_val"), [nb(v)])]
        return el("dl", C(f"kv:{block.name}"), kids)
    if block.style == "table":
        rows: list[Child] = [
            el(
                "tr",
                C("kv_row"),
                [el(block.key_tag, C("kv_key"), [k]), el("td", C("kv_val"), [nb(v)])],
            )
            for k, v in pairs
        ]
        return el("table", C(f"kv:{block.name}"), [el("tbody", "", rows)])
    return el(
        "div",
        C(f"kv:{block.name}"),
        [
            el(
                "div",
                C("kv_row"),
                [el("span", C("kv_key"), [k]), " ", el("span", C("kv_val"), [nb(v)])],
            )
            for k, v in pairs
        ],
    )


def _kv_block(t: TemplateDef, name: str, d: PageData, nb: Nb) -> Node:
    block = next(x for x in t.kvs if x.name == name)
    react = t.markup == "react"
    return el(
        "div" if react else "section",
        t.cls(f"section:{name}"),
        [
            el("div" if react else "h2", t.cls("h2"), [block.heading]),
            _kv_node(t, block, d.kv[name], nb),
        ],
    )


def _record_node(t: TemplateDef, tab: TableDef, d: PageData, nb: Nb, *, thead: bool) -> Node:
    data = d.tables[tab.name]
    if not data.rows:
        return el("p", t.cls(f"empty:{tab.name}"), [f"No {tab.heading.lower()} yet."])
    head: Node = el(
        "tr", "", [el("th", t.cls("th"), [c.label]) for c in tab.columns if c.name in data.columns]
    )
    body: list[Child] = [
        el(
            "tr",
            "",
            [
                el("td", t.cls(f"col:{tab.name}:{c}"), [nb(v)])
                for c, v in zip(data.columns, row, strict=True)
            ],
        )
        for row in data.rows
    ]
    if thead:
        return el(
            "table", t.cls(f"table:{tab.name}"), [el("thead", "", [head]), el("tbody", "", body)]
        )
    return el("table", t.cls(f"table:{tab.name}"), [head, *body], attr("border", "1"))


def _table_block(t: TemplateDef, name: str, d: PageData, nb: Nb) -> Node:
    tab = next(x for x in t.tables if x.name == name)
    react = t.markup == "react"
    return el(
        "div" if react else "section",
        t.cls(f"section:{name}"),
        [
            el("div" if react else "h2", t.cls("h2"), [tab.heading]),
            _record_node(t, tab, d, nb, thead=True),
        ],
    )


def _related_block(t: TemplateDef, d: PageData, nb: Nb) -> Node:
    react = t.markup == "react"
    items: list[Child] = [
        el(
            "div" if react else "li",
            t.cls("related_item"),
            [el("a", "", [nb(text)], attr("href", href))],
        )
        for text, href in d.related
    ]
    return el(
        "div" if react else "aside",
        t.cls("related"),
        [
            el("div" if react else "h3", t.cls("h2"), [t.related_heading]),
            el("div" if react else "ul", t.cls("related_list"), items),
        ],
    )


def _group_block(t: TemplateDef, group: str, d: PageData, nb: Nb) -> Node | None:
    kids: list[Child] = []
    for s in t.scalars:
        if s.group == group:
            node = _scalar_block(t, s, d.scalars[s.name], nb)
            if node is not None:
                kids.append(node)
    if not kids:
        return None
    tag = "ul" if t.field_layout == "li" else "div"
    return el(tag, t.cls(f"group:{group}"), kids)


def content_tree(t: TemplateDef, d: PageData, nb: Nb = _ident) -> Node:
    """The template's content region for one page."""
    if t.markup == "table":
        return _table_content(t, d, nb)
    react = t.markup == "react"
    title_attrs = attr("itemprop", "name") if t.field_scheme == "itemprop" else ""
    title_tag = "h1" if (not react or t.react_h1) else "div"
    blocks: list[Node] = [
        el(
            "div" if react else "header",
            t.cls("header"),
            [el(title_tag, t.cls("title"), [nb(d.title)], title_attrs)],
        )
    ]
    for kind, name in t.order:
        node: Node | None
        if kind == "group":
            node = _group_block(t, name, d, nb)
        elif kind == "para":
            s = t.scalar(name)
            node = _scalar_block(t, s, d.scalars[name], nb)
        elif kind == "list":
            node = _list_block(t, name, d, nb)
        elif kind == "kv":
            node = _kv_block(t, name, d, nb)
        else:
            node = _table_block(t, name, d, nb)
        if node is not None:
            blocks.append(_nest(t, node, f"{kind}:{name}") if react else node)
    if t.related:
        blocks.append(
            _nest(t, _related_block(t, d, nb), "related") if react else _related_block(t, d, nb)
        )
    root_attrs = ""
    if t.field_scheme == "itemprop":
        root_attrs = " itemscope" + attr("itemtype", f"https://schema.org/{t.kind.capitalize()}")
    return el(t.root_tag, t.cls("root"), list(blocks), root_attrs)


def _nest(t: TemplateDef, node: Node, key: str) -> Node:
    for k in range(t.react_depth):
        node = el("div", t.cls(f"inner:{key}:{k}"), [node])
    return node


def _table_content(t: TemplateDef, d: PageData, nb: Nb) -> Node:
    C = t.cls
    title_attrs = attr("itemprop", "name") if t.field_scheme == "itemprop" else ""
    rows: list[Child] = [
        el(
            "tr",
            "",
            [
                el(
                    "td",
                    C("header"),
                    [el("h1", C("title"), [nb(d.title)], title_attrs)],
                    attr("colspan", "2"),
                )
            ],
        )
    ]

    def wide(cls: str, kids: list[Child]) -> Node:
        return el("tr", "", [el("td", cls, kids, attr("colspan", "2"))])

    for kind, name in t.order:
        if kind == "group":
            frows: list[Child] = []
            for s in t.scalars:
                if s.group != name:
                    continue
                value = d.scalars[s.name]
                if value is None and t.absent == "omitted":
                    continue
                wcls, wattrs = _wrap_attrs(t, s, value)
                frows.append(
                    el(
                        "tr",
                        wcls,
                        [
                            el("td", t.label_cls(s.name), [_label(t, s)]),
                            _value_node(t, s, value, "td", nb),
                        ],
                        wattrs,
                    )
                )
            if frows:
                rows.append(
                    wide("", [el("table", C(f"group:{name}"), frows, attr("width", "100%"))])
                )
        elif kind == "para":
            s = t.scalar(name)
            value = d.scalars[name]
            if value is None and t.absent == "omitted":
                continue
            wcls, wattrs = _wrap_attrs(t, s, value)
            rows.append(
                el(
                    "tr",
                    wcls,
                    [
                        el(
                            "td",
                            "",
                            [
                                el("b", t.label_cls(name), [s.label]),
                                _value_node(t, s, value, "div", nb),
                            ],
                            attr("colspan", "2"),
                        )
                    ],
                    wattrs,
                )
            )
        elif kind == "list":
            lst = next(x for x in t.lists if x.name == name)
            items: list[Child] = [
                el(
                    "li",
                    C(f"item:{name}"),
                    [
                        el("a", "", [nb(x)], attr("href", f"/{name}/{_slug(x)}"))
                        if lst.link
                        else nb(x)
                    ],
                )
                for x in d.lists[name]
            ]
            rows.append(
                wide(
                    C(f"section:{name}"),
                    [
                        el("b", C("h2"), [lst.heading]),
                        el("ol" if lst.ordered else "ul", C(f"list:{name}"), items),
                    ],
                )
            )
        elif kind == "kv":
            block = next(x for x in t.kvs if x.name == name)
            rows.append(
                wide(
                    C(f"section:{name}"),
                    [el("b", C("h2"), [block.heading]), _kv_node(t, block, d.kv[name], nb)],
                )
            )
        else:
            tab = next(x for x in t.tables if x.name == name)
            rows.append(
                wide(
                    C(f"section:{name}"),
                    [el("b", C("h2"), [tab.heading]), _record_node(t, tab, d, nb, thead=False)],
                )
            )
    if t.related:
        links: list[Child] = [
            el("li", "", [el("a", "", [nb(x)], attr("href", h))]) for x, h in d.related
        ]
        rows.append(
            wide(
                C("related"),
                [el("b", C("h2"), [t.related_heading]), el("ul", C("related_list"), links)],
            )
        )
    root_attrs = attr("width", "100%")
    if t.field_scheme == "itemprop":
        root_attrs += " itemscope" + attr("itemtype", f"https://schema.org/{t.kind.capitalize()}")
    return el("table", C("root"), rows, root_attrs)
