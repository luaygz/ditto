"""Union page internals on real HTML: table-aware location and splicing, the collapse rules,
text helpers and rendering."""

from __future__ import annotations

from typing import Any

from lxml import html as lhtml
from union_fixtures import cand

from ditto.analyze.models import CandidateTable, Table
from ditto.fingerprint.dom import clean_html
from ditto.union import build_union_page
from ditto.union.compress import COLLAPSE_LEVELS, candidate_note, collapse, render
from ditto.union.prepare import X_C, X_LABEL, truncate_middle
from ditto.union.tree import CandIndex, Tree, find_label, kv_key, locate

PRODUCT = "html > body > div.product"
SPECS_ROW = f"{PRODUCT} > table.specs > tr"
PRICES = f"{PRODUCT} > table.prices"


def product_page(
    name: str,
    specs: list[tuple[str, str]],
    rows: list[tuple[str, ...]],
    headers: tuple[str, ...] = ("Item", "Price"),
) -> str:
    spec_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in specs)
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>" if headers else ""
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return (
        f'<html><body><div class="product"><h1>{name}</h1>'
        f'<table class="specs">{spec_rows}</table>'
        f'<table class="prices">{head}{body}</table></div></body></html>'
    )


def product_table() -> CandidateTable:
    kv = {"table_id": "t0", "table_role": "kv_value"}
    col = {"table_id": "t1", "table_role": "record_column"}
    candidates = [
        cand("c0", f"{PRODUCT} > h1", "mandatory", 1.0),
        cand("c1", f"{SPECS_ROW} > th", "label", 1.0, m=1.0, table_id="t0", table_role="kv_key"),
        cand("c2", f"{SPECS_ROW} > td", "mandatory", 1.0, column="Brand", **kv),
        cand("c3", f"{SPECS_ROW} > td", "optional", 0.3, column="Color", **kv),
        cand("c4", f"{SPECS_ROW} > td", "mandatory", 1.0, column="Weight", **kv),
        cand("c5", f"{PRICES} > tr > th", "label", 1.0, m=1.0, table_id="t1"),
        cand("c6", f"{PRICES} > tr > td", "list", 1.0, column="Item", **col),
        cand("c7", f"{PRICES} > tr > td", "list", 1.0, column="Price", **col),
        cand("c8", f"{PRICES} > tr > td", "list", 0.2, column="Discount", **col),
    ]
    tables = [
        Table(
            id="t0",
            kind="key_value",
            layout="pairs",
            container_path=f"{PRODUCT} > table.specs",
            row_path=SPECS_ROW,
            key_candidate="c1",
            keys_seen=["Brand", "Color", "Weight"],
        ),
        Table(
            id="t1",
            kind="record_list",
            layout="table",
            container_path=PRICES,
            row_path=f"{PRICES} > tr",
        ),
    ]
    return CandidateTable(
        template_id="t_prod", cluster_size=30, n_sampled=30, candidates=candidates, tables=tables
    )


def _kids(el: Any) -> list[Any]:
    return [c for c in el if isinstance(c.tag, str)]


def _text(el: Any) -> str:
    return " ".join(el.text_content().split())


def test_locate_matches_kv_keys_and_record_headers() -> None:
    page = product_page(
        "Lamp",
        [("Brand", "Acme"), ("Weight", "2 kg")],
        [("Bulb", "$3"), ("Shade", "$9")],
    )
    located = locate(Tree(clean_html(page)), CandIndex.build(product_table()))
    texts = {cid: [el.text for el in els] for cid, els in located.items()}
    assert texts == {
        "c0": ["Lamp"],
        "c2": ["Acme"],
        "c4": ["2 kg"],
        "c6": ["Bulb", "Shade"],
        "c7": ["$3", "$9"],
    }


def test_headerless_record_table_uses_column_numbers() -> None:
    table = product_table()
    for c in table.candidates:
        if c.id == "c6":
            c.column = "col1"
        if c.id == "c7":
            c.column = "col2"
    page = product_page("Lamp", [("Brand", "Acme")], [("Bulb", "$3")], headers=())
    located = locate(Tree(clean_html(page)), CandIndex.build(table))
    assert [el.text for el in located["c6"]] == ["Bulb"]
    assert [el.text for el in located["c7"]] == ["$3"]


def test_missing_kv_row_is_spliced_whole_between_shared_keys() -> None:
    pages = {
        0: product_page("Lamp", [("Brand", "Acme"), ("Weight", "2 kg")], [("Bulb", "$3")]),
        1: product_page(
            "Desk", [("Brand", "Oak"), ("Color", "Brown"), ("Weight", "9 kg")], [("Leg", "$1")]
        ),
    }
    page = build_union_page(product_table(), pages, min_tokens=0)
    assert page.base_page_id == 1  # holds the Color row
    pages[1] = product_page("Desk", [("Brand", "Oak"), ("Weight", "9 kg")], [("Leg", "$1")])
    pages[2] = product_page("Cup", [("Brand", "B"), ("Color", "Red"), ("Weight", "1")], [])
    page = build_union_page(product_table(), pages, min_tokens=0)
    assert page.spliced == ["c3"]
    specs = lhtml.fromstring(page.html).find(".//table[@class='specs']")
    assert specs is not None
    # base: the smaller of the two equally covering pages; Color lands between Brand and Weight
    assert page.base_page_id == 1
    assert [_text(r) for r in _kids(specs)] == ["Brand Oak", "Color Red", "Weight 9 kg"]
    assert "<!-- spliced from another page: c3 --><tr>" in page.html


def test_missing_record_column_splices_the_donor_table() -> None:
    pages = {
        0: product_page(
            "Lamp",
            [("Brand", "A"), ("Color", "Red"), ("Weight", "1")],
            [("Bulb", "$3"), ("Cord", "$2")],
        ),
        1: product_page(
            "Desk", [("Brand", "B")], [("Leg", "$1", "5%")], headers=("Item", "Price", "Discount")
        ),
    }
    page = build_union_page(product_table(), pages, min_tokens=0)
    assert page.base_page_id == 0
    assert page.spliced == ["c8"]
    tables = lhtml.fromstring(page.html).findall(".//table[@class='prices']")
    assert len(tables) == 2
    # aligned after the shared sibling preceding it on the donor (the specs table)
    assert "Discount" in tables[0].text_content() and "Discount" not in tables[1].text_content()
    assert "c8 · list · on 20% of pages" in page.html
    # other candidates are annotated once per page, the spliced one inside its unit
    assert page.html.count("<!-- c6 · list") == 1 and page.html.count("<!-- c8 · list") == 1


def test_record_rows_collapse_but_header_row_stays() -> None:
    rows = [(f"Part {i}", f"${i}") for i in range(25)]
    page = build_union_page(
        product_table(), {0: product_page("Lamp", [("Brand", "A")], rows)}, min_tokens=0
    )
    prices = lhtml.fromstring(page.html).find(".//table[@class='prices']")
    assert prices is not None
    kept = [_text(r) for r in _kids(prices)]
    assert kept == ["Item Price", "Part 0 $0", "Part 24 $24"]
    assert "collapsed: 23 more siblings like the one above (variant 2 of 2)" in page.html


def test_collapse_skips_groups_whose_members_carry_distinct_labels() -> None:
    def rows(labels: list[str]) -> Any:
        body = "".join(
            f'<div class="row"><span class="k" {X_LABEL}="c1">{k}</span>'
            f'<span class="v" {X_C}="c1">v{i}</span></div>'
            for i, k in enumerate(labels)
        )
        return lhtml.fromstring(f'<div class="list">{body}</div>')

    kv = rows([f"Key {i}" for i in range(8)])
    assert collapse(kv, COLLAPSE_LEVELS[0]) == 0
    assert len(_kids(kv)) == 8
    same = rows(["Price:"] * 8)
    assert collapse(same, COLLAPSE_LEVELS[0]) == 1
    assert len(_kids(same)) == 2  # first and last; one variant
    html_out = render(same, {"c1": "c1 · list · on 100% of pages"})
    assert "collapsed: 6 more siblings like the one above (variant 1 of 1)" in html_out
    assert "data-x-" not in html_out


def test_highest_collapse_level_keeps_one_example_per_group() -> None:
    body = "".join(f'<li class="i"><b>{i}</b>{"<i>x</i>" if i % 2 else ""}</li>' for i in range(9))
    root = lhtml.fromstring(f"<ul>{body}</ul>")
    assert collapse(root, COLLAPSE_LEVELS[-1]) == 1
    assert [li.text_content() for li in _kids(root)] == ["0"]
    assert "collapsed: 8 more siblings like the one above (variant 1 of 1)" in render(root, {})


def test_find_label_and_kv_key_without_key_path() -> None:
    root = lhtml.fromstring(
        "<div><p>Contact</p><span>Email:</span><a>a@b.org</a>"
        "<dl><dt>Fax</dt><dd>123</dd></dl></div>"
    )
    a = root.find("a")
    assert a is not None
    label = find_label(a, "email")
    assert label is not None and label.text == "Email:"
    assert find_label(a, "Contact") is not None  # the parent's first child
    assert find_label(a, "Phone") is None and find_label(a, None) is None
    dd = root.find(".//dd")
    assert dd is not None
    key = kv_key(Tree(root), dd, None, None, ["Fax"])
    assert key is not None and key.text == "Fax"


def test_truncate_middle_and_notes() -> None:
    assert truncate_middle("short", 10) == "short"
    cut = truncate_middle("a" * 100 + "b" * 100, 21)
    assert cut.startswith("a" * 9) and cut.endswith("b" * 9) and " … " in cut
    rare = cand("c9", "html > body > p", "optional", 0.004, examples=["", " x  -- y "])
    assert candidate_note(rare) == "c9 · optional · on <1% of pages · e.g. x -- y"
    root = lhtml.fromstring(f'<div><p {X_C}="c9">text</p></div>')
    out = render(root, {"c9": candidate_note(rare)})
    assert "<!-- c9 · optional · on <1% of pages · e.g. x - - y -->" in out


def test_tail_value_pair_row_is_spliced_whole() -> None:
    facts = "html > body > div.product > ul.facts > li"
    kv = {"table_id": "t0", "table_role": "kv_value"}
    table = CandidateTable(
        template_id="t_facts",
        cluster_size=30,
        n_sampled=30,
        candidates=[
            cand("c0", "html > body > div.product > h1", "mandatory", 1.0),
            cand("c1", f"{facts} > b", "label", 1.0, m=1.0, table_id="t0", table_role="kv_key"),
            cand("c2", facts, "mandatory", 1.0, column="Brand:", **kv),
            cand("c3", facts, "optional", 0.4, column="Color:", **kv),
            cand("c4", facts, "mandatory", 1.0, column="Weight:", **kv),
        ],
        tables=[
            Table(
                id="t0",
                kind="key_value",
                layout="pairs",
                container_path="html > body > div.product > ul.facts",
                row_path=facts,
                key_candidate="c1",
            )
        ],
    )

    def page(name: str, rows: list[tuple[str, str]]) -> str:
        items = "".join(f"<li><b>{k}</b> {v}</li>" for k, v in rows)
        return f'<html><body><div class="product"><h1>{name}</h1><ul class="facts">{items}</ul></div></body></html>'

    pages = {
        0: page("Lamp", [("Brand:", "Acme"), ("Weight:", "2 kg")]),
        1: page("Cup", [("Color:", "Red")]),
    }
    union = build_union_page(table, pages, min_tokens=0)
    assert union.base_page_id == 0 and union.spliced == ["c3"]
    ul = lhtml.fromstring(union.html).find(".//ul")
    assert ul is not None
    assert [_text(li) for li in _kids(ul)] == ["Brand: Acme", "Weight: 2 kg", "Color: Red"]
    assert union.missing == []


def test_colspan_survives_pruning_so_numbered_columns_still_locate() -> None:
    cell = "html > body > table.res > tr > td"
    table = CandidateTable(
        template_id="t_cs",
        cluster_size=30,
        n_sampled=30,
        candidates=[
            cand("c0", cell, "list", 1.0, table_id="t0", table_role="record_column", column="col4")
        ],
        tables=[
            Table(
                id="t0",
                kind="record_list",
                layout="table",
                container_path="html > body > table.res",
                row_path="html > body > table.res > tr",
            )
        ],
    )
    rows = "".join(
        f'<tr><td>{i}</td><td colspan="2" style="x">wide {i}</td><td>buy {i}</td></tr>'
        for i in range(3)
    )
    page = f'<html><body><table class="res">{rows}</table></body></html>'
    union = build_union_page(table, {0: page}, min_tokens=0)
    assert union.shown == ["c0"]
    assert '<td colspan="2">wide 0</td>\n<!-- c0 · list' in union.html
