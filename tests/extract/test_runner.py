from __future__ import annotations

from typing import Any

import pytest
from lxml import etree

from ditto.extract.runner import (
    _header_cells,
    _header_column,
    apply_spec,
    apply_spec_html,
    candidate_values,
    clean_key,
    compile_xpath,
    evaluate,
    field_value,
    is_missing,
)
from ditto.fingerprint.dom import clean_html
from ditto.specgen.models import KeyValueField, ListField, RecordListField, ScalarField, Spec
from ditto.specgen.schema_export import EXAMPLE_HTML, EXAMPLE_SPEC

PRODUCT = """<!doctype html>
<html lang="en"><head>
  <title>Trail Runner 3 | Outdoor Shop</title>
  <meta itemprop="price" content="129.00">
  <script type="application/ld+json">{"@type": "Product", "name": "LEAKED"}</script>
</head>
<body class="page product">
  <nav class="crumbs"><a href="/">Home</a> › <a href="/shoes">Shoes</a></nav>
  <div id="product" itemscope>
    <h1 class="product-title" itemprop="name">
       Trail   Runner 3
    </h1>
    <p class="price"><span class="old">€149,00</span> <span class="now">€1.129,50</span></p>
    <p class="sku">SKU: <b>TR-3-0042</b></p>
    <p class="published">Listed on 12 Mar 2021</p>
    <ul class="tags"><li> trail </li><li></li><li>running</li><li>  </li><li>waterproof</li></ul>
    <a class="brand" href="/brands/peak?ref=pdp">Peak Outfitters</a>
    <table class="variants">
      <thead><tr><th>Color</th><th>Size</th><th>Price</th><th>Stock</th></tr></thead>
      <tbody>
        <tr><td>Red</td><td>42</td><td>€129,00</td><td>3 left</td></tr>
        <tr><td>Blue</td><td>43</td><td>€1.129,50</td><td></td></tr>
        <tr><td></td><td></td><td></td><td></td></tr>
        <tr><td>Green</td><td>44</td><td>n/a</td><td>12 left</td></tr>
      </tbody>
    </table>
    <dl class="specs">
      <dt>Weight:</dt><dd>280 g</dd>
      <dt> Drop </dt><dd>6 mm</dd>
      <dt>Weight</dt><dd>999 g</dd>
      <dt>Upper :</dt><dd></dd>
      <dt></dt><dd>orphan value</dd>
    </dl>
    <table class="facts">
      <tr><th>Origin</th><td>Portugal</td></tr>
      <tr><th>Warranty</th><td>2 years</td></tr>
    </table>
  </div>
  <footer><a href="mailto:help@shop.example">help@shop.example</a></footer>
</body></html>"""

ROOT = clean_html(PRODUCT)


def _col(header: str) -> str:
    return f"./td[count(ancestor::table//th[normalize-space()='{header}']/preceding-sibling::th)+1]"


def test_evaluate_shapes() -> None:
    assert [e.tag for e in evaluate("//h1[@itemprop='name']", ROOT)] == ["h1"]
    assert evaluate("//a[@class='brand']/@href", ROOT) == ["/brands/peak?ref=pdp"]
    assert evaluate("count(//ul[@class='tags']/li)", ROOT) == [5.0]
    assert evaluate("boolean(//h1)", ROOT) == [True]
    assert evaluate("boolean(//h7)", ROOT) == []
    assert evaluate("number('x')", ROOT) == []  # NaN
    assert evaluate("string(//p[@class='sku']/b)", ROOT) == ["TR-3-0042"]
    assert evaluate("string(//h7)", ROOT) == []
    assert evaluate("//a[ends-with(@href, 'x')]", ROOT) == []  # fails at evaluation time
    assert evaluate("//script", ROOT) == []  # cleaned away


def test_compile_cache_and_smart_strings() -> None:
    assert compile_xpath("//h1") is compile_xpath("//h1")
    href = evaluate("//a[@class='brand']/@href", ROOT)[0]
    assert href.getparent().get("class") == "brand"


def test_head_is_stripped_except_title() -> None:
    title = ScalarField(candidates=["//title", "//head/title"])
    assert field_value(title, ROOT) == "Trail Runner 3 | Outdoor Shop"
    meta = ScalarField(candidates=["//meta[@itemprop='price']/@content", "//p[@class='price']"])
    assert field_value(meta, ROOT) is None  # <meta> is not part of the cleaned page


def test_scalar_first_non_null_then_fallbacks() -> None:
    name = ScalarField(candidates=["//h1[@itemprop='name']", "//header//h1"])
    assert field_value(name, ROOT) == "Trail Runner 3"
    # the selected candidate matches only empty nodes first: the first non-null match wins
    tag = ScalarField(candidates=["//ul[@class='tags']/li[position() > 1]", "//a[@rel='tag']"])
    assert field_value(tag, ROOT) == "running"  # li[2] is empty and skipped
    price = ScalarField(
        candidates=["//span[@class='sale']", "//p[@class='price']/span[@class='now']"],
        fallbacks=[1],
        transforms=["cast(float)"],
    )
    assert field_value(price, ROOT) == 1129.5
    no_fallback = price.model_copy(update={"fallbacks": []})
    assert field_value(no_fallback, ROOT) is None
    sku = ScalarField(
        candidates=["//p[@class='sku']", "//p[@class='sku']/b"],
        selected=0,
        transforms=["regex('SKU:\\s*(\\S+)', 1)"],
    )
    assert field_value(sku, ROOT) == "TR-3-0042"
    date = ScalarField(
        candidates=["//p[@class='published']", "//p[contains(., 'Listed')]"],
        transforms=["cast(date)"],
        type_check="date",
    )
    assert field_value(date, ROOT) == "2021-03-12"
    brand = ScalarField(
        candidates=["//a[@class='brand']", "//a[contains(@href, '/brands/')]"],
        transforms=["attr(href)", "split('?', 0)", "split('/', -1)"],
    )
    assert field_value(brand, ROOT) == "peak"


def test_list_all_matches_in_order() -> None:
    tags = ListField(candidates=["//ul[@class='tags']/li", "//ul[contains(@class,'tags')]//li"])
    assert field_value(tags, ROOT) == ["trail", "running", "waterproof"]
    crumbs = ListField(
        candidates=["//nav[@class='breadcrumbs']//a", "//nav[@class='crumbs']/a/@href"],
        fallbacks=[1],
    )
    assert field_value(crumbs, ROOT) == ["/", "/shoes"]
    upper = ListField(
        candidates=["//ul[@class='tags']/li", "//nav[@class='crumbs']/a"],
        transforms=["regex('^w.*', 0)"],
    )
    assert field_value(upper, ROOT) == ["waterproof"]
    none = ListField(candidates=["//ul[@class='colors']/li", "//ol[@class='x']/li"])
    assert field_value(none, ROOT) == []


def test_record_list_columns_by_header_text() -> None:
    variants = RecordListField(
        container="//table[contains(@class,'variants')]//tr[td]",
        columns={"price": _col("Price"), "color": _col("Color"), "stock": _col("Stock")},
        column_source="header_row",
        transforms={"price": ["cast(float)"], "stock": ["cast(int)"]},
    )
    assert field_value(variants, ROOT) == [
        {"price": 129.0, "color": "Red", "stock": 3},
        {"price": 1129.5, "color": "Blue", "stock": None},
        # the all-empty row is skipped
        {"price": None, "color": "Green", "stock": 12},
    ]


TABLES = """<div>
<table class="t1"><thead><tr><th>Name</th><th>Price</th><th>Link</th></tr></thead>
  <tbody><tr><td>A</td><td>€1</td><td><a href="/a">a</a></td></tr>
         <tr><td>B</td><td>€2</td><td><a href="/b">b</a> <a href="/b2">b2</a></td></tr></tbody>
  <tbody><tr><td>C</td><td>€3</td></tr></tbody></table>
<table class="t2"><tr><th>Price</th><th>Name</th></tr><tr><td>9</td><td>Z</td></tr>
  <tr><th>Total</th><td>row header</td></tr></table>
</div>"""


@pytest.mark.parametrize(
    "expr",
    [
        _col("Name"),
        _col("Price"),
        _col("Link") + "//a/@href",
        _col("Link") + "/a",
        _col("Missing"),  # count() is 0, so the brief's pattern falls back to the first cell
        "./th[count(ancestor::table//th[normalize-space()='Price']/preceding-sibling::th)+1]",
        " ./td[ count( ancestor::table[1] // th[normalize-space()='Name'] / preceding-sibling::th"
        " ) + 1 ]/text() ",
    ],
)
def test_header_column_fast_path_matches_literal_xpath(expr: str) -> None:
    plan = _header_column(expr)
    assert plan is not None
    root = clean_html(TABLES)
    positions: dict[tuple[Any, str], int] = {}
    for row in root.xpath("//tr"):
        assert _header_cells(plan, row, positions) == evaluate(expr, row)
    assert 1 <= len(positions) <= 3  # one lookup per row parent holding such cells, not per row


@pytest.mark.parametrize(
    "expr",
    [
        "./td[2]",
        ".//td[count(ancestor::table//th[normalize-space()='A']/preceding-sibling::th)+1]",
        _col("A") + " | ./th",
        _col("A") + "[1]",
        "./td[count(ancestor::table//th[normalize-space()='A']/following-sibling::th)+1]",
    ],
)
def test_other_column_shapes_use_plain_xpath(expr: str) -> None:
    assert _header_column(expr) is None


def test_record_list_over_several_row_groups() -> None:
    root = clean_html(TABLES)
    field = RecordListField(
        container="//table[@class='t1']//tr[td]",
        columns={"name": _col("Name"), "price": _col("Price"), "links": _col("Link") + "//a/@href"},
        column_source="header_row",
        transforms={"price": ["cast(int)"]},
    )
    assert field_value(field, root) == [
        {"name": "A", "price": 1, "links": "/a"},
        {"name": "B", "price": 2, "links": "/b"},
        {"name": "C", "price": 3, "links": None},
    ]
    big = "<table class='t'><tr><th>N</th><th>V</th></tr>" + "".join(
        f"<tr><td>{i}</td><td>v{i}</td></tr>" for i in range(1500)
    )
    rows = field_value(
        RecordListField(
            container="//table[@class='t']//tr[td]",
            columns={"v": _col("V")},
            column_source="header_row",
        ),
        clean_html(big + "</table>"),
    )
    assert len(rows) == 1500 and rows[-1] == {"v": "v1499"}


def test_record_list_skips_non_element_rows_and_handles_no_rows() -> None:
    attrs = RecordListField(
        container="//table[contains(@class,'variants')]/@class",
        columns={"x": "./td[1]"},
    )
    assert field_value(attrs, ROOT) == []
    missing = RecordListField(container="//table[@id='nope']//tr", columns={"x": "./td[1]"})
    assert field_value(missing, ROOT) == []


def test_key_value_dl_first_key_wins_and_colons_stripped() -> None:
    specs = KeyValueField(pairs="//dl[@class='specs']/dt", value="./following-sibling::dd[1]")
    assert field_value(specs, ROOT) == {
        "Weight": "280 g",  # the later "Weight" is ignored
        "Drop": "6 mm",
        "Upper": None,  # key present, empty value
    }


def test_key_value_table_rows_with_transforms() -> None:
    facts = KeyValueField(
        pairs="//table[@class='facts']//tr",
        pairs_alternates=["//table[contains(@class,'facts')]//tr[th]"],
        key="./th",
        value="./td",
        transforms={"key": ["replace('Warranty', 'Guarantee')"], "value": ["cast(int)"]},
    )
    assert field_value(facts, ROOT) == {"Origin": None, "Guarantee": 2}
    nothing = KeyValueField(pairs="//dl[@id='x']/dt", value="./following-sibling::dd[1]")
    assert field_value(nothing, ROOT) == {}
    on_attrs = KeyValueField(pairs="//dl[@class='specs']/@class", value=".")
    assert field_value(on_attrs, ROOT) == {}


@pytest.mark.parametrize(
    ("raw", "key"),
    [("Email:", "Email"), ("  Upper :  ", "Upper"), ("Höhe：", "Höhe"), ("a\n b", "a b"), (3, "3")],
)
def test_clean_key(raw: Any, key: str) -> None:
    assert clean_key(raw) == key


def test_candidate_values() -> None:
    assert candidate_values("//ul[@class='tags']/li", ROOT, ["strip"]) == [
        "trail",
        "running",
        "waterproof",
    ]
    assert candidate_values("//h7", ROOT, []) == []


def test_apply_spec_meta_and_missing() -> None:
    spec = Spec.model_validate(
        {
            "template_id": "t_prod",
            "site": "shop.example",
            "generated_by": {"model": "m", "prompt_version": "specgen.v1"},
            "fields": {
                "name": {
                    "type": "scalar",
                    "candidates": ["//h1[@itemprop='name']", "//h1[@class='product-title']"],
                },
                "rating": {
                    "type": "scalar",
                    "candidates": ["//span[@class='rating']", "//*[@itemprop='ratingValue']"],
                },
                "tags": {
                    "type": "list",
                    "candidates": ["//ul[@class='tags']/li", "//a[@rel='tag']"],
                },
                "reviews": {
                    "type": "record_list",
                    "container": "//ol[@class='reviews']/li",
                    "columns": {"text": "./p"},
                },
                "facts": {
                    "type": "key_value",
                    "pairs": "//table[@class='facts']//tr",
                    "key": "./th",
                    "value": "./td",
                },
                "specs": {
                    "type": "key_value",
                    "pairs": "//dl[@class='nope']/dt",
                    "value": "./following-sibling::dd[1]",
                },
            },
        }
    )
    out = apply_spec(spec, ROOT)
    assert out["name"] == "Trail Runner 3"
    assert out["tags"] == ["trail", "running", "waterproof"]
    assert out["facts"] == {"Origin": "Portugal", "Warranty": "2 years"}
    assert out["_meta"] == {"missing": ["rating", "reviews", "specs"], "spec_version": "1"}
    assert list(out) == ["name", "rating", "tags", "reviews", "facts", "specs", "_meta"]
    # Spec.apply cleans raw HTML (str or bytes) itself and gives the same answer
    assert spec.apply(PRODUCT) == out
    assert spec.apply(PRODUCT.encode("utf-8")) == out
    assert apply_spec_html(spec, PRODUCT) == out


def test_brief_example_extracts_as_expected() -> None:
    out = Spec.model_validate(EXAMPLE_SPEC).apply(EXAMPLE_HTML)
    assert out == {
        "name": "Dr. Amal Khoury",
        "email": "a.khoury@example.org",
        "publications": [
            {"title": "Wrapper induction with one call per template", "year": 2021},
            {"title": "Template clustering at web scale", "year": 2019},
        ],
        "details": {
            "Email": "a.khoury@example.org",
            "Office": "Building B, Room 214",
            "Phone": "+961 1 350 000 ext. 4321",
        },
        "_meta": {"missing": [], "spec_version": "1"},
    }


def test_brief_example_optional_email_missing() -> None:
    page = EXAMPLE_HTML.replace(
        '<dt>Email</dt><dd><a href="mailto:a.khoury@example.org">a.khoury@example.org</a></dd>',
        "",
    )
    out = Spec.model_validate(EXAMPLE_SPEC).apply(page)
    assert out["email"] is None and "Email" not in out["details"]
    assert out["_meta"]["missing"] == ["email"]


def test_is_missing() -> None:
    assert is_missing(None) and is_missing([]) and is_missing({})
    assert not is_missing(0) and not is_missing([None]) and not is_missing({"a": None})


def test_evaluate_on_subtree_is_relative() -> None:
    row = ROOT.xpath("//table[@class='facts']//tr")[1]
    assert [etree.tostring(e, encoding="unicode") for e in evaluate("./th", row)] == [
        "<th>Warranty</th>"
    ]


@pytest.mark.parametrize(
    ("expr", "fast"),
    [
        ("//div[@id='x']/h1", "/descendant::div[@id='x']/h1"),
        (
            "//main[@id='m']//h1[contains(concat(' ',normalize-space(@class),' '),' t ')]",
            "/descendant::main[@id='m']/descendant::h1[contains(@class,'t')][contains(concat("
            "' ',normalize-space(@class),' '),' t ')]",
        ),
        (
            'p[contains(concat(" ", normalize-space( @class ), " "), " a-b ")][1]',
            'p[contains(@class,"a-b")][contains(concat(" ", normalize-space( @class ), " "), '
            '" a-b ")][1]',
        ),
        (
            "//a[contains(concat(' ',normalize-space(@class),' '),' a b ')]",
            "/descendant::a[contains(concat(' ',normalize-space(@class),' '),' a b ')]",
        ),
        (".//th[normalize-space()='Topic']", "./descendant::th[normalize-space()='Topic']"),
        ("//li[a[1]]", "/descendant::li[a[1]]"),
        (
            "//tr[td[@class='l'][starts-with(., 'X')]]",
            "/descendant::tr[td[@class='l'][starts-with(., 'X')]]",
        ),
        ("//div[@a and @b]//*[@c]", "/descendant::div[@a and @b]/descendant::*[@c]"),
        ("//div[div]", "/descendant::div[div]"),
        ("//x[@n * 2 > 3]", "/descendant::x[@n * 2 > 3]"),
        ("//a[not(@href)] | //b[@x|@y]", "/descendant::a[not(@href)] | /descendant::b[@x|@y]"),
        ("//text()[normalize-space()]", "/descendant::text()[normalize-space()]"),
        ("(//a[@x])[1]", "(/descendant::a[@x])[1]"),
        ("//a[.='//b[@c]']", "/descendant::a[.='//b[@c]']"),
        ("//td[preceding-sibling::th]", "/descendant::td[preceding-sibling::th]"),
        ("//a[count(b) = 0]", "/descendant::a[count(b) = 0]"),
        ("//li[text()]", "/descendant::li[text()]"),
    ],
)
def test_fast_xpath_rewrites_node_by_node_predicates(expr: str, fast: str) -> None:
    from ditto.extract.runner import fast_xpath

    assert fast_xpath(expr) == fast


@pytest.mark.parametrize(
    "expr",
    [
        "//div",  # libxml2 already scans descendants once
        "//li[1]",
        "//li[last()]",
        "//li[position() < 3]",
        "//a[@x][2]",
        "//text()[normalize-space()][1]",
        "//li[count(a)]",
        "//a[string-length(.)]",
        "//li[$v]",
        "//li['x']",
        "//li[(@x)]",
        "//@href[.='x']",
        "//self::div[@a]",
        "//li[last()-1]/a",
        "//\u00e9l[@a]",  # does not tokenize: left alone
        "//a[@x",
        "a b",
    ],
)
def test_fast_xpath_leaves_positional_and_unknown_steps(expr: str) -> None:
    from ditto.extract.runner import fast_xpath

    assert fast_xpath(expr) == expr


TRICKY = """<html><body>
<div id="a" class="box x"><ul><li>1<a href="/1">one</a></li><li>2</li><li>3<a>three</a></li></ul>
  <div class="box"><ul><li>4</li><li>5<a href="/5">five</a></li></ul></div></div>
<table><tr><th>K</th><th>V</th></tr><tr><td>k1</td><td>v1</td></tr><tr><td>k2</td><td> </td></tr>
</table><p>t1<b>b</b>t2</p><p class="x"> </p><p class="boxes xx">y</p></body></html>"""


@pytest.mark.parametrize(
    "expr",
    [
        "//li[a]",
        "//li[a][1]",
        "//li[1]",
        "//div[contains(concat(' ',normalize-space(@class),' '),' box ')]//li[a[@href]]/a",
        "//ul[li]/li[last()]",
        "//tr[td[1]='k2']/td[2]",
        "//td[count(preceding-sibling::td) = 1]",
        "//text()[normalize-space()][1]",
        "//text()[normalize-space()]",
        "//p[@class]|//li[a]",
        "(//li[a])[2]",
        "//div[@id='a']//div//li[. = '4']",
        "//*[@class='x']",
        "count(//li[a])",
        "//li[a and position() > 1]",
        "//*[contains(concat(' ',normalize-space(@class),' '),' x ')][1]",
        "(//*[contains(concat(' ',normalize-space(@class),' '),' box ')])[2]//li",
        "//div[contains(concat(' ',normalize-space(@class),' '),' bo ')]",
        "//p[contains(concat(' ',normalize-space(@class),' '),' x ')]/text()",
    ],
)
def test_compiled_xpath_selects_what_the_expression_says(expr: str) -> None:
    root = clean_html(TRICKY)
    rows = root.xpath("//ul")
    for node in [root, *rows]:
        want = etree.XPath(expr)(node)
        got = compile_xpath(expr)(node)
        if isinstance(want, list):
            assert [id(x) if isinstance(x, etree._Element) else x for x in got] == [
                id(x) if isinstance(x, etree._Element) else x for x in want
            ]
        else:
            assert got == want


def test_compile_xpath_raises_for_invalid_selector() -> None:
    with pytest.raises(etree.XPathSyntaxError):
        compile_xpath("//a[@x")
    assert evaluate("//a[@x", ROOT) == []


ANCHORED = """<html><body>
<div id="main"><h1>Title</h1><ul id="list"><li><a href="/1">one</a></li><li>two</li></ul>
  <div id="dup"><p>outer</p><div id="dup"><p>inner</p></div></div></div>
<span id="dup">third</span><p id="it's">quoted</p><section id="s"><b>x</b></section>
<div id="s">not a section</div></body></html>"""


@pytest.mark.parametrize(
    ("expr", "anchor"),
    [
        ("//div[@id='main']//h1", ("div", "main", ".//h1")),
        ('//div[ @id = "main" ]/h1/text()', ("div", "main", "./h1/text()")),
        ("//*[@id='list']", ("*", "list", None)),
        ("//ul[@id='list']/li[a]/a/@href", ("ul", "list", "./li[a]/a/@href")),
        (
            "//div[@id='main']/following-sibling::span[1]",
            ("div", "main", "./following-sibling::span[1]"),
        ),
        ("//div[@id='main']/..", ("div", "main", "./..")),
    ],
)
def test_id_anchor_splits_anchor_and_rest(expr: str, anchor: tuple[str, str, str | None]) -> None:
    from ditto.extract.runner import _id_anchor

    found = _id_anchor(expr)
    assert found is not None and (found.tag, found.id, found.rest) == anchor


@pytest.mark.parametrize(
    "expr",
    [
        "//div[@id='main'][1]//h1",  # another predicate on the anchor
        "//div[@id='main']//h1 | //h2",  # union
        "//div[@id='main']//li = 'two'",  # operator
        "(//div[@id='main']//li)[1]",
        "//div[@class='main']//h1",
        "//div[@id=$x]//h1",
        "//div[@id='main' or @id='x']",
        "//x:div[@id='main']",
        "//text()[@id='main']",
        "/html//div[@id='main']",
        ".//div[@id='main']",
        "//div[@id='main']]",
    ],
)
def test_id_anchor_rejects_other_shapes(expr: str) -> None:
    from ditto.extract.runner import _id_anchor

    assert _id_anchor(expr) is None


@pytest.mark.parametrize(
    "expr",
    [
        "//div[@id='main']//h1",
        "//div[@id='main']",
        "//*[@id='list']/li",
        "//ul[@id='list']/li[a]/a/@href",
        "//ul[@id='list']//text()",
        "//div[@id='dup']//p",  # nested duplicates: evaluated as written
        "//*[@id='dup']",
        "//span[@id='dup']",  # one span among three ids 'dup'
        '//p[@id="it\'s"]',
        "//section[@id='s']/b",
        "//div[@id='s']",
        "//div[@id='nope']//p",
        "//div[@id='main']/..",
        "//div[@id='main']//h1[",  # invalid: no matches either way
        "//div[@id='main']//h1 | //p",
    ],
)
def test_anchored_evaluation_selects_what_the_expression_says(expr: str) -> None:
    from ditto.extract.runner import Anchors, _evaluate

    root = clean_html(ANCHORED)
    ids = frozenset({"main", "list", "dup", "it's", "s", "nope"})
    anchors = Anchors(root, ids)
    for node in (root, *root.iter("li")):
        want = evaluate(expr, node)
        got = _evaluate(expr, node, anchors)
        assert [id(x) if isinstance(x, etree._Element) else x for x in got] == [
            id(x) if isinstance(x, etree._Element) else x for x in want
        ]
    assert _evaluate("//div[@id='main']//h1", root, Anchors(root, frozenset())) == evaluate(
        "//div[@id='main']//h1", root
    )


def test_apply_spec_collects_anchor_ids_of_every_field_type() -> None:
    from ditto.extract.runner import Anchors

    spec = Spec.model_validate(
        {
            "template_id": "t_x",
            "site": "x.example",
            "generated_by": {"model": "m", "prompt_version": "specgen.v1"},
            "fields": {
                "title": {"type": "scalar", "candidates": ["//div[@id='main']/h1", "//h1"]},
                "shared": {"type": "scalar", "candidates": ["//*[@id='s']/b", "//*[@id='s']"]},
                "items": {"type": "list", "candidates": ["//ul[@id='list']/li", "//li[a]"]},
                "rows": {
                    "type": "record_list",
                    "container": "//div[@id='main']/ul/li",
                    "columns": {"text": ".", "link": "./a/@href"},
                },
                "kv": {
                    "type": "key_value",
                    "pairs": "//section[@id='s']",
                    "key": "./b",
                    "value": "./b",
                },
            },
        }
    )
    root = clean_html(ANCHORED)
    assert Anchors.for_spec(spec, root).ids == {"main", "list", "s"}
    assert apply_spec(spec, root) == {
        "title": "Title",
        "shared": "x",
        "items": ["one", "two"],
        "rows": [{"text": "one", "link": "/1"}, {"text": "two", "link": None}],
        "kv": {"x": "x"},
        "_meta": {"missing": [], "spec_version": spec.spec_version},
    }


NBSP_PAGE = """<html><body><table class="info">
<tr><td>Type:\u00a0\u00a0</td><td>Public</td></tr>
<tr><td>\u00a0Size:</td><td>Large</td></tr>
<tr><td>Plain:</td><td>Value</td></tr></table>
<p class="a\u00a0b">nbsp class</p></body></html>"""


@pytest.mark.parametrize(
    ("expr", "want"),
    [
        ("//td[normalize-space(.)='Type:']/following-sibling::td[1]", ["Public"]),
        ("//td[normalize-space()='Size:']/following-sibling::td[1]", ["Large"]),
        ("//td[normalize-space(text())='Plain:']/following-sibling::td[1]", ["Value"]),
        ("//td[normalize-space(.)='Type:\u00a0']", []),  # literals are left alone
        ("//td[contains(., 'Type:\u00a0')]/following-sibling::td[1]", ["Public"]),
        ("normalize-space(//table[@class='info']//td[1])", ["Type:"]),
        ("//p[contains(concat(' ',normalize-space(@class),' '),' b ')]", ["nbsp class"]),
    ],
)
def test_normalize_space_treats_nbsp_as_space(expr: str, want: list[str]) -> None:
    root = clean_html(NBSP_PAGE)
    got = [x if isinstance(x, str) else x.text_content() for x in evaluate(expr, root)]
    assert got == want


def test_spec_apply_and_extraction_path_share_nbsp_semantics() -> None:
    from ditto.fingerprint.dom import clean_etree

    spec = Spec.model_validate(
        {
            "template_id": "t_nbsp",
            "site": "x.example",
            "generated_by": {"model": "m", "prompt_version": "specgen.v1"},
            "fields": {
                "type": {
                    "type": "scalar",
                    "candidates": [
                        "//td[normalize-space(.)='Type:']/following-sibling::td[1]",
                        "//table[@class='info']//tr[1]/td[2]",
                    ],
                },
                "info": {
                    "type": "key_value",
                    "pairs": "//table[@class='info']//tr",
                    "key": "./td[1]",
                    "value": "./td[normalize-space()='Large']",
                },
            },
        }
    )
    want = {
        "type": "Public",
        "info": {"Type": None, "Size": "Large", "Plain": None},
        "_meta": {"missing": [], "spec_version": spec.spec_version},
    }
    data = NBSP_PAGE.encode()
    assert spec.apply(NBSP_PAGE) == want  # standalone Spec.apply
    assert apply_spec_html(spec, data, "utf-8") == want
    assert apply_spec(spec, clean_etree(data, "utf-8")) == want  # what extract_run does


@pytest.mark.parametrize(
    ("expr", "rewritten"),
    [
        ("//td[normalize-space()='x']", "//td[normalize-space(translate(.,'\u00a0',' '))='x']"),
        (
            "//td[normalize-space( td[1] ) = 'a']",
            "//td[normalize-space(translate( td[1] ,'\u00a0',' ')) = 'a']",
        ),
        (
            "normalize-space(concat(normalize-space(),'x'))",
            "normalize-space(translate(concat(normalize-space(translate(.,'\u00a0',' ')),'x'),"
            "'\u00a0',' '))",
        ),
        ("//td['normalize-space()']", "//td['normalize-space()']"),
        ("//normalize-space", "//normalize-space"),
        ("//td[normalize-space(]", "//td[normalize-space(]"),
        ("a b", "a b"),
    ],
)
def test_nbsp_as_space_rewrites_only_normalize_space_calls(expr: str, rewritten: str) -> None:
    from ditto.extract.runner import nbsp_as_space

    assert nbsp_as_space(expr) == rewritten
