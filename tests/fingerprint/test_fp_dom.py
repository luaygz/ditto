from __future__ import annotations

import pytest
from lxml import etree

from ditto.fingerprint.dom import (
    Step,
    canonical_classes,
    clean_html,
    is_utility_class,
    iter_paths,
    iter_text_items,
    keep_identifier,
    looks_hashed,
    node_text,
    own_text,
    parse_html,
    parse_step,
    split_path,
    step_label,
    step_of,
)

PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"><title> Blue  Widget </title>
  <link rel="stylesheet" href="/s.css"><style>.x{color:red}</style>
  <script>var x = "<p>not text</p>";</script>
</head>
<body class="page mt-4 flex">
  <!-- a comment -->
  <noscript><p>enable js</p></noscript>
  <header id="top" class="site-header css-1a2b3c">
    <nav role="navigation"><a href="/">Home</a> <a href="/shop">Shop</a></nav>
  </header>
  <main id="post-1234" class="product sc-AxjAm">
    <h1 itemprop="name" class="title text-xl">Blue Widget</h1>
    <dl class="specs">
      <dt>Price</dt><dd data-testid="price">$9.99</dd>
      <dt>Color</dt><dd>blue&nbsp;and   green</dd>
    </dl>
    <p class="desc">A <b>very</b> fine widget.<img src="/w.png"></p>
    <svg><text>icon</text></svg>
    <iframe src="https://ads.example"></iframe>
    <template><p>later</p></template>
  </main>
</body>
</html>"""


@pytest.fixture
def root() -> etree._Element:
    return clean_html(PAGE)


def test_canonical_classes_drops_utility_state_and_hashed() -> None:
    assert canonical_classes("product mt-4 flex text-xl css-1a2b3c sc-AxjAm") == ["product"]
    assert canonical_classes("card active is-open hidden _x9f") == ["card"]
    assert canonical_classes("zeta beta alpha gamma") == ["alpha", "beta", "gamma"]
    assert canonical_classes("Price  price price") == ["Price", "price"]
    assert canonical_classes("a8f3k9 title") == ["title"]
    assert canonical_classes("") == []
    assert canonical_classes(None) == []
    assert canonical_classes("1col :hover title") == ["title"]


def test_class_helpers() -> None:
    assert looks_hashed("css-abc")
    assert looks_hashed("Xy7Qz1")
    assert not looks_hashed("product-title")
    assert not looks_hashed("h1")
    assert is_utility_class("px-2")
    assert is_utility_class("Active")
    assert not is_utility_class("specs")


def test_keep_identifier() -> None:
    assert keep_identifier(" main ") == "main"
    assert keep_identifier("post-1234") is None
    assert keep_identifier(":r1:") is None
    assert keep_identifier("abc12def") is None
    assert keep_identifier("") is None
    assert keep_identifier(None) is None


def test_clean_html_strips_non_content(root: etree._Element) -> None:
    text = etree.tostring(root, encoding="unicode")
    for gone in ("<script", "<style", "<noscript", "<svg", "<iframe", "<template", "<!--"):
        assert gone not in text
    assert "<meta" not in text and "<link" not in text
    head = root.find("head")
    assert head is not None
    assert [c.tag for c in head] == ["title"]
    assert "fine widget" in text


def test_clean_html_bytes_and_empty() -> None:
    root = clean_html("<p>caf\u00e9</p>".encode("latin-1"), encoding="latin-1")
    assert node_text(root) == "caf\u00e9"
    assert node_text(clean_html("")) == ""
    assert node_text(clean_html(b"")) == ""
    xml = parse_html('<?xml version="1.0" encoding="utf-8"?><html><body>x</body></html>')
    assert node_text(xml) == "x"


def test_step_labels(root: etree._Element) -> None:
    main = root.find(".//main")
    assert main is not None
    assert step_label(main) == "main.product"  # numeric id dropped, hashed class dropped
    header = root.find(".//header")
    assert header is not None
    assert step_label(header) == "header#top.site-header"
    h1 = root.find(".//h1")
    assert h1 is not None
    assert step_of(h1) == Step("h1", None, ("title",), (("itemprop", "name"),))
    assert step_label(h1) == "h1.title[itemprop=name]"
    dd = root.find(".//dd")
    assert dd is not None
    assert step_label(dd) == "dd[data-testid=price]"


def test_paths_have_no_indices_and_repeat(root: etree._Element) -> None:
    paths = [p for _, p in iter_paths(root)]
    assert paths[0] == "html"
    assert "html > body.page > header#top.site-header > nav[role=navigation] > a" in paths
    dts = [p for p in paths if p.endswith("> dt")]
    assert dts == ["html > body.page > main.product > dl.specs > dt"] * 2
    assert not any("[1]" in p or "[2]" in p for p in paths)
    # document order
    assert paths.index("html > head > title") < paths.index("html > body.page")


def test_iter_text_items(root: etree._Element) -> None:
    items = [(i.path, i.text, i.attr) for i in iter_text_items(root)]
    assert ("html > head > title", "Blue Widget", None) in items
    assert (
        "html > body.page > main.product > dl.specs > dd",
        "blue and green",
        None,
    ) in items
    assert ("html > body.page > main.product > p.desc", "A fine widget.", None) in items
    assert ("html > body.page > main.product > p.desc > img @src", "/w.png", "src") in items
    hrefs = [t for p, t, a in items if a == "href"]
    assert hrefs == ["/", "/shop"]


def test_own_text_and_node_text(root: etree._Element) -> None:
    p = root.find(".//p")
    assert p is not None
    assert own_text(p) == "A fine widget."
    assert node_text(p) == "A very fine widget."


def test_parse_step_and_split_path_round_trip() -> None:
    step = Step("dd", "main", ("a", "b"), (("itemprop", "price"), ("role", "cell")))
    assert parse_step(step.label()) == step
    steps, attr = split_path("html > body.page > a.more @href")
    assert [s.label() for s in steps] == ["html", "body.page", "a.more"]
    assert attr == "href"
    steps, attr = split_path("html > body")
    assert attr is None and len(steps) == 2
    with pytest.raises(ValueError):
        parse_step("not a step")


def test_step_xpath_matches_its_element(root: etree._Element) -> None:
    for el, _ in iter_paths(root):
        xp = "//" + step_of(el).xpath()
        assert el in root.xpath(xp)
    h1 = Step("h1", "x", ("title",), (("itemprop", "name"),)).xpath()
    assert h1 == (
        "h1[@id='x'][contains(concat(' ', normalize-space(@class), ' '), ' title ')]"
        "[@itemprop='name']"
    )


def test_attribute_values_with_markup_chars_are_dropped() -> None:
    root = clean_html('<div itemprop="a[b]" role=" main  region ">x</div>')
    div = root.find(".//div")
    assert div is not None
    assert step_label(div) == "div[role=main region]"


def test_clean_etree_matches_clean_html() -> None:
    from ditto.fingerprint.dom import clean_etree

    plain = clean_etree(PAGE)
    rich = clean_html(PAGE)
    assert not isinstance(plain, type(rich))
    assert [p for _, p in iter_paths(plain)] == [p for _, p in iter_paths(rich)]
    assert etree.tostring(plain) == etree.tostring(rich)


def test_unparseable_or_blank_input_gives_empty_document() -> None:
    for data in (b"   ", "  \n ", "<!-- only a comment -->", b"<!-- c -->"):
        root = clean_html(data)
        assert root.tag == "html"
        assert node_text(root) == ""


def test_step_label_fast_path_and_tag_label() -> None:
    from ditto.fingerprint.dom import STEP_ATTRS, tag_label

    assert tag_label("DIV") == "div"
    assert set(STEP_ATTRS) == {"id", "class", "itemprop", "data-testid", "role"}
    root = clean_html('<div href="/x" title="t"><SPAN lang="en">a</SPAN></div>')
    div = root.find(".//div")
    assert div is not None
    assert step_label(div) == "div" == step_of(div).label()
    span = root.find(".//span")
    assert span is not None
    assert step_label(span) == "span"
