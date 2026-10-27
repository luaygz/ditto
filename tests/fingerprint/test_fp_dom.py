from __future__ import annotations

import pytest
from lxml import etree

from ditto.fingerprint.dom import (
    Step,
    canonical_classes,
    clean_etree,
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


def _xml(root: etree._Element) -> str:
    return etree.tostring(root.getroottree(), encoding="unicode")


@pytest.mark.parametrize(
    ("data", "encoding"),
    [
        ("<p>caf\u00e9 \u20ac \U0001f600</p>".encode(), "utf-8"),
        (b"\xef\xbb\xbf<p>caf\xc3\xa9</p>", "utf-8-sig"),
        ("<p>caf\u00e9</p>".encode(), "utf-8-sig"),  # utf-8-sig without a BOM
        ("<p>Cr\u00e8me \u2019s</p>".encode("cp1252"), "cp1252"),
        (b'<?xml version="1.0" encoding="utf-8"?>\n<html><body>x \xc3\xa9</body></html>', "utf8"),
        ('\u00a0<?xml version="1.0"?><p>x</p>'.encode(), "utf-8"),
        (b" \n\x1c<?xml version='1.0'?><p>x</p>", "utf-8"),
        (b"<p>caf\xe9 \xff ok</p>", "utf-8"),  # invalid UTF-8: decoded with replacement
        (b'<meta charset="iso-8859-1"><p>caf\xc3\xa9</p>', "utf-8"),
        (b" \t\r\n", "utf-8"),
        ("\u00a0\u3000".encode(), "utf-8"),
        (b"<!-- only a comment -->", "utf-8"),
        (b"\xef\xbb\xbf\xef\xbb\xbf<p>x</p>", "utf-8-sig"),
    ],
)
def test_bytes_parse_matches_decoded_text(data: bytes, encoding: str) -> None:
    want = _xml(clean_html(data.decode(encoding, errors="replace")))
    assert _xml(clean_etree(data, encoding)) == want
    assert _xml(clean_html(data, encoding)) == want


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", None])
def test_xml_invalid_characters_are_removed(encoding: str | None) -> None:
    page = '<p title="O\x1bToole\x00 a\x0cb">x\x1by\x00z\x0bw \ufffe\uffff</p>'
    data = page.encode("utf-8" if encoding is None else encoding, errors="ignore")
    for root in (clean_html(data, encoding), clean_html(page)):
        p = root.find(".//p")
        assert p is not None
        title, text = p.get("title"), p.text
        assert title == "OToole\ufffd a b"  # lxml turns NUL into U+FFFD itself
        assert text is not None and text.startswith("xy\ufffdz w")
        assert "\ufffe" not in text and "\uffff" not in text
        p.set("title", title)  # setters accept every value of the cleaned tree
        p.text = text


def test_lone_surrogates_are_dropped() -> None:
    root = clean_html('<p title="a\udc80b">c\ud800d</p>')
    p = root.find(".//p")
    assert p is not None and p.get("title") == "ab" and p.text == "cd"


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", None])
def test_character_references_to_xml_invalid_characters_are_removed(
    encoding: str | None,
) -> None:
    # SWDE movie-amctv: <a title="Biography for actor Peter O&#27;Toole">
    page = (
        '<a title="Peter O&#27;Toole &#x1B;x&#0000027y">O&#27Toole &#11;w&#xc;v</a>'
        "<p>a&#xFFFE;b&#65535;c &#0; &#xD800; &#1114112; &#39;&#160;&#x2019;&#13;</p>"
    )
    data = page.encode("utf-8") if encoding is None else page.encode(encoding)
    for root in (clean_html(data, encoding), clean_html(page), clean_etree(data, encoding)):
        a, p = root.find(".//a"), root.find(".//p")
        assert a is not None and p is not None
        assert a.get("title") == "Peter OToole xy"
        assert a.text == "OToole  w v"  # whitespace controls become spaces
        assert p.text == "abc \ufffd \ufffd \ufffd '\u00a0\u2019\r"
        a.set("title", a.get("title") or "")  # setters accept every value of the cleaned tree
        p.text = p.text


def test_valid_character_references_leave_the_tree_alone() -> None:
    from ditto.fingerprint.dom import _refers_to_forbidden

    assert not _refers_to_forbidden(b"&#39; &#x27; &#160; &#8217; &#9; &#xA; &#13; &#x10FFFF;")
    assert not _refers_to_forbidden(b"&#; &#x; &#xg; &amp;#27;")
    for ref in (b"&#27", b"&#x1b", b"&#X1B;", b"&#000011;", b"&#xFFFE;", b"&#65535;"):
        assert _refers_to_forbidden(b"a " + ref + b" b"), ref
    # code points libxml2 turns into U+FFFD count too (a false alarm costs one tree pass)
    for ref in (b"&#0;", b"&#xD800;", b"&#1114112;", b"&#x123456789;", b"&#99999999999;"):
        assert _refers_to_forbidden(ref), ref
