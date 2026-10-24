from __future__ import annotations

from typing import Any

import pytest
from lxml import etree, html

from ditto.specgen.transforms import (
    CAST_TYPES,
    TRANSFORM_DOCS,
    TRANSFORM_NAMES,
    Transform,
    apply_transforms,
    parse_transform,
    to_text,
)

PAGE = html.fromstring(
    """<div>
      <a class="more" href="/staff/a-khoury" data-id="42">  Dr.  Amal
         Khoury </a>
      <img src="/img/a.png" alt="">
      <span class="price">€1.234,56</span>
    </div>"""
)
LINK = PAGE.find(".//a")
IMG = PAGE.find(".//img")
PRICE = PAGE.find(".//span")


@pytest.mark.parametrize(
    ("text", "name", "args"),
    [
        ("strip", "strip", ()),
        ("normalize_ws", "normalize_ws", ()),
        (" normalize_ws() ", "normalize_ws", ()),
        (r"regex(\d+, 0)", "regex", (r"\d+", 0)),
        (r"regex('(\d+)-(\d+)', 2)", "regex", (r"(\d+)-(\d+)", 2)),
        (r"regex('(\d+),(\d+)', 1)", "regex", (r"(\d+),(\d+)", 1)),  # quoted comma
        (r"regex(\d{1,3}(,\d{3})*, 1)", "regex", (r"\d{1,3}(,\d{3})*", 1)),  # unquoted commas
        (r"regex('\d+')", "regex", (r"\d+", 0)),  # group defaults to 0
        (r'regex("it\"s (\w+)", 1)', "regex", ('it"s (\\w+)', 1)),  # escaped quote
        ("split('/', -1)", "split", ("/", -1)),
        ("split(',', 0)", "split", (",", 0)),
        ("split(' | ', 1)", "split", (" | ", 1)),
        ("split( / , 2 )", "split", ("/", 2)),
        ("replace(', ', '; ')", "replace", (", ", "; ")),
        ("replace('$', '')", "replace", ("$", "")),
        ("replace('it\\'s', 'its')", "replace", ("it's", "its")),
        ("replace('a\\\\b', 'x')", "replace", ("a\\b", "x")),
        ("cast(int)", "cast", ("int",)),
        ("cast( 'float' )", "cast", ("float",)),
        ("cast(date)", "cast", ("date",)),
        ("attr(href)", "attr", ("href",)),
        ("attr('data-id')", "attr", ("data-id",)),
        ("attr(xml:lang)", "attr", ("xml:lang",)),
    ],
)
def test_parse_transform_accepts(text: str, name: str, args: tuple[Any, ...]) -> None:
    t = parse_transform(text)
    assert t == Transform(name, args)
    assert parse_transform(str(t)) == t  # canonical form round-trips


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("lower", "unknown transform 'lower'"),
        ("Cast(int)", "unknown transform 'Cast'"),
        ("normalize-ws", "unknown transform"),
        ("json_path($.a)", "unknown transform"),
        ("strip extra", "not a transform call"),
        ("regex('a', 1) junk", "not a transform call"),
        ("", "not a transform call"),
        ("strip(x)", "takes no arguments"),
        ("regex", "needs arguments"),
        ("regex()", "needs arguments"),
        ("regex('', 0)", "pattern is empty"),
        ("regex('(', 0)", "invalid regex"),
        ("regex('(a)', 2)", "1 groups"),
        ("regex('a', -1)", "groups"),
        ("regex('a', x)", "regex group must be an integer"),
        ("regex('a', 1, 2)", "regex takes"),
        ("split('/')", "split takes"),
        ("split(, 0)", "separator is empty"),
        ("split(',', one)", "split index must be an integer"),
        ("split(',', )", "split index must be an integer"),
        ("replace('a')", "replace takes"),
        ("replace('', 'b')", "non-empty old"),
        ("cast(bool)", "cast takes one of"),
        ("cast(int, float)", "cast takes one of"),
        ("attr(1x)", "attr takes one attribute name"),
        ("attr(href, src)", "attr takes one attribute name"),
        ("split('/, 0)", "unterminated"),
        ("split('/'x, 0)", "unexpected text after a quoted argument"),
    ],
)
def test_parse_transform_rejects(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        parse_transform(text)


def test_closed_set_is_documented() -> None:
    assert set(TRANSFORM_DOCS) == set(TRANSFORM_NAMES)
    assert CAST_TYPES == ("int", "float", "date")


@pytest.mark.parametrize(
    ("value", "chain", "expected"),
    [
        # regex groups
        ("Order #1234-56", [r"regex('#(\d+)-(\d+)', 2)"], "56"),
        ("Order #1234-56", [r"regex('#(\d+)', 0)"], "#1234"),
        ("no digits", [r"regex('\d+', 0)"], None),
        ("a-", [r"regex('a(b)?', 1)"], None),  # optional group that did not take part
        ("1,234,567 views", [r"regex(\d{1,3}(,\d{3})*, 0)", "cast(int)"], 1234567),
        # quoted arguments with commas
        ("Beirut, Lebanon, Earth", ["split(',', 1)"], "Lebanon"),
        ("a, b, c", ["replace(', ', ';')"], "a;b;c"),
        ("10,20", [r"regex('(\d+),(\d+)', 2)"], "20"),
        # split, negative index and out of range
        ("/staff/dept/a-khoury", ["split('/', -1)"], "a-khoury"),
        ("/staff/dept/a-khoury", ["split('/', -2)"], "dept"),
        ("a|b", ["split('|', 5)"], None),
        ("a|b", ["split('|', -3)"], None),
        # strip / normalize / replace
        ("  hi  ", ["strip"], "hi"),
        ("a \n\t b\u00a0c", ["normalize_ws"], "a b c"),
        ("$12.00", ["replace('$', '')"], "12.00"),
        ("N/A", ["replace('N/A', '')"], None),  # empty result is null
        # casts
        ("1,234.50", ["cast(int)"], 1234),
        ("1,234.50", ["cast(float)"], 1234.5),
        ("€1.234,56", ["cast(float)"], 1234.56),
        ("€1.234,56", ["cast(int)"], 1234),
        ("March 3, 2021", ["cast(date)"], "2021-03-03"),
        ("March 3, 2021", ["cast(int)"], 3),
        ("1,234.50", ["cast(date)"], None),
        ("Posted 12 Mar 2021", ["cast(date)"], "2021-03-12"),
        ("free", ["cast(float)"], None),
        ("1,234.50", ["cast(float)", "cast(int)"], 1234),
        ("1,234.50", ["cast(int)", "strip"], "1234"),  # string transforms read numbers as text
        # chains stop at the first null
        ("abc", [r"regex('\d+', 0)", "cast(int)"], None),
        # final value is trimmed, empty is null
        ("   ", [], None),
        ("  x ", [], "x"),
    ],
)
def test_apply_transforms_on_strings(value: str, chain: list[str], expected: Any) -> None:
    assert apply_transforms(value, chain) == expected


def test_attr_on_element_and_on_string() -> None:
    assert apply_transforms(LINK, ["attr(href)"]) == "/staff/a-khoury"
    assert apply_transforms(LINK, ["attr('data-id')", "cast(int)"]) == 42
    assert apply_transforms(LINK, ["attr(title)"]) is None  # missing attribute
    assert apply_transforms(IMG, ["attr(alt)"]) is None  # empty attribute
    assert apply_transforms("/staff/a-khoury", ["attr(href)"]) is None  # not an element
    assert apply_transforms(LINK, ["normalize_ws", "attr(href)"]) is None  # text by then
    assert apply_transforms(LINK, ["attr(href)", "split('/', -1)"]) == "a-khoury"


def test_elements_read_as_normalized_text() -> None:
    assert apply_transforms(LINK, []) == "Dr. Amal Khoury"
    assert apply_transforms(LINK, ["regex('Dr. (.+)', 1)"]) == "Amal Khoury"
    assert apply_transforms(PRICE, ["cast(float)"]) == 1234.56
    assert apply_transforms(IMG, []) is None  # element without text


def test_xpath_scalar_results() -> None:
    assert apply_transforms(3.0, []) == "3"
    assert apply_transforms(2.5, ["cast(float)"]) == 2.5
    assert apply_transforms(True, []) == "true"
    assert apply_transforms(None, ["strip"]) is None


def test_smart_strings_from_xpath() -> None:
    href = PAGE.xpath("//a/@href")[0]
    assert apply_transforms(href, ["split('/', 1)"]) == "staff"
    text = PAGE.xpath("//a/text()")[0]
    assert apply_transforms(text, []) == "Dr.  Amal\n         Khoury"


def test_unknown_transform_fails_at_apply_time() -> None:
    with pytest.raises(ValueError, match="unknown transform"):
        apply_transforms("x", ["upper"])


def test_to_text() -> None:
    assert to_text(None) is None
    assert to_text(LINK) == "Dr. Amal Khoury"
    assert to_text(False) == "false"
    assert to_text(1.5) == "1.5" and to_text(7.0) == "7"
    assert to_text(12) == "12"
    assert to_text(etree.fromstring("<p>a<b>b</b> c</p>")) == "ab c"
