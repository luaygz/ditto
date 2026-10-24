from __future__ import annotations

import pytest

from ditto.specgen.lint import (
    is_positional_only,
    lint_selector,
    location_paths,
    split_steps,
)

POSITIONAL = "purely positional path"

# Absolute selectors that must pass: the brief's own examples first, then real-world shapes.
ACCEPT_ABSOLUTE = [
    "//h1[@itemprop='name']",
    "//header//h1",
    "//dt[normalize-space()='Email']/following-sibling::dd[1]//a",
    "//a[starts-with(@href,'mailto:')]",
    "//table[contains(@class,'pubs')]//tr[td]",
    "//dl[@class='profile']/dt",
    "//div[@id='main']/div[2]/span",  # positional step next to an anchored one
    "//*[@id='price']",
    "//meta[@itemprop='price']/@content",
    "//span[@data-testid='sku']/text()",
    "//title",
    "//h1/text()",
    "(//div[@class='x']//a)[1]",
    "//div[contains(concat(' ', normalize-space(@class), ' '), ' price ')]",
    "//div[contains(@class,'price') and not(contains(@class,'old'))]",
    "//a[@x and (@y or @z)]",  # 'and (' is an operator, not a function call
    "//li[position() mod 2 = 1][@class='item']",
    "//td[normalize-space(translate(., 'ABC', 'abc'))='price:']/following-sibling::td[1]",
    "//th[normalize-space()='Price']/../td",
    "//h1[@itemprop='name'] | //header//h1",
    "//*[local-name()='h1'][string-length(normalize-space()) > 0]",
    "//a[@x = 'and (']",  # function-looking text inside a string literal
    "//span[@class='price'][substring-after(., '$') != '']",
    "//ul[@class='tags']/li[last()]",
    "//span[contains(., 'data')][@class='x']",
]

REJECT_ABSOLUTE = [
    ("", "empty selector"),
    ("//h1[", "does not compile"),
    ("//a except //b", "does not compile"),
    ("//a[@x=('a','b')]", "does not compile"),
    ("//a[matches(., 'x')]", "matches() which is not XPath 1.0"),
    ("//a[lower-case(.)='x']", "lower-case() which is not XPath 1.0"),
    ("//a[upper-case(.)='X']", "upper-case() which is not XPath 1.0"),
    ("//a[ends-with(@href,'.pdf')]", "ends-with() which is not XPath 1.0"),
    ("//a[string-join(., '')]", "non-XPath-1.0 function string-join()"),
    ("//a[Contains(., 'x')]", "non-XPath-1.0 function Contains()"),
    ("//a[current()]", "non-XPath-1.0 function current()"),
    ("//a[re:test(., 'x')]", "prefixed function re:test()"),
    ("//x:a", "namespace prefix"),
    ("//a[$x]", "variable $x"),
    ("//comment()", "comment()"),
    ("//div[comment()]", "comment()"),
    ("//div[@data-x-id='c17']", "data-x-"),
    ("//div[@class='data-x-c17']", "data-x-"),
    ("count(//li)", "absolute selector must be a location path"),
    ("normalize-space(//h1)", "absolute selector must be a location path"),
    ("./h1", "absolute selector must be a location path"),
    ("/html/body/div[2]/h1", POSITIONAL),
    ("/html/body//h1", POSITIONAL),  # rooted at '/'
    ("/html/body/div[3]/a/@href", POSITIONAL),
    ("//div", POSITIONAL),
    ("//div[3]", POSITIONAL),
    ("//li[position()=2]", POSITIONAL),
    ("//li[last()]", POSITIONAL),
    ("//table//tr[2]/td[3]", POSITIONAL),
    ("(//h1)[1]", POSITIONAL),
    ("//h1 | //div", POSITIONAL),  # every union branch needs an anchor
    ("//p", POSITIONAL),
    ("//h2/following-sibling::p", POSITIONAL),
    ("//a/@href", POSITIONAL),
    ("//*", POSITIONAL),
    ("//body/div/descendant::span", POSITIONAL),
]


@pytest.mark.parametrize("xpath", ACCEPT_ABSOLUTE)
def test_accepts_absolute(xpath: str) -> None:
    assert lint_selector(xpath) == []


@pytest.mark.parametrize(("xpath", "message"), REJECT_ABSOLUTE)
def test_rejects_absolute(xpath: str, message: str) -> None:
    errors = lint_selector(xpath)
    assert any(message in e for e in errors), errors


ACCEPT_RELATIVE = [
    ".",
    "./td[count(ancestor::table//th[normalize-space()='Title']/preceding-sibling::th)+1]",
    "./td[count(../../tr/th[normalize-space()='Year']/preceding-sibling::th)+1]//a/@href",
    ".//a/@href",
    ".//span[@class='price']",
    "./h3",
    "./dd",
    "./following-sibling::dd[@class='value']",
]
REJECT_RELATIVE = [
    ("./td[2]", POSITIONAL),
    ("./td", POSITIONAL),
    ("./td[position()=3]", POSITIONAL),
    ("./td[last()]", POSITIONAL),
    ("./*[1]", POSITIONAL),
    ("./following-sibling::dd[1]", POSITIONAL),
    ("td[1]", "relative selector must start with '.'"),
    ("//td[@class='x']", "relative selector must start with '.'"),
    (".//a | //b[@id='x']", "relative selector must start with '.'"),
]


@pytest.mark.parametrize("xpath", ACCEPT_RELATIVE)
def test_accepts_relative(xpath: str) -> None:
    assert lint_selector(xpath, relative=True) == []


@pytest.mark.parametrize(("xpath", "message"), REJECT_RELATIVE)
def test_rejects_relative(xpath: str, message: str) -> None:
    errors = lint_selector(xpath, relative=True)
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize(
    "xpath", ["./td[2]", "./td", "./following-sibling::dd[1]", "./*[3]/text()", "."]
)
def test_allow_positional_for_headerless_and_key_value_parts(xpath: str) -> None:
    assert lint_selector(xpath, relative=True, allow_positional=True) == []


def test_allow_positional_still_enforces_other_rules() -> None:
    errors = lint_selector("./td[lower-case(.)='x']", relative=True, allow_positional=True)
    assert errors == ["uses lower-case() which is not XPath 1.0"]


def test_errors_are_deduplicated_and_returned_as_fresh_lists() -> None:
    errors = lint_selector("//a[matches(., 'x')][matches(., 'y')]")
    assert errors == ["uses matches() which is not XPath 1.0"]
    errors.append("mutated")
    assert lint_selector("//a[matches(., 'x')][matches(., 'y')]") == [
        "uses matches() which is not XPath 1.0"
    ]


def test_split_steps_respects_brackets_and_strings() -> None:
    assert split_steps("//dt[normalize-space()='a/b']/following-sibling::dd[1]") == [
        "",
        "",
        "dt[normalize-space()='a/b']",
        "following-sibling::dd[1]",
    ]
    assert split_steps("./td[count(a/b)+1]") == [".", "td[count(a/b)+1]"]


def test_location_paths_unfolds_unions_and_groups() -> None:
    assert location_paths("(//a | //b)[1]") == ["//a[1]", "//b[1]"]
    assert location_paths("//a[@x='|'] | //b") == ["//a[@x='|']", "//b"]
    assert location_paths("(//h1)") == ["//h1"]


@pytest.mark.parametrize(
    ("xpath", "expected"),
    [
        ("//header//h1", False),
        ("//h1[1]", True),
        ("/html/body/h1", True),
        ("//span", True),
        ("//span[@class='x']", False),
        ("//div[@id='a']/span", False),
        ("./td[2]", True),
        (".//a", False),
        ("//@href", False),
        ("id('main')//span", False),
    ],
)
def test_is_positional_only(xpath: str, expected: bool) -> None:
    assert is_positional_only(xpath) is expected
