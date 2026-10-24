from __future__ import annotations

import pytest

from ditto.specgen.typecheck import (
    check_value,
    has_currency,
    infer_value_type,
    is_date,
    number_string,
    parse_date,
    parse_int,
    parse_number,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234.50", 1234.5),
        ("€1.234,56", 1234.56),
        ("1 234,56", 1234.56),
        ("1\u00a0234,56 €", 1234.56),
        ("1\u202f234\u202f567,89", 1234567.89),
        ("CHF 1'234.50", 1234.5),
        ("Rs. 1,23,456", 123456.0),
        ("1,234,567.89", 1234567.89),
        ("1.234.567", 1234567.0),
        ("0,5", 0.5),
        ("1,5 kg", 1.5),
        ("1,234", 1234.0),
        ("1.234", 1.234),  # a lone '.' is a decimal mark
        ("Price: 25 USD, was 30", 25.0),
        ("-3", -3.0),
        ("\u22123.5", -3.5),
        ("45%", 45.0),
        ("(1,234)", 1234.0),
        ("2019-2020", 2019.0),
        ("4.5 out of 5 stars", 4.5),
        ("N/A", None),
        ("", None),
        ("v1.2.3", None),  # inconsistent grouping is not a number
        ("1,23.4,5", None),
    ],
)
def test_parse_number(text: str, expected: float | None) -> None:
    assert parse_number(text) == expected


def test_parse_int_truncates_and_keeps_long_digit_strings_exact() -> None:
    assert parse_int("1,234.99") == 1234
    assert parse_int("-7.9") == -7
    assert parse_int("12345678901234567890") == 12345678901234567890
    assert parse_int("none") is None
    assert number_string("$ 1,299") == "1299"


@pytest.mark.parametrize(
    ("text", "iso"),
    [
        ("March 3, 2021", "2021-03-03"),
        ("12 Mar 2021", "2021-03-12"),
        ("12 March 2021, 10:30", "2021-03-12"),
        ("2021-03-12T10:00:00Z", "2021-03-12"),
        ("2021-03-12T10:00:00.123+02:00", "2021-03-12"),
        ("2021-03-12 10:00", "2021-03-12"),
        ("2021/3/4", "2021-03-04"),
        ("03/12/2021", "2021-03-12"),  # '/' reads month first
        ("13/03/2021", "2021-03-13"),  # ... unless the first part cannot be a month
        ("12.03.2021", "2021-03-12"),  # '.' reads day first
        ("12/03/21", "2021-12-03"),
        ("12-Mar-2021", "2021-03-12"),
        ("Mar. 3, 2021", "2021-03-03"),
        ("Sept 3 2021", "2021-09-03"),
        ("Tuesday, March 3rd, 2021", "2021-03-03"),
        ("the 3rd of March, 2021", "2021-03-03"),
        ("3 de marzo de 2021", "2021-03-03"),
        ("3. März 2021", "2021-03-03"),
        ("1er mars 2021", "2021-03-01"),
        ("Published on March 3, 2021 by A. Khoury", "2021-03-03"),
        ("Updated 2020-01-01, first posted March 3, 2019", "2020-01-01"),
        ("March 2021", "2021-03"),
        ("2021-03", "2021-03"),
        ("2021", "2021"),
        ("2021-02-30", None),
        ("3 days ago", None),
        ("version 1.2.3", None),
        ("", None),
    ],
)
def test_parse_date(text: str, iso: str | None) -> None:
    assert parse_date(text) == iso


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("12 Mar 2021", True),
        ("Tuesday, 12 March 2021", True),
        ("12 Mar 2021 10:30 AM", True),
        ("2021-03-12T10:00:00Z", True),
        ("March 2021", True),
        ("on 2021-03-12 at 10:00 UTC", True),
        ("2021", False),  # a bare year is a number
        ("Published March 3, 2021", False),
        ("hello", False),
    ],
)
def test_is_date_is_whole_value(text: str, ok: bool) -> None:
    assert is_date(text) is ok


VALID = {
    "email": [
        "a.khoury@example.org",
        "mailto:a.khoury@example.org",
        "mailto:press@example.org?subject=Hello",
        "<l.haddad@example.org>",
        "first.last+tag@sub.example.co.uk",
        "José.Pérez@ejemplo.es",
    ],
    "url": [
        "https://example.org/staff/a-khoury?tab=pubs#top",
        "http://localhost.example:8080/x",
        "//cdn.example.org/img/logo.png",
        "/staff/a-khoury",
        "./next",
        "../products/42",
        "staff/a-khoury.html",
        "page.php?id=3",
        "example.org",
        "www.example.org/path",
        "?page=2",
        "https://例え.jp/パス",
    ],
    "number": ["1,234", "1.234,56", "-3.5", "\u22123", "45%", "45 %", "1 234", "+7", "007"],
    "price": [
        "$1,299.99",
        "€ 1.234,56",
        "1.234,56 €",
        "£12",
        "USD 25.00",
        "25.00 USD",
        "Rs. 1,23,456",
        "₹1,23,456",
        "CHF 1'234.50",
        "R$ 10,00",
        "¥1,000",
        "1 299 kr",
        "$25 - $40",
        "From $25",
        "25,-",
        "12,99 €",
        "1299",
    ],
    "date": ["12 Mar 2021", "March 3, 2021", "2021-03-12T10:00:00Z", "03/12/2021"],
    "text": ["anything", "x"],
}
INVALID = {
    "email": ["Email: a@b.org", "a@b", "user@localhost", "a @b.org", "a@1.2", "@example.org"],
    "url": [
        "#top",
        "javascript:void(0)",
        "mailto:a@b.org",
        "tel:+9611350000",
        "Mr.Smith",
        "hello world",
        "2021/03/12",
        "1.5",
        "https://exa mple.org",
    ],
    "number": ["1e5", "12 kg", "1.2.3", "abc", "1,23.4,5"],
    "price": ["Free", "Call for price", "$", "about twenty dollars", "1.2.3 €"],
    "date": ["2021", "yesterday", "Q1 2021", "20210312"],
    "text": ["", "   "],
}


@pytest.mark.parametrize(("kind", "value"), [(k, v) for k, vs in VALID.items() for v in vs])
def test_check_value_accepts(kind: str, value: str) -> None:
    assert check_value(kind, value)


@pytest.mark.parametrize(("kind", "value"), [(k, v) for k, vs in INVALID.items() for v in vs])
def test_check_value_rejects(kind: str, value: str) -> None:
    assert not check_value(kind, value)


def test_check_value_non_strings() -> None:
    assert not check_value("text", None)
    assert check_value("number", 3) and check_value("price", 4.5) and check_value(None, 0)
    assert not check_value("date", 2021) and not check_value("email", 1)
    assert check_value("text", True) and not check_value("number", True)
    assert check_value("unknown-kind", "x")  # unknown checks accept any value


def test_has_currency() -> None:
    assert has_currency("€12") and has_currency("12 USD") and has_currency("Rs. 5")
    assert not has_currency("12.50") and not has_currency("USDA certified")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["a@b.org", "mailto:c@d.org"], "email"),
        (["/a/b", "/c", "https://x.org"], "url"),
        (["12 Mar 2021", "2021-03-14"], "date"),
        (["$1", "$2", "3"], "price"),
        (["1", "2", "3"], "number"),
        ([1, 2.5], "number"),
        (["2019", "2021"], "number"),  # years are numbers, not dates
        (["2021/03/12", "2021/04/01"], "date"),
        (["hello", "x"], "text"),
        (["1", "2", "three"], "text"),
        ([], None),
        (["", "  ", None], None),
    ],
)
def test_infer_value_type(values: list[object], expected: str | None) -> None:
    assert infer_value_type(values) == expected


def test_infer_value_type_threshold() -> None:
    values = ["1", "2", "3", "x"]
    assert infer_value_type(values) == "text"
    assert infer_value_type(values, threshold=0.75) == "number"
