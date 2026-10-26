"""Path-induced fallback selectors: anchored, lint-clean, and matching the analysis node."""

from __future__ import annotations

from pathlib import Path

from verify_sitekit import build_run, cand, staff_page, staff_table, text_path

from ditto.extract.runner import evaluate, field_value
from ditto.fingerprint.dom import clean_html, split_path
from ditto.specgen.lint import lint_selector
from ditto.specgen.models import KeyValueField, ListField, RecordListField, ScalarField
from ditto.specgen.transforms import to_text
from ditto.verify import induced_field, path_induced_selectors
from ditto.verify.fallback import combined_selectors, induced_variants, relative_selector

PAGE = staff_page(3, email=True, tags=2, pubs=3)


def _texts(xpath: str, html: str = PAGE) -> list[str | None]:
    return [to_text(x) for x in evaluate(xpath, clean_html(html))]


def test_two_anchored_variants_pass_lint_and_match() -> None:
    c = cand("c0", text_path(PAGE, "Person 003"), 1.0)
    sels = path_induced_selectors(c)
    assert len(sels) == 2
    for sel in sels:
        assert lint_selector(sel) == []
        assert _texts(sel) == ["Person 003"]
    assert sels[0] == "//h1[@itemprop='name']"
    li = cand("c4", text_path(PAGE, "topic3-0"), 1.0, "list")
    tag_sels = path_induced_selectors(li)
    assert tag_sels[0].startswith(
        "//ul[contains(concat(' ', normalize-space(@class), ' '), ' tags ')]"
    )
    assert all(_texts(s) == ["topic3-0", "topic3-1"] for s in tag_sels)


def test_attribute_path_selects_the_attribute() -> None:
    c = cand("c9", text_path(PAGE, "mailto:p3@example.org", attr="href"), 0.5)
    sels = path_induced_selectors(c)
    assert sels and all(s.endswith("/@href") and not lint_selector(s) for s in sels)
    assert _texts(sels[0]) == ["mailto:p3@example.org"]


def test_unanchored_generic_path_uses_label_or_nothing() -> None:
    html = "<html><body><div><p>Label</p><p>Value 1</p></div></body></html>"
    path = text_path(html, "Value 1")
    assert path_induced_selectors(cand("c1", path, 1.0)) == []
    labelled = path_induced_selectors(cand("c1", path, 1.0, label_hint="Label"))
    assert labelled == ["//*[normalize-space()='Label']/following-sibling::p[1]"]
    assert _texts(labelled[0], html) == ["Value 1"]


def test_relative_selector_child_and_sibling() -> None:
    row, _ = split_path("html > body > dl.x > dt")
    dd, _ = split_path("html > body > dl.x > dd")
    inner, _ = split_path("html > body > dl.x > dt > b")
    other, _ = split_path("html > body > ul > li")
    assert relative_selector(row, row, None) == "."
    assert relative_selector(row, inner, "href") == "./b/@href"
    assert relative_selector(row, dd, None) == "./following-sibling::dd[1]"
    assert relative_selector(row, other, None) is None
    assert induced_variants([]) == []


def test_combined_selectors_join_variants() -> None:
    b_page = staff_page(4, variant="b")
    a = cand("c0", text_path(PAGE, "Person 003"), 0.5)
    b = cand("c1", text_path(b_page, "Person 004"), 0.5)
    sels = combined_selectors([a, b])
    assert len(sels) == 2 and all(" | " in s for s in sels)
    assert _texts(sels[0], b_page) == ["Person 004"]
    assert _texts(sels[0], PAGE) == ["Person 003"]


def test_induced_fields_from_analysis(tmp_path: Path) -> None:
    run = build_run(tmp_path, [("https://example.org/staff/a", PAGE)])
    table = staff_table(run, PAGE)
    name = induced_field(["c0"], table, description="person name")
    assert isinstance(name, ScalarField) and name.description == "person name"
    tags = induced_field(["c4"], table)
    assert isinstance(tags, ListField)
    tree = clean_html(PAGE)
    assert field_value(tags, tree) == ["topic3-0", "topic3-1"]
    pubs = induced_field(["t0"], table, field_type="record_list")
    assert isinstance(pubs, RecordListField) and pubs.column_source == "header_row"
    rows = field_value(pubs, tree)
    assert rows == [{"Title": f"Paper 3.{k}", "Year": str(2000 + (3 + k) % 20)} for k in range(3)]
    details = induced_field(["c6"], table, field_type="key_value")
    assert isinstance(details, KeyValueField)
    assert field_value(details, tree) == {
        "Email": "p3@example.org",
        "Phone": "+961 1 003 021",
        "Office": "Room 3",
    }
    assert induced_field(["c404"], table) is None
    assert field_value(name, tree) == "Person 003"
