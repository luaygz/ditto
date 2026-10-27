"""The v0.2 gate metric: is a known field shown on its template's union page?"""

from __future__ import annotations

from typing import Any

from ditto.analyze.models import Candidate, CandidateTable
from ditto.union.coverage import (
    Annotated,
    FieldPath,
    KnownField,
    annotated_nodes,
    field_coverage,
    field_path,
    page_text_items,
    template_coverage,
    text_matches,
)

DD = "html > body > dl.contact > dd"
NOTE = "html > body > p.note"


def _page(email: str | None, phone: str | None, note: str = "Call us") -> str:
    rows = ""
    if email:
        rows += f"<dt>Email</dt><dd>{email}</dd>"
    if phone:
        rows += f"<dt>Phone</dt><dd>{phone}</dd>"
    return (
        "<html><head><title>Contact</title><script>x()</script></head><body>"
        f'<h1>Ann</h1><dl class="contact">{rows}</dl><p class="note">{note}</p></body></html>'
    )


PAGES = {
    "p1": _page("ann@example.org", None),
    "p2": _page(None, "555-0101"),
    "p3": _page("bo@example.org", "555-0102"),
}
EMAIL = {"p1": ["ann@example.org"], "p3": ["bo@example.org"]}
PHONE = {"p2": ["555-0101"], "p3": ["555-0102"]}


def _cand(cid: str, path: str, cls: str, **kw: Any) -> Candidate:
    return Candidate.model_validate(
        {"id": cid, "path": path, "class": cls, "p": 0.5, "m": 0.1, **kw}
    )


def _table(*, phone_cls: str = "optional") -> CandidateTable:
    return CandidateTable(
        template_id="t_1",
        cluster_size=3,
        n_sampled=3,
        candidates=[
            _cand("c1", "html > body > h1", "mandatory"),
            _cand("c2", DD, "optional", column="Email", table_id="tb0", table_role="kv_value"),
            _cand("c3", DD, phone_cls, column="Phone", table_id="tb0", table_role="kv_value"),
        ],
    )


def _union(*, phone: str | None = "555-0102", email: str = "ann@example.org") -> str:
    parts = [
        '<html><body><!-- c1 · mandatory · on 100% of pages · e.g. Ann --><h1>Ann</h1><dl class="contact">',
        f"<dt>Email</dt><!-- c2 · optional · on 67% of pages · e.g. x --><dd>{email}</dd>",
        "<!-- collapsed: 2 more siblings like the one above (variant 1 of 1) -->",
    ]
    if phone:
        parts.append(f"<dt>Phone</dt><!-- c3 · optional · on 67% of pages --><dd>{phone}</dd>")
    parts.append("</dl></body></html>")
    return "".join(parts)


def _items(key: str) -> list[tuple[str, str]]:
    return page_text_items(PAGES[key])


def _cover(name: str, values: dict[str, list[str]], union: str, table: CandidateTable) -> Any:
    others = {"email": ("Phone",), "phone": ("Email",)}[name]
    (result,) = template_coverage([KnownField(name, values, others)], table, union, _items)
    return result


def test_page_text_items_are_cleaned_structural_paths() -> None:
    items = _items("p3")
    assert (DD, "bo@example.org") in items and (DD, "555-0102") in items
    assert ("html > head > title", "Contact") in items
    assert not any("x()" in text for _, text in items)  # scripts are cleaned away


def test_annotated_nodes_pair_each_annotation_with_its_nodes() -> None:
    nodes = annotated_nodes(
        '<html><body><!-- c4 @href · optional · e.g. /a --><a href="/a">Go</a>'
        "<!-- spliced from another page: c5 --><!-- c5 · list · on 3% --> <i>x</i>"
        "<!-- collapsed: 3 more siblings like the one above (variant 1 of 1) --><i>y</i>"
        "<!-- c7 · list --><i>z</i><b>w</b>"
        "<!-- c6 · optional · on 1% --></body></html>"
    )
    assert nodes == [
        Annotated("c4", "href", ("Go",)),
        Annotated("c5", None, ("x", "y")),  # same-tag siblings up to the next annotation
        Annotated("c7", None, ("z",)),
        Annotated("c6", None, ()),  # annotation with no element after it
    ]
    assert annotated_nodes("   ") == []


def test_text_matches_exact_and_middle_cut_values() -> None:
    values = {"Alpha beta gamma delta", "42"}
    assert text_matches(" 42 ", values)
    assert text_matches("Alpha … delta", values)
    assert not text_matches("Alpha … omega", values)
    assert not text_matches("4", values)
    assert not text_matches("", values)


def test_field_path_is_the_modal_path_of_value_nodes() -> None:
    located = field_path((_items(k), v) for k, v in EMAIL.items())
    assert located == FieldPath(DD, 2, 2, 1)
    assert field_path([(_items("p1"), ["  "])]) == FieldPath(None, 0, 0, 0)


def test_covered_by_its_own_annotated_candidate() -> None:
    result = _cover("email", EMAIL, _union(), _table())
    assert result.covered and result.shown == ["c2"] and result.reason is None
    assert result.to_dict()["path_support"] == "2/2"
    phone = _cover("phone", PHONE, _union(), _table())
    assert phone.covered and phone.shown == ["c3"]


def test_shared_path_is_told_apart_by_label_not_just_by_path() -> None:
    # the phone cell is missing from the union page; the email cell sits at the same path
    result = _cover("phone", PHONE, _union(phone=None), _table(phone_cls="noise"))
    assert not result.covered
    assert result.reason == "not_shown"
    assert "c3 noise" in result.detail and "c2" not in result.detail


def test_shared_path_value_collision_does_not_count() -> None:
    # c2 (labeled Email) shows a value that is also a phone value on another page
    phone = {"p2": ["555-0101"], "p3": ["ann@example.org"]}
    result = _cover("phone", phone, _union(phone=None), _table())
    assert not result.covered and result.reason == "not_shown"


def test_value_mismatch_when_its_candidate_shows_another_value() -> None:
    result = _cover("email", EMAIL, _union(email="someone@else.org"), _table())
    assert not result.covered and result.reason == "value_mismatch"
    assert "someone@else.org" in result.detail


def test_unlocated_and_no_candidate() -> None:
    table = _table()
    missing = _cover("email", {"p1": ["not on the page"]}, _union(), table)
    assert missing.reason == "unlocated" and missing.located.examined == 1
    # the note paragraph has a path but no candidate in the table
    note = field_coverage(
        "note", field_path([(_items("p1"), ["Call us"])]), table, annotated_nodes(_union()), []
    )
    assert note.located.path == NOTE
    assert note.reason == "no_candidate" and not note.covered


def test_label_and_value_sharing_a_path_on_an_unambiguous_path() -> None:
    # the analysis merged "Area:" and its value into one candidate (both spans have the path),
    # and its label hint names the neighboring field: the path alone identifies the field
    path = "html > body > div.area > span"
    page = '<html><body><div class="area"><span>Area:</span><span>90 m2</span></div></body></html>'
    table = CandidateTable(
        template_id="t_1",
        cluster_size=2,
        n_sampled=2,
        candidates=[_cand("c9", path, "optional", label_hint="Type:")],
    )
    union = (
        '<html><body><div class="area"><!-- c9 · optional · on 13% of pages -->'
        "<span>Area:</span><span>75 m2</span></div></body></html>"
    )
    (result,) = template_coverage(
        [KnownField("area", {"p1": ["90 m2"], "p2": ["75 m2"]}, ("Type",))],
        table,
        union,
        lambda key: page_text_items(page),
    )
    assert result.covered and result.shown == ["c9"]
    assert result.located.path == path


def test_template_coverage_locates_on_at_most_max_pages() -> None:
    seen: list[str] = []

    def items(key: str) -> list[tuple[str, str]]:
        seen.append(key)
        return _items(key)

    fields = [KnownField("email", EMAIL, ("Phone",)), KnownField("phone", PHONE, ("Email",))]
    results = template_coverage(fields, _table(), _union(), items, max_pages=1)
    assert [r.covered for r in results] == [True, True]
    assert seen == ["p1", "p2"]  # the first present page (sorted) of each field
