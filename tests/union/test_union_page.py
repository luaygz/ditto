"""build_union_page end to end on real HTML snippets."""

from __future__ import annotations

from typing import Any

from lxml import html as lhtml
from union_fixtures import PUBS, P, cand, profile_page

from ditto.analyze.models import CandidateTable
from ditto.union import UnionPage, build_union_page, count_tokens


def _pages() -> dict[int, str]:
    return {
        0: profile_page("Ann Lee", pubs=["A", "B", "C"]),
        1: profile_page("Bo Chen", email="bo@example.org", pubs=["D", "E"]),
        2: profile_page(
            "Cy Diaz",
            specs=[("Department", "HR"), ("Fax", "555-0199"), ("Office", "C3")],
            awards=["Prize A", "Prize B"],
        ),
        3: profile_page("Di Ek"),
    }


def _build(table: CandidateTable, pages: dict[int, str], **kw: Any) -> UnionPage:
    return build_union_page(table, pages, min_tokens=0, **kw)


def _kids(el: Any) -> list[Any]:
    """Element children (the union page also holds annotation comments)."""
    return [c for c in el if isinstance(c.tag, str)]


def test_count_tokens_uses_o200k() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2
    assert count_tokens("<|endoftext|>") > 1  # special tokens count as plain text


def test_base_is_page_with_most_field_candidates(table: CandidateTable) -> None:
    page = _build(table, _pages())
    # page 2 holds title, name, phone, 3 kv values, pubs and awards (8); page 1 holds 7
    assert page.base_page_id == 2
    assert page.donor_page_ids == [1]
    assert page.spliced == ["c3"]  # the email text; its @href comes with the same node
    assert page.missing == []
    assert page.shown == [f"c{i}" for i in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)]
    assert page.template_id == "t_test"
    assert page.tokens == count_tokens(page.html)


def test_splice_optional_field_lands_at_aligned_position_with_label(
    table: CandidateTable,
) -> None:
    table.candidates = [c for c in table.candidates if c.id != "c4"]
    pages = {
        0: profile_page(
            "Ann Lee",
            specs=[("Department", "HR"), ("Fax", "1"), ("Office", "C3")],
            awards=["Prize"],
        ),
        1: profile_page("Bo Chen", email="bo@example.org"),
    }
    page = _build(table, pages)
    assert page.base_page_id == 0
    assert page.spliced == ["c3"]
    root = lhtml.fromstring(page.html)
    contact = root.find(".//div[@class='contact']")
    assert contact is not None
    kids = [(c.tag, c.get("class"), c.text_content()) for c in _kids(contact)]
    assert kids == [
        ("span", "label", "Phone:"),
        ("span", "phone", "555-0100"),
        ("span", "label", "Email:"),
        ("a", "email", "bo@example.org"),
    ]
    # the spliced marker precedes the copied label; the annotation precedes the node
    assert (
        '<span class="phone">555-0100</span><!-- spliced from another page: c3 -->'
        '<span class="label">Email:</span>'
        "<!-- c3 · optional · on 41% of pages · e.g. a.khoury@example.org -->"
        '<a class="email" href="mailto:bo@example.org">' in page.html
    )


def test_splice_key_value_row_between_shared_keys(table: CandidateTable) -> None:
    pages = {
        0: profile_page("Ann Lee", email="ann@example.org", pubs=["A"]),
        1: profile_page("Bo Chen", specs=[("Department", "HR"), ("Fax", "555"), ("Office", "C")]),
    }
    page = _build(table, pages)
    assert page.base_page_id == 0
    assert page.spliced == ["c8"]
    root = lhtml.fromstring(page.html)
    dl = root.find(".//dl")
    assert dl is not None
    texts = [c.text_content() for c in _kids(dl)]
    assert texts == ["Department", "Sales", "Fax", "555", "Office", "B12"]
    assert "<!-- spliced from another page: c8 --><dt>Fax</dt>" in page.html


def test_splice_creates_missing_ancestors(table: CandidateTable) -> None:
    pages = {
        0: profile_page("Ann Lee", email="a@x.org", pubs=["A", "B"]),
        1: profile_page("Bo Chen", awards=["Prize A", "Prize B"]),
    }
    page = _build(table, pages)
    assert page.spliced == ["c10"]
    root = lhtml.fromstring(page.html)
    awards = root.find(".//main/div[@class='awards']")
    assert awards is not None
    # the ul ancestor is created, the label h3 comes along, one award is copied
    assert [c.tag for c in _kids(awards)] == ["h3", "ul"]
    assert [li.text_content() for li in awards.iter("li")] == ["Prize A"]
    assert '<!-- spliced from another page: c10 --><div class="awards">' in page.html


def test_annotation_format(table: CandidateTable) -> None:
    page = _build(table, _pages())
    assert "<!-- c3 · optional · on 41% of pages · e.g. a.khoury@example.org -->" in page.html
    assert (
        "<!-- c4 @href · optional · on 41% of pages · e.g. mailto:a.khoury@example.org -->"
        in page.html
    )
    assert "<!-- c1 · mandatory · on 100% of pages · e.g. Ann Lee -->" in page.html
    assert "<!-- c2 · mandatory · on 100% of pages -->" in page.html
    # every annotation sits immediately before its node
    assert '<!-- c1 · mandatory · on 100% of pages · e.g. Ann Lee --><h1 class="name">' in (
        page.html
    )
    # labels and boilerplate are not annotated
    assert "c5 ·" not in page.html and "c16 ·" not in page.html


def test_no_internal_marks_and_no_boilerplate(table: CandidateTable) -> None:
    page = _build(table, _pages())
    assert "data-x-" not in page.html
    for gone in ("site-nav", "Home", "cookie", "Copyright", "onclick", "<script", "<style"):
        assert gone not in page.html
    # labels adjacent to candidates stay
    assert "<h2>Details</h2>" in page.html
    assert '<span class="label">Phone:</span>' in page.html


def test_collapse_by_variant_keeps_each_variant_first_and_last(table: CandidateTable) -> None:
    pubs: list[str | tuple[str, str]] = []
    for i in range(30):
        pubs.append((f"Paper {i}", "2020") if i % 3 == 1 else f"Paper {i}")
    pages = {0: profile_page("Ann Lee", email="a@x.org", pubs=pubs)}
    page = _build(table, pages)
    root = lhtml.fromstring(page.html)
    titles = [s.text_content() for s in root.iter("span") if s.get("class") == "title"]
    # first, last, one example of the second variant (with a year)
    assert titles == ["Paper 0", "Paper 1", "Paper 29"]
    assert page.collapsed_groups == 1
    # 20 plain pubs, 2 kept (first + last) -> 18 removed; 10 with a year, 1 kept -> 9 removed
    assert (
        '<span class="title">Paper 0</span></li>\n<!-- collapsed: 18 more siblings like the one '
        "above (variant 1 of 2) -->" in page.html
    )
    assert "<!-- collapsed: 9 more siblings like the one above (variant 2 of 2) -->" in page.html


def test_key_value_list_is_never_collapsed(table: CandidateTable) -> None:
    specs = [(f"Key {i}", f"Value {i}") for i in range(12)]
    pages = {0: profile_page("Ann Lee", specs=specs)}
    page = _build(table, pages)
    root = lhtml.fromstring(page.html)
    dl = root.find(".//dl")
    assert dl is not None
    assert len(_kids(dl)) == 24
    assert "collapsed" not in page.html


def test_attributes_are_pruned(table: CandidateTable) -> None:
    long_href = "https://example.org/" + "x" * 120
    html_page = (
        profile_page("Ann Lee")
        .replace(
            '<h1 class="name">',
            '<h1 class="name flex mt-4 css-1x2y3z" id="post-1234" style="color:red" '
            'data-foo="1" data-x-evil="1" itemprop="name" role="heading" aria-label="Full name" '
            'data-testid="person-name" title="t">',
        )
        .replace("</h1>", f' <a href="{long_href}" type="text/html" onclick="go()">home</a></h1>')
    )
    h1_path = f"{P} > h1.name[itemprop=name][data-testid=person-name][role=heading]"
    for c in table.candidates:
        if c.id == "c1":
            c.path = h1_path
    page = _build(table, {0: html_page})
    h1 = lhtml.fromstring(page.html).find(".//h1")
    assert h1 is not None
    assert dict(h1.attrib) == {
        "class": "name",
        "itemprop": "name",
        "data-testid": "person-name",
        "role": "heading",
        "aria-label": "Full name",
    }
    link = h1.find("a")
    assert link is not None
    assert dict(link.attrib) == {"href": long_href[:77] + "...", "type": "text/html"}
    assert "c1 · mandatory" in page.html


def test_text_truncated_from_the_middle(table: CandidateTable) -> None:
    bio = "Start of the bio. " + "filler words " * 40 + "End of the bio."
    name = "N" * 150 + "Z" * 250
    page = _build(table, {0: profile_page(name, bio=bio)})
    root = lhtml.fromstring(page.html)
    p = root.find(".//p")
    assert p is not None and p.text is not None
    assert p.text.startswith("Start of the bio.") and p.text.endswith("End of the bio.")
    assert " … " in p.text and len(p.text) <= 121
    h1 = root.find(".//h1")
    assert h1 is not None and h1.text is not None
    # candidate text keeps up to 300 characters, also cut in the middle
    assert len(h1.text) <= 301 and h1.text.startswith("NNN") and h1.text.endswith("ZZZ")


def test_budget_drops_candidate_free_subtrees(table: CandidateTable) -> None:
    filler = "".join(
        f'<div class="note-{chr(97 + i % 26)}{chr(97 + i // 26)}"><p>Note {i}: something '
        f"unique about item {i * 7919}</p></div>"
        for i in range(400)
    )
    big = profile_page("Ann Lee", email="a@x.org").replace(
        "<h2>Details</h2>", filler + "<h2>Details</h2>"
    )
    full = _build(table, {0: big}, max_tokens=10**6)
    assert full.tokens > 5000
    page = _build(table, {0: big}, max_tokens=1500)
    assert page.tokens <= 1500
    assert page.dropped
    assert page.missing == ["c8", "c10"]  # no sampled page has them
    assert '<h1 class="name">Ann Lee</h1>' in page.html
    assert page.log[-1].startswith("final page:")


def test_budget_raises_collapse_aggressiveness(table: CandidateTable) -> None:
    table.candidates.append(cand("c19", f"{PUBS} > span.year", "optional", 0.5))
    pubs: list[str | tuple[str, str]] = []
    for i in range(60):
        pubs.append((f"Paper {i} " + "words " * 20, str(2000 + i)) if i % 2 else f"Paper {i}")
    pages = {0: profile_page("Ann Lee", email="a@x.org", pubs=pubs)}
    loose = _build(table, pages)
    assert loose.collapse_level == 0
    assert not loose.dropped
    tight = _build(table, pages, max_tokens=loose.tokens - 20)
    assert tight.tokens <= loose.tokens - 20
    assert tight.collapse_level >= 1
    assert "c9 · list" in tight.html and "c19 · optional" in tight.html
    assert tight.missing == ["c8", "c10"]


def test_hard_cap_drops_least_frequent_candidates_last(table: CandidateTable) -> None:
    page = _build(table, _pages(), max_tokens=120)
    assert page.tokens <= 120
    assert any(d.startswith("candidate ") for d in page.dropped)
    assert page.missing  # reported, not silently lost


def test_focus_trims_to_neighborhoods(table: CandidateTable) -> None:
    page = _build(table, _pages(), focus={"c3", "c8", "c99"})
    assert page.focus == ["c3", "c8"]
    assert "c3 · optional" in page.html and "c8 · optional" in page.html
    assert "<h1" not in page.html and "Paper" not in page.html and "pubs" not in page.html
    # neighborhoods keep labels: the contact row and the whole key-value list
    assert "Email:" in page.html and "<dt>Fax</dt>" in page.html and "<dt>Office</dt>" in page.html
    assert any("c99" in line for line in page.log)


def test_single_page_and_small_page_is_not_padded(table: CandidateTable) -> None:
    page = build_union_page(table, {7: profile_page("Ann Lee")})
    assert page.base_page_id == 7
    assert page.donor_page_ids == []
    assert page.under_min
    assert "not padded" in page.log[-1]
    meta = page.meta()
    assert meta["base_page_id"] == 7 and meta["under_min"] is True
    assert set(meta) >= {"tokens", "donor_page_ids", "spliced", "log", "focus", "missing"}
