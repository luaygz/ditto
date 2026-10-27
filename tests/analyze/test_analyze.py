from __future__ import annotations

from pathlib import Path

import pytest
from analyze_sites import N_PAGES, NOISE_PAGE, color_on, email_on, n_pubs, product_page, site_pages

from ditto.analyze import Candidate, CandidateTable, analyze_pages, classify, locate_candidates
from ditto.analyze.walk import walk
from ditto.fingerprint.dom import clean_html, iter_text_items

MAIN = "html > body > main"


@pytest.fixture(scope="module")
def table() -> CandidateTable:
    return analyze_pages("t_test", site_pages(), cluster_size=500)


def cand(table: CandidateTable, path: str, column: str | None = None) -> Candidate:
    found = [c for c in table.candidates if c.path == path and c.column == column]
    assert len(found) == 1, (path, column, [(c.id, c.column) for c in found])
    return found[0]


# -- walk ------------------------------------------------------------------------------------------


def test_walk_yields_exactly_the_text_items_of_iter_text_items() -> None:
    html = product_page(1) + "<p>tail <b>bold</b> after</p>"
    expected = [(t.path, t.text, t.attr) for t in iter_text_items(clean_html(html))]
    got = [(t.path, t.text, t.attr) for t in walk(clean_html(html)).items]
    assert got == expected
    assert got  # non-trivial page


# -- classification rules ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((1.0, 1.0, 100, 1, 1), "boilerplate"),
        ((1.0, 0.98, 100, 2, 2), "boilerplate"),
        ((0.96, 0.5, 96, 1, 1), "mandatory"),
        ((0.95, 0.97, 95, 1, 1), "mandatory"),
        ((0.5, 0.3, 50, 1, 1), "optional"),
        ((0.02, 0.1, 2, 1, 1), "optional"),
        ((1.0, 0.2, 100, 1, 6), "list"),
        ((0.3, 0.2, 30, 0, 4), "list"),
        ((0.01, 1.0, 1, 1, 1), "noise"),
        ((0.01, 0.2, 25, 1, 1), "optional"),  # < 2% but on >= 20 sampled pages
        ((0.015, 0.2, 30, 1, 3), "list"),
    ],
)
def test_classify_rules(args: tuple[float, float, int, int, int], expected: str) -> None:
    assert classify(*args) == expected


def test_boilerplate_mandatory_optional_list_noise(table: CandidateTable) -> None:
    assert cand(table, "html > body > nav > ul > li > a").cls == "boilerplate"
    assert cand(table, "html > body > footer > p").cls == "boilerplate"
    name = cand(table, f"{MAIN} > h1.name")
    assert (name.cls, name.p, name.repeat_min, name.repeat_max) == ("mandatory", 1.0, 1, 1)
    badge = cand(table, f"{MAIN} > p.badge")
    assert badge.cls == "optional"
    assert badge.p == pytest.approx(sum(email_on(i) for i in range(N_PAGES)) / N_PAGES)
    tags = cand(table, f"{MAIN} > ul.tags > li")
    assert (tags.cls, tags.repeat_min, tags.repeat_max) == ("list", 1, 4)
    noise = cand(table, f"{MAIN} > p.promo")
    assert (noise.cls, noise.pages_present, noise.examples) == (
        "noise",
        1,
        [f"Special offer {NOISE_PAGE}"],
    )
    assert table.n_sampled == N_PAGES and table.cluster_size == 500


def test_low_cardinality_field_survives(table: CandidateTable) -> None:
    avail = cand(table, f"{MAIN} > p.avail")
    assert avail.cls == "mandatory"
    assert avail.m == pytest.approx(0.7)
    assert set(avail.examples) == {"In stock", "Out of stock"}


def test_label_class_and_preceding_sibling_hint(table: CandidateTable) -> None:
    price = cand(table, f"{MAIN} > div.buy > span.price")
    assert price.label_hint == "Price:"
    assert price.value_type == "price"
    assert cand(table, f"{MAIN} > div.buy > span.lbl").cls == "label"


def test_label_hint_from_parents_first_child(table: CandidateTable) -> None:
    body = cand(table, f"{MAIN} > div.desc > p.body")
    assert body.label_hint == "Description"
    assert cand(table, f"{MAIN} > div.desc > h4").cls == "label"


def test_label_hint_never_crosses_another_field(table: CandidateTable) -> None:
    # p.avail follows the price block whose nearest text is a value, so it has no label.
    assert cand(table, f"{MAIN} > p.avail").label_hint is None
    # the h1 is preceded by <nav>, a section, never a label
    assert cand(table, f"{MAIN} > h1.name").label_hint is None


def test_attribute_label_is_its_elements_text(table: CandidateTable) -> None:
    href = cand(table, f"{MAIN} > a.more @href")
    assert (href.attr, href.label_hint, href.value_type) == ("href", "Read more", "url")
    assert cand(table, f"{MAIN} > a.more").cls == "label"


def test_examples_are_three_distinct_values(table: CandidateTable) -> None:
    name = cand(table, f"{MAIN} > h1.name")
    assert name.examples == ["Widget 0", "Widget 1", "Widget 2"]
    tags = cand(table, f"{MAIN} > ul.tags > li")
    assert len(tags.examples) == 3 and len(set(tags.examples)) == 3
    assert all(len(c.examples) <= 3 for c in table.candidates)


# -- tables ---------------------------------------------------------------------------------------


def test_key_value_dl_with_invariant_keys(table: CandidateTable) -> None:
    dl = f"{MAIN} > dl.profile"
    kv = next(t for t in table.tables if t.container_path == dl)
    assert (kv.kind, kv.layout, kv.row_path) == ("key_value", "dl", f"{dl} > dt")
    assert kv.keys_seen[:2] == ["Name", "Email"] or kv.keys_seen[:2] == ["Name", "Phone"]
    assert set(kv.keys_seen) == {"Name", "Email", "Phone", "Color"}
    assert (kv.rows_min, kv.rows_max) == (2, 4)
    key = table.by_id(kv.key_candidate or "")
    assert (key.path, key.cls, key.table_role) == (f"{dl} > dt", "label", "kv_key")

    email = cand(table, f"{dl} > dd > a", "Email")
    assert (email.cls, email.table_id, email.table_role) == ("optional", kv.id, "kv_value")
    assert email.p == pytest.approx(0.4)
    assert (email.label_hint, email.value_type) == ("Email", "email")
    assert cand(table, f"{dl} > dd > a @href", "Email").examples[0] == "mailto:p0@example.org"
    phone = cand(table, f"{dl} > dd", "Phone")
    assert (phone.cls, phone.p, phone.label_hint) == ("mandatory", 1.0, "Phone")
    color = cand(table, f"{dl} > dd", "Color")
    assert color.cls == "optional"
    assert color.p == pytest.approx(sum(color_on(i) for i in range(N_PAGES)) / N_PAGES)
    # the shared dd path is split per key: no path-level dd candidate remains
    assert not [c for c in table.candidates if c.path == f"{dl} > dd" and c.column is None]


def test_record_list_with_th_header(table: CandidateTable) -> None:
    pubs = f"{MAIN} > table.pubs"
    rec = next(t for t in table.tables if t.container_path == pubs)
    assert (rec.kind, rec.layout, rec.row_path) == ("record_list", "table", f"{pubs} > tbody > tr")
    assert [(c.name, c.header_source) for c in rec.columns] == [("Title", "th"), ("Year", "th")]
    assert (rec.rows_min, rec.rows_max) == (1, 4)
    title = table.by_id(rec.columns[0].candidate_id)
    assert title.path == f"{pubs} > tbody > tr > td > a"
    assert (title.cls, title.table_role, title.label_hint) == ("list", "record_column", "Title")
    year = table.by_id(rec.columns[1].candidate_id)
    assert (year.path, year.label_hint, year.value_type) == (
        f"{pubs} > tbody > tr > td",
        "Year",
        "number",
    )
    header = cand(table, f"{pubs} > thead > tr > th")
    assert (header.cls, header.table_role, header.table_id) == ("label", "record_header", rec.id)


def test_headerless_repeated_rows_table(table: CandidateTable) -> None:
    log = f"{MAIN} > table.log"
    rec = next(t for t in table.tables if t.container_path == log)
    assert (rec.kind, rec.layout, rec.rows_min, rec.rows_max) == ("record_list", "table", 2, 3)
    assert [(c.name, c.header_source) for c in rec.columns] == [("col1", "none"), ("col2", "none")]
    assert table.by_id(rec.columns[0].candidate_id).value_type == "date"


def test_two_cell_table_rows_with_invariant_keys_are_key_value(table: CandidateTable) -> None:
    facts = f"{MAIN} > table.facts"
    kv = next(t for t in table.tables if t.container_path == facts)
    assert (kv.kind, kv.layout, kv.row_path) == ("key_value", "pairs", f"{facts} > tr")
    assert kv.keys_seen == ["Material", "Origin"]
    origin = cand(table, f"{facts} > tr > td", "Origin")
    assert (origin.cls, origin.label_hint, origin.table_role) == ("mandatory", "Origin", "kv_value")


def test_generic_repeated_rows_become_record_list(table: CandidateTable) -> None:
    row = f"{MAIN} > ul.results > li.result"
    rec = next(t for t in table.tables if t.row_path == row)
    assert (rec.kind, rec.layout, rec.container_path) == (
        "record_list",
        "rows",
        f"{MAIN} > ul.results",
    )
    assert [c.name for c in rec.columns] == ["title", "title_href", "cost"]
    assert all(c.header_source == "none" for c in rec.columns)
    title = table.by_id(rec.columns[0].candidate_id)
    assert (title.path, title.cls, title.table_role) == (
        f"{row} > a.title",
        "list",
        "record_column",
    )
    # no table for a single-path list
    assert not [t for t in table.tables if t.row_path == f"{MAIN} > ul.tags > li"]


# -- ids -------------------------------------------------------------------------------------------


def test_ids_are_deterministic_and_follow_the_base_page() -> None:
    pages = site_pages(40)
    first = analyze_pages("t_x", pages, cluster_size=40)
    second = analyze_pages("t_x", pages, cluster_size=40)
    assert first.model_dump() == second.model_dump()
    assert [c.id for c in first.candidates] == [f"c{i}" for i in range(len(first.candidates))]
    assert [t.id for t in first.tables] == [f"tb{i}" for i in range(len(first.tables))]
    assert first.candidates[0].path == "html > head > title"
    base = first.base_page_id
    assert base is not None and email_on(base) and color_on(base)
    email = cand(first, f"{MAIN} > dl.profile > dd > a", "Email")
    footer = cand(first, "html > body > footer > p")
    assert int(email.id[1:]) < int(footer.id[1:])


def test_save_load_round_trip(table: CandidateTable, tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    table.save(path)
    loaded = CandidateTable.load(path)
    assert loaded == table
    assert '"class": "mandatory"' in path.read_text()


# -- locate ----------------------------------------------------------------------------------------


def test_locate_candidates_splits_keys_and_columns(table: CandidateTable) -> None:
    page = 1
    found = locate_candidates(clean_html(product_page(page)), table)
    email = cand(table, f"{MAIN} > dl.profile > dd > a", "Email")
    assert [el.text for el in found[email.id]] == [f"p{page}@example.org"]
    phone = cand(table, f"{MAIN} > dl.profile > dd", "Phone")
    assert [el.text for el in found[phone.id]] == [f"555-{page:04d}"]
    year = cand(table, f"{MAIN} > table.pubs > tbody > tr > td", "Year")
    assert len(found[year.id]) == n_pubs(page)
    href = cand(table, f"{MAIN} > a.more @href")
    assert [el.get("href") for el in found[href.id]] == [f"/widgets/{page}/details"]
    name = cand(table, f"{MAIN} > h1.name")
    assert [el.text for el in found[name.id]] == [f"Widget {page}"]


def test_locate_ignores_unseen_keys(table: CandidateTable) -> None:
    html = product_page(2).replace("<dt>Phone</dt>", "<dt>Fax</dt>")
    found = locate_candidates(clean_html(html), table)
    phone = cand(table, f"{MAIN} > dl.profile > dd", "Phone")
    assert phone.id not in found


# -- edge cases ------------------------------------------------------------------------------------


def test_layout_tables_are_not_record_tables() -> None:
    def page(i: int) -> str:
        cells = "".join(f"<p>para {i}-{k}</p>" for k in range(70))
        return (
            "<html><body><table class='layout'><tr><td><table class='inner'><tr><td>menu</td>"
            f"</tr></table></td><td>{cells}</td></tr><tr><td>a{i}</td><td>b{i}</td></tr>"
            "</table></body></html>"
        )

    result = analyze_pages("t_l", [(i, page(i)) for i in range(10)], cluster_size=10)
    assert not [t for t in result.tables if "table.layout" in t.container_path.split(" > ")[-1]]
    assert all(c.column is None for c in result.candidates if "table.layout" in c.path)


def test_single_page_and_empty_input() -> None:
    one = analyze_pages("t_1", [(7, product_page(7))], cluster_size=1)
    assert one.n_sampled == 1 and one.base_page_id == 7
    assert {c.cls for c in one.candidates} <= {"boilerplate", "label"}
    empty = analyze_pages("t_0", [], cluster_size=0)
    assert (empty.candidates, empty.tables, empty.base_page_id) == ([], [], None)


def test_alternating_siblings_and_label_tail_rows_are_key_value() -> None:
    def page(i: int) -> str:
        return (
            "<html><body><div class='info'><span class='k'>Brand</span>"
            f"<span class='v'>Acme {i}</span><span class='k'>Size</span>"
            f"<span class='v'>{['S', 'M', 'L'][i % 3]}</span></div>"
            f"<ul class='facts'><li><b>Maker:</b> Corp {i}</li><li><b>Weight:</b> {i % 7} kg</li>"
            "</ul></body></html>"
        )

    result = analyze_pages("t_a", [(i, page(i)) for i in range(20)], cluster_size=20)
    info = "html > body > div.info"
    alt = next(t for t in result.tables if t.container_path == info)
    assert (alt.kind, alt.layout, alt.row_path) == ("key_value", "alternating", f"{info} > span.k")
    assert alt.keys_seen == ["Brand", "Size"]
    size = cand(result, f"{info} > span.v", "Size")
    assert (size.label_hint, size.table_role, size.cls) == ("Size", "kv_value", "mandatory")
    facts = "html > body > ul.facts"
    tail = next(t for t in result.tables if t.container_path == facts)
    assert (tail.layout, tail.row_path, tail.keys_seen) == (
        "pairs",
        f"{facts} > li",
        ["Maker:", "Weight:"],
    )
    maker = cand(result, f"{facts} > li", "Maker:")
    assert (maker.label_hint, maker.examples[0]) == ("Maker:", "Corp 0")
    key = result.by_id(tail.key_candidate or "")
    assert (key.path, key.cls) == (f"{facts} > li > b", "label")
    found = locate_candidates(clean_html(page(4)), result)
    assert [el.tag for el in found[maker.id]] == ["li"]
    assert [el.text for el in found[size.id]] == ["M"]


def test_leading_child_label_and_single_row_header_table() -> None:
    def page(i: int) -> str:
        return (
            f"<html><body><p class='note'><b>Note:</b> remark {i}</p>"
            "<table class='one'><tr><th>Size</th><th>Mass</th></tr>"
            f"<tr><td>{i}</td><td>{i * 2} g</td></tr></table></body></html>"
        )

    result = analyze_pages("t_b", [(i, page(i)) for i in range(12)], cluster_size=12)
    # a lone "<b>Note:</b> value" row is no key-value table, but its key labels the value
    note = cand(result, "html > body > p.note")
    assert (note.label_hint, note.table_id) == ("Note:", None)
    # a header over one row: each column is a mandatory scalar labelled by its <th>
    mass = cand(result, "html > body > table.one > tr > td", "Mass")
    assert (mass.cls, mass.label_hint, mass.repeat_max) == ("mandatory", "Mass", 1)


def test_pair_rows_with_spacer_cells_and_leading_key_text_are_key_value() -> None:
    rows = [("Cuisine", lambda i: ["Thai", "Deli"][i % 2]), ("Phone", lambda i: f"555-01{i:02d}")]
    rows += [("Hours", lambda i: f"{i % 5 + 8}am-9pm")]

    def page(i: int) -> str:
        shown = [r for k, r in enumerate(rows) if (i + k) % 4 != 0]  # rows vary per page
        spacer = "<td><img src='/s.gif'></td>"
        table = "".join(
            f"<tr>{spacer}<td><span><b>{key}</b></span></td><td><span>{value(i)}</span></td>"
            f"{spacer}</tr>"
            for key, value in shown
        )
        facts = "".join(
            f"<li>{key}: <span>{value(i)}</span></li>" for key, value in reversed(shown)
        )
        return (
            f"<html><body><table class='info'>{table}</table>"
            f"<ul class='facts'>{facts}</ul></body></html>"
        )

    result = analyze_pages("t_p", [(i, page(i)) for i in range(24)], cluster_size=24)
    info = "html > body > table.info"
    kv = next(t for t in result.tables if t.container_path == info)
    assert (kv.kind, kv.layout, kv.row_path) == ("key_value", "pairs", f"{info} > tr")
    assert set(kv.keys_seen) == {"Cuisine", "Phone", "Hours"}
    phone = cand(result, f"{info} > tr > td > span", "Phone")
    assert (phone.label_hint, phone.table_role, phone.examples[0]) == (
        "Phone",
        "kv_value",
        "555-0100",
    )
    key = result.by_id(kv.key_candidate or "")
    assert (key.path, key.cls, key.table_role) == (
        f"{info} > tr > td > span > b",
        "label",
        "kv_key",
    )

    facts = "html > body > ul.facts"
    lead = next(t for t in result.tables if t.container_path == facts)
    assert (lead.kind, lead.layout, lead.row_path) == ("key_value", "pairs", f"{facts} > li")
    assert set(lead.keys_seen) == {"Cuisine:", "Phone:", "Hours:"}
    cuisine = cand(result, f"{facts} > li > span", "Cuisine:")
    assert (cuisine.label_hint, cuisine.table_role) == ("Cuisine:", "kv_value")
    key = result.by_id(lead.key_candidate or "")
    assert (key.path, key.cls, key.table_role) == (f"{facts} > li", "label", "kv_key")
    found = locate_candidates(clean_html(page(2)), result)
    assert [el.text for el in found[phone.id]] == ["555-0102"]
    assert [el.text for el in found[cuisine.id]] == ["Thai"]


def test_text_less_cells_beyond_two_or_a_trailing_tail_are_no_pair() -> None:
    def page(i: int) -> str:
        three = "".join(f"<tr><td>Key {k}</td><td>v{i}</td><td>x{i}</td></tr>" for k in range(3))
        tails = "".join(f"<li>Size: <b>{i}</b> cm</li><li>Mass: <b>{i}</b> g</li>" for _ in "a")
        return (
            f"<html><body><table class='three'>{three}</table><ul class='tails'>{tails}</ul>"
            "</body></html>"
        )

    result = analyze_pages("t_n", [(i, page(i)) for i in range(12)], cluster_size=12)
    assert not [t for t in result.tables if t.kind == "key_value"]
