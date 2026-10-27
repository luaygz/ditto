"""Rule-based probe matching on hand-built candidate tables (no LLM, no I/O)."""

from __future__ import annotations

from checkpoint_builders import BODY, cand, job_table, kv_table, table

from ditto.analyze.models import CandidateTable, Table, TableColumn
from ditto.checkpoints.match import (
    expects_list,
    match_template,
    name_variants,
    phrase_similarity,
    tokens,
    usable_candidates,
)
from ditto.schema import ABSENT, FieldRequest, FieldsSchema


def match(t, fields):  # type: ignore[no-untyped-def]
    return match_template(t, FieldsSchema.from_obj(fields).fields)


def test_tokens_normalize_case_punctuation_and_plurals() -> None:
    assert tokens("Date Posted:") == ("date", "posted")
    assert tokens("hiringOrganization") == ("hiring", "organization")
    assert tokens("job_title") == ("job", "title")
    assert tokens("Categories") == ("category",)
    assert tokens("the price of the item") == ("price", "item")


def test_phrase_similarity_exact_overlap_and_fuzzy() -> None:
    assert phrase_similarity(("email",), ("e", "mail")) == 1.0  # compact forms equal
    assert phrase_similarity(("title",), ("job", "title")) >= 0.8
    assert phrase_similarity(("organisation",), ("organization",)) > 0.8
    assert phrase_similarity(("price",), ("location",)) == 0.0


def test_name_variants_substitute_synonyms() -> None:
    variants = dict(name_variants("company name"))
    assert variants[("company",)] == 1.0  # generic "name" dropped
    assert ("employer",) in variants
    assert ("hiring", "organization") in variants


def test_synonyms_map_fields_to_differently_labelled_candidates() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > span.a", label="Cost:", examples=["$12.00", "$15.50"]),
            cand("c2", f"{BODY} > span.b", label="Employer", examples=["Acme", "Globex"]),
            cand("c3", f"{BODY} > span.c", label="City", examples=["Austin", "Paris"]),
            cand("c4", f"{BODY} > span.d", label="Published", examples=["2026-01-02"]),
        ],
    )
    got = match(t, ["price", "company", "location", "date posted"])
    assert {name: (m.status, m.use) for name, m in got.items()} == {
        "price": ("found", "c1"),
        "company": ("found", "c2"),
        "location": ("found", "c3"),
        "date posted": ("found", "c4"),
    }
    assert "synonym" in got["company"].ranked[0].reasons[0]


def test_itemprop_and_class_identifiers_count_as_labels() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > div[itemprop=hiringOrganization]", examples=["Acme"]),
            cand("c2", f"{BODY} > span.job-location", examples=["Austin"]),
            cand("c3", f"{BODY} > span.other", examples=["x"]),
        ],
    )
    got = match(t, ["company", "location"])
    assert got["company"].use == "c1"
    assert got["location"].use == "c2"


def test_type_match_separates_equally_labelled_candidates() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > p.a", label="Contact", examples=["555-0100", "555-0199"]),
            cand(
                "c2",
                f"{BODY} > p.b",
                label="Contact",
                examples=["ann@example.org", "bo@example.org"],
                value_type="email",
            ),
        ],
    )
    got = match(t, {"contact": {"type_check": "email"}})["contact"]
    assert got.status == "found"
    assert got.use == "c2"
    assert got.ranked[0].score > got.ranked[1].score + 0.3


def test_text_field_prefers_text_over_link_attribute() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > a.company", label="Company", examples=["Acme", "Globex"]),
            cand(
                "c2",
                f"{BODY} > a.company @href",
                attr="href",
                label="Company",
                examples=["/c/acme", "/c/globex"],
                value_type="url",
            ),
        ],
    )
    got = match(t, ["company"])["company"]
    assert got.use == "c1"
    assert got.status == "found"


def test_user_examples_identify_an_unlabelled_candidate() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > div > span", examples=["Acme Corp", "Globex Inc", "Initech"]),
            cand("c2", f"{BODY} > div > em", examples=["Austin", "Denver", "Remote"]),
        ],
    )
    got = match(t, {"brand": {"examples": ["Globex Inc"]}})["brand"]
    assert got.use == "c1"
    assert got.status == "found"
    assert "example value matches" in got.ranked[0].reasons


def test_example_shape_disagreement_lowers_a_labelled_candidate() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > span.a", label="Rating", examples=["great", "ok", "bad"]),
            cand("c2", f"{BODY} > span.b", label="Rating", examples=["4.5", "3.0", "5.0"]),
        ],
    )
    got = match(t, {"rating": {"examples": ["4.0"]}})["rating"]
    assert got.use == "c2"


def test_labels_and_boilerplate_are_never_fields() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > dl > dt", "label", examples=["Price:"]),
            cand("c2", f"{BODY} > span.price", "boilerplate", label="Price", examples=["$1"]),
            cand("c3", f"{BODY} > span.noise", "noise", label="Price", examples=["$9"]),
        ],
    )
    assert usable_candidates(t) == []
    got = match(t, {"price": {"examples": ["Price:"]}})["price"]
    assert got.status == "missing"
    assert got.use == ABSENT
    assert got.ranked == []


def test_kv_keys_and_record_headers_are_not_fields() -> None:
    usable = {c.id for c in usable_candidates(kv_table())}
    assert "c1" not in usable  # kv_key
    assert "c3" not in usable  # record_header
    assert {"c0", "c2", "c4", "c5"} <= usable


def test_found_ambiguous_missing() -> None:
    got = match(job_table(), ["company", "salary", "remote policy", "title"])
    assert (got["company"].status, got["company"].use) == ("found", "c4")
    assert (got["title"].status, got["title"].use) == ("found", "c2")  # h1 over <title>
    assert got["salary"].status == "ambiguous"  # only a parent class "pay" hints at it
    assert got["salary"].use == "c7"
    assert (got["remote policy"].status, got["remote policy"].use) == ("missing", ABSENT)


def test_two_strong_candidates_make_a_field_ambiguous() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > span.a", label="Date", examples=["2026-01-02", "2026-02-03"]),
            cand("c2", f"{BODY} > span.b", label="Date", examples=["2025-05-06", "2025-07-08"]),
        ],
    )
    got = match(t, ["date"])["date"]
    assert got.status == "ambiguous"
    assert [s.id for s in got.ranked] == ["c1", "c2"]


def test_same_values_at_two_nodes_do_not_make_a_field_ambiguous() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > span.a", label="Date", examples=["2026-01-02", "2026-02-03"]),
            cand("c2", f"{BODY} > span.b", label="Date", examples=["2026-02-03", "2026-01-02"]),
        ],
    )
    assert match(t, ["date"])["date"].status == "found"


def test_kv_key_match_and_record_list_table() -> None:
    got = match(
        kv_table(),
        {
            "salary": "pay",
            "job type": "",
            "reviews": {"type": "record_list", "description": "user reviews"},
        },
    )
    assert (got["salary"].status, got["salary"].use) == ("found", "c2")
    assert got["salary"].ranked[0].key == "Salary"
    assert got["job type"].ranked[0].key == "Job type"
    assert got["reviews"].status == "found"
    assert got["reviews"].use == ["c4", "c5"]
    assert got["reviews"].ranked[0].id == "tb1"


def test_list_field_prefers_repeating_candidate() -> None:
    t = table(
        "t_a",
        [
            cand("c1", f"{BODY} > ul.tags > li", "list", repeat=(1, 6), examples=["a", "b"]),
            cand("c2", f"{BODY} > span.tags", examples=["a, b"]),
        ],
    )
    got = match(t, {"tags": {"type": "list"}})["tags"]
    assert got.use == "c1"


def test_expects_list_from_type_name_and_description() -> None:
    assert expects_list(FieldRequest(name="tags"))
    assert expects_list(FieldRequest(name="ingredients"))
    assert expects_list(FieldRequest(name="plotKeywords"))
    assert expects_list(FieldRequest(name="genre", description="the genre or genres"))
    assert expects_list(FieldRequest(name="author", description="a page may list several"))
    assert expects_list(FieldRequest(name="item", type="list"))
    for singular in ("title", "address", "status", "price", "analysis"):
        assert not expects_list(FieldRequest(name=singular, description="the " + singular))
    assert not expects_list(FieldRequest(name="tags", type="scalar"))  # declared type wins


def test_plural_field_prefers_a_list_over_a_scalar_badge() -> None:
    # an article's badge ("Top rated") sits in a key/value block whose keys include "Label";
    # the real tags are a repeated list labelled "Topics" (fixture T04)
    kv = Table(
        id="tb0",
        kind="key_value",
        container_path=f"{BODY} > div.summary",
        row_path=f"{BODY} > div.summary > p",
        p=1.0,
        keys_seen=["Label", "Posted"],
    )
    t = table(
        "t_a",
        [
            cand(
                "c7",
                f"{BODY} > div.summary > p > span.badge",
                label="Label",
                examples=["Top rated", "Staff pick"],
                table_id="tb0",
                table_role="kv_value",
                column="Label",
            ),
            cand(
                "c31",
                f"{BODY} > section > ul > li > a",
                "list",
                repeat=(1, 5),
                label="Topics",
                examples=["camera", "return"],
            ),
        ],
        [kv],
    )
    got = match(t, {"tags": "the topic tags, keywords or topics listed on the page"})["tags"]
    assert (got.status, got.use) == ("found", "c31")
    assert all(s.id != "c7" for s in got.ranked)  # "Label" is no longer a synonym of tags
    # a plural field still takes a single node when nothing repeats, just with less confidence
    single = table("t_b", [cand("c1", f"{BODY} > span.x", label="Topics", examples=["a, b"])])
    one = match(single, ["topics"])["topics"]
    assert one.use == "c1" and "plural field, single node" in one.ranked[0].reasons


def _course_table() -> CandidateTable:
    syllabus = Table(
        id="tb3",
        kind="record_list",
        container_path=f"{BODY} > section.syllabus-section > table.syllabus",
        row_path=f"{BODY} > section.syllabus-section > table.syllabus > tr",
        p=0.86,
        columns=[
            TableColumn(name="Week", candidate_id="c38"),
            TableColumn(name="Topic", candidate_id="c39"),
        ],
    )
    cell = f"{BODY} > section.syllabus-section > table.syllabus > tr > td"
    return table(
        "t_course",
        [
            cand(
                "c28",
                f"{BODY} > section > ul.class-topics > li.class-topics-item",
                "list",
                repeat=(1, 5),
                label="What you'll learn",
                examples=["Machine learning", "UX research"],
            ),
            cand(
                "c38",
                cell,
                "list",
                p=0.86,
                repeat=(1, 5),
                label="Week",
                examples=["1", "2"],
                table_id="tb3",
                table_role="record_column",
                column="Week",
            ),
            cand(
                "c39",
                cell,
                "list",
                p=0.86,
                repeat=(1, 5),
                label="Topic",
                examples=["Write Coach in Data analysis"],
                table_id="tb3",
                table_role="record_column",
                column="Topic",
            ),
        ],
        [syllabus],
    )


def test_standalone_list_beats_a_record_column_for_a_plain_list_field() -> None:
    # fixture T01: tags must be the course's topic list, not the syllabus table's Topic column
    desc = "the topic tags, keywords or topics listed on the page"
    got = match(_course_table(), {"tags": desc})["tags"]
    assert got.use == "c28"
    assert [s.id for s in got.ranked[:2]] == ["c28", "c39"]
    assert "one column of record table tb3" in got.ranked[1].reasons
    # ... unless the description names the table
    named = match(_course_table(), {"tags": "the topics of each week in the syllabus"})["tags"]
    assert named.use == "c39"
    # a requested record_list still gets the table
    rows = match(_course_table(), {"syllabus": {"type": "record_list"}})["syllabus"]
    assert rows.use == ["c38", "c39"]
