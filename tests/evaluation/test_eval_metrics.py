from __future__ import annotations

import pytest

from ditto.evaluation.metrics import (
    METRIC_VERSION,
    MODES,
    Counts,
    OptionalRecall,
    aggregate,
    align_field_names,
    canonical,
    canonical_text,
    joins_alternatives,
    normalize_text,
    optional_field_recall,
    rename_fields,
    sample_keys,
    score_site,
    score_value,
    values_match,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hello \n  World ", "hello world"),
        ("New\u00a0York", "new york"),
        ("$1,299.00", "1299"),
        ("1,234,567", "1234567"),
        ("1\u202f000", "1000"),
        ("€ 12,50", "12,50"),  # a decimal comma is not a thousands separator
        ("£5", "5"),
        ("¥ 3,000", "3000"),
        ("12.50", "12.5"),
        ("ACME Inc.", "acme inc"),
        ("Hello!!", "hello"),
        ("Price: ", "price"),
        ("Route 66 —", "route 66"),
        ("STRASSE", "strasse"),
        (": Canon", "canon"),  # SWDE text-node label prefix
        (": (702) 214-3500", "(702) 214-3500"),
        ("-5", "-5"),  # a leading minus is not a separator
        ("25\x9630 mpg", "25\u201330 mpg"),  # C1 control read as cp1252 (&#150;)
        ("\x80 12", "12"),  # cp1252 euro sign, then removed as a currency symbol
        ("Straße", "strasse"),  # casefold
        ("U.S.A.", "u.s.a"),
        ("(555) 123-4567", "(555) 123-4567"),
    ],
)
def test_normalize_text_table(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_canonical_modes_and_types() -> None:
    assert canonical(1299.0, "norm") == "1299"
    assert canonical(1299.0, "raw") == "1299"
    assert canonical(12.5, "raw") == "12.5"
    assert canonical(True, "raw") == "true"
    assert canonical(None, "norm") is None
    assert canonical("   ", "raw") is None
    assert canonical("—", "norm") is None  # normalizes to nothing
    assert canonical("—", "raw") == "—"
    assert canonical(" Widget ", "raw") == " Widget "
    assert values_match("$1,299.00", 1299, "norm")
    assert not values_match("$1,299.00", 1299, "raw")
    assert not values_match(None, None, "norm")


def test_scalar_counts() -> None:
    assert score_value("scalar", "widget", "Widget", "norm") == Counts(tp=1)
    assert score_value("scalar", "widget", "Widget", "raw") == Counts(fp=1, fn=1)
    # optional field: null where the label is empty is correct (true negative)
    assert score_value("scalar", None, None, "norm") == Counts(tn=1)
    assert score_value("scalar", "", [], "norm") == Counts(tn=1)
    # null where the label has a value is a miss; a value where the label is empty is an FP
    assert score_value("scalar", None, "x", "norm") == Counts(fn=1)
    assert score_value("scalar", "x", None, "norm") == Counts(fp=1)


def test_scalar_multi_value_labels_and_list_predictions() -> None:
    label = ["Drama", "Comedy"]
    assert score_value("scalar", "comedy", label, "norm") == Counts(tp=1)
    assert score_value("scalar", "Horror", label, "norm") == Counts(fp=1, fn=1)
    # a list prediction is scored by its first non-empty item (top-1)
    assert score_value("scalar", ["", "Drama", "Horror"], label, "raw") == Counts(tp=1)
    assert score_value("scalar", ["Horror", "Drama"], label, "raw") == Counts(fp=1, fn=1)


# -- canon column (rules derived from SWDE dev misses, metric version 2) -------------------------


def test_match_columns_and_version() -> None:
    assert MODES == ("norm", "raw", "canon")
    assert METRIC_VERSION == 2


@pytest.mark.parametrize(
    ("pred", "label"),
    [
        # trailing separator: SWDE restaurant labels end with " |"
        ("2477 Arctic Blvd., Anchorage, AK, 99503", "2477 Arctic Blvd., Anchorage, AK, 99503 |"),
        ("907/243-0507", "907/243-0507 |"),
        ("12.5", "12.50 |"),  # the number form applies after the strip
        # diacritics (combining marks U+0300-U+036F)
        ("Sacks Cafe", "Sacks Café"),
        ("Creme Brulee", "Crème brûlée"),
        ("Cafe\u0301", "Café"),  # decomposed vs precomposed
        ("Muller", "MÜLLER"),
        # digit groups of a long identifier (ISBN-13/10, ten digits or more)
        ("9780307269997", "978-0307269997"),
        ("978 0 307 26999 7", "978-0-307-26999-7"),
        ("0307269997", "0-307-26999-7"),  # leading zero: same number form on both sides
        ("080442957X", "0-8044-2957-x"),  # ISBN-10 check character
    ],
)
def test_canon_credits_formatting_differences(pred: str, label: str) -> None:
    assert not values_match(pred, label, "norm")
    assert values_match(pred, label, "canon")
    assert score_value("scalar", pred, label, "canon") == Counts(tp=1)


@pytest.mark.parametrize(
    ("pred", "label"),
    [
        # extra or different facts are never credited
        ("907/274-2739 | Reserve Online", "907/274-2739 |"),
        ("Ray Allen", "Ray Allen\u00a0\u00a0#20\u00a0\u00a0SG"),
        ("May 25, 2010", "Knopf; 1 edition (May 25, 2010)"),
        ("Knopf", "Knopf; 1 edition (May 25, 2010)"),
        ("George Roy Hill", "Roy Hill"),
        ("PG13", "P"),
        ("Price: $12.99", "$12.99"),  # no label-prefix rule
        ("ISBN 978-0307269997", "9780307269997"),
        ("a | b", "a b"),  # a pipe inside a value stays
        ("Cafe", "Cafes"),
        # digit groups with fewer than ten digits are left alone: dates, ranges, ZIP+4
        ("1-12-2010", "11-2-2010"),
        ("2010-05-25", "20100525"),
        ("10-12", "1012"),
        ("12345-6789", "123456789"),
        ("978-0307269997", "978-0307269998"),
        ("(907) 274-2739", "9072742739"),  # only hyphens and spaces separate digit groups
    ],
)
def test_canon_does_not_credit_different_values(pred: str, label: str) -> None:
    assert not values_match(pred, label, "canon")
    assert score_value("scalar", pred, label, "canon") == Counts(fp=1, fn=1)


@pytest.mark.parametrize(
    "text",
    [
        "  Hello \n  World ",
        "$1,299.00",
        "€ 12,50",
        ": Canon",
        "25\x9630 mpg",
        "Straße",
        "U.S.A.",
        "(555) 123-4567",
        "Route 66 —",
    ],
)
def test_canon_extends_norm(text: str) -> None:
    # canon is a function of the norm form: whatever norm equates, canon equates
    assert canonical_text(text) == canonical_text(normalize_text(text))
    assert normalize_text(text) == canonical_text(text)  # no canon rule applies to these


def test_canonical_text_leaves_other_scripts_and_inner_separators() -> None:
    assert canonical_text("שָׁלוֹם") == "שָׁלוֹם"  # Hebrew points are not U+0300-U+036F
    assert canonical_text("किताब") == "किताब"
    assert canonical_text("a | b |") == "a | b"
    assert normalize_text("Sacks Café") == "sacks café"  # norm itself is unchanged
    assert normalize_text("99503 |") == "99503 |"
    assert canonical("—", "canon") is None
    assert canonical("- |", "canon") is None
    assert canonical(None, "canon") is None
    assert canonical(9780307269997, "canon") == "9780307269997"


@pytest.mark.parametrize(
    ("pred", "label"),
    [
        ("Comedy Western", ["Comedy", "Western"]),
        ("Seafood,American", ["Seafood", "American"]),
        ("New American, American", ["New American", "American"]),
        (
            "Fantasy Musical Romance Children's/Family",
            ["Fantasy", "Musical", "Romance", "Children's/Family"],
        ),
        ("Drama; Comedy", ["Comedy", "Drama", "Horror"]),  # a subset, in any order
        ("Drama / Comedy |", ["Comedy", "Drama"]),
    ],
)
def test_canon_credits_joined_label_values_on_scalars(pred: str, label: list[str]) -> None:
    assert score_value("scalar", pred, label, "norm") == Counts(fp=1, fn=1)
    assert score_value("scalar", pred, label, "canon") == Counts(tp=1)


@pytest.mark.parametrize(
    ("pred", "label"),
    [
        ("Comedy Western 1969", ["Comedy", "Western"]),  # an extra fact
        ("Comedy Horror", ["Comedy", "Western"]),  # a value the label does not hold
        ("Comedy Comedy", ["Comedy", "Western"]),  # repeats are not a join
        ("ComedyWestern", ["Comedy", "Western"]),  # no separator
        ("Comedy and Western", ["Comedy", "Western"]),
        ("Comedy Comedy", "Comedy"),  # a single labeled value has nothing to join
    ],
)
def test_canon_join_rule_rejects_other_content(pred: str, label: str | list[str]) -> None:
    assert score_value("scalar", pred, label, "canon") == Counts(fp=1, fn=1)


def test_join_rule_is_scalar_only() -> None:
    assert joins_alternatives("drama comedy", ["comedy", "drama"])
    assert not joins_alternatives("drama", ["comedy", "drama"])
    assert not joins_alternatives("drama comedy", ["drama comedy"])
    # list fields keep per-item matching: a joined string is one wrong item
    c = score_value("list", ["Comedy Western"], ["Comedy", "Western"], "canon")
    assert c == Counts(fp=1, fn=2)
    # the first item of a list prediction is what a scalar scores
    assert score_value("scalar", ["Comedy Western", "x"], ["Comedy", "Western"], "canon") == Counts(
        tp=1
    )


def test_canon_applies_to_every_field_kind() -> None:
    label_rows = [{"name": "Café Uno", "phone": "555 |"}]
    pred_rows = [{"name": "Cafe Uno", "phone": "555"}]
    assert score_value("record_list", pred_rows, label_rows, "norm") == Counts(fp=1, fn=1)
    assert score_value("record_list", pred_rows, label_rows, "canon") == Counts(tp=1)
    kv = score_value("key_value", {"Matière": "Coton"}, {"Matiere": "coton |"}, "canon")
    assert kv == Counts(tp=1)
    assert score_value("list", ["Cafe", "Bar"], ["Café", "Bar |"], "canon") == Counts(tp=2)


def test_score_site_and_aggregate_report_canon() -> None:
    labels = {"p0": {"name": "Café |"}, "p1": {"name": "Bar"}}
    predictions = {"p0": {"name": "cafe"}, "p1": {"name": "Bar"}}
    s = score_site("site", "v", predictions, labels)
    assert s.fields["name"]["norm"] == Counts(tp=1, fp=1, fn=1)
    assert s.fields["name"]["canon"] == Counts(tp=2)
    assert s.to_dict()["canon"]["f1"] == pytest.approx(1.0)
    agg = aggregate([s])
    assert agg["overall"]["canon"]["f1"] == pytest.approx(1.0)
    assert agg["per_vertical"]["v"]["fields_canon_f1"] == {"name": pytest.approx(1.0)}


def test_list_counts_are_per_item() -> None:
    c = score_value("list", ["a", "b", "c"], ["A", "b", "d"], "norm")
    assert c == Counts(tp=2, fp=1, fn=1)
    assert score_value("list", ["a", "a"], ["a"], "norm") == Counts(tp=1, fp=1)
    assert score_value("list", [], [], "norm") == Counts(tn=1)
    assert score_value("list", "a", ["a"], "norm") == Counts(tp=1)


def test_record_list_rows_align_by_position() -> None:
    label = [
        {"size": "S", "stock": "3"},
        {"size": "M", "stock": "0"},
        {"size": "L", "stock": "7"},
    ]
    pred = [{"Size": "s", "stock": "3"}, {"size": "M", "stock": "1"}]
    assert score_value("record_list", pred, label, "norm") == Counts(tp=1, fp=1, fn=2)
    assert score_value("record_list", pred, label, "raw") == Counts(tp=0, fp=2, fn=3)
    # shifted rows do not match (positional alignment)
    shifted = [{"size": "M", "stock": "0"}, {"size": "L", "stock": "7"}]
    assert score_value("record_list", shifted, label, "norm") == Counts(tp=0, fp=2, fn=3)
    # an optional column absent in the label must be absent in the prediction too
    assert score_value(
        "record_list", [{"a": "1", "b": "x"}], [{"a": "1", "b": None}], "norm"
    ) == Counts(fp=1, fn=1)
    assert score_value("record_list", [{"a": "1"}], [{"a": "1", "b": None}], "norm") == Counts(tp=1)
    # all-empty rows are ignored
    assert score_value("record_list", [{"a": None}], [], "norm") == Counts(tn=1)


def test_key_value_scored_per_key() -> None:
    label = {"Color": "Red", "Weight": "2 kg"}
    pred = {"color:": "red", "Weight": "3 kg", "Size": "XL"}
    assert score_value("key_value", pred, label, "norm") == Counts(tp=1, fp=2, fn=1)
    assert score_value("key_value", pred, label, "raw") == Counts(tp=0, fp=3, fn=2)
    pairs = [["Color", "Red"], {"key": "Weight", "value": "2 kg"}, "junk"]
    assert score_value("key_value", pairs, label, "raw") == Counts(tp=2)
    assert score_value("key_value", {}, {}, "norm") == Counts(tn=1)


def test_counts_metrics() -> None:
    c = Counts(tp=3, fp=1, fn=2)
    assert c.precision == pytest.approx(0.75)
    assert c.recall == pytest.approx(0.6)
    assert c.f1 == pytest.approx(2 * 3 / (2 * 3 + 1 + 2))
    assert Counts(tn=4).f1 is None
    assert Counts(fn=1).precision is None
    assert Counts(fn=1).f1 == 0.0


def test_score_site_handles_missing_predictions_exclusions_and_unannotated_fields() -> None:
    labels = {
        "p0": {"title": "A", "price": "$1"},
        "p1": {"title": "B", "price": None},
        "p2": {"title": "C", "price": "$3"},
        "p3": {"title": "D"},  # price not annotated here
        "p4": {"title": "E", "price": "$5"},
    }
    predictions = {
        "p0": {"title": "A", "price": "$1"},
        "p1": {"title": "b", "price": None},
        "p2": {"title": "C", "price": "$4"},
        "p3": {"title": "D", "price": "$9"},
    }
    s = score_site("site", "v", predictions, labels, exclude=["p0"])
    assert s.pages_excluded == 1
    assert s.pages_scored == 4
    assert s.pages_without_prediction == 1
    assert s.fields["title"]["norm"] == Counts(tp=3, fn=1)
    assert s.fields["title"]["raw"] == Counts(tp=2, fp=1, fn=2)
    # p1 null/null -> TN; p2 wrong -> FP+FN; p3 not annotated -> skipped; p4 no prediction -> FN
    assert s.fields["price"]["norm"] == Counts(fp=1, fn=2, tn=1)
    m = s.metrics("norm")
    assert m["f1"] == pytest.approx((6 / 7 + 0.0) / 2)
    assert set(s.to_dict()["fields"]) == {"title", "price"}


def test_aggregate_averages_sites_then_verticals() -> None:
    def site(name: str, vertical: str, correct: bool) -> object:
        pred = {"k": {"f": "yes" if correct else "no"}}
        return score_site(name, vertical, pred, {"k": {"f": "yes"}})

    scores = [site("a1", "A", True), site("a2", "A", False), site("b1", "B", True)]
    agg = aggregate(scores)  # type: ignore[arg-type]
    assert agg["per_vertical"]["A"]["norm"]["f1"] == pytest.approx(0.5)
    assert agg["per_vertical"]["A"]["n_sites"] == 2
    assert agg["per_vertical"]["B"]["norm"]["f1"] == pytest.approx(1.0)
    # macro over verticals (0.75), not micro over sites (2/3)
    assert agg["overall"]["norm"]["f1"] == pytest.approx(0.75)
    assert agg["overall"]["n_sites"] == 3
    assert agg["per_vertical"]["A"]["fields_norm_f1"] == {"f": pytest.approx(0.5)}


def test_group_metrics() -> None:
    s = score_site("s", "v", {"k": {"a": "1", "b": "x"}}, {"k": {"a": "1", "b": "2", "c": "3"}})
    groups = s.group_metrics({"type_1": ["a"], "type_2": ["b", "c"], "empty": ["zzz"]}, "norm")
    assert groups["type_1"]["f1"] == 1.0
    assert groups["type_2"]["f1"] == 0.0
    assert groups["empty"]["f1"] is None


def test_optional_field_recall_only_counts_fields_unseen_at_generation() -> None:
    labels = {
        "p0": {"title": "A", "author": None},  # the generating page lacks author
        "p1": {"title": "B", "author": "X"},
        "p2": {"title": "C", "author": "Y"},
        "p3": {"title": "D", "author": None},
        "q0": {"title": "Q", "author": "Z"},  # template without known generating pages
    }
    predictions = {"p1": {"author": "x"}, "p2": {"title": "C"}, "q0": {"author": "Z"}}
    page_template = {"p0": "T", "p1": "T", "p2": "T", "p3": "T", "q0": "U"}
    result = optional_field_recall(predictions, labels, page_template, {"T": ["p0"], "U": []})
    assert result.pages == 2
    assert set(result.fields) == {"author"}  # title was present on the generating page
    assert result.total("norm") == Counts(tp=1, fn=1)
    assert result.total("raw") == Counts(tp=0, fn=2)
    d = result.to_dict()
    assert d["norm"]["recall"] == pytest.approx(0.5)
    merged = OptionalRecall()
    merged.add(result)
    merged.add(result)
    assert merged.pages == 4
    assert merged.total("norm") == Counts(tp=2, fn=2)


def test_unprompted_alignment_by_value_overlap() -> None:
    labels = {
        f"k{i}": {"title": f"Item {i}", "price": f"${i}.00", "color": "red"} for i in range(20)
    }
    predictions = {
        f"k{i}": {
            "product_name": f"item {i}",
            "cost": f"{i}",
            "headline": f"Item {i}" if i == 0 else "other",
            "url": "ignored",
            "_meta": {"missing": []},
        }
        for i in range(20)
    }
    keys = sample_keys(predictions, 50, seed=1)
    assert len(keys) == 20
    alignment = align_field_names(predictions, labels, keys)
    # one-to-one: "headline" overlaps title on 1 page only and loses to product_name;
    # color has no overlapping prediction
    assert alignment == {"title": "product_name", "price": "cost"}
    renamed = rename_fields(predictions["k3"], alignment)
    assert renamed == {"title": "item 3", "price": "3"}
    # minimum overlap fraction: a field matching on 1 of 20 pages is not aligned at 10%
    only_headline = {k: {"headline": v["headline"]} for k, v in predictions.items()}
    assert align_field_names(only_headline, labels, keys) == {}
    assert align_field_names(only_headline, labels, keys, min_fraction=0.0) == {"title": "headline"}


def test_alignment_handles_structured_values() -> None:
    labels = {"k": {"specs": {"Color": "Red"}, "rows": [{"size": "S"}]}}
    predictions = {"k": {"attributes": {"colour": "red"}, "table": [{"s": "s"}]}}
    assert align_field_names(predictions, labels, ["k"]) == {
        "specs": "attributes",
        "rows": "table",
    }


def test_sample_keys_is_deterministic() -> None:
    keys = [f"p{i:03d}" for i in range(500)]
    a = sample_keys(keys, 200, seed=7)
    assert a == sample_keys(reversed(keys), 200, seed=7)
    assert len(a) == 200 and a == sorted(a)
    assert a != sample_keys(keys, 200, seed=8)
