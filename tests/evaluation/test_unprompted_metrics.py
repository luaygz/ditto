from __future__ import annotations

import pytest

from ditto.evaluation.metrics import (
    align_field_names,
    field_parts,
    part_value,
    rename_fields,
    schema_precision,
    score_value,
)


def _book_pages(n: int) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    labels: dict[str, dict[str, object]] = {}
    preds: dict[str, dict[str, object]] = {}
    for i in range(n):
        key = f"k{i}"
        labels[key] = {
            "title": [f"Book {i}"],
            "publisher": [f"House {i % 3}"],
            "publication_date": [f"200{i % 10}"],
            "isbn_13": [f"97800000000{i:02d}"],
        }
        preds[key] = {
            "title": f"Book {i}",
            # one key-value field holds three labeled attributes
            "bibliographic_details": {
                "ISBN 13": f"97800000000{i:02d}",
                "Publisher": f"House {i % 3}",
                "Publication  Date": f"200{i % 10}",
                "Binding": "Hardcover",
            },
            "offers": [
                {"seller": f"Shop {i}", "price": f"${i}.50"},
                {"seller": "Other shop", "price": "$1.00"},
            ],
        }
    return labels, preds


def test_field_parts_address_keys_and_columns() -> None:
    parts = field_parts("specs", {"Engine": "V6", " Fuel  type ": "Gas", "Empty": ""})
    assert parts == {
        "specs": {"Engine": "V6", " Fuel  type ": "Gas", "Empty": ""},
        "specs[Engine]": "V6",
        "specs[Fuel type]": "Gas",
    }
    rows = [{"name": "A", "role": "x"}, {"name": "B", "role": None}]
    assert field_parts("cast", rows) == {
        "cast": rows,
        "cast.name": ["A", "B"],
        "cast.role": ["x"],
    }
    assert field_parts("tags", ["a", "b"]) == {"tags": ["a", "b"]}
    assert field_parts("title", "T") == {"title": "T"}


def test_part_value_resolves_whole_fields_and_parts() -> None:
    row = {"specs": {"Engine": "V6"}, "cast": [{"name": "A"}], "title": "T"}
    assert part_value(row, "title") == "T"
    assert part_value(row, "specs[Engine]") == "V6"
    assert part_value(row, "specs[Doors]") is None
    assert part_value(row, "cast.name") == ["A"]
    assert part_value(row, "missing[Key]") is None
    assert part_value(row, "absent") is None


def test_labels_align_to_key_value_keys_instead_of_the_whole_field() -> None:
    labels, preds = _book_pages(20)
    alignment = align_field_names(preds, labels, sorted(preds))
    assert alignment == {
        "title": "title",
        "publisher": "bibliographic_details[Publisher]",
        "publication_date": "bibliographic_details[Publication Date]",
        "isbn_13": "bibliographic_details[ISBN 13]",
    }
    renamed = rename_fields(preds["k4"], alignment)
    assert renamed == {
        "title": "Book 4",
        "publisher": "House 1",
        "publication_date": "2004",
        "isbn_13": "978000000000" + "4",
    }
    for field in ("publisher", "publication_date", "isbn_13"):
        counts = score_value("scalar", renamed[field], labels["k4"][field], "norm")
        assert counts.tp == 1


def test_labels_align_to_record_columns_and_a_whole_field_blocks_its_parts() -> None:
    labels = {f"k{i}": {"seller": [f"Shop {i}"]} for i in range(10)}
    preds = {
        f"k{i}": {"offers": [{"seller": f"Shop {i}", "price": "$1"}, {"seller": "X"}]}
        for i in range(10)
    }
    assert align_field_names(preds, labels, sorted(preds)) == {"seller": "offers.seller"}
    # a structured label aligns only to a whole structured field ...
    kv_labels = {f"k{i}": {"specs": {"Seller": f"Shop {i}"}} for i in range(10)}
    kv_preds = {f"k{i}": {"details": {"Seller": f"Shop {i}"}} for i in range(10)}
    assert align_field_names(kv_preds, kv_labels, sorted(kv_preds)) == {"specs": "details"}
    # ... and once the whole field is taken, none of its parts can be aligned to another label
    both = {
        k: {**v, "seller": [f"Shop {i}" if i < 5 else "nope"]}
        for i, (k, v) in enumerate(sorted(kv_labels.items()))
    }
    assert align_field_names(kv_preds, both, sorted(kv_preds)) == {"specs": "details"}


def test_plain_labels_never_align_to_a_whole_structured_field() -> None:
    # the whole "prices" overlaps on every page, the "Sale" key on only some: the key wins
    labels = {f"k{i}": {"price": [f"${i}"]} for i in range(10)}
    preds = {
        f"k{i}": {"prices": {"Sale" if i < 6 else "Price": f"${i}", "List": "$99"}}
        for i in range(10)
    }
    assert align_field_names(preds, labels, sorted(preds)) == {"price": "prices[Sale]"}


def test_a_part_blocks_its_whole_field_but_not_its_siblings() -> None:
    labels = {f"k{i}": {"a": [f"A{i}"], "b": [f"B{i}"], "spec": {"A": f"A{i}"}} for i in range(10)}
    preds = {f"k{i}": {"kv": {"A": f"A{i}", "B": f"B{i}"}} for i in range(10)}
    alignment = align_field_names(preds, labels, sorted(preds))
    # "a" sorts before "spec" on the tie and takes kv[A]: the whole kv is no longer available
    assert alignment == {"a": "kv[A]", "b": "kv[B]"}


def _rows(verdicts: dict[tuple[str, str, str], str]) -> list[dict[str, str]]:
    return [
        {"site": s, "template": t, "field": f, "verdict": v, "reason": "r"}
        for (s, t, f), v in verdicts.items()
    ]


def test_schema_precision_per_site_and_overall() -> None:
    rows = _rows(
        {
            ("a", "t1", "title"): "entity",
            ("a", "t1", "share_count"): "noise",
            ("a", "t2", "title"): "entity",
            ("a", "t2", "price"): "entity",
            ("b", "t9", "name"): "entity",
        }
    )
    out = schema_precision(rows)
    assert out["fields"] == 5 and out["entity"] == 4
    assert out["precision"] == pytest.approx(0.8)
    assert out["per_site"]["a"] == {"fields": 4, "entity": 3, "precision": 0.75}
    assert out["per_site"]["b"]["precision"] == 1.0
    assert out["precision_site_mean"] == pytest.approx(0.875)
    assert "run_check" not in out


def test_schema_precision_checks_the_judgments_against_a_run() -> None:
    rows = _rows({("a", "t1", "title"): "entity", ("a", "t1", "old"): "noise"})
    rows.append({"site": "gone", "template": "t", "field": "f", "verdict": "entity"})
    proposed = {"a": {"t1": ["title", "new"]}, "other": {"t": ["x"]}}
    check = schema_precision(rows, proposed)["run_check"]
    assert check == {
        "matches_run": False,
        "unjudged": ["a t1.new"],
        "stale": ["a t1.old"],
        "sites_missing": ["gone"],
    }
    exact = schema_precision(rows[:1], {"a": {"t1": ["title"]}})["run_check"]
    assert exact["matches_run"] is True


def test_schema_precision_rejects_bad_rows() -> None:
    with pytest.raises(ValueError, match="verdict"):
        schema_precision([{"site": "a", "template": "t", "field": "f", "verdict": "maybe"}])
    row = {"site": "a", "template": "t", "field": "f", "verdict": "entity"}
    with pytest.raises(ValueError, match="duplicate"):
        schema_precision([row, row])
    assert schema_precision([])["precision"] is None
