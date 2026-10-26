"""Repair helpers: page compaction, prompt spec, answer validation, variants; generation input."""

from __future__ import annotations

from pathlib import Path

import pytest
from verify_sitekit import cls, staff_page

from ditto.fingerprint.dom import clean_html
from ditto.run import RunDir, write_json
from ditto.specgen.models import KeyValueField, RecordListField, ScalarField, Spec
from ditto.verify.repair import (
    compact_page,
    describe_nodes,
    make_validator,
    repair_variants,
    spec_for_prompt,
)
from ditto.verify.stage import failed_fields, load_generation

COL = "./td[count(ancestor::table[1]//th[normalize-space()='{}']/preceding-sibling::th)+1]"


def _spec() -> Spec:
    return Spec.model_validate(
        {
            "template_id": "t_x",
            "site": "example.org",
            "generated_by": {"model": "m", "prompt_version": "specgen.v1"},
            "fields": {
                "name": {
                    "type": "scalar",
                    "candidates": ["//h1[@itemprop='name']", "//main/h1"],
                    "description": "person name",
                    "source_candidates": ["c0"],
                    "stats": {"presence_rate": 0.5},
                },
                "tags": {"type": "list", "candidates": ["//ul/li[@id='a']", "//ol/li[@id='b']"]},
            },
        }
    )


def test_compact_page_prunes_and_truncates() -> None:
    long_text = "x" * 300
    html = (
        "<html><head><script>bad()</script></head><body>"
        f'<div class="card mt-4 a1b2c3d4" data-foo="1" style="color:red" itemprop="name">{long_text}</div>'
        f'<a href="https://example.org/{"p" * 100}">link</a></body></html>'
    )
    out = compact_page(html)
    assert "bad()" not in out and "data-foo" not in out and "style" not in out
    assert 'class="card"' in out and 'itemprop="name"' in out
    assert "…" in out and long_text not in out
    assert len(compact_page(staff_page(1), max_tokens=10)) == 40


def test_spec_for_prompt_drops_bookkeeping_and_hides_working_fields() -> None:
    text = spec_for_prompt(_spec(), only=["name"])
    assert "presence_rate" not in text and "source_candidates" not in text
    assert "person name" in text and "working; not shown" in text


def test_validator_keeps_description_and_sources() -> None:
    validate = make_validator(_spec(), ["name"])
    out = validate(
        {"fields": {"name": {"candidates": ["//h1[@id='n']", "//h2[@id='n']"], "stats": {}}}}
    )
    field = out["name"]
    assert isinstance(field, ScalarField)
    assert field.description == "person name" and field.source_candidates == ["c0"]
    with pytest.raises(ValueError, match="field definition must be an object"):
        validate({"fields": {"name": "//h1"}})
    with pytest.raises(ValueError, match=r"purely positional"):
        validate({"fields": {"name": {"candidates": ["//div", "//span"]}}})
    with pytest.raises(ValueError, match="no field definition"):
        validate({"fields": {}})


def test_repair_variants_for_each_field_type() -> None:
    old = _spec().fields["name"]
    new = ScalarField(candidates=["//h1[@id='n']", "//h2[@id='n']", "//h3[@id='n']"])
    variants = repair_variants(old, new, "//h1[@itemprop='name']")
    assert len(variants) == 2
    assert variants[1].candidates == ["//h1[@id='n']", "//h2[@id='n']", "//h1[@itemprop='name']"]  # type: ignore[union-attr]
    columns = {"title": COL.format("Title")}
    rec_old = RecordListField(container=f"//table[{cls('pubs')}]//tr[td]", columns=columns)
    rec_new = RecordListField(container=f"//table[{cls('papers')}]//tr[td]", columns=columns)
    rec = repair_variants(rec_old, rec_new, rec_old.container)
    assert len(rec) == 2 and isinstance(rec[1], RecordListField)
    assert rec[1].container == rec_old.container
    assert rec[1].container_alternates == [rec_new.container]
    kv_old = KeyValueField(pairs=f"//dl[{cls('a')}]/dt", value="./following-sibling::dd[1]")
    kv_new = kv_old.model_copy(update={"pairs": f"//dl[{cls('b')}]/dt"})
    kv = repair_variants(kv_old, kv_new, kv_old.pairs)
    assert len(kv) == 2 and kv[1].pairs_alternates == [kv_new.pairs]  # type: ignore[union-attr]
    other_cols = rec_new.model_copy(update={"columns": {"t": "./td[@class='t']"}})
    assert len(repair_variants(rec_old, other_cols, rec_old.container)) == 1


def test_describe_nodes() -> None:
    tree = clean_html("<html><body><p>a</p><p>b</p><p>c</p><p>d</p></body></html>")
    assert describe_nodes([]) == "0 nodes"
    assert describe_nodes(list(tree.iter("p"))) == "4 node(s): 'a', 'b', 'c' (+1 more)"
    assert describe_nodes(["x"]) == "1 node(s): 'x'"


def test_generation_input_shapes(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "run").ensure()
    assert load_generation(run) == {}
    write_json(run.generation_json, {"t_a": {"status": "ok"}, "junk": 3})
    assert load_generation(run) == {"t_a": {"status": "ok"}}
    write_json(run.generation_json, [1, 2])
    assert load_generation(run) == {}
    assert failed_fields({"failed_fields": ["a", {"name": "b", "type": "list"}, 7]}) == {
        "a": {},
        "b": {"name": "b", "type": "list"},
    }
    assert failed_fields({"failed_fields": {"a": None}}) == {"a": {}}
    assert failed_fields(None) == {}
