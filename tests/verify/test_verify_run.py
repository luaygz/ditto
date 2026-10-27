"""verify_run end to end on an ingested site with hand-built clusters, analysis and drafts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest
from verify_sitekit import (
    BASE,
    TID,
    FakeLLM,
    build_run,
    cls,
    ids_of,
    save_clusters,
    save_draft,
    staff_page,
    staff_table,
    write_generation,
)

from ditto.cluster.models import Clusters
from ditto.run import RunDir, read_json
from ditto.specgen.models import ListField, RecordListField, ScalarField, Spec
from ditto.verify import VerifyReport, repair_rejected, verify_run

PROFILE = f"//dl[{cls('profile')}]"
TAGS = f"//ul[{cls('tags')}]"
PUBS = f"//table[{cls('pubs')}]"
COL = "./td[count(ancestor::table[1]//th[normalize-space()='{}']/preceding-sibling::th)+1]"
NAME_FIX = {
    "fields": {
        "name": {
            "type": "scalar",
            "candidates": [f"//h1[@itemprop='name'] | //h1[{cls('title')}]", "//main/h1"],
            "transforms": ["normalize_ws"],
        }
    }
}


def _pages(n_a: int, n_b: int = 0, n_c: int = 0) -> list[tuple[str, str]]:
    pages = [
        (f"{BASE}a{i:03d}", staff_page(i, email=i % 2 == 0, tags=1 + i % 3, pubs=1 + i % 3))
        for i in range(n_a)
    ]
    pages += [(f"{BASE}b{i:03d}", staff_page(100 + i, variant="b")) for i in range(n_b)]
    pages += [(f"{BASE}c{i:03d}", staff_page(200 + i, variant="c")) for i in range(n_c)]
    return pages


def _site(tmp_path: Path, n_a: int, n_b: int = 0, n_c: int = 0) -> tuple[RunDir, list[list[int]]]:
    pages = _pages(n_a, n_b, n_c)
    run = build_run(tmp_path, pages)
    urls = [u for u, _ in pages]
    groups = [ids_of(run, urls[:n_a])]
    if n_b:
        groups.append(ids_of(run, urls[n_a : n_a + n_b]))
    if n_c:
        groups.append(ids_of(run, urls[n_a + n_b :]))
    save_clusters(run, groups)
    total = n_a + n_b + n_c
    staff_table(
        run,
        pages[0][1],
        sample_b=pages[n_a][1] if n_b else None,
        sample_c=pages[n_a + n_b][1] if n_c else None,
        p_a=n_a / total,
        size=total,
    )
    return run, groups


def _fields(name_candidates: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "name": {
            "type": "scalar",
            "candidates": name_candidates or ["//main/h1", "//h1[@itemprop='name']"],
            "source_candidates": ["c0", "c1", "c7"],
        },
        "email": {
            "type": "scalar",
            "candidates": [f"{PROFILE}/dd[1]", "//a[starts-with(@href,'mailto:')]"],
            "type_check": "email",
            "source_candidates": ["c2"],
        },
        "phone": {
            "type": "scalar",
            "candidates": [
                f"{PROFILE}/dd",
                "//dt[normalize-space()='Phone']/following-sibling::dd[1]",
            ],
            "source_candidates": ["c3"],
        },
        "tags": {
            "type": "list",
            "candidates": [f"{TAGS}/li[1]", f"{TAGS}/li"],
            "source_candidates": ["c4"],
        },
        "pubs": {
            "type": "record_list",
            "container": f"{PUBS}//tr[td][1]",
            "container_alternates": [f"{PUBS}//tr[td]"],
            "columns": {"title": COL.format("Title"), "year": COL.format("Year")},
            "column_source": "header_row",
            "transforms": {"year": ["cast(int)"]},
            "source_candidates": ["t0"],
        },
        "details": {
            "type": "key_value",
            "pairs": f"{PROFILE}/dt",
            "key": ".",
            "value": "./following-sibling::dd[1]",
            "source_candidates": ["c6"],
        },
    }
    fields.update(extra)
    return fields


def test_winners_stats_and_final_spec(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 24)
    save_draft(run, _fields())
    reports = verify_run(run, max_repair_calls=0, workers=1, merge_split=False)
    spec = Spec.load(run.spec(TID))
    assert spec.status == "final"
    report = reports[TID]
    assert VerifyReport.load(run.verify_report(TID)).fields.keys() == spec.fields.keys()

    name = spec.fields["name"]
    assert isinstance(name, ScalarField)
    assert name.selected == 1  # tie on score -> fewer positional steps
    assert "fewer positional steps" in (name.selection_reason or "")
    assert name.fallbacks == [0]
    assert not name.optional and name.stats.presence_rate == 1.0

    email = spec.fields["email"]
    assert isinstance(email, ScalarField) and email.selected == 1
    assert email.optional  # analysis path on 50% of pages
    assert email.stats.presence_rate == 0.5 and email.stats.expected_presence == 0.5
    assert email.fallbacks == []  # dd[1] disagrees with the winner where both match
    dd1 = report.fields["email"].candidates[0]
    assert dd1.type_check_pass == 0.5 and dd1.score == 0.5

    phone = spec.fields["phone"]
    assert isinstance(phone, ScalarField) and phone.selected == 1
    assert report.fields["phone"].candidates[0].multiplicity == 0.0

    tags = spec.fields["tags"]
    assert isinstance(tags, ListField) and tags.selected == 1
    first, every = report.fields["tags"].candidates[:2]
    assert first.multiplicity == every.multiplicity == 1.0  # li[1] reads part of the path...
    assert first.positional_steps > every.positional_steps  # ... and loses on its index

    pubs = spec.fields["pubs"]
    assert isinstance(pubs, RecordListField)
    assert pubs.container == f"{PUBS}//tr[td]"
    assert pubs.container_alternates == [f"{PUBS}//tr[td][1]"]
    assert (pubs.stats.rows_min, pubs.stats.rows_max) == (1, 3)
    assert spec.fields["details"].stats.keys_seen == 3

    assert all(f.failures == 0 for f in report.fields.values())
    assert report.degraded == [] and report.repair_calls == 0
    assert all(f.stats.pages_scored == 24 for f in spec.fields.values())
    timings = read_json(run.timings_json)
    assert timings["verify"]["pages"] == 24


def test_failures_grouped_by_signature_group_and_repaired(tmp_path: Path) -> None:
    run, groups = _site(tmp_path, 20, n_b=8)
    save_draft(run, _fields(["//h1[@itemprop='name']", "//*[@itemprop='name']"]))
    llm = FakeLLM([NAME_FIX])
    reports = verify_run(run, max_repair_calls=3, llm=llm, workers=1, merge_split=False)
    assert len(llm.calls) == 1
    assert llm.kwargs[0] == {"prompt_version": "repair.v1", "template_id": TID}
    prompt = llm.user_text()
    assert "Signature group 1: 8 failing pages of 8" in prompt
    assert "missing" in prompt and "//h1[@itemprop='name']" in prompt
    assert "Person 1" in prompt  # expected value read from the analysis path on that page
    record = reports[TID].repairs[0]
    assert record.signature_group == 1 and record.page_id in groups[1]
    assert record.fields == ["name"] and record.accepted == ["name"]
    assert record.failures_before == {"name": 8} and record.failures_after == {"name": 0}
    spec = Spec.load(run.spec(TID))
    name = spec.fields["name"]
    assert isinstance(name, ScalarField) and not name.degraded
    assert name.stats.presence_rate == 1.0
    assert spec.generated_by.repair_calls == 1
    assert spec.generated_by.repair_prompt_version == "repair.v1"
    origins = {c.xpath: c.origin for c in reports[TID].fields["name"].candidates}
    assert origins[NAME_FIX["fields"]["name"]["candidates"][0]] == "repair"


def test_repair_cap_then_degraded(tmp_path: Path) -> None:
    run, _groups = _site(tmp_path, 20, n_b=8, n_c=5)
    save_draft(run, _fields(["//h1[@itemprop='name']", "//*[@itemprop='name']"]))
    useless = {
        "fields": {
            "name": {
                "type": "scalar",
                "candidates": ["//h1[@itemprop='name']", "//*[@itemprop='name']"],
            }
        }
    }
    llm = FakeLLM([useless])
    reports = verify_run(run, max_repair_calls=1, llm=llm, workers=1, merge_split=False)
    assert len(llm.calls) == 1
    report = reports[TID]
    assert report.repairs[0].signature_group == 1  # the group with the most failing pages
    assert report.repairs[0].accepted == []
    spec = Spec.load(run.spec(TID))
    name = spec.fields["name"]
    assert name.degraded and report.degraded == ["name"]
    assert name.stats.model_extra is not None
    assert name.stats.model_extra["failure_rate"] == round(13 / 33, 4)
    assert "degraded" in (name.selection_reason or "")
    assert report.fields["name"].failures_by_group == {"1": 8, "2": 5}


def test_repair_that_rewrites_passing_values_is_rejected(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 20, n_b=8)
    cut = ["regex('([0-9]+)$', 1)"]  # the draft reads only the number of "Person 7"
    save_draft(run, _fields(["//h1[@itemprop='name']", "//*[@itemprop='name']"]))
    draft = Spec.load(run.draft_spec(TID))
    name = draft.fields["name"].model_copy(update={"transforms": cut})
    draft.fields["name"] = name
    draft.save(run.draft_spec(TID))
    fixed = NAME_FIX["fields"]["name"]["candidates"]
    whole = {"fields": {"name": {"type": "scalar", "candidates": fixed, "transforms": []}}}
    llm = FakeLLM([whole])
    reports = verify_run(run, max_repair_calls=1, llm=llm, workers=1, merge_split=False)
    # the answer fills the 8 failing pages but replaces the 20 passing numbers with whole names
    assert reports[TID].repairs[0].accepted == []
    spec = Spec.load(run.spec(TID))
    assert spec.fields["name"].transforms == cut and spec.fields["name"].degraded

    run2, _ = _site(tmp_path / "kept", 20, n_b=8)
    save_draft(run2, _fields(["//h1[@itemprop='name']", "//*[@itemprop='name']"]))
    draft.save(run2.draft_spec(TID))
    kept = {"fields": {"name": {"type": "scalar", "candidates": fixed, "transforms": cut}}}
    reports = verify_run(
        run2, max_repair_calls=1, llm=FakeLLM([kept]), workers=1, merge_split=False
    )
    assert reports[TID].repairs[0].accepted == ["name"]
    assert not Spec.load(run2.spec(TID)).fields["name"].degraded


def test_no_llm_marks_failing_fields_degraded(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 20, n_b=8)
    save_draft(run, _fields(["//h1[@itemprop='name']", "//*[@itemprop='name']"]))
    reports = verify_run(run, max_repair_calls=3, llm=None, workers=1, merge_split=False)
    assert reports[TID].repairs == [] and reports[TID].degraded == ["name"]


def test_invalid_repair_answers_count_as_calls(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 20, n_b=8)
    save_draft(run, _fields(["//h1[@itemprop='name']", "//*[@itemprop='name']"]))
    bad = {"fields": {"name": {"type": "list", "candidates": ["//h1", "//h2"]}}}
    llm = FakeLLM(["not json", bad])
    reports = verify_run(run, max_repair_calls=1, llm=llm, workers=1, merge_split=False)
    assert len(llm.calls) == 2  # one request, one retry with the errors
    record = reports[TID].repairs[0]
    assert record.error and "InvalidResponse" in record.error
    assert "must keep type 'scalar'" in record.error
    assert reports[TID].repair_calls == 1


def test_path_induced_fallback_when_every_candidate_fails(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 12)
    save_draft(run, _fields(["//h1[@itemprop='title']", "//*[@itemprop='title']"]))
    reports = verify_run(run, max_repair_calls=0, workers=1, merge_split=False)
    fr = reports[TID].fields["name"]
    assert fr.path_induced and fr.failures == 0
    assert fr.candidates[fr.selected].origin == "path_induced"
    assert any("path-induced" in n for n in reports[TID].notes)
    spec = Spec.load(run.spec(TID))
    assert spec.apply(staff_page(3))["name"] == "Person 003"


def test_generation_failed_fields_built_from_paths(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 12)
    fields = _fields()
    for gone in ("tags", "pubs"):
        fields.pop(gone)
    save_draft(run, fields)
    write_generation(
        run,
        {
            TID: {
                "status": "partial",
                "failed_fields": {
                    "tags": {"type": "list", "source_candidates": ["c4"], "errors": ["bad xpath"]},
                    "pubs": {"type": "record_list", "source_candidates": ["t0"]},
                    "ghost": {"type": "scalar", "source_candidates": ["c99"], "errors": ["boom"]},
                },
            }
        },
    )
    reports = verify_run(run, max_repair_calls=0, workers=1, merge_split=False)
    spec = Spec.load(run.spec(TID))
    assert {"tags", "pubs"} <= set(spec.fields)
    assert reports[TID].fields["tags"].path_induced
    assert "ghost" in spec.not_present and "boom" in spec.not_present["ghost"]
    row = spec.apply(staff_page(5, tags=2, pubs=1))
    assert row["tags"] == ["topic5-0", "topic5-1"]
    assert row["pubs"] == [{"Title": "Paper 5.0", "Year": "2005"}]


def test_whole_template_failure_gets_fallback_spec(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 12)
    write_generation(
        run,
        {
            TID: {
                "status": "failed",
                "draft": None,
                "failed_fields": {"name": {"type": "scalar", "source_candidates": ["c0"]}},
            }
        },
    )
    reports = verify_run(run, max_repair_calls=0, workers=1, merge_split=False)
    spec = Spec.load(run.spec(TID))
    assert spec.generated_by.mode == "fallback" and reports[TID].spec_source == "fallback"
    assert spec.fields["name"].stats.presence_rate == 1.0


def test_split_moves_the_failing_signature_group(tmp_path: Path) -> None:
    run, groups = _site(tmp_path, 24, n_b=6)
    save_draft(run, _fields(["//h1[@itemprop='name']", "//*[@itemprop='name']"]))
    reports = verify_run(run, max_repair_calls=0, workers=1, merge_split=True)
    clusters = Clusters.load(run.clusters_json)
    assert len(clusters.clusters) == 2
    child = next(c for c in clusters.clusters if c.split_from == TID)
    assert sorted(child.page_ids) == sorted(groups[1]) and child.size == 6
    assert clusters.get(TID).size == 24
    assert next(g for g in clusters.signature_groups if g.id == 1).template_id == child.template_id
    assert child.url_examples and child.url_examples[0].startswith(BASE + "b")
    parent = Spec.load(run.spec(TID))
    assert not parent.fields["name"].degraded
    child_spec = Spec.load(run.spec(child.template_id))
    # the child is re-verified on its own pages: every copied candidate fails there, so the
    # path-induced fallback (from the variant-b analysis path) takes over
    assert child_spec.template_id == child.template_id
    assert not child_spec.fields["name"].degraded
    assert reports[child.template_id].fields["name"].path_induced
    assert child_spec.apply(staff_page(100, variant="b"))["name"] == "Person 100"
    events = reports[TID].merge_split
    assert events and events[0]["action"] == "split" and events[0]["fields"] == ["name"]
    assert reports[child.template_id].spec_source == "split_parent"
    assert reports[TID].n_pages == 24


def test_selector_errors_do_not_split(tmp_path: Path) -> None:
    # one markup variant, two signature groups; the name selector misses every page of the
    # second group although the analysis sees the name's path there: a selector error, which
    # repair handles, not a different template
    pages = [(f"{BASE}a{i:03d}", staff_page(i)) for i in range(24)]
    pages += [(f"{BASE}z{i:03d}", staff_page(500 + i)) for i in range(6)]
    run = build_run(tmp_path, pages)
    urls = [u for u, _ in pages]
    save_clusters(run, [ids_of(run, urls[:24]), ids_of(run, urls[24:])])
    staff_table(run, pages[0][1], size=30)
    miss = "[not(starts-with(normalize-space(),'Person 5'))]"
    save_draft(run, _fields([f"//h1[@itemprop='name']{miss}", f"//*[@itemprop='name']{miss}"]))
    reports = verify_run(run, max_repair_calls=0, workers=1, merge_split=True)
    assert reports[TID].fields["name"].failures == 6  # the misses are counted as failures
    assert [c.template_id for c in Clusters.load(run.clusters_json).clusters] == [TID]
    assert reports[TID].merge_split == []


def test_merge_carries_pages_into_the_larger_cluster(tmp_path: Path) -> None:
    pages = _pages(30)
    run = build_run(tmp_path, pages)
    urls = [u for u, _ in pages]
    big, small = ids_of(run, urls[:22]), ids_of(run, urls[22:])
    save_clusters(run, [big], extra=[("t_small", [small])])
    staff_table(run, pages[0][1], size=22)
    save_draft(run, _fields())
    reports = verify_run(run, max_repair_calls=0, workers=1, merge_split=True)
    clusters = Clusters.load(run.clusters_json)
    assert [c.template_id for c in clusters.clusters] == [TID]
    assert clusters.get(TID).merged_from == ["t_small"] and clusters.get(TID).size == 30
    assert reports[TID].merge_split[0]["absorbed"] == "t_small"
    assert Spec.load(run.spec(TID)).fields["name"].stats.pages_scored == 30


def test_rejected_field_goes_back_through_repair(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 12)
    fields = _fields()
    fields["phone"]["candidates"] = [
        "//dt[normalize-space()='Office']/following-sibling::dd[1]",
        f"{PROFILE}/dd[last()]",
    ]
    save_draft(run, fields)
    verify_run(run, max_repair_calls=0, workers=1, merge_split=False)
    fix = {
        "fields": {
            "phone": {
                "type": "scalar",
                "candidates": [
                    "//dt[normalize-space()='Phone']/following-sibling::dd[1]",
                    f"//dl[{cls('profile')}]/dt[normalize-space()='Phone']/following-sibling::dd[1]",
                ],
            }
        }
    }
    llm = FakeLLM([fix])
    out = repair_rejected(run, {TID: {"phone": "that is the office", "nope": "x"}}, llm=llm)
    assert "that is the office" in llm.user_text()
    report = out[TID]
    assert report.rejections["phone"]["repaired"] is True
    assert report.rejections["nope"]["repaired"] is False
    spec = Spec.load(run.spec(TID))
    assert spec.apply(staff_page(3))["phone"] == "+961 1 003 021"
    assert spec.generated_by.repair_calls == 1
    saved = orjson.loads(run.verify_report(TID).read_bytes())
    assert saved["rejections"]["phone"]["note"] == "that is the office"


def test_rejections_without_llm_are_recorded(tmp_path: Path) -> None:
    run, _ = _site(tmp_path, 12)
    save_draft(run, _fields())
    verify_run(run, max_repair_calls=0, workers=1, merge_split=False)
    out = repair_rejected(run, {TID: {"name": "wrong"}, "t_none": {"x": "y"}})
    assert out[TID].rejections["name"] == {
        "note": "wrong",
        "repaired": False,
        "reason": "no LLM available for repair",
    }
    assert "t_none" not in out


def test_parallel_workers_match_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _site(tmp_path, 40)
    monkeypatch.setattr("ditto.verify.pages.PAGES_PER_CHUNK", 8)
    save_draft(run, _fields())
    serial = verify_run(run, max_repair_calls=0, workers=1, merge_split=False)
    run.spec(TID).unlink()
    parallel = verify_run(run, max_repair_calls=0, workers=2, merge_split=False)
    for name in serial[TID].fields:
        a, b = serial[TID].fields[name], parallel[TID].fields[name]
        assert a.candidates == b.candidates


@pytest.mark.parametrize("bad", [[], {"fields": []}, {"fields": {"other": {}}}])
def test_repair_validator_rejects_bad_shapes(bad: Any) -> None:
    from ditto.verify.repair import make_validator

    spec = Spec.model_validate(
        {
            "template_id": TID,
            "site": "example.org",
            "generated_by": {"model": "m", "prompt_version": "p"},
            "fields": {"name": _fields()["name"]},
        }
    )
    with pytest.raises(ValueError):
        make_validator(spec, ["name"])(bad)
