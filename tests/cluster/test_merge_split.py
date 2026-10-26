"""Stage-3 step-5 merge and split rules on hand-built clusters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ditto.analyze.models import Candidate, CandidateTable
from ditto.cluster.merge_split import (
    MergePlan,
    apply_merges,
    apply_splits,
    merge_spec_fields,
    plan_merges,
    plan_splits,
    reaches,
    spec_presence,
    split_template_id,
    tolerance,
)
from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.ingest import PageStore, ingest
from ditto.specgen.models import Spec


def _clusters(spec: dict[str, list[list[int]]], min_size: int = 5) -> Clusters:
    clusters, groups = [], []
    for tid, members in spec.items():
        gids = []
        for pids in members:
            gids.append(len(groups))
            groups.append(
                SignatureGroup(
                    id=len(groups),
                    sig_hash=f"{len(groups):x}",
                    size=len(pids),
                    template_id=tid,
                    page_ids=pids,
                )
            )
        pages = sorted(p for g in members for p in g)
        clusters.append(
            Cluster(
                template_id=tid,
                size=len(pages),
                is_tail=len(pages) < min_size,
                signature_groups=gids,
                url_patterns=[f"/{tid}/{{n}}"],
                url_examples=[f"https://x.org/{tid}/1"],
                page_ids=pages,
            )
        )
    n = sum(c.size for c in clusters)
    return Clusters(
        n_pages=n, min_cluster_size=min_size, clusters=clusters, signature_groups=groups
    )


def _spec(tid: str, rates: dict[str, float], xpath: str = "//h1[@id='t']") -> Spec:
    fields = {
        name: {
            "type": "scalar",
            "candidates": [xpath, f"//h2[@id='{name}']"],
            "stats": {"presence_rate": p},
            "source_candidates": ["c0"],
        }
        for name, p in rates.items()
    }
    return Spec.model_validate(
        {
            "template_id": tid,
            "site": "x.org",
            "generated_by": {"model": "m", "prompt_version": "specgen.v1"},
            "fields": fields,
        }
    )


def test_split_when_failures_concentrate_in_one_group() -> None:
    page_ids = list(range(30))
    groups = [0] * 20 + [1] * 6 + [2] * 4
    price = np.array([True] * 20 + [False] * 5 + [True] + [True] * 4)
    title = np.ones(30, bool)
    plans = plan_splits("t_a", page_ids, groups, {"price": price, "title": title}, {1: "ab"})
    assert len(plans) == 1
    plan = plans[0]
    assert plan.group_ids == (1,) and plan.fields == ("price",)
    assert plan.page_ids == tuple(range(20, 26))
    assert plan.new_template_id == split_template_id("t_a", ["ab"])
    assert plan.new_template_id.startswith("t_") and len(plan.new_template_id) == 10
    assert plan.evidence["1:price"] == {"in_group": round(1 / 6, 4), "elsewhere": 1.0}
    assert plan.event()["action"] == "split"


def test_no_split_when_elsewhere_is_not_reliable_or_group_too_small() -> None:
    groups = [0] * 20 + [1] * 6
    weak = np.array([True] * 18 + [False] * 2 + [False] * 6)  # 90% elsewhere
    assert plan_splits("t_a", list(range(26)), groups, {"f": weak}, {}) == []
    tiny = np.array([True] * 20 + [False] * 2)
    assert plan_splits("t_a", list(range(22)), [0] * 20 + [1] * 2, {"f": tiny}, {}) == []
    unknown = np.array([True] * 20 + [False] * 6)
    assert plan_splits("t_a", list(range(26)), [0] * 20 + [-1] * 6, {"f": unknown}, {}) == []


def test_groups_with_the_same_profile_split_together() -> None:
    # each small group is < 50% while > 95% holds on the rest: both leave, as one cluster
    groups = [0] * 100 + [1] * 3 + [2] * 3
    f = np.array([True] * 100 + [False] * 6)
    plans = plan_splits("t_a", list(range(106)), groups, {"f": f}, {1: "a", 2: "b"})
    assert len(plans) == 1 and plans[0].group_ids == (1, 2)
    # two large failing groups: failures do not concentrate in one group, no split
    groups = [0] * 20 + [1] * 4 + [2] * 4
    f = np.array([True] * 20 + [False] * 8)
    assert plan_splits("t_a", list(range(28)), groups, {"f": f}, {}) == []


def test_apply_splits_updates_membership() -> None:
    clusters = _clusters({"t_a": [list(range(20)), list(range(20, 26))]})
    plan = plan_splits(
        "t_a",
        list(range(26)),
        [0] * 20 + [1] * 6,
        {"f": np.array([True] * 20 + [False] * 6)},
        {1: "1"},
    )[0]
    out = apply_splits(clusters, [plan], url_of=lambda pid: f"https://x.org/staff/{pid}")
    parent, child = out.get("t_a"), out.get(plan.new_template_id)
    assert parent.size == 20 and parent.signature_groups == [0] and not parent.is_tail
    assert child.size == 6 and child.split_from == "t_a" and child.signature_groups == [1]
    assert child.url_examples[0] == "https://x.org/staff/20"
    assert child.url_patterns
    assert out.template_of(21) == plan.new_template_id
    assert out.signature_groups[1].template_id == plan.new_template_id


def test_reaches_and_tolerance() -> None:
    assert tolerance(1.0, 20) == 0.05
    assert tolerance(0.5, 4) == 0.5
    own = {"title": 1.0, "price": 0.4}
    assert reaches(own, {"title": 0.95, "price": 0.3}, 20)
    assert not reaches(own, {"title": 0.8, "price": 0.4}, 20)
    assert not reaches(own, {"title": 1.0}, 20)  # price missing entirely
    assert not reaches({"a": 0.5}, {"a": 0.5}, 20)  # no anchor field
    assert not reaches(own, {"title": 1.0, "price": 0.4}, 0)


def test_plan_and_apply_merges() -> None:
    clusters = _clusters(
        {
            "t_big": [list(range(30))],
            "t_small": [list(range(30, 36))],
            "t_other": [list(range(36, 46))],
        }
    )
    specs = {"t_big": _spec("t_big", {"title": 1.0}), "t_other": _spec("t_other", {"name": 1.0})}
    samples = {
        "t_big": list(range(10)),
        "t_small": list(range(30, 36)),
        "t_other": list(range(36, 46)),
    }
    rates = {
        ("t_big", "t_small"): {"title": 1.0},
        ("t_big", "t_other"): {"title": 1.0},
        ("t_other", "t_big"): {"name": 0.0},  # mutual check fails for t_other
    }
    plans = plan_merges(clusters, specs, rates, samples)
    assert plans == [MergePlan("t_big", "t_small", plans[0].evidence)]
    assert plans[0].evidence["pages_sampled"] == 6
    merged = apply_merges(clusters, plans)
    big = merged.get("t_big")
    assert big.size == 36 and big.merged_from == ["t_small"]
    assert big.signature_groups == [0, 1]
    assert {c.template_id for c in merged.clusters} == {"t_big", "t_other"}
    assert merged.template_of(33) == "t_big"
    assert merged.signature_groups[1].template_id == "t_big"


def test_merge_spec_fields_carries_missing_fields() -> None:
    a = _spec("t_a", {"title": 1.0})
    b = _spec("t_b", {"title": 1.0, "price": 0.9})
    a = a.model_copy(update={"not_present": {"price": "not found"}})
    ta = CandidateTable(
        template_id="t_a",
        cluster_size=1,
        n_sampled=1,
        candidates=[
            Candidate.model_validate(
                {"id": "c5", "path": "html > body > h1", "class": "mandatory", "p": 1, "m": 0}
            )
        ],
    )
    tb = CandidateTable(
        template_id="t_b",
        cluster_size=1,
        n_sampled=1,
        candidates=[
            Candidate.model_validate(
                {"id": "c0", "path": "html > body > h1", "class": "mandatory", "p": 1, "m": 0}
            )
        ],
    )
    merged, added = merge_spec_fields(a, b, ta, tb)
    assert added == ["price"] and "price" not in merged.not_present
    assert merged.fields["price"].source_candidates == ["c5"]
    _, none = merge_spec_fields(a, b, None, None)
    assert none == ["price"]


def test_spec_presence_on_real_pages(tmp_path: Path) -> None:
    src = tmp_path / "site"
    src.mkdir()
    rows = []
    for i in range(8):
        body = f"<h1 id='t'>Item {i}</h1>" if i < 6 else f"<h2 id='x'>Other {i}</h2>"
        (src / f"p{i}.html").write_text(f"<html><body>{body}<p>p{i}</p></body></html>")
        rows.append(f"p{i}.html\thttps://x.org/p/{i}")
    (src / "urls.tsv").write_text("\n".join(rows) + "\n")
    ingest(src, tmp_path / "run", workers=1)
    store = PageStore.open(tmp_path / "run")
    ids = [store.lookup(f"https://x.org/p/{i}") for i in range(8)]
    a = [p for p in ids[:4] if p is not None]
    b = [p for p in ids[4:6] if p is not None]
    c = [p for p in ids[6:] if p is not None]
    specs = {"t_a": _spec("t_a", {"title": 1.0})}
    rates = spec_presence(tmp_path / "run", specs, {"t_a": a, "t_b": b, "t_c": c}, workers=1)
    assert rates[("t_a", "t_b")] == {"title": 1.0}
    assert rates[("t_a", "t_c")] == {"title": 0.0}
    assert ("t_a", "t_a") not in rates
