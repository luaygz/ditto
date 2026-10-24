from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ditto.synth import SynthParams, load_params
from ditto.synth.plan import allocate, plan_sizes

ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_matches_model_defaults() -> None:
    assert SynthParams.load(ROOT / "eval/synth/default.yaml") == SynthParams()


def test_defaults_follow_the_brief() -> None:
    p = SynthParams()
    assert p.templates == 12
    assert p.pages_median == 2000
    assert p.optional_fields == 3
    assert p.optional_presence == (0.02, 0.95)
    assert p.record_tables == 1 and p.kv_blocks == 1
    assert p.hashed_class_share == 0.3
    assert p.boilerplate_share == 0.4
    assert p.near_duplicate_pairs == 1 and p.near_duplicate_jaccard == 0.7
    assert p.absent_rendering == "omitted"
    assert p.markup_style == "server"
    assert p.noise == ()


def test_load_params_overrides_and_fixture() -> None:
    p = load_params(ROOT / "eval/synth/fixture.yaml", {"noise": ["nbsp"]})
    assert p.templates == 4 and p.total_pages == 200 and p.noise == ("nbsp",)
    assert load_params() == SynthParams()


@pytest.mark.parametrize(
    "bad",
    [
        {"table_rows": (5, 2)},
        {"optional_presence": (0.0, 0.5)},
        {"near_duplicate_pairs": 7},
        {"noise": ["nbsp", "nbsp"]},
        {"pages_min": 100, "pages_max": 10},
        {"total_pages": 5},
        {"unknown_knob": 1},
        {"markup_style": "jsp"},
    ],
)
def test_invalid_params_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SynthParams.model_validate(bad)


def test_load_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        SynthParams.load(path)


@pytest.mark.parametrize("total", [10_000, 20_000, 100_000])
def test_total_pages_rescales_sizes(total: int) -> None:
    params = SynthParams(total_pages=total)
    main, tails = plan_sizes(params, 1)
    raw_main, raw_tails = plan_sizes(SynthParams(), 1)
    assert sum(main) + sum(tails) == total
    assert tails == raw_tails and all(t < 20 for t in tails)
    assert all(m >= 20 for m in main)
    # rescaling keeps the relative template sizes (up to rounding / bounds)
    big = max(range(len(main)), key=main.__getitem__)
    small = min(range(len(main)), key=main.__getitem__)
    assert big == max(range(len(raw_main)), key=raw_main.__getitem__)
    assert main[big] / main[small] == pytest.approx(raw_main[big] / raw_main[small], rel=0.05)


def test_unscaled_sizes_are_lognormal_and_clamped() -> None:
    main, tails = plan_sizes(SynthParams(templates=200, tail_templates=0), 3)
    assert len(main) == 200 and not tails
    assert all(20 <= m <= 60000 for m in main)
    median = sorted(main)[100]
    assert 1200 < median < 3300


def test_allocate_respects_bounds_and_total() -> None:
    sizes = allocate(1000, [1, 1, 100, 1000], 50, 600)
    assert sum(sizes) == 1000
    assert all(50 <= s <= 600 for s in sizes)
    assert sizes[3] == 600
    with pytest.raises(ValueError):
        allocate(10, [1, 1], 20, 30)
    assert allocate(0, [], 1, 2) == []
