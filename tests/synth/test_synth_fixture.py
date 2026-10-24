"""The committed 200-page fixture site is exactly what the generator produces."""

from __future__ import annotations

import json
from pathlib import Path

from _synth_checks import sidecar_problems, tree_bytes

from ditto.synth import synth_main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/site"


def test_fixture_shape() -> None:
    meta = json.loads((FIXTURE / "synth.json").read_text())
    assert meta["n_pages"] == 200
    assert len(list((FIXTURE / "site").glob("*.html"))) == 200
    sizes = sorted(meta["template_pages"].values())
    main = [n for n in sizes if n >= 20]
    tail = [n for n in sizes if n < 20]
    assert len(main) == 4 and all(25 <= n <= 70 for n in main)
    assert 5 <= sum(tail) <= 20
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    assert total < 2_200_000
    assert (FIXTURE / "README.md").exists()


def test_fixture_sidecars_match_pages() -> None:
    for truth_path in sorted((FIXTURE / "truth").glob("p*.json"))[:60]:
        truth = json.loads(truth_path.read_text())
        raw = (FIXTURE / "site" / f"{truth['page']}.html").read_bytes()
        assert sidecar_problems(raw, truth) == []


def test_fixture_regenerates_identically(tmp_path: Path) -> None:
    meta = json.loads((FIXTURE / "synth.json").read_text())
    synth_main(ROOT / "eval/synth/fixture.yaml", meta["seed"], tmp_path)
    assert tree_bytes(tmp_path) == tree_bytes(FIXTURE)
