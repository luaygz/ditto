from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest
from baseline_helpers import load_module

from ditto.evaluation.baseline_api import NotRun, PageInput
from ditto.evaluation.datasets import swde
from ditto.schema import FieldsSchema
from ditto.specgen.models import Spec

SITES = ["auto-yahoo", "book-barnesandnoble", "movie-imdb", "nbaplayer-yahoo", "university-usnews"]
FIELDS_DIR = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "swde" / "fields"

# Hand-made snippets with the sites' markup structure and invented values.
NBA = """<html><head><title>Jo Doe - NBA</title></head><body>
<div id="ysp-player-nav" class="nba-player-nav"><div class="player-info"><ul>
<li class="player-name">Jo Doe</li><li class="team-name"><a href="/nba/teams/bos">Boston Celtics</a>
<div class="flyout"><ul><li><a href="/nba/teams/bos/news">News</a></li></ul></div></li></ul>
<ul class="stats"><li class="height"><strong>Height:</strong> 6-5</li>
<li class="weight"><strong>Weight:</strong> 205</li></ul></div></div></body></html>"""

USNEWS = """<html><body><div class="vcard"><h1 class="fn org">Example College</h1>
<div class="adr"><div class="tel"><span class="type">Work</span><span class="value">(800) 555-0100</span></div></div>
<div class="contact"><span class="url"> Web site: <span>http://www.example.edu</span></span></div></div>
<div id="profile-details"><table class="index"><tr><td class="label">Institutional Control: </td>
<td class="value">Private</td></tr><tr><td class="label">Year founded: </td><td class="value">1906</td></tr>
</table></div></body></html>"""

IMDB = """<html><body><table><tr><td id="overview-top"><h1 class="header">An Example Film
<span>(<a href="/year/1994/">1994</a>)</span></h1>
<div class="infobar">154 min - <a href="/genre/Crime">Crime</a> | <a href="/genre/Drama">Drama</a></div>
<div class="txt-block"><h4 class="inline">Director:</h4><a href="/name/nm1/">Ann Director</a></div>
</td></tr></table><div class="txt-block"><h4>Motion Picture Rating (<a href="/mpaa">MPAA</a>)</h4>
Rated R for violence. <span class="see-more inline"><a href="x">See all</a></span></div></body></html>"""


@pytest.fixture(scope="module")
def hw() -> ModuleType:
    return load_module("handwritten")


def test_specs_validate_and_match_swde_fields(hw: ModuleType) -> None:
    assert hw.available_sites() == SITES
    verticals = set()
    for site in SITES:
        spec = Spec.load(hw.SPEC_DIR / f"{site}.json")
        vertical, name = site.split("-", 1)
        verticals.add(vertical)
        assert spec.site == site and spec.status == "final"
        assert "build agent" in spec.generated_by.model_dump()["author"]
        fields = FieldsSchema.load(FIELDS_DIR / f"{vertical}.yaml")
        assert sorted(spec.fields) == sorted(fields.names())
        assert swde.split_of(swde.SiteRef(vertical, name)) == "test"
    assert len(verticals) == 5


@pytest.mark.parametrize(
    ("site", "html", "expected"),
    [
        (
            "nbaplayer-yahoo",
            NBA,
            {"name": "Jo Doe", "team": "Boston Celtics", "height": "6-5", "weight": "205"},
        ),
        (
            "university-usnews",
            USNEWS,
            {
                "name": "Example College",
                "phone": "(800) 555-0100",
                "website": "http://www.example.edu",
                "type": "Private",
            },
        ),
        (
            "movie-imdb",
            IMDB,
            {
                "title": "An Example Film",
                "director": "Ann Director",
                "genre": "Crime",
                "mpaa_rating": "Rated R for violence.",
            },
        ),
    ],
)
def test_run_baseline_extracts(
    hw: ModuleType, tmp_path: Path, site: str, html: str, expected: dict[str, str]
) -> None:
    vertical = site.split("-", 1)[0]
    schema = FieldsSchema.load(FIELDS_DIR / f"{vertical}.yaml")
    pages = [PageInput("0000", "u", html, "t"), PageInput("0001", "u1", "<html></html>", "t")]
    out = hw.run_baseline(pages, schema, out_dir=tmp_path, site=site, workers=1)
    assert out.predictions["0000"] == expected
    assert all(v is None for v in out.predictions["0001"].values())
    assert out.calls == 0 and out.cost_usd == 0.0 and out.spend_usd == 0.0 and out.pages_run == 2
    summary = json.loads((tmp_path / "handwritten.summary.json").read_text())
    assert summary["site"] == site


def test_extra_requested_field_is_empty(hw: ModuleType, tmp_path: Path) -> None:
    schema = FieldsSchema.from_obj({"name": "player", "salary": "salary"})
    out = hw.run_baseline(
        [PageInput("0", "u", NBA, None)], schema, out_dir=tmp_path, site="nbaplayer-yahoo"
    )
    assert out.predictions["0"] == {"name": "Jo Doe", "salary": None}
    assert any("salary" in n for n in out.notes)


def test_unknown_site_is_not_run(hw: ModuleType, tmp_path: Path) -> None:
    schema = FieldsSchema.from_obj({"name": "x"})
    with pytest.raises(NotRun, match="no hand-written wrapper"):
        hw.run_baseline([], schema, out_dir=tmp_path, site="auto-aol")
    with pytest.raises(NotRun, match="needs the site name"):
        hw.run_baseline([], schema, out_dir=tmp_path)
