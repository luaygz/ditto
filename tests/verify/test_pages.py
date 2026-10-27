"""Per-page observations on real HTML: exact path presence, selectors, rows and siblings."""

from __future__ import annotations

from pathlib import Path

import pytest
from verify_sitekit import cls, staff_page, text_path

from ditto.extract.runner import evaluate
from ditto.fingerprint.dom import clean_html
from ditto.specgen.models import KeyValueField, RecordListField
from ditto.verify import pages
from ditto.verify.pages import (
    PathProbe,
    RowsProbe,
    SelectorProbe,
    observe,
    observe_path,
    path_texts,
    sibling_count,
)

PAGE = staff_page(7, email=True, tags=3, pubs=2)


def test_observe_path_is_exact() -> None:
    tree = clean_html(PAGE)
    path = text_path(PAGE, "Person 007")
    assert path.endswith("h1[itemprop=name]")
    assert observe_path(path, tree) == 1
    # an extra canonical class changes the structural path: not the analysis's path any more
    other = clean_html(PAGE.replace('itemprop="name"', 'itemprop="name" class="big"'))
    assert observe_path(path, other) == 0
    li = text_path(PAGE, "topic7-0")
    assert observe_path(li, tree) == 3
    assert path_texts(li, tree) == ["topic7-0", "topic7-1", "topic7-2"]
    href = text_path(PAGE, "mailto:p7@example.org", attr="href")
    assert href.endswith(" @href")
    assert observe_path(href, tree) == 1
    assert observe_path("not a path ][", tree) == 0


def test_observe_selector_scalar_and_list() -> None:
    tree = clean_html(PAGE)
    email = observe(SelectorProbe("//a[starts-with(@href,'mailto:')]", (), "email"), tree)
    assert (email.nodes, email.values, email.type_ok) == (1, 1, 1.0)
    wrong = observe(SelectorProbe(f"//dl[{cls('profile')}]/dd[2]", (), "email"), tree)
    assert wrong.values == 1 and wrong.type_ok == 0.0
    several = observe(SelectorProbe(f"//dl[{cls('profile')}]/dd", (), None), tree)
    assert several.nodes == 3
    tags = observe(SelectorProbe(f"//ul[{cls('tags')}]/li", (), None, plural=True), tree)
    assert (tags.nodes, tags.values, tags.type_ok) == (3, 3, 1.0)
    same = observe(SelectorProbe("//ul/li", ("normalize_ws",), None, plural=True), tree)
    assert same.vhash == tags.vhash
    none = observe(SelectorProbe("//h3[@id='nope']", (), None), tree)
    assert none.values == 0 and none.type_ok < 0


def test_sibling_count_ignores_header_rows() -> None:
    tree = clean_html(PAGE)
    data_rows = evaluate(f"//table[{cls('pubs')}]//tr[td]", tree)
    assert len(data_rows) == 2
    assert sibling_count(data_rows) == 2
    assert sibling_count(data_rows[:1]) == 2


def test_observe_rows_record_list_and_key_value() -> None:
    tree = clean_html(PAGE)
    col = "./td[count(ancestor::table[1]//th[normalize-space()='{}']/preceding-sibling::th)+1]"
    pubs = RecordListField(
        container=f"//table[{cls('pubs')}]//tr[td]",
        columns={"title": col.format("Title"), "year": col.format("Year")},
        column_source="header_row",
        transforms={"year": ["cast(int)"]},
    )
    full = observe(RowsProbe(pubs, (("year", "number"),)), tree, sampled=True)
    assert (full.nodes, full.values, full.siblings, full.type_ok) == (2, 2, 2, 1.0)
    first = RowsProbe(pubs.model_copy(update={"container": f"//table[{cls('pubs')}]//tr[td][1]"}))
    one = observe(first, tree, sampled=True)
    assert (one.nodes, one.siblings) == (1, 2)
    assert observe(first, tree, sampled=False).siblings == -1
    kv = KeyValueField(
        pairs=f"//dl[{cls('profile')}]/dt", key=".", value="./following-sibling::dd[1]"
    )
    obs = observe(RowsProbe(kv), tree, sampled=True)
    assert set(obs.keys) == {"Email", "Phone", "Office"}
    assert obs.values == 3 and obs.type_ok == 1.0
    nothing = observe(RowsProbe(kv.model_copy(update={"pairs": "//dl[@id='x']/dt"})), tree)
    assert nothing.nodes == 0


def test_path_probe_observation() -> None:
    tree = clean_html(PAGE)
    obs = observe(PathProbe(text_path(PAGE, "topic7-1")), tree)
    assert obs.nodes == obs.values == 3


def test_small_templates_skip_the_worker_pool_without_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int | None] = []

    def fake_pmap(fn, items, *, workers=None, **kw):  # type: ignore[no-untyped-def]
        seen.append(workers)
        return iter([pages._empty(len(items), 1)])

    monkeypatch.setattr(pages, "pmap", fake_pmap)
    probe = SelectorProbe("//h1", (), None)
    monkeypatch.setenv("DITTO_MP_START", "spawn")
    pages.evaluate_probes(Path("/nonexistent"), list(range(40)), [probe], workers=8)
    pages.evaluate_probes(Path("/nonexistent"), list(range(pages.SPAWN_MIN_PAGES)), [probe])
    monkeypatch.setenv("DITTO_MP_START", "fork")
    pages.evaluate_probes(Path("/nonexistent"), list(range(40)), [probe], workers=8)
    assert seen == [1, None, 8]
