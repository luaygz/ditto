"""Rendering knobs: absent-field renderings, key-value styles, markup styles."""

from __future__ import annotations

import random
from functools import lru_cache

import pytest
from lxml import html

from ditto.fingerprint.dom import node_text
from ditto.synth import SynthParams
from ditto.synth.page import PageData, make_page_data, truth_record
from ditto.synth.plan import SitePlan, build_plan
from ditto.synth.render import content_tree, serialize
from ditto.synth.values import build_pools

SEED = 131


@lru_cache(maxsize=32)
def _plan(**kw: object) -> SitePlan:
    base: dict[str, object] = {
        "templates": 1,
        "tail_templates": 0,
        "near_duplicate_pairs": 0,
        "total_pages": 30,
        "optional_fields": 3,
    }
    base.update(kw)
    return build_plan(SynthParams.model_validate(base), SEED)


def _exemplar(plan: SitePlan, t_index: int = 0) -> PageData:
    t = plan.templates[t_index]
    return make_page_data(
        t, plan.params, build_pools(SEED), random.Random(7), host=plan.host, exemplar=True
    )


def _render(plan: SitePlan, d: PageData, t_index: int = 0) -> html.HtmlElement:
    out, _ = serialize(content_tree(plan.templates[t_index], d))
    return html.fragment_fromstring(out)


def _texts(root: html.HtmlElement) -> list[str]:
    return [node_text(e) for e in root.iter() if isinstance(e.tag, str)]


@pytest.mark.parametrize("markup", ["server", "react", "table"])
@pytest.mark.parametrize("mode", ["omitted", "empty", "na", "hidden"])
def test_absent_renderings(mode: str, markup: str) -> None:
    plan = _plan(absent_rendering=mode, markup_style=markup)
    t = plan.templates[0]
    assert t.absent == mode and t.markup == markup
    d = _exemplar(plan)
    s = t.optional_scalars()[0]
    value = d.scalars[s.name]
    assert value is not None
    present = _render(plan, d)
    d.scalars[s.name] = None
    absent = _render(plan, d)
    n_present, n_absent = len(list(present.iter())), len(list(absent.iter()))
    assert value in _texts(present) and value not in _texts(absent)
    assert truth_record(t, d, "p0", "u")["fields"][s.name] is None
    if mode == "omitted":
        assert n_absent < n_present
        return
    assert n_absent == n_present  # node kept, only its text changes
    if mode == "na":
        assert "N/A" in _texts(absent)
    if mode == "hidden":
        hidden = [e for e in absent.iter() if "hidden" in (e.get("class") or "").split()]
        assert hidden and hidden[0].get("aria-hidden") == "true"
        assert all(node_text(e) == s.label or s.label in node_text(e) for e in hidden)


@pytest.mark.parametrize("style", ["dl", "table", "div"])
def test_kv_styles(style: str) -> None:
    plan = _plan(kv_style=style, kv_blocks=2)
    t = plan.templates[0]
    assert [b.style for b in t.kvs] == [style, style]
    d = _exemplar(plan)
    root = _render(plan, d)
    by_text = {}
    for e in root.iter():
        if isinstance(e.tag, str) and len(e) == 0:
            by_text.setdefault(node_text(e), e)
    for block in t.kvs:
        for key, value in d.kv[block.name]:
            key_el = by_text[key]
            val_el = key_el.getnext()
            assert val_el is not None and node_text(val_el) == value
            parent = key_el.getparent()
            assert parent is not None
            if style == "dl":
                assert (key_el.tag, val_el.tag, parent.tag) == ("dt", "dd", "dl")
            elif style == "table":
                assert parent.tag == "tr" and key_el.tag in ("th", "td") and val_el.tag == "td"
            else:
                assert parent.tag == "div" and (key_el.tag, val_el.tag) == ("span", "span")


def test_record_tables_have_header_cells_and_optional_columns() -> None:
    plan = _plan(record_tables=3, optional_columns=2, table_rows=(1, 4))
    t = plan.templates[0]
    assert len(t.tables) == 3
    d = _exemplar(plan)
    root = _render(plan, d)
    for tab in t.tables:
        assert sum(c.optional for c in tab.columns) == 2
        headers = [node_text(th) for th in root.iter("th")]
        assert all(c.label in headers for c in tab.columns)
    tab = t.tables[0]
    d.tables[tab.name].rows = []
    empty = _render(plan, d)
    assert f"No {tab.heading.lower()} yet." in _texts(empty)


def test_markup_styles() -> None:
    server = _plan(markup_style="server")
    root = _render(server, _exemplar(server))
    assert root.tag in ("article", "div", "section") and root.find(".//h1") is not None
    assert root.find(".//section") is not None

    react = _plan(markup_style="react", kv_style="div")
    t = react.templates[0]
    root = _render(react, _exemplar(react))
    tags = {e.tag for e in root.iter() if isinstance(e.tag, str)}
    assert not tags & {"section", "article", "header", "ul", "ol", "li", "h2", "aside", "dl"}
    assert root.tag == "div" and t.react_depth >= 1

    table = _plan(markup_style="table")
    root = _render(table, _exemplar(table))
    assert root.tag == "table" and root.find("tr") is not None


def test_mixed_styles_cover_all_options() -> None:
    plan = _plan(
        templates=24,
        total_pages=500,
        markup_style="mixed",
        absent_rendering="mixed",
        kv_style="mixed",
    )
    assert {t.markup for t in plan.templates} == {"server", "react", "table"}
    assert {t.absent for t in plan.templates} == {"omitted", "empty", "na", "hidden"}
    assert {b.style for t in plan.templates for b in t.kvs} == {"dl", "table", "div"}
    assert plan.chrome.style == "server"
