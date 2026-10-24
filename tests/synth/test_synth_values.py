from __future__ import annotations

import random

import pytest
from lxml import html

from ditto.fingerprint.dom import canonical_classes, node_text
from ditto.specgen.typecheck import check_value
from ditto.synth import vocab as V
from ditto.synth.blueprint import class_is_hashed, ensure_semantic, hashed_class, semantic_class
from ditto.synth.render import WS_MODES, el, entity_escaper, serialize
from ditto.synth.values import (
    CURRENCIES,
    DATE_FORMATS,
    GENERATORS,
    TIME_FORMATS,
    ValueCtx,
    build_pools,
    generate,
    slugify,
)
from ditto.textutil import normalize_ws


def _all_protos() -> list[tuple[str, str]]:
    """(generator, value type) of every field, column and kv value in the catalogs."""
    out: list[tuple[str, str]] = []
    for k in V.KINDS:
        out.append((k.title.gen, k.title.vtype))
        out += [(f.gen, f.vtype) for f in k.scalars]
        out += [(x.gen, x.vtype) for x in k.lists]
        out += [(c.gen, c.vtype) for t in k.tables for c in t.columns + t.extra]
        out += [(g, "text") for kv in k.kvs for _, g in kv.keys]
    out += [(f.gen, f.vtype) for f in V.GENERIC_SCALARS]
    out += [(x.gen, x.vtype) for x in V.GENERIC_LISTS]
    out += [(c.gen, c.vtype) for t in V.GENERIC_TABLES for c in t.columns + t.extra]
    out += [(c.gen, c.vtype) for c in V.GENERIC_COLUMNS]
    out += [(g, "text") for kv in V.GENERIC_KVS for _, g in kv.keys]
    return out


def test_every_catalog_generator_exists() -> None:
    missing = {g for g, _ in _all_protos() if g not in GENERATORS}
    assert not missing


def test_generated_values_are_clean_and_typed() -> None:
    pools = build_pools(101)
    rng = random.Random(5)
    for gen, vtype in set(_all_protos()):
        for n in range(12):
            ctx = ValueCtx(
                currency=n % len(CURRENCIES),
                date_fmt=DATE_FORMATS[n % len(DATE_FORMATS)],
                time_fmt=TIME_FORMATS[n % 2],
            )
            value = generate(gen, rng, pools, ctx)
            assert value and normalize_ws(value) == value, (gen, value)
            assert check_value(vtype, value), (gen, vtype, value)


def test_catalog_names_unique_per_kind() -> None:
    for k in V.KINDS:
        names = [k.title.name] + [f.name for f in k.scalars]
        assert len(names) == len(set(names)), k.kind
        for kv in k.kvs:
            keys = [key for key, _ in kv.keys]
            assert len(keys) == len(set(keys)), kv.name
        for t in k.tables:
            cols = [c.name for c in t.columns + t.extra]
            assert len(cols) == len(set(cols)), t.name


def test_pools_are_deterministic_and_seeded() -> None:
    assert build_pools(101) == build_pools(101)
    a, b = build_pools(101), build_pools(102)
    assert a.first_names != b.first_names
    assert len(a.companies) > 100 and len(a.words) > 500


def test_slugify() -> None:
    assert slugify("José García & Co.") == "jose-garcia-co"
    assert slugify("!!!") == "item"


def test_semantic_classes_survive_canonicalization() -> None:
    for prefix in ("pdp", "product-alt", "listing", "item"):
        for naming in ("kebab", "bem", "camel"):
            for role in ("price", "seller_email", "tags-item", "row", "flex", "content"):
                cls = semantic_class(prefix, naming, role)  # type: ignore[arg-type]
                assert canonical_classes(cls) == [cls], cls
    assert ensure_semantic("flex") == "s-flex"


@pytest.mark.parametrize("markup", ["server", "react", "table"])
def test_hashed_classes_are_dropped(markup: str) -> None:
    rng = random.Random(3)
    for _ in range(500):
        value = hashed_class(rng, markup, "price-value")  # type: ignore[arg-type]
        assert canonical_classes(value) == [], value
        assert class_is_hashed(value)
    assert not class_is_hashed("pdp-price mt-2")
    assert not class_is_hashed("")


def _tree():
    return el(
        "div",
        "outer",
        [
            el("h1", "", ["Café & Bar <b>"]),
            el("p", "", [el("strong", "", ["Price:"]), " ", el("span", "v", ["€\xa012,50"])]),
            el("ul", "", [el("li", "", ["one"]), el("li", "", ["two 'quoted' \"x\""])]),
            el("input", "", None, ' type="text"'),
        ],
    )


def test_writer_modes_keep_visible_text() -> None:
    texts = []
    for mode in WS_MODES:
        out, count = serialize(_tree(), mode)
        assert count == 9
        root = html.fragment_fromstring(out)
        # inter-tag whitespace may differ between modes; text-bearing elements may not
        tags = ("h1", "p", "strong", "span", "li")
        texts.append([node_text(e) for tag in tags for e in root.iter(tag)])
        if mode == "minified":
            assert "\n" not in out
    assert texts[0] == texts[1] == texts[2]
    assert texts[0][0] == "Café & Bar <b>"
    assert texts[0][1] == "Price: € 12,50"


def test_entity_escaper_round_trips() -> None:
    rng = random.Random(9)
    esc = entity_escaper(rng)
    text = "Ünïcode & <tags> 'q' \"d\" m² – café\xa0ok plain words here"
    seen_numeric = False
    for _ in range(50):
        encoded = esc(text)
        seen_numeric |= "&#" in encoded
        out, _ = serialize(el("p", "", [text]), "pretty", esc=lambda _t, e=encoded: e)
        assert normalize_ws(html.fragment_fromstring(out).text_content()) == normalize_ws(text)
    assert seen_numeric
