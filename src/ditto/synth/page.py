"""Per-page values of a synthetic template and their ground-truth sidecar form."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ditto.synth.blueprint import TemplateDef
from ditto.synth.params import SynthParams
from ditto.synth.values import Pools, generate, slugify


@dataclass
class TableData:
    columns: list[str]  # names of the columns present on this page, in display order
    rows: list[list[str]]


@dataclass
class PageData:
    """Everything a page shows. ``scalars[name] is None`` means the optional field is absent."""

    title: str
    scalars: dict[str, str | None]
    lists: dict[str, list[str]]
    tables: dict[str, TableData]
    kv: dict[str, list[tuple[str, str]]]
    related: list[tuple[str, str]]


def make_page_data(
    t: TemplateDef,
    params: SynthParams,
    pools: Pools,
    rng: random.Random,
    *,
    host: str,
    exemplar: bool = False,
) -> PageData:
    """Draw one page's values. ``exemplar`` renders every optional part (all optional fields and
    columns, two table rows, every key-value key) for structural measurements."""
    ctx = t.value_ctx

    def gen(name: str) -> str:
        return generate(name, rng, pools, ctx)

    title = gen(t.title.gen)
    scalars: dict[str, str | None] = {}
    for s in t.scalars:
        if s.optional and not exemplar and rng.random() >= s.presence:
            scalars[s.name] = None
        else:
            scalars[s.name] = gen(s.gen)
    lists: dict[str, list[str]] = {}
    for lst in t.lists:
        n = 3 if exemplar else rng.randint(*params.list_items)
        items: list[str] = []
        for _ in range(4 * n):  # distinct items, like real lists; small vocabularies may cap n
            if len(items) == n:
                break
            item = gen(lst.gen)
            if item not in items:
                items.append(item)
        lists[lst.name] = items
    tables: dict[str, TableData] = {}
    for tab in t.tables:
        n_rows = 2 if exemplar else rng.randint(*params.table_rows)
        cols = [c for c in tab.columns if exemplar or not c.optional or rng.random() < c.presence]
        rows = [[gen(c.gen) for c in cols] for _ in range(n_rows)]
        tables[tab.name] = TableData([c.name for c in cols], rows)
    kv: dict[str, list[tuple[str, str]]] = {}
    for block in t.kvs:
        if exemplar:
            idx = list(range(len(block.keys)))
        else:
            lo, hi = params.kv_keys
            k = rng.randint(min(lo, len(block.keys)), min(hi, len(block.keys)))
            idx = sorted(rng.sample(range(len(block.keys)), k))
        kv[block.name] = [(block.keys[i][0], gen(block.keys[i][1])) for i in idx]
    related: list[tuple[str, str]] = []
    if t.related:
        for _ in range(3 if exemplar else rng.randint(3, 5)):
            text = gen(t.title.gen)
            related.append((text, page_url(t, host, rng.randint(10000, 9999999), text, rng)))
    return PageData(title, scalars, lists, tables, kv, related)


def page_url(t: TemplateDef, host: str, uid: int, title: str, rng: random.Random) -> str:
    """URL of a page of template ``t``; ``uid`` makes it unique within the site."""
    slug = slugify(title)
    seg = t.segment
    style = t.url_style
    if style == "id-slug":
        path = f"/{seg}/{uid}/{slug}"
    elif style == "id":
        path = f"/{seg}/{uid}"
    elif style == "slug":
        path = f"/{seg}/{slug}-{uid}/"
    elif style == "date-slug":
        path = f"/{seg}/{rng.randint(2016, 2026)}/{rng.randint(1, 12):02d}/{slug}-{uid}"
    elif style == "html":
        path = f"/{seg}/{slug}-{uid}.html"
    elif style == "query":
        path = f"/{seg}.php?id={uid}"
    else:
        path = f"/{seg}/{slug}-{uid}"
    return f"https://{host}{path}"


def truth_record(t: TemplateDef, d: PageData, page: str, url: str) -> dict[str, Any]:
    """The sidecar of one page (contract layout in ``local://contracts.md``)."""
    fields: dict[str, str | None] = {t.title.name: d.title}
    fields.update((s.name, d.scalars[s.name]) for s in t.scalars)
    records: dict[str, list[dict[str, str | None]]] = {}
    for tab in t.tables:
        data = d.tables[tab.name]
        pos = {name: i for i, name in enumerate(data.columns)}
        records[tab.name] = [
            {c.name: (row[pos[c.name]] if c.name in pos else None) for c in tab.columns}
            for row in data.rows
        ]
    return {
        "page": page,
        "url": url,
        "template_id": t.tid,
        "fields": fields,
        "lists": {name: list(items) for name, items in d.lists.items()},
        "records": records,
        "kv": {name: dict(pairs) for name, pairs in d.kv.items()},
        "optional_present": [
            s.name for s in t.scalars if s.optional and d.scalars[s.name] is not None
        ],
    }
