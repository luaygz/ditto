"""Fixtures for the LiveWeb-IE tests: a tiny release laid out like the real one.

Ground-truth lines and values are copied from the real release files (sbY99/LiveWeb-IE@34b5649),
trimmed to a few pages.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "eval" / "datasets" / "livewebie"

D1 = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
D2 = "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html"
D3 = "https://books.toscrape.com/catalogue/soumission_998/index.html"
L1 = "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
Q1 = "http://quotes.toscrape.com/author/Albert-Einstein"

DETAIL_HTML = (
    "<html><head><title>A Light in the Attic | Books to Scrape</title></head><body>"
    '<div class="product_main"><h1>A Light in the Attic</h1>'
    '<p class="price_color">\u00a351.77</p></div>'
    '<div class="item active"><img src="../../media/cache/fe/72/fe72f0532301ec28892ae79a629a293c.jpg"'
    ' alt="A Light in the Attic"></div></body></html>'
)
LISTING_HTML = (
    "<html><body><ol class='row'>"
    "<li><article class='product_pod'><h3><a href='../../../its-only-the-himalayas_981/index.html'"
    " title=\"It's Only the Himalayas\">It's Only the Himalayas</a></h3></article></li>"
    "<li><article class='product_pod'><h3><a href='../../../full-moon-over-noahs-ark_811/index.html'"
    ' title="Full Moon over Noah\u2019s Ark">Full Moon over Noah\u2019s ...</a></h3></article></li>'
    "</ol></body></html>"
)


def load_script(name: str) -> ModuleType:
    """Import ``eval/datasets/livewebie/<name>.py`` under a unique module name."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    mod_name = f"livewebie_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _gt(domain: str, website: str, gid: str, label: str, rows: list[tuple[str, list[str]]]) -> str:
    total = sum(len(v) for _, v in rows)
    lines = [f"{domain} {website} {gid} {label}", f"{len(rows)} {total} {total}"]
    for url, vals in rows:
        lines.append(f'"{url}" {len(vals)} ' + "\t".join(vals))
    return "\n".join(lines) + "\n"


def _q(
    website_id: str,
    website: str,
    key: str,
    gid: str,
    qid: str,
    query: str,
    attrs: list[str],
    typ: str,
    domain: str,
) -> dict[str, object]:
    return {
        "website_id": website_id,
        "website": website,
        "group_id": gid,
        "group_url": f"./{key}/group/group.jsonl",
        "query_id": qid,
        "query": query,
        "attribute": attrs,
        "type": typ,
        "label": {a: f"./groundtruth/{domain}/{website}-{gid}-{a}.txt" for a in attrs},
        "sample_id": f"{website_id}_{gid}_{qid}",
    }


def build_release(tmp_path: Path) -> Path:
    """Write the tiny release under ``tmp_path/data/livewebie/release/LiveWeb_IE``."""
    root = tmp_path / "data" / "livewebie" / "release" / "LiveWeb_IE"
    books, quotes = "books.toscrape.com", "quotes.toscrape.com"
    dataset = {
        "booktoscrape": [
            _q(
                "w_014",
                books,
                "booktoscrape",
                "g_000",
                "q_000",
                "Extract the title of the book",
                ["title"],
                "type_1",
                "book",
            ),
            _q(
                "w_014",
                books,
                "booktoscrape",
                "g_000",
                "q_001",
                "Scrape the book stock status and associated image",
                ["title", "book_image"],
                "type_2",
                "book",
            ),
            _q(
                "w_014",
                books,
                "booktoscrape",
                "g_002",
                "q_000",
                "Get all the titles of the books",
                ["title"],
                "type_3",
                "book",
            ),
            _q(
                "w_014",
                books,
                "booktoscrape",
                "g_002",
                "q_001",
                "Return all book titles, prices, and URLs",
                ["title", "book_link"],
                "type_4",
                "book",
            ),
        ],
        "quotestoscrape": [
            _q(
                "w_013",
                quotes,
                "quotestoscrape",
                "g_000",
                "q_000",
                "Fetch the quote author's name",
                ["author"],
                "type_1",
                "other",
            ),
        ],
    }
    root.mkdir(parents=True)
    (root / "dataset.json").write_text(json.dumps(dataset, indent=1))
    groups = root / "booktoscrape" / "group"
    groups.mkdir(parents=True)
    (groups / "group.jsonl").write_text(
        json.dumps({"g_000": {"group_url": [D1, D2, D3, D1]}})
        + "\n"
        + json.dumps({"g_002": {"group_url": [L1]}})
        + "\n"
    )
    qgroups = root / "quotestoscrape" / "group"
    qgroups.mkdir(parents=True)
    (qgroups / "group.jsonl").write_text(json.dumps({"g_000": {"group_url": [Q1]}}) + "\n")
    gt = root / "groundtruth"
    (gt / "book").mkdir(parents=True)
    (gt / "other").mkdir(parents=True)
    (gt / "book" / f"{books}-g_000-title.txt").write_text(
        _gt(
            "book",
            books,
            "g_000",
            "book_title",
            [(D1, ["A Light in the Attic"]), (D2, ["Tipping the Velvet"]), (D3, [])],
        )
    )
    (gt / "book" / f"{books}-g_000-book_image.txt").write_text(
        _gt(
            "book",
            books,
            "g_000",
            "book_image",
            [
                (D1, ["../../media/cache/fe/72/fe72f0532301ec28892ae79a629a293c.jpg"]),
                (D2, ["../../media/cache/08/e9/08e94f3731d7d6b760dfbfbc02ca5c62.jpg"]),
            ],
        )
    )
    (gt / "book" / f"{books}-g_002-title.txt").write_text(
        _gt(
            "book",
            books,
            "g_002",
            "book_title",
            [(L1, ["It's Only the Himalayas", "Full Moon over Noah's Ark"])],
        )
    )
    (gt / "book" / f"{books}-g_002-book_link.txt").write_text(
        _gt(
            "book",
            books,
            "g_002",
            "book_link",
            [
                (
                    L1,
                    [
                        "../../../its-only-the-himalayas_981/index.html",
                        "../../../full-moon-over-noahs-ark_811/index.html",
                    ],
                )
            ],
        )
    )
    (gt / "other" / f"{quotes}-g_000-author.txt").write_text(
        _gt("other", quotes, "g_000", "quote_author", [(Q1, ["Albert Einstein"])])
    )
    return root
