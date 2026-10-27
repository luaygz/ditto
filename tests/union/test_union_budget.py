"""The union page token budget (splicing most frequent candidates first within it, searching
past the sample only while budget is left, the hard cap), undoing splices that hide another
candidate, and rows whose own text is the key."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from lxml import html as lhtml
from union_fixtures import cand

from ditto.analyze.models import CandidateTable, Table
from ditto.union import UnionPage, build_union_page
from ditto.union.splice import cid_key

MAIN = "html > body > main"
EXTRAS = "abcdef"
LONG = "a sentence of words that keeps going " * 12


def extras_table() -> CandidateTable:
    """``c0`` the title on every page; ``c1``..``c6`` one optional block each, rarer and rarer."""
    candidates = [cand("c0", f"{MAIN} > h1", "mandatory", 1.0)]
    for i, x in enumerate(EXTRAS, start=1):
        candidates.append(cand(f"c{i}", f"{MAIN} > div.extra-{x} > p", "optional", 1.0 - i / 10))
    return CandidateTable(
        template_id="t_extras", cluster_size=40, n_sampled=40, candidates=candidates
    )


def extras_page(title: str, extras: str) -> str:
    blocks = "".join(f'<div class="extra-{x}"><p>{x}: {LONG}</p></div>' for x in extras)
    return f"<html><body><main><h1>{title}</h1>{blocks}</main></body></html>"


def test_most_frequent_candidates_are_spliced_first_within_the_budget() -> None:
    pages = {0: extras_page("Base", "")} | {
        i: extras_page(f"Page {i}", x) for i, x in enumerate(EXTRAS, start=1)
    }
    full = build_union_page(extras_table(), pages, min_tokens=0, max_tokens=10**6)
    assert len(full.spliced) == len(EXTRAS) - 1 and not full.budget_skipped
    cap = full.tokens // 4  # spliced units may add up to the cap: not all of them fit
    page = build_union_page(extras_table(), pages, min_tokens=0, max_tokens=cap)
    assert page.tokens <= cap
    missing = sorted(set(full.spliced), key=lambda c: -extras_table().candidates[int(c[1:])].p)
    assert page.spliced == missing[: len(page.spliced)]  # a prefix, by frequency
    assert page.budget_skipped == sorted(missing[len(page.spliced) :], key=cid_key)
    assert page.budget_skipped  # the budget ran out
    assert page.meta()["budget_skipped"] == page.budget_skipped
    assert any(line.startswith("splice budget of") for line in page.log)


def test_pages_past_the_sample_are_searched_only_while_budget_is_left() -> None:
    sample = {1: extras_page("One", "ab"), 2: extras_page("Two", "c")}
    consumed: list[int] = []

    def pool() -> Iterator[tuple[int, str]]:
        for pid, extras in ((3, "d"), (4, "ef")):
            consumed.append(pid)
            yield pid, extras_page(f"Page {pid}", extras)

    roomy = build_union_page(extras_table(), sample, min_tokens=0, more_pages=pool())
    assert consumed == [3, 4] and roomy.searched_pages == 2
    assert {"c4", "c5", "c6"} <= set(roomy.spliced) and roomy.missing == []

    consumed.clear()
    tight = build_union_page(extras_table(), sample, min_tokens=0, max_tokens=60, more_pages=pool())
    assert consumed == [] and tight.searched_pages == 0
    assert any("were not searched for: the splice budget is spent" in x for x in tight.log)
    assert tight.tokens <= 60


def test_hard_cap_holds_when_labels_of_value_less_tables_remain() -> None:
    dl = f"{MAIN} > dl.facts"
    table = CandidateTable(
        template_id="t_dl",
        cluster_size=30,
        n_sampled=30,
        candidates=[
            cand("c0", f"{MAIN} > h1", "mandatory", 1.0),
            # the key-value table has keys but no value candidate: its keys are kept as labels
            cand("c1", f"{dl} > dt", "label", 1.0, m=1.0, table_id="t0", table_role="kv_key"),
        ],
        tables=[
            Table(
                id="t0",
                kind="key_value",
                layout="dl",
                container_path=dl,
                row_path=f"{dl} > dt",
                key_candidate="c1",
            )
        ],
    )
    rows = "".join(f"<dt>Key number {i} of many</dt><dd>{i}</dd>" for i in range(300))
    page = f'<html><body><main><h1>T</h1><dl class="facts">{rows}</dl></main></body></html>'
    union = build_union_page(table, {0: page}, min_tokens=0, max_tokens=300)
    assert union.tokens <= 300
    assert "every remaining label" in union.dropped


def test_a_value_whose_key_is_its_row_text_is_spliced_with_the_row() -> None:
    facts = "html > body > div.film > ul.facts > li"
    kv = {"table_id": "t0", "table_role": "kv_value"}
    table = CandidateTable(
        template_id="t_film",
        cluster_size=30,
        n_sampled=30,
        candidates=[
            cand("c0", "html > body > div.film > h1", "mandatory", 1.0),
            cand("c1", facts, "label", 1.0, m=1.0, table_id="t0", table_role="kv_key"),
            cand("c2", f"{facts} > span", "mandatory", 1.0, column="Studio:", **kv),
            cand("c3", f"{facts} > span", "mandatory", 1.0, column="Rating:", **kv),
            cand("c4", f"{facts} > span", "optional", 0.4, column="Genre/Type:", **kv),
        ],
        tables=[
            Table(
                id="t0",
                kind="key_value",
                layout="pairs",
                container_path="html > body > div.film > ul.facts",
                row_path=facts,
                key_candidate="c1",
            )
        ],
    )

    def page(name: str, rows: list[tuple[str, str]]) -> str:
        items = "".join(f"<li>{k} <span>{v}</span></li>" for k, v in rows)
        return (
            f'<html><body><div class="film"><h1>{name}</h1><ul class="facts">{items}</ul>'
            "</div></body></html>"
        )

    pages = {
        0: page("A", [("Studio:", "Acme"), ("Rating:", "PG")]),
        1: page("B", [("Genre/Type:", "Western"), ("Studio:", "Bolt")]),
    }
    union = build_union_page(table, pages, min_tokens=0)
    assert union.base_page_id == 0 and union.spliced == ["c4"] and union.missing == []
    ul: Any = lhtml.fromstring(union.html).find(".//ul")
    assert ul is not None
    rows = [" ".join(li.text_content().split()) for li in ul.iter("li")]
    assert "Genre/Type: Western" in rows and rows[:2] == ["Studio: Acme", "Rating: PG"]


def _price_union(price_p: float, extra_p: float) -> UnionPage:
    """Base row ``Price: $1`` (a key-value pair, value ``c1``); the donor row adds a third cell
    ``c2``, which on the base row turns the pair into something else."""
    row = "html > body > table.product > tr"
    table = CandidateTable(
        template_id="t_price",
        cluster_size=30,
        n_sampled=30,
        candidates=[
            cand("c0", f"{row} > td", "label", 1.0, m=1.0, table_id="t0", table_role="kv_key"),
            cand(
                "c1",
                f"{row} > td > b.big",
                "optional",
                price_p,
                column="Price:",
                table_id="t0",
                table_role="kv_value",
            ),
            cand("c2", f"{row} > td.extra", "optional", extra_p),
        ],
        tables=[
            Table(
                id="t0",
                kind="key_value",
                layout="pairs",
                container_path="html > body > table.product",
                row_path=row,
                key_candidate="c0",
            )
        ],
    )

    def page(price: str, extra: str = "") -> str:
        cell = f'<td class="extra">{extra}</td>' if extra else ""
        return (
            '<html><body><table class="product"><tr><td>Price:</td>'
            f'<td><b class="big">{price}</b></td>{cell}</tr></table></body></html>'
        )

    union = build_union_page(table, {0: page("$1"), 1: page("$2", "was $3")}, min_tokens=0)
    assert union.base_page_id == 0
    return union


def test_a_splice_that_hides_a_more_frequent_candidate_is_undone() -> None:
    union = _price_union(price_p=0.9, extra_p=0.1)
    assert union.spliced == [] and union.missing == ["c2"] and union.shown == ["c1"]
    assert any("splice of c2 from page 1 undone: it hides c1" in x for x in union.log)
    assert "was $3" not in union.html


def test_a_splice_that_hides_only_rarer_candidates_is_kept() -> None:
    union = _price_union(price_p=0.1, extra_p=0.9)
    assert union.spliced == ["c2"] and union.missing == ["c1"] and union.shown == ["c2"]
    assert any("splice of c2 hides the rarer c1" in x for x in union.log)
    assert "was $3" in union.html
