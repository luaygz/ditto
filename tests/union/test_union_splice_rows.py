"""Splicing into the right row, splicing whole rows, and finding rare donors past the sample."""

from __future__ import annotations

from collections.abc import Iterator

from union_fixtures import cand

from ditto.analyze.models import CandidateTable, Table
from ditto.union import build_union_page
from ditto.union.coverage import annotated_nodes
from ditto.union.tree import CandIndex, shown_candidate

M = "html > body.page > main.movie"
ROW = f"{M} > ul.facts > li.wrap"
SPAN = f"{ROW} > span"
SEC = f"{M} > section"


def movie_page(
    title: str,
    *,
    facts: list[tuple[str, str]],
    standfirst: str | None = None,
    takeaways: list[str] = ("One",),  # type: ignore[assignment]
    prices: str | None = None,
) -> str:
    rows = "".join(f'<li class="wrap"><span>{k}</span> <span>{v}</span></li>' for k, v in facts)
    stand = f'<section><h2>Standfirst</h2><p class="summary">{standfirst}</p></section>'
    points = "".join(f"<li>{t}</li>" for t in takeaways)
    price = f'<div class="price"><span>Prices</span><span>{prices}</span></div>'
    return (
        f'<html><head><title>{title}</title></head><body class="page"><main class="movie">'
        f"<h1>{title}</h1>"
        f'<ul class="facts">{rows}</ul>'
        f"{stand if standfirst else ''}"
        f'<section><h2 class="heading">Takeaways</h2><ul class="points">{points}</ul></section>'
        f"{price if prices else ''}"
        "</main></body></html>"
    )


def movie_table(*, budget_cls: str = "optional", budget_p: float = 0.4) -> CandidateTable:
    return CandidateTable(
        template_id="t_movie",
        cluster_size=50,
        n_sampled=50,
        candidates=[
            cand("c0", f"{M} > h1", "mandatory", 1.0),
            cand("c1", SPAN, "label", 1.0, table_id="tb0", table_role="kv_key"),
            cand(
                "c2",
                SPAN,
                "mandatory",
                1.0,
                table_id="tb0",
                table_role="kv_value",
                column="Studio:",
                label_hint="Studio:",
            ),
            cand(
                "c3",
                SPAN,
                budget_cls,
                budget_p,
                table_id="tb0",
                table_role="kv_value",
                column="Budget:",
                label_hint="Budget:",
            ),
            cand("c4", f"{SEC} > h2", "label", 0.3, table_id="tb1", table_role="kv_key"),
            cand("c5", f"{SEC} > h2.heading", "label", 1.0, table_id="tb1", table_role="kv_key"),
            cand("c6", f"{SEC} > p.summary", "optional", 0.3, label_hint="Standfirst"),
            cand(
                "c7",
                f"{SEC} > ul.points > li",
                "list",
                1.0,
                label_hint="Takeaways",
                repeat_min=1,
                repeat_max=4,
            ),
            cand(
                "c8",
                f"{M} > div.price > span",
                "optional",
                0.2,
                label_hint="Takeaways",
                repeat_min=2,
                repeat_max=2,
            ),
            cand(
                "c9", f"{M} > section > h2.junk", "noise", 0.01, table_id="tb1", table_role="kv_key"
            ),
        ],
        tables=[
            Table(
                id="tb0",
                kind="key_value",
                layout="pairs",
                container_path=f"{M} > ul.facts",
                row_path=ROW,
                p=1.0,
                rows_min=1,
                rows_max=2,
                key_candidate="c1",
                keys_seen=["Studio:", "Budget:"],
            ),
        ],
    )


def _shown(html: str) -> dict[str, tuple[str, ...]]:
    return {a.cid: a.texts for a in annotated_nodes(html)}


def test_a_value_sharing_its_key_path_is_spliced_as_its_own_row() -> None:
    pages = {
        1: movie_page("A", facts=[("Studio:", "Acme")]),
        2: movie_page("B", facts=[("Studio:", "Bolt"), ("Budget:", "USD 5")]),
    }
    page = build_union_page(movie_table(), {1: pages[1]}, more_pages=[(2, pages[2])])
    assert page.spliced == ["c3"] and "c3" not in page.missing
    shown = _shown(page.html)
    assert shown["c3"] == ("USD 5",)
    assert shown["c2"] == ("Acme",)  # the base row keeps its own key and value
    assert page.html.count('<li class="wrap">') == 2


def test_a_labeled_section_is_spliced_whole_not_into_another_section() -> None:
    pages = {
        1: movie_page("A", facts=[("Studio:", "Acme"), ("Budget:", "USD 1")], prices="$"),
        2: movie_page("B", facts=[("Studio:", "Bolt")], standfirst="In short: good."),
    }
    page = build_union_page(movie_table(), pages)
    assert page.base_page_id == 1 and page.spliced == ["c6"] and page.missing == []
    shown = _shown(page.html)
    assert shown["c6"] == ("In short: good.",) and shown["c7"] == ("One",)
    # Standfirst got its own section with its heading; Takeaways keeps only its list
    assert "<section><h2>Standfirst</h2>" in page.html.replace("\n", "")


def test_a_fixed_group_of_same_path_siblings_is_spliced_with_its_parent() -> None:
    pages = {
        1: movie_page("A", facts=[("Studio:", "Acme"), ("Budget:", "USD 1")], standfirst="S"),
        2: movie_page("B", facts=[("Studio:", "Bolt")], prices="$$"),
    }
    page = build_union_page(movie_table(), pages)
    assert page.base_page_id == 1 and page.spliced == ["c8"]
    assert _shown(page.html)["c8"] == ("Prices", "$$")


def test_donors_past_the_sample_are_searched_only_while_needed() -> None:
    pool = [
        (3, movie_page("C", facts=[("Studio:", "Cine")])),
        (4, movie_page("D", facts=[("Studio:", "Dune"), ("Budget:", "USD 9")])),
        (5, movie_page("E", facts=[("Studio:", "Echo"), ("Budget:", "USD 7")])),
    ]
    read: list[int] = []

    def lazy() -> Iterator[tuple[int, str]]:
        for pid, html in pool:
            read.append(pid)
            yield pid, html

    sample = {1: movie_page("A", facts=[("Studio:", "Acme")], standfirst="S", prices="$")}
    page = build_union_page(movie_table(), sample, more_pages=lazy())
    assert read == [3, 4]  # stops once every candidate is found
    assert page.searched_pages == 2 and page.donor_page_ids == [4]
    assert page.sample_page_ids == [1] and page.meta()["searched_pages"] == 2
    assert any("searched 2 more pages for 1 field candidates" in x for x in page.log)
    # without a pool nothing is searched
    again = build_union_page(movie_table(), {2: movie_page("B", facts=[("Studio:", "B")])})
    assert again.searched_pages == 0


def test_rare_table_values_are_shown_but_other_noise_is_not() -> None:
    table = movie_table(budget_cls="noise", budget_p=0.01)
    index = CandIndex.build(table)
    assert "c3" in index.fields  # a rare key-value value: shown
    assert "c9" not in index.fields  # noise that is a key, not a value
    assert shown_candidate(table.candidates[0]) and not shown_candidate(table.candidates[1])
    pages = {
        1: movie_page("A", facts=[("Studio:", "Acme")]),
        2: movie_page("B", facts=[("Studio:", "Bolt"), ("Budget:", "USD 5")]),
    }
    page = build_union_page(table, pages)
    assert "c3" in page.shown and page.html.count("c3 · noise · on 1% of pages") == 1
