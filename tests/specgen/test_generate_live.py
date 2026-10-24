"""Live spec generation: one real main-model call on a realistic product template.

Skipped unless ``OPENAI_API_KEY`` is set. Costs about two cents per run (cached afterwards by
the LLM layer's response cache).
"""

from __future__ import annotations

import pytest

from ditto.analyze.models import Candidate, CandidateTable, Table, TableColumn
from ditto.llm import get_llm
from ditto.schema import FieldRequest
from ditto.specgen.generate import TemplateInput, generate_spec

BOOKS = [
    {
        "slug": "the-salt-road",
        "title": "The Salt Road",
        "author": "Nadia Karam",
        "price": "$18.99",
        "list": "$24.00",
        "isbn": "978-1-4028-9462-6",
        "publisher": "Harbor Press",
        "pages": "342",
        "series": "Desert Trilogy #1",
        "reviews": [("Lea M.", "5/5", "Could not put it down."), ("Omar T.", "4/5", "Slow start.")],
    },
    {
        "slug": "cedar-and-ash",
        "title": "Cedar and Ash",
        "author": "Jad Rahme",
        "price": "$12.50",
        "list": None,
        "isbn": "978-0-3164-8521-3",
        "publisher": "North Lane",
        "pages": "218",
        "series": None,
        "reviews": [("Hana S.", "3/5", "Fine.")],
    },
    {
        "slug": "night-markets",
        "title": "Night Markets of Beirut",
        "author": "Rima Aoun",
        "price": "$21.00",
        "list": "$28.00",
        "isbn": "978-1-9848-0122-9",
        "publisher": "Harbor Press",
        "pages": "410",
        "series": None,
        "reviews": [],
    },
    {
        "slug": "a-quiet-orbit",
        "title": "A Quiet Orbit",
        "author": "Sami Haddad",
        "price": "$9.99",
        "list": None,
        "isbn": "978-0-0623-1595-4",
        "publisher": "Kite Books",
        "pages": "189",
        "series": "Orbit Cycle #2",
        "reviews": [
            ("Dana K.", "4/5", "Lovely prose."),
            ("Ziad F.", "5/5", "Best of the year."),
            ("Maya R.", "2/5", "Not for me."),
        ],
    },
]


def page(b: dict[str, object]) -> str:
    old = f' <s class="list-price">{b["list"]}</s>' if b["list"] else ""
    series = f"<tr><th>Series</th><td>{b['series']}</td></tr>" if b["series"] else ""
    reviews = "".join(
        f'<div class="review"><span class="reviewer">{r}</span><span class="rating">{s}</span>'
        f'<p class="review-text">{t}</p></div>'
        for r, s, t in b["reviews"]  # type: ignore[attr-defined]
    )
    return f"""<html><head><title>{b["title"]} by {b["author"]} | Paperleaf Books</title></head>
<body class="product-page"><header class="site-header"><a class="logo" href="/">Paperleaf</a>
<nav><a href="/fiction">Fiction</a><a href="/history">History</a></nav></header>
<div class="breadcrumbs"><a href="/">Home</a> › <a href="/fiction">Fiction</a> › <span>{b["title"]}</span></div>
<main id="main"><div class="product" itemscope="">
<h1 class="product-title" itemprop="name">{b["title"]}</h1>
<p class="byline">by <a class="author" href="/author/{b["slug"]}" itemprop="author">{b["author"]}</a></p>
<div class="price-box"><span class="price" itemprop="price">{b["price"]}</span>{old}</div>
<table class="details"><tr><th>ISBN-13</th><td>{b["isbn"]}</td></tr>
<tr><th>Publisher</th><td>{b["publisher"]}</td></tr><tr><th>Pages</th><td>{b["pages"]}</td></tr>{series}</table>
</div><section class="reviews"><h2>Customer reviews</h2>{reviews}</section>
<section class="related"><h2>You may also like</h2><ul class="carousel">
<li><a href="/book/x">Other Book</a></li><li><a href="/book/y">Another Book</a></li></ul></section></main>
<footer>© 2026 Paperleaf Books · <span class="updated">Page generated 2026-10-01</span> ·
<span class="views">1,234 views</span></footer></body></html>"""


UNION = """<html><body class="product-page"><main id="main"><div class="product">
<!-- c1 · mandatory · on 100% of pages · e.g. The Salt Road -->
<h1 class="product-title" itemprop="name">The Salt Road</h1>
<p class="byline">by <!-- c2 · mandatory · on 100% of pages · e.g. Nadia Karam -->
<a class="author" href="/author/the-salt-road" itemprop="author">Nadia Karam</a>
<!-- c3 @href · mandatory · on 100% of pages · e.g. /author/the-salt-road --></p>
<div class="price-box"><!-- c4 · mandatory · on 100% of pages · e.g. $18.99 -->
<span class="price" itemprop="price">$18.99</span>
<!-- c5 · optional · on 48% of pages · e.g. $24.00 --><s class="list-price">$24.00</s></div>
<table class="details"><tr><th>ISBN-13</th><!-- c6 · mandatory · on 100% of pages · e.g. 978-1-4028-9462-6 --><td>978-1-4028-9462-6</td></tr>
<tr><th>Publisher</th><!-- c7 · mandatory · on 100% of pages · e.g. Harbor Press --><td>Harbor Press</td></tr>
<tr><th>Pages</th><!-- c8 · mandatory · on 100% of pages · e.g. 342 --><td>342</td></tr>
<!-- spliced from another page: c9 -->
<tr><th>Series</th><!-- c9 · optional · on 37% of pages · e.g. Desert Trilogy #1 --><td>Desert Trilogy #1</td></tr></table>
</div><section class="reviews"><h2>Customer reviews</h2>
<div class="review"><!-- c10 · list · on 81% of pages · e.g. Lea M. --><span class="reviewer">Lea M.</span>
<!-- c11 · list · on 81% of pages · e.g. 5/5 --><span class="rating">5/5</span>
<!-- c12 · list · on 81% of pages · e.g. Could not put it down. --><p class="review-text">Could not put it down.</p></div>
<!-- collapsed: 4 more siblings like the one above (variant 1 of 1) --></section>
<section class="related"><h2>You may also like</h2><ul class="carousel">
<li><!-- c13 · list · on 100% of pages · e.g. Other Book --><a href="/book/x">Other Book</a></li>
<!-- collapsed: 7 more siblings like the one above (variant 1 of 1) --></ul></section></main>
<footer><!-- c14 · mandatory · on 100% of pages · e.g. Page generated 2026-10-01 -->
<span class="updated">Page generated 2026-10-01</span>
<!-- c15 · mandatory · on 100% of pages · e.g. 1,234 views --><span class="views">1,234 views</span></footer>
</body></html>"""


def table() -> CandidateTable:
    def c(cid: str, path: str, cls: str, p: float, ex: list[str], **kw: object) -> Candidate:
        return Candidate.model_validate(
            {"id": cid, "path": path, "class": cls, "p": p, "m": 0.05, "examples": ex, **kw}
        )

    prod = "html > body.product-page > main#main > div.product"
    rev = "html > body.product-page > main#main > section.reviews > div.review"
    return CandidateTable(
        template_id="t_book",
        cluster_size=2000,
        n_sampled=2000,
        candidates=[
            c(
                "c1",
                f"{prod} > h1.product-title[itemprop=name]",
                "mandatory",
                1.0,
                ["The Salt Road", "Cedar and Ash"],
            ),
            c(
                "c2",
                f"{prod} > p.byline > a.author[itemprop=author]",
                "mandatory",
                1.0,
                ["Nadia Karam", "Jad Rahme"],
                label_hint="by",
            ),
            c(
                "c3",
                f"{prod} > p.byline > a.author[itemprop=author] @href",
                "mandatory",
                1.0,
                ["/author/the-salt-road"],
                attr="href",
                value_type="url",
            ),
            c(
                "c4",
                f"{prod} > div.price-box > span.price[itemprop=price]",
                "mandatory",
                1.0,
                ["$18.99", "$12.50"],
                value_type="price",
            ),
            c(
                "c5",
                f"{prod} > div.price-box > s.list-price",
                "optional",
                0.48,
                ["$24.00"],
                value_type="price",
            ),
            c(
                "c6",
                f"{prod} > table.details > tr > td",
                "mandatory",
                1.0,
                ["978-1-4028-9462-6"],
                label_hint="ISBN-13",
                table_id="T1",
                table_role="kv_value",
                column="ISBN-13",
            ),
            c(
                "c7",
                f"{prod} > table.details > tr > td",
                "mandatory",
                1.0,
                ["Harbor Press"],
                label_hint="Publisher",
                table_id="T1",
                table_role="kv_value",
                column="Publisher",
            ),
            c(
                "c8",
                f"{prod} > table.details > tr > td",
                "mandatory",
                1.0,
                ["342", "218"],
                label_hint="Pages",
                table_id="T1",
                table_role="kv_value",
                column="Pages",
                value_type="number",
            ),
            c(
                "c9",
                f"{prod} > table.details > tr > td",
                "optional",
                0.37,
                ["Desert Trilogy #1"],
                label_hint="Series",
                table_id="T1",
                table_role="kv_value",
                column="Series",
            ),
            c(
                "c10",
                f"{rev} > span.reviewer",
                "list",
                0.81,
                ["Lea M.", "Hana S."],
                repeat_min=1,
                repeat_max=40,
            ),
            c(
                "c11",
                f"{rev} > span.rating",
                "list",
                0.81,
                ["5/5", "3/5"],
                repeat_min=1,
                repeat_max=40,
            ),
            c(
                "c12",
                f"{rev} > p.review-text",
                "list",
                0.81,
                ["Could not put it down."],
                repeat_min=1,
                repeat_max=40,
            ),
            c(
                "c13",
                "html > body.product-page > main#main > section.related > ul.carousel > li > a",
                "list",
                1.0,
                ["Other Book"],
                repeat_min=8,
                repeat_max=8,
            ),
            c(
                "c14",
                "html > body.product-page > footer > span.updated",
                "mandatory",
                1.0,
                ["Page generated 2026-10-01"],
                value_type="date",
            ),
            c(
                "c15",
                "html > body.product-page > footer > span.views",
                "mandatory",
                1.0,
                ["1,234 views"],
                value_type="number",
            ),
        ],
        tables=[
            Table(
                id="T1",
                kind="key_value",
                container_path=f"{prod} > table.details",
                row_path=f"{prod} > table.details > tr",
                p=1.0,
                rows_min=3,
                rows_max=4,
                key_candidate=None,
                value_candidate="c6",
                keys_seen=["ISBN-13", "Publisher", "Pages", "Series"],
            ),
            Table(
                id="T2",
                kind="record_list",
                container_path="html > body.product-page > main#main > section.reviews",
                row_path=rev,
                p=0.81,
                rows_min=1,
                rows_max=40,
                columns=[
                    TableColumn(name="reviewer", candidate_id="c10", header_source="none"),
                    TableColumn(name="rating", candidate_id="c11", header_source="none"),
                    TableColumn(name="text", candidate_id="c12", header_source="none"),
                ],
            ),
        ],
    )


FIELDS = [
    FieldRequest(name="title", description="book title"),
    FieldRequest(name="author", description="author name"),
    FieldRequest(name="price", description="current selling price", type_check="price"),
    FieldRequest(name="isbn", description="ISBN-13"),
    FieldRequest(name="series", description="series name and number"),
    FieldRequest(
        name="reviews",
        description="customer reviews with reviewer, rating, text",
        type="record_list",
    ),
    FieldRequest(name="weight", description="shipping weight of the book"),
]


@pytest.mark.live
def test_live_prompted_spec_extracts_held_out_pages() -> None:
    inp = TemplateInput(
        template_id="t_book",
        site="paperleaf.example",
        mode="prompted",
        table=table(),
        union_html=UNION,
        size=2000,
        url_patterns=["/book/{slug}"],
        url_examples=["https://paperleaf.example/book/the-salt-road"],
        pages_seen=["https://paperleaf.example/book/the-salt-road"],
        shown={f"c{i}" for i in range(1, 16)},
        fields=FIELDS,
    )
    gen = generate_spec(inp, llm=get_llm("generation"))
    assert gen.status == "ok", (gen.reason, gen.failed_fields)
    spec = gen.spec
    assert spec is not None
    assert "weight" in spec.not_present
    assert set(spec.fields) == {"title", "author", "price", "isbn", "series", "reviews"}
    assert spec.fields["reviews"].type == "record_list"
    for b in BOOKS[1:]:  # pages the model never saw
        row = spec.apply(page(b))
        assert row["title"] == b["title"]
        assert row["author"] == b["author"]
        assert b["price"].lstrip("$") in str(row["price"])  # type: ignore[union-attr]
        assert row["isbn"] == b["isbn"]
        assert (row["series"] is None) == (b["series"] is None)
        assert len(row["reviews"]) == len(b["reviews"])  # type: ignore[arg-type]


@pytest.mark.live
def test_live_unprompted_spec_filters_noise() -> None:
    inp = TemplateInput(
        template_id="t_book",
        site="paperleaf.example",
        mode="unprompted",
        table=table(),
        union_html=UNION,
        size=2000,
        pages_seen=["https://paperleaf.example/book/the-salt-road"],
        shown={f"c{i}" for i in range(1, 16)},
    )
    gen = generate_spec(inp, llm=get_llm("generation"))
    assert gen.status == "ok", (gen.reason, gen.failed_fields)
    spec = gen.spec
    assert spec is not None
    used = {c for f in spec.fields.values() for c in f.source_candidates}
    assert {"c1", "c2", "c4"} <= used
    assert not {"c13", "c14", "c15"} & used  # carousel, page timestamp, view counter
    row = spec.apply(page(BOOKS[3]))
    assert "A Quiet Orbit" in row.values()
