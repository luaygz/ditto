"""extract_run: final specs over every page of their clusters into JSONL (no LLM)."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.extract import ExtractSummary, extract_run
from ditto.ingest import PageStore, ingest
from ditto.run import RunDir, read_json
from ditto.specgen.models import Spec

PRODUCT = """<html><head><title>{name}</title></head><body class="shop">
<h1 itemprop="name">{name}</h1>{price}<ul class="tags">{tags}</ul></body></html>"""


def _run(tmp_path: Path, n: int = 12) -> tuple[RunDir, list[int], list[int]]:
    src = tmp_path / "site"
    src.mkdir()
    rows = []
    for i in range(n):
        price = f'<span class="price">${10 + i}.00</span>' if i % 3 else ""
        tags = "".join(f"<li>t{i}{k}</li>" for k in range(i % 3))
        html = PRODUCT.format(name=f"Item {i}", price=price, tags=tags)
        (src / f"p{i}.html").write_text(html)
        rows.append(f"p{i}.html\thttps://shop.example/item/{i:03d}")
    (src / "about.html").write_text("<html><body><p>About us</p></body></html>")
    rows.append("about.html\thttps://shop.example/about")
    (src / "urls.tsv").write_text("\n".join(rows) + "\n")
    ingest(src, tmp_path / "run", workers=1)
    run = RunDir(tmp_path / "run")
    store = PageStore.open(run)
    items = [store.lookup(f"https://shop.example/item/{i:03d}") for i in range(n)]
    about = store.lookup("https://shop.example/about")
    product_ids = [p for p in items if p is not None]
    tail_ids = [about] if about is not None else []
    Clusters(
        n_pages=n + 1,
        min_cluster_size=5,
        clusters=[
            Cluster(
                template_id="t_prod",
                size=n,
                is_tail=False,
                signature_groups=[0],
                page_ids=product_ids,
            ),
            Cluster(
                template_id="t_about", size=1, is_tail=True, signature_groups=[1], page_ids=tail_ids
            ),
        ],
        signature_groups=[
            SignatureGroup(id=0, sig_hash="a", size=n, template_id="t_prod", page_ids=product_ids),
            SignatureGroup(id=1, sig_hash="b", size=1, template_id="t_about", page_ids=tail_ids),
        ],
    ).save(run.clusters_json)
    return run, product_ids, tail_ids


def _spec(tid: str, status: str = "final", extra: dict[str, object] | None = None) -> Spec:
    fields: dict[str, object] = {
        "name": {"type": "scalar", "candidates": ["//h1[@itemprop='name']", "//title"]},
        "price": {
            "type": "scalar",
            "candidates": ["//span[@class='price']", "//*[@class='price']"],
            "transforms": ["cast(float)"],
        },
        "tags": {"type": "list", "candidates": ["//ul[@class='tags']/li", "//ul/li[@id='x']"]},
    }
    fields.update(extra or {})
    return Spec.model_validate(
        {
            "template_id": tid,
            "site": "shop.example",
            "status": status,
            "generated_by": {"model": "m", "prompt_version": "specgen.v1"},
            "fields": fields,
        }
    )


def test_jsonl_rows_shape_and_summary(tmp_path: Path) -> None:
    run, product_ids, _ = _run(tmp_path)
    _spec("t_prod").save(run.spec("t_prod"))
    _spec("t_about", status="draft").save(run.spec("t_about"))
    tail = run.out_jsonl(run.site, None)
    tail.parent.mkdir(parents=True, exist_ok=True)
    tail.write_text('{"template_id": null}\n')

    summary = extract_run(run, workers=1)
    assert isinstance(summary, ExtractSummary)
    path = run.out_jsonl(run.site, "t_prod")
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines()]
    assert len(rows) == 12
    assert [r["page_id"] for r in rows] == sorted(product_ids)
    first = rows[0]
    assert list(first) == ["url", "page_id", "template_id", "name", "price", "tags", "_meta"]
    assert first["url"] == "https://shop.example/item/000" and first["template_id"] == "t_prod"
    assert first["name"] == "Item 0" and first["price"] is None and first["tags"] == []
    assert first["_meta"] == {"missing": ["price", "tags"], "spec_version": "1"}
    assert rows[5]["price"] == 15.0 and rows[5]["tags"] == ["t50", "t51"]
    assert tail.read_text() == '{"template_id": null}\n'  # long-tail output left alone
    assert not run.out_jsonl(run.site, "t_about").exists()
    assert summary.skipped == {"t_about": "spec is not final (run `ditto verify`)"}
    te = summary.templates["t_prod"]
    assert te.pages == 12 and te.filled == {"name": 12, "price": 8, "tags": 8}
    assert summary.pages == 12 and summary.workers == 1
    assert summary.pages_per_sec_per_core and summary.pages_per_cpu_sec
    timing = read_json(run.timings_json)["extract"]
    assert timing["pages"] == 12 and timing["pages_per_sec_per_core"] > 0
    assert run.read_meta()["extract"]["templates"]["t_prod"]["pages"] == 12


def test_parallel_extraction_matches_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, _ = _run(tmp_path, n=40)
    _spec("t_prod").save(run.spec("t_prod"))
    extract_run(run, workers=1)
    serial = run.out_jsonl(run.site, "t_prod").read_bytes()
    monkeypatch.setattr("ditto.extract.stage.MIN_PAGES_PER_CHUNK", 4)
    summary = extract_run(run, workers=3)  # 40 pages in chunks of 4: 10 chunks on 3 workers
    assert run.out_jsonl(run.site, "t_prod").read_bytes() == serial
    assert summary.workers == 3 and summary.templates["t_prod"].filled["name"] == 40


def test_chunk_size_balances_workers() -> None:
    from ditto.extract.stage import MAX_PAGES_PER_CHUNK, MIN_PAGES_PER_CHUNK, chunk_size

    assert chunk_size(2000, 16) == 32  # four chunks per worker
    assert chunk_size(100, 16) == MIN_PAGES_PER_CHUNK
    assert chunk_size(1_000_000, 4) == MAX_PAGES_PER_CHUNK
    assert chunk_size(0, 1) == MIN_PAGES_PER_CHUNK


def test_tail_cluster_with_final_spec_is_extracted(tmp_path: Path) -> None:
    run, _, tail_ids = _run(tmp_path, n=6)
    about = Spec.model_validate(
        {
            "template_id": "t_about",
            "site": "shop.example",
            "status": "final",
            "generated_by": {"model": "m", "prompt_version": "specgen.v1"},
            "fields": {
                "text": {
                    "type": "scalar",
                    "candidates": [
                        "//p[normalize-space()='About us']",
                        "//body/p[contains(., 'About')]",
                    ],
                }
            },
        }
    )
    about.save(run.spec("t_about"))
    summary = extract_run(run)
    rows = run.out_jsonl(run.site, "t_about").read_bytes().splitlines()
    assert orjson.loads(rows[0])["text"] == "About us"
    assert summary.templates["t_about"].pages == len(tail_ids) == 1
