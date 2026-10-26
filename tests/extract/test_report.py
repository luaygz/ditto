"""write_report: report.json and report.md contents from a hand-built run directory."""

from __future__ import annotations

from pathlib import Path

import orjson

from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.extract import write_report
from ditto.run import RunDir, read_json, write_json
from ditto.specgen.models import Spec
from ditto.verify.models import FieldReport, RepairRecord, VerifyReport


def _setup(tmp_path: Path) -> RunDir:
    run = RunDir(tmp_path / "run").ensure()
    run.write_meta(site="shop.example")
    Clusters(
        n_pages=45,
        min_cluster_size=20,
        clusters=[
            Cluster(
                template_id="t_prod",
                size=40,
                is_tail=False,
                signature_groups=[0],
                url_patterns=["/item/{n}"],
                url_examples=["https://shop.example/item/001"],
                page_ids=list(range(40)),
                merged_from=["t_old"],
            ),
            Cluster(
                template_id="t_misc",
                size=5,
                is_tail=True,
                signature_groups=[1],
                url_patterns=["/misc/{slug}"],
                url_examples=["https://shop.example/misc/a"],
                page_ids=list(range(40, 45)),
            ),
        ],
        signature_groups=[
            SignatureGroup(
                id=0, sig_hash="a", size=40, template_id="t_prod", page_ids=list(range(40))
            ),
            SignatureGroup(
                id=1, sig_hash="b", size=5, template_id="t_misc", page_ids=list(range(40, 45))
            ),
        ],
    ).save(run.clusters_json)
    Spec.model_validate(
        {
            "template_id": "t_prod",
            "site": "shop.example",
            "status": "final",
            "generated_by": {
                "model": "openai/m1",
                "prompt_version": "specgen.v1",
                "repair_calls": 1,
            },
            "fields": {
                "name": {
                    "type": "scalar",
                    "candidates": ["//h1[@itemprop='name']", "//title"],
                    "stats": {
                        "presence_rate": 1.0,
                        "expected_presence": 1.0,
                        "multiplicity": 1.0,
                        "type_check_pass": 1.0,
                        "failure_rate": 0.0,
                    },
                },
                "price": {
                    "type": "scalar",
                    "candidates": ["//span[@class='price']", "//*[@itemprop='price']"],
                    "degraded": True,
                    "optional": True,
                    "stats": {"presence_rate": 0.5, "expected_presence": 0.7, "failure_rate": 0.28},
                },
                "reviews": {
                    "type": "record_list",
                    "container": "//ol[@class='reviews']/li",
                    "columns": {"text": "./p"},
                    "stats": {"presence_rate": 0.6, "rows_min": 1, "rows_max": 12},
                },
            },
            "not_present": {"sku": "no candidate holds a SKU"},
        }
    ).save(run.spec("t_prod"))
    VerifyReport(
        template_id="t_prod",
        fields={
            "price": FieldReport(
                name="price", type="scalar", selected_xpath="//span[@class='price']", failures=8
            )
        },
        repairs=[RepairRecord(call=1, signature_group=0, page_id=3, fields=["price"])],
        repair_calls=1,
        merge_split=[{"action": "merge", "target": "t_prod", "absorbed": "t_old"}],
    ).save(run.verify_report("t_prod"))
    write_json(
        run.generation_json, {"templates": {"t_misc": {"status": "skipped", "reason": "tail"}}}
    )
    usage = [
        {
            "role": "generation",
            "template_id": "t_prod",
            "model": "openai/m1",
            "input_tokens": 9000,
            "output_tokens": 800,
            "cost_usd": 0.02,
            "list_cost_usd": 0.02,
            "estimated": False,
            "cached": False,
        },
        {
            "role": "repair",
            "template_id": "t_prod",
            "model": "openai/m1",
            "input_tokens": 3000,
            "output_tokens": 200,
            "cost_usd": 0.0,
            "list_cost_usd": 0.006,
            "estimated": True,
            "cached": True,
        },
    ]
    run.usage_jsonl.write_bytes(b"".join(orjson.dumps(u) + b"\n" for u in usage))
    run.record_timing("extract", 0.08, 40, 1)
    run.record_timing("verify", 0.5, 40, 2)
    run.write_meta(
        extract={
            "pages": 40,
            "seconds": 0.08,
            "workers": 1,
            "pages_per_sec_per_core": 500.0,
            "pages_per_cpu_sec": 520.0,
            "templates": {
                "t_prod": {"pages": 40, "filled": {"name": 40, "price": 20, "reviews": 24}}
            },
        }
    )
    tail = run.out_jsonl("shop.example", None)
    tail.parent.mkdir(parents=True)
    tail.write_text('{"a": 1}\n{"a": 2}\n')
    return run


def test_report_json_contents(tmp_path: Path) -> None:
    run = _setup(tmp_path)
    report = write_report(run)
    assert read_json(run.report_json) == orjson.loads(orjson.dumps(report))
    cl = report["clusters"]
    assert (cl["n_templates"], cl["n_tail_clusters"], cl["tail_pages"]) == (1, 1, 5)
    prod = cl["templates"][0]
    assert prod["template_id"] == "t_prod" and prod["size"] == 40 and prod["spec"] == "final"
    assert prod["url_examples"] == ["https://shop.example/item/001"]
    assert prod["merged_from"] == ["t_old"]
    assert cl["templates"][1]["generation"] == "skipped"
    t = report["templates"]["t_prod"]
    assert t["fields"]["name"]["presence_rate"] == 1.0
    assert t["fields"]["name"]["fill_rate"] == 1.0
    price = t["fields"]["price"]
    assert price["degraded"] and price["failure_rate"] == 0.28 and price["failures"] == 8
    assert price["selected"] == "//span[@class='price']" and price["fill_rate"] == 0.5
    assert t["fields"]["reviews"]["rows_max"] == 12
    assert t["not_present"] == {"sku": "no candidate holds a SKU"}
    assert t["degraded"] == ["price"] and t["generated_by"]["repair_calls"] == 1
    llm = report["llm"]
    assert llm["totals"]["calls"] == 2 and llm["totals"]["input_tokens"] == 12000
    assert llm["by_role"]["repair"]["cached_calls"] == 1
    assert llm["by_template"]["t_prod"]["output_tokens"] == 1000
    assert report["timings"]["extract"]["pages_per_sec_per_core"] == 500.0
    assert report["extract"]["pages_per_cpu_sec"] == 520.0
    assert report["tail_direct"]["rows"] == 2
    assert report["generation_issues"] == {"t_misc": {"status": "skipped", "reason": "tail"}}


def test_report_markdown_contents(tmp_path: Path) -> None:
    run = _setup(tmp_path)
    write_report(run)
    md = run.report_md.read_text()
    assert md.startswith("# ditto report — shop.example")
    assert "45 pages · 1 templates · 1 tail clusters (5 pages) · 40 rows extracted" in md
    assert (
        "| t_prod | 40 |  | final | 40 | /item/{n} | https://shop.example/item/001 | merged t_old |"
        in md
    )
    assert "degraded (28.0% failing), optional" in md
    assert "1–12" in md
    assert "- `sku`: no candidate holds a SKU" in md
    assert "Merged t_old into t_prod." in md
    assert "| repair | 1 | 1 | 3000 | 200 | $0.0000 (est.) | $0.0060 |" in md
    assert "| extract | 0.080 | 40 | 1 | 500.0 | 500.0 |" in md
    assert "500.0 pages/s/core wall-clock, 520.0 pages per CPU-second" in md
    assert "Long tail (direct): 2 rows." in md
    assert "t_misc: skipped — tail" in md


def test_report_on_empty_run(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "empty").ensure()
    report = write_report(run)
    assert report["templates"] == {} and report["llm"]["totals"]["calls"] == 0
    assert run.report_md.exists()
