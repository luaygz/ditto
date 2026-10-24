"""A tiny generated-style synthetic site and a fake ditto pipeline for the eval harness tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ditto.cluster.models import Cluster, Clusters
from ditto.run import RunDir

HOST = "https://shop.example"

# Source of a fake baseline module (eval/baselines/<name>.py) used by the harness tests.
BASELINE = """
from ditto.evaluation.baseline_api import BaselineOutput, NotRun
from lxml import html


def run_baseline(pages, schema, *, out_dir, settings=None, max_pages=None, seed=0, site=None):
    if site == "refuse":
        raise NotRun("no hand-written wrapper for refuse")
    preds = {}
    for p in pages:
        assert "<script" not in p.html and p.template_id
        h1 = html.fromstring(p.html).xpath("//h1/text()")
        preds[p.key] = {"title": h1[0] if h1 else None}
    n = len(pages)
    return BaselineOutput(
        predictions=preds, calls=n, input_tokens=500 * n, output_tokens=20 * n,
        cost_usd=0.01 * n, model="openai/gpt-test", spend_usd=0.0,
        notes=[f"site={site} max_pages={max_pages} fields={len(schema.fields)}"],
    )
"""


PRODUCTS: list[dict[str, Any]] = [
    {
        "title": "Widget One",
        "price": "$19.99",
        "tags": ["tools", "home"],
        "sizes": [{"size": "S", "stock": "3"}, {"size": "M", "stock": "0"}],
        "specs": {"Color": "Red", "Weight": "2 kg"},
    },
    {
        "title": "Widget Two",
        "price": "$1,299.00",
        "tags": ["garden"],
        "sizes": [{"size": "L", "stock": "7"}],
        "specs": {"Color": "Blue"},
    },
    {
        "title": "Widget Three",
        "price": "$5.00",
        "tags": [],
        "sizes": [],
        "specs": {"Color": "Green", "Weight": "1 kg"},
    },
    {
        "title": "Widget Four",
        "price": None,
        "tags": ["tools"],
        "sizes": [{"size": "S", "stock": "1"}],
        "specs": {"Weight": "5 kg"},
    },
]
ARTICLES: list[dict[str, Any]] = [
    {"headline": "Spring sale starts", "author": None},
    {"headline": "New widgets arrive", "author": "Ana Lima"},
    {"headline": "Store hours change", "author": "Bo Chen"},
    {"headline": "Garden tips", "author": None},
]


def _product_html(p: dict[str, Any]) -> str:
    price = f'<span class="price">{p["price"]}</span>' if p["price"] else ""
    tags = "".join(f"<li>{t}</li>" for t in p["tags"])
    rows = "".join(f"<tr><td>{r['size']}</td><td>{r['stock']}</td></tr>" for r in p["sizes"])
    specs = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in p["specs"].items())
    return (
        "<html><head><title>Shop</title><script>var x = 1;</script></head><body>"
        '<nav><a href="/">Home</a></nav>'
        f'<main class="product"><h1>{p["title"]}</h1>{price}'
        f'<ul class="tags">{tags}</ul>'
        f"<table><tr><th>Size</th><th>Stock</th></tr>{rows}</table>"
        f"<dl>{specs}</dl></main><footer>(c) Shop</footer></body></html>"
    )


def _article_html(a: dict[str, Any]) -> str:
    author = f'<p class="byline">{a["author"]}</p>' if a["author"] else ""
    return (
        "<html><head><title>News</title></head><body>"
        f'<article><h2 class="headline">{a["headline"]}</h2>{author}'
        "<p>Lorem ipsum dolor sit amet.</p></article></body></html>"
    )


def make_tiny_site(root: Path) -> Path:
    """Write ``root/site``, ``root/truth`` and ``root/synth.json`` in the synthetic layout."""
    site, truth = root / "site", root / "truth"
    (truth / "fields").mkdir(parents=True)
    site.mkdir(parents=True)
    tsv: list[str] = []
    n = 0
    for p in PRODUCTS:
        page = f"p{n:06d}"
        url = f"{HOST}/product/{n}"
        (site / f"{page}.html").write_text(_product_html(p))
        car = {
            "page": page,
            "url": url,
            "template_id": "T01",
            "fields": {"title": p["title"], "price": p["price"]},
            "lists": {"tags": p["tags"]},
            "records": {"sizes": p["sizes"]},
            "kv": {"specs": p["specs"]},
            "optional_present": ["price"] if p["price"] else [],
        }
        (truth / f"{page}.json").write_text(json.dumps(car))
        tsv.append(f"{page}.html\t{url}")
        n += 1
    for a in ARTICLES:
        page = f"p{n:06d}"
        url = f"{HOST}/news/{n}"
        (site / f"{page}.html").write_text(_article_html(a))
        car = {
            "page": page,
            "url": url,
            "template_id": "T02",
            "fields": {"headline": a["headline"], "author": a["author"]},
            "lists": {},
            "records": {},
            "kv": {},
            "optional_present": ["author"] if a["author"] else [],
        }
        (truth / f"{page}.json").write_text(json.dumps(car))
        tsv.append(f"{page}.html\t{url}")
        n += 1
    (site / "urls.tsv").write_text("\n".join(tsv) + "\n")
    (truth / "templates.json").write_text(json.dumps({"T01": {}, "T02": {}}))
    (truth / "fields" / "T01.yaml").write_text(
        "title: the product name\n"
        "price: {description: listed price, type_check: price}\n"
        "tags: {description: tag list, type: list}\n"
        "sizes: {description: size table, type: record_list}\n"
        "specs: {description: specifications, type: key_value}\n"
    )
    (truth / "fields" / "T02.yaml").write_text(
        "headline: article headline\nauthor: article author\ntitle: duplicate name, first wins\n"
    )
    (root / "synth.json").write_text(json.dumps({"seed": 1, "pages": n}))
    return root


class FakePipeline:
    """Stands in for the ditto stages: ingests the tiny site, clusters it by its true template,
    and 'extracts' the given predictions, writing the run-directory files the runner reads."""

    def __init__(
        self,
        site_root: Path,
        predictions: dict[str, dict[str, Any]],
        *,
        usage_cost: float = 0.002,
        seen: dict[str, list[str]] | None = None,
        merge_templates: bool = False,
    ) -> None:
        self.site_root = site_root
        self.predictions = predictions
        self.usage_cost = usage_cost
        self.seen = seen if seen is not None else {"T01": ["page:0"], "T02": ["/news/4"]}
        self.merge_templates = merge_templates
        self.calls: list[tuple[str, dict[str, Any]]] = []
        lines = (site_root / "site" / "urls.tsv").read_text().splitlines()
        self.pages = [tuple(line.split("\t")) for line in lines if line]
        self.truth = {
            p: json.loads((site_root / "truth" / f"{Path(p).stem}.json").read_text())
            for p, _ in self.pages
        }

    # -- stages ------------------------------------------------------------------------------
    def ingest(self, src: Any, out: Path, **kw: Any) -> RunDir:
        self.calls.append(("ingest", {"src": src, **kw}))
        run = RunDir(out).ensure()
        run.write_meta(site=kw.get("site"))
        run.record_timing("ingest", 0.5, len(self.pages), 1)
        return run

    def cluster(self, run: RunDir, **kw: Any) -> None:
        self.calls.append(("cluster", kw))
        groups: dict[str, list[int]] = {}
        for pid, (path, _) in enumerate(self.pages):
            tid = "t_all" if self.merge_templates else "t_" + self.truth[path]["template_id"]
            groups.setdefault(tid, []).append(pid)
        clusters = [
            Cluster(template_id=t, size=len(p), is_tail=False, signature_groups=[], page_ids=p)
            for t, p in groups.items()
        ]
        Clusters(
            n_pages=len(self.pages), min_cluster_size=2, clusters=clusters, signature_groups=[]
        ).save(run.clusters_json)
        run.record_timing("fingerprint", 0.25, len(self.pages), 2)
        run.record_timing("cluster", 0.1, len(self.pages), 1)

    def analyze(self, run: RunDir, **kw: Any) -> None:
        self.calls.append(("analyze", kw))

    @staticmethod
    def run_options(**kw: Any) -> SimpleNamespace:
        return SimpleNamespace(**kw)

    def continue_run(self, run: RunDir, opts: SimpleNamespace, schema: Any) -> None:
        self.calls.append(("continue_run", {"opts": opts, "schema": schema}))
        site = run.site
        by_tid: dict[str, list[dict[str, Any]]] = {}
        for pid, (path, url) in enumerate(self.pages):
            tid = "t_" + self.truth[path]["template_id"]
            key = Path(path).stem
            row = {"url": url, "page_id": pid, "template_id": tid}
            row.update(self.predictions.get(key, {}))
            row["_meta"] = {"missing": [], "spec_version": "1"}
            by_tid.setdefault(tid, []).append(row)
        for tid, rows in by_tid.items():
            out = run.out_jsonl(site, tid)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
            spec = {
                "template_id": tid,
                "site": site,
                "status": "final",
                "generated_by": {
                    "model": "openai/gpt-test",
                    "prompt_version": "specgen.v1",
                    "repair_prompt_version": "repair.v1",
                    "pages_seen": self.seen.get(tid.removeprefix("t_"), []),
                    "mode": opts.mode,
                },
                "fields": {
                    "title": {"type": "scalar", "candidates": ["//h1", "//title"]},
                },
            }
            run.spec(tid).parent.mkdir(parents=True, exist_ok=True)
            run.spec(tid).write_text(json.dumps(spec))
        usage = [
            {
                "role": "generation",
                "model": "openai/gpt-test",
                "template_id": tid,
                "prompt_version": "specgen.v1",
                "input_tokens": 10_000,
                "output_tokens": 1_000,
                "cost_usd": self.usage_cost,
                "cached": False,
                "latency_s": 2.0,
            }
            for tid in sorted(by_tid)
        ]
        usage.append(
            {
                **usage[0],
                "role": "repair",
                "prompt_version": "repair.v1",
                "cached": True,
                "cost_usd": 0.0,
            }
        )
        run.usage_jsonl.write_text("".join(json.dumps(u) + "\n" for u in usage))
        run.record_timing("verify", 0.2, len(self.pages), 2)
        run.record_timing("extract", 0.01, len(self.pages), 2)

    # -- other hooks -------------------------------------------------------------------------
    def page_urls(self, run: RunDir) -> dict[int, str]:
        return {pid: url for pid, (_, url) in enumerate(self.pages)}

    def read_html(self, run: RunDir, pid: int) -> str:
        return (self.site_root / "site" / self.pages[pid][0]).read_text()

    @staticmethod
    def settings(models: dict[str, str], budget: float | None) -> dict[str, Any]:
        return {"models": dict(models), "budget": budget}

    @staticmethod
    def resolve_model(role: str, settings: dict[str, Any]) -> str:
        return str(settings["models"].get(role, "openai/default"))

    @staticmethod
    def estimate_cost(model: str, tin: int, tout: int) -> tuple[float, bool]:
        return tin * 1e-6 + tout * 4e-6, False


def truth_predictions() -> dict[str, dict[str, Any]]:
    """Predictions equal to the truth (keys p000000...)."""
    out: dict[str, dict[str, Any]] = {}
    for i, p in enumerate(PRODUCTS):
        out[f"p{i:06d}"] = {
            "title": p["title"],
            "price": p["price"],
            "tags": p["tags"],
            "sizes": p["sizes"],
            "specs": p["specs"],
        }
    for j, a in enumerate(ARTICLES, start=len(PRODUCTS)):
        out[f"p{j:06d}"] = {"headline": a["headline"], "author": a["author"]}
    return out
