"""Speed of the page-level stages, in pages per second per core.

Times ``ingest``, ``fingerprint`` (parse + clean + structural fingerprint) and ``extract`` (read +
parse + clean + every spec field + JSON) through their public entry points, once in a single
process and once with all workers, and prints a Markdown table. Each measurement is the best
of ``--repeat`` runs (the machine may be shared); CPU seconds include worker processes.

Cases (no LLM involved; the specs are data under ``eval/bench/specs/``):

* ``fixture``: the synthetic test site (200 small pages) with the final specs and clusters of a
  ``ditto run`` over it.
* ``auto-aol``, ``book-abebooks``: SWDE development-slice sites (2,000 pages of ~50-65 KB each),
  every page extracted with a hand-written benchmark spec (one cluster of all pages). The
  specs exist for speed only; they were never scored or tuned against SWDE labels.
* ``--handwritten``: the five SWDE sites of ``eval/baselines/handwritten`` with those specs.
* ``--warc``: each SWDE case again, read from a single ``.warc.gz`` built from its pages
  (gzip record per page, as Common Crawl stores them).

Usage::

    uv run python eval/bench/speed.py [--handwritten] [--warc] [--profile] [--json out.json]
    uv run python eval/bench/speed.py --run RUN_DIR             # fingerprint + extract only
    uv run python eval/bench/speed.py --pages DIR --spec SPEC [--manifest urls.tsv]
    uv run python eval/bench/speed.py compare RUN_A RUN_B       # extraction outputs identical?

``--profile`` adds a single-process breakdown of extraction per page: read, input preparation
(decode / UTF-8 check / XML-safe), libxml2 parse, the character-reference check (and tree
sanitizing when a reference names an XML-invalid character), freeing the previous tree,
cleaning, spec evaluation and JSON. ``compare`` checks every ``out/**/*.jsonl`` row of two
runs byte for byte after re-serializing with sorted keys.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import orjson

REPO = Path(__file__).resolve().parents[2]
BENCH_SPECS = Path(__file__).resolve().parent / "specs"
HANDWRITTEN = REPO / "eval" / "baselines" / "handwritten"
SWDE = REPO / "data" / "swde"
BENCH_TID = "t_bench"


@dataclass
class Case:
    name: str
    source: Path  # HTML directory, or a WARC file
    manifest: Path | None = None
    specs: list[Path] = field(default_factory=list)
    clusters: Path | None = None  # None: one cluster of every page, extracted with specs[0]
    run: Path | None = None  # an existing run: no ingest, its own clusters and specs


@dataclass
class Row:
    case: str
    stage: str
    workers: int
    pages: int
    kb_per_page: float
    seconds: float
    cpu_seconds: float

    @property
    def pages_per_sec(self) -> float:
        return self.pages / self.seconds if self.seconds else 0.0

    @property
    def per_worker(self) -> float:
        return self.pages_per_sec / self.workers

    @property
    def pages_per_cpu_sec(self) -> float:
        return self.pages / self.cpu_seconds if self.cpu_seconds else 0.0


def swde_case(site: str, spec: Path) -> Case:
    vertical = site.split("-", 1)[0]
    found = sorted((SWDE / vertical).glob(f"{site}(*)"))
    if not found:
        raise SystemExit(f"SWDE site {site} not found under {SWDE / vertical}")
    return Case(site, found[0], SWDE / "urls" / f"{site}.tsv", [spec])


def default_cases(handwritten: bool) -> list[Case]:
    fixture = BENCH_SPECS / "fixture"
    cases = [
        Case(
            "fixture",
            REPO / "tests" / "fixtures" / "site" / "site",
            specs=sorted(fixture.glob("t_*.json")),
            clusters=fixture / "clusters.json",
        ),
        swde_case("auto-aol", BENCH_SPECS / "auto-aol.json"),
        swde_case("book-abebooks", BENCH_SPECS / "book-abebooks.json"),
    ]
    if handwritten:
        cases += [swde_case(p.stem, p) for p in sorted(HANDWRITTEN.glob("*.json"))]
    return cases


def _cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def best_of(
    repeat: int, fn: Callable[[], int], before: Callable[[], None] | None = None
) -> tuple[float, float, int]:
    """(wall seconds, CPU seconds, pages) of the fastest of ``repeat`` calls of ``fn``."""
    best: tuple[float, float, int] | None = None
    for _ in range(repeat):
        if before is not None:
            before()
        gc.collect()
        cpu0, t0 = _cpu(), time.perf_counter()
        pages = fn()
        wall, cpu = time.perf_counter() - t0, _cpu() - cpu0
        if best is None or wall < best[0]:
            best = (wall, cpu, pages)
    assert best is not None
    return best


def build_warc(case: Case, out: Path) -> Path:
    """One gzipped WARC holding an HTTP 200 ``response`` record per page of ``case``."""
    if out.exists():
        return out
    from io import BytesIO

    from warcio.statusandheaders import StatusAndHeaders
    from warcio.warcwriter import WARCWriter

    from ditto.ingest.directory import list_html_files, read_manifest

    urls = read_manifest(case.manifest, case.source).urls if case.manifest else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        writer = WARCWriter(fh, gzip=True)
        for rel, _ in list_html_files(case.source).files:
            body = (case.source / rel).read_bytes()
            http = StatusAndHeaders(
                "200 OK",
                [("Content-Type", "text/html"), ("Content-Length", str(len(body)))],
                protocol="HTTP/1.1",
            )
            record = writer.create_warc_record(
                urls.get(rel, rel),
                "response",
                payload=BytesIO(body),
                http_headers=http,
                warc_headers_dict={"WARC-Date": "2026-01-01T00:00:00Z"},
            )
            writer.write_record(record)
    tmp.replace(out)
    return out


def install_specs(case: Case, run_root: Path) -> None:
    """clusters.json and final specs for ``case`` in a freshly ingested run."""
    from ditto.cluster.models import Cluster, Clusters
    from ditto.ingest import PageStore
    from ditto.run import RunDir
    from ditto.specgen.models import Spec

    run = RunDir(run_root)
    shutil.rmtree(run.specs_dir, ignore_errors=True)
    run.specs_dir.mkdir(parents=True)
    if case.clusters is not None:
        shutil.copyfile(case.clusters, run.clusters_json)
        for path in case.specs:
            shutil.copyfile(path, run.spec(path.stem))
        return
    n = len(PageStore.open(run))
    Clusters(
        n_pages=n,
        min_cluster_size=1,
        clusters=[
            Cluster(
                template_id=BENCH_TID,
                size=n,
                is_tail=False,
                signature_groups=[],
                page_ids=list(range(n)),
            )
        ],
        signature_groups=[],
    ).save(run.clusters_json)
    spec = Spec.load(case.specs[0])
    spec.template_id, spec.status = BENCH_TID, "final"
    spec.save(run.spec(BENCH_TID))


def kb_per_page(run_root: Path) -> float:
    from ditto.ingest import PageStore

    store = PageStore.open(run_root)
    total = sum(len(store.read_bytes(pid)) for pid in store.ids())
    return total / max(1, len(store)) / 1024


def bench_case(
    case: Case, work: Path, workers: list[int], repeat: int, stages: set[str]
) -> list[Row]:
    from ditto.extract import extract_run
    from ditto.fingerprint import fingerprint_run
    from ditto.ingest import ingest
    from ditto.run import RunDir

    rows: list[Row] = []
    if case.run is not None:
        run_root = case.run
    else:
        run_root = work / case.name / "run"
        if "ingest" in stages or not (run_root / "pages.sqlite").exists():
            for w in workers:

                def fresh() -> None:
                    shutil.rmtree(run_root, ignore_errors=True)

                def do_ingest(w: int = w) -> int:
                    return ingest(case.source, run_root, manifest=case.manifest, workers=w).n_pages

                wall, cpu, pages = best_of(repeat if "ingest" in stages else 1, do_ingest, fresh)
                if "ingest" in stages:
                    used = used_workers(run_root, "ingest", w)
                    rows.append(Row(case.name, "ingest", used, pages, 0.0, wall, cpu))
        install_specs(case, run_root)
    run = RunDir(run_root)
    kb = kb_per_page(run_root)
    for row in rows:
        row.kb_per_page = kb
    for stage in ("fingerprint", "extract"):
        if stage not in stages:
            continue
        for w in workers:
            if stage == "fingerprint":

                def do(w: int = w) -> int:
                    return fingerprint_run(run, workers=w).n_pages
            else:

                def do(w: int = w) -> int:
                    return extract_run(run, workers=w).pages

            wall, cpu, pages = best_of(repeat, do)
            used = used_workers(run_root, stage, w)
            rows.append(Row(case.name, stage, used, pages, kb, wall, cpu))
    return rows


def used_workers(run_root: Path, stage: str, asked: int) -> int:
    """Worker processes the stage actually used (it caps them by its chunk count), from the
    run's ``timings.json``."""
    try:
        return int(json.loads((run_root / "timings.json").read_text())[stage]["workers"])
    except (OSError, KeyError, ValueError):
        return asked


def profile_case(case_name: str, run_root: Path, max_pages: int) -> dict[str, float]:
    """Per-page milliseconds of each extraction phase, one process, over the run's specs."""
    from lxml import etree

    from ditto.cluster.models import Clusters
    from ditto.extract.runner import apply_spec
    from ditto.extract.stage import row_for
    from ditto.fingerprint import dom
    from ditto.ingest import PageStore
    from ditto.run import RunDir
    from ditto.specgen.models import Spec

    run = RunDir(run_root)
    store = PageStore.open(run)
    clusters = Clusters.load(run.clusters_json)
    items: list[tuple[int, Spec]] = []
    for path in sorted(run.specs_dir.glob("*.json")):
        spec = Spec.load(path)
        items += [(pid, spec) for pid in sorted(clusters.get(spec.template_id).page_ids)]
    items = items[:max_pages]
    parser = dom._parser(True, utf8=True)
    phases = dict.fromkeys(
        ("read", "prepare", "parse", "refs", "free", "clean", "apply", "json"), 0.0
    )
    clock = time.perf_counter
    root = None
    for pid, spec in items:
        t0 = clock()
        raw, encoding = store.read_page(pid)
        t1 = clock()
        source, sniffed = dom._source(raw, encoding)
        t2 = clock()
        del root  # the previous page's tree: freed here, as in the stage loop
        t3 = clock()
        if sniffed or not source.strip():  # never for ingested pages (they have an encoding)
            root = dom._parse(raw, encoding, True)
            t4 = t5 = clock()
        else:
            root = etree.fromstring(source, parser)
            t4 = clock()
            if dom._refers_to_forbidden(source):
                dom._sanitize_tree(root)
            t5 = clock()
        dom._clean(root, comments=False)
        t6 = clock()
        values = apply_spec(spec, root)
        t7 = clock()
        orjson.dumps(row_for(spec, pid, store.url(pid), values))
        t8 = clock()
        for key, a, b in (
            ("read", t0, t1),
            ("prepare", t1, t2),
            ("free", t2, t3),
            ("parse", t3, t4),
            ("refs", t4, t5),
            ("clean", t5, t6),
            ("apply", t6, t7),
            ("json", t7, t8),
        ):
            phases[key] += b - a
    n = max(1, len(items))
    out = {k: v / n * 1000 for k, v in phases.items()}
    total = sum(out.values())
    print(f"\n{case_name}: extraction profile, one process, {len(items)} pages")
    print("| phase | ms/page | share |\n| --- | ---: | ---: |")
    for key, ms in out.items():
        print(f"| {key} | {ms:.3f} | {ms / total * 100:.1f}% |")
    print(f"| total | {total:.3f} | {1000 / total:.0f} pages/s |")
    return out


def compare_runs(a: Path, b: Path) -> int:
    """Exit status 0 when every extraction row of ``a`` and ``b`` is identical."""
    files_a = {p.relative_to(a / "out") for p in (a / "out").rglob("*.jsonl")}
    files_b = {p.relative_to(b / "out") for p in (b / "out").rglob("*.jsonl")}
    status = 0
    if files_a != files_b:
        print(
            f"different files: only in A {sorted(files_a - files_b)}, "
            f"only in B {sorted(files_b - files_a)}"
        )
        status = 1
    rows = diffs = 0
    for rel in sorted(files_a & files_b):
        la = (a / "out" / rel).read_bytes().splitlines()
        lb = (b / "out" / rel).read_bytes().splitlines()
        if len(la) != len(lb):
            print(f"{rel}: {len(la)} vs {len(lb)} rows")
            status = 1
        for i, (x, y) in enumerate(zip(la, lb, strict=False)):
            rows += 1
            sx = orjson.dumps(orjson.loads(x), option=orjson.OPT_SORT_KEYS)
            sy = orjson.dumps(orjson.loads(y), option=orjson.OPT_SORT_KEYS)
            if sx != sy:
                diffs += 1
                status = 1
                if diffs <= 5:
                    print(f"{rel} row {i}:\n  A {sx[:300]!r}\n  B {sy[:300]!r}")
    print(f"compared {len(files_a & files_b)} files, {rows} rows: {diffs} differ")
    return status


def machine() -> dict[str, Any]:
    import lxml.etree as etree
    import psutil

    model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": model,
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
        "lxml": ".".join(map(str, etree.LXML_VERSION)),
        "libxml2": ".".join(map(str, etree.LIBXML_VERSION)),
        "load_average": [round(x, 2) for x in os.getloadavg()],
    }


def print_table(rows: list[Row]) -> None:
    print(
        "\n| case | stage | workers | pages | KB/page | seconds | pages/s | pages/s/worker "
        "| CPU s | pages/CPU-s |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r.case} | {r.stage} | {r.workers} | {r.pages} | {r.kb_per_page:.1f} "
            f"| {r.seconds:.3f} | {r.pages_per_sec:,.0f} | {r.per_worker:,.0f} "
            f"| {r.cpu_seconds:.2f} | {r.pages_per_cpu_sec:,.0f} |"
        )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compare"]:
        if len(argv) != 3:
            raise SystemExit("usage: speed.py compare RUN_A RUN_B")
        return compare_runs(Path(argv[1]), Path(argv[2]))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "data" / "bench")
    ap.add_argument("--workers", default="1,all", help="comma list; 'all' = default_workers()")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--stages", default="ingest,fingerprint,extract")
    ap.add_argument("--handwritten", action="store_true")
    ap.add_argument("--warc", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-pages", type=int, default=1000)
    ap.add_argument("--only", default="", help="comma list of case names to run")
    ap.add_argument("--run", type=Path, help="benchmark an existing run directory")
    ap.add_argument("--pages", type=Path, help="HTML directory or WARC file")
    ap.add_argument("--spec", type=Path, help="spec applied to every page of --pages")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--json", type=Path, help="also write results as JSON")
    args = ap.parse_args(argv)

    from ditto.parallel import default_workers

    workers = sorted(
        {default_workers() if w.strip() == "all" else int(w) for w in args.workers.split(",")}
    )
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    if args.run is not None:
        name = args.run.parent.name if args.run.name == "run" else args.run.name
        cases = [Case(name, args.run, run=args.run)]
        stages.discard("ingest")
    elif args.pages is not None:
        if args.spec is None:
            raise SystemExit("--pages needs --spec")
        cases = [Case(args.pages.stem, args.pages, args.manifest, [args.spec])]
    else:
        cases = default_cases(args.handwritten)
        if args.warc:
            cases += [
                Case(
                    f"{c.name}-warc",
                    build_warc(c, args.work / "warc" / f"{c.name}.warc.gz"),
                    specs=c.specs,
                )
                for c in cases
                if c.clusters is None
            ]
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        cases = [c for c in cases if c.name in keep]
    info = machine()
    print(f"machine: {info}")
    rows: list[Row] = []
    profiles: dict[str, dict[str, float]] = {}
    for case in cases:
        case_rows = bench_case(case, args.work, workers, args.repeat, stages)
        rows += case_rows
        print_table(case_rows)
        if args.profile:
            run_root = case.run or args.work / case.name / "run"
            profiles[case.name] = profile_case(case.name, run_root, args.profile_pages)
    print("\nAll results:")
    print_table(rows)
    info["load_average_end"] = [round(x, 2) for x in os.getloadavg()]
    print(f"load average start {info['load_average']}, end {info['load_average_end']}")
    if args.json is not None:
        payload = {
            "machine": info,
            "repeat": args.repeat,
            "rows": [
                {
                    **asdict(r),
                    "pages_per_sec": round(r.pages_per_sec, 1),
                    "pages_per_sec_per_worker": round(r.per_worker, 1),
                    "pages_per_cpu_sec": round(r.pages_per_cpu_sec, 1),
                }
                for r in rows
            ],
            "profiles": profiles,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
