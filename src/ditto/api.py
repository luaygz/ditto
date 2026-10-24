"""Python API: ``Site`` wraps a run directory and exposes each stage as a method.

    site = Site.from_dir("pages/", run_dir="runs/example")
    site.cluster(); site.analyze()
    probe = site.probe(FieldsSchema.load("fields.yaml"))
    mapping = probe.confirm()
    site.generate("prompted", schema=..., mapping=mapping)
    site.verify(); site.review(); site.extract()

For one finished spec on one page, use ``Spec.load(path).apply(html)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import orjson

from ditto.run import RunDir
from ditto.schema import FieldsSchema, Mapping


def _default_run_dir(src: Path) -> Path:
    return Path.cwd() / "ditto-runs" / (src.stem or "site")


@dataclass
class Probe:
    """Result of ``Site.probe``; ``confirm`` turns it into ``mapping.json``."""

    site: Site
    result: Any

    def confirm(self, edits: dict[str, dict[str, str | list[str]]] | None = None) -> Mapping:
        """Accept the proposal, optionally overriding ``{template_id: {field: cid|"absent"}}``."""
        from ditto.checkpoints import confirm_run
        from ditto.schema import TemplateMapping

        mapping = confirm_run(self.site.run, yes=True, interactive=False)
        if edits:
            for tid, fields in edits.items():
                tm = mapping.templates.setdefault(tid, TemplateMapping())
                tm.fields.update(fields)
            mapping.save(self.site.run.mapping_json)
        return mapping


class Site:
    """A downloaded site and its run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run = RunDir(Path(run_dir))

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        manifest: str | Path | None = None,
        *,
        run_dir: str | Path | None = None,
        workers: int | None = None,
    ) -> Site:
        from ditto.ingest import ingest

        src = Path(path)
        out = Path(run_dir) if run_dir else _default_run_dir(src)
        ingest(src, out, manifest=Path(manifest) if manifest else None, workers=workers)
        return cls(out)

    @classmethod
    def from_warc(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        run_dir: str | Path | None = None,
        workers: int | None = None,
    ) -> Site:
        from ditto.ingest import ingest

        items = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
        out = Path(run_dir) if run_dir else _default_run_dir(items[0])
        ingest(items if len(items) > 1 else items[0], out, workers=workers)
        return cls(out)

    @classmethod
    def open(cls, run_dir: str | Path) -> Site:
        """Reopen an existing run directory."""
        return cls(run_dir)

    # -- stages ---------------------------------------------------------------------------------
    def cluster(self, *, min_cluster_size: int = 20, workers: int | None = None) -> Any:
        from ditto.pipeline import stage_cluster

        return stage_cluster(self.run, min_cluster_size=min_cluster_size, workers=workers)

    def analyze(self, *, include_tail: bool = False, workers: int | None = None) -> dict[str, Any]:
        from ditto.pipeline import stage_analyze

        return stage_analyze(self.run, include_tail=include_tail, workers=workers)

    def probe(self, schema: FieldsSchema | str | Path, *, use_llm: bool = False) -> Probe:
        from ditto.checkpoints import probe_run

        fs = schema if isinstance(schema, FieldsSchema) else FieldsSchema.load(schema)
        llm = None
        if use_llm:
            from ditto.llm import get_llm

            llm = get_llm("probe", run=self.run)
        return Probe(self, probe_run(self.run, fs, use_llm=use_llm, llm=llm))

    def generate(
        self,
        mode: Literal["prompted", "unprompted"],
        schema: FieldsSchema | str | Path | None = None,
        mapping: Mapping | None = None,
        llm: Any = None,
        *,
        min_cluster_size: int = 20,
    ) -> Any:
        from ditto.llm import get_llm
        from ditto.specgen import generate_run

        fs = None
        if schema is not None:
            fs = schema if isinstance(schema, FieldsSchema) else FieldsSchema.load(schema)
        if mapping is None and mode == "prompted" and self.run.mapping_json.exists():
            mapping = Mapping.load(self.run.mapping_json)
        return generate_run(
            self.run,
            mode=mode,
            schema=fs,
            mapping=mapping,
            llm=llm or get_llm("generation", run=self.run),
            min_cluster_size=min_cluster_size,
        )

    def verify(self, max_repair_calls: int = 3, *, llm: Any = None) -> Any:
        from ditto.verify import verify_run

        if llm is None and max_repair_calls > 0:
            from ditto.llm import get_llm

            llm = get_llm("repair", run=self.run)
        return verify_run(self.run, max_repair_calls=max_repair_calls, llm=llm)

    def review(self, *, n: int = 5, seed: int = 0) -> Any:
        """Non-interactive review: returns the sampled records and saves golden samples."""
        from ditto.checkpoints import review_run

        return review_run(self.run, yes=True, n=n, seed=seed)

    def extract(self, *, workers: int | None = None) -> Any:
        from ditto.extract import extract_run, write_report

        summary = extract_run(self.run, workers=workers)
        write_report(self.run)
        return summary

    # -- results --------------------------------------------------------------------------------
    def records(self) -> Iterator[dict[str, Any]]:
        """Iterate every extracted record of every template."""
        out = self.run.root / "out"
        for path in sorted(out.glob("*/*.jsonl")):
            with path.open("rb") as fh:
                for line in fh:
                    if line.strip():
                        yield orjson.loads(line)

    def specs(self) -> dict[str, Any]:
        from ditto.specgen.models import Spec

        return {p.stem: Spec.load(p) for p in sorted(self.run.specs_dir.glob("t_*.json"))}
