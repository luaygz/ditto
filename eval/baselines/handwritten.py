"""Hand-written XPath wrappers for five SWDE test sites: the selector-quality ceiling.

The wrappers are ordinary ditto specs (``eval/baselines/handwritten/<vertical>-<site>.json``,
validated by :class:`ditto.specgen.models.Spec`) written by the build agent, not by a human,
before any ditto-generated spec existed; each spec's ``generated_by`` records the author, the
pages read while writing it and how it was checked. Applying them uses ``Spec.apply`` (the same
cleaning and extractor as ditto) and calls no model, so calls, tokens and cost are zero.

Sites: ``auto-yahoo``, ``book-barnesandnoble``, ``movie-imdb``, ``nbaplayer-yahoo``,
``university-usnews`` (five verticals, all in the SWDE test split). Any other site raises
``NotRun``.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ditto.evaluation.baseline_api import BaselineOutput, NotRun, PageInput

if TYPE_CHECKING:
    from ditto.schema import FieldsSchema
    from ditto.specgen.models import Spec


def _load_common() -> ModuleType:
    name = "ditto_eval_baseline_common"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("common.py"))
        if spec is None or spec.loader is None:  # pragma: no cover - file ships with this one
            raise ImportError("eval/baselines/common.py not found")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


common = _load_common()

NAME = "handwritten"
SPEC_DIR = Path(__file__).with_name("handwritten")


def available_sites(spec_dir: Path = SPEC_DIR) -> list[str]:
    """Site names (``<vertical>-<site>``) that have a hand-written wrapper."""
    return sorted(p.stem for p in spec_dir.glob("*.json"))


@lru_cache(maxsize=32)
def load_spec(path: str) -> Spec:
    from ditto.specgen.models import Spec

    return Spec.load(path)


def spec_path(site: str, spec_dir: Path = SPEC_DIR) -> Path:
    path = spec_dir / f"{site}.json"
    if not path.is_file():
        raise NotRun(
            f"no hand-written wrapper for site {site!r}; available: {available_sites(spec_dir)}"
        )
    return path


def predict(spec: Spec, html: str, schema: FieldsSchema) -> dict[str, Any]:
    """Apply ``spec`` to one page; keep exactly the schema's fields in ditto's shapes."""
    row = spec.apply(html)
    out: dict[str, Any] = {}
    for f in schema.fields:
        kind = common.field_kind(f)
        out[f.name] = row[f.name] if f.name in spec.fields else common.empty_value(kind)
    return out


def _apply_chunk(
    chunk: Sequence[tuple[str, str, str, str]],
) -> list[tuple[str, dict[str, Any]]]:
    """pmap worker: ``(key, html, spec_path, fields_schema_json)`` per page."""
    from ditto.schema import FieldsSchema

    out: list[tuple[str, dict[str, Any]]] = []
    schemas: dict[str, FieldsSchema] = {}
    for key, html, path, fields_json in chunk:
        fields = schemas.get(fields_json)
        if fields is None:
            fields = schemas[fields_json] = FieldsSchema.model_validate_json(fields_json)
        out.append((key, predict(load_spec(path), html, fields)))
    return out


def run_baseline(
    pages: list[PageInput],
    schema: FieldsSchema,
    *,
    out_dir: Path,
    settings: Any = None,
    max_pages: int | None = None,
    seed: int = 0,
    site: str | None = None,
    workers: int | None = None,
    spec_dir: Path = SPEC_DIR,
) -> BaselineOutput:
    """Apply the site's hand-written wrapper to every page (no model calls)."""
    from ditto.parallel import pmap_flat

    if site is None:
        raise NotRun("the hand-written baseline needs the site name (site=...)")
    path = spec_path(site, spec_dir)
    spec = load_spec(str(path))
    out_dir = Path(out_dir)
    selected = common.sample_pages(pages, max_pages, seed)
    fields_json = schema.model_dump_json()
    work = [(p.key, p.html, str(path), fields_json) for p in selected]

    started = time.perf_counter()
    predictions = dict(pmap_flat(_apply_chunk, work, workers=workers, chunk_size=64))
    seconds = time.perf_counter() - started

    missing = [f.name for f in schema.fields if f.name not in spec.fields]
    notes = [
        f"hand-written XPath wrapper {path.name} (written by the build agent, not a human; "
        f"see generated_by in the spec) applied with Spec.apply to {len(selected)} pages "
        f"in {seconds:.2f}s; no model calls",
    ]
    if missing:
        notes.append(f"requested fields without a hand-written selector (empty): {missing}")
    output = BaselineOutput(
        predictions=predictions,
        calls=0,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        notes=notes,
        model=None,
        pages_run=len(selected),
        spend_usd=0.0,
    )
    common.write_jsonl(
        out_dir / f"{NAME}.predictions.jsonl", common.predictions_rows(selected, predictions)
    )
    common.write_summary(
        out_dir, NAME, output, {"site": site, "spec": str(path), "seconds": seconds}
    )
    return output
