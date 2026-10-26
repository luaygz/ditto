"""The evaluation baseline interface (``eval/baselines/<name>.py``).

Each baseline module defines::

    def run_baseline(pages: list[PageInput], schema: FieldsSchema, *, out_dir: Path,
                     settings=None, max_pages: int | None = None, seed: int = 0) -> BaselineOutput

and imports :class:`PageInput`, :class:`BaselineOutput` and :class:`NotRun` from this module.
The harness loads baseline modules by file path, so ``eval/`` need not be a package.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

DEFAULT_BASELINES_DIR = Path("eval/baselines")


@dataclass
class PageInput:
    key: str
    url: str
    html: str
    """The page exactly as ditto's extractor sees it (scripts and head except title removed)."""
    template_id: str | None


@dataclass
class BaselineOutput:
    predictions: dict[str, dict[str, Any]]
    """Page key -> {field: value}, same value shapes as ditto output fields."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    """Every call priced at list price, cache hits included (drives cost per 1,000 pages)."""
    notes: list[str] = field(default_factory=list)
    model: str | None = None
    pages_run: int | None = None
    """Pages actually processed (when sampled below the input count)."""
    spend_usd: float | None = None
    """Real spend of this run (cache hits are free); ``None`` means ``cost_usd``."""
    pages_failed: int = 0
    """Pages whose call errored (provider/LLM error or invalid answer after the retry); they
    score as empty. When every processed page failed the harness marks the site failed."""


class NotRun(Exception):
    """Raised by a baseline that cannot run; the message is recorded as the reason."""


RunBaseline = Callable[..., BaselineOutput]


def load_baseline(name: str, baselines_dir: Path = DEFAULT_BASELINES_DIR) -> RunBaseline:
    """Import ``<baselines_dir>/<name>.py`` by file path and return its ``run_baseline``."""
    path = Path(baselines_dir) / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"baseline module not found: {path}")
    module_name = f"ditto_eval_baseline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "run_baseline", None)
    if not callable(fn):
        raise ImportError(f"{path} does not define run_baseline(...)")
    return cast(RunBaseline, fn)
