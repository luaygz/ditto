"""Stage 8: spec runner, JSONL writer, report writer.

``extract_run`` / ``write_report`` load lazily so ``Spec.apply`` (which imports the runner)
stays light enough to embed in a user's own scraper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ditto.extract.report import build_report, write_report
    from ditto.extract.stage import ExtractSummary, TemplateExtract, extract_run

_LAZY = {
    "ExtractSummary": "ditto.extract.stage",
    "TemplateExtract": "ditto.extract.stage",
    "extract_run": "ditto.extract.stage",
    "build_report": "ditto.extract.report",
    "write_report": "ditto.extract.report",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'ditto.extract' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


__all__ = [
    "ExtractSummary",
    "TemplateExtract",
    "build_report",
    "extract_run",
    "write_report",
]
