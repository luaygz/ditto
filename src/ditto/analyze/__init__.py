"""Stage 4: cluster analysis (candidate tables per template)."""

from __future__ import annotations

from ditto.analyze.core import analyze_pages, classify
from ditto.analyze.locate import locate_candidates
from ditto.analyze.models import Candidate, CandidateTable, Table, TableColumn
from ditto.analyze.stage import analyze_run

__all__ = [
    "Candidate",
    "CandidateTable",
    "Table",
    "TableColumn",
    "analyze_pages",
    "analyze_run",
    "classify",
    "locate_candidates",
]
