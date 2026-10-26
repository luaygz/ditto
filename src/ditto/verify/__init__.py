"""Stage 7: verification and repair.

Every candidate selector of every field runs on every page of its cluster; winners are chosen
by presence × multiplicity × type pass; failing signature groups get one repair call each
(capped per template); fields still failing are marked degraded; the stage-3 merge/split rules
run last and the final specs are written.
"""

from __future__ import annotations

from ditto.verify.fallback import induced_field, path_induced_selectors
from ditto.verify.models import CandidateScore, FieldReport, RepairRecord, VerifyReport
from ditto.verify.stage import repair_rejected, verify_run

__all__ = [
    "CandidateScore",
    "FieldReport",
    "RepairRecord",
    "VerifyReport",
    "induced_field",
    "path_induced_selectors",
    "repair_rejected",
    "verify_run",
]
