"""Verification report written to ``verify/<template_id>.json`` (one per template)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

CandidateOrigin = Literal["llm", "repair", "path_induced", "merged"]
FailureKind = Literal["missing", "multiple", "type"]


class CandidateScore(BaseModel):
    """Scores of one candidate selector (or one container/pairs selector) over every page."""

    index: int
    role: str = Field(description="candidates[i] | container | container_alternates[i] | pairs …")
    xpath: str
    origin: CandidateOrigin = "llm"
    pages_matched: int = 0
    presence_rate: float = 0.0
    expected_presence: float | None = None
    presence_score: float = Field(
        0.0, description="Share of the pages where the analysis expects the field that it fills."
    )
    multiplicity: float | None = None
    type_check_pass: float | None = None
    score: float = Field(0.0, description="presence_score × multiplicity × type_check_pass")
    positional_steps: int = 0
    agreement: dict[str, float | None] = Field(
        default_factory=dict,
        description="Other candidate index -> value agreement where both match.",
    )
    rows_min: int | None = None
    rows_max: int | None = None
    keys_seen: int | None = None


class PageFailure(BaseModel):
    page_id: int
    signature_group: int | None = None
    kind: FailureKind


class FieldReport(BaseModel):
    name: str
    type: str
    selected: int = 0
    selected_xpath: str = ""
    selection_reason: str | None = None
    value_type: str | None = Field(None, description="Type checked (type_check or inferred).")
    expected_presence: float | None = None
    analysis_p: float | None = None
    optional: bool = False
    degraded: bool = False
    path_induced: bool = Field(False, description="Path-induced fallback selectors were added.")
    candidates: list[CandidateScore] = Field(default_factory=list)
    failures: int = 0
    failure_rate: float = 0.0
    failures_by_kind: dict[str, int] = Field(default_factory=dict)
    failures_by_group: dict[str, int] = Field(
        default_factory=dict, description="Signature group id -> failing pages."
    )
    failure_examples: list[PageFailure] = Field(default_factory=list, max_length=10)


class RepairRecord(BaseModel):
    call: int
    signature_group: int | None = None
    page_id: int
    fields: list[str]
    accepted: list[str] = Field(default_factory=list)
    failures_before: dict[str, int] = Field(default_factory=dict)
    failures_after: dict[str, int] = Field(default_factory=dict)
    note: str | None = Field(None, description="User note (review rejections).")
    error: str | None = None


class VerifyReport(BaseModel):
    """Everything verification measured and changed for one template."""

    template_id: str
    llm_seconds: float = Field(0.0, description="Time spent waiting on repair calls.")
    status: Literal["final", "merged", "failed"] = "final"
    spec_source: str = Field("draft", description="draft | final | fallback | split_parent")
    n_pages: int = 0
    seconds: float = 0.0
    fields: dict[str, FieldReport] = Field(default_factory=dict)
    not_present: dict[str, str] = Field(default_factory=dict)
    repairs: list[RepairRecord] = Field(default_factory=list)
    repair_calls: int = 0
    degraded: list[str] = Field(default_factory=list)
    merge_split: list[dict[str, Any]] = Field(default_factory=list)
    rejections: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> VerifyReport:
        return cls.model_validate_json(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=1) + "\n")
