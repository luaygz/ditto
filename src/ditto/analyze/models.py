"""Candidate table written by cluster analysis (``candidates/<template_id>.json``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CandidateClass = Literal["boilerplate", "label", "mandatory", "optional", "list", "noise"]
TableRole = Literal["kv_key", "kv_value", "record_header", "record_column"]
ValueType = Literal["email", "url", "number", "date", "price", "text"]
TableLayout = Literal["dl", "alternating", "pairs", "table", "rows"]

FIELD_CLASSES: frozenset[str] = frozenset({"mandatory", "optional", "list"})


class Candidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Stable within the template: c0, c1, ...")
    path: str = Field(description="Structural path; attribute paths end with ' @href' / ' @src'.")
    attr: str | None = None
    cls: CandidateClass = Field(alias="class")
    p: float = Field(description="Presence rate over the sampled pages.")
    m: float = Field(description="Modal share: fraction of present pages whose text is the mode.")
    pages_present: int = 0
    repeat_min: int = Field(1, description="Min nodes with this path per page, where present.")
    repeat_max: int = Field(1, description="Max nodes with this path per page, where present.")
    label_hint: str | None = None
    examples: list[str] = Field(default_factory=list, max_length=3)
    value_type: ValueType | None = None
    table_id: str | None = None
    table_role: TableRole | None = None
    column: str | None = Field(None, description="Header text for record columns / key for kv.")

    @property
    def is_field(self) -> bool:
        return self.cls in FIELD_CLASSES


class TableColumn(BaseModel):
    name: str
    candidate_id: str
    header_source: Literal["th", "none"] = "th"


class Table(BaseModel):
    id: str
    kind: Literal["key_value", "record_list"]
    layout: TableLayout | None = Field(
        None,
        description="How rows are built: dl (dt/dd), alternating (dl-like generic siblings), "
        "pairs (row element = key child + value child or tail text), table (<table> rows/cells), "
        "rows (generic repeated sibling rows; columns are distinct sub-paths).",
    )
    container_path: str = Field(description="Path of the table / dl / list element.")
    row_path: str = Field(
        description="Path of the repeated row element (tr, div, li; for dl/alternating the key "
        "element, e.g. dt)."
    )
    p: float = 0.0
    rows_min: int = 0
    rows_max: int = 0
    # key_value
    key_candidate: str | None = None
    value_candidate: str | None = None
    keys_seen: list[str] = Field(default_factory=list)
    # record_list
    columns: list[TableColumn] = Field(default_factory=list)


class CandidateTable(BaseModel):
    template_id: str
    cluster_size: int
    n_sampled: int
    base_page_id: int | None = Field(
        None, description="Sampled page covering the most text paths; ids follow its doc order."
    )
    candidates: list[Candidate]
    tables: list[Table] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> CandidateTable:
        return cls.model_validate_json(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=1, by_alias=True) + "\n")

    def by_id(self, cid: str) -> Candidate:
        for c in self.candidates:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def field_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.is_field]
