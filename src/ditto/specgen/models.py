"""The spec format: the single source of truth (exported to ``docs/spec.schema.json``).

A spec is one versioned JSON file per template. It holds XPath 1.0 selectors plus a closed set of
transforms, never code. Validation compiles every selector and lints it (see ``lint.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ditto.specgen.lint import lint_selector
from ditto.specgen.transforms import parse_transform

SPEC_VERSION: Literal["1"] = "1"

FieldType = Literal["scalar", "list", "record_list", "key_value"]
TypeCheck = Literal["email", "url", "number", "date", "price", "text"]


class FieldStats(BaseModel):
    """Per-field statistics written by verification (and needed later for drift detection)."""

    model_config = ConfigDict(extra="allow")

    presence_rate: float | None = None
    expected_presence: float | None = None
    multiplicity: float | None = None
    type_check_pass: float | None = None
    rows_min: int | None = None
    rows_max: int | None = None
    keys_seen: int | None = None
    pages_scored: int | None = None


class _BaseField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    optional: bool = False
    degraded: bool = False
    selection_reason: str | None = None
    source_candidates: list[str] = Field(
        default_factory=list, description="Analysis candidate ids (e.g. c17) this field maps to."
    )
    stats: FieldStats = Field(default_factory=FieldStats)

    def selectors(self) -> list[tuple[str, str, bool, bool]]:
        """``(role, xpath, relative, allow_positional)`` for every selector in the field."""
        raise NotImplementedError

    def transform_lists(self) -> list[list[str]]:
        raise NotImplementedError

    @model_validator(mode="after")
    def _validate_selectors(self) -> _BaseField:
        errors: list[str] = []
        for role, xpath, relative, allow_positional in self.selectors():
            errors.extend(
                f"{role}: {msg}"
                for msg in lint_selector(
                    xpath, relative=relative, allow_positional=allow_positional
                )
            )
        for chain in self.transform_lists():
            for t in chain:
                try:
                    parse_transform(t)
                except ValueError as exc:
                    errors.append(f"transform {t!r}: {exc}")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class _SelectorField(_BaseField):
    candidates: list[str] = Field(min_length=2, max_length=3)
    selected: int = 0
    fallbacks: list[int] = Field(
        default_factory=list,
        description="Candidate indices tried in order when the selected one matches nothing; "
        "verification only lists candidates that agree with the winner.",
    )
    transforms: list[str] = Field(default_factory=list)
    type_check: TypeCheck | None = None

    def selectors(self) -> list[tuple[str, str, bool, bool]]:
        return [(f"candidates[{i}]", x, False, False) for i, x in enumerate(self.candidates)]

    def transform_lists(self) -> list[list[str]]:
        return [self.transforms]

    @model_validator(mode="after")
    def _check_indices(self) -> _SelectorField:
        n = len(self.candidates)
        if not 0 <= self.selected < n:
            raise ValueError(f"selected={self.selected} out of range for {n} candidates")
        if any(not 0 <= i < n for i in self.fallbacks):
            raise ValueError("fallbacks index out of range")
        return self


class ScalarField(_SelectorField):
    type: Literal["scalar"] = "scalar"


class ListField(_SelectorField):
    type: Literal["list"] = "list"


class RecordListField(_BaseField):
    type: Literal["record_list"] = "record_list"
    container: str
    container_alternates: list[str] = Field(default_factory=list, max_length=2)
    columns: dict[str, str] = Field(min_length=1)
    column_source: Literal["header_row", "llm"] = "llm"
    transforms: dict[str, list[str]] = Field(default_factory=dict)

    def selectors(self) -> list[tuple[str, str, bool, bool]]:
        out = [("container", self.container, False, False)]
        out += [
            (f"container_alternates[{i}]", x, False, False)
            for i, x in enumerate(self.container_alternates)
        ]
        # Headerless tables (column_source == "llm") may address cells by position.
        positional_ok = self.column_source == "llm"
        out += [(f"columns.{k}", x, True, positional_ok) for k, x in self.columns.items()]
        return out

    def transform_lists(self) -> list[list[str]]:
        return list(self.transforms.values())

    @model_validator(mode="after")
    def _check_transform_keys(self) -> RecordListField:
        unknown = set(self.transforms) - set(self.columns)
        if unknown:
            raise ValueError(f"transforms for unknown columns: {sorted(unknown)}")
        return self


class KeyValueField(_BaseField):
    type: Literal["key_value"] = "key_value"
    pairs: str
    pairs_alternates: list[str] = Field(default_factory=list, max_length=2)
    key: str = "."
    value: str
    transforms: dict[Literal["key", "value"], list[str]] = Field(default_factory=dict)

    def selectors(self) -> list[tuple[str, str, bool, bool]]:
        out = [("pairs", self.pairs, False, False)]
        out += [
            (f"pairs_alternates[{i}]", x, False, False) for i, x in enumerate(self.pairs_alternates)
        ]
        out += [("key", self.key, True, True), ("value", self.value, True, True)]
        return out

    def transform_lists(self) -> list[list[str]]:
        return list(self.transforms.values())


SpecField = Annotated[
    ScalarField | ListField | RecordListField | KeyValueField, Field(discriminator="type")
]


class GeneratedBy(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt_version: str
    pages_seen: list[str] = Field(default_factory=list)
    mode: Literal["prompted", "unprompted", "fallback"] = "prompted"
    repair_calls: int = 0
    repair_prompt_version: str | None = None
    created_at: str | None = None


class Spec(BaseModel):
    """One template's extraction spec."""

    model_config = ConfigDict(extra="forbid")

    spec_version: Literal["1"] = SPEC_VERSION
    template_id: str
    site: str
    status: Literal["draft", "final"] = "draft"
    generated_by: GeneratedBy
    fields: dict[str, SpecField]
    not_present: dict[str, str] = Field(
        default_factory=dict, description="Prompted fields the LLM could not find -> reason."
    )

    _compiled: Any = PrivateAttr(default=None)

    @classmethod
    def load(cls, path: str | Path) -> Spec:
        return cls.model_validate_json(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2, exclude_none=True) + "\n")

    def apply(self, html: str | bytes) -> dict[str, Any]:
        """Run this spec on one page. Returns ``{<fields...>, "_meta": {missing, spec_version}}``.

        Needs nothing but the HTML: the page is cleaned exactly as during extraction.
        """
        from ditto.extract.runner import apply_spec_html

        return apply_spec_html(self, html)


def spec_json_schema() -> dict[str, Any]:
    return Spec.model_json_schema()
