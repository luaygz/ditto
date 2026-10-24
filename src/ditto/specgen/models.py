"""The spec format: the single source of truth (exported to ``docs/spec.schema.json``).

A spec is one versioned JSON file per template. It holds XPath 1.0 selectors plus a closed set of
transforms, never code. Validation compiles every selector and lints it (see ``lint.py``),
parses every transform (see ``transforms.py``) and checks indices and names; a spec that loads
is a spec that runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ditto.specgen.lint import lint_selector
from ditto.specgen.transforms import parse_transform

SPEC_VERSION: Literal["1"] = "1"
#: Keys every output row already uses; a field may not take these names (nor start with ``_``).
RESERVED_FIELD_NAMES = ("url", "page_id", "template_id")

FieldType = Literal["scalar", "list", "record_list", "key_value"]
TypeCheck = Literal["email", "url", "number", "date", "price", "text"]

_TRANSFORMS_DOC = (
    "Transforms applied in order to each matched item (closed set: strip, normalize_ws, "
    "regex(pattern, group), split(sep, index), replace(old, new), cast(int|float|date), "
    "attr(name))."
)


class FieldStats(BaseModel):
    """Per-field statistics written by verification over every page of the cluster; kept in the
    spec for drift detection on later crawls. Extra keys are allowed."""

    model_config = ConfigDict(extra="allow")

    presence_rate: float | None = Field(
        default=None, ge=0, le=1, description="Share of pages where the field has a value."
    )
    expected_presence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Presence rate the analysis expected from the field's path (`p`).",
    )
    multiplicity: float | None = Field(
        default=None,
        ge=0,
        description="For scalars: share of pages with a match where the selector matched "
        "exactly one node.",
    )
    type_check_pass: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Share of extracted values that pass the field's `type_check`.",
    )
    rows_min: int | None = Field(
        default=None, ge=0, description="record_list: fewest rows on a page where present."
    )
    rows_max: int | None = Field(
        default=None, ge=0, description="record_list: most rows on a page where present."
    )
    keys_seen: int | None = Field(
        default=None, ge=0, description="key_value: distinct keys seen across the cluster."
    )
    pages_scored: int | None = Field(
        default=None, ge=0, description="Pages the statistics were computed on."
    )


class _BaseField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(
        default=None, description="What the field holds, in words (from fields.yaml or the LLM)."
    )
    optional: bool = Field(
        default=False,
        description="True when the analysis found the field's path on fewer than 95% of the "
        "cluster's pages, so a null is not a failure.",
    )
    degraded: bool = Field(
        default=False,
        description="Set by verification when the field still fails on some pages after the "
        "repair budget is spent; see `stats` for its rates.",
    )
    selection_reason: str | None = Field(
        default=None, description="Why verification picked the selected candidate."
    )
    source_candidates: list[str] = Field(
        default_factory=list, description="Analysis candidate ids (e.g. c17) this field maps to."
    )
    stats: FieldStats = Field(
        default_factory=FieldStats, description="Verification statistics (see `stats`)."
    )

    def selectors(self) -> list[tuple[str, str, bool, bool]]:
        """``(role, xpath, relative, allow_positional)`` for every selector in the field."""
        raise NotImplementedError  # pragma: no cover - every concrete field overrides it

    def transform_lists(self) -> list[list[str]]:
        raise NotImplementedError  # pragma: no cover - every concrete field overrides it

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
    candidates: list[str] = Field(
        min_length=2,
        max_length=3,
        description="Two or three distinct absolute XPath 1.0 selectors for the same value.",
    )
    selected: int = Field(
        default=0, ge=0, description="Index of the candidate extraction uses (set by verification)."
    )
    fallbacks: list[int] = Field(
        default_factory=list,
        description="Candidate indices tried in order when the selected one matches nothing; "
        "verification only lists candidates that agree with the winner.",
    )
    transforms: list[str] = Field(default_factory=list, description=_TRANSFORMS_DOC)
    type_check: TypeCheck | None = Field(
        default=None, description="Expected value type, checked by verification."
    )

    def selectors(self) -> list[tuple[str, str, bool, bool]]:
        return [(f"candidates[{i}]", x, False, False) for i, x in enumerate(self.candidates)]

    def transform_lists(self) -> list[list[str]]:
        return [self.transforms]

    @model_validator(mode="after")
    def _check_indices(self) -> _SelectorField:
        n = len(self.candidates)
        if len({c.strip() for c in self.candidates}) != n:
            raise ValueError("candidates must be distinct selectors")
        if not 0 <= self.selected < n:
            raise ValueError(f"selected={self.selected} out of range for {n} candidates")
        if any(not 0 <= i < n for i in self.fallbacks):
            raise ValueError("fallbacks index out of range")
        return self


class ScalarField(_SelectorField):
    """One value or null. The selector must match exactly one node where the field exists;
    extraction keeps the first non-null match."""

    type: Literal["scalar"] = Field(default="scalar", description="Field type discriminator.")


class ListField(_SelectorField):
    """An array of values: every node the selector matches (0..N), in document order."""

    type: Literal["list"] = Field(default="list", description="Field type discriminator.")


class RecordListField(_BaseField):
    """An array of objects: `container` matches each row, each column XPath is relative to the
    row. Rows whose columns are all null are dropped."""

    type: Literal["record_list"] = Field(
        default="record_list", description="Field type discriminator."
    )
    container: str = Field(description="Absolute XPath matching every row element.")
    container_alternates: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Up to two alternative container selectors (the 2-3 candidates rule); "
        "verification may promote one, extraction uses `container` only.",
    )
    columns: dict[str, str] = Field(
        min_length=1,
        description="Column name -> XPath relative to the row (starts with `.`). With "
        "`column_source: header_row` cells are addressed by header text, never by position.",
    )
    column_source: Literal["header_row", "llm"] = Field(
        default="llm",
        description="`header_row` when column names came from `<th>` text, `llm` when the LLM "
        "named them (headerless tables may then address cells by position).",
    )
    transforms: dict[str, list[str]] = Field(
        default_factory=dict, description="Column name -> transform chain for that column."
    )

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
    def _check_columns(self) -> RecordListField:
        if any(not name.strip() for name in self.columns):
            raise ValueError("column names must not be empty")
        unknown = set(self.transforms) - set(self.columns)
        if unknown:
            raise ValueError(f"transforms for unknown columns: {sorted(unknown)}")
        return self


class KeyValueField(_BaseField):
    """An object: `pairs` matches one element per pair, `key` and `value` are relative to it.
    Keys lose a trailing colon; the first occurrence of a key wins."""

    type: Literal["key_value"] = Field(default="key_value", description="Field type discriminator.")
    pairs: str = Field(description="Absolute XPath matching one element per key/value pair.")
    pairs_alternates: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Up to two alternative pair selectors (the 2-3 candidates rule); "
        "extraction uses `pairs` only.",
    )
    key: str = Field(default=".", description="XPath relative to the pair element for the key.")
    value: str = Field(description="XPath relative to the pair element for the value.")
    transforms: dict[Literal["key", "value"], list[str]] = Field(
        default_factory=dict, description="Transform chains for `key` and/or `value`."
    )

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
    """Provenance: which model and prompt wrote the spec, from which pages. Extra keys allowed."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(description="Model id that wrote the spec.")
    prompt_version: str = Field(description="Prompt id used for generation, e.g. `specgen.v1`.")
    pages_seen: list[str] = Field(
        default_factory=list, description="URLs of the pages merged into the union page."
    )
    mode: Literal["prompted", "unprompted", "fallback"] = Field(
        default="prompted",
        description="`prompted` (user fields.yaml), `unprompted` (LLM named the fields) or "
        "`fallback` (path-induced selectors, no LLM).",
    )
    repair_calls: int = Field(default=0, ge=0, description="Repair LLM calls spent on this spec.")
    repair_prompt_version: str | None = Field(
        default=None, description="Prompt id used for repair, e.g. `repair.v1`."
    )
    created_at: str | None = Field(default=None, description="ISO 8601 creation time.")


class Spec(BaseModel):
    """One template's extraction spec."""

    model_config = ConfigDict(extra="forbid")

    spec_version: Literal["1"] = Field(
        default=SPEC_VERSION, description='Format version; `"1"` for this release.'
    )
    template_id: str = Field(min_length=1, description="Template (cluster) id, e.g. `t_3f9a`.")
    site: str = Field(min_length=1, description="Site the template belongs to (its domain).")
    status: Literal["draft", "final"] = Field(
        default="draft",
        description="`draft` after generation, `final` once verification selected candidates.",
    )
    generated_by: GeneratedBy = Field(description="Provenance (see `generated_by`).")
    fields: dict[str, SpecField] = Field(
        description="Output field name -> field definition. Names are the keys of every "
        "extracted record; `url`, `page_id`, `template_id` and names starting with `_` are "
        "reserved."
    )
    not_present: dict[str, str] = Field(
        default_factory=dict, description="Prompted fields the LLM could not find -> reason."
    )

    @model_validator(mode="after")
    def _check_names(self) -> Spec:
        bad = [
            name
            for name in self.fields
            if not name.strip() or name.startswith("_") or name in RESERVED_FIELD_NAMES
        ]
        if bad:
            raise ValueError(
                f"reserved or empty field names {bad}; {', '.join(RESERVED_FIELD_NAMES)} and "
                "names starting with '_' are output-row keys"
            )
        both = sorted(set(self.fields) & set(self.not_present))
        if both:
            raise ValueError(f"fields listed both in fields and not_present: {both}")
        return self

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
    """JSON Schema of :class:`Spec`. The ``type`` discriminator of every field definition is
    listed as required: parsing a spec needs it even though the Python models default it."""
    schema = Spec.model_json_schema()
    for name in ("ScalarField", "ListField", "RecordListField", "KeyValueField"):
        definition = schema["$defs"][name]
        definition["required"] = ["type", *definition.get("required", [])]
    return schema
