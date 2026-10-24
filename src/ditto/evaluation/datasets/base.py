"""Dataset adapter protocol for the evaluation harness.

Every evaluation dataset (synthetic, SWDE, Common Crawl silver, LiveWeb-IE) lives in a module
``ditto.evaluation.datasets.<name>`` that exposes one function::

    def load_sites(selection: Selection, work_dir: Path) -> list[EvalSite]: ...

``work_dir`` is a scratch directory owned by the evaluation run (adapters may write derived
inputs there, e.g. a ``urls.tsv`` manifest or a merged ``fields.yaml``). Adapters raise
:class:`DatasetUnavailable` when the data has not been downloaded, so the runner can record the
evaluation as not run with the reason instead of crashing.

Label value shapes per field kind. A field key absent from a page's label dict means the field
is not annotated for that page (it is not scored there); a present key with an empty value
(``None``, ``""``, ``[]`` or ``{}``) means the page is labeled as lacking it.

- ``scalar``: ``str`` or a ``list[str]`` of acceptable alternatives (SWDE multi-value attributes:
  a prediction matching any of them counts).
- ``list``: ``list[str]``, the full list of values on the page.
- ``record_list``: ``list[dict[column, str | None]]``, rows in page order.
- ``key_value``: ``dict[key, str]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urldefrag

InputKind = Literal["dir", "warc"]
LabelKind = Literal["scalar", "list", "record_list", "key_value"]
Split = Literal["dev", "test", "all"]


class DatasetUnavailable(RuntimeError):
    """The dataset is not on disk (not downloaded, or skipped with a recorded reason)."""


@dataclass
class EvalSite:
    """One site to evaluate: what ditto ingests plus the labels it is scored against.

    ``labels`` maps a page key to ``{field: value}``; ``url_to_key`` maps the URL that ingest
    records for a page (the ``urls.tsv`` URL for directory input, ``WARC-Target-URI`` for WARC
    input) to that page key. Pages without a labels entry are not scored.
    """

    name: str
    vertical: str
    input_paths: list[Path]
    input_kind: InputKind
    fields_schema_path: Path | None
    labels: dict[str, dict[str, Any]]
    url_to_key: dict[str, str]
    manifest: Path | None = None
    """``urls.tsv`` (``path<TAB>url``) passed to ingest for directory input."""
    field_kinds: dict[str, LabelKind] = field(default_factory=dict)
    """Field -> label kind; fields not listed are ``scalar``."""
    truth_templates: dict[str, str] = field(default_factory=dict)
    """Page key -> true template id (synthetic sites; enables clustering metrics)."""
    field_groups: dict[str, list[str]] = field(default_factory=dict)
    """Named field groups reported separately, e.g. LiveWeb-IE complexity types."""
    meta: dict[str, Any] = field(default_factory=dict)

    def kind(self, name: str) -> LabelKind:
        return self.field_kinds.get(name, "scalar")

    def fields(self) -> list[str]:
        """Every labeled field name, in first-seen order."""
        seen: dict[str, None] = {}
        for page in self.labels.values():
            for name in page:
                seen.setdefault(name, None)
        for name in self.field_kinds:
            seen.setdefault(name, None)
        return list(seen)

    def key_for_url(self, url: str) -> str | None:
        """Page key for an ingested URL (exact match, then without fragment / trailing slash)."""
        key = self.url_to_key.get(url)
        if key is not None:
            return key
        bare = urldefrag(url)[0]
        for candidate in (bare, bare.rstrip("/"), bare + "/"):
            key = self.url_to_key.get(candidate)
            if key is not None:
                return key
        return None


@dataclass(frozen=True)
class Selection:
    """Which sites of a dataset an evaluation config selects."""

    split: Split = "all"
    sites: tuple[str, ...] = ()
    verticals: tuple[str, ...] = ()
    max_sites: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    """Dataset-specific options from the config's ``dataset_options`` (paths, synth params)."""


class DatasetAdapter(Protocol):
    def __call__(self, selection: Selection, work_dir: Path) -> list[EvalSite]: ...


def select_sites(sites: Iterable[EvalSite], selection: Selection) -> list[EvalSite]:
    """Apply the site / vertical / max_sites filters (split is the adapter's job).

    Sites are ordered by (vertical, name) so ``max_sites`` is deterministic.
    """
    chosen = sorted(sites, key=lambda s: (s.vertical, s.name))
    if selection.verticals:
        wanted = set(selection.verticals)
        chosen = [s for s in chosen if s.vertical in wanted]
    if selection.sites:
        wanted = set(selection.sites)
        chosen = [s for s in chosen if s.name in wanted]
    if selection.max_sites is not None:
        chosen = chosen[: selection.max_sites]
    return chosen


def is_empty(value: Any) -> bool:
    """True for a missing label or prediction: ``None``, blank string, empty list or dict."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False
