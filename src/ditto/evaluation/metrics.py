"""Evaluation metrics: value normalization, page-level per-field P/R/F1 and their aggregation.

Every score is computed in three match columns: ``norm`` (normalized match: whitespace, case,
currency symbols, thousands separators, trailing punctuation, plus leading separators such as
``": "``, C1 control characters read as cp1252, and the canonical form of plain numbers), ``raw``
(exact string equality) and ``canon`` (the ``norm`` form plus the formatting rules derived from
the SWDE development-slice misses, see :func:`canonical_text`; on a scalar field a prediction
that joins several labeled values with list separators also counts). Units per field kind:

- ``scalar``: one unit per page. A list label holds acceptable alternatives; a list prediction
  is scored by its first non-empty item (top-1).
- ``list``: one unit per item (multiset match).
- ``record_list``: one unit per row after positional alignment; a row is correct when every
  labeled column matches.
- ``key_value``: one unit per key; correct when the key exists with a matching value.

A null prediction where the label is empty is correct (a true negative, outside P and R); a null
where the label has a value is a miss (FN); a value where the label is empty is a false positive;
a wrong value is both a false positive and a miss. A field missing from a page's label dict is
not annotated for that page and is not scored there.

Aggregation: per field over the pages of a site (micro), then the site's P/R/F1 is the mean over
its fields, the vertical's the mean over its sites and the overall the mean over verticals.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from statistics import fmean
from typing import Any, Literal

from ditto.evaluation.datasets.base import LabelKind, is_empty
from ditto.textutil import normalize_ws

MatchMode = Literal["norm", "raw", "canon"]
MODES: tuple[MatchMode, ...] = ("norm", "raw", "canon")

METRIC_VERSION = 2
"""Version of the match-column definitions, recorded in every evaluation's ``meta.json``.

1: ``norm`` and ``raw``. 2: adds ``canon``, whose rules were derived from SWDE development-slice
misses only (``eval/analysis/swde_dev_misses.md``) and frozen before any test set was rescored;
``norm`` and ``raw`` are unchanged."""

# Keys of an extracted output row that are bookkeeping, not fields.
ROW_META_KEYS = frozenset({"url", "page_id", "template_id", "_meta"})

_CURRENCY = {
    cp: None
    for cp in (*range(0x3000), *range(0xFDF0, 0x10000))
    if unicodedata.category(chr(cp)) == "Sc"
}
_THOUSANDS = re.compile(r"(?<=\d)[,\u00a0\u2009\u202f'](?=\d{3}(?!\d))")
_TRAILING_PUNCT = ".,;:!?\u2026\u00b7-\u2013\u2014\u3002\u3001 "
_LEADING_SEPARATORS = ":;,|\u00b7\u2022\u2013\u2014 "
# C1 controls (U+0080-U+009F) come from numeric references like ``&#150;`` that some decoders
# keep as code points while others read them as cp1252; map them to the cp1252 character.
_C1 = {cp: bytes([cp]).decode("cp1252", errors="ignore") or None for cp in range(0x80, 0xA0)}
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")
_COLKEY = re.compile(r"[^0-9a-z]+")
# canon: separators also stripped at the end (SWDE labels such as "99503 |"), combining
# diacritical marks (U+0300-U+036F after canonical decomposition), digit groups of a long
# identifier ("978-0307269997") and the separators between joined label values.
_CANON_TRAILING = _TRAILING_PUNCT + "|"
_DIACRITICS = re.compile("[\u0300-\u036f]")
_DIGIT_GROUPS = re.compile(r"\d+(?:[- ]\d+)+(?:[- ]?x)?|\d+[- ]x")
_DIGIT_GROUP_SEPARATORS = re.compile(r"[- ]")
_MIN_GROUPED_LENGTH = 10
_JOIN_SEPARATOR = re.compile(r"\s*[,;/|]\s*|\s+")


# -- normalization -------------------------------------------------------------------------------


def to_text(value: Any) -> str | None:
    """Text form of one predicted or labeled value (``None`` for a missing value)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return str(int(value)) if value.is_integer() else repr(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize_text(text: str) -> str:
    """Normalized form used by the ``norm`` match column."""
    text = text.translate(_C1).translate(_CURRENCY)
    text = _THOUSANDS.sub("", text)
    text = normalize_ws(text).casefold()
    text = text.rstrip(_TRAILING_PUNCT).lstrip(_LEADING_SEPARATORS).strip()
    return _number_form(text)


def _number_form(text: str) -> str:
    """Canonical decimal form of a plain number (``"12.50"`` -> ``"12.5"``); other text as is."""
    if _PLAIN_NUMBER.fullmatch(text):
        try:
            number = Decimal(text)
        except InvalidOperation:  # pragma: no cover - the regex admits only valid decimals
            return text
        if number == number.to_integral_value():
            return str(number.to_integral_value())
        return format(number.normalize(), "f")
    return text


def canonical_text(text: str) -> str:
    """Canonical form used by the ``canon`` match column: a function of the ``norm`` form.

    On top of :func:`normalize_text`: combining diacritical marks are dropped (``"café"`` ->
    ``"cafe"``); a trailing ``|`` is stripped like trailing punctuation (``"99503 |"``); a value
    made only of digit groups separated by single hyphens or spaces (an ISBN-10 may end in the
    check character ``x``) and at least ten characters long without them loses the separators
    (``"978-0307269997"`` -> ``"9780307269997"``), so dates and ranges such as ``"10-12"`` are
    untouched. Equal ``norm`` forms always give equal ``canon`` forms.
    """
    text = normalize_text(text)
    text = _DIACRITICS.sub("", unicodedata.normalize("NFD", text))
    text = unicodedata.normalize("NFC", text).rstrip(_CANON_TRAILING)
    if _DIGIT_GROUPS.fullmatch(text):
        joined = _DIGIT_GROUP_SEPARATORS.sub("", text)
        if len(joined) >= _MIN_GROUPED_LENGTH:
            text = joined
    return normalize_text(text)


def canonical(value: Any, mode: MatchMode) -> str | None:
    """Comparable form of a value under ``mode`` (``None`` when the value is empty).

    Under ``norm`` and ``canon`` a value that normalizes to nothing (e.g. a lone dash) counts as
    empty.
    """
    text = to_text(value)
    if text is None or not text.strip():
        return None
    if mode == "raw":
        return text
    if mode == "canon":
        return canonical_text(text) or None
    return normalize_text(text) or None


def values_match(pred: Any, label: Any, mode: MatchMode) -> bool:
    a, b = canonical(pred, mode), canonical(label, mode)
    return a is not None and a == b


# -- counts --------------------------------------------------------------------------------------


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> float | None:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else None

    def metrics(self) -> dict[str, float | None]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1}

    def to_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn, **self.metrics()}


def _first_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return next((v for v in value if not is_empty(v)), None)
    return value


def _items(value: Any) -> list[Any]:
    if is_empty(value):
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if not is_empty(v)]
    return [value]


def _colkey(name: str) -> str:
    return _COLKEY.sub("_", name.casefold()).strip("_")


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for row in value:
        if isinstance(row, Mapping) and any(not is_empty(v) for v in row.values()):
            rows.append({_colkey(str(k)): v for k, v in row.items()})
    return rows


def _pairs(value: Any) -> dict[str, Any]:
    """Key-value field as a dict; accepts a dict, ``[[k, v], ...]`` or ``[{key, value}, ...]``."""
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items() if not is_empty(v)}
    out: dict[str, Any] = {}
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Mapping) and "key" in item:
                k, v = item.get("key"), item.get("value")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
            else:
                continue
            if k is not None and not is_empty(v):
                out[str(k)] = v
    return out


def joins_alternatives(text: str, alternatives: Iterable[str]) -> bool:
    """``text`` is two or more distinct ``alternatives`` joined by list separators.

    Separators: ``,`` ``;`` ``/`` ``|`` (with optional spaces) or whitespace. Every piece must be
    a whole alternative, so a value with any other content never qualifies (``"comedy western"``
    joins ``comedy`` and ``western``; ``"comedy western 1969"`` does not).
    """
    options = sorted(set(alternatives), key=len, reverse=True)
    if len(options) < 2:
        return False

    def joined(rest: str, used: frozenset[str]) -> bool:
        for option in options:
            if option in used or not rest.startswith(option):
                continue
            tail = rest[len(option) :]
            if not tail:
                if used:
                    return True
                continue
            sep = _JOIN_SEPARATOR.match(tail)
            if sep and sep.end() < len(tail) and joined(tail[sep.end() :], used | {option}):
                return True
        return False

    return joined(text, frozenset())


def _score_scalar(pred: Any, label: Any, mode: MatchMode) -> Counts:
    alternatives = [a for a in (canonical(v, mode) for v in _items(label)) if a is not None]
    guess = canonical(_first_item(pred), mode)
    if guess is None:
        return Counts(fn=1) if alternatives else Counts(tn=1)
    if not alternatives:
        return Counts(fp=1)
    if guess in alternatives or (mode == "canon" and joins_alternatives(guess, alternatives)):
        return Counts(tp=1)
    return Counts(fp=1, fn=1)


def _score_list(pred: Any, label: Any, mode: MatchMode) -> Counts:
    got = Counter(c for c in (canonical(v, mode) for v in _items(pred)) if c is not None)
    want = Counter(c for c in (canonical(v, mode) for v in _items(label)) if c is not None)
    if not got and not want:
        return Counts(tn=1)
    tp = sum((got & want).values())
    return Counts(tp=tp, fp=sum(got.values()) - tp, fn=sum(want.values()) - tp)


def _row_matches(
    pred_row: Mapping[str, Any], label_row: Mapping[str, Any], mode: MatchMode
) -> bool:
    for col, want in label_row.items():
        got = pred_row.get(col)
        if is_empty(want):
            if not is_empty(got):
                return False
        elif not values_match(got, want, mode):
            return False
    return True


def _score_record_list(pred: Any, label: Any, mode: MatchMode) -> Counts:
    got, want = _rows(pred), _rows(label)
    if not got and not want:
        return Counts(tn=1)
    tp = sum(1 for g, w in zip(got, want, strict=False) if _row_matches(g, w, mode))
    return Counts(tp=tp, fp=len(got) - tp, fn=len(want) - tp)


def _score_key_value(pred: Any, label: Any, mode: MatchMode) -> Counts:
    got: dict[str, Any] = {}
    for k, v in _pairs(pred).items():
        ck = canonical(k, mode)
        if ck is not None:
            got.setdefault(ck, v)
    want: dict[str, Any] = {}
    for k, v in _pairs(label).items():
        ck = canonical(k, mode)
        if ck is not None:
            want.setdefault(ck, v)
    if not got and not want:
        return Counts(tn=1)
    tp = sum(1 for k, v in want.items() if k in got and values_match(got[k], v, mode))
    return Counts(tp=tp, fp=len(got) - tp, fn=len(want) - tp)


_SCORERS = {
    "scalar": _score_scalar,
    "list": _score_list,
    "record_list": _score_record_list,
    "key_value": _score_key_value,
}


def score_value(kind: LabelKind, pred: Any, label: Any, mode: MatchMode) -> Counts:
    """Score one field on one page."""
    return _SCORERS[kind](pred, label, mode)


# -- site scores ---------------------------------------------------------------------------------


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return fmean(kept) if kept else None


def _macro(per_field: Iterable[Counts]) -> dict[str, float | None]:
    counts = list(per_field)
    return {
        "precision": _mean(c.precision for c in counts),
        "recall": _mean(c.recall for c in counts),
        "f1": _mean(c.f1 for c in counts),
    }


@dataclass
class SiteScore:
    """Per-field counts of one site, in both match modes."""

    site: str
    vertical: str
    fields: dict[str, dict[MatchMode, Counts]]
    pages_scored: int = 0
    pages_without_prediction: int = 0
    pages_excluded: int = 0
    """Labeled pages left out because the LLM saw them while generating a spec."""

    def field_metrics(self, mode: MatchMode) -> dict[str, dict[str, float | None]]:
        return {name: modes[mode].metrics() for name, modes in self.fields.items()}

    def metrics(self, mode: MatchMode) -> dict[str, float | None]:
        """Mean over fields of the per-field precision, recall and F1."""
        return _macro(modes[mode] for modes in self.fields.values())

    def group_metrics(
        self, groups: Mapping[str, Sequence[str]], mode: MatchMode
    ) -> dict[str, dict[str, float | None]]:
        """Mean field metrics per named field group (e.g. LiveWeb-IE complexity types)."""
        return {
            group: _macro(self.fields[f][mode] for f in names if f in self.fields)
            for group, names in groups.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "vertical": self.vertical,
            "pages_scored": self.pages_scored,
            "pages_without_prediction": self.pages_without_prediction,
            "pages_excluded": self.pages_excluded,
            **{mode: self.metrics(mode) for mode in MODES},
            "fields": {
                name: {mode: counts.to_dict() for mode, counts in modes.items()}
                for name, modes in self.fields.items()
            },
        }


def score_site(
    site: str,
    vertical: str,
    predictions: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    kinds: Mapping[str, LabelKind] | None = None,
    *,
    fields: Sequence[str] | None = None,
    exclude: Iterable[str] = (),
) -> SiteScore:
    """Score every labeled page of one site.

    ``predictions`` and ``labels`` map page keys to ``{field: value}``; a labeled page without a
    prediction scores as all-null. ``fields`` defaults to every labeled field; a field absent from
    a page's label dict is skipped on that page. ``exclude`` lists page keys that are not scored
    (pages the LLM saw at generation time).
    """
    kinds = kinds or {}
    if fields is None:
        seen: dict[str, None] = {}
        for row in labels.values():
            for name in row:
                seen.setdefault(name, None)
        fields = list(seen)
    excluded = set(exclude)
    counts: dict[str, dict[MatchMode, Counts]] = {f: {m: Counts() for m in MODES} for f in fields}
    score = SiteScore(site=site, vertical=vertical, fields=counts)
    empty: Mapping[str, Any] = {}
    for key, label_row in labels.items():
        if key in excluded:
            score.pages_excluded += 1
            continue
        pred_row = predictions.get(key)
        if pred_row is None:
            score.pages_without_prediction += 1
            pred_row = empty
        score.pages_scored += 1
        for name in fields:
            if name not in label_row:
                continue
            kind = kinds.get(name, "scalar")
            for mode in MODES:
                counts[name][mode].add(
                    score_value(kind, pred_row.get(name), label_row.get(name), mode)
                )
    return score


def aggregate(scores: Sequence[SiteScore]) -> dict[str, Any]:
    """Average per site, then per vertical (mean of its sites), then overall (mean of verticals).

    Also reports the per-vertical mean F1 of every field across the vertical's sites.
    """
    by_vertical: dict[str, list[SiteScore]] = defaultdict(list)
    for s in scores:
        by_vertical[s.vertical].append(s)
    per_vertical: dict[str, Any] = {}
    for vertical in sorted(by_vertical):
        sites = by_vertical[vertical]
        entry: dict[str, Any] = {"n_sites": len(sites)}
        for mode in MODES:
            site_metrics = [s.metrics(mode) for s in sites]
            entry[mode] = {
                k: _mean(m[k] for m in site_metrics) for k in ("precision", "recall", "f1")
            }
            names = sorted({f for s in sites for f in s.fields})
            entry[f"fields_{mode}_f1"] = {
                f: _mean(s.fields[f][mode].f1 for s in sites if f in s.fields) for f in names
            }
        per_vertical[vertical] = entry
    overall: dict[str, Any] = {"n_sites": len(scores), "n_verticals": len(per_vertical)}
    for mode in MODES:
        overall[mode] = {
            k: _mean(v[mode][k] for v in per_vertical.values())
            for k in ("precision", "recall", "f1")
        }
    return {"per_vertical": per_vertical, "overall": overall}


# -- optional-field recall -----------------------------------------------------------------------


@dataclass
class OptionalRecall:
    """Recall on (page, field) pairs whose field had no value on any page the LLM saw."""

    fields: dict[str, dict[MatchMode, Counts]] = field(default_factory=dict)
    pages: int = 0

    def total(self, mode: MatchMode) -> Counts:
        out = Counts()
        for modes in self.fields.values():
            out.add(modes[mode])
        return out

    def add(self, other: OptionalRecall) -> None:
        self.pages += other.pages
        for name, modes in other.fields.items():
            mine = self.fields.setdefault(name, {m: Counts() for m in MODES})
            for mode in MODES:
                mine[mode].add(modes[mode])

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            **{
                mode: {"recall": self.total(mode).recall, **_tp_fn(self.total(mode))}
                for mode in MODES
            },
            "fields": {
                name: {mode: {"recall": c.recall, **_tp_fn(c)} for mode, c in modes.items()}
                for name, modes in self.fields.items()
            },
        }


def _tp_fn(c: Counts) -> dict[str, int]:
    return {"tp": c.tp, "fn": c.fn}


def optional_field_recall(
    predictions: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    page_template: Mapping[str, str],
    seen_by_template: Mapping[str, Iterable[str]],
    kinds: Mapping[str, LabelKind] | None = None,
) -> OptionalRecall:
    """Recall where the attribute exists on a page but on none of its template's generating pages.

    ``page_template`` maps page key -> template id the page was extracted with;
    ``seen_by_template`` maps template id -> page keys the spec was generated from (spec
    ``generated_by.pages_seen``). Templates without known generating pages are skipped, as are
    the generating pages themselves.
    """
    kinds = kinds or {}
    seen = {tid: set(keys) for tid, keys in seen_by_template.items()}
    seen = {tid: keys for tid, keys in seen.items() if keys}
    all_seen: set[str] = set().union(*seen.values()) if seen else set()
    present_on_seen: dict[str, set[str]] = {}
    for tid, keys in seen.items():
        present_on_seen[tid] = {
            name for k in keys for name, value in labels.get(k, {}).items() if not is_empty(value)
        }
    result = OptionalRecall()
    for key, label_row in labels.items():
        template = page_template.get(key)
        if template is None or template not in seen or key in all_seen:
            continue
        pred_row = predictions.get(key, {})
        counted = False
        for name, label in label_row.items():
            if is_empty(label) or name in present_on_seen[template]:
                continue
            counted = True
            modes = result.fields.setdefault(name, {m: Counts() for m in MODES})
            for mode in MODES:
                c = score_value(kinds.get(name, "scalar"), pred_row.get(name), label, mode)
                modes[mode].add(Counts(tp=c.tp, fn=c.fn))
        result.pages += counted
    return result


# -- unprompted name alignment -------------------------------------------------------------------


def _flat_values(value: Any) -> set[str]:
    """Normalized leaf values of any field shape (scalar, list, rows, key-value)."""
    out: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            text = canonical(item, "norm")
            if text is not None:
                out.add(text)
    return out


def _structured(value: Any) -> bool:
    """A key-value mapping or a list of records (rows)."""
    if isinstance(value, Mapping):
        return True
    return isinstance(value, (list, tuple)) and any(isinstance(v, Mapping) for v in value)


def field_parts(name: str, value: Any) -> dict[str, Any]:
    """A predicted field and, for a structured value, its addressable parts.

    An unprompted spec may group what the labels keep apart: a specification sheet becomes one
    ``key_value`` field, a listing table one ``record_list``. Each key of a mapping is a part
    ``name[Key]`` (its value); each column of a record list is a part ``name.column`` (the
    column's values in row order). Plain values have no parts.
    """
    parts: dict[str, Any] = {name: value}
    if isinstance(value, Mapping):
        for key, v in _pairs(value).items():
            parts[f"{name}[{normalize_ws(key)}]"] = v
    elif isinstance(value, (list, tuple)):
        columns: dict[str, list[Any]] = {}
        for row in value:
            if isinstance(row, Mapping):
                for col, v in row.items():
                    if not is_empty(v):
                        columns.setdefault(str(col), []).append(v)
        for col, values in columns.items():
            parts[f"{name}.{col}"] = values
    return parts


def _part_base(part: str) -> str:
    """The predicted field a part belongs to (the part itself for a whole field)."""
    cut = min((i for i in (part.find("["), part.find(".")) if i > 0), default=len(part))
    return part[:cut]


def part_value(row: Mapping[str, Any], part: str) -> Any:
    """Value of a whole predicted field or of one of its parts (see :func:`field_parts`)."""
    if part in row:
        return row[part]
    base = _part_base(part)
    if base == part or base not in row:
        return None
    return field_parts(base, row[base]).get(part)


def align_field_names(
    predictions: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    keys: Iterable[str],
    *,
    min_fraction: float = 0.1,
) -> dict[str, str]:
    """Map label field names to predicted (LLM-named) fields by value overlap on ``keys``.

    Candidates are the predicted fields and their parts (a key of a key-value field, a column
    of a record list; :func:`field_parts`). A plain label (scalar or list) can only align to a
    plain field or a part, a structured label (key-value or records) only to a whole structured
    field: anything else could never score. For every compatible (label field, candidate) pair,
    count the pages where their normalized values share at least one value. Pairs are assigned
    one-to-one greedily by descending overlap; a whole field and its parts exclude each other,
    sibling parts do not. A pair needs an overlap on at least ``min_fraction`` of the pages that
    carry the label (and at least one page). Returns ``{label_field: predicted field or part}``;
    :func:`rename_fields` resolves parts.
    """
    overlap: Counter[tuple[str, str]] = Counter()
    labeled: Counter[str] = Counter()
    structured_labels: set[str] = set()
    structured_fields: set[str] = set()  # whole fields with a mapping or record value
    for key in keys:
        label_row = labels.get(key)
        pred_row = predictions.get(key)
        if not label_row:
            continue
        pred_values: dict[str, set[str]] = {}
        for name, whole in (pred_row or {}).items():
            if name in ROW_META_KEYS:
                continue
            if _structured(whole):
                structured_fields.add(name)
            for part, value in field_parts(name, whole).items():
                pred_values[part] = _flat_values(value)
        for lf, label in label_row.items():
            want = _flat_values(label)
            if not want:
                continue
            labeled[lf] += 1
            if _structured(label):
                structured_labels.add(lf)
            for pf, got in pred_values.items():
                if got & want:
                    overlap[(lf, pf)] += 1

    def compatible(lf: str, pf: str) -> bool:
        return (pf in structured_fields) == (lf in structured_labels)

    alignment: dict[str, str] = {}
    used: set[str] = set()
    whole_used: set[str] = set()  # fields aligned whole: none of their parts is available
    part_used: set[str] = set()  # fields with an aligned part: the whole is not available
    for (lf, pf), n in sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0])):
        base = _part_base(pf)
        if lf in alignment or pf in used or base in whole_used or not compatible(lf, pf):
            continue
        if base == pf and base in part_used:
            continue
        if n < max(1.0, min_fraction * labeled[lf]):
            continue
        alignment[lf] = pf
        used.add(pf)
        (whole_used if base == pf else part_used).add(base)
    return alignment


def rename_fields(row: Mapping[str, Any], alignment: Mapping[str, str]) -> dict[str, Any]:
    """Predicted row with LLM field names (or field parts) replaced by the aligned label names."""
    out: dict[str, Any] = {}
    for lf, pf in alignment.items():
        value = part_value(row, pf)
        if value is not None:
            out[lf] = value
    return out


def sample_keys(keys: Iterable[str], n: int, seed: int) -> list[str]:
    """Deterministic sample of ``n`` page keys (all keys when there are fewer)."""
    ordered = sorted(keys)
    if len(ordered) <= n:
        return ordered
    return sorted(random.Random(seed).sample(ordered, n))


# -- unprompted schema precision (judged) -------------------------------------------------------

VERDICTS = ("entity", "noise")


def schema_precision(
    judgments: Sequence[Mapping[str, Any]],
    proposed: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
) -> dict[str, Any]:
    """Schema precision from per-field verdicts: entity fields / all judged fields.

    ``judgments`` are ``{site, template, field, verdict, ...}`` rows (verdict ``entity`` or
    ``noise``). Precision is reported per site and overall (micro: over all judged fields), plus
    the mean of the per-site values. With ``proposed`` (``{site: {template: field names}}`` of a
    run), the result also says whether the judgments cover exactly that run's proposed fields on
    the judged sites: ``unjudged`` lists proposed fields without a verdict, ``stale`` verdicts for
    fields the run did not propose, ``sites_missing`` judged sites the run lacks.
    """
    per_site: dict[str, dict[str, Any]] = {}
    judged: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in judgments:
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r} in {row}")
        site, key = str(row["site"]), (str(row["template"]), str(row["field"]))
        if key in judged[site]:
            raise ValueError(f"duplicate verdict for {site} {key[0]}.{key[1]}")
        judged[site].add(key)
        entry = per_site.setdefault(site, {"fields": 0, "entity": 0})
        entry["fields"] += 1
        entry["entity"] += verdict == "entity"
    for entry in per_site.values():
        entry["precision"] = entry["entity"] / entry["fields"]
    n_fields = sum(e["fields"] for e in per_site.values())
    n_entity = sum(e["entity"] for e in per_site.values())
    out: dict[str, Any] = {
        "n_sites": len(per_site),
        "fields": n_fields,
        "entity": n_entity,
        "precision": n_entity / n_fields if n_fields else None,
        "precision_site_mean": _mean(e["precision"] for e in per_site.values()),
        "per_site": dict(sorted(per_site.items())),
    }
    if proposed is not None:
        unjudged: list[str] = []
        stale: list[str] = []
        for site in sorted(judged):
            if site not in proposed:
                continue
            run_fields = {(t, f) for t, names in proposed[site].items() for f in names}
            unjudged.extend(f"{site} {t}.{f}" for t, f in sorted(run_fields - judged[site]))
            stale.extend(f"{site} {t}.{f}" for t, f in sorted(judged[site] - run_fields))
        missing = sorted(set(judged) - set(proposed))
        out["run_check"] = {
            "matches_run": not (unjudged or stale or missing),
            "unjudged": unjudged,
            "stale": stale,
            "sites_missing": missing,
        }
    return out
