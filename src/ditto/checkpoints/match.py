"""Rule-based matching of requested fields to analysis candidates (probe, no LLM).

Every candidate of a template is scored against every requested field. The score multiplies
one *evidence* term (how strongly something names the candidate as this field) by factors that
can only lower it:

- evidence: the best of label similarity (``label_hint``, table column/key, path identifiers,
  tag hints; normalized string similarity plus a small synonym list), description overlap with
  those labels, identity with a user-supplied example, and a weak type-only signal for very
  specific types (email, price, date);
- type factor: share of the candidate's example values passing the expected type check;
- example factor: shape agreement between the user's examples and the candidate's;
- shape factor: scalar vs list vs record_list vs key_value;
- a small presence tie-breaker.

Labels, boilerplate and noise are never fields; neither are key/header cells of tables.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from ditto.analyze.models import Candidate, CandidateTable, Table
from ditto.fingerprint.dom import Step, split_path
from ditto.schema import ABSENT, FieldRequest
from ditto.specgen.typecheck import check_value, infer_value_type
from ditto.textutil import normalize_ws

FOUND = 0.6  # top score at or above this, with a clear margin, is "found"
AMBIGUOUS = 0.35  # top score at or above this (but not found) is "ambiguous"
MARGIN = 0.15  # found needs the runner-up this far below (or under AMBIGUOUS)
MIN_LISTED = 0.2  # candidates under this score are not listed
TOP_K = 3

FieldStatus = Literal["found", "ambiguous", "missing"]

SYNONYM_DISCOUNT = 0.92
SPECIFIC_TYPES = ("email", "price", "date")

# Small synonym list: phrases in one group name the same thing.
SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("price", "cost", "amount", "fee", "msrp", "list price", "sale price"),
    ("salary", "pay", "compensation", "wage", "pay range", "base salary"),
    (
        "company",
        "employer",
        "hiring organization",
        "organization",
        "hiring company",
        "firm",
        "business",
    ),
    ("location", "address", "city", "place", "locality", "region", "job location", "where"),
    ("title", "name", "heading", "headline"),
    (
        "date posted",
        "posted",
        "posted on",
        "published",
        "published on",
        "publication date",
        "date published",
        "posting date",
        "posted date",
    ),
    ("description", "summary", "overview", "about", "details", "body"),
    ("author", "by", "written by", "writer", "byline"),
    ("email", "e mail", "mail", "contact email"),
    ("phone", "telephone", "tel", "phone number", "mobile"),
    ("url", "link", "website", "homepage", "web site", "href"),
    ("image", "photo", "picture", "img", "thumbnail", "logo"),
    ("rating", "stars", "score", "review score"),
    ("category", "genre", "section", "department"),
    ("tags", "keywords", "labels", "topics"),
    ("employment type", "job type", "contract type", "type of employment"),
)

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "for",
        "in",
        "on",
        "at",
        "to",
        "and",
        "or",
        "is",
        "are",
        "was",
        "its",
        "this",
        "that",
        "with",
        "from",
        "as",
        "be",
        "it",
    ]
)
# Tokens that say little about *which* field this is ("company name" is about the company).
GENERIC = frozenset(["name", "value", "text", "info", "information", "field", "data", "label"])

# Field-name keywords implying a value type (softer than an explicit type_check).
_NAME_TYPES: tuple[tuple[str, frozenset[str]], ...] = (
    ("email", frozenset({"email", "mail"})),
    ("url", frozenset({"url", "link", "website", "homepage", "href", "image", "photo", "logo"})),
    ("date", frozenset({"date", "posted", "published", "updated", "deadline"})),
    ("price", frozenset({"price", "cost", "fee", "msrp"})),
)

# Tag -> hint words and weights (an h1 is usually a detail page's title; <title> often carries
# the site name too, so it ranks below the h1).
_TAG_HINTS: dict[str, tuple[tuple[str, float], ...]] = {
    "h1": (("title", 0.75), ("heading", 0.75)),
    "h2": (("heading", 0.55),),
    "h3": (("heading", 0.5),),
    "title": (("title", 0.55),),
    "time": (("date", 0.7),),
    "address": (("address", 0.7),),
    "img": (("image", 0.7),),
}
_ATTR_HINTS: dict[str, tuple[tuple[str, float], ...]] = {
    "href": (("link", 0.6), ("url", 0.6)),
    "src": (("image", 0.65),),
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


@lru_cache(maxsize=1 << 15)
def tokens(text: str | None) -> tuple[str, ...]:
    """Lowercase, split camelCase / snake_case / punctuation, drop stopwords, crude stemming."""
    if not text:
        return ()
    low = _CAMEL.sub(" ", text).lower()
    out: list[str] = []
    for tok in _NON_ALNUM.split(low):
        if not tok or tok in STOPWORDS:
            continue
        out.append(_stem(tok))
    return tuple(out)


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def _core(toks: tuple[str, ...]) -> tuple[str, ...]:
    core = tuple(t for t in toks if t not in GENERIC)
    return core or toks


_SYN_GROUPS: tuple[tuple[tuple[str, ...], ...], ...] = tuple(
    tuple(tokens(p) for p in group) for group in SYNONYMS
)


@lru_cache(maxsize=1 << 16)
def phrase_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Similarity of two token phrases in [0, 1]: exact, token overlap, or fuzzy characters."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ca, cb = "".join(a), "".join(b)
    if ca == cb:
        return 1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    token_score = 0.0
    if inter:
        cover_a = inter / len(sa)
        cover_b = inter / len(sb)
        dice = 2 * inter / (len(sa) + len(sb))
        token_score = max(dice, 0.6 * cover_a + 0.4 * cover_b)
    if token_score >= 0.9:  # fuzzy characters cannot beat it (they score at most 0.9)
        return token_score
    # ratio <= 2*min/(len_a+len_b): skip the matcher when it cannot reach the 0.85 bar, then
    # use its cheap upper bound before the full ratio
    if 2 * min(len(ca), len(cb)) < 0.85 * (len(ca) + len(cb)):
        return token_score
    matcher = difflib.SequenceMatcher(None, ca, cb, autojunk=False)
    if matcher.quick_ratio() < 0.85:
        return token_score
    ratio = matcher.ratio()
    return max(token_score, ratio * 0.9 if ratio >= 0.85 else 0.0)


def name_variants(name: str) -> list[tuple[tuple[str, ...], float]]:
    """The field's own phrase (weight 1) plus synonym substitutions (discounted)."""
    base = _core(tokens(name))
    if not base:
        return []
    out: dict[tuple[str, ...], float] = {base: 1.0}
    for group in _SYN_GROUPS:
        for member in group:
            n = len(member)
            if not n:
                continue
            for i in range(len(base) - n + 1):
                if base[i : i + n] != member:
                    continue
                for other in group:
                    if other == member or not other:
                        continue
                    variant = base[:i] + other + base[i + n :]
                    out.setdefault(variant, SYNONYM_DISCOUNT)
    return list(out.items())


@dataclass(frozen=True)
class LabelSource:
    """One piece of text that may name a candidate: label hint, column, key, path identifier."""

    toks: tuple[str, ...]
    weight: float
    kind: str  # label | column | key | itemprop | path | parent | tag
    text: str
    human: bool  # written for readers (label/column/key), so a description may mention it
    tokset: frozenset[str] = field(init=False, repr=False)
    content: frozenset[str] = field(init=False, repr=False)  # tokens a description may share

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokset", frozenset(self.toks))
        content = frozenset(t for t in self.toks if len(t) >= 3 and t not in GENERIC)
        object.__setattr__(self, "content", content)


def _step_identifiers(step: Step) -> list[tuple[str, str]]:
    """``(kind, text)`` identifiers of a structural step: itemprop, id, testid, classes."""
    out: list[tuple[str, str]] = []
    for name, value in step.attrs:
        if name == "itemprop":
            out.append(("itemprop", value))
        elif name == "data-testid":
            out.append(("path", value))
    if step.id:
        out.append(("path", step.id))
    out.extend(("path", c) for c in step.classes)
    return out


def label_sources(cand: Candidate, table: Table | None) -> list[LabelSource]:
    srcs: list[LabelSource] = []

    def add(text: str | None, weight: float, kind: str, human: bool) -> None:
        toks = _core(tokens(text))
        if toks:
            srcs.append(LabelSource(toks, weight, kind, normalize_ws(text or ""), human))

    add(cand.label_hint, 1.0, "label", True)
    if cand.column:
        add(cand.column, 1.0, "column" if cand.table_role == "record_column" else "key", True)
    if table is not None and table.kind == "key_value" and cand.table_role == "kv_value":
        for key in table.keys_seen:
            add(key, 0.9, "key", True)
    try:
        steps, attr = split_path(cand.path)
    except ValueError:
        steps, attr = [], cand.attr
    attr = attr or cand.attr
    if steps:
        last = steps[-1]
        for kind, text in _step_identifiers(last):
            add(text, 0.9 if kind == "itemprop" else 0.8, kind, False)
        if len(steps) > 1:
            for kind, text in _step_identifiers(steps[-2]):
                add(text, 0.75 if kind == "itemprop" else 0.6, "parent", False)
        if last.tag in _TAG_HINTS and attr is None:
            for words, weight in _TAG_HINTS[last.tag]:
                add(words, weight, "tag", False)
    if attr in _ATTR_HINTS:
        for words, weight in _ATTR_HINTS[attr]:
            add(words, weight, "tag", False)
    return srcs


def table_label_sources(table: Table, by_id: dict[str, Candidate]) -> list[LabelSource]:
    """Label sources of a whole table (container identifiers, column names / keys, heading)."""
    srcs: list[LabelSource] = []

    def add(text: str | None, weight: float, kind: str, human: bool) -> None:
        toks = _core(tokens(text))
        if toks:
            srcs.append(LabelSource(toks, weight, kind, normalize_ws(text or ""), human))

    try:
        steps, _ = split_path(table.container_path)
    except ValueError:
        steps = []
    if steps:
        for kind, text in _step_identifiers(steps[-1]):
            add(text, 0.85, kind, False)
    texts = [c.name for c in table.columns] if table.kind == "record_list" else table.keys_seen
    add(" ".join(texts), 0.5, "columns", True)
    for cid in (table.key_candidate, *(c.candidate_id for c in table.columns)):
        cand = by_id.get(cid) if cid else None
        if cand is not None and cand.table_role != "record_column":
            add(cand.label_hint, 0.9, "label", True)
    return srcs


def expected_type(req: FieldRequest) -> tuple[str | None, bool]:
    """Expected value type and whether it is firm (explicit / from examples) or name-inferred."""
    if req.type_check and req.type_check != "text":
        return req.type_check, True
    if req.examples:
        inferred = infer_value_type(req.examples)
        if inferred and inferred != "text":
            return inferred, True
    if req.type_check == "text":
        return None, True
    name_toks = set(tokens(req.name))
    for kind, words in _NAME_TYPES:
        if name_toks & words:
            return kind, False
    return None, False


def _norm_value(v: str) -> str:
    return normalize_ws(v).casefold()


_SHAPE_SUBS = ((re.compile(r"\d+"), "9"), (re.compile(r"[^\W\d_]+"), "a"))


def value_shape(v: str) -> str:
    s = normalize_ws(v)
    for pat, rep in _SHAPE_SUBS:
        s = pat.sub(rep, s)
    return s


def example_agreement(user: Sequence[str], cand: Sequence[str]) -> tuple[float, float]:
    """``(identity, shape)``: identity 1.0 for an equal value, 0.8 for containment; shape is the
    best character-class similarity between any user example and any candidate example."""
    identity = 0.0
    shape = 0.0
    for u in user:
        nu = _norm_value(u)
        su = value_shape(u)
        for c in cand:
            nc = _norm_value(c)
            if not nu or not nc:
                continue
            if nu == nc:
                identity = 1.0
            elif min(len(nu), len(nc)) >= 3 and (nu in nc or nc in nu):
                identity = max(identity, 0.8)
            ratio = difflib.SequenceMatcher(None, su, value_shape(c), autojunk=False).ratio()
            shape = max(shape, ratio)
    return identity, shape


@dataclass
class Scored:
    """A scored candidate (or table) for one field."""

    id: str
    score: float
    reasons: list[str]
    examples: list[str]
    use: str | list[str]
    candidate: Candidate | None = None
    table: Table | None = None
    key: str | None = None  # key of a key_value table the field matched

    @property
    def example_key(self) -> tuple[str, ...]:
        return tuple(sorted(_norm_value(e) for e in self.examples))


@dataclass
class FieldMatch:
    name: str
    status: FieldStatus
    use: str | list[str]
    ranked: list[Scored] = field(default_factory=list)

    @property
    def top(self) -> Scored | None:
        return self.ranked[0] if self.ranked else None


@dataclass(frozen=True)
class FieldContext:
    """Everything about one requested field that scoring needs, computed once per field."""

    req: FieldRequest
    variants: tuple[tuple[tuple[str, ...], frozenset[str], float], ...]
    desc: frozenset[str]  # content tokens of the description
    kind: str | None  # expected value type
    firm: bool  # kind comes from type_check / examples, not just the name
    # (source tokens, human) -> (score before the source weight, variant used); label texts and
    # path identifiers repeat across candidates, so each is compared with the variants once
    _seen: dict[tuple[tuple[str, ...], bool], tuple[float, tuple[str, ...] | None, float]] = field(
        default_factory=dict, compare=False, repr=False
    )

    @classmethod
    def of(cls, req: FieldRequest) -> FieldContext:
        kind, firm = expected_type(req)
        return cls(
            req=req,
            variants=tuple((v, frozenset(v), w) for v, w in name_variants(req.name)),
            desc=frozenset(t for t in tokens(req.description) if len(t) >= 3 and t not in GENERIC),
            kind=kind,
            firm=firm,
        )

    def source_match(self, src: LabelSource) -> tuple[float, tuple[str, ...] | None, float]:
        """Best ``(score, variant, variant weight)`` of the field's variants against ``src``
        (without the source's own weight)."""
        key = (src.toks, src.human)
        hit = self._seen.get(key)
        if hit is not None:
            return hit
        best: tuple[float, tuple[str, ...] | None, float] = (0.0, None, 1.0)
        for toks, vset, vw in self.variants:
            s = phrase_similarity(toks, src.toks)
            if not s:
                continue
            covered = s >= 0.85 or vset <= src.tokset
            # a synonym counts only when the label holds all of it ("job location" must not
            # match a bare "job"); a partial match on an identifier (class "job" for field
            # "job type") is weak
            if vw < 1.0 and not covered:
                continue
            if not src.human and not covered:
                s *= 0.5
            s *= vw
            if s > best[0]:
                best = (s, toks, vw)
        self._seen[key] = best
        return best


@dataclass
class Prepared:
    """Per-template precomputation shared by all fields."""

    table: CandidateTable
    usable: list[Candidate]
    sources: dict[str, list[LabelSource]]
    tables: dict[str, Table]
    table_sources: dict[str, list[LabelSource]]
    by_id: dict[str, Candidate]
    type_pass: dict[tuple[str, str], float] = field(default_factory=dict)

    def type_fraction(self, cand: Candidate, kind: str) -> float:
        """Share of the candidate's examples passing ``kind`` (cached per candidate and kind)."""
        key = (cand.id, kind)
        frac = self.type_pass.get(key)
        if frac is None:
            if cand.examples:
                frac = sum(check_value(kind, e) for e in cand.examples) / len(cand.examples)
            elif cand.value_type is not None:
                frac = 1.0 if cand.value_type == kind else 0.0
            else:
                frac = 0.5
            self.type_pass[key] = frac
        return frac


def usable_candidates(table: CandidateTable) -> list[Candidate]:
    """Candidates that may hold a field: never labels, boilerplate, noise, keys or headers."""
    return [
        c
        for c in table.candidates
        if c.is_field and c.table_role not in ("kv_key", "record_header")
    ]


def prepare(table: CandidateTable) -> Prepared:
    tables = {t.id: t for t in table.tables}
    usable = usable_candidates(table)
    by_id = {c.id: c for c in table.candidates}
    return Prepared(
        table=table,
        usable=usable,
        sources={c.id: label_sources(c, tables.get(c.table_id or "")) for c in usable},
        tables=tables,
        table_sources={t.id: table_label_sources(t, by_id) for t in table.tables},
        by_id=by_id,
    )


def _label_evidence(
    ctx: FieldContext, srcs: Iterable[LabelSource]
) -> tuple[float, str | None, LabelSource | None]:
    best, best_src = 0.0, None
    best_var: tuple[str, ...] | None = None
    best_vw = 1.0
    for src in srcs:
        s, var, vw = ctx.source_match(src)
        s *= src.weight
        if s > best:
            best, best_src, best_var, best_vw = s, src, var, vw
    if best_src is None or best < MIN_LISTED:
        return best, None, best_src
    syn = "" if best_vw == 1.0 or best_var is None else f" (synonym {' '.join(best_var)})"
    return best, f"{best_src.kind} '{best_src.text}'{syn}", best_src


def _desc_evidence(ctx: FieldContext, srcs: Iterable[LabelSource]) -> tuple[float, str | None]:
    if not ctx.desc:
        return 0.0, None
    best, best_src = 0.0, None
    for src in srcs:
        # identifiers like a body class say little next to a sentence: readable labels only
        if not src.human or not src.content:
            continue
        s = 0.6 * len(ctx.desc & src.content) / len(src.content) * src.weight
        if s > best:
            best, best_src = s, src
    if best_src is None or best < MIN_LISTED:
        return best, None
    return best, f"description mentions {best_src.kind} '{best_src.text}'"


def _shape_factor(req: FieldRequest, cand: Candidate | None, table: Table | None) -> float:
    want = req.type
    if want is None:
        return 1.0
    if cand is None:  # whole table
        assert table is not None
        if want == "record_list":
            return 1.0 if table.kind == "record_list" else 0.7
        if want == "key_value":
            return 1.0 if table.kind == "key_value" else 0.7
        return 0.6
    repeated = cand.cls == "list" or cand.repeat_max > 1
    if want == "scalar":
        return 0.75 if repeated else 1.0
    if want == "list":
        return 1.0 if repeated else 0.75
    # record_list / key_value requested: a single column/value candidate is a weak stand-in
    return 0.7 if cand.table_id else 0.5


def _type_terms(
    ctx: FieldContext, cand: Candidate, prep: Prepared
) -> tuple[float, float, str | None]:
    """``(factor, evidence, reason)`` from the expected value type."""
    kind = ctx.kind
    if kind is None:
        if cand.value_type in ("url", "email") or cand.attr is not None:
            return 0.7, 0.0, f"value type {cand.value_type or cand.attr} for a text field"
        return 1.0, 0.0, None
    frac = prep.type_fraction(cand, kind)
    factor = (0.35 + 0.65 * frac) if ctx.firm else (0.6 + 0.4 * frac)
    # type alone is weak evidence, and only when the type is firm or the analysis agrees
    # (bare numbers pass the price check, so a name-inferred "price" needs value_type price)
    weight = 0.4 if kind in SPECIFIC_TYPES else 0.25
    evidence = weight * frac if ctx.firm else (0.3 if cand.value_type == kind else 0.0)
    reason = f"type {kind} {frac:.0%}" if frac else f"not {kind}"
    return factor, evidence, reason


def score_candidate(ctx: FieldContext, cand: Candidate, prep: Prepared) -> Scored:
    """Score one candidate against one requested field."""
    req = ctx.req
    srcs = prep.sources[cand.id]
    reasons: list[str] = []
    label, why, src = _label_evidence(ctx, srcs)
    evidence = label
    if why and label >= MIN_LISTED:
        reasons.append(f"{why} ~ {label:.2f}")
    d, dwhy = _desc_evidence(ctx, srcs)
    if d > evidence:
        evidence = d
    if dwhy and d >= MIN_LISTED:
        reasons.append(f"{dwhy} ~ {d:.2f}")
    factor = 1.0
    if req.examples:
        identity, shape = example_agreement(req.examples, cand.examples)
        if identity:
            reasons.append("example value matches" if identity == 1.0 else "example overlaps")
            evidence = max(evidence, identity)
        factor *= 0.6 + 0.4 * shape
    tfactor, tevidence, twhy = _type_terms(ctx, cand, prep)
    if tevidence > evidence:
        evidence = tevidence
    factor *= tfactor
    if twhy:
        reasons.append(twhy)
    factor *= _shape_factor(req, cand, None)
    score = evidence * factor * (0.9 + 0.1 * min(cand.p, 1.0))
    key = src.text if src is not None and src.kind == "key" and label >= AMBIGUOUS else None
    return Scored(
        id=cand.id,
        score=round(score, 4),
        reasons=reasons,
        examples=list(cand.examples[:3]),
        use=cand.id,
        candidate=cand,
        key=key,
    )


def _table_use(table: Table) -> list[str]:
    if table.kind == "key_value":
        return [c for c in (table.key_candidate, table.value_candidate) if c]
    return [c.candidate_id for c in table.columns]


def _table_examples(table: Table, by_id: dict[str, Candidate]) -> list[str]:
    if table.kind == "key_value":
        return list(table.keys_seen[:3])
    out: list[str] = []
    for col in table.columns[:3]:
        cand = by_id.get(col.candidate_id)
        if cand is not None and cand.examples:
            out.append(f"{col.name}: {cand.examples[0]}")
    return out


def score_table(ctx: FieldContext, table: Table, prep: Prepared) -> Scored:
    """Score a whole table (record list / key-value list) against a group field."""
    srcs = prep.table_sources[table.id]
    label, why, _ = _label_evidence(ctx, srcs)
    d, dwhy = _desc_evidence(ctx, srcs)
    reasons = [f"{why} ~ {label:.2f}"] if why and label >= MIN_LISTED else []
    if dwhy and d >= MIN_LISTED:
        reasons.append(f"{dwhy} ~ {d:.2f}")
    shape = _shape_factor(ctx.req, None, table)
    score = max(label, d) * shape * (0.9 + 0.1 * min(table.p, 1.0))
    return Scored(
        id=table.id,
        score=round(score, 4),
        reasons=[f"{table.kind} table", *reasons],
        examples=_table_examples(table, prep.by_id),
        use=_table_use(table),
        table=table,
    )


def classify(ranked: list[Scored]) -> FieldStatus:
    """found / ambiguous / missing from the ranked scores (best first)."""
    if not ranked or ranked[0].score < AMBIGUOUS:
        return "missing"
    top = ranked[0]
    if top.score < FOUND:
        return "ambiguous"
    for other in ranked[1:]:
        if other.score < AMBIGUOUS or top.score - other.score >= MARGIN:
            break
        if other.examples and other.example_key == top.example_key:
            continue  # same values at another node: not a real competitor
        return "ambiguous"
    return "found"


def score_all(ctx: FieldContext, prep: Prepared) -> list[Scored]:
    """Every candidate (and, for group fields, every table) scored, best first."""
    scored = [score_candidate(ctx, c, prep) for c in prep.usable]
    if ctx.req.type in ("record_list", "key_value"):
        scored += [score_table(ctx, t, prep) for t in prep.tables.values()]
    scored.sort(key=lambda s: -s.score)  # stable: ties keep candidate (document) order
    return scored


def match_field(req: FieldRequest, prep: Prepared, *, top_k: int = TOP_K) -> FieldMatch:
    listed = [s for s in score_all(FieldContext.of(req), prep) if s.score >= MIN_LISTED]
    status = classify(listed)
    ranked = listed[:top_k]
    use: str | list[str] = ranked[0].use if status != "missing" else ABSENT
    return FieldMatch(name=req.name, status=status, use=use, ranked=ranked)


def match_template(table: CandidateTable, fields: Sequence[FieldRequest]) -> dict[str, FieldMatch]:
    """Match every requested field against one template's candidate table."""
    prep = prepare(table)
    return {req.name: match_field(req, prep) for req in fields}


def score_by_id(req: FieldRequest, prep: Prepared, cid: str) -> Scored | None:
    """Score one specific candidate or table id (used when the LLM or the user picks it)."""
    ctx = FieldContext.of(req)
    for c in prep.usable:
        if c.id == cid:
            return score_candidate(ctx, c, prep)
    if cid in prep.tables:
        return score_table(ctx, prep.tables[cid], prep)
    return None
