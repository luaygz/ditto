"""Candidate scoring, winner selection and failure detection (pure functions over arrays).

Per candidate (brief, stage 7):

* presence — share of the pages where the analysis finds the field's path that the selector
  fills (``presence_score``); the raw ``presence_rate`` over all pages is reported too;
* multiplicity — scalar: share of matched pages with exactly one node; list: share of pages
  whose value count does not exceed the analysis path's node count (a list may read part of
  a path that several fields share, such as one labelled row of a details block, but never
  nodes beyond it); record_list/key_value: share of sampled pages whose matched row count
  equals the real sibling count;
* type check — share of matched pages whose value passes the field's type check;
* agreement — share of the pages where two candidates both match on which their values agree.

Winner: highest ``presence × multiplicity × type_pass`` (rounded to 3 decimals), ties broken
toward fewer positional steps, then higher mean agreement, then the lower index. Candidates
with the same multiplicity and type pass whose scores differ by at most ``PRESENCE_TIE`` count
as tied: the analysis' expected presence comes from the source path, which overstates a field
that reads only part of a shared path, and the winner's fallbacks fill such small gaps.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ditto.specgen.lint import split_steps
from ditto.verify.models import CandidateOrigin, CandidateScore
from ditto.verify.pages import NO_VALUE, ProbeMatrix

Kind = Literal["scalar", "list", "rows"]

FALLBACK_AGREEMENT = 0.98
SCORE_DECIMALS = 3
PRESENCE_TIE = 0.02
OK, MISSING, MULTIPLE, TYPE = 0, 1, 2, 3
FAILURE_NAMES = {MISSING: "missing", MULTIPLE: "multiple", TYPE: "type"}

_AXIS = (
    r"(?:(?:child|descendant|descendant-or-self|following-sibling|preceding-sibling|following"
    r"|preceding|parent|ancestor|ancestor-or-self|self)::)?"
)
_POSITIONAL_STEP = re.compile(rf"^{_AXIS}(?:[A-Za-z][\w.-]*|\*)(?:\[\s*\d+\s*\])*$")
_INDEX = re.compile(r"\[\s*\d+\s*\]")
_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")


def positional_steps(xpath: str) -> int:
    """Element steps that carry no anchor (a bare tag, optionally with a numeric index); a
    numeric index counts once more (``li[1]`` is less general than ``li``)."""
    count = 0
    for branch in _STRING.sub("''", xpath).split("|"):
        for step in split_steps(branch.strip()):
            if step and step not in (".", "..") and _POSITIONAL_STEP.match(step):
                count += 1 + len(_INDEX.findall(step))
    return count


@dataclass
class Obs:
    """One probe's observations over the pages of a cluster (1-D arrays, same page order)."""

    nodes: np.ndarray[Any, np.dtype[np.int32]]
    values: np.ndarray[Any, np.dtype[np.int32]]
    type_ok: np.ndarray[Any, np.dtype[np.float32]]
    vhash: np.ndarray[Any, np.dtype[np.uint64]]
    siblings: np.ndarray[Any, np.dtype[np.int32]]
    keys: set[str] = field(default_factory=set)

    @classmethod
    def column(cls, m: ProbeMatrix, j: int) -> Obs:
        return cls(
            m.nodes[:, j].copy(),
            m.values[:, j].copy(),
            m.type_ok[:, j].copy(),
            m.vhash[:, j].copy(),
            m.siblings[:, j].copy(),
            set(m.keys[j]),
        )

    @classmethod
    def build(
        cls,
        nodes: Sequence[int],
        values: Sequence[int] | None = None,
        type_ok: Sequence[float] | None = None,
        vhash: Sequence[int] | None = None,
        siblings: Sequence[int] | None = None,
    ) -> Obs:
        """Convenience constructor (tests, hand-built observations)."""
        n = len(nodes)
        vals = list(nodes) if values is None else list(values)
        return cls(
            np.asarray(nodes, np.int32),
            np.asarray(vals, np.int32),
            np.asarray(
                type_ok if type_ok is not None else [1.0 if v else NO_VALUE for v in vals],
                np.float32,
            ),
            np.asarray(vhash if vhash is not None else [0] * n, np.uint64),
            np.asarray(siblings if siblings is not None else [-1] * n, np.int32),
        )

    @property
    def matched(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return self.values > 0


@dataclass
class Expectation:
    """What the analysis expects per page: ``mask`` (its path is present), ``counts`` (nodes
    at the path), and the sample-based ``analysis_p``. ``mask`` is None when no source
    candidate path could be checked."""

    mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None
    counts: np.ndarray[Any, np.dtype[np.int32]] | None = None
    analysis_p: float | None = None

    @property
    def known(self) -> bool:
        return self.mask is not None and bool(self.mask.any())

    @property
    def expected_presence(self) -> float | None:
        if self.mask is not None and self.known:
            return float(self.mask.mean())
        return self.analysis_p


def _mean(arr: np.ndarray[Any, Any]) -> float | None:
    return float(arr.mean()) if arr.size else None


def multiplicity(kind: Kind, obs: Obs, exp: Expectation) -> float | None:
    if kind == "scalar":
        hit = obs.nodes > 0
        return _mean(obs.nodes[hit] == 1)
    if kind == "list":
        if exp.counts is None:
            return 1.0 if obs.matched.any() else None
        both = obs.matched & (exp.counts > 0)
        if not both.any():
            return 1.0 if obs.matched.any() else None
        return _mean(obs.values[both] <= exp.counts[both])
    measured = (obs.siblings >= 0) & (obs.nodes > 0)
    return _mean(obs.nodes[measured] == obs.siblings[measured])


def score_candidate(
    kind: Kind,
    obs: Obs,
    exp: Expectation,
    *,
    index: int = 0,
    role: str = "",
    xpath: str = "",
    origin: CandidateOrigin = "llm",
) -> CandidateScore:
    n = obs.values.shape[0]
    matched = obs.matched
    presence_rate = float(matched.mean()) if n else 0.0
    if exp.mask is not None and exp.known:
        presence_score = int((matched & exp.mask).sum()) / int(exp.mask.sum())
    elif exp.analysis_p:
        presence_score = min(1.0, presence_rate / exp.analysis_p)
    else:
        presence_score = presence_rate
    mult = multiplicity(kind, obs, exp)
    typed = obs.type_ok >= 0
    type_pass = _mean(obs.type_ok[typed])
    score = presence_score * (1.0 if mult is None else mult) * (type_pass or 0.0)
    rows = obs.values[matched]
    return CandidateScore(
        index=index,
        role=role,
        xpath=xpath,
        origin=origin,
        pages_matched=int(matched.sum()),
        presence_rate=round(presence_rate, 4),
        expected_presence=None
        if exp.expected_presence is None
        else round(exp.expected_presence, 4),
        presence_score=round(presence_score, 4),
        multiplicity=None if mult is None else round(mult, 4),
        type_check_pass=None if type_pass is None else round(type_pass, 4),
        score=round(score, 4),
        positional_steps=positional_steps(xpath) if xpath else 0,
        rows_min=int(rows.min()) if kind == "rows" and rows.size else None,
        rows_max=int(rows.max()) if kind == "rows" and rows.size else None,
        keys_seen=len(obs.keys) if kind == "rows" and obs.keys else None,
    )


def agreement(a: Obs, b: Obs) -> float | None:
    both = a.matched & b.matched
    if not both.any():
        return None
    return float((a.vhash[both] == b.vhash[both]).mean())


def fill_agreement(scores: Sequence[CandidateScore], obs: Sequence[Obs]) -> None:
    for i, si in enumerate(scores):
        si.agreement = {}
        for j in range(len(scores)):
            if i != j:
                value = agreement(obs[i], obs[j])
                si.agreement[str(j)] = None if value is None else round(value, 4)


def _mean_agreement(s: CandidateScore) -> float:
    vals = [v for v in s.agreement.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def winner_key(s: CandidateScore) -> tuple[float, int, float, int]:
    return (-round(s.score, SCORE_DECIMALS), s.positional_steps, -_mean_agreement(s), s.index)


def tied(a: CandidateScore, b: CandidateScore) -> bool:
    """Same multiplicity and type pass, scores within ``PRESENCE_TIE`` of each other."""
    gap = abs(round(a.score, SCORE_DECIMALS) - round(b.score, SCORE_DECIMALS))
    return (
        a.multiplicity == b.multiplicity
        and a.type_check_pass == b.type_check_pass
        and gap <= PRESENCE_TIE + 1e-9
    )


def ranking(scores: Sequence[CandidateScore]) -> list[int]:
    """Candidate indices (into ``scores``), best first: by score, except that among candidates
    tied with the best remaining one the fewest positional steps, then the highest mean
    agreement, then the lowest index goes first."""
    remaining = sorted(range(len(scores)), key=lambda i: winner_key(scores[i]))
    out: list[int] = []
    while remaining:
        lead = scores[remaining[0]]
        group = [i for i in remaining if tied(scores[i], lead)]
        pick = min(group, key=lambda i: winner_key(scores[i])[1:])
        out.append(pick)
        remaining.remove(pick)
    return out


def choose_winner(scores: Sequence[CandidateScore]) -> int:
    """Index (into ``scores``) of the winning candidate."""
    return ranking(scores)[0]


def fallback_indices(scores: Sequence[CandidateScore], winner: int) -> list[int]:
    """Candidates agreeing with the winner on >= 98% of the pages where both match."""
    out = []
    for i in ranking(scores):
        if i == winner:
            continue
        agree = scores[i].agreement.get(str(winner))
        if agree is not None and agree >= FALLBACK_AGREEMENT:
            out.append(i)
    return out


def selection_reason(scores: Sequence[CandidateScore], winner: int, value_type: str | None) -> str:
    w = scores[winner]
    exp = "?" if w.expected_presence is None else f"{w.expected_presence:.3f}"
    mult = "n/a" if w.multiplicity is None else f"{w.multiplicity:.3f}"
    typ = "n/a" if w.type_check_pass is None else f"{w.type_check_pass:.3f}"
    parts = [
        f"{w.role} wins with score {w.score:.3f} = presence {w.presence_score:.3f} "
        f"(rate {w.presence_rate:.3f}, analysis expects {exp}) × multiplicity {mult} × "
        f"type {typ} ({value_type or 'any'})"
    ]
    for i, s in enumerate(scores):
        if i == winner:
            continue
        note = f"{s.role} {s.score:.3f}"
        if tied(s, w):
            close = round(s.score, SCORE_DECIMALS) != round(w.score, SCORE_DECIMALS)
            prefix = f"within {PRESENCE_TIE} of the winner, " if close else ""
            if s.positional_steps != w.positional_steps:
                note += (
                    f" ({prefix}tie broken toward fewer positional steps: "
                    f"{w.positional_steps} vs {s.positional_steps})"
                )
            else:
                note += f" ({prefix}tie)"
        parts.append(note)
    return "; ".join(parts)


def effective(obs: Sequence[Obs], order: Sequence[int]) -> Obs:
    """What extraction sees per page: the first candidate in ``order`` that yields a value."""
    first = obs[order[0]]
    out = Obs(
        first.nodes.copy(),
        first.values.copy(),
        first.type_ok.copy(),
        first.vhash.copy(),
        first.siblings.copy(),
        set(first.keys),
    )
    filled = out.values > 0
    for idx in order[1:]:
        o = obs[idx]
        take = ~filled & (o.values > 0)
        if not take.any():
            continue
        out.nodes[take] = o.nodes[take]
        out.values[take] = o.values[take]
        out.type_ok[take] = o.type_ok[take]
        out.vhash[take] = o.vhash[take]
        out.siblings[take] = o.siblings[take]
        out.keys |= o.keys
        filled |= take
    return out


def page_failures(kind: Kind, eff: Obs, exp: Expectation) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Failure code per page: the analysis says the path is present but the field yields
    nothing (MISSING) or a scalar matches several nodes (MULTIPLE), or the value fails its
    type check (TYPE). Without an analysis expectation only MULTIPLE/TYPE apply."""
    n = eff.values.shape[0]
    codes = np.zeros(n, np.uint8)
    expected = exp.mask if exp.mask is not None and exp.known else None
    type_floor = 1.0 if kind == "scalar" else 0.5
    typed = eff.type_ok >= 0
    codes[typed & (eff.type_ok < type_floor)] = TYPE
    if kind == "scalar":
        several = eff.nodes > 1
        if expected is not None:
            several &= expected
        codes[several] = MULTIPLE
    if expected is not None:
        codes[expected & (eff.values == 0)] = MISSING
    return codes
