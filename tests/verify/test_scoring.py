"""Verification scorer: presence vs expected, multiplicity, type pass, tie-break, failures."""

from __future__ import annotations

import numpy as np

from ditto.verify.scoring import (
    MISSING,
    MULTIPLE,
    TYPE,
    Expectation,
    Obs,
    agreement,
    choose_winner,
    effective,
    fallback_indices,
    fill_agreement,
    multiplicity,
    page_failures,
    positional_steps,
    score_candidate,
    selection_reason,
)


def _scores(kind, obs, exp, xpaths):  # type: ignore[no-untyped-def]
    scores = [
        score_candidate(kind, o, exp, index=i, role=f"candidates[{i}]", xpath=x)
        for i, (o, x) in enumerate(zip(obs, xpaths, strict=True))
    ]
    fill_agreement(scores, obs)
    return scores


def test_positional_steps_counts_unanchored_element_steps() -> None:
    assert positional_steps("//h1[@itemprop='name']") == 0
    assert positional_steps("//header//h1") == 2
    assert positional_steps("//dt[normalize-space()='Email']/following-sibling::dd[1]//a") == 3
    assert positional_steps("//a/@href") == 1
    assert positional_steps("//h1[@id='x'] | //div/h2") == 2
    assert positional_steps(".") == 0
    assert positional_steps("//ul[@id='t']/li[1]") == 2  # an index counts once more
    assert positional_steps("//ul[@id='t']/li") == 1


def test_multiplicity_penalty_picks_single_node_candidate() -> None:
    exp = Expectation(mask=np.ones(4, bool), counts=np.ones(4, np.int32), analysis_p=1.0)
    single = Obs.build([1, 1, 1, 1], vhash=[1, 2, 3, 4])
    several = Obs.build([3, 3, 3, 3], vhash=[1, 2, 3, 4])
    scores = _scores("scalar", [several, single], exp, ["//dd", "//dd[@class='x']"])
    assert scores[0].multiplicity == 0.0 and scores[0].score == 0.0
    assert scores[1].multiplicity == 1.0 and scores[1].score == 1.0
    assert choose_winner(scores) == 1


def test_type_check_penalty() -> None:
    exp = Expectation(mask=np.ones(4, bool), analysis_p=1.0)
    good = Obs.build([1, 1, 1, 1], type_ok=[1, 1, 1, 1])
    half = Obs.build([1, 1, 1, 1], type_ok=[1, 0, 1, 0])
    scores = _scores("scalar", [half, good], exp, ["//dd[@id='a']", "//a[@id='b']"])
    assert scores[0].type_check_pass == 0.5
    assert scores[0].score == 0.5
    assert choose_winner(scores) == 1


def test_tie_broken_toward_fewer_positional_steps() -> None:
    exp = Expectation(mask=np.ones(3, bool), analysis_p=1.0)
    a = Obs.build([1, 1, 1], vhash=[7, 8, 9])
    b = Obs.build([1, 1, 1], vhash=[7, 8, 9])
    scores = _scores("scalar", [a, b], exp, ["//main/div/h1", "//h1[@itemprop='name']"])
    assert scores[0].score == scores[1].score == 1.0
    w = choose_winner(scores)
    assert w == 1
    reason = selection_reason(scores, w, None)
    assert "fewer positional steps: 0 vs 3" in reason
    assert fallback_indices(scores, w) == [0]


def test_presence_against_expected_for_optional_field() -> None:
    mask = np.array([True, False, True, False])
    exp = Expectation(mask=mask, counts=mask.astype(np.int32), analysis_p=0.5)
    exact = Obs.build([1, 0, 1, 0])
    partial = Obs.build([1, 0, 0, 0])
    s_exact = score_candidate("scalar", exact, exp)
    s_partial = score_candidate("scalar", partial, exp)
    assert s_exact.presence_rate == 0.5 and s_exact.presence_score == 1.0
    assert s_exact.expected_presence == 0.5
    assert s_partial.presence_score == 0.5


def test_presence_without_page_mask_uses_analysis_p() -> None:
    obs = Obs.build([1, 1, 0, 0])
    assert score_candidate("scalar", obs, Expectation(analysis_p=0.5)).presence_score == 1.0
    assert score_candidate("scalar", obs, Expectation()).presence_score == 0.5


def test_list_multiplicity_against_path_counts_and_rows_against_siblings() -> None:
    counts = np.array([2, 3, 1], np.int32)
    exp = Expectation(mask=counts > 0, counts=counts, analysis_p=1.0)
    assert multiplicity("list", Obs.build([2, 3, 1]), exp) == 1.0
    # part of a shared path (one labelled row of a details block) is fine...
    assert multiplicity("list", Obs.build([1, 1, 1]), exp) == 1.0
    # ... nodes beyond the source path are not
    assert multiplicity("list", Obs.build([3, 3, 2]), exp) == 1 / 3
    rows_ok = Obs.build([2, 3, 1], siblings=[2, 3, -1])
    rows_first = Obs.build([1, 1, 1], siblings=[2, 3, -1])
    assert multiplicity("rows", rows_ok, Expectation()) == 1.0
    assert multiplicity("rows", rows_first, Expectation()) == 0.0
    s = score_candidate("rows", rows_ok, Expectation(), xpath="//tr[td]")
    assert (s.rows_min, s.rows_max) == (1, 3)


def test_label_anchored_subset_beats_a_whole_shared_path() -> None:
    # a details block whose rows all share one path; the director is one labelled row and is
    # missing on one page in 60, so the analysis (which sees the shared path everywhere)
    # expects it on every page
    n = 60
    counts = np.full(n, 6, np.int32)
    exp = Expectation(mask=np.ones(n, bool), counts=counts, analysis_p=1.0)
    director = Obs.build([1] * (n - 1) + [0], vhash=list(range(n)))
    everything = Obs.build([6] * n, vhash=list(range(100, 100 + n)))
    xpaths = [
        "//div[@id='details']//li[starts-with(normalize-space(),'Director:')]/a",
        "//div[@id='details']/ul/li/span/a",
    ]
    scores = _scores("list", [director, everything], exp, xpaths)
    assert scores[0].score < scores[1].score == 1.0  # presence 59/60 against the whole path
    assert choose_winner(scores) == 0  # within PRESENCE_TIE: fewer positional steps wins
    assert "within 0.02 of the winner, tie broken toward fewer positional steps: 1 vs 4" in (
        selection_reason(scores, 0, None)
    )
    # a real shortfall is not a tie
    sparse = Obs.build([1] * (n // 2) + [0] * (n // 2), vhash=list(range(n)))
    assert choose_winner(_scores("list", [sparse, everything], exp, xpaths)) == 1
    # neither is a multiplicity difference (a scalar matching several nodes on some pages)
    one = Expectation(mask=np.ones(n, bool), counts=np.ones(n, np.int32), analysis_p=1.0)
    exact = Obs.build([1] * n, vhash=list(range(n)))
    multi = Obs.build([1] * (n - 1) + [2], vhash=list(range(n)))
    s = _scores("scalar", [exact, multi], one, ["//main/div/h1", "//h1[@itemprop='name']"])
    assert choose_winner(s) == 0


def test_agreement_and_fallbacks_need_98_percent() -> None:
    a = Obs.build([1] * 50, vhash=list(range(50)))
    b = Obs.build([1] * 50, vhash=[*range(49), 999])
    c = Obs.build([0] * 50)
    assert agreement(a, b) == 0.98
    assert agreement(a, c) is None
    exp = Expectation(mask=np.ones(50, bool), analysis_p=1.0)
    scores = _scores("scalar", [a, b, c], exp, ["//h1[@id='a']", "//h1[@id='b']", "//h1[@id='c']"])
    assert fallback_indices(scores, 0) == [1]


def test_effective_uses_fallback_where_selected_is_empty() -> None:
    sel = Obs.build([1, 0, 0], vhash=[5, 0, 0])
    fb = Obs.build([1, 1, 0], vhash=[5, 6, 0])
    eff = effective([sel, fb], [0, 1])
    assert eff.values.tolist() == [1, 1, 0]
    assert eff.vhash.tolist() == [5, 6, 0]


def test_page_failures_codes() -> None:
    mask = np.array([True, True, True, False, False])
    exp = Expectation(mask=mask, analysis_p=0.6)
    eff = Obs.build([0, 2, 1, 3, 1], values=[0, 2, 1, 3, 1], type_ok=[-1, 1, 0, 1, 1])
    codes = page_failures("scalar", eff, exp)
    # page 3 has several nodes but the analysis does not expect the field there
    assert codes.tolist() == [MISSING, MULTIPLE, TYPE, 0, 0]
    list_codes = page_failures("list", Obs.build([1, 1], type_ok=[0.6, 0.4]), Expectation())
    assert list_codes.tolist() == [0, TYPE]
