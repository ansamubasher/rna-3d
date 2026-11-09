"""Unit tests for src/rank.py: within-target z-score, edge cases, ranking, confidence."""
import math

import numpy as np
import pytest

from rank import confidence_label, rank_order, rank_target, within_target_normalize

CFG = {"rank": {"top_k": 2, "min_candidates_for_zscore": 3, "std_ddof": 0},
       "confidence": {"high_max_ppl": None, "medium_max_ppl": None}}


# --- z-score ---------------------------------------------------------------

def test_zscore_matches_formula():
    ppl = [2.1, 3.5, 2.8, 2.1, 3.9]
    r = within_target_normalize(ppl)
    x = np.array(ppl)
    z = (x - x.mean()) / x.std(ddof=0)
    assert r.method == "zscore"
    assert np.allclose(r.z_scores, z)
    assert np.allclose(r.final_scores, -z)            # final = -z for perplexity


def test_zscore_has_mean_zero_unit_std():
    r = within_target_normalize([1.0, 2.0, 3.0, 4.0, 10.0])
    assert abs(np.mean(r.z_scores)) < 1e-12
    assert abs(np.std(r.z_scores) - 1.0) < 1e-12


def test_lower_perplexity_gets_higher_score():
    r = within_target_normalize([1.5, 3.0, 2.0])
    assert r.final_scores[0] > r.final_scores[2] > r.final_scores[1]


def test_higher_is_better_keeps_sign():
    r = within_target_normalize([0.8, 0.5, 0.3], higher_is_better=True)
    assert r.final_scores[0] > r.final_scores[1] > r.final_scores[2]


def test_zscore_is_shift_and_scale_invariant():
    a = within_target_normalize([2.0, 3.0, 5.0]).final_scores
    b = within_target_normalize([20.0, 30.0, 50.0]).final_scores
    c = within_target_normalize([12.0, 13.0, 15.0]).final_scores
    assert np.allclose(a, b) and np.allclose(a, c)


# --- edge cases --------------------------------------------------------------

def test_fallback_when_too_few_candidates():
    r = within_target_normalize([3.0, 2.5], min_candidates=3)
    assert r.method == "fallback_raw"
    assert r.final_scores == [-3.0, -2.5]             # final = -ppl
    assert r.z_scores == [None, None]
    assert "valid candidates" in r.reason


def test_fallback_when_std_is_zero():
    r = within_target_normalize([3.0, 3.0, 3.0])
    assert r.method == "fallback_raw"
    assert r.final_scores == [-3.0, -3.0, -3.0]
    assert "std == 0" in r.reason


def test_fallback_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        within_target_normalize([1.0, 2.0], label="T1")
    assert "T1" in caplog.text


def test_single_candidate_falls_back():
    assert within_target_normalize([2.0]).method == "fallback_raw"


@pytest.mark.parametrize("bad", [[], [1.0, float("nan"), 2.0], [1.0, float("inf"), 2.0]])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        within_target_normalize(bad)


# --- ranking -----------------------------------------------------------------

def test_rank_order_best_first_and_ties_by_id():
    assert rank_order([0.5, 2.0, 2.0, -1.0], ["d", "c", "a", "b"]) == [2, 1, 0, 3]


def test_rank_target_puts_failed_last_and_returns_top_k():
    cands = [{"candidate_id": "x", "status": "ok", "perplexity": "3.0"},
             {"candidate_id": "y", "status": "ok", "perplexity": "2.0"},
             {"candidate_id": "bad", "status": "error", "perplexity": ""},
             {"candidate_id": "z", "status": "ok", "perplexity": "2.5"}]
    res = rank_target(cands, CFG, "T")
    assert [c["candidate_id"] for c in res["ranked"]] == ["y", "z", "x", "bad"]
    assert [c["rank"] for c in res["ranked"]] == [1, 2, 3, None]
    assert [c["candidate_id"] for c in res["top_k"]] == ["y", "z"]
    assert res["n_valid"] == 3 and res["n_failed"] == 1
    assert res["norm_method"] == "zscore"


def test_rank_target_no_valid_candidates():
    res = rank_target([{"candidate_id": "a", "status": "error", "perplexity": ""}], CFG, "T")
    assert res["top_k"] == [] and res["confidence"] == "none"


# --- confidence ----------------------------------------------------------------

@pytest.mark.parametrize("ppl,expected", [(2.0, "high"), (2.2, "high"), (2.5, "medium"),
                                          (3.0, "medium"), (3.01, "low")])
def test_confidence_label_bands(ppl, expected):
    assert confidence_label(ppl, 2.2, 3.0) == expected


def test_confidence_uncalibrated_and_none():
    assert confidence_label(2.0, None, None) == "uncalibrated"
    assert confidence_label(None, 2.0, 3.0) == "none"
    assert confidence_label(math.nan, 2.0, 3.0) == "none"


def test_confidence_bad_config_raises():
    with pytest.raises(ValueError):
        confidence_label(2.0, 3.0, 2.0)
