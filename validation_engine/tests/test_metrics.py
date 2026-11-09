"""Unit tests for src/evaluate.py: per-target metrics, random baseline, Wilcoxon, calibration."""
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from evaluate import (evaluate, expected_best_of_k, fit_thresholds, loto_calibration,
                      paired_wilcoxon, random_metrics, spearman, target_metrics, truth_label,
                      write_thresholds_to_config)

TM = [0.20, 0.80, 0.50, 0.35, 0.65, 0.10]
IDS = ["a", "b", "c", "d", "e", "f"]


# --- per-target metrics ----------------------------------------------------------

def test_perfect_scores_give_oracle_metrics():
    m = target_metrics(TM, TM, IDS, k=5)
    assert m["top1_tm"] == 0.80 and m["selection_loss"] == 0.0
    assert m["best5_tm"] == 0.80 and m["spearman"] == pytest.approx(1.0)


def test_reversed_scores():
    m = target_metrics([-t for t in TM], TM, IDS, k=2)
    assert m["top1_tm"] == 0.10                      # picks the worst
    assert m["best5_tm"] == 0.20                     # best of the 2 worst
    assert m["selection_loss"] == pytest.approx(0.70)
    assert m["spearman"] == pytest.approx(-1.0)


def test_best_of_k_uses_top_k_only():
    scores = [5, 4, 3, 2, 1, 0]                      # picks a, b, ... in order
    assert target_metrics(scores, TM, IDS, k=1)["best5_tm"] == 0.20
    assert target_metrics(scores, TM, IDS, k=2)["best5_tm"] == 0.80


def test_tie_break_by_candidate_id():
    m = target_metrics([1, 1, 0, 0, 0, 0], TM, IDS, k=1)
    assert m["top1_tm"] == 0.20                      # 'a' wins the tie with 'b'


def test_spearman_undefined_for_constant_input():
    assert math.isnan(spearman([1, 1, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(spearman([1, 2, 3], [0.5, 0.5, 0.5]))


# --- random baseline (exact expectation) ---------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3, 5, 6, 10])
def test_expected_best_of_k_matches_brute_force(k):
    if k >= len(TM):
        assert expected_best_of_k(TM, k) == max(TM)
        return
    brute = np.mean([max(c) for c in combinations(TM, k)])
    assert expected_best_of_k(TM, k) == pytest.approx(brute)


def test_random_metrics():
    r = random_metrics(TM, k=5)
    assert r["top1_tm"] == pytest.approx(np.mean(TM))
    assert r["selection_loss"] == pytest.approx(0.80 - np.mean(TM))
    assert r["spearman"] == 0.0


# --- paired test ------------------------------------------------------------------------

def test_wilcoxon_identical_inputs():
    r = paired_wilcoxon([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert math.isnan(r["p_value"]) and "identical" in r["note"]
    assert r["n_ties"] == 3


def test_wilcoxon_clear_difference():
    a = [0.9, 0.8, 0.85, 0.7, 0.95, 0.75, 0.88, 0.92]
    b = [x - 0.2 for x in a]
    r = paired_wilcoxon(a, b)
    assert r["n_a_better"] == 8 and r["p_value"] < 0.05
    assert r["mean_diff"] == pytest.approx(0.2)


# --- evaluate() over a small dataset ------------------------------------------------------

def _toy_df():
    rows = []
    for tid, tms, ppls in [("T1", [0.2, 0.8, 0.5, 0.4], [3.5, 1.8, 2.5, 3.0]),   # gRNAde perfect
                           ("T2", [0.3, 0.6, 0.1], [2.0, 3.0, 2.5])]:           # gRNAde imperfect
        for i, (t, p) in enumerate(zip(tms, ppls)):
            rows.append({"target_id": tid, "candidate_id": f"{tid}_{i}", "tm": t,
                         "perplexity": p, "pmol": t})                         # pMoL = perfect here
    return pd.DataFrame(rows)


def test_evaluate_raw_and_norm_identical_per_target():
    pt, pooled = evaluate(_toy_df(), ["grnade_raw", "grnade_norm"], k=5, min_cands=3)
    raw = pt[pt.method == "grnade_raw"].set_index("target_id")
    norm = pt[pt.method == "grnade_norm"].set_index("target_id")
    for col in ["top1_tm", "best5_tm", "selection_loss", "spearman"]:
        assert np.allclose(raw[col], norm[col])
    assert raw.loc["T1", "top1_tm"] == 0.8 and raw.loc["T2", "top1_tm"] == 0.3


def test_evaluate_all_methods_and_oracle_is_upper_bound():
    methods = ["oracle", "random", "grnade_raw", "grnade_norm", "lociparse_pmol", "grnade_lociparse_avg"]
    pt, _ = evaluate(_toy_df(), methods, k=5, min_cands=3)
    for tid, g in pt.groupby("target_id"):
        oracle = g[g.method == "oracle"]["top1_tm"].iloc[0]
        assert (g["top1_tm"] <= oracle + 1e-12).all()
        assert g[g.method == "oracle"]["selection_loss"].iloc[0] == 0


def test_evaluate_skips_small_targets():
    df = _toy_df()
    df = df[~df.candidate_id.isin(["T2_1", "T2_2"])]           # T2 now has 1 candidate
    pt, _ = evaluate(df, ["grnade_norm"], k=5, min_cands=3)
    assert set(pt.target_id) == {"T1"}


# --- confidence calibration ---------------------------------------------------------------------

def test_truth_label():
    assert truth_label(0.5, 0.45, 0.30) == "high"
    assert truth_label(0.3, 0.45, 0.30) == "medium"
    assert truth_label(0.1, 0.45, 0.30) == "low"


def test_fit_thresholds_separable_data():
    ppl = [1.5, 1.6, 2.4, 2.5, 3.5, 3.6]
    truth = ["high", "high", "medium", "medium", "low", "low"]
    h, m, acc = fit_thresholds(ppl, truth)
    assert acc == 1.0 and 1.6 < h < 2.4 and 2.5 < m < 3.5


def test_loto_never_uses_the_held_out_target():
    ppl = [1.5, 1.6, 1.7, 2.4, 2.5, 2.6, 3.5, 3.6, 3.7]
    truth = ["high"] * 3 + ["medium"] * 3 + ["low"] * 3
    assert loto_calibration(ppl, truth)["loto_accuracy"] == 1.0
    # A target that contradicts all the others must be predicted wrongly by LOTO,
    # which shows that its own label was not used when fitting its thresholds.
    ppl2, truth2 = ppl + [1.55], truth + ["low"]
    preds = loto_calibration(ppl2, truth2)["loto_predictions"]
    assert preds[-1] == "high"


def test_write_thresholds_keeps_comments(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("confidence:\n  # comment kept\n  high_max_ppl: null\n  medium_max_ppl: null\n", encoding="utf-8")
    write_thresholds_to_config(p, 2.1, 2.9)
    text = p.read_text(encoding="utf-8")
    assert "# comment kept" in text and "high_max_ppl: 2.1000" in text and "medium_max_ppl: 2.9000" in text
