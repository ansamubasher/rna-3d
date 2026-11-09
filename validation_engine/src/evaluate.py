"""
evaluate.py — Step 8: how well does each method PICK good structures?

For every target (an RNA with many candidate structures and a known native
structure) we let each method rank the candidates, then compare its choice
with the true quality (TM-score vs the native, from label_usalign.py).

METRICS (per target, then mean and median over targets)
-------------------------------------------------------
* spearman        Spearman correlation between the method's score and the true
                  TM-score over that target's candidates (does it order them well?)
* top1_tm         TM-score of the method's #1 pick
* best5_tm        best TM-score among the method's top 5 (the web app shows 5)
* selection_loss  (best TM-score available) - top1_tm   (0 = perfect pick)
Also reported: pooled_spearman = one Spearman over ALL candidates of ALL
targets. That is where within-target normalisation can make a difference (see below).

METHODS (rows of the results table)
-----------------------------------
1. oracle               picks by the true TM-score (upper bound)
2. random               exact expected value of a uniformly random pick
                        (not a simulation, so no randomness at all)
3. grnade_raw           -perplexity (ablation: no normalisation)
4. grnade_norm          -z(perplexity) within target (OURS)
5. lociparse_pmol       lociPARSE pMoL (higher = better)
6. grnade_lociparse_avg mean of the z-scored gRNAde score and the z-scored lociPARSE score

AN IMPORTANT POINT FOR THE REPORT
---------------------------------
Inside ONE target, z = (ppl - mean)/std is a shift-and-scale of ppl, so
grnade_raw and grnade_norm produce the SAME ORDER. Their per-target metrics
(spearman, top1, best5, loss) are therefore identical BY CONSTRUCTION, and the
Wilcoxon test between them reports "identical". Normalisation matters when scores
are compared ACROSS targets (pooled_spearman, confidence calibration) or COMBINED
with another method (method 6). The summary states this explicitly.

FAIR COMPARISON
---------------
The main analysis uses, per target, only candidates that have a TM-score label
AND a score from every method being compared, so all methods choose from the same
pool. Targets listed in data.excluded_rna_protein_targets or data.excluded_duplicate_targets
are excluded (logged).
Targets with fewer than rank.min_candidates_for_zscore candidates are skipped (logged).
Every metric is reported for subset "all" and for subset "no_overlap" (without
data.possible_training_overlap_targets).

PAIRED TESTS
------------
Wilcoxon signed-rank on per-target top1_tm for the pairs in evaluate.paired_tests.

CONFIDENCE CALIBRATION (leave-one-target-out)
---------------------------------------------
Truth label per target, from the TM-score of gRNAde's top pick:
    >= calib_tm_good -> "high", >= calib_tm_ok -> "medium", else "low".
Predictor: the raw perplexity of gRNAde's top pick, with two thresholds
(high_max_ppl <= medium_max_ppl). For each target t, the thresholds are fitted
(max accuracy) on all OTHER targets and applied to t. That gives the honest
LOTO accuracy, because thresholds are never fitted on the target they are tested on.
The final thresholds (fitted on all targets) are written to
results/confidence_thresholds.json, and to config.yaml if you pass --write_config.

Usage:
    python src/evaluate.py
    python src/evaluate.py --methods oracle random grnade_raw grnade_norm   # before lociPARSE is run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import load_config, resolve_path                             # noqa: E402
from rank import confidence_label, rank_order, within_target_normalize   # noqa: E402

log = logging.getLogger("evaluate")

ALL_METHODS = ["oracle", "random", "grnade_raw", "grnade_norm",
               "lociparse_pmol", "grnade_lociparse_avg"]
NEEDS = {  # which score columns each method needs
    "oracle": [], "random": [], "grnade_raw": ["perplexity"], "grnade_norm": ["perplexity"],
    "lociparse_pmol": ["pmol"], "grnade_lociparse_avg": ["perplexity", "pmol"],
}
METRICS = ["spearman", "top1_tm", "best5_tm", "selection_loss"]


# ---------------------------------------------------------------------------
# Metric functions (pure, unit-tested in tests/test_metrics.py)
# ---------------------------------------------------------------------------

def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation. NaN if either input is constant (undefined)."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).correlation)


def target_metrics(scores: Sequence[float], tm: Sequence[float], ids: Sequence[str],
                   k: int = 5) -> Dict[str, float]:
    """Metrics for ONE method on ONE target. `scores`: higher = better."""
    tm = np.asarray(tm, float)
    order = rank_order(list(map(float, scores)), list(ids))   # same tie-break as rank.py
    top1 = float(tm[order[0]])
    best = float(tm.max())
    return {"spearman": spearman(scores, tm), "top1_tm": top1,
            "best5_tm": float(tm[order[:k]].max()), "selection_loss": best - top1,
            "best_available_tm": best, "n_candidates": len(tm)}


def expected_best_of_k(tm: Sequence[float], k: int) -> float:
    """Exact E[max TM-score] of k candidates drawn uniformly without replacement.

    Sort ascending t_1 <= ... <= t_n. The maximum of a random k-subset is t_i with
    probability C(i-1, k-1) / C(n, k) (i-1 smaller ones to choose the other k-1 from).
    """
    t = np.sort(np.asarray(tm, float))
    n = len(t)
    if k >= n:
        return float(t.max())
    total = math.comb(n, k)
    return float(sum(t[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / total)


def random_metrics(tm: Sequence[float], k: int = 5) -> Dict[str, float]:
    """Expected metrics of a uniformly random ranking (no sampling)."""
    tm = np.asarray(tm, float)
    best = float(tm.max())
    return {"spearman": 0.0,                 # expected rank correlation of a random order
            "top1_tm": float(tm.mean()), "best5_tm": expected_best_of_k(tm, k),
            "selection_loss": best - float(tm.mean()),
            "best_available_tm": best, "n_candidates": len(tm)}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
    """Wilcoxon signed-rank test, a vs b, paired by target (two-sided)."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"n_targets": int(len(d)), "mean_diff": float(d.mean()) if len(d) else float("nan"),
           "n_a_better": int((d > 0).sum()), "n_b_better": int((d < 0).sum()),
           "n_ties": int((d == 0).sum())}
    if len(d) == 0 or np.all(d == 0):
        out.update(statistic=float("nan"), p_value=float("nan"),
                   note="identical on every target (no test possible)")
        return out
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    out.update(statistic=float(res.statistic), p_value=float(res.pvalue),
               note="ties dropped (zero_method='wilcox')" if out["n_ties"] else "")
    return out


# ---------------------------------------------------------------------------
# Confidence calibration (unit-tested)
# ---------------------------------------------------------------------------

def truth_label(top1_tm: float, tm_good: float, tm_ok: float) -> str:
    return "high" if top1_tm >= tm_good else ("medium" if top1_tm >= tm_ok else "low")


def fit_thresholds(ppl: Sequence[float], truth: Sequence[str]) -> Tuple[float, float, float]:
    """Choose (high_max_ppl, medium_max_ppl) that maximise accuracy on these targets.

    Candidate cut points are midpoints between consecutive observed perplexities,
    plus one just below the minimum and one just above the maximum. All pairs with
    high <= medium are tried. The first best pair in sorted order wins, so the result
    is deterministic. Returns (high_max, medium_max, training_accuracy).
    """
    v = np.unique(np.asarray(ppl, float))
    cuts = [float(v[0] - 1e-6)] + [float((a + b) / 2) for a, b in zip(v[:-1], v[1:])] + [float(v[-1] + 1e-6)]
    best = (cuts[0], cuts[-1], -1.0)
    for i, h in enumerate(cuts):
        for m in cuts[i:]:
            acc = float(np.mean([confidence_label(p, h, m) == t for p, t in zip(ppl, truth)]))
            if acc > best[2]:
                best = (h, m, acc)
    return best


def loto_calibration(ppl: Sequence[float], truth: Sequence[str]) -> Dict[str, object]:
    """Leave-one-target-out: fit on n-1 targets, predict the held-out one."""
    ppl, truth = list(map(float, ppl)), list(truth)
    preds = []
    for i in range(len(ppl)):
        tr_p = ppl[:i] + ppl[i + 1:]
        tr_t = truth[:i] + truth[i + 1:]
        h, m, _ = fit_thresholds(tr_p, tr_t)
        preds.append(confidence_label(ppl[i], h, m))
    acc = float(np.mean([p == t for p, t in zip(preds, truth)])) if preds else float("nan")
    return {"loto_predictions": preds, "loto_accuracy": acc}


# ---------------------------------------------------------------------------
# Data loading and method scores
# ---------------------------------------------------------------------------

def _ok(path: Path, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"target_id": str, "candidate_id": str})
    df = df[df["status"] == "ok"]
    return df[["target_id", "candidate_id"] + cols]


def load_data(labels: Path, grnade: Optional[Path], lociparse: Optional[Path],
              methods: List[str]) -> pd.DataFrame:
    """One row per candidate with tm (+ perplexity / pmol as needed). Inner join = same pool."""
    df = _ok(labels, ["tm_native"]).rename(columns={"tm_native": "tm"})
    need = {c for m in methods for c in NEEDS[m]}
    if "perplexity" in need:
        df = df.merge(_ok(grnade, ["perplexity"]), on=["target_id", "candidate_id"], how="inner")
    if "pmol" in need:
        df = df.merge(_ok(lociparse, ["pmol"]), on=["target_id", "candidate_id"], how="inner")
    return df.sort_values(["target_id", "candidate_id"]).reset_index(drop=True)


def method_scores(g: pd.DataFrame, method: str, min_cands: int) -> Optional[np.ndarray]:
    """Per-candidate scores (higher = better) for one target, or None for 'random'."""
    if method == "oracle":
        return g["tm"].to_numpy(float)
    if method == "random":
        return None
    if method == "grnade_raw":
        return -g["perplexity"].to_numpy(float)
    if method == "grnade_norm":
        return np.array(within_target_normalize(g["perplexity"], False, min_cands).final_scores)
    if method == "lociparse_pmol":
        return g["pmol"].to_numpy(float)
    if method == "grnade_lociparse_avg":
        zg = within_target_normalize(g["perplexity"], False, min_cands)
        zl = within_target_normalize(g["pmol"], True, min_cands)
        if zg.method != "zscore" or zl.method != "zscore":
            raise ValueError("z-score unavailable for this target (zero spread); cannot average")
        return (np.array(zg.final_scores) + np.array(zl.final_scores)) / 2
    raise KeyError(method)


def evaluate(df: pd.DataFrame, methods: List[str], k: int, min_cands: int) -> Tuple[pd.DataFrame, Dict]:
    """Per-target metrics for every method + pooled (score, tm) pairs for pooled Spearman."""
    rows, pooled = [], {m: ([], []) for m in methods}
    for tid, g in df.groupby("target_id", sort=True):
        if len(g) < min_cands:
            log.warning("[%s] only %d candidates with all scores; target skipped", tid, len(g))
            continue
        ids, tm = g["candidate_id"].tolist(), g["tm"].to_numpy(float)
        for m in methods:
            try:
                s = method_scores(g, m, min_cands)
            except ValueError as exc:
                log.warning("[%s] %s skipped: %s", tid, m, exc)
                continue
            met = random_metrics(tm, k) if s is None else target_metrics(s, tm, ids, k)
            rows.append({"target_id": tid, "method": m, **met})
            if s is not None:
                pooled[m][0].extend(s.tolist())
                pooled[m][1].extend(tm.tolist())
    return pd.DataFrame(rows), pooled


def summarise(per_target: pd.DataFrame, pooled: Dict, methods: List[str], subset: str) -> pd.DataFrame:
    out = []
    for m in methods:
        g = per_target[per_target["method"] == m]
        if g.empty:
            continue
        row = {"subset": subset, "method": m, "n_targets": len(g)}
        for met in METRICS:
            row[f"{met}_mean"] = float(np.nanmean(g[met])) if g[met].notna().any() else float("nan")
            row[f"{met}_median"] = float(np.nanmedian(g[met])) if g[met].notna().any() else float("nan")
        s, t = pooled.get(m, ([], []))
        row["pooled_spearman"] = spearman(s, t) if s else float("nan")
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plots (static PNGs for the report)
# ---------------------------------------------------------------------------
# Colours: the validated default categorical slots 1-2 from the dataviz reference
# palette (blue #2a78d6, orange #eb6834), on the light chart surface #fcfcfb,
# with text in the palette's ink colours. Thin marks, recessive grid, one y-axis.
BLUE, ORANGE, SURFACE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_scatter(df: pd.DataFrame, methods: List[str], min_cands: int, out_dir: Path) -> None:
    """One PNG per target: method score (x) vs true TM-score (y), one panel per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [m for m in ("grnade_norm", "lociparse_pmol", "grnade_lociparse_avg") if m in methods]
    labels = {"grnade_norm": "gRNAde score (−z perplexity)", "lociparse_pmol": "lociPARSE pMoL",
              "grnade_lociparse_avg": "Average of z-scores"}
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid, g in df.groupby("target_id"):
        if len(g) < min_cands or not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), squeeze=False,
                                 facecolor=SURFACE)
        for ax, m in zip(axes[0], panels):
            try:
                s = method_scores(g, m, min_cands)
            except ValueError:
                ax.set_visible(False)
                continue
            _style(ax)
            ax.scatter(s, g["tm"], s=18, color=BLUE, edgecolor=SURFACE, linewidth=0.8, label="candidate")
            i = int(np.argmax(s))
            ax.scatter([s[i]], [g["tm"].iloc[i]], s=60, facecolor="none", edgecolor=ORANGE, linewidth=2,
                       label="#1 pick")
            ax.set_xlabel(labels[m], color=INK, fontsize=9)
            ax.set_ylabel("TM-score vs native", color=INK, fontsize=9)
            ax.set_title(f"ρ = {spearman(s, g['tm']):.2f}", color=INK2, fontsize=9, loc="left")
            # Legend above the plot area (not inside it), so its markers can't be mistaken for data.
            ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                      borderaxespad=0.1, handletextpad=0.3, columnspacing=1.0, labelcolor=INK2)
        fig.suptitle(f"{tid}  ({len(g)} candidates)", color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{tid}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)


def plot_bars(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bars: mean top-1 and mean best-of-5 TM-score per method (subset 'all')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary[summary["subset"] == "all"]
    if s.empty:
        return
    x = np.arange(len(s))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.3 * len(s) + 2, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    b1 = ax.bar(x - w / 2 - 0.01, s["top1_tm_mean"], w, color=BLUE, label="Top-1 TM-score")
    b2 = ax.bar(x + w / 2 + 0.01, s["best5_tm_mean"], w, color=ORANGE, label="Best-of-5 TM-score")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK2, padding=2)
    short = {"oracle": "Oracle", "random": "Random", "grnade_raw": "gRNAde raw",
             "grnade_norm": "gRNAde norm. (ours)", "lociparse_pmol": "lociPARSE",
             "grnade_lociparse_avg": "gRNAde + lociPARSE"}
    ax.set_xticks(x, [short.get(m, m) for m in s["method"]], rotation=20, ha="right", fontsize=8, color=INK)
    ax.set_ylabel("Mean over targets", color=INK, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", labelcolor=INK2)
    ax.set_title("Top-1 and best-of-5 TM-score by method", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------

NAMES = {"oracle": "Oracle (best possible pick)", "random": "Random pick (expected)",
         "grnade_raw": "gRNAde raw perplexity", "grnade_norm": "gRNAde normalised (ours)",
         "lociparse_pmol": "lociPARSE (pMoL)", "grnade_lociparse_avg": "gRNAde + lociPARSE average"}


def _f(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def write_summary(path: Path, summary: pd.DataFrame, tests: pd.DataFrame, calib: Dict,
                  n_targets: Dict[str, int], excluded: List[str], overlap: List[str]) -> None:
    L = ["# Validation engine — evaluation summary", "",
         "This report was generated automatically by `src/evaluate.py`. Each RNA target has many predicted "
         "structures (candidates). Every method picks the candidates it believes are best, and we check those "
         "picks against the true quality: the TM-score against the experimental structure (1 = identical, "
         "above about 0.45 usually means the same overall fold).", "",
         f"* Targets evaluated: {n_targets.get('all', 0)} (all), {n_targets.get('no_overlap', 0)} "
         "(excluding possible gRNAde training overlap)",
         f"* Excluded RNA–protein complexes: {', '.join(excluded) if excluded else 'none listed'}",
         f"* Possible training-overlap targets: {', '.join(overlap) if overlap else 'none listed'}", ""]
    for subset in ("all", "no_overlap"):
        s = summary[summary["subset"] == subset]
        if s.empty:
            continue
        L += [f"## Results — subset `{subset}`", "",
              "| Method | Targets | Top-1 TM (mean / median) | Best-of-5 TM (mean / median) | "
              "Selection loss (mean) | Spearman (mean) | Pooled Spearman |",
              "|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            L.append(f"| {NAMES.get(r['method'], r['method'])} | {r['n_targets']} | "
                     f"{_f(r['top1_tm_mean'])} / {_f(r['top1_tm_median'])} | "
                     f"{_f(r['best5_tm_mean'])} / {_f(r['best5_tm_median'])} | "
                     f"{_f(r['selection_loss_mean'])} | {_f(r['spearman_mean'])} | {_f(r['pooled_spearman'])} |")
        L.append("")
    L += ["### How to read this", "",
          "* **Top-1 TM**: quality of the single structure the method would pick. Higher is better.",
          "* **Best-of-5 TM**: quality of the best structure among the 5 the method shows the user.",
          "* **Selection loss**: how much TM-score is lost compared with a perfect pick (0 = perfect).",
          "* **Spearman**: how well the method orders the candidates of one RNA (1 = perfect order, 0 = random).",
          "* **Pooled Spearman**: the same over all candidates of all RNAs together. This is where comparing "
          "raw perplexities across different RNAs hurts, and where normalisation should help.",
          "* **Raw vs normalised gRNAde:** inside one RNA, the z-score keeps the order of the perplexities, so the "
          "two give identical per-target picks by construction. The normalisation matters for pooled comparisons, "
          "for confidence, and for combining with lociPARSE.", ""]
    if not tests.empty:
        L += ["## Paired comparisons (Wilcoxon signed-rank on per-target top-1 TM-score)", "",
              "| Comparison | Subset | Targets | Mean difference | A better / B better / ties | p-value | Note |",
              "|---|---|---|---|---|---|---|"]
        for _, t in tests.iterrows():
            L.append(f"| {NAMES.get(t['method_a'], t['method_a'])} vs {NAMES.get(t['method_b'], t['method_b'])} | "
                     f"{t['subset']} | {t['n_targets']} | {_f(t['mean_diff'])} | "
                     f"{t['n_a_better']} / {t['n_b_better']} / {t['n_ties']} | {_f(t['p_value'], 4)} | {t['note']} |")
        L += ["", "A p-value below 0.05 means the difference is unlikely to be chance. With only about a dozen "
              "targets the test has little power, so a non-significant result does not show that the methods are equal.", ""]
    if calib:
        L += ["## Confidence calibration (leave-one-target-out)", "",
              f"* Truth label: TM-score of gRNAde's top pick ≥ {calib['tm_good']} → high, "
              f"≥ {calib['tm_ok']} → medium, otherwise low.",
              "* Predictor: raw perplexity of gRNAde's top pick.",
              f"* LOTO accuracy: **{_f(calib['loto_accuracy'])}** over {calib['n_targets']} targets "
              f"(each target was predicted with thresholds fitted on the other targets only).",
              f"* Final thresholds fitted on all targets: high if ppl ≤ {_f(calib['high_max_ppl'])}, "
              f"medium if ≤ {_f(calib['medium_max_ppl'])}, otherwise low "
              f"(training accuracy {_f(calib['train_accuracy'])}).",
              "* With so few targets these thresholds are rough. Treat the confidence label as a coarse guide.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def write_thresholds_to_config(config_path: Path, high: float, medium: float) -> None:
    """Replace only the two threshold values in config.yaml, keeping every comment."""
    text = Path(config_path).read_text(encoding="utf-8")
    for key, val in (("high_max_ppl", high), ("medium_max_ppl", medium)):
        text, n = re.subn(rf"^(\s*{key}:\s*)\S+", rf"\g<1>{val:.4f}", text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"{key} not found in {config_path}")
    Path(config_path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate ranking methods against US-align TM-scores.")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--labels", help="US-align labels CSV (default: usalign.cache_csv)")
    ap.add_argument("--grnade", help="gRNAde scores CSV (default: <processed_dir>/grnade_scores.csv)")
    ap.add_argument("--lociparse", help="lociPARSE scores CSV (default: <processed_dir>/lociparse_scores.csv)")
    ap.add_argument("--methods", nargs="+", help="subset of methods (default: evaluate.methods in config)")
    ap.add_argument("--write_config", action="store_true",
                    help="write the calibrated confidence thresholds into config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    processed = resolve_path(cfg, cfg["data"]["processed_dir"])
    labels = Path(args.labels) if args.labels else resolve_path(cfg, cfg["usalign"]["cache_csv"])
    grnade = Path(args.grnade) if args.grnade else processed / "grnade_scores.csv"
    loci = Path(args.lociparse) if args.lociparse else processed / "lociparse_scores.csv"
    methods = args.methods or list(cfg["evaluate"]["methods"])
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        ap.error(f"unknown methods: {sorted(unknown)}")
    if any("pmol" in NEEDS[m] for m in methods) and not loci.exists():
        dropped = [m for m in methods if "pmol" in NEEDS[m]]
        log.warning("lociPARSE scores not found (%s); dropping methods %s", loci, dropped)
        methods = [m for m in methods if m not in dropped]

    k = int(cfg["rank"]["top_k"])
    min_c = int(cfg["rank"]["min_candidates_for_zscore"])
    excluded = list(cfg["data"].get("excluded_rna_protein_targets") or [])
    overlap = list(cfg["data"].get("possible_training_overlap_targets") or [])

    df = load_data(labels, grnade, loci, methods)
    if excluded:
        log.info("Excluding RNA-protein targets: %s", excluded)
        df = df[~df["target_id"].isin(excluded)]
    duplicates = list(cfg["data"].get("excluded_duplicate_targets") or [])
    if duplicates:
        log.info("Excluding duplicate targets (same models as another target): %s", duplicates)
        df = df[~df["target_id"].isin(duplicates)]
    log.info("%d candidates over %d targets with labels and all scores", len(df), df["target_id"].nunique())

    out = cfg["output"]
    results_dir = resolve_path(cfg, out["results_dir"])
    summaries, per_targets, tests, n_targets = [], [], [], {}
    for subset, d in (("all", df), ("no_overlap", df[~df["target_id"].isin(overlap)])):
        pt, pooled = evaluate(d, methods, k, min_c)
        if pt.empty:
            continue
        pt.insert(0, "subset", subset)
        per_targets.append(pt)
        n_targets[subset] = pt["target_id"].nunique()
        summaries.append(summarise(pt, pooled, methods, subset))
        for a, b in cfg["evaluate"].get("paired_tests", []):
            if a in methods and b in methods:
                wide = pt.pivot(index="target_id", columns="method", values="top1_tm").dropna(subset=[a, b])
                tests.append({"subset": subset, "method_a": a, "method_b": b,
                              **paired_wilcoxon(wide[a], wide[b])})
    if not per_targets:
        log.error("Nothing to evaluate (no target with enough labelled, scored candidates).")
        return 2
    per_target = pd.concat(per_targets, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    tests_df = pd.DataFrame(tests)

    # Confidence calibration on subset "all", using gRNAde's top pick per target
    calib = {}
    if "grnade_norm" in methods:
        ccfg = cfg["confidence"]
        rows = []
        for tid, g in df.groupby("target_id", sort=True):
            if len(g) < min_c:
                continue
            s = method_scores(g, "grnade_norm", min_c)
            i = rank_order(list(s), g["candidate_id"].tolist())[0]
            rows.append((tid, float(g["perplexity"].iloc[i]), float(g["tm"].iloc[i])))
        if len(rows) >= 3:
            ppl = [r[1] for r in rows]
            truth = [truth_label(r[2], ccfg["calib_tm_good"], ccfg["calib_tm_ok"]) for r in rows]
            h, m, train_acc = fit_thresholds(ppl, truth)
            loto = loto_calibration(ppl, truth)
            calib = {"tm_good": ccfg["calib_tm_good"], "tm_ok": ccfg["calib_tm_ok"],
                     "n_targets": len(rows), "high_max_ppl": h, "medium_max_ppl": m,
                     "train_accuracy": train_acc, "loto_accuracy": loto["loto_accuracy"],
                     "per_target": [{"target_id": t, "top1_ppl": p, "top1_tm": tm, "truth": tr, "loto_pred": pr}
                                    for (t, p, tm), tr, pr in zip(rows, truth, loto["loto_predictions"])]}
            (results_dir / "confidence_thresholds.json").parent.mkdir(parents=True, exist_ok=True)
            (results_dir / "confidence_thresholds.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
            if args.write_config:
                write_thresholds_to_config(Path(args.config), h, m)
                log.info("Wrote thresholds to %s: high<=%.4f, medium<=%.4f", args.config, h, m)
        else:
            log.warning("Fewer than 3 targets: confidence calibration skipped")

    metrics_csv = resolve_path(cfg, out["metrics_csv"])
    per_target_csv = resolve_path(cfg, out["per_target_csv"])
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_csv, index=False)
    per_target.to_csv(per_target_csv, index=False)
    tests_df.to_csv(results_dir / "paired_tests.csv", index=False)
    plots = resolve_path(cfg, out["plots_dir"])
    plot_scatter(df, methods, min_c, plots)
    plot_bars(summary, plots / "bar_top1_best5.png")
    write_summary(resolve_path(cfg, out["summary_md"]), summary, tests_df, calib, n_targets, excluded, overlap)
    log.info("Wrote %s, %s, paired_tests.csv, plots and %s", metrics_csv, per_target_csv, out["summary_md"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
