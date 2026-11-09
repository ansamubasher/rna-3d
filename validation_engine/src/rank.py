"""
rank.py — Step 5: within-target normalisation, ranking, top-k and confidence.

WHY NORMALISE?
--------------
Raw gRNAde perplexity is NOT comparable across different RNAs: some sequences
are intrinsically easier to "explain" than others (length, GC content,
how common the fold is in the training data). Following ESM-IF (Hsu et al., ICML 2022),
which compares likelihoods relative to a reference on the SAME target, we
only ever compare a candidate with the OTHER candidates of the same target:
 
    z_i           = (ppl_i - mean(ppl of this target)) / std(ppl of this target)
    final_score_i = -z_i              (higher = better, because low perplexity = good)

Edge cases (as specified for the project):
    * fewer than `min_candidates_for_zscore` valid candidates, or std == 0
      -> fall back to final_score = -ppl and log a WARNING.

Note: inside ONE target, ranking by -z gives exactly the same ORDER as ranking
by -ppl (z is a shift-and-scale of ppl). The z-score matters when scores are
COMBINED or COMPARED across targets: the gRNAde+lociPARSE average in Step 8,
and confidence calibration.

The same function also works for "higher is better" scores (e.g. lociPARSE pMoL):
then final_score = +z, and the fallback is final_score = +value.

CONFIDENCE (low / medium / high)
--------------------------------
Based on the RAW perplexity of the top-ranked candidate (an absolute measure
of how well the best structure explains the sequence):
    ppl <= high_max_ppl                 -> "high"
    high_max_ppl < ppl <= medium_max_ppl  -> "medium"
    ppl > medium_max_ppl                -> "low"
The two thresholds are NOT chosen by hand: Step 8 (evaluate.py) derives them
with leave-one-target-out calibration and writes them to config.yaml. Until
then they are null, and the label is "uncalibrated".

Usage:
    python src/rank.py                         # ranks data/processed/grnade_scores.csv
    python src/rank.py --scores my_scores.csv --out results/ranked.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import load_config, resolve_path  # noqa: E402

log = logging.getLogger("rank")

RANKED_COLUMNS = ["target_id", "candidate_id", "rank", "final_score", "z_score",
                  "perplexity", "norm_method", "confidence", "status"]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@dataclass
class NormResult:
    final_scores: List[float]       # higher = better, same order as the input
    z_scores: List[Optional[float]] # None when the fallback was used
    method: str                     # "zscore" or "fallback_raw"
    reason: str = ""                # why the fallback was used (empty for zscore)


def within_target_normalize(values: Sequence[float], higher_is_better: bool = False,
                            min_candidates: int = 3, ddof: int = 0,
                            label: str = "") -> NormResult:
    """Z-score the scores of ONE target's valid candidates.

    values           : raw scores of the valid candidates (e.g. perplexities)
    higher_is_better : False for perplexity (lower = better), True for pMoL
    min_candidates   : below this many values, use the fallback
    ddof             : 0 = population std (stated explicitly in config.yaml)

    Returns final scores where HIGHER IS ALWAYS BETTER, whatever the input score.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("values must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite (remove failed candidates first)")
    sign = 1.0 if higher_is_better else -1.0
# THIS IS THE PART WHERE WE CHECK IF WE HAVE ENOUGH CANDIDATES TO CALCULATE Z SCORE, OTHERWISE WE FALL BACK TO RAW SCORE

# SO ITS LOWKEY JUST VALDIATION
    reason = ""
    if len(x) < min_candidates:
        reason = f"only {len(x)} valid candidates (< {min_candidates})"
    else:
        std = float(np.std(x, ddof=ddof))
        if std == 0.0 or not math.isfinite(std):
            reason = "std == 0 (all candidates have the same score)"

    if reason:
        log.warning("%s: z-score not used (%s); falling back to raw score", label or "target", reason)
        return NormResult(final_scores=[float(sign * v) for v in x],
                          z_scores=[None] * len(x), method="fallback_raw", reason=reason)



#CALCUALTES TEH Z SCORE

    z = (x - x.mean()) / np.std(x, ddof=ddof)
    return NormResult(final_scores=[float(sign * v) for v in z],
                      z_scores=[float(v) for v in z], method="zscore")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
# SORTS THE Z SCORE TO RANK
def rank_order(final_scores: Sequence[float], ids: Sequence[str]) -> List[int]:
    """Return indices sorted best-first.

    Ties (equal final_score) are broken by candidate id (alphabetical),
    so the ranking is the same on every run.
    """
    return sorted(range(len(final_scores)), key=lambda i: (-final_scores[i], ids[i]))

# ACTUALLYA DDA  LABEL USIGN THE UN NORALSIED PART 
def confidence_label(best_ppl: Optional[float], high_max_ppl: Optional[float],
                     medium_max_ppl: Optional[float]) -> str:
    """Map the top candidate's raw perplexity to low / medium / high.

    Returns "uncalibrated" until Step 8 has written both thresholds, and
    "none" if there is no valid candidate at all.
    """
    if best_ppl is None or not math.isfinite(best_ppl):
        return "none"
    if high_max_ppl is None or medium_max_ppl is None:
        return "uncalibrated"
    if high_max_ppl > medium_max_ppl:
        raise ValueError("config error: high_max_ppl must be <= medium_max_ppl")
    if best_ppl <= high_max_ppl:
        return "high"
    if best_ppl <= medium_max_ppl:
        return "medium"
    return "low"

# NOW TAKES ONE TARGE TAND ANAKS ITS CANDIDATES AND RANKS THEM BASED ON THE Z SCORE AND FALLBACKS IF NEEDED
def rank_target(candidates: List[dict], cfg: dict, target_id: str = "") -> dict:
    # THIS SI BASICALLYT EH FUCNTION TAHAT CALSLAS LL THE ABVOE FUNCTION
    """Rank the candidates of ONE target.

    candidates: list of dicts with at least candidate_id, status, perplexity
                (as written by score_grnade.py; status != "ok" = failed).
    Returns a dict with:
        ranked      : all candidates, best first, each with rank/final_score/z_score
                      (failed candidates come last with rank None)
        top_k       : the first top_k valid candidates
        norm_method : "zscore" or "fallback_raw"
        confidence  : low / medium / high / uncalibrated / none
    """
    rcfg, ccfg = cfg["rank"], cfg["confidence"]
    top_k = int(rcfg["top_k"])
# GETS EH VALIDE ONES
    valid = [c for c in candidates
             if c.get("status") == "ok" and c.get("perplexity") not in (None, "")
             and math.isfinite(float(c["perplexity"]))]
    failed = [c for c in candidates if c not in valid]
# RANKS THEM 
    ranked: List[dict] = []
    norm_method, best_ppl = "none", None
    # IF VALID NTOAMLSIES THEM AND RANKS THEM, IF NOT FALLS BACK TO RAW SCORE
    if valid:
        ppl = [float(c["perplexity"]) for c in valid]
        ids = [c["candidate_id"] for c in valid]
        norm = within_target_normalize(ppl, higher_is_better=False,
                                       min_candidates=int(rcfg["min_candidates_for_zscore"]),
                                       ddof=int(rcfg.get("std_ddof", 0)), label=target_id)
        # TH PART EHRET HEY ARE RNAKIED
        norm_method = norm.method
        for r, i in enumerate(rank_order(norm.final_scores, ids), start=1):
            ranked.append({**valid[i], "rank": r, "final_score": norm.final_scores[i],
                           "z_score": norm.z_scores[i], "perplexity": ppl[i],
                           "norm_method": norm.method})
        best_ppl = ranked[0]["perplexity"]
# THIS IS THE PART WHERE WE ADD THE FAILED ONES TO THE END OF THE RANKED LIST WITH NONE VALUES FOR RANK AND SCORES
    for c in sorted(failed, key=lambda c: c["candidate_id"]):
        ranked.append({**c, "rank": None, "final_score": None, "z_score": None,
                       "norm_method": norm_method})
# GIVES HTEM A CONFIDENT CLAVE; PBASED ON THE PERPLEXITY
    conf = confidence_label(best_ppl, ccfg.get("high_max_ppl"), ccfg.get("medium_max_ppl"))
    return {"target_id": target_id, "ranked": ranked,
            "top_k": [c for c in ranked if c["rank"] is not None][:top_k],
            "norm_method": norm_method, "confidence": conf,
            "n_valid": len(valid), "n_failed": len(failed)}


# ---------------------------------------------------------------------------
# CLI: rank every target in a scores CSV
# ---------------------------------------------------------------------------

def _fmt(v, nd=6):
    return "" if v is None else (round(v, nd) if isinstance(v, float) else v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Within-target z-score ranking of gRNAde scores.")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--scores", help="scores CSV from score_grnade.py "
                                     "(default: <processed_dir>/grnade_scores.csv)")
    ap.add_argument("--out", help="ranked CSV (default: <results_dir>/ranked_grnade.csv)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    scores = Path(args.scores) if args.scores else resolve_path(cfg, cfg["data"]["processed_dir"]) / "grnade_scores.csv"
    out = Path(args.out) if args.out else resolve_path(cfg, cfg["output"]["results_dir"]) / "ranked_grnade.csv"

    by_target: Dict[str, List[dict]] = defaultdict(list)
    with open(scores, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_target[row["target_id"]].append(row)

    rows = []
    for tid in sorted(by_target):
        res = rank_target(by_target[tid], cfg, tid)
        log.info("[%s] %d valid, %d failed, method=%s, best ppl=%s, confidence=%s",
                 tid, res["n_valid"], res["n_failed"], res["norm_method"],
                 _fmt(res["ranked"][0]["perplexity"] if res["n_valid"] else None, 3), res["confidence"])
        for c in res["ranked"]:
            rows.append({"target_id": tid, "candidate_id": c["candidate_id"], "rank": _fmt(c["rank"]),
                         "final_score": _fmt(c["final_score"]), "z_score": _fmt(c["z_score"]),
                         "perplexity": _fmt(c["perplexity"] if c["rank"] is not None else None),
                         "norm_method": c["norm_method"], "confidence": res["confidence"],
                         "status": c.get("status", "")})

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RANKED_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log.info("Ranked %d targets -> %s", len(by_target), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
