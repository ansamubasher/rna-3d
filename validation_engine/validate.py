"""
validate.py — Step 6: the public entry point of the validation engine.

Give it ONE RNA sequence and a folder of candidate 3D structures from ANY
predictor (RNAPro, DRfold2, RhoFold+, ...). It scores every candidate with
gRNAde, ranks them with the within-target z-score, and writes results.json
for the web app. No ground truth is needed.

    python validate.py --sequence <seq or fasta> --candidates <dir> --top_k 5 --out results.json

Pipeline (each step reuses the module that implements it):
    1. src/prepare.py       prepare_target()   parse, split models, map residues, write clean PDBs
    2. src/score_grnade.py  GrnadeScorer       teacher-forced perplexity of the sequence per candidate
    3. src/rank.py          rank_target()      z-score within this target, rank, top-k, confidence
    4. this file            build_results()    assemble results.json

results.json contains:
    ranked      : every valid candidate, best first (file, rank, final_score, z_score, perplexity)
    top_k       : the first k of `ranked`
    confidence  : low / medium / high ("uncalibrated" until Step 8 sets the thresholds)
    best_candidate_per_nucleotide : p(true nucleotide) per residue of the #1 candidate,
                  with its ORIGINAL chain and residue number, for colouring the 3D viewer
    skipped     : candidates rejected by preprocessing or scoring, with the reason
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from prepare import load_config, prepare_target, read_sequence_arg, write_manifest  # noqa: E402
from rank import rank_target                                                         # noqa: E402

log = logging.getLogger("validate")


def _num(v) -> Optional[float]:
    """CSV/JSON values arrive as strings or numbers; return a float or None."""
    if v is None or v == "":
        return None
    return float(v)

# THIS IS THE PART WHERE WE BUILD THE RESULTS.JSON FILE FROM THE OUTPUTS OF THE THREE STEPS
# IT SI BACSICALLY COLLECTIIGN THE RESULTS FROM THE PREPARE, SCORE AND RANKING STEPS AND PUTTING THEM INTO A DICTIONARY
def build_results(target_id: str, sequence: str, prep_rows: List[dict],
                  score_rows: List[dict], ranking: dict, cfg: dict,
                  checkpoint: str) -> dict:
    """Assemble the results.json dictionary from the outputs of the three steps."""
    prep_by_id = {r["candidate_id"]: r for r in prep_rows}

    ranked = []
    for c in ranking["ranked"]:
        if c["rank"] is None:
            continue

        # GETS THSI FROM PAPREP.PY
        p = prep_by_id.get(c["candidate_id"], {})
        ranked.append({
            "rank": c["rank"],
            "candidate_id": c["candidate_id"],
            "candidate_file": p.get("source_file", ""),
            "model_index": int(p["model_index"]) if p.get("model_index") not in (None, "") else None,
            "final_score": _num(c["final_score"]),
            "z_score": _num(c["z_score"]),
            "perplexity": _num(c["perplexity"]),
            "mean_nll": _num(c.get("mean_nll")),
            "n_scored": int(c["n_scored"]) if c.get("n_scored") not in (None, "") else None,
            "coverage": _num(p.get("coverage")),
            "flags": [f for f in (p.get("flags", "").split(";") + str(c.get("flags", "")).split(";")) if f],
        })

    # Everything that did not make it into the ranking, with the reason
    skipped = [{"candidate_id": r["candidate_id"], "candidate_file": r["source_file"],
                "stage": "prepare", "reason": r["skip_reason"]}
               for r in prep_rows if r["status"] != "ok"]
    skipped += [{"candidate_id": s["candidate_id"],
                 "candidate_file": prep_by_id.get(s["candidate_id"], {}).get("source_file", ""),
                 "stage": "score_grnade", "reason": s["error"]}
                for s in score_rows if s["status"] != "ok"]
# RNAKS IT CLALS AOLF CUNTIONS
    # Per-nucleotide scores of the best candidate, read from its score cache JSON
    per_nt = None
    if ranked:
        best = ranked[0]
        cache = next(s["cache_json"] for s in score_rows if s["candidate_id"] == best["candidate_id"])
        rec = json.loads(Path(cache).read_text(encoding="utf-8"))
        per_nt = {
            "candidate_id": best["candidate_id"],
            "candidate_file": best["candidate_file"],
            "model_index": best["model_index"],
            "score_meaning": "probability gRNAde gives to the true nucleotide (0-1, higher = fits better; 0.25 = random)",
            "residues": [
                {"target_position": ti + 1, "target_nt": nt, "chain": ch, "res_id": rid,
                 "p_true": None if p is None else round(p, 4)}
                for ti, nt, ch, rid, p in zip(rec["target_index"], rec["target_nt"],
                                               rec["orig_chain"], rec["orig_res_id"], rec["p_true"])
            ],
        }

    ccfg = cfg["confidence"]
    warnings = []
    if ranking["norm_method"] == "fallback_raw":
        warnings.append("z-score normalisation not used (too few valid candidates or zero spread); "
                        "ranked by raw perplexity instead")
    if ranking["confidence"] == "uncalibrated":
        warnings.append("confidence thresholds not calibrated yet (Step 8); label is 'uncalibrated'")
    if not ranked:
        warnings.append("no valid candidates; see 'skipped'")

    k = int(cfg["rank"]["top_k"])
    return {
        "engine": "gRNAde validation engine (teacher-forced perplexity, within-target z-score)",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "target_id": target_id,
        "sequence": sequence,
        "length": len(sequence),
        "counts": {
            "candidates_found": len(prep_rows),
            "valid_ranked": len(ranked),
            "skipped": len(skipped),
        },
        "normalization": ranking["norm_method"],
        "confidence": ranking["confidence"],
        "confidence_basis": {
            "metric": ccfg.get("metric", "best_candidate_ppl"),
            "value": ranked[0]["perplexity"] if ranked else None,
            "high_max_ppl": ccfg.get("high_max_ppl"),
            "medium_max_ppl": ccfg.get("medium_max_ppl"),
        },
        "top_k": ranked[:k],
        "ranked": ranked,
        "best_candidate_per_nucleotide": per_nt,
        "skipped": skipped,
        "warnings": warnings,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rank candidate RNA 3D structures with gRNAde.")
    ap.add_argument("--sequence", required=True, help="RNA sequence, or path to a FASTA file")
    ap.add_argument("--candidates", required=True, help="folder of candidate PDB/mmCIF files")
    ap.add_argument("--top_k", type=int, help="how many to return (default: rank.top_k in config)")
    ap.add_argument("--out", default="results.json", help="output JSON path")
    ap.add_argument("--target_id", default="query", help="name used in outputs and file names")
    ap.add_argument("--work_dir", help="where clean PDBs and score caches go "
                                       "(default: <out>_work next to the output)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args(argv)
#LOAD STHE CONFIG
    cfg = load_config(args.config)
    if args.top_k is not None:
        if args.top_k < 1:
            ap.error("--top_k must be >= 1")
        cfg["rank"]["top_k"] = args.top_k
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out = Path(args.out).resolve()
    work = Path(args.work_dir).resolve() if args.work_dir else out.with_name(out.stem + "_work")
    cand_dir = Path(args.candidates)
    if not cand_dir.is_dir():
        ap.error(f"candidates folder not found: {cand_dir}")
    sequence = read_sequence_arg(args.sequence)
    log.info("Target %s: %d nt, candidates from %s", args.target_id, len(sequence), cand_dir)
# CALLS PREPARE .PY FUCNTIONS
    # 1. Preprocess
    prep_rows = prepare_target(args.target_id, sequence, cand_dir, work / "clean", cfg)
    write_manifest(prep_rows, work / "manifest.csv")
    ok_rows = [r for r in prep_rows if r["status"] == "ok"]
# CLALSS COREERR GUN
    # 2. Score with gRNAde (imported here: needs torch + the gRNAde repo)
    score_rows: List[dict] = []
    checkpoint = ""
    if ok_rows:
        from score_grnade import GrnadeScorer, score_candidate, set_seeds
        set_seeds(int(cfg["seed"]))
        scorer = GrnadeScorer(cfg)
        checkpoint = scorer.ckpt_path
        for r in ok_rows:
            score_rows.append(score_candidate(scorer, r, work / "scores"))
#RTANKS
    # 3. Rank this target's scored candidates
    ranking = rank_target(score_rows, cfg, args.target_id)
#BUILD REUSLTS
    # 4. Write results.json
    results = build_results(args.target_id, sequence, prep_rows, score_rows,
                            ranking, cfg, checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if results["ranked"]:
        b = results["ranked"][0]
        log.info("Best: %s (ppl %.3f, z %s) | confidence: %s | %d ranked, %d skipped -> %s",
                 b["candidate_id"], b["perplexity"],
                 "n/a" if b["z_score"] is None else f"{b['z_score']:.2f}",
                 results["confidence"], len(results["ranked"]), len(results["skipped"]), out)
        return 0
    log.error("No valid candidates. See 'skipped' in %s", out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
