"""
score_lociparse.py — Step 8 baseline: score every candidate with lociPARSE.

lociPARSE (Tarafder & Bhattacharya, J. Chem. Inf. Model. 2024) is a structure-
quality model for RNA. For each nucleotide it predicts pNuL (predicted
nucleotide-wise lDDT, 0-1). Its global score pMoL is the mean of the pNuL values.
Higher = better. It is the main baseline that gRNAde is compared against.

TWO ENVIRONMENTS
----------------
lociPARSE needs torch 1.12, gRNAde needs torch 2.1, so lociPARSE lives in its own
conda env (environment.md, section 2). This file therefore has two modes:
  * driver mode (default, any env): reads the manifest, works out which candidates
    are not cached yet, and launches the WORKER with the lociparse env's Python
    (`tools.lociparse_python` in config.yaml). Then it collects the results into a CSV.
  * worker mode (--worker jobs.json, runs inside the lociparse env): loads the
    lociPARSE model ONCE and scores every job. It imports only the standard
    library, numpy, torch and lociPARSE (the lociparse env has no PyYAML,
    biotite, ...).

WHAT WE CALL IN lociPARSE (read from lociPARSE/core.py, commit 8c7acbe)
----------------------------------------------------------------------
`lociparse().score(path)` = genFeat(path) -> model(V, E, G, X) -> sigmoid -> pNuL,
pMoL = mean(pNuL). It then ROUNDS both to 2 decimals (classes pMoL / pNuL in core.py).
With ~200 candidates per target, 2-decimal pMoL gives many ties, so the worker
runs the SAME steps as `lociparse.score` (the same model object, the same
extract_features, the same sigmoid) without the rounding. We store both the
unrounded pMoL (used for ranking) and the official rounded value (for reference).

INPUT FILE FORMAT FOR lociPARSE
-------------------------------
lociPARSE's PDB reader (feature_generation.py) is line-based and fragile:
  * it stops at the first EMPTY line (`while (linef := readline().rstrip())`);
  * it finishes the last residue only when it meets a final `TER` line.
So for each candidate we write a minimal copy of prepare.py's clean PDB:
ATOM records only, then one `TER` line, and nothing after it. The clean PDB
already has one chain and residues numbered 1..L with no insertion codes,
which is what its parser expects.

Outputs:
  <processed_dir>/lociparse_scores/<target>/<candidate>.json   (cache: pMoL + pNuL)
  <processed_dir>/lociparse_scores.csv                          (one row per candidate)

Usage:
  python src/score_lociparse.py                 # driver; launches the lociparse env itself
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("score_lociparse")

SCORE_COLUMNS = ["target_id", "candidate_id", "status", "error", "n_residues",
                 "pmol", "pmol_official", "cache_json"]


# ===========================================================================
# WORKER MODE (runs inside the lociparse env)
# ===========================================================================

def write_lociparse_input(clean_pdb: str, out_path: str) -> int:
    """Copy only the ATOM records, then a final TER line (see module docstring).
    Returns the number of residues written."""
    residues = set()
    with open(clean_pdb, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("ATOM  "):
                fout.write(line.rstrip("\r\n") + "\n")
                residues.add(line[21:27])  # chain + residue number + insertion code
        fout.write("TER\n")
    return len(residues)


def score_one(lp, clean_pdb: str) -> dict:
    """Same steps as lociPARSE core.py `lociparse.score`, minus the 2-decimal rounding."""
    import numpy as np
    import torch
    from torch import nn

    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "model.pdb")
        n_res = write_lociparse_input(clean_pdb, inp)
        V, E, G, X = lp.extract_features(inp)                       # lociPARSE genFeat
        V = V.unsqueeze(0).to(torch.float32).to(lp.device)
        E = E.unsqueeze(0).to(torch.float32).to(lp.device)
        G = G.unsqueeze(0).to(lp.device)
        X = X.unsqueeze(0).to(torch.float32).to(lp.device)
        with torch.no_grad():
            pred = lp.model(V, E, G, X)
            pnul = nn.Sigmoid()(pred.squeeze()).detach().cpu().numpy()
    pnul = np.atleast_1d(pnul).astype(float)
    if len(pnul) != n_res:
        raise ValueError(f"lociPARSE returned {len(pnul)} scores for {n_res} residues")
    pmol = float(np.mean(pnul))
    return {"n_residues": int(n_res), "pmol": pmol,
            "pmol_official": float(f"{pmol:.2f}"),   # what lociPARSE itself would report
            "pnul": [float(v) for v in pnul]}


def worker(jobs_path: str) -> int:
    """Score every job in jobs.json: [{"clean_pdb":..., "map_json":..., "out_json":...}, ...]."""
    from lociPARSE import lociparse

    jobs = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
    lp = lociparse()   # loads QAmodel_lociPARSE.pt shipped with the package
    print(f"[lociPARSE worker] {len(jobs)} jobs on {lp.device}", flush=True)
    for i, job in enumerate(jobs, 1):
        try:
            res = score_one(lp, job["clean_pdb"])
            mapping = json.loads(Path(job["map_json"]).read_text(encoding="utf-8"))
            if len(mapping["target_index"]) != res["n_residues"]:
                raise ValueError("residue count differs from map.json")
            rec = {"status": "ok", "error": "", "clean_pdb": job["clean_pdb"],
                   "target_index": mapping["target_index"], **res}
        except Exception as exc:  # report and continue with the next job
            rec = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "clean_pdb": job["clean_pdb"]}
        Path(job["out_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["out_json"]).write_text(json.dumps(rec), encoding="utf-8")
        if i % 50 == 0:
            print(f"[lociPARSE worker] {i}/{len(jobs)}", flush=True)
    return 0


# ===========================================================================
# DRIVER MODE (any env with PyYAML; the rna env is fine)
# ===========================================================================

def driver(argv) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare import load_config, resolve_path

    ap = argparse.ArgumentParser(description="Score candidates with lociPARSE (baseline).")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--manifest", help="manifest CSV from prepare.py (default: data.manifest_csv)")
    ap.add_argument("--out_csv", help="default: <processed_dir>/lociparse_scores.csv")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log_dir = resolve_path(cfg, cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "score_lociparse.log", encoding="utf-8")],
    )

    processed = resolve_path(cfg, cfg["data"]["processed_dir"])
    manifest = Path(args.manifest) if args.manifest else resolve_path(cfg, cfg["data"]["manifest_csv"])
    out_csv = Path(args.out_csv) if args.out_csv else processed / "lociparse_scores.csv"
    cache_dir = processed / "lociparse_scores"

    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "ok"]

    def cache_path(r):
        return cache_dir / r["target_id"] / f"{r['candidate_id']}.json"

    def cached_ok(r):
        p = cache_path(r)
        if args.force or not p.exists():
            return False
        rec = json.loads(p.read_text(encoding="utf-8"))
        return rec.get("status") == "ok" and rec.get("clean_pdb") == r["clean_pdb"]

    todo = [r for r in rows if not cached_ok(r)]
    log.info("%d candidates: %d cached, %d to score", len(rows), len(rows) - len(todo), len(todo))

    if todo:
        py = str(resolve_path(cfg, cfg["tools"]["lociparse_python"]))
        if not Path(py).exists():
            raise FileNotFoundError(f"lociparse env Python not found: {py} "
                                    "(set tools.lociparse_python in config.yaml; environment.md section 2)")
        jobs = [{"clean_pdb": r["clean_pdb"], "map_json": r["map_json"],
                 "out_json": str(cache_path(r))} for r in todo]
        cache_dir.mkdir(parents=True, exist_ok=True)
        jobs_file = cache_dir / "_jobs.json"
        jobs_file.write_text(json.dumps(jobs), encoding="utf-8")
        # -I (isolated mode): don't import modules from the current folder by accident
        cmd = [py, "-I", str(Path(__file__).resolve()), "--worker", str(jobs_file)]
        log.info("Running lociPARSE worker: %s", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            log.error("lociPARSE worker exited with code %d (results so far are cached)", proc.returncode)

    out_rows = []
    for r in rows:
        p = cache_path(r)
        row = {c: "" for c in SCORE_COLUMNS}
        row.update(target_id=r["target_id"], candidate_id=r["candidate_id"], cache_json=str(p))
        if not p.exists():
            row.update(status="error", error="no lociPARSE result (worker did not reach it)")
        else:
            rec = json.loads(p.read_text(encoding="utf-8"))
            row.update(status=rec["status"], error=rec.get("error", ""))
            if rec["status"] == "ok":
                row.update(n_residues=rec["n_residues"], pmol=round(rec["pmol"], 6),
                           pmol_official=rec["pmol_official"])
        if row["status"] != "ok":
            log.warning("[%s] %s: %s", r["target_id"], r["candidate_id"], row["error"])
        out_rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    n_ok = sum(r["status"] == "ok" for r in out_rows)
    log.info("lociPARSE scores written: %s (%d ok, %d errors)", out_csv, n_ok, len(out_rows) - n_ok)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2]))
    sys.exit(driver(sys.argv[1:]))
