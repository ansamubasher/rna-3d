"""
label_usalign.py — Step 7 (EVALUATION ONLY): ground-truth quality of each candidate.

For every candidate we compare its 3D structure with the experimental (native)
structure using US-align (Zhang et al., Nature Methods 2022) and record:
    * TM-score normalised by the NATIVE length  (0-1, higher = better; this is
      the "true quality" label used in Step 8)
    * TM-score normalised by the model length    (kept for reference)
    * RMSD over the aligned residues, and aligned length
The validation engine itself (validate.py) never uses these numbers. They are
only used to measure how good its ranking is.

WHAT IS ALIGNED
---------------
We align the CLEAN PDBs written by prepare.py, not the raw downloads:
    model  = clean candidate PDB  (one model, one chain, standard A/C/G/U)
    native = clean native PDB     (the experimental structure, prepared with
                                   prepare.py against the same target sequence)
Reasons:
  * native Windows US-align cannot read .gz files (US-align readme.txt);
  * CASP files can hold several MODELs, and US-align's default `-ter 2`
    aligns only the FIRST chain. The clean files have exactly one chain and
    one model, so there is no ambiguity about what was compared;
  * both files contain the same kind of residues (no ligands, ions or water).

COMMAND (from config.yaml: usalign_bin + usalign_args)
    USalign.exe model.pdb native.pdb -mol RNA -outfmt 2
`-outfmt 2` prints one tab-separated line, which is far easier to parse
reliably than the full text output. Its columns, read from US-align's source
(TMalign.h, output_results(), outfmt_opt==2, commit 97325d3):
    PDBchain1  PDBchain2  TM1  TM2  RMSD  ID1  ID2  IDali  L1  L2  Lali
where TM1 is normalised by Structure_1 (the model) and TM2 by Structure_2
(the native). Because the native is passed SECOND, TM2 is the
"normalised by the reference" TM-score we report.

Inputs:
    --manifest          candidate manifest from prepare.py (default: data.manifest_csv)
    --natives_manifest  manifest from running prepare.py on the native structures;
                        the first status=ok row per target_id is used as that target's native
Output / cache: usalign.cache_csv (re-runs only compute missing pairs).

Usage:
    python src/label_usalign.py --natives_manifest data/processed/natives_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import load_config, resolve_path  # noqa: E402

log = logging.getLogger("label_usalign")

LABEL_COLUMNS = ["target_id", "candidate_id", "model_pdb", "native_pdb", "usalign_args",
                 "tm_native", "tm_model", "rmsd", "aligned_length",
                 "len_model", "len_native", "seq_id_aligned", "status", "error"]

# Column names of US-align's -outfmt 2 header line (USalign.cpp, commit 97325d3)
OUTFMT2_HEADER = ["PDBchain1", "PDBchain2", "TM1", "TM2", "RMSD",
                  "ID1", "ID2", "IDali", "L1", "L2", "Lali"]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_outfmt2(stdout: str) -> Dict[str, float]:
    """Parse US-align `-outfmt 2` output (one header line starting with '#', one result line).

    Returns tm_model, tm_native, rmsd, seq_id_aligned, len_model, len_native,
    aligned_length. Raises ValueError if the output doesn't look as expected,
    so a format change is never silently mis-read.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    header = [ln for ln in lines if ln.startswith("#")]
    data = [ln for ln in lines if not ln.startswith("#")]
    if header:
        cols = header[0].lstrip("#").split("\t")
        if cols[:len(OUTFMT2_HEADER)] != OUTFMT2_HEADER:
            raise ValueError(f"unexpected US-align header: {header[0]!r}")
    if len(data) != 1:
        raise ValueError(f"expected exactly 1 result line from US-align, got {len(data)}: {data[:3]!r}")
    f = data[0].split("\t")
    if len(f) < 11:
        raise ValueError(f"expected 11 tab-separated fields, got {len(f)}: {data[0]!r}")
    out = {
        "tm_model": float(f[2]),        # TM1: normalised by Structure_1 = model
        "tm_native": float(f[3]),       # TM2: normalised by Structure_2 = native (REPORTED)
        "rmsd": float(f[4]),
        "seq_id_aligned": float(f[7]),
        "len_model": int(f[8]),
        "len_native": int(f[9]),
        "aligned_length": int(f[10]),
    }
    for k in ("tm_model", "tm_native"):
        if not 0.0 <= out[k] <= 1.0:
            raise ValueError(f"{k}={out[k]} outside [0, 1]")
    return out


# ---------------------------------------------------------------------------
# Running US-align
# ---------------------------------------------------------------------------

def run_usalign(usalign_bin: str, model_pdb: str, native_pdb: str,
                extra_args: List[str], timeout: int) -> Dict[str, object]:
    """Run US-align on one pair and return parsed numbers (or status=error)."""
    # Drop any user-given "-outfmt <n>" pair: the parser below needs -outfmt 2
    args, skip = [], False
    for a in extra_args:
        if skip:
            skip = False
            continue
        if a == "-outfmt":
            log.warning("-outfmt in usalign_args is ignored; we always use -outfmt 2")
            skip = True
            continue
        args.append(a)
    cmd = [usalign_bin, model_pdb, native_pdb, *args, "-outfmt", "2"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"timeout after {timeout}s"}
    except OSError as exc:  # e.g. binary not found
        return {"status": "error", "error": f"could not run US-align: {exc}"}
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        return {"status": "error", "error": f"exit {proc.returncode}: {msg[-1] if msg else ''}"}
    try:
        return {"status": "ok", "error": "", **parse_outfmt2(proc.stdout)}
    except ValueError as exc:
        return {"status": "error", "error": f"parse error: {exc}"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _key(row: dict) -> tuple:
    """A cached result is valid only for the same files and the same US-align options."""
    return (row["target_id"], row["candidate_id"], row["model_pdb"],
            row["native_pdb"], row["usalign_args"])


def load_cache(path: Path) -> Dict[tuple, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {_key(r): r for r in csv.DictReader(fh) if r.get("status") == "ok"}


def write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def read_ok_rows(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["status"] == "ok"]


def pick_natives(native_rows: List[dict]) -> Dict[str, str]:
    """One native clean PDB per target: the first accepted row of that target."""
    natives: Dict[str, str] = {}
    for r in native_rows:
        if r["target_id"] in natives:
            log.warning("[%s] several native structures; using %s, ignoring %s",
                        r["target_id"], natives[r["target_id"]], r["clean_pdb"])
            continue
        natives[r["target_id"]] = r["clean_pdb"]
    return natives


def label_all(cand_rows: List[dict], natives: Dict[str, str], cfg: dict,
              cache_path: Path, workers: int = 4, force: bool = False) -> List[dict]:
    ucfg = cfg["usalign"]
    usalign_bin = str(resolve_path(cfg, cfg["tools"]["usalign_bin"]))
    extra = [str(a) for a in cfg["tools"].get("usalign_args", ["-mol", "RNA"])]
    args_str = " ".join(extra)
    timeout = int(ucfg.get("timeout_sec", 300))
    cache = {} if force else load_cache(cache_path)

    rows, todo = [], []
    for c in cand_rows:
        row = {k: "" for k in LABEL_COLUMNS}
        row.update(target_id=c["target_id"], candidate_id=c["candidate_id"],
                   model_pdb=c["clean_pdb"], native_pdb=natives.get(c["target_id"], ""),
                   usalign_args=args_str)
        if not row["native_pdb"]:
            row.update(status="error", error="no native structure for this target")
        elif _key(row) in cache:
            row = cache[_key(row)]
        else:
            todo.append(row)
        rows.append(row)

    n_no_native = sum(r["status"] == "error" for r in rows)
    log.info("%d pairs: %d cached, %d to compute, %d without native",
             len(rows), len(rows) - len(todo) - n_no_native, len(todo), n_no_native)
    if todo and not Path(usalign_bin).exists():
        raise FileNotFoundError(f"US-align binary not found: {usalign_bin} (see environment.md section 3)")

    def work(row: dict) -> None:
        row.update(run_usalign(usalign_bin, row["model_pdb"], row["native_pdb"], extra, timeout))

    # US-align runs as a separate process, so a thread pool is enough to run several at once
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, _ in enumerate(pool.map(work, todo), start=1):
            if i % 100 == 0:
                log.info("US-align: %d/%d done", i, len(todo))
                write_rows(cache_path, rows)   # save progress in case the run is interrupted

    for r in rows:
        if r["status"] == "error":
            log.warning("[%s] %s: %s", r["target_id"], r["candidate_id"], r["error"])
    write_rows(cache_path, rows)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TM-score / RMSD of every candidate vs its native (US-align).")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--manifest", help="candidate manifest (default: data.manifest_csv)")
    ap.add_argument("--natives_manifest", required=True,
                    help="manifest from running prepare.py on the native structures")
    ap.add_argument("--out", help="labels CSV / cache (default: usalign.cache_csv)")
    ap.add_argument("--workers", type=int, default=4, help="US-align processes in parallel")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log_dir = resolve_path(cfg, cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "label_usalign.log", encoding="utf-8")],
    )

    manifest = Path(args.manifest) if args.manifest else resolve_path(cfg, cfg["data"]["manifest_csv"])
    out = Path(args.out) if args.out else resolve_path(cfg, cfg["usalign"]["cache_csv"])
    natives = pick_natives(read_ok_rows(Path(args.natives_manifest)))
    rows = label_all(read_ok_rows(manifest), natives, cfg, out, args.workers, args.force)

    n_ok = sum(r["status"] == "ok" for r in rows)
    log.info("Labels written: %s (%d ok, %d errors)", out, n_ok, len(rows) - n_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
