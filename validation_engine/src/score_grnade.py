"""
score_grnade.py — Step 4: score each candidate structure with gRNAde.

IDEA (inverse folding as a validator)
-------------------------------------
gRNAde (Joshi et al., ICLR 2025) is an RNA inverse-folding model: given a 3D
backbone, it predicts a probability distribution over A/G/C/U at each
position. If a candidate structure is correct, the TRUE target sequence
should "fit" it well. We measure that fit with the TEACHER-FORCED likelihood
of the target sequence on the candidate backbone:

    NLL_i      = -log p(target_nt_i | backbone, target_nt_<i)   per position i
    mean NLL   = average of NLL_i over scored positions
    perplexity = exp(mean NLL)        (4.0 = random guessing over 4 bases)

Lower perplexity means the backbone explains the sequence better, so the
candidate is more likely to be correct.

WHY NOT USE gRNAde's design FUNCTIONS?
--------------------------------------
`design.py` and `src/evaluator.py::evaluate` call `model.sample(...)` and
report the perplexity of the sequences that gRNAde itself SAMPLES
(evaluator.py, the "perplexity per sample" block: cross_entropy(logits, samples)).
That measures how confident the model is in its own designs, NOT how well
OUR target sequence fits the backbone.

WHAT WE USE INSTEAD (all read from gRNAde commit 2453b18)
---------------------------------------------------------
* `src/models.py::gRNAde.forward(batch)`: the training/likelihood forward
  pass. It encodes the backbone graph and then runs the autoregressive decoder
  with the TRUE sequence `batch.seq` as decoder input (teacher forcing). The
  decoder only lets node i see sequence information from nodes j < i: see
  `h_S[edge_index[0] >= edge_index[1]] = 0`. It returns logits of shape
  [n_nodes, 4].
* `src/trainer.py::loop`: gRNAde's own validation/test loss is
  `CrossEntropyLoss(label_smoothing=0.0)(logits[mask_seq], seq[mask_seq])`
  and the logged perplexity is exp(loss). We compute exactly that quantity,
  per position (log_softmax + gather) so that we can also save the per-
  nucleotide probabilities.
* `src/data/data_utils.py::pdb_to_tensor` + `get_backbone_coords`: gRNAde's
  own PDB parsing and 3-bead backbone (P, C4', N1 for pyrimidines / N9 for
  purines). We call them the same way `RNAGraphFeaturizer.featurize_from_pdb_file`
  does: secondary structure from the 3D coordinates (biotite) with
  pseudoknots kept, and no SASA.
* `src/data/featurizer.py::RNAGraphFeaturizer(split="test", ...)`: builds
  the graph. With split="test", no coordinate noise and no 3D dropout are
  applied (those happen only when "train" is in the split name).
  Hyper-parameters (top_k, num_rbf, ...) come from the checkpoint's own config.yaml.
* Model construction and `load_state_dict(torch.load(...))` exactly as in
  `design.py::main`.

TWO DETAILS THAT MATTER
-----------------------
1. The clean PDB from prepare.py carries the CANDIDATE's nucleotides, so
   `get_backbone_coords` picks the base atom (N1/N9) that is really present.
   The sequence we put into the graph (`rna["sequence"]`) is the TARGET
   sequence at those positions, taken from the .map.json. Usually the two are
   identical; they differ only at the few mismatch positions prepare.py allowed.
2. Decoding order is the natural 5'->3' order (no permutation), the same
   order `model.sample` uses. The checkpoint was trained with random
   orders (random_order: True in its config), so 5'->3' is one valid order.
   We fix it so results are deterministic.

OUTPUTS
-------
* One cached JSON per candidate: <cache_dir>/<target_id>/<candidate_id>.json
  with mean_nll, perplexity, n_scored, and per-position p(true nt), logp and
  the full 4-way probabilities.
* A summary CSV (default data/processed/grnade_scores.csv) with one row per candidate.

SANITY CHECK MODE (--sanity)
----------------------------
For native (experimental) structures: the native sequence on its own
backbone should have perplexity clearly < 4, and shuffled versions of the
same sequence (same base composition) on the same backbone should be higher.

Usage:
    conda activate rna
    python src/score_grnade.py                       # score everything in the manifest
    python src/score_grnade.py --manifest path.csv --out_csv scores.csv
    python src/score_grnade.py --sanity --manifest natives_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# prepare.py sits next to this file; reuse its config helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import load_config, resolve_path  # noqa: E402

log = logging.getLogger("score_grnade")

SCORE_COLUMNS = ["target_id", "candidate_id", "status", "error",
                 "n_scored", "mean_nll", "perplexity", "flags", "cache_json"]


# ---------------------------------------------------------------------------
# Loading gRNAde
# ---------------------------------------------------------------------------

def _read_ckpt_config(path: Path) -> Dict[str, object]:
    """The checkpoint config.yaml stores each key as {value: ..., desc: ...}."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return {k: (v["value"] if isinstance(v, dict) and "value" in v else v)
            for k, v in raw.items()}


def _set_grnade_env(grnade_repo: Path) -> None:
    """Environment variables + sys.path that gRNAde needs before import (see import_grnade)."""
    for var in ("X3DNA", "ETERNAFOLD"):
        os.environ.setdefault(var, str(grnade_repo / "tools" / f"_unused_{var.lower()}"))
    os.environ.setdefault("PROJECT_PATH", str(grnade_repo) + os.sep)
    os.environ.setdefault("DATA_PATH", str(grnade_repo / "data") + os.sep)
    os.environ.setdefault("WANDB_MODE", "disabled")
    if str(grnade_repo) not in sys.path:
        sys.path.insert(0, str(grnade_repo))


# ---------------------------------------------------------------------------
# Secondary structure with a time limit
# ---------------------------------------------------------------------------
# gRNAde gets base pairs from the 3D structure with biotite (pdb_to_sec_struct ->
# base_pairs + pseudoknot ordering). On physically broken models (e.g. CASP15 R1138
# models with nucleotides < 1 A apart) the pseudoknot step can run for a very long
# time: >2 min vs ~11 s for a normal 720-nt model. A Python thread cannot be
# interrupted, so the call runs in a separate helper process that we kill if it
# exceeds grnade.sec_struct_timeout_sec. The candidate then gets the same fallback as
# any other secondary-structure failure (all-unpaired, flagged).

def _sec_struct_job(grnade_repo: str, filepath: str, sequence: str) -> str:
    """Runs in the helper process: exactly gRNAde's own call inside pdb_to_tensor."""
    _set_grnade_env(Path(grnade_repo))
    from src.data.sec_struct_utils import pdb_to_sec_struct
    return pdb_to_sec_struct(filepath, sequence, True)   # keep_pseudoknots=True, as in pdb_to_tensor


def _sec_struct_warmup(grnade_repo: str) -> bool:
    _set_grnade_env(Path(grnade_repo))
    from src.data.sec_struct_utils import pdb_to_sec_struct  # noqa: F401
    return True


class SecStructRunner:
    """Calls gRNAde's pdb_to_sec_struct in one persistent helper process, with a timeout."""

    def __init__(self, grnade_repo: Path, timeout_sec: float):
        self.repo = str(grnade_repo)
        self.timeout = float(timeout_sec)
        self.pool = None

    def _start(self):
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(max_workers=1)
        # Pay the helper's start-up/import cost here, not inside the timed call.
        self.pool.submit(_sec_struct_warmup, self.repo).result()

    def _kill(self):
        if self.pool is None:
            return
        for p in list(getattr(self.pool, "_processes", {}).values()):
            p.kill()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pool = None

    def __call__(self, filepath: str, sequence: str) -> str:
        from concurrent.futures import TimeoutError as FutureTimeout
        if self.pool is None:
            self._start()
        fut = self.pool.submit(_sec_struct_job, self.repo, filepath, sequence)
        try:
            return fut.result(timeout=self.timeout)
        except FutureTimeout:
            self._kill()            # the next call starts a fresh helper
            raise TimeoutError(f"secondary structure took > {self.timeout:.0f} s")

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


def import_grnade(grnade_repo: Path):
    """Import gRNAde's modules from its cloned repo.

    Notes for running on Windows / without gRNAde's optional tools:
      * gRNAde's `src/data/sec_struct_utils.py` has function DEFAULT ARGUMENTS
        like `os.path.join(X3DNA_PATH, "bin/find_pair")`, evaluated at IMPORT
        time. If the X3DNA / ETERNAFOLD environment variables are unset this
        raises a TypeError (os.path.join(None, ...)). We do not use X3DNA or
        EternaFold, so we set harmless placeholders BEFORE importing. If one of
        them were ever actually called, it would fail loudly (file not found)
        and not silently give wrong results.
      * gRNAde's package is called `src`, like our own folder. Our folder has
        NO __init__.py, so it is only a "namespace package". Python always
        prefers a regular package (gRNAde's `src` has an __init__.py), and we
        also put the gRNAde repo first on sys.path.
    """
    _set_grnade_env(grnade_repo)
    from src.models import gRNAde                                    # noqa: E402
    from src.data.featurizer import RNAGraphFeaturizer               # noqa: E402
    from src.data.data_utils import pdb_to_tensor, get_backbone_coords  # noqa: E402
    import src as _grnade_src
    if Path(_grnade_src.__file__).resolve().parent != (grnade_repo / "src").resolve():
        raise ImportError(f"'src' resolved to {_grnade_src.__file__}, not gRNAde's repo")
    return gRNAde, RNAGraphFeaturizer, pdb_to_tensor, get_backbone_coords



class GrnadeScorer:
    """Holds the loaded model + featurizer and scores one structure at a time."""

    def __init__(self, cfg: dict):
        import torch

        self.torch = torch
        tools = cfg["tools"]
        repo = resolve_path(cfg, tools["grnade_repo"])
        ckpt = resolve_path(cfg, tools["grnade_ckpt"])
        ckpt_cfg_path = resolve_path(cfg, tools["grnade_ckpt_config"])
        for p in (repo, ckpt, ckpt_cfg_path):
            if not p.exists():
                raise FileNotFoundError(f"Not found: {p}  (see environment.md, section 1)")

        gRNAde, RNAGraphFeaturizer, self.pdb_to_tensor, self.get_backbone_coords = \
            import_grnade(repo)
        self.sec_struct = SecStructRunner(repo, cfg["grnade"].get("sec_struct_timeout_sec", 60))
        c = _read_ckpt_config(ckpt_cfg_path)
        self.ckpt_path = str(ckpt)
        self.max_len = int(c.get("max_nodes_sample", 500))

        dev = str(cfg["grnade"].get("device", "auto"))
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        # Same arguments as design.py::main, with values from the checkpoint config.
        # split="test" -> no noise and no 3D dropout (featurizer.__call__).
        self.featurizer = RNAGraphFeaturizer(
            split="test",
            radius=c["radius"],
            top_k=c["top_k"],
            num_rbf=c["num_rbf"],
            num_posenc=c["num_posenc"],
            max_num_conformers=c["max_num_conformers"],
            noise_scale=c["noise_scale"],
            drop_prob_3d=c["drop_prob_3d"],
        )
        self.model = gRNAde(
            node_in_dim=tuple(c["node_in_dim"]),
            node_h_dim=tuple(c["node_h_dim"]),
            edge_in_dim=tuple(c["edge_in_dim"]),
            edge_h_dim=tuple(c["edge_h_dim"]),
            num_layers=c["num_layers"],
            drop_rate=c["drop_rate"],
            out_dim=c["out_dim"],
        )
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu"))
        self.model = self.model.to(self.device).eval()   # eval(): dropout off

        # gRNAde's own letter <-> index map: {'A': 0, 'G': 1, 'C': 2, 'U': 3, '_': 4}
        self.letter_to_num = dict(self.featurizer.letter_to_num)
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        log.info("gRNAde loaded: %s on %s", self.ckpt_path, self.device)

    # -- structure -> gRNAde input -----------------------------------------
    def load_structure(self, clean_pdb: str):
        """Parse a clean PDB with gRNAde's own loader.

        Returns (candidate_sequence, backbone_coords [L,3,3], sec_struct, flags).
        Same calls as RNAGraphFeaturizer.featurize_from_pdb_file.
        """
        flags = []
        # GRNADES OWN FUCNTION THAT CONVERTS PDB TO TENSOR (sequence + 3D coordinates).
        # The secondary structure is computed separately below, with the SAME gRNAde
        # function pdb_to_tensor would call, but with a time limit.
        out = self.pdb_to_tensor(clean_pdb, return_sec_struct=False, return_sasa=False)
        if out is None:
            raise ValueError("gRNAde pdb_to_tensor returned None (<= 1 residue)")
        seq, coords, _, _ = out
        try:
            sec_struct = self.sec_struct(clean_pdb, seq)
            assert len(sec_struct) == len(seq), "Sequence and secondary structure must be the same length"
        except TimeoutError as exc:
            # Broken geometry made base-pair detection explode (see SecStructRunner).
            log.warning("sec-struct timed out for %s (%s); using no 2D edges", clean_pdb, exc)
            sec_struct = "." * len(seq)
            flags += ["no_sec_struct", "sec_struct_timeout"]
        except Exception as exc:
            # gRNAde's pdb_to_sec_struct falls back to X3DNA (Linux binary) when
            # biotite fails on very short RNAs. We have no X3DNA, so we drop
            # 2D edges for that candidate (an all-unpaired structure) and FLAG it.
            # The 3D kNN and sequence edges remain.
            log.warning("sec-struct failed for %s (%s); using no 2D edges", clean_pdb, exc)
            sec_struct = "." * len(seq)
            flags.append("no_sec_struct")
        coords = self.get_backbone_coords(coords, seq)  # P, C4', N1/N9 by CANDIDATE base
        return seq, coords, sec_struct, flags

    # -- teacher-forced likelihood -----------------------------------------
    def score(self, sequence: str, coords, sec_struct: str) -> dict:
        """Teacher-forced log-likelihood of `sequence` on the given backbone.

        `sequence` must have the same length and residue order as `coords`.
        Characters outside A/G/C/U become '_' and are not scored (mask_seq),
        exactly as in gRNAde's training loss.
        """
        torch = self.torch
        F = torch.nn.functional
        seq = "".join(ch if ch in ("A", "G", "C", "U") else "_" for ch in sequence)
        if len(seq) != coords.shape[0]:
            raise ValueError(f"sequence length {len(seq)} != structure length {coords.shape[0]}")

        rna = {"sequence": seq, "coords_list": [coords], "sec_struct_list": [sec_struct]}
        with torch.no_grad():
            data = self.featurizer(rna).to(self.device)   # permutation=None -> 5'->3'
            logits = self.model(data)                     # [L, 4], teacher-forced on data.seq
            logp = F.log_softmax(logits, dim=-1)          # [L, 4]
            mask = data.mask_seq                           # True where seq is A/G/C/U
            idx = data.seq.clone()
            idx[~mask] = 0                                 # placeholder; masked out below
            logp_true = logp.gather(1, idx.view(-1, 1)).squeeze(1)   # [L]

        logp_true = logp_true.cpu().numpy().astype(float)
        mask_np = mask.cpu().numpy().astype(bool)
        probs = logp.exp().cpu().numpy()
        n_scored = int(mask_np.sum())
        if n_scored == 0:
            raise ValueError("no scorable positions")
        mean_nll = float(-logp_true[mask_np].mean())

        # Probability columns in gRNAde's index order, renamed to letters for readability
        order = [self.num_to_letter[i] for i in range(probs.shape[1])]
        return {
            "n_scored": n_scored,
            "mean_nll": mean_nll,
            "perplexity": float(np.exp(mean_nll)),
            "p_true": [float(np.exp(v)) if m else None for v, m in zip(logp_true, mask_np)],
            "logp_true": [float(v) if m else None for v, m in zip(logp_true, mask_np)],
            "prob_order": order,
            "probs": np.round(probs, 5).tolist(),
        }


# ---------------------------------------------------------------------------
# Per-candidate scoring with caching
# ---------------------------------------------------------------------------

def score_candidate(scorer: GrnadeScorer, row: dict, cache_dir: Path,
                    force: bool = False) -> dict:
    """Score one manifest row (status == ok). Returns a summary row."""
    out = {c: "" for c in SCORE_COLUMNS}
    out.update(target_id=row["target_id"], candidate_id=row["candidate_id"])
    cache_json = cache_dir / row["target_id"] / f"{row['candidate_id']}.json"
    out["cache_json"] = str(cache_json)

    # Cache: reuse if it was computed from the same clean PDB and checkpoint
    if cache_json.exists() and not force:
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
        if cached.get("clean_pdb") == row["clean_pdb"] and cached.get("checkpoint") == scorer.ckpt_path:
            out.update(status="ok", n_scored=cached["n_scored"], mean_nll=cached["mean_nll"],
                       perplexity=cached["perplexity"], flags=";".join(cached.get("flags", [])))
            return out

    try:
        mapping = json.loads(Path(row["map_json"]).read_text(encoding="utf-8"))
        cand_seq, coords, sec_struct, flags = scorer.load_structure(row["clean_pdb"])
        # The clean PDB must contain exactly the residues listed in the mapping
        if cand_seq != mapping["candidate_nt"]:
            raise ValueError("clean PDB sequence does not match map.json candidate_nt")
        target_nt = mapping["target_nt"]              # the sequence we are scoring
        if len(target_nt) > scorer.max_len:
            flags.append(f"longer_than_{scorer.max_len}")
        res = scorer.score(target_nt, coords, sec_struct)
    except Exception as exc:
        out.update(status="error", error=f"{type(exc).__name__}: {exc}")
        log.warning("[%s] %s: %s", row["target_id"], row["candidate_id"], out["error"])
        return out

    record = {
        "target_id": row["target_id"],
        "candidate_id": row["candidate_id"],
        "clean_pdb": row["clean_pdb"],
        "checkpoint": scorer.ckpt_path,
        "flags": flags,
        "sec_struct": sec_struct,
        "target_index": mapping["target_index"],   # 0-based target position per scored residue
        "target_nt": target_nt,
        "orig_chain": mapping["orig_chain"],       # for colouring the original model in the viewer
        "orig_res_id": mapping["orig_res_id"],
        **res,
    }
    cache_json.parent.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(json.dumps(record), encoding="utf-8")
    out.update(status="ok", n_scored=res["n_scored"], mean_nll=round(res["mean_nll"], 6),
               perplexity=round(res["perplexity"], 6), flags=";".join(flags))
    return out


def read_manifest(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(rows: List[dict], path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Sanity checks on native structures
# ---------------------------------------------------------------------------

SANITY_COLUMNS = ["target_id", "candidate_id", "length", "native_ppl",
                  "shuffled_ppl_mean", "shuffled_ppl_min", "n_shuffles",
                  "native_below_random", "native_below_all_shuffles", "flags"]


def run_sanity(scorer: GrnadeScorer, rows: List[dict], cfg: dict) -> List[dict]:
    """Native sequence vs shuffled sequences on each native backbone.

    Shuffling keeps the base composition, so a lower native perplexity
    shows that gRNAde uses the 3D structure, not just base frequencies.
    """
    n_shuf = int(cfg["grnade"]["sanity_n_shuffles"])
    random_ppl = float(cfg["grnade"]["sanity_random_ppl"])
    rng = random.Random(int(cfg["seed"]))
    out = []
    for row in rows:
        mapping = json.loads(Path(row["map_json"]).read_text(encoding="utf-8"))
        cand_seq, coords, sec_struct, flags = scorer.load_structure(row["clean_pdb"])
        native = mapping["target_nt"]
        native_ppl = scorer.score(native, coords, sec_struct)["perplexity"]
        shuf_ppls = []
        for _ in range(n_shuf):
            s = list(native)
            rng.shuffle(s)
            shuf_ppls.append(scorer.score("".join(s), coords, sec_struct)["perplexity"])
        r = {
            "target_id": row["target_id"], "candidate_id": row["candidate_id"],
            "length": len(native), "native_ppl": round(native_ppl, 4),
            "shuffled_ppl_mean": round(float(np.mean(shuf_ppls)), 4),
            "shuffled_ppl_min": round(float(np.min(shuf_ppls)), 4),
            "n_shuffles": n_shuf,
            "native_below_random": native_ppl < random_ppl,
            "native_below_all_shuffles": native_ppl < min(shuf_ppls),
            "flags": ";".join(flags),
        }
        log.info("SANITY %s: native ppl %.3f | shuffled mean %.3f (min %.3f) | <4: %s | < all shuffles: %s",
                 row["candidate_id"], native_ppl, r["shuffled_ppl_mean"], r["shuffled_ppl_min"],
                 r["native_below_random"], r["native_below_all_shuffles"])
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score candidates with gRNAde (teacher-forced perplexity).")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--manifest", help="manifest CSV from prepare.py (default: data.manifest_csv)")
    ap.add_argument("--out_csv", help="summary CSV (default: <processed_dir>/grnade_scores.csv)")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    ap.add_argument("--sanity", action="store_true",
                    help="treat manifest rows as NATIVE structures and run the sanity checks")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log_dir = resolve_path(cfg, cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "score_grnade.log", encoding="utf-8")],
    )
    set_seeds(int(cfg["seed"]))

    manifest = Path(args.manifest) if args.manifest else resolve_path(cfg, cfg["data"]["manifest_csv"])
    rows = [r for r in read_manifest(manifest) if r["status"] == "ok"]
    log.info("%d scorable candidates in %s", len(rows), manifest)

    scorer = GrnadeScorer(cfg)

    if args.sanity:
        res = run_sanity(scorer, rows, cfg)
        out = Path(args.out_csv) if args.out_csv else resolve_path(cfg, cfg["output"]["results_dir"]) / "sanity_grnade.csv"
        write_csv(res, out, SANITY_COLUMNS)
        n_ok = sum(r["native_below_random"] and r["native_below_all_shuffles"] for r in res)
        log.info("Sanity: %d/%d natives pass both checks. Written to %s", n_ok, len(res), out)
        return 0

    cache_dir = resolve_path(cfg, cfg["grnade"]["cache_dir"])
    results = []
    for i, row in enumerate(rows, 1):
        results.append(score_candidate(scorer, row, cache_dir, force=args.force))
        if i % 50 == 0:
            log.info("scored %d/%d", i, len(rows))

    out = Path(args.out_csv) if args.out_csv else resolve_path(cfg, cfg["data"]["processed_dir"]) / "grnade_scores.csv"
    write_csv(results, out, SCORE_COLUMNS)
    n_err = sum(r["status"] != "ok" for r in results)
    log.info("Scores written: %s (%d ok, %d errors)", out, len(results) - n_err, n_err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
