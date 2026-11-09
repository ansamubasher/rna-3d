"""
prepare.py — Step 3: preprocess candidate 3D structures for gRNAde scoring.

What this script does, for one target (one RNA sequence) and a folder of
candidate structures produced by ANY predictor (RNAPro, DRfold2, RhoFold+,
CASP15 groups, ...):

  1. Read every candidate file (PDB or mmCIF, optionally .gz). A file holding
     several MODEL blocks is split into one candidate per model.
  2. Keep only RNA nucleotide residues and rename them to A/C/G/U. Old atom
     names are normalised too (C4* -> C4', O1P -> OP1).
  3. Map the candidate's residues onto the target sequence with a sequence
     alignment, because residue NUMBERING in predicted files often differs
     from the target (it may start at 0 or 1, use gaps, or have several chains).
  4. Skip candidates whose coverage of the target is < min_coverage, or whose
     aligned residues mismatch the target too often. Every skip reason is logged.
  5. For each accepted candidate write:
       - <candidate_id>.pdb       : a clean, single-model, single-chain PDB whose
                                    residue numbers are TARGET positions (1-based)
       - <candidate_id>.map.json  : the residue mapping (target position <->
                                    original chain/residue number), which is
                                    needed to colour the original model in the
                                    3D viewer later
  6. Write a manifest CSV with one row per candidate, accepted or skipped.

Why write a clean PDB instead of tensors?
  gRNAde's own loader `src/data/data_utils.py::pdb_to_tensor` (gRNAde repo,
  commit 2453b18) parses PDB files with `cpdb`, keeps only residues named
  A/G/C/U (everything else becomes "_" and is masked), does not split
  MODEL blocks, and does not read mmCIF. If every candidate becomes a clean
  single-model PDB here, Step 4 can feed it straight into gRNAde's own
  `pdb_to_tensor` + `get_backbone_coords` (P, C4', N1/N9) without any
  re-implemented featurisation. That is what the task asks for.

Usage (single target, which validate.py will also use):
    python src/prepare.py --target_id R1107 --sequence R1107.fasta \
        --candidates data/raw/casp15/predictions/R1107

Usage (many targets, for the CASP15 evaluation):
    python src/prepare.py --targets_csv data/raw/casp15/targets.csv
  where targets.csv has columns: target_id, sequence, candidates_dir
  ("sequence" may be the sequence itself or a path to a FASTA file).
  The CSV is built in Step 2 once the real CASP15 file layout is known.

For multi-chain targets, give the chains concatenated in order as ONE
sequence. The alignment then maps the candidate's chains (in file order) onto it.

References: Joshi et al. 2025 (gRNAde, ICLR) for the backbone representation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

log = logging.getLogger("prepare")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Copied from gRNAde src/constants.py (RNA_ATOMS, commit 2453b18). Only these
# atoms are written to the clean PDB. gRNAde ignores every other atom, and
# dropping hydrogens keeps the files small. Copied rather than imported so that
# prepare.py does not need the gRNAde environment (gRNAde's constants.py
# imports wandb at module level).
RNA_ATOMS = [
    "P", "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    "N1", "C2", "O2", "N2", "N3", "C4", "O4", "N4", "C5", "C6", "O6",
    "N6", "N7", "C8", "N9", "OP1", "OP2",
]
STANDARD_NT = {"A", "C", "G", "U"}

# Residue-name spellings seen in RNA PDB files that mean a standard nucleotide.
# Modified nucleotides (PSU, 5MC, 2MG, ...) are deliberately NOT mapped:
# guessing their parent base would be a silent assumption. They are dropped,
# and so count against coverage.
RESNAME_ALIASES = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "RA": "A", "RC": "C", "RG": "G", "RU": "U",          # older / AMBER style
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U",      # three-letter style
    "RA5": "A", "RC5": "C", "RG5": "G", "RU5": "U",      # AMBER 5'-terminal
    "RA3": "A", "RC3": "C", "RG3": "G", "RU3": "U",      # AMBER 3'-terminal
}

# Old-style atom names -> current PDB names
ATOM_ALIASES = {"O1P": "OP1", "O2P": "OP2"}

STRUCTURE_EXTS_PDB = {".pdb", ".ent"}
STRUCTURE_EXTS_CIF = {".cif", ".mmcif"}

MANIFEST_COLUMNS = [
    "target_id", "candidate_id", "source_file", "model_index",
    "clean_pdb", "map_json",
    "target_length", "length", "coverage",
    "n_mismatch", "mismatch_fraction", "n_extra_residues",
    "n_nonstandard_dropped", "n_missing_c4_dropped",
    "status", "skip_reason", "flags",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | os.PathLike) -> dict:
    """Load config.yaml and return it as a dict.

    The path of the config file is stored under cfg["_root"]. Relative paths
    in the config are resolved against that folder by `resolve_path`, not
    against the current working directory.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(config_path.parent)
    return cfg


def resolve_path(cfg: dict, p: str | os.PathLike) -> Path:
    """Resolve a path from the config: absolute stays as is, relative is
    taken from the folder that holds config.yaml."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else Path(cfg["_root"]) / p


# ---------------------------------------------------------------------------
# Target sequence handling
# ---------------------------------------------------------------------------

def clean_sequence(seq: str) -> str:

    """Uppercase, strip whitespace, convert DNA-style T to U."""
    # whenever there is a space
    # it splits the sequecne into two chains
    # and then joins them together to form a single chain
    seq = "".join(seq.split()).upper().replace("T", "U")

    # raises an error
    if not seq:
        raise ValueError("Empty target sequence")

    # if they have stuff that is not int he standard nucleotides, we will warn them
    bad = sorted(set(seq) - STANDARD_NT)
    if bad:
        # We keep these positions (so numbering stays correct), but they can
        # never be matched or scored. Warn so the user knows.
        log.warning("Target sequence contains non-ACGU characters %s; "
                    "these positions cannot be scored.", bad)
    return seq



def read_sequence_arg(value: str) -> str:
    """--sequence may be a raw sequence or a path to a FASTA/plain text file.

    For a FASTA with several records (e.g. a multi-chain target), the records
    are concatenated in file order.
    """
    path = Path(value)
    # path is actually a file, not just a string that looks like one. Read it.
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        # read it and strip it down to just the sequence letters, ignoring FASTA headers and whitespace
        seq_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith(">")]
        #calls the cleans equence function to clean the sequence and return it
        return clean_sequence("".join(seq_lines))
    return clean_sequence(value)


# ---------------------------------------------------------------------------
# Sequence alignment (residue mapping)
# ---------------------------------------------------------------------------
# A small, self-contained global alignment with FREE END GAPS (a
# "semi-global" or "glocal" alignment) and AFFINE GAP PENALTIES (Gotoh 1982).
#
# Free end gaps, because:
#   * a candidate may be missing a few terminal residues of the target, and
#   * a candidate may have extra residues at its ends (e.g. a tag or a second
#     chain copy).
#   Neither should be penalised. Only internal gaps cost something.
#
# Affine gaps (opening a gap costs more than extending it), because missing
# residues in a model are almost always ONE contiguous segment. With a single
# flat gap cost, three 1-residue gaps cost the same as one 3-residue gap, and
# inside repetitive sequence (e.g. GGGAGAG) the aligner may scatter them. That
# assigns some residues to target positions that are off by one. Affine costs
# make one contiguous gap the clear winner.
#
# The code is written out (instead of calling a library) so it can be
# explained line by line.

MATCH, MISMATCH = 2, -1
GAP_OPEN = -5      # cost of the FIRST position of a gap
GAP_EXTEND = -1    # cost of every further position of the same gap
_NEG = -10**9      # "impossible" score


def align_semiglobal(target: str, query: str) -> List[Tuple[int, int]]:
    """Align `query` (candidate sequence) to `target`.

    Returns the aligned column pairs (target_index, query_index), 0-based, in
    increasing order. Positions aligned to a gap are not returned.

    THREE SCORE MATRICES (Gotoh), each (n+1) x (m+1), for the best alignment of
    target[:i] with query[:j] that ENDS in:
        H[i, j]  any state (the overall best)
        F[i, j]  a gap in the query: target[i-1] is unmatched   ("up" move)
        E[i, j]  a gap in the target: query[j-1] is unmatched   ("left" move)
    Recurrences:
        F[i, j] = max(H[i-1, j] + GAP_OPEN, F[i-1, j] + GAP_EXTEND)
        E[i, j] = max(H[i, j-1] + GAP_OPEN, E[i, j-1] + GAP_EXTEND)
        H[i, j] = max(H[i-1, j-1] + match/mismatch, F[i, j], E[i, j])
    Row 0 and column 0 of H are 0, because leading gaps are free.

    TRACEBACK matrices record which option won in each cell, so we can walk back
    from the end and read off the aligned pairs. Ties are broken in a fixed order
    (diagonal > up > left, and extend > open), so the result is deterministic.
    """
    n, m = len(target), len(query)
    if n == 0 or m == 0:
        return []

    H = np.zeros((n + 1, m + 1), dtype=np.int64)
    F = np.full((n + 1, m + 1), _NEG, dtype=np.int64)
    E = np.full((n + 1, m + 1), _NEG, dtype=np.int64)
    TH = np.zeros((n + 1, m + 1), dtype=np.int8)   # H came from: 0 diag, 1 F (up), 2 E (left)
    TF = np.zeros((n + 1, m + 1), dtype=np.int8)   # F came from: 0 open (from H), 1 extend (from F)
    TE = np.zeros((n + 1, m + 1), dtype=np.int8)   # E came from: 0 open (from H), 1 extend (from E)

    q = np.frombuffer(query.encode(), dtype=np.uint8)
    for i in range(1, n + 1):
        # F and the diagonal only need row i-1, so they are computed for the whole row at once
        f_open, f_ext = H[i - 1, 1:] + GAP_OPEN, F[i - 1, 1:] + GAP_EXTEND
        TF[i, 1:] = f_ext >= f_open
        F[i, 1:] = np.maximum(f_open, f_ext)
        diag = H[i - 1, :-1] + np.where(q == ord(target[i - 1]), MATCH, MISMATCH)

        # E needs the cell just to the left in the SAME row, so loop over j
        for j in range(1, m + 1):
            e_open, e_ext = H[i, j - 1] + GAP_OPEN, E[i, j - 1] + GAP_EXTEND
            if e_ext >= e_open:
                E[i, j], TE[i, j] = e_ext, 1
            else:
                E[i, j], TE[i, j] = e_open, 0
            d, f, e = diag[j - 1], F[i, j], E[i, j]
            if d >= f and d >= e:
                H[i, j], TH[i, j] = d, 0
            elif f >= e:
                H[i, j], TH[i, j] = f, 1
            else:
                H[i, j], TH[i, j] = e, 2

    # Trailing gaps are free: the best end point lies on the last row or the
    # last column of H.
    best_last_row_j = int(np.argmax(H[n, :]))
    best_last_col_i = int(np.argmax(H[:, m]))
    if H[n, best_last_row_j] >= H[best_last_col_i, m]:
        i, j = n, best_last_row_j
    else:
        i, j = best_last_col_i, m

    # Walk back. `state` says which matrix we are in.
    pairs: List[Tuple[int, int]] = []
    state = "H"
    while i > 0 and j > 0:
        if state == "H":
            t = TH[i, j]
            if t == 0:                       # aligned pair
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
            else:
                state = "F" if t == 1 else "E"
        elif state == "F":                   # target residue unmatched
            state = "F" if TF[i, j] == 1 else "H"
            i -= 1
        else:                                # query residue unmatched
            state = "E" if TE[i, j] == 1 else "H"
            j -= 1
    pairs.reverse()
    return pairs


@dataclass
class MappingResult:
    pairs: List[Tuple[int, int]]   # (target_index, candidate_index), 0-based
    coverage: float                # aligned target positions / target length
    n_mismatch: int                # aligned columns where the bases differ
    mismatch_fraction: float       # n_mismatch / aligned columns
    n_extra: int                   # candidate residues not aligned to the target


def map_candidate_to_target(target: str, cand_seq: str,
                            _cache: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
                            ) -> MappingResult:
    """Map candidate residues onto target positions.

    Fast path: an identical sequence maps 1:1 with no alignment.
    Otherwise the alignment for each (target, candidate sequence) pair is
    cached, because most candidates of a target share the same sequence.
    (The mutable default `_cache` is intentional: it lives for one process.)
    """


    # CHECKS WHEN TO URN AN ALIGNMENT FUNCTIONA ND WHEN NOT TO


    # If the candidate sequence is identical to the target, we can skip the
    if cand_seq == target:
        pairs = [(i, i) for i in range(len(target))]


# ELSE IT CALLS THE ALIGNMENT FUNCTION
# Caching results simply means saving the result of a calculation so you don't have to calculate it again next time.   
    else:
        key = (target, cand_seq)
        if key not in _cache:
            _cache[key] = align_semiglobal(target, cand_seq)
        pairs = _cache[key]

    n_aligned = len(pairs)
    n_mismatch = sum(1 for ti, ci in pairs if target[ti] != cand_seq[ci])


    # and returns a MappingResult object with the mapping information
    return MappingResult(
        pairs=pairs,
        coverage=n_aligned / len(target),
        n_mismatch=n_mismatch,
        mismatch_fraction=(n_mismatch / n_aligned) if n_aligned else 1.0,
        n_extra=len(cand_seq) - n_aligned,
    )


# ---------------------------------------------------------------------------
# Structure reading
# ---------------------------------------------------------------------------

# figures out which formati it s
# DOE SIT END WITH GZ
def _open_text(path: Path) -> str:
    """Return the file contents as text, transparently un-gzipping .gz files."""
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="replace")

# ALSO DOES THE SAME THIGN
# ENDS WITH GZ, THEN STRIPS IT OFF AND CHECKS THE EXTENSION
# GZ IS A FILE COMPRESSION METHOD
#GZIP
def detect_format(path: Path, text: str) -> Optional[str]:
    """Return 'pdb', 'cif' or None.

    Uses the file extension first. Predictor and CASP output files often have
    no extension at all, so if the extension doesn't decide, sniff the content:
    mmCIF starts with a 'data_' block, and PDB/CASP-TS files contain ATOM or
    HETATM records.
    """
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    ext = os.path.splitext(name)[1]
    if ext in STRUCTURE_EXTS_PDB:
        return "pdb"
    if ext in STRUCTURE_EXTS_CIF:
        return "cif"
    head = text.lstrip()[:20000]
    if head.startswith("data_"):
        return "cif"
    for line in head.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            return "pdb"
    return None


# NOW ACTYALLY OPENS THE FILE
# AND READ THE 3D MODEL

def read_models(path: Path):
    """Read a structure file and return a list of biotite AtomArrays, one per model."""
    from biotite.structure.io import pdb as bpdb
    from biotite.structure.io import pdbx as bpdbx

    text = _open_text(path)
    fmt = detect_format(path, text)
    if fmt is None:
        raise ValueError("unrecognised file format (no ATOM records / not mmCIF)")

    models = []
    if fmt == "pdb":
        f = bpdb.PDBFile.read(io.StringIO(text))
        n_models = f.get_model_count()
        for k in range(1, n_models + 1):
            models.append(f.get_structure(model=k, altloc="first"))
    else:
        f = bpdbx.CIFFile.read(io.StringIO(text))
        n_models = bpdbx.get_model_count(f)
        for k in range(1, n_models + 1):
            # use_author_fields=True: the residue numbers/chains the depositor
            # wrote, which match what users see in viewers.
            models.append(bpdbx.get_structure(f, model=k, altloc="first",
                                              use_author_fields=True))
    return models


# ---------------------------------------------------------------------------
# Residue extraction and clean-PDB writing
# ---------------------------------------------------------------------------

# NORMALISES STUFFFF
def normalize_resname(name: str) -> Optional[str]:
    """Map a residue name to A/C/G/U, or None if it is not a standard RNA nucleotide."""
    return RESNAME_ALIASES.get(name.strip().upper())

# ALSO NORMALSIESS SACCORIDN GTO THE FORMAT OR NAMING CONVENTION WE SPECICIFED AT THE TOP INT HE LISTTT
def normalize_atom_name(name: str) -> str:
    name = name.strip().replace("*", "'")   # C4* -> C4'
    return ATOM_ALIASES.get(name, name)


@dataclass
class Residue:
    nt: str                 # A/C/G/U
    chain: str              # original chain id
    res_id: int             # original residue number
    ins_code: str           # original insertion code
    atoms: "object"         # biotite AtomArray of this residue's kept atoms


@dataclass
class ExtractResult:
    residues: List[Residue] = field(default_factory=list)
    n_nonstandard: int = 0  # residues dropped: not a standard nucleotide (incl. ligands, ions, water, protein)
    n_missing_c4: int = 0   # residues dropped: no C4' atom (gRNAde cannot place them)



# DISCARS EVERYTHIGN EXCEPT THE USELFUL ONES

def extract_rna_residues(atoms) -> ExtractResult:
    """Walk residues in file order and keep standard RNA nucleotides with a C4' atom."""
    import biotite.structure as struc

    out = ExtractResult()
    # Normalise atom names on a copy (old C4* style, O1P/O2P)
    atoms = atoms.copy()
    atoms.atom_name = np.array([normalize_atom_name(a) for a in atoms.atom_name],
                               dtype=atoms.atom_name.dtype)
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for s, e in zip(starts[:-1], starts[1:]):
        res = atoms[s:e]
        nt = normalize_resname(res.res_name[0])
        if nt is None:
            # Water and ions are expected and are not worth counting.
            if res.res_name[0].strip().upper() not in {"HOH", "WAT", "MG", "K", "NA", "CL"}:
                out.n_nonstandard += 1
            continue
        keep = np.isin(res.atom_name, RNA_ATOMS)
        res = res[keep]
        if "C4'" not in set(res.atom_name):
            out.n_missing_c4 += 1
            continue
        # Duplicate atom names inside one residue (alternative locations that
        # were not resolved): keep the first occurrence of each.
        _, first_idx = np.unique(res.atom_name, return_index=True)
        res = res[np.sort(first_idx)]
        out.residues.append(Residue(nt=nt, chain=str(res.chain_id[0]),
                                    res_id=int(res.res_id[0]),
                                    ins_code=str(res.ins_code[0]),
                                    atoms=res))
    return out


def write_clean_pdb(residues: List[Residue], pairs: List[Tuple[int, int]],
                    out_path: Path) -> None:
    """Write mapped residues as ONE chain 'A', numbered by TARGET position (1-based).

    The residue NAME stays the candidate's own nucleotide. gRNAde's
    `get_backbone_coords` uses that name to pick N1 (pyrimidine) or N9 (purine),
    and it must match the atoms that are really present. The target nucleotide
    at each position is stored in the .map.json and used in Step 4.
    """
    import biotite.structure as struc
    from biotite.structure.io import pdb as bpdb

    parts = []
    for ti, ci in pairs:
        r = residues[ci].atoms.copy()
        r.chain_id[:] = "A"
        r.res_id[:] = ti + 1
        r.ins_code[:] = ""
        r.res_name[:] = residues[ci].nt
        r.hetero[:] = False
        parts.append(r)
    arr = struc.concatenate(parts)
    f = bpdb.PDBFile()
    f.set_structure(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out_path))


# ---------------------------------------------------------------------------
# Per-target driver
# ---------------------------------------------------------------------------

def list_candidate_files(cand_dir: Path) -> List[Path]:
    """Every regular file in the folder, sorted so the order is reproducible.
    The format check happens later, so extension-less CASP files are kept too."""
    files = [p for p in sorted(cand_dir.iterdir()) if p.is_file()
             and not p.name.startswith(".")
             and not p.name.endswith((".map.json", ".csv", ".txt", ".log", ".md"))]
    return files


def _base_row(target_id, candidate_id, src, model_index, target_len) -> dict:
    row = {c: "" for c in MANIFEST_COLUMNS}
    row.update(target_id=target_id, candidate_id=candidate_id,
               source_file=str(src), model_index=model_index,
               target_length=target_len)
    return row


def prepare_target(target_id: str, sequence: str, cand_dir: Path,
                   out_dir: Path, cfg: dict) -> List[dict]:
    """Process every candidate of one target. Returns the manifest rows."""
    pcfg = cfg["prepare"]
    min_cov = float(pcfg["min_coverage"])
    max_mm = float(pcfg["max_mismatch_fraction"])
    max_len = int(pcfg["grnade_max_len"])
    split_models = bool(pcfg.get("split_multi_model", True))

    rows: List[dict] = []

    # GETS ALL THE CANDIATE FIKES
    files = list_candidate_files(cand_dir)
    if not files:
        log.warning("[%s] no files found in %s", target_id, cand_dir)



# FIGURES OTUT EH FORMAT OF THESE FULES
    for src in files:
        stem = src.name[:-3] if src.name.lower().endswith(".gz") else src.name
        stem = os.path.splitext(stem)[0] if os.path.splitext(stem)[1].lower() in (
            STRUCTURE_EXTS_PDB | STRUCTURE_EXTS_CIF) else stem
# READS THE 3D MODELS OT HESE FILES
        try:
            models = read_models(src)
        except Exception as exc:  # report the error, never crash the whole run
            row = _base_row(target_id, stem, src, "", len(sequence))
            row.update(status="skipped", skip_reason=f"parse_error: {exc}")
            rows.append(row)
            log.info("[%s] skip %s: %s", target_id, src.name, row["skip_reason"])
            continue
#DECIDES WHTEHR TO SPLIT THE MDOEL OR NTO
        if not split_models:
            models = models[:1]
        multi = len(models) > 1
# process eahc model separtely  
        for k, atoms in enumerate(models, start=1):
            # GIVE SIT A CAND ID
            cand_id = f"{stem}_m{k}" if multi else stem
            row = _base_row(target_id, cand_id, src, k, len(sequence))
#GET STHE RESIDEUES
            ex = extract_rna_residues(atoms)
            row.update(n_nonstandard_dropped=ex.n_nonstandard,
                       n_missing_c4_dropped=ex.n_missing_c4)
            if len(ex.residues) < 2:
                row.update(status="skipped",
                           skip_reason="fewer than 2 RNA residues with a C4' atom")
                rows.append(row)
                log.info("[%s] skip %s: %s", target_id, cand_id, row["skip_reason"])
                continue
# GETS TEH SEQURNCE
            cand_seq = "".join(r.nt for r in ex.residues)
            #MAPS TOT ARGET
            m = map_candidate_to_target(sequence, cand_seq)
            row.update(length=len(m.pairs), coverage=round(m.coverage, 4),
                       n_mismatch=m.n_mismatch,
                       mismatch_fraction=round(m.mismatch_fraction, 4),
                       n_extra_residues=m.n_extra)
# REASONS OF EVERYHTING
            reasons = []
            if m.coverage < min_cov:
                reasons.append(f"coverage {m.coverage:.3f} < {min_cov}")
            if m.mismatch_fraction > max_mm:
                reasons.append(f"mismatch_fraction {m.mismatch_fraction:.3f} > {max_mm}")
            if reasons:
                row.update(status="skipped", skip_reason="; ".join(reasons))
                rows.append(row)
                log.info("[%s] skip %s: %s", target_id, cand_id, row["skip_reason"])
                continue
# PSOSIBEL PLAGS LIKE MSOMATCH ETC
            flags = []
            if len(m.pairs) > max_len:
                flags.append(f"longer_than_grnade_max_len_{max_len}")
            if m.n_extra > 0:
                flags.append(f"extra_residues_{m.n_extra}")
            if m.n_mismatch > 0:
                flags.append(f"mismatches_{m.n_mismatch}")
#MAKES A PDB FILE AND A MAP.JSON
            clean_pdb = out_dir / f"{cand_id}.pdb"
            map_json = out_dir / f"{cand_id}.map.json"
            write_clean_pdb(ex.residues, m.pairs, clean_pdb)
            mapping = {
                "target_id": target_id,
                "candidate_id": cand_id,
                "source_file": str(src),
                "model_index": k,
                "target_sequence": sequence,
                # One entry per residue in the clean PDB, in the same order:
                "target_index": [ti for ti, _ in m.pairs],            # 0-based
                "target_nt": "".join(sequence[ti] for ti, _ in m.pairs),
                "candidate_nt": "".join(cand_seq[ci] for _, ci in m.pairs),
                "orig_chain": [ex.residues[ci].chain for _, ci in m.pairs],
                "orig_res_id": [ex.residues[ci].res_id for _, ci in m.pairs],
                "orig_ins_code": [ex.residues[ci].ins_code for _, ci in m.pairs],
            }
            map_json.write_text(json.dumps(mapping, indent=1), encoding="utf-8")

            row.update(clean_pdb=str(clean_pdb), map_json=str(map_json),
                       status="ok", flags=";".join(flags))
            rows.append(row)

    n_ok = sum(r["status"] == "ok" for r in rows)
    log.info("[%s] %d candidates: %d ok, %d skipped",
             target_id, len(rows), n_ok, len(rows) - n_ok)
    return rows


def write_manifest(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict) -> None:
    log_dir = resolve_path(cfg, cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "prepare.log", encoding="utf-8")],
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--target_id", help="single-target mode: target name")
    ap.add_argument("--sequence", help="single-target mode: sequence or FASTA path")
    ap.add_argument("--candidates", help="single-target mode: folder of candidate files")
    ap.add_argument("--targets_csv", help="batch mode: CSV with target_id,sequence,candidates_dir")
    ap.add_argument("--out_dir", help="where clean PDBs go (default: data.processed_dir)")
    ap.add_argument("--manifest", help="manifest CSV path (default: data.manifest_csv)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.targets_csv:
        with open(args.targets_csv, newline="", encoding="utf-8") as fh:
            targets = [(r["target_id"], read_sequence_arg(r["sequence"]),
                        Path(r["candidates_dir"])) for r in csv.DictReader(fh)]
    elif args.target_id and args.sequence and args.candidates:
        targets = [(args.target_id, read_sequence_arg(args.sequence), Path(args.candidates))]
    else:
        ap.error("give either --targets_csv, or all of --target_id --sequence --candidates")

    out_root = Path(args.out_dir) if args.out_dir else resolve_path(cfg, cfg["data"]["processed_dir"])
    manifest = Path(args.manifest) if args.manifest else resolve_path(cfg, cfg["data"]["manifest_csv"])

    all_rows: List[dict] = []
    for target_id, seq, cand_dir in targets:
        if not cand_dir.is_dir():
            log.error("[%s] candidates folder not found: %s", target_id, cand_dir)
            continue
        log.info("[%s] length %d, candidates from %s", target_id, len(seq), cand_dir)
        all_rows += prepare_target(target_id, seq, cand_dir, out_root / "candidates" / target_id, cfg)

    write_manifest(all_rows, manifest)
    skipped = [r for r in all_rows if r["status"] != "ok"]
    log.info("Manifest written: %s (%d rows, %d skipped)", manifest, len(all_rows), len(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
