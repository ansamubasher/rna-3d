"""Step 2: build targets.csv and natives.csv for the CASP15 RNA evaluation.

Reads only local files that were downloaded by hand (see README):
    data/raw/casp15/casp15.seq.txt              official CASP15 sequences
    data/raw/casp15/predictions/<target>/       extracted prediction archives
    data/raw/casp15/natives/<PDB>.cif           RCSB entries listed on the CASP15 target page

Writes:
    data/raw/casp15/natives_clean/<target>/<PDB>_<chain>.pdb   one RNA chain, model 1 only
    data/raw/casp15/targets.csv    target_id,sequence,candidates_dir,...   (candidates = CASP15 predictions)
    data/raw/casp15/natives.csv    target_id,sequence,candidates_dir,...   (candidates = the single native)

Both CSVs are read by src/prepare.py --targets_csv. Paths in them are relative to validation_engine/,
so run prepare.py from that folder.

Why cut out one chain: several natives contain protein chains or two copies of the RNA (e.g. 7QR3 has
RNA chains C and D). prepare.py keeps every RNA residue it finds, so giving it a single chain avoids
mixing copies. Why model 1: NMR/cryo-EM ensembles (8UYS, 8UYE) have 20-30 models; the deposited first
model is used as the reference.

Usage (from validation_engine/):  python src/build_casp15.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import (extract_rna_residues, map_candidate_to_target, read_models)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "casp15"

# PDB codes: CASP15 target list, https://predictioncenter.org/casp15/targetlist.cgi (checked 2026-11-09).
# Chains: chosen by running this script's checks (the chain whose RNA matches the CASP sequence, first copy).
# seq_id = record in casp15.seq.txt. R1117v2, RT1189 and RT1190 have no record of their own; their
# prediction files carry exactly the R1117 / R1189 / R1190 sequence (checked: coverage 1.00, 0 mismatches).
TARGETS = [
    # target    seq_id     PDB     chain  note
    ("R1107",   "R1107",   "7QR4", "B",   "RNA-protein crystal; RNA chain only"),
    ("R1108",   "R1108",   "7QR3", "C",   "RNA-protein crystal; chain D is a second RNA copy"),
    ("R1116",   "R1116",   "8S95", "C",   "native models 146 of 157 nt"),
    ("R1117",   "R1117",   "8FZA", "A",   "native models 29 of 30 nt; chain B is a second copy"),
    ("R1117v2", "R1117",   "8FZA", "A",   "same RNA as R1117; LG (ligand-only) files are skipped"),
    ("R1126",   "R1126",   "8TVZ", "C",   ""),
    ("R1128",   "R1128",   "8BTZ", "A",   ""),
    ("R1136",   "R1136",   "7ZJ4", "E",   ""),
    ("R1138",   "R1138v1", "7PTK", "B",   "CASP lists 7PTK and 7PTL; 7PTK used"),
    ("R1149",   "R1149",   "8UYS", "A",   "ensemble; model 1 used"),
    ("R1156",   "R1156",   "8UYE", "A",   "CASP lists 8UYE, 8UYG, 8UYJ; 8UYE model 1 used"),
    ("R1189",   "R1189",   "7YR7", "A",   "RNA-protein complex; RNA chain only"),
    ("R1190",   "R1190",   "7YR6", "A",   "RNA-protein complex; RNA chain only"),
    ("RT1189",  "R1189",   "7YR7", "A",   "same RNA as R1189 (RNA-protein category)"),
    ("RT1190",  "R1190",   "7YR6", "A",   "same RNA as R1190 (RNA-protein category)"),
]

CSV_COLUMNS = ["target_id", "sequence", "candidates_dir", "seq_source", "native_pdb", "native_chain", "note"]


def read_fasta(path: Path) -> dict:
    seqs, name = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name and line.strip():
            seqs[name] += line.strip().upper()
    return seqs


def write_native_chain(pdb_id: str, chain: str, out_path: Path, target_seq: str) -> str:
    """Write model 1, one chain, as PDB. Returns a short check string (coverage / mismatches)."""
    from biotite.structure.io import pdb as bpdb

    atoms = read_models(RAW / "natives" / f"{pdb_id}.cif")[0]
    atoms = atoms[atoms.chain_id == chain]
    if len(atoms) == 0:
        raise ValueError(f"{pdb_id}: chain {chain} not found")
    seq = "".join(r.nt for r in extract_rna_residues(atoms).residues)
    m = map_candidate_to_target(target_seq, seq)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = bpdb.PDBFile()
    f.set_structure(atoms)
    f.write(str(out_path))
    return f"len {len(seq)}, coverage {m.coverage:.2f}, mismatches {m.n_mismatch}"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def main() -> int:
    seqs = read_fasta(RAW / "casp15.seq.txt")
    t_rows, n_rows = [], []
    for tid, seq_id, pdb_id, chain, note in TARGETS:
        seq = seqs[seq_id]
        pred_dir = RAW / "predictions" / tid
        if not pred_dir.is_dir():
            print(f"[{tid}] WARNING predictions folder missing: {rel(pred_dir)}; target left out")
            continue
        nat_dir = RAW / "natives_clean" / tid
        check = write_native_chain(pdb_id, chain, nat_dir / f"{pdb_id}_{chain}.pdb", seq)
        n_files = sum(1 for _ in pred_dir.iterdir())
        print(f"[{tid}] seq {seq_id} ({len(seq)} nt), {n_files} prediction files, "
              f"native {pdb_id}:{chain} -> {check}")
        common = {"target_id": tid, "sequence": seq, "seq_source": seq_id,
                  "native_pdb": pdb_id, "native_chain": chain, "note": note}
        t_rows.append({**common, "candidates_dir": rel(pred_dir)})
        n_rows.append({**common, "candidates_dir": rel(nat_dir)})

    for name, rows in (("targets.csv", t_rows), ("natives.csv", n_rows)):
        with open(RAW / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {rel(RAW / name)} ({len(rows)} targets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
