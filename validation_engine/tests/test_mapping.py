"""Unit tests for src/prepare.py: residue mapping, name normalisation, format detection,
and an end-to-end run on a small synthetic structure (no downloads needed)."""
import csv
import json
from pathlib import Path

import pytest

from prepare import (align_semiglobal, clean_sequence, detect_format, map_candidate_to_target,
                     normalize_atom_name, normalize_resname, prepare_target, read_sequence_arg)

TARGET = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGG"


# --- alignment / mapping ---------------------------------------------------------------

def test_identical_sequence_maps_one_to_one():
    m = map_candidate_to_target(TARGET, TARGET)
    assert m.pairs == [(i, i) for i in range(len(TARGET))]
    assert m.coverage == 1.0 and m.n_mismatch == 0 and m.n_extra == 0


def test_single_mismatch_is_counted():
    cand = TARGET[:10] + ("A" if TARGET[10] != "A" else "C") + TARGET[11:]
    m = map_candidate_to_target(TARGET, cand)
    assert m.coverage == 1.0 and m.n_mismatch == 1
    assert m.mismatch_fraction == pytest.approx(1 / len(TARGET))


def test_missing_termini_are_free_and_offset_is_correct():
    cand = TARGET[3:-2]
    m = map_candidate_to_target(TARGET, cand)
    assert m.pairs[0] == (3, 0) and m.pairs[-1] == (len(TARGET) - 3, len(cand) - 1)
    assert m.coverage == pytest.approx(len(cand) / len(TARGET)) and m.n_mismatch == 0


def test_extra_tail_residues_are_counted_not_aligned():
    m = map_candidate_to_target(TARGET, TARGET + "AAAAA")
    assert m.coverage == 1.0 and m.n_extra == 5


def test_internal_deletion_is_one_contiguous_gap():
    # Deleting TARGET[20:23] = "AGA" inside the repetitive stretch "GGGAGAG".
    # Inside a repeat, the exact place of the gap is ambiguous from sequence alone,
    # but affine gap costs must make it ONE block of 3 consecutive target positions.
    cand = TARGET[:20] + TARGET[23:]
    m = map_candidate_to_target(TARGET, cand)
    unmapped = sorted(set(range(len(TARGET))) - {ti for ti, _ in m.pairs})
    assert len(unmapped) == 3 and unmapped == list(range(unmapped[0], unmapped[0] + 3))
    assert m.n_mismatch == 0 and len(m.pairs) == len(cand)


def test_internal_deletion_in_unique_sequence_is_exact():
    target = "ACGUUGCAAGCUAGCGUAACG"
    cand = target[:8] + target[12:]                 # delete 4 residues, unique context
    m = map_candidate_to_target(target, cand)
    unmapped = sorted(set(range(len(target))) - {ti for ti, _ in m.pairs})
    assert unmapped == [8, 9, 10, 11] and m.n_mismatch == 0


def test_alignment_is_deterministic():
    assert align_semiglobal(TARGET, TARGET[5:30]) == align_semiglobal(TARGET, TARGET[5:30])


def test_unrelated_sequence_has_low_coverage_or_many_mismatches():
    m = map_candidate_to_target("ACGUACGUACGUACGUACGU", "GGGGGGGGGG")
    assert m.coverage < 0.9 or m.mismatch_fraction > 0.05


# --- names, sequence, format -------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [("A", "A"), ("  G", "G"), ("RU", "U"), ("CYT", "C"),
                                           ("RA5", "A"), ("PSU", None), ("HOH", None), ("DA", None)])
def test_normalize_resname(name, expected):
    assert normalize_resname(name) == expected


@pytest.mark.parametrize("name,expected", [("C4*", "C4'"), ("O1P", "OP1"), ("O2P", "OP2"), ("N9", "N9")])
def test_normalize_atom_name(name, expected):
    assert normalize_atom_name(name) == expected


def test_clean_sequence():
    assert clean_sequence(" acg t\nu ") == "ACGUU"
    with pytest.raises(ValueError):
        clean_sequence("   ")


def test_read_sequence_from_multi_record_fasta(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(">chainA\nGGCA\nCU\n>chainB\nUUCG\n", encoding="utf-8")
    assert read_sequence_arg(str(f)) == "GGCACUUUCG"


def test_detect_format():
    assert detect_format(Path("x.pdb"), "") == "pdb"
    assert detect_format(Path("x.cif.gz"), "") == "cif"
    assert detect_format(Path("R1107TS001_1"), "PFRMAT TS\nATOM      1  P  ...") == "pdb"
    assert detect_format(Path("noext"), "data_1ABC\n_entry.id 1ABC") == "cif"
    assert detect_format(Path("junk"), "hello") is None


# --- end-to-end on a synthetic structure ----------------------------------------------------------

def _atom(serial, name, resn, chain, resi, x, y, z, element):
    return (f"ATOM  {serial:5d} {name:<4s} {resn:>3s} {chain}{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")


def _write_pdb(path, seq, start_resi=1, models=1):
    """Toy RNA: per residue P, C4' and N1/N9, spread along a line. Biologically
    meaningless, but enough to test parsing, model splitting and mapping."""
    lines, s = [], 1
    for mdl in range(1, models + 1):
        if models > 1:
            lines.append(f"MODEL     {mdl:4d}\n")
        for i, nt in enumerate(seq):
            base_atom = "N9" if nt in "AG" else "N1"
            x = 6.0 * i + mdl
            for name, dx, el in (("P", 0.0, "P"), ("C4'", 1.5, "C"), (base_atom, 3.0, "N")):
                lines.append(_atom(s, name, nt, "A", start_resi + i, x + dx, 0.0, 0.0, el))
                s += 1
        if models > 1:
            lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


CFG = {"prepare": {"min_coverage": 0.9, "max_mismatch_fraction": 0.05,
                   "grnade_max_len": 500, "split_multi_model": True}}


def test_prepare_target_end_to_end(tmp_path):
    seq = "GGCACUUCGGUGCC"
    cands = tmp_path / "cands"
    cands.mkdir()
    _write_pdb(cands / "renumbered.pdb", seq, start_resi=101)          # numbering differs
    _write_pdb(cands / "R0000TS001_1", seq, models=3)                   # no extension, 3 models
    _write_pdb(cands / "short.pdb", seq[3:])                            # coverage 11/14 < 0.9
    (cands / "junk.dat").write_text("not a structure", encoding="utf-8")

    rows = prepare_target("T", seq, cands, tmp_path / "out", CFG)
    by_id = {r["candidate_id"]: r for r in rows}

    assert {"R0000TS001_1_m1", "R0000TS001_1_m2", "R0000TS001_1_m3"} <= set(by_id)
    assert by_id["renumbered"]["status"] == "ok"
    assert by_id["short"]["status"] == "skipped" and "coverage" in by_id["short"]["skip_reason"]
    # an unknown extension is kept in the id ("junk.dat"), like extension-less CASP files
    assert by_id["junk.dat"]["status"] == "skipped" and "parse_error" in by_id["junk.dat"]["skip_reason"]

    # The mapping keeps the ORIGINAL residue numbers and maps them to target positions 1..14
    m = json.loads(Path(by_id["renumbered"]["map_json"]).read_text(encoding="utf-8"))
    assert m["orig_res_id"] == list(range(101, 115))
    assert m["target_index"] == list(range(14)) and m["target_nt"] == seq

    # The clean PDB is numbered by target position (1-based), as one chain 'A'
    clean = Path(by_id["renumbered"]["clean_pdb"]).read_text(encoding="utf-8").splitlines()
    atoms = [ln for ln in clean if ln.startswith("ATOM")]
    assert {ln[21] for ln in atoms} == {"A"}
    assert sorted({int(ln[22:26]) for ln in atoms}) == list(range(1, 15))
