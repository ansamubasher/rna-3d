# gRNAde Validation Engine

Final Year Project (FYP-1) component of an RNA 3D structure prediction pipeline built around the
Stanford RNA 3D Folding (Kaggle) task.

```
RNA sequence
  -> template retrieval (teammates)
  -> 3D structure prediction (RNAPro; DRfold2 / RhoFold+)
  -> N candidate 3D structures
  -> VALIDATION ENGINE (this repo): score, rank, top-5 + confidence
  -> web app 3D viewer
```

Given **one RNA sequence** and **a folder of candidate 3D structures from any predictor**, the engine
scores each candidate, ranks them, and returns the top 5 with a confidence label. It works with any
predictor: RNAPro output plugs in later with no code changes.

## Method in one paragraph

gRNAde (Joshi et al., ICLR 2025) is an RNA *inverse folding* model: from a 3D backbone it predicts a
probability for A/G/C/U at every position. If a candidate structure is right, the true sequence should
fit it well. We compute the **teacher-forced perplexity** of the given sequence on each candidate's
backbone, using gRNAde's own training forward pass (no sampling, no training, pretrained checkpoint used
zero-shot). Perplexities are only compared **within one target**, following the relative-likelihood idea
of ESM-IF (Hsu et al., ICML 2022):

```
z_i = (ppl_i - mean(ppl of this target)) / std(ppl of this target)
final_score_i = -z_i          # higher = better
```

If a target has fewer than 3 valid candidates, or std = 0, the engine falls back to `final_score = -ppl`
and logs a warning.

## Repository layout

```
validation_engine/
  README.md            this file
  environment.md       exact install steps (Windows), versions, commits, checkpoint
  config.yaml          every path, threshold and seed
  validate.py          PUBLIC ENTRY POINT (Step 6)
  src/
    prepare.py         Step 3  parse PDB/mmCIF(.gz), split models, map residues, clean PDBs, manifest
    score_grnade.py    Step 4  gRNAde teacher-forced perplexity (+ --sanity checks)
    rank.py            Step 5  within-target z-score, ranking, top-k, confidence
    label_usalign.py   Step 7  TM-score / RMSD vs native with US-align (evaluation only)
    score_lociparse.py Step 8  lociPARSE baseline (runs in its own conda env)
    evaluate.py        Step 8  metrics, Wilcoxon tests, LOTO confidence calibration, plots, summary.md
  tests/               unit tests: normalisation, metrics, residue mapping, edge cases
  data/                raw + processed (gitignored)
  results/             metrics.csv, per_target.csv, plots/, summary.md
```

## Setup

Follow [environment.md](environment.md). In short, there are two conda environments:

* `rna`: Python 3.10, torch 2.1, PyG, gRNAde @ `2453b18`, checkpoint `gRNAde_drop3d@0.75_maxlen@500.h5`
* `lociparse`: Python 3.10, torch 1.12, lociPARSE @ `8c7acbe` (called automatically by `score_lociparse.py`)

plus `USalign.exe` built from source with MinGW-w64 (evaluation only). Paths are set in `config.yaml`.

## Using the engine (no ground truth needed)

```powershell
conda activate rna
python validate.py --sequence query.fasta --candidates path\to\predicted_models --top_k 5 --out results.json
```

`results.json` contains:

| Field | Meaning |
|---|---|
| `ranked` | every valid candidate, best first: file, model index, rank, `final_score`, `z_score`, raw `perplexity`, coverage, flags |
| `top_k` | the first k of `ranked` |
| `confidence` | `high` / `medium` / `low`, or `uncalibrated` until Step 8 has set the thresholds |
| `best_candidate_per_nucleotide` | for the #1 candidate: original chain + residue number and p(true nucleotide) per residue, for colouring in the 3D viewer |
| `skipped`, `warnings` | rejected candidates (with stage and reason) and caveats |

## Reproducing the evaluation (CASP15)

**Data.** The CASP15 Prediction Center download area
(https://predictioncenter.org/download_area/CASP15/) has, as checked on 2026-11-08:
* `predictions/RNA/` with one archive per target: R1107, R1108, R1116, R1117 (+ R1117v2), R1126, R1128,
  R1136, R1138, R1149, R1156, R1189, R1190, RT1189, RT1190
* `targets/` with experimental structures (e.g. `casp15.targets.TS-domains.public_12.20.2022.tar.gz`,
  `casp15.targets.oligo.public_12.20.2022.tar.gz`)
* `sequences/casp15.seq.txt`

The archives' internal layout has not been inspected yet. That is Step 2, which also decides which
targets are RNA–protein complexes (`data.excluded_rna_protein_targets`) and which may overlap with gRNAde's
training data (RNASolo, ≤ 4 Å, downloaded 2023-10-31; `data.possible_training_overlap_targets`).

**Pipeline** (all in the `rna` env):

```powershell
# 2. build targets.csv / natives.csv (natives = RCSB mmCIFs in data/raw/casp15/natives, one RNA chain cut out)
python src/build_casp15.py

# 3. preprocess candidates and natives (targets.csv / natives.csv: target_id,sequence,candidates_dir)
python src/prepare.py --targets_csv data/raw/casp15/targets.csv
python src/prepare.py --targets_csv data/raw/casp15/natives.csv `
    --out_dir data/processed/natives --manifest data/processed/natives_manifest.csv

# 4. gRNAde scores + sanity checks on natives (native ppl < 4 and < shuffled)
python src/score_grnade.py
python src/score_grnade.py --sanity --manifest data/processed/natives_manifest.csv

# 5. ranking
python src/rank.py

# 7. ground truth (evaluation only)
python src/label_usalign.py --natives_manifest data/processed/natives_manifest.csv

# 8. baseline + evaluation
python src/score_lociparse.py
python src/evaluate.py --write_config
```

**Metrics** (per target, then mean and median): Spearman (score vs TM-score), top-1 TM-score, best-of-5
TM-score, selection loss (best available − top-1), plus pooled Spearman over all candidates. **Methods:**
oracle, random (exact expectation), gRNAde raw perplexity, gRNAde normalised (ours), lociPARSE pMoL, and the
average of z-scored gRNAde and lociPARSE. Paired Wilcoxon signed-rank tests on per-target top-1 TM-score.
Results are reported with and without the possible training-overlap targets.

> Note: inside one target, z-scoring keeps the order of the perplexities, so *raw* and *normalised* gRNAde
> give identical per-target picks by construction. Normalisation matters for pooled comparisons, for
> confidence calibration, and for combining with lociPARSE. `summary.md` states this.

## Tests

```powershell
conda activate rna
pip install pytest
python -m pytest tests -q
```

## Current status

| Part | Status |
|---|---|
| prepare.py | tested on public PDB entries 1EHZ and 2KOC (PDB, mmCIF, .gz, multi-model, extension-less, renumbered, truncated, junk) |
| score_grnade.py | written against gRNAde @ `2453b18` source; **not yet run** (needs the `rna` env + checkpoint) |
| rank.py, validate.py | tested (validate.py end-to-end with a stand-in scorer) |
| label_usalign.py | parser/caching tested against US-align's `-outfmt 2` format from source; **real binary not yet run** |
| score_lociparse.py | lociPARSE's own parser verified on our input files; **model not yet run** |
| evaluate.py | unit-tested on toy data; **no CASP15 results yet** |

## References

1. Joshi et al. (2025). gRNAde: Geometric Deep Learning for 3D RNA Inverse Design. ICLR 2025. arXiv:2305.14749. https://github.com/chaitjo/geometric-rna-design
2. Hsu et al. (2022). Learning inverse folding from millions of predicted structures. ICML 2022.
3. Tarafder & Bhattacharya (2024). lociPARSE: A Locality-aware Invariant Point Attention Model for Scoring RNA 3D Structures. J. Chem. Inf. Model. 64(22), 8655–8664. DOI 10.1021/acs.jcim.4c01621
4. Das et al. (2023). Assessment of three-dimensional RNA structure prediction in CASP15. Proteins 91(12), 1747–1770. DOI 10.1002/prot.26602
5. Zhang et al. (2022). US-align. Nature Methods 19(9), 1109–1115. DOI 10.1038/s41592-022-01585-1
6. Lee et al. (2025). RNAPro (preprint). Pipeline context only; not used here.
