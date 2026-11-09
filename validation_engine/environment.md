# Environment setup — gRNAde Validation Engine (Windows)

This file lists the exact install steps, versions and checkpoint used by the validation engine,
on **native Windows 11 with PowerShell**. Everything here was checked against the upstream
repositories and package indexes on **2026-11-08**.

> **Status (2026-11-09):** the `rna` environment, gRNAde, the checkpoint and US-align are installed and
> verified on this machine. The route used differs slightly from the planned steps (venv instead of conda);
> see "Recorded after install" at the bottom. lociPARSE is not installed yet.

---

## 0. Platform and layout

| Item | Choice | Notes |
|---|---|---|
| OS | Windows 11, native (no WSL) | gRNAde's own docs only test Linux and macOS. **Windows is untested upstream**, so see the "Windows risks" section below. |
| Shell | PowerShell | Every command below is PowerShell. |
| Env manager | Miniforge (includes `conda` and `mamba`) | gRNAde docs use `mamba`. |
| GPU | Optional | Teacher-forced scoring is one forward pass per candidate, so CPU is enough for CASP15 sizes. |

We checked that **Windows (`win_amd64`, Python 3.10) builds exist** for every pinned package:

| Package | Version | Windows wheel |
|---|---|---|
| torch (gRNAde) | 2.1.0 (+cu121 and +cpu) | yes |
| torch-scatter / torch-sparse / torch-cluster | 2.1.2 / 0.6.18 / 1.6.3 (pt21, cu121 and cpu) | yes, on data.pyg.org |
| cpdb-protein | 0.3.1 | yes |
| arnie, draw_rna | 0.2.7, 0.4.3 | pure Python, so no wheel needed |
| torch (lociPARSE) | 1.12.0+cpu | yes |
| numpy / scipy (lociPARSE) | 1.22.3 / 1.7.3 | yes |
| US-align | none for Windows on conda/bioconda | **build it from source with MinGW-w64** (see Section 3) |

We use **two separate environments** because the dependencies conflict:

| Env | Python | torch | Used for |
|---|---|---|---|
| `rna` | 3.10 | 2.1.0 | gRNAde scoring, plus all engine code (`prepare.py`, `rank.py`, `evaluate.py`, ...) |
| `lociparse` | 3.10 | 1.12.0 | lociPARSE baseline only, called from `score_lociparse.py` through `subprocess` |

lociPARSE's `setup.py` pins `torch==1.12.0` and `numpy==1.22.3`, but gRNAde needs `torch==2.1.0`, so the two cannot share an env.

**Folder layout.** `config.yaml` resolves relative paths against `validation_engine/`, so external tools
go in `validation_engine/tools/`. That folder is gitignored and is never committed.
```
validation_engine/
  config.yaml
  environment.md
  tools/
    geometric-rna-design/     # gRNAde
    lociPARSE/
    USalign/                  # contains USalign.exe after the build
  data/  results/  src/ ...
```

### 0a. Install Miniforge
1. Download and run **`Miniforge3-Windows-x86_64.exe`** from
   https://github.com/conda-forge/miniforge/releases/latest. Install "Just Me" to the default
   `C:\Users\<you>\miniforge3`, which is the path `config.yaml` expects.
2. Open the **Miniforge Prompt** once and run the line below so that `conda activate` works in normal PowerShell:
   ```powershell
   conda init powershell
   ```
3. Close and reopen PowerShell. If you get a script-execution error, run this once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 0b. Git
Git for Windows is already installed on this machine (`git` is on PATH). No action needed.

Throughout this file, `$ROOT` means the engine folder:
```powershell
$ROOT = "C:\Users\AnsaMubasher\tmep2\grenade\validation_engine"
New-Item -ItemType Directory -Force "$ROOT\tools" | Out-Null
```

---

## 1. gRNAde (`rna` env)

**Reference:** Joshi et al., *gRNAde: Geometric Deep Learning for 3D RNA Inverse Design*, ICLR 2025, arXiv:2305.14749.
**Repo:** https://github.com/chaitjo/geometric-rna-design
**Pinned commit:** `2453b18778a1981ac25f314790e5395f1e9ab218` (main, 2025-12-15; latest tag is `v1.0.0`)

These steps follow the repo's `docs/installation.md`, translated to PowerShell, with the commit pinned.

```powershell
cd "$ROOT\tools"
git clone https://github.com/chaitjo/geometric-rna-design.git
cd geometric-rna-design
git checkout 2453b18778a1981ac25f314790e5395f1e9ab218

mamba create -n rna python=3.10 -y
conda activate rna
```

**PyTorch and PyG.** Pick ONE of the two blocks.

GPU (NVIDIA, with a driver that supports CUDA 12.1):
```powershell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric
```
CPU only:
```powershell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric
```

**Other libraries** (as listed upstream):
```powershell
mamba install jupyterlab matplotlib seaborn pandas biopython biotite -c conda-forge -y
pip install wandb gdown pyyaml ipdb python-dotenv tqdm cpdb-protein torchmetrics einops ml_collections mdanalysis MDAnalysisTests draw_rna arnie
```

### 1a. Extra packages for the validation engine (same `rna` env)
```powershell
pip install scipy pytest
```
`scipy` provides the Spearman correlation and the Wilcoxon test in `evaluate.py`. `pytest` runs the unit tests.

### 1b. Pretrained checkpoint

**Source:** HuggingFace model repo `chaitjo/gRNAde` (MIT licence)
**Pinned revision:** `91491fc71584bf37fee8297781edf7df07b1d349` (2025-12-15)
**File we use:** `gRNAde_drop3d@0.75_maxlen@500.h5` (and the `config.yaml` next to it)

The HF repo also holds `ribonanzanet/`, `ribonanzanet_sec_struct/` and `rhofold/` checkpoints.
Upstream downloads all of them, but **the validation engine does not need them**: we only run gRNAde's
encoder/decoder on a fixed sequence and never use the sampling or refolding filters.

```powershell
conda activate rna
pip install -U huggingface_hub
cd "$ROOT\tools\geometric-rna-design"
# The repo is public and not gated, so `hf auth login` is optional.
hf download chaitjo/gRNAde "gRNAde_drop3d@0.75_maxlen@500.h5" config.yaml `
    --revision 91491fc71584bf37fee8297781edf7df07b1d349 `
    --local-dir checkpoints
```
(The upstream docs install the `hf` CLI with a bash script. On Windows, `pip install huggingface_hub` provides the same `hf` command.)

What the checkpoint name means: the model was trained with 3D-feature dropout 0.75 and a maximum length
of 500 nt. Longer candidates are outside its training regime, and `prepare.py` flags them (`grnade_max_len` in `config.yaml`).

### 1c. `.env` file (gRNAde reads it through python-dotenv)
```powershell
cd "$ROOT\tools\geometric-rna-design"
$g = "$ROOT\tools\geometric-rna-design".Replace('\','/')
@"
PROJECT_PATH=$g/
DATA_PATH=$g/data/
WANDB_PROJECT=gRNAde
WANDB_ENTITY=none
WANDB_DIR=$g/
WANDB_MODE=disabled
"@ | Out-File -Encoding utf8 .env
```
- The upstream example uses `export VAR=...` lines (bash style). python-dotenv also accepts plain `VAR=value`, which is what we write here.
- `WANDB_MODE=disabled` stops logging to Weights & Biases, since we only run inference.

### 1d. Training data and the CASP15 overlap check (for Step 2)
From the repo's `docs/data_setup.md`: the training data are **RNASolo** structures at **≤ 4 Å resolution**,
downloaded on **31 October 2023**. The exact PDB IDs per split are in `data/split_ids/`
(the ICLR 2025 splits are `das_split.pt` and `structsim_split_v2.pt`).

CASP15 experimental structures were released during and after 2022, so some may be in RNASolo before the
2023-10-31 cutoff. The overlap check in Step 2 will compare CASP15 target PDB IDs and sequences against
these split files. That check has **not been done yet**.

### 1e. Windows risks for gRNAde (check during Step 4)
gRNAde has not been tested on Windows. Things to watch for:
- **Hard-coded `/` paths or `os.system` shell calls** in data-loading or utility code. We only import the
  model and featurisation code, which should avoid most of this, but we will check each function before using it.
- **Optional external tools** (EternaFold, X3DNA, ViennaRNA) are Linux/macOS builds. We **do not need them**,
  because we score a given sequence instead of designing new ones. They are skipped on purpose.
- If anything fails on Windows and can't be worked around cleanly, the fallback is WSL2 (Ubuntu) with the
  same commands translated to bash. The engine code itself stays the same.

---

## 2. lociPARSE (`lociparse` env, baseline)

**Reference:** Tarafder & Bhattacharya, *J. Chem. Inf. Model.* 64(22), 8655–8664 (2024), DOI 10.1021/acs.jcim.4c01621.
**Repo:** https://github.com/Bhattacharya-Lab/lociPARSE
**Pinned commit:** `8c7acbe4e7c486122a4c261b1ea68fff7247b796` (main, 2024-11-11)

We install from source at the pinned commit. The PyPI package `lociPARSE==0.1.1` exists, but the source
install ships the model weights `QAmodel_lociPARSE.pt` through `package_data`, which makes it easier to check.

```powershell
mamba create -n lociparse python=3.10 -y
conda activate lociparse

# CPU build. Windows wheel torch-1.12.0+cpu-cp310-win_amd64 exists.
pip install torch==1.12.0 --index-url https://download.pytorch.org/whl/cpu

cd "$ROOT\tools"
git clone https://github.com/Bhattacharya-Lab/lociPARSE.git
cd lociPARSE
git checkout 8c7acbe4e7c486122a4c261b1ea68fff7247b796
pip install .
```
`setup.py` pins: numpy 1.22.3, tqdm 4.64.0, torch 1.12.0, scipy 1.7.3, scikit-learn 1.1.1, pandas 1.4.4,
seaborn 0.12.0, matplotlib 3.5.2, openpyxl.

**Usage (Python API, from the README):**
```python
from lociPARSE import lociparse
lp = lociparse()
score = lp.score("R1108.pdb")
score.pMoL.show()        # global score
score.pNuL.show()        # list of per-nucleotide scores
score.save("score.txt")
```
**`score.txt` format (README):** line 1 holds the global pMoL. Every later line holds two columns:
the nucleotide index in the PDB, then that nucleotide's pNuL (predicted lDDT).
The README has no example file, so `score_lociparse.py` will check its parser against a real `score.txt` before trusting it.

**Smoke test** (the repo ships `R1108.pdb`, which is a CASP15 target):
```powershell
conda activate lociparse
cd "$ROOT\tools\lociPARSE"
python -c "from lociPARSE import lociparse; s=lociparse().score('R1108.pdb'); s.pMoL.show(); s.save('score_test.txt')"
Get-Content score_test.txt -TotalCount 3
```

---

## 3. US-align (evaluation only)

**Reference:** Zhang et al., *Nature Methods* 19(9), 1109–1115 (2022), DOI 10.1038/s41592-022-01585-1.
**Repo:** https://github.com/pylelab/USalign (the source behind https://zhanggroup.org/US-align/)
**Pinned commit:** `97325d3aad852f8a4407649f25e697bbaa17e186`. This is the commit gRNAde's own install
docs pin, which keeps the two consistent.

No prebuilt Windows binary is available through conda or bioconda, so we compile it. The US-align
`readme.txt` (at this commit) says it builds with **mingw-w64 version 9.3 or later**, and that a native
Windows build **cannot read `.gz` files**. Candidates must therefore be unzipped first, which `prepare.py`
will handle.

**Install a MinGW-w64 compiler through MSYS2:**
1. Install MSYS2 from https://www.msys2.org/ (default `C:\msys64`), or run `winget install MSYS2.MSYS2`.
2. Open the **"MSYS2 UCRT64"** terminal and run:
   ```bash
   pacman -S --needed mingw-w64-ucrt-x86_64-gcc
   ```
3. Back in PowerShell, check that the compiler works. It must be version 9.3 or later:
   ```powershell
   C:\msys64\ucrt64\bin\g++.exe --version
   ```

**Build US-align:**
```powershell
cd "$ROOT\tools"
git clone https://github.com/pylelab/USalign.git
cd USalign
git checkout 97325d3aad852f8a4407649f25e697bbaa17e186
C:\msys64\ucrt64\bin\g++.exe -static -O3 -ffast-math -o USalign.exe USalign.cpp -lm
.\USalign.exe -h | Select-Object -First 5
```
`-static` bundles the MinGW runtime into the binary, so `USalign.exe` runs without MSYS2 on PATH.

**Quick check** with the two example files that ship in the repo (they are protein, so no `-mol RNA` here):
```powershell
.\USalign.exe PDB1.pdb PDB2.pdb
```

How the engine calls it (`label_usalign.py`):
```powershell
USalign.exe model.pdb native.pdb -mol RNA
```
In that output, `TM-score=... (normalized by length of Structure_2...)` is the score normalised by the
**native/reference** length, and that is the TM-score we report. The parser is checked against real output in Step 7.

---

## 4. Config hook-up

`config.yaml` already points to these locations, relative to `validation_engine/`:
```yaml
tools:
  grnade_repo: tools/geometric-rna-design
  grnade_ckpt: tools/geometric-rna-design/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5
  lociparse_python: C:/Users/AnsaMubasher/miniforge3/envs/lociparse/python.exe
  usalign_bin: tools/USalign/USalign.exe
```
If Miniforge went somewhere else, find the real path with:
```powershell
conda activate lociparse; (Get-Command python).Source
```

---

## 5. Verification commands

Run these after installing, and paste the output into the table below.
```powershell
conda activate rna
python -c "import sys,torch,torch_geometric,Bio,biotite; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), torch_geometric.__version__, Bio.__version__, biotite.__version__)"
git -C "$ROOT\tools\geometric-rna-design" rev-parse HEAD
Get-FileHash -Algorithm SHA256 "$ROOT\tools\geometric-rna-design\checkpoints\gRNAde_drop3d@0.75_maxlen@500.h5"

conda activate lociparse
python -c "import sys,torch,numpy,lociPARSE; print(sys.version.split()[0], torch.__version__, numpy.__version__)"
git -C "$ROOT\tools\lociPARSE" rev-parse HEAD

git -C "$ROOT\tools\USalign" rev-parse HEAD
C:\msys64\ucrt64\bin\g++.exe --version | Select-Object -First 1
```

### Recorded after install (fill in, do not guess)

**What was actually done on this machine (2026-11-09), and why it differs from the steps above:**
- Miniforge could not be downloaded: GitHub release downloads returned `504 Gateway Time-out`, both with
  curl and with `winget install CondaForge.Miniforge3`. Instead, the `rna` environment is a plain **venv**:
  Python 3.10.11 from python.org (installer signed by the Python Software Foundation, installed per-user,
  PATH unchanged), venv at `tools\venvs\rna`. Use it with `tools\venvs\rna\Scripts\python.exe` (or
  `tools\venvs\rna\Scripts\Activate.ps1`). Packages come from the same pip indexes as in Section 1 (CPU block),
  plus `numpy<2`, which torch 2.1 needs. `jupyterlab`, `ipdb` and `MDAnalysisTests` were left out because the engine doesn't need them.
- US-align was built with the MinGW-w64 g++ that ships with Strawberry Perl (`C:\Strawberry\c\bin\g++.exe`,
  13.2.0, which meets the ≥ 9.3 requirement), so MSYS2 was not installed. Build command as in Section 3.
- The checkpoint was fetched with curl from
  `https://huggingface.co/chaitjo/gRNAde/resolve/<revision>/gRNAde_drop3d%400.75_maxlen%40500.h5`
  (plus `config.yaml`) at the pinned revision.
- lociPARSE: not installed yet. `config.yaml` → `tools.lociparse_python` must be updated once it is.

| Item | Value |
|---|---|
| Windows version | Windows 11 Enterprise 10.0.26300 |
| GPU / CUDA (or CPU) | CPU only (no NVIDIA GPU) |
| `rna`: Python / torch / PyG / biopython / biotite / numpy | 3.10.11 / 2.1.0+cpu / 2.8.1 / 1.88 / 1.2.0 / 1.26.4 |
| PyG extensions | torch-scatter 2.1.2+pt21cpu, torch-sparse 0.6.18+pt21cpu, torch-cluster 1.6.3+pt21cpu |
| gRNAde commit | 2453b18778a1981ac25f314790e5395f1e9ab218 |
| Checkpoint SHA256 | D43454DEAEEC773644BFA52385B6FCC8264DB256878FD9CA19ED8711EC62971C (8,726,092 bytes) |
| `lociparse`: Python / torch / numpy | not installed yet |
| lociPARSE commit | not installed yet |
| lociPARSE smoke test pMoL on R1108.pdb | not run yet |
| US-align commit | 97325d3aad852f8a4407649f25e697bbaa17e186 (prints "Version 20230609") |
| MinGW g++ version | 13.2.0 (MinGW-W64 x86_64-ucrt-posix-seh, Strawberry Perl) |
| Date installed | 2026-11-09 |
| gRNAde sanity (Step 4) | 15/15 natives: ppl < 4 and < all 10 shuffles; `results/sanity_grnade.csv` |
