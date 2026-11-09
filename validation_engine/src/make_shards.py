"""Split the candidate manifest into N shards so gRNAde scoring can run in N parallel processes.

score_grnade.py runs one candidate at a time and mostly uses one CPU core, so on a multi-core PC
running several copies on different shards is much faster. Each copy writes the same per-candidate
cache (data/processed/grnade_scores/<target>/<candidate>.json); no candidate is in two shards,
so they never write the same file.

Candidates are sorted by length and dealt out round-robin, so every shard gets a similar mix of
short and long RNAs (long ones are the slow ones).

Writes  <out_dir>/manifest_eval.csv  (all scorable candidates, minus excluded targets)
        <out_dir>/manifest_<i>.csv   (shard i, i = 0..N-1)

Usage:  python src/make_shards.py --n 8
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of shards / parallel processes")
    ap.add_argument("--manifest", default=str(ROOT / "data/processed/manifest.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "data/processed/shards"))
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    skip = set(cfg["data"].get("excluded_duplicate_targets") or [])

    m = pd.read_csv(args.manifest)
    ok = m[(m["status"] == "ok") & (~m["target_id"].isin(skip))]
    ok = ok.sort_values("target_length", ascending=False).reset_index(drop=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok.to_csv(out / "manifest_eval.csv", index=False)
    for i in range(args.n):
        ok.iloc[i::args.n].to_csv(out / f"manifest_{i}.csv", index=False)
    print(f"{len(ok)} candidates in {args.n} shards -> {out} (left out: {sorted(skip) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
