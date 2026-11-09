# Score all prepared CASP15 candidates with gRNAde, using several CPU processes in parallel.
#
# Run from PowerShell inside validation_engine:
#     .\run_scoring.ps1                 # 8 parallel processes
#     .\run_scoring.ps1 -Workers 12     # more, if you have the cores
#
# Already-scored candidates are read from the cache, so it is safe to stop and re-run.
# Logs: results\logs\score_shard_<i>.out
# Final output: data\processed\grnade_scores.csv  (used by rank.py and evaluate.py)

param([int]$Workers = 8, [int]$ThreadsPerWorker = 2)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "tools\venvs\rna\Scripts\python.exe"
New-Item -ItemType Directory -Force "results\logs" | Out-Null

# 1. Split the manifest into one shard per worker
& $py src\make_shards.py --n $Workers

# 2. Start one scoring process per shard (each limited to a few threads so they don't fight)
$env:OMP_NUM_THREADS = "$ThreadsPerWorker"
$env:MKL_NUM_THREADS = "$ThreadsPerWorker"
$procs = foreach ($i in 0..($Workers - 1)) {
    Start-Process -FilePath $py -NoNewWindow -PassThru `
        -ArgumentList "src\score_grnade.py", "--manifest", "data\processed\shards\manifest_$i.csv",
                      "--out_csv", "data\processed\shards\grnade_scores_$i.csv" `
        -RedirectStandardOutput "results\logs\score_shard_$i.out" `
        -RedirectStandardError  "results\logs\score_shard_$i.err"
}
Write-Host "Started $Workers workers. Waiting... (progress: Get-Content results\logs\score_shard_0.err -Tail 1)"
$procs | Wait-Process

# 3. Merge: one more pass over all candidates; everything is cached, so this only writes the CSV
& $py src\score_grnade.py --manifest data\processed\shards\manifest_eval.csv
Write-Host "Done: data\processed\grnade_scores.csv"
