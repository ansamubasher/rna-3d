# Validation engine — evaluation summary

This report was generated automatically by `src/evaluate.py`. Each RNA target has many predicted structures (candidates). Every method picks the candidates it believes are best, and we check those picks against the true quality: the TM-score against the experimental structure (1 = identical, above about 0.45 usually means the same overall fold).

* Targets evaluated: 13 (all), 6 (excluding possible gRNAde training overlap)
* Excluded RNA–protein complexes: none listed
* Possible training-overlap targets: R1107, R1108, R1116, R1117, R1117v2, R1189, R1190

## Results — subset `all`

| Method | Targets | Top-1 TM (mean / median) | Best-of-5 TM (mean / median) | Selection loss (mean) | Spearman (mean) | Pooled Spearman |
|---|---|---|---|---|---|---|
| Oracle (best possible pick) | 13 | 0.552 / 0.564 | 0.552 / 0.564 | 0.000 | 1.000 | 1.000 |
| Random pick (expected) | 13 | 0.279 / 0.285 | 0.393 / 0.390 | 0.274 | 0.000 | n/a |
| gRNAde raw perplexity | 13 | 0.464 / 0.476 | 0.505 / 0.494 | 0.088 | 0.519 | 0.561 |
| gRNAde normalised (ours) | 13 | 0.464 / 0.476 | 0.505 / 0.494 | 0.088 | 0.519 | 0.511 |

## Results — subset `no_overlap`

| Method | Targets | Top-1 TM (mean / median) | Best-of-5 TM (mean / median) | Selection loss (mean) | Spearman (mean) | Pooled Spearman |
|---|---|---|---|---|---|---|
| Oracle (best possible pick) | 6 | 0.649 / 0.631 | 0.649 / 0.631 | 0.000 | 1.000 | 1.000 |
| Random pick (expected) | 6 | 0.281 / 0.297 | 0.416 / 0.397 | 0.368 | 0.000 | n/a |
| gRNAde raw perplexity | 6 | 0.509 / 0.485 | 0.570 / 0.543 | 0.140 | 0.602 | 0.620 |
| gRNAde normalised (ours) | 6 | 0.509 / 0.485 | 0.570 / 0.543 | 0.140 | 0.602 | 0.588 |

### How to read this

* **Top-1 TM**: quality of the single structure the method would pick. Higher is better.
* **Best-of-5 TM**: quality of the best structure among the 5 the method shows the user.
* **Selection loss**: how much TM-score is lost compared with a perfect pick (0 = perfect).
* **Spearman**: how well the method orders the candidates of one RNA (1 = perfect order, 0 = random).
* **Pooled Spearman**: the same over all candidates of all RNAs together. This is where comparing raw perplexities across different RNAs hurts, and where normalisation should help.
* **Raw vs normalised gRNAde:** inside one RNA, the z-score keeps the order of the perplexities, so the two give identical per-target picks by construction. The normalisation matters for pooled comparisons, for confidence, and for combining with lociPARSE.

## Paired comparisons (Wilcoxon signed-rank on per-target top-1 TM-score)

| Comparison | Subset | Targets | Mean difference | A better / B better / ties | p-value | Note |
|---|---|---|---|---|---|---|
| gRNAde normalised (ours) vs Random pick (expected) | all | 13 | 0.185 | 12 / 1 / 0 | 0.0005 |  |
| gRNAde normalised (ours) vs gRNAde raw perplexity | all | 13 | 0.000 | 0 / 0 / 13 | n/a | identical on every target (no test possible) |
| gRNAde normalised (ours) vs Random pick (expected) | no_overlap | 6 | 0.228 | 5 / 1 / 0 | 0.0625 |  |
| gRNAde normalised (ours) vs gRNAde raw perplexity | no_overlap | 6 | 0.000 | 0 / 0 / 6 | n/a | identical on every target (no test possible) |

A p-value below 0.05 means the difference is unlikely to be chance. With only about a dozen targets the test has little power, so a non-significant result does not show that the methods are equal.

## Confidence calibration (leave-one-target-out)

* Truth label: TM-score of gRNAde's top pick ≥ 0.45 → high, ≥ 0.3 → medium, otherwise low.
* Predictor: raw perplexity of gRNAde's top pick.
* LOTO accuracy: **0.538** over 13 targets (each target was predicted with thresholds fitted on the other targets only).
* Final thresholds fitted on all targets: high if ppl ≤ 2.232, medium if ≤ 2.232, otherwise low (training accuracy 0.769).
* With so few targets these thresholds are rough. Treat the confidence label as a coarse guide.
