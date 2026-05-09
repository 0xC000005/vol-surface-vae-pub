# World Model HEAD035: Direct Predictor Tradeoff Analysis

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Question

What should be done with the HEAD034 direct predictor result, given that it
improves retrieval/top-k but worsens fixed-target MSE and decoded delta MSE?

## Evidence

HEAD028 fixed-PCA predictor:

```text
epoch                  25
target-space MSE       0.987130
MRR                    0.096374
top1                   0.035156
top5                   0.132031
top10                  0.203125
decoded delta MSE      0.015369
pred effective rank    3.863753
```

HEAD034 MSE-selected checkpoint:

```text
epoch                  16
target-space MSE       1.093260
MRR                    0.098682
top1                   0.036719
top5                   0.132812
top10                  0.203125
decoded delta MSE      0.016982
pred effective rank    5.273018
context rank           11.586305
```

HEAD034 retrieval-selected checkpoint:

```text
epoch                  15
target-space MSE       1.093676
MRR                    0.110811
top1                   0.049219
top5                   0.150781
top10                  0.221875
decoded delta MSE      0.016992
pred effective rank    5.180739
context rank           11.372300
```

HEAD034 best decoded-delta checkpoint:

```text
epoch                  9
target-space MSE       1.118770
MRR                    0.092990
top5                   0.125000
top10                  0.198438
decoded delta MSE      0.016267
pred effective rank    4.686083
context rank           10.751725
```

## Interpretation

HEAD034 is not a candidate model because it loses the MSE and decoded-delta
surface. But it is useful evidence:

- direct flattened-past features expose more retrieval signal than the GRU-only
  HEAD028 predictor;
- the signal appears as higher effective rank and better nearest-neighbor
  ordering;
- the same representation is less accurate in fixed PCA coordinates and decodes
  worse to frame deltas.

This is not solved by checkpoint selection. The best decoded-delta checkpoint
still trails HEAD028, and the retrieval-selected checkpoint worsens decoded MSE.

The next move should therefore not be a retrieval loss or a target sweep. The
clean question is whether a context architecture can preserve HEAD028's
coordinate accuracy while borrowing HEAD034's direct high-rank signal.

## Decision

Run one architecture-only experiment:

```text
past window
-> GRU context branch
-> direct flattened-past branch
-> fused context
-> horizon predictor
-> fixed z_pca

loss = MSE(predicted z_pca, fixed z_pca)
```

Keep the rest fixed:

- same fixed delta-PCA target contract;
- same horizons `(1, 5, 10, 20, 30)`;
- same target dim `8`;
- same train/validation window counts;
- no target encoder;
- no retrieval/neighborhood loss;
- no target-dimension sweep;
- no decoder.

Falsifier for the next experiment:

- it does not beat HEAD028 on MSE or decoded delta MSE;
- or it cannot preserve HEAD034's retrieval/top-k gain.

If the fused architecture improves both surfaces, it becomes the next Part 1
reference. If it only improves retrieval while damaging MSE, treat the problem
as a representation tradeoff that needs a principled gate definition before
more model work.

## Artifacts

- `experiments/world/reports/world_model_head035_direct_predictor_analysis.md`
