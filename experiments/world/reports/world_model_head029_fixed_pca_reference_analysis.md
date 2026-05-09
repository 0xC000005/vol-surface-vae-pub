# World Model HEAD029: Fixed PCA Reference Analysis

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Question

How should HEAD028 be compared to the supervised delta and persistence
references without mixing incompatible metrics?

## Metric Boundary

HEAD028 has two evaluation surfaces:

- **Fixed-target latent space:** MRR/top-k against whitened PCA horizon-delta
  targets. This is comparable to HEAD025's learned-target retrieval only as a
  target-space diagnostic, not as frame-space retrieval.
- **Decoded delta/frame space:** inverse PCA decoded delta MSE. This is
  comparable to raw persistence frame MSE and supervised delta-frame predictors.

## Evidence

HEAD028 fixed target diagnostic:

```text
fixed-target MRR      0.096374
fixed-target top5     0.132031
fixed-target top10    0.203125
decoded delta MSE     0.015369
pred effective rank   3.863753
target effective rank 3.327211
context rank          4.874274
```

HEAD025 learned delta EMA JEPA:

```text
learned-target MRR    0.042356
pred effective rank   1.262683
```

Raw persistence reference:

```text
frame MSE             0.022376
frame-space MRR       0.052426
frame-space top5      0.086719
frame-space top10     0.135156
```

HEAD007 supervised delta lower bound:

```text
frame MSE             0.015180
frame-space MRR       0.052198
frame-space top5      0.062500
```

HEAD013 fixed-delta context reference:

```text
frame MSE             0.021323
frame-space MRR       0.058137
frame-space top5      0.078906
context rank          5.284253
```

## Interpretation

HEAD028 should not be promoted as "better frame-space retrieval" because its
MRR/top-k are measured in the fixed PCA target space. The correct claim is
narrower and stronger:

- a stable, whitened horizon-delta target space is learnable from past windows;
- it avoids the low-rank learned-target collapse seen in HEAD025;
- decoded delta MSE `0.015369` is close to HEAD007's supervised delta lower
  bound `0.015180` and better than persistence `0.022376`;
- the fixed target contract has enough discriminative structure to support MRR
  `0.096374` in its own latent space.

This makes fixed delta-PCA a valid target-space contract for the next learned
target attempt, not the final representation claim.

## Decision

The next non-ad-hoc learned-target step should imitate the fixed target
contract instead of inventing another objective:

```text
future horizon delta block
-> target encoder
-> z_target
trained/pretrained to match fixed PCA target z_pca
```

Then the JEPA predictor can be evaluated against either:

- frozen fixed PCA targets directly, or
- a frozen learned target encoder that has already been validated against fixed
  PCA targets.

Do not move to the decoder yet. Part 1 still needs a learned target encoder or
fixed target contract story that survives retrieval, rank, and decoded-frame
checks.

## Next Step

Run `research_ideation` for a target-encoder distillation contract:

- classify it under the literature gate;
- keep it to one bounded path;
- do not add retrieval/neighborhood loss;
- do not add a target-dimension sweep.
