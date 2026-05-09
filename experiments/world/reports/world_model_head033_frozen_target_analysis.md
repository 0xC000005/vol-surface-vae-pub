# World Model HEAD033: Frozen Target Predictor Analysis

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Question

Why did HEAD032's frozen distilled target encoder not improve the JEPA context
predictor over the HEAD028 fixed-PCA predictor, even though HEAD031 showed the
target encoder can imitate the fixed target contract?

## Evidence

HEAD032 target encoder quality versus fixed PCA stayed strong on every horizon:

```text
horizon  target-MSE  target-MRR  top5      decoded-MSE  rank
1        0.008850    0.992188    1.000000  0.000955     3.024002
5        0.015953    0.974935    1.000000  0.001780     3.380905
10       0.017700    0.987630    1.000000  0.001803     3.297969
20       0.013536    0.998047    1.000000  0.002626     3.418469
30       0.014493    1.000000    1.000000  0.003057     3.338669
```

Per-horizon predictor comparison:

```text
horizon  H28-MSE   H28-MRR  H28-top5  H28-decMSE  H28-rank  H32-MSE   H32-MRR  H32-top5  H32-decMSE  H32-rank
1        0.642477  0.081196 0.117188  0.009832    3.729984  0.643060  0.078426 0.101562  0.010226    3.394946
5        0.851433  0.088257 0.128906  0.013317    3.800845  0.854622  0.077070 0.109375  0.013521    3.549734
10       1.010910  0.104040 0.144531  0.015791    3.791976  1.012193  0.083298 0.125000  0.015972    3.534849
20       1.195735  0.099184 0.121094  0.018698    3.729061  1.175904  0.084456 0.109375  0.018526    3.510448
30       1.235094  0.109192 0.148438  0.019206    3.737418  1.210571  0.102448 0.117188  0.019264    3.462891
```

Checkpoint-selection audit:

```text
HEAD028 best MSE epoch: epoch 23, MSE 0.986971, MRR 0.091640, top5 0.128906, decoded MSE 0.015164
HEAD028 best MRR/top5 epoch: epoch 25, MSE 0.987130, MRR 0.096374, top5 0.132031, decoded MSE 0.015369

HEAD032 best MSE epoch: epoch 19, MSE 0.979270, MRR 0.085140, top5 0.112500, decoded MSE 0.015502
HEAD032 best MRR epoch: epoch 24, MSE 1.000972, MRR 0.089344, top5 0.116406, decoded MSE 0.015935
HEAD032 best top5 epoch: epoch 17, MSE 0.988635, MRR 0.088808, top5 0.120313, decoded MSE 0.015496
```

## Interpretation

The target encoder is not the active failure. HEAD032's frozen target encoder
is close to fixed PCA across horizons. The context predictor is the limiting
piece.

The predictor also does not show the old HEAD025 collapse pattern. Its
effective rank stays around `3.4` to `3.8`, and the context rank reaches about
`4.7` to `4.9`. So the failure is not simple rank collapse either.

The issue is more specific:

- short/mid horizons are slightly worse than HEAD028 on MSE and clearly worse
  on retrieval;
- long horizons are slightly better on MSE but not on retrieval or decoded MSE;
- selecting by MRR or top5 does not recover a hidden better checkpoint;
- target-space anchoring helps the target encoder, but does not increase the
  recoverable signal in the past-window context representation.

This points to a `latent_prediction` bottleneck in the mapping from past window
to future-delta target coordinates, not a decoder problem and not a target
encoder anchoring problem.

## Decision

Do not add another loss or target sweep. HEAD032 should remain a negative bridge
diagnostic.

The next useful step is a predictability/capacity audit against the fixed
delta-PCA target space:

```text
past window -> candidate predictor/probe -> fixed z_pca
```

The audit should answer one question: is HEAD028 capped because the GRU context
predictor is too weak, or because the fixed future-delta target coordinates are
only weakly predictable from the available past window?

Keep it bounded:

- no decoder;
- no target encoder;
- no retrieval/neighborhood objective;
- no target-dimension sweep;
- use the same fixed PCA contract and validation slice;
- compare a simple direct probe/predictor to the HEAD028 GRU predictor on the
  same MSE, retrieval, decoded-MSE, and rank metrics.

If a direct predictor cannot beat HEAD028, the next paradigm shift should be
context-object or data-object redesign rather than another objective patch. If
it can beat HEAD028, the next experiment should port only that predictor/context
change back into the JEPA path.

## Artifacts

- `experiments/world/reports/world_model_head033_frozen_target_analysis.md`
