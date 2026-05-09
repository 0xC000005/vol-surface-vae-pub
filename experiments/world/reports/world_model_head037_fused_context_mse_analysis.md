# World Model HEAD037: Fused Context MSE Analysis

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Question

HEAD036 improves retrieval/top-k and keeps decoded delta MSE at the HEAD028
level, but has worse whitened fixed-target MSE. Is that MSE regression a reason
to reject HEAD036, or is it concentrated in PCA directions that matter less for
decoded delta quality?

## Component Audit

The fixed target is whitened PCA, so target-space MSE gives equal weight to all
PCA coordinates. Decoded delta error weights each coordinate by its PCA scale.

```text
component  scale     H28 z-MSE  H36 z-MSE  delta z-MSE  H28 raw contrib  H36 raw contrib  delta raw
1          0.460737  0.867833   0.824827  -0.043007    0.00736891      0.00700374      -0.00036518
2          0.304683  0.687009   0.735544   0.048534    0.00255106      0.00273128       0.00018022
3          0.220330  0.333785   0.346420   0.012635    0.00064815      0.00067269       0.00002453
4          0.173319  0.159370   0.201818   0.042449    0.00019150      0.00024250       0.00005101
5          0.144438  0.421550   0.591873   0.170322    0.00035178      0.00049391       0.00014213
6          0.125886  0.815536   0.693490  -0.122046    0.00051696      0.00043960      -0.00007736
7          0.103925  3.391192   3.436517   0.045325    0.00146505      0.00148463       0.00001958
8          0.093717  1.220760   1.275772   0.055012    0.00042888      0.00044820       0.00001933
```

Totals:

```text
HEAD028 whitened z-MSE       0.987130
HEAD036 whitened z-MSE       1.013283
delta z-MSE                  0.026153

HEAD028 raw contribution     0.01352228
HEAD036 raw contribution     0.01351655
delta raw contribution      -0.00000574
```

Horizon/component delta in whitened z-MSE (`HEAD036 - HEAD028`):

```text
h1   0.003126  0.004048 -0.000724  0.051659  0.041904 -0.016414  0.168552  0.053595
h5  -0.018102  0.035127 -0.006824  0.069096  0.070221 -0.060033  0.145208  0.136145
h10 -0.018717  0.046861 -0.053896  0.040985  0.112063 -0.113548  0.077044  0.210397
h20 -0.081345  0.047777  0.000276  0.017465  0.249206 -0.219388 -0.119493 -0.031976
h30 -0.099995  0.108858  0.124341  0.033040  0.378217 -0.200844 -0.044688 -0.093106
```

## Interpretation

HEAD036's target-space MSE regression is real, but it is not damaging the
decoded delta surface. The model improves PC1, the largest-scale direction, and
also improves PC6. It regresses most in PC5 plus smaller-scale directions.
Because decoded delta quality is scale-weighted, the raw contribution is
essentially tied and slightly better than HEAD028.

The result is therefore not a simple MSE failure. It is a tradeoff between:

- **whitened coordinate accuracy**, where HEAD028 is still better;
- **retrieval/top-k and raw decoded delta quality**, where HEAD036 is better.

For Part 1, retrieval/top-k and non-collapse matter because this is a latent
world-state test, not just a frame regression test. HEAD036 is the first model
to improve retrieval materially while preserving decoded-frame quality.

## Decision

Treat HEAD036 as the provisional fixed-target Part 1 reference, with one caveat:
the whitened target-space MSE regression must be tracked and should not be
hidden.

Do not move to the decoder yet. Before changing objectives, run a robustness
check with the exact same fused-context configuration and one new seed. This is
not a new research knob; it tests whether the fused-context gain is stable or
seed noise.

Acceptance for the robustness check:

- MRR and top5 remain above HEAD028 (`0.096374`, `0.132031`);
- decoded delta MSE remains near or below HEAD028 (`0.015369`);
- target-space MSE does not degrade substantially beyond HEAD036 (`1.013283`);
- predicted rank stays non-collapsed, near the HEAD036 range.

If the repeat holds, keep HEAD036 as the current Part 1 fixed-target reference.
If it fails, downgrade the fused architecture to a promising but unstable
diagnostic.

## Artifacts

- `experiments/world/reports/world_model_head037_fused_context_mse_analysis.md`
