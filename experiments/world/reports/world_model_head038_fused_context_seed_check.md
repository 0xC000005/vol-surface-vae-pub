# World Model HEAD038: Fused Context Seed Check

Date: 2026-05-09

## Iteration Type

`experiment`

## Question

Does the fused-context gain from HEAD036 survive a same-configuration seed
repeat, or was it seed noise?

## Setup

Same configuration as HEAD036:

```text
past window
-> GRU sequence branch
-> direct flattened-past branch
-> fused context
-> horizon-conditioned predictor
-> fixed z_pca

loss = MSE(predicted z_pca, fixed z_pca)
```

Only the seed and output paths changed:

```text
--seed 7711
--output_json results/world/part1_fused_context_delta_pca_head038_seed7711.json
--checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt
```

No target encoder, retrieval/neighborhood loss, target sweep, or decoder was
used.

## Acceptance From HEAD037

The repeat should:

- keep MRR/top5 above HEAD028 `0.096374`/`0.132031`;
- keep decoded delta MSE near or below HEAD028 `0.015369`;
- avoid substantially worsening target-space MSE beyond HEAD036 `1.013283`;
- stay non-collapsed.

## Result

MSE-selected checkpoint:

```text
epoch                  14
target-space MSE       1.000193
MRR                    0.103029
top1                   0.039062
top5                   0.133594
top10                  0.226562
decoded delta MSE      0.015176
pred effective rank    4.585648
context rank           6.648142
```

Best retrieval row:

```text
epoch                  19
target-space MSE       1.024813
MRR                    0.122457
top1                   0.058594
top5                   0.158594
top10                  0.240625
decoded delta MSE      0.015882
pred effective rank    4.963474
context rank           7.411700
```

Best decoded-delta row:

```text
epoch                  12
target-space MSE       1.008128
MRR                    0.103713
top5                   0.137500
top10                  0.217969
decoded delta MSE      0.015051
pred effective rank    4.346399
context rank           6.390465
```

Reference comparison:

```text
HEAD028 fixed-PCA:
MSE                    0.987130
MRR                    0.096374
top5                   0.132031
top10                  0.203125
decoded delta MSE      0.015369
pred effective rank    3.863753

HEAD036 fused seed 7710:
MSE                    1.013283
MRR                    0.117733
top5                   0.157812
top10                  0.242188
decoded delta MSE      0.015363
pred effective rank    4.595201
```

## Interpretation

The fused-context improvement is not a one-seed artifact. The repeat is milder
than HEAD036 on the MSE-selected retrieval row, but it still beats HEAD028 on
MRR/top5/top10 and improves decoded delta MSE. It also has better target-space
MSE than HEAD036 while staying non-collapsed.

The strongest retrieval checkpoint in the repeat beats both HEAD036's selected
checkpoint and HEAD028 on retrieval, though its decoded MSE is worse than the
MSE-selected checkpoint. That confirms the same tradeoff seen in HEAD036:
retrieval can be pushed higher, but the safer reference point is the checkpoint
that preserves decoded quality.

## Decision

Promote the fused-context fixed delta-PCA predictor as the current fixed-target
Part 1 reference:

- primary reference checkpoint: HEAD038 MSE-selected seed `7711`, because it
  preserves decoded delta quality and improves retrieval/top-k over HEAD028;
- supporting checkpoint: HEAD036 seed `7710`, because it shows the same
  architecture can produce stronger retrieval/top-k with decoded quality tied
  to HEAD028.

Do not move to the decoder yet. The next Part 1 gate should be a frozen-probe
audit on the fused-context representation, using the existing probe/evaluation
patterns where possible. This tests whether the improved retrieval is useful
for future-state attributes beyond nearest-neighbor ranking.

## Artifacts

- Ignored outputs: `results/world/part1_fused_context_delta_pca_head038_seed7711.json`,
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt`
- `experiments/world/reports/world_model_head038_fused_context_seed_check.md`
