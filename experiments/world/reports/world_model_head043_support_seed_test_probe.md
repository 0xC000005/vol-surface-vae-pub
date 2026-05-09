# World Model HEAD043: Support Seed Test Probe

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: the supporting fused-context seed `7710` should also pass the split-aware held-out test probe audit, confirming that HEAD042 was not a primary-seed-only result.

Falsifier: seed `7710` collapses on test, fails to improve over the zero-delta baseline, or loses the fixed-PCA/raw-delta retrieval signal that made the fused fixed-target model the current Part 1 candidate.

## Implementation

Reused the split-aware audit script from HEAD042:

```bash
python experiments/world/part1_jepa_latent/fused_context_probe_audit.py \
  --device cpu \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt \
  --max_train_windows 2048 \
  --eval_split test --max_eval_windows 256 \
  --batch_size 128 --ridge_alpha 1e-3 \
  --output_json results/world/part1_fused_context_probe_audit_head043_test_seed7710.json
```

No model objective, target, decoder, retrieval loss, or Barlow/VICReg-style term was added.

## Result

Test context health:

- variance min `0.000939`, mean `0.142549`;
- effective rank `7.508093`, participation ratio `4.865114`;
- off-diagonal absolute mean `0.321090`.

Trained head to fixed delta-PCA on test:

- MSE `1.052584`;
- MRR `0.124176`, top5 `0.175781`, top10 `0.259375`;
- decoded delta MSE `0.017537`;
- predicted rank `5.100427`.

Frozen context ridge probe to fixed delta-PCA on test:

- MSE `1.046969`;
- MRR `0.131317`, top5 `0.191406`, top10 `0.269531`;
- predicted rank `5.428738`.

Frozen context ridge probe to raw horizon deltas on test:

- MSE `0.018021`;
- MRR `0.115288`, top5 `0.160156`, top10 `0.238281`;
- predicted rank `2.567909`.

Zero-delta test baseline:

- MSE `0.025450`;
- MRR `0.023923`, top5 `0.019531`, top10 `0.039062`.

## Cross-Test Read

| metric | HEAD042 seed 7711 test | HEAD043 seed 7710 test |
|---|---:|---:|
| context rank | 7.534067 | 7.508093 |
| context offdiag | 0.310717 | 0.321090 |
| trained decoded delta MSE | 0.017769 | 0.017537 |
| trained fixed-PCA MRR | 0.119919 | 0.124176 |
| trained fixed-PCA top5 | 0.155469 | 0.175781 |
| fixed-PCA ridge MRR | 0.116466 | 0.131317 |
| fixed-PCA ridge top5 | 0.148438 | 0.191406 |
| raw-delta ridge MSE | 0.017923 | 0.018021 |
| raw-delta ridge MRR | 0.110315 | 0.115288 |
| raw-delta ridge top5 | 0.142969 | 0.160156 |
| raw-delta MSE improvement over zero | 0.295762 | 0.291907 |

## Interpretation

- Support-seed held-out test probing passes.
- Test context health is stable across both seeds, with rank near `7.5` and offdiag near `0.31` to `0.32`.
- Fixed-PCA retrieval is stronger on seed `7710` than seed `7711`, including the frozen-context ridge probe.
- Raw-delta MSE improvement over the zero baseline remains stable near `29%`.
- The HEAD040 caveat still matters: the fused fixed-target model is the current best coordinate/decoded-surface representation, while older retrieval-trained diagnostic contexts can still be sharper on some raw-delta retrieval comparisons.

## Decision / Next Step

- Treat the fused-context fixed delta-PCA predictor as the current validated Part 1 JEPA-style representation reference.
- Keep primary reference seed `7711` and supporting seed `7710`.
- Do not add Barlow Twins, retrieval/neighborhood losses, or more Part 1 knobs.
- Next HEAD should be post-experiment analysis to write the Part 1 reference closeout and decide the minimum decoder-conditioning handoff criteria, without starting decoder work unless explicitly requested.

## Artifacts

- Ignored output: `results/world/part1_fused_context_probe_audit_head043_test_seed7710.json`
