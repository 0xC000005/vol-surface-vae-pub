# World Model HEAD041: Fused Context Probe Seed Check

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: the HEAD039 frozen-probe result is not a seed-specific artifact. The supporting HEAD036 fused-context checkpoint should show similar non-collapse, fixed-PCA probe retrieval, and raw-delta probe MSE.

Falsifier: seed `7710` loses context rank, fails to beat the zero-delta baseline in raw-delta MSE/retrieval, or cannot reproduce fixed-PCA probe retrieval near the trained head.

## Implementation

Reused the existing HEAD039 audit script without changing code:

```bash
python experiments/world/part1_jepa_latent/fused_context_probe_audit.py \
  --device cpu \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt \
  --max_train_windows 2048 --max_val_windows 256 \
  --batch_size 128 --ridge_alpha 1e-3 \
  --output_json results/world/part1_fused_context_probe_audit_head041_seed7710.json
```

No model objective, target, decoder, retrieval loss, or Barlow/VICReg-style term was added.

## Result

Validation context health:

- variance min `0.002247`, mean `0.153385`;
- effective rank `6.551514`, participation ratio `4.425208`;
- off-diagonal absolute mean `0.347958`.

Trained head to fixed delta-PCA:

- MSE `1.013283`;
- MRR `0.117733`, top5 `0.157812`, top10 `0.242188`;
- decoded delta MSE `0.015363`;
- predicted rank `4.595201`.

Frozen context ridge probe to fixed delta-PCA:

- MSE `1.026062`;
- MRR `0.118667`, top5 `0.157031`, top10 `0.239063`;
- predicted rank `4.865685`.

Frozen context ridge probe to raw horizon deltas:

- MSE `0.015880`;
- MRR `0.106107`, top5 `0.125000`, top10 `0.208594`;
- predicted rank `2.826555`.

Zero-delta baseline:

- MSE `0.022376`;
- MRR `0.023923`, top5 `0.019531`, top10 `0.039062`.

## Cross-Seed Read

| metric | HEAD039 seed 7711 | HEAD041 seed 7710 |
|---|---:|---:|
| context rank | 6.648142 | 6.551514 |
| context offdiag | 0.323006 | 0.347958 |
| trained fixed-PCA MRR | 0.103029 | 0.117733 |
| trained decoded delta MSE | 0.015176 | 0.015363 |
| fixed-PCA ridge MRR | 0.103763 | 0.118667 |
| fixed-PCA ridge top5 | 0.136719 | 0.157031 |
| raw-delta ridge MSE | 0.015701 | 0.015880 |
| raw-delta ridge MRR | 0.096147 | 0.106107 |

## Interpretation

- The fused-context representation is cross-seed stable on non-collapse and decoded/coordinate quality.
- Fixed-PCA neighborhood structure is also cross-seed stable; the supporting seed is stronger than the primary seed on fixed-PCA MRR/top5.
- Raw-delta probe retrieval remains mixed: HEAD041 improves over HEAD039 MRR (`0.106107` vs `0.096147`) but still trails the strongest older raw-delta context probes on top5.
- This strengthens the current Part 1 reference while preserving the HEAD040 caveat: fixed-target quality does not fully solve raw-delta retrieval sharpness.

## Decision / Next Step

- Cross-seed probe robustness passes.
- Do not add Barlow Twins, retrieval/neighborhood losses, or decoder work.
- Next HEAD should add or use a split-aware probe audit so the same fused-context checks can run on the held-out test split, while training ridge probes only on the train split.

## Artifacts

- Ignored output: `results/world/part1_fused_context_probe_audit_head041_seed7710.json`
