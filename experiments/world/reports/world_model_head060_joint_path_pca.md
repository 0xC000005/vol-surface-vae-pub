# World Model HEAD060: Joint Path-PCA Target

Date: 2026-05-09

Iteration type: `experiment`

## Literature Status

`supported_adjacent`.

This is still a JEPA-style predictive representation experiment because the
model predicts a future latent target from past context with an MSE loss. The
target is not a canonical EMA I-JEPA/A-JEPA target encoder. It is a fixed,
train-fit, whitened PCA code over the full future horizon-delta path, used as a
structured low-rank future-state target. The support is adjacent to predictive
coding and fixed low-rank representation learning, and it directly follows
HEAD059's target-geometry diagnosis.

## Hypothesis / Falsifier

Hypothesis: a single joint future-path target over horizons `(1, 5, 10, 20, 30)`
preserves cross-horizon and cross-cell covariance better than isolated
per-horizon targets, improving Part 1 prediction/retrieval and frozen-context
raw-delta probes without adding a new objective.

Falsifier: decoded horizon-delta MSE is not competitive with the current
fixed-PCA reference, path-code retrieval is weak, frozen context remains poor on
raw-delta probes, or context health collapses.

## Implementation

Added `experiments/world/part1_jepa_latent/joint_path_pca_predictor.py`.

Target contract:

```text
future horizon-delta path over horizons (1, 5, 10, 20, 30)
-> flatten full H x C path
-> train-fit whitened PCA path code
past -> fused GRU/direct context
-> predictor -> single future-path code
loss = MSE(z_pred_path, z_target_path)
```

Added a focused round-trip and shape test in
`test_code/test_world_model_evaluation.py`.

No EMA target extension, Barlow Twins, VICReg, retrieval/neighborhood loss,
target sweep, decoder component, or new scalar loss weight was added.

## Run

```bash
python experiments/world/part1_jepa_latent/joint_path_pca_predictor.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_dim 16 --hidden_dim 64 --context_dim 32 \
  --predictor_hidden_dim 128 --seed 7714 \
  --output_json results/world/part1_joint_path_pca_head060.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/joint_path_pca_head060.pt
```

Train/validation contract:

| object | shape |
| --- | ---: |
| train past | `2048 x 30 x 25` |
| train future delta path | `2048 x 5 x 25` |
| train target code | `2048 x 16` |
| val past | `256 x 30 x 25` |
| val future delta path | `256 x 5 x 25` |
| val target code | `256 x 16` |

## Result

Best checkpoint selected by validation decoded-delta MSE:

| metric | value |
| --- | ---: |
| epoch | 13 |
| path-code MSE | 0.951244 |
| path-code MRR | 0.133841 |
| path-code top5 | 0.175781 |
| decoded-delta MSE | 0.015530 |
| decoded-delta MRR | 0.094780 |
| decoded-delta top5 | 0.131250 |
| predicted effective rank | 5.802974 |
| target effective rank | 7.812441 |
| context effective rank | 9.204105 |
| context offdiag abs mean | 0.282073 |

Frozen context ridge probes:

| probe | MSE | MRR | top5 | top10 |
| --- | ---: | ---: | ---: | ---: |
| joint path-PCA code | 0.976923 | 0.141879 | 0.207031 | 0.312500 |
| raw horizon deltas | 0.015436 | 0.101809 | 0.126563 | 0.213281 |

Reference comparison:

| gate | HEAD038 primary reference | HEAD060 |
| --- | ---: | ---: |
| trained decoded-delta MSE | 0.015176 | 0.015530 |
| raw-delta ridge MSE | 0.015701 | 0.015436 |
| raw-delta ridge MRR | 0.096147 | 0.101809 |
| raw-delta ridge top5 | 0.128125 | 0.126563 |
| context effective rank | 6.648142 | 9.204105 |
| context offdiag abs mean | 0.323006 | 0.282073 |

The path-code ridge retrieval is not directly comparable to the old fixed-PCA
per-horizon retrieval because the target object changed. Still, it is a useful
internal sign that the context linearly exposes the new joint target geometry.

## Mechanism Read

HEAD060 is the first post-reference Part 1 experiment in this branch that is
competitive with the fixed-PCA reference on validation. It does not beat the
primary reference's trained decoded-delta MSE, but it improves the frozen
context raw-delta ridge MSE and MRR while producing a healthier context rank and
lower off-diagonal correlation.

This supports HEAD059's diagnosis: the main issue was target geometry, not a
need for Barlow/VICReg, retrieval loss, or another EMA tweak.

Primary failure class remaining: `split_robustness`. The validation result is
promising, but it is one seed and one split.

## Decision / Next Step

Promote HEAD060 as a provisional Part 1 candidate, not as the replacement
reference yet.

Next iteration should run a split-aware held-out test audit for the joint
path-PCA checkpoint, using train-only PCA/ridge fitting and evaluating on test
windows. Do not update `reference_manifest.json` until the test audit and/or a
seed repeat confirms the validation result.

## Artifacts

- `experiments/world/part1_jepa_latent/joint_path_pca_predictor.py`
- `test_code/test_world_model_evaluation.py`
- `results/world/part1_joint_path_pca_head060.json`
- `models/world/checkpoints/part1_jepa_latent/joint_path_pca_head060.pt`
