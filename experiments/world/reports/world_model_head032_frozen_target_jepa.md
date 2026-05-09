# World Model HEAD032: Frozen Target JEPA Predictor

Date: 2026-05-09

## Iteration Type

`experiment`

## Literature Status

`supported_adjacent`, following HEAD030 and HEAD031.

This experiment keeps the target-encoder distillation bridge but returns to a
JEPA-style context-to-target prediction objective. The target encoder is trained
to imitate the fixed delta-PCA contract, frozen, and then used as the target
space for a past-window context predictor. No retrieval, neighborhood,
variance/covariance, or decoder objective is added.

## Hypothesis

If HEAD025 failed because the target encoder was unanchored, then training the
context predictor against a frozen distilled target encoder should preserve the
healthy target space from HEAD031 while matching or improving the fixed-PCA
predictor from HEAD028.

Falsifier:

- target encoder quality is good, but context prediction remains near HEAD028
  fixed-target MSE `0.987130`;
- retrieval fails to beat HEAD028 fixed-target MRR `0.096374` or top-k;
- decoded delta MSE does not improve beyond HEAD028 `0.015369`.

## Implementation

Added:

- `experiments/world/part1_jepa_latent/frozen_target_jepa.py`
- focused test in `test_code/test_world_model_evaluation.py`

Two-stage contract:

```text
Stage 1:
future horizon delta -> target encoder -> z_target
loss = MSE(z_target, fixed z_pca)

Stage 2:
past window -> context encoder -> horizon predictor -> z_pred
future horizon delta -> frozen target encoder -> stopgrad(z_target)
loss = MSE(z_pred, z_target)
```

The script recreates the target-encoder contract in-process so the experiment is
reproducible without depending on an ignored checkpoint.

## Validation

Red test:

```text
pytest test_code/test_world_model_evaluation.py::test_frozen_target_jepa_predicts_horizon_codes_and_freezes_target_encoder -q
```

Initially failed with:

```text
ModuleNotFoundError: No module named 'experiments.world.part1_jepa_latent.frozen_target_jepa'
```

Post-implementation checks:

```text
pytest test_code/test_world_model_evaluation.py -q
24 passed in 0.74s

python -m py_compile experiments/world/part1_jepa_latent/frozen_target_jepa.py test_code/test_world_model_evaluation.py
```

Run:

```text
python experiments/world/part1_jepa_latent/frozen_target_jepa.py \
  --device cpu --target_epochs 25 --predictor_epochs 25 \
  --batch_size 128 --max_train_windows 2048 --max_val_windows 256 \
  --target_dim 8 --hidden_dim 64 --context_dim 32 \
  --predictor_hidden_dim 128 \
  --output_json results/world/part1_frozen_target_jepa_head032.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/frozen_target_jepa_head032.pt
```

## Result

Target encoder stage, validation against fixed PCA:

```text
target-to-PCA MSE        0.014106
target-to-PCA MRR        0.990560
decoded delta MSE        0.002044
PCA oracle residual      0.001846
target encoder rank      3.430619
```

Best context-predictor epoch by MSE:

```text
epoch                    19
val MSE vs frozen target 0.979270
MRR vs frozen target     0.085140
top1                    0.028125
top5                    0.112500
top10                   0.183594
decoded delta MSE        0.015502
predicted rank           3.633565
context rank             4.716221
```

Best retrieval row:

```text
epoch                    24
MRR vs frozen target     0.089344
top5                    0.116406
top10                   0.198438
decoded delta MSE        0.015935
predicted rank           3.796872
context rank             4.872305
```

Comparison to HEAD028 fixed-PCA predictor:

```text
HEAD028 fixed-target MSE       0.987130
HEAD028 fixed-target MRR       0.096374
HEAD028 fixed-target top5      0.132031
HEAD028 decoded delta MSE      0.015369
HEAD032 best MSE               0.979270
HEAD032 best MRR row           0.089344
HEAD032 best top5 row          0.120312
HEAD032 best decoded MSE       0.015502
```

Against the original fixed PCA target, the best-MSE checkpoint reports:

```text
MSE vs fixed PCA          1.018668
MRR vs fixed PCA          0.089965
top5 vs fixed PCA         0.116406
```

## Mechanism Read

The distilled target encoder is not the active bottleneck. It remains close to
the fixed PCA teacher. The context predictor does not collapse, and its
effective rank is healthy, but its retrieval and decoded-frame quality remain
around the HEAD028 fixed-PCA predictor.

This falsifies the simple bridge hypothesis. Freezing a distilled learned target
encoder does not by itself improve the predictive world-state gate. The next
failure class remains `latent_prediction`, specifically the mapping from past
window to future-delta target space.

## Decision

Do not promote HEAD032 as the current Part 1 candidate. Keep HEAD028 as the
fixed-target reference and HEAD031 as evidence that the target encoder can be
anchored.

Next HEAD should be `post_experiment_analysis`, not another objective patch:

- compare HEAD028 and HEAD032 per-horizon error/retrieval dynamics;
- decide whether the predictor is capacity-limited, context-object limited, or
  selection-metric limited;
- only then choose one next experiment.

Do not add retrieval/neighborhood loss, target sweeps, or decoder components
yet.

## Artifacts

- `experiments/world/part1_jepa_latent/frozen_target_jepa.py`
- `test_code/test_world_model_evaluation.py`
- Ignored outputs: `results/world/part1_frozen_target_jepa_head032.json`,
  `models/world/checkpoints/part1_jepa_latent/frozen_target_jepa_head032.pt`
