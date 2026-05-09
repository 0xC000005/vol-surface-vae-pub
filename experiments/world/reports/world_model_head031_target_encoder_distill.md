# World Model HEAD031: Target Encoder Distillation

Date: 2026-05-09

## Iteration Type

`experiment`

## Literature Status

`supported_adjacent`, following HEAD030.

The experiment does not add a new JEPA objective. It uses the fixed delta-PCA
target contract as a teacher for a learned target encoder, then evaluates
whether that target encoder can preserve the fixed target space before any
context-predictor JEPA training resumes.

## Hypothesis

A small learned target encoder can imitate the fixed delta-PCA contract when it
is trained directly on future horizon deltas. If true, the HEAD025 target
collapse was a coupled target-space anchoring failure, not evidence that learned
target encoders are unusable for this data object.

Falsifier:

- target-to-PCA MSE remains close to HEAD028's context-predictor fixed-target
  MSE `0.987130`;
- decoded delta MSE remains close to HEAD028's context-predictor decoded delta
  MSE `0.015369` instead of approaching the PCA oracle residual `0.001846`;
- target representation health collapses toward the HEAD025 rank pattern.

## Implementation

Added:

- `experiments/world/part1_jepa_latent/target_encoder_distill.py`
- focused tests in `test_code/test_world_model_evaluation.py`

Core contract:

```text
delta_h = future[:, h - 1, :] - past[:, -1, :]
z_pca = fixed whitened PCA_8(delta_h), fitted on train deltas
delta_h + horizon embedding -> learned target encoder -> z_target
loss = MSE(z_target, z_pca)
```

The model is target-encoder-only:

- no context encoder;
- no JEPA predictor;
- no retrieval/neighborhood loss;
- no variance/covariance weights;
- no decoder.

## Validation

Red test:

```text
pytest test_code/test_world_model_evaluation.py::test_target_encoder_distill_model_outputs_horizon_codes \
  test_code/test_world_model_evaluation.py::test_evaluate_distilled_targets_scores_perfect_fixed_pca_codes -q
```

Initially failed with:

```text
ModuleNotFoundError: No module named 'experiments.world.part1_jepa_latent.target_encoder_distill'
```

Post-implementation checks:

```text
pytest test_code/test_world_model_evaluation.py -q
23 passed in 0.78s

python -m py_compile experiments/world/part1_jepa_latent/target_encoder_distill.py test_code/test_world_model_evaluation.py
```

Run:

```text
python experiments/world/part1_jepa_latent/target_encoder_distill.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_dim 8 --hidden_dim 64 \
  --output_json results/world/part1_target_encoder_distill_head031.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/target_encoder_distill_head031.pt
```

## Result

Best epoch:

```text
epoch                         25
target-to-PCA MSE             0.008652
target-to-PCA cosine mean     0.997209
target-space MRR              0.998438
target-space top1             0.996875
target-space top5             1.000000
target-space top10            1.000000
decoded delta MSE             0.001990
pred effective rank           3.426355
fixed PCA target rank         3.327211
```

Reference comparisons:

```text
HEAD028 context predictor fixed-target MSE   0.987130
HEAD028 context predictor decoded delta MSE  0.015369
PCA oracle residual delta MSE                0.001846
HEAD025 collapsed predicted rank             1.262683
```

## Mechanism Read

The target encoder passed the diagnostic:

- fixed-target MSE fell by roughly two orders of magnitude versus HEAD028's
  context predictor;
- target-space retrieval is nearly exact;
- decoded delta MSE `0.001990` is close to the PCA oracle residual `0.001846`;
- predicted rank `3.426355` matches the fixed PCA target rank `3.327211` rather
  than collapsing toward the HEAD025 rank pattern.

This is not yet a validated JEPA world model because the context predictor has
not been trained against the learned target encoder. It does show that a learned
target encoder can preserve the fixed delta-PCA contract when it is anchored by
the teacher target.

## Decision

Promote the distilled target encoder as the next target-space bridge.

Next HEAD should train a JEPA context predictor against a frozen distilled
target encoder:

```text
past window -> context encoder -> horizon predictor -> z_pred
future horizon delta -> frozen distilled target encoder -> z_target
loss = MSE(z_pred, stopgrad(z_target))
```

Keep the next experiment bounded:

- load or recreate the HEAD031 target encoder contract;
- freeze the target encoder;
- use MSE prediction only;
- keep the same horizons and target dimension;
- do not add retrieval/neighborhood loss, target sweeps, or decoder components.

## Artifacts

- `experiments/world/part1_jepa_latent/target_encoder_distill.py`
- `test_code/test_world_model_evaluation.py`
- Ignored outputs: `results/world/part1_target_encoder_distill_head031.json`,
  `models/world/checkpoints/part1_jepa_latent/target_encoder_distill_head031.pt`
