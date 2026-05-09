# World Model HEAD011: Context Regularization Trial

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Adding direct variance/covariance regularization to the supervised fixed-delta
context state should improve representation health without sacrificing the
fixed-delta prediction and retrieval advantage.

Falsifier: context effective rank/off-diagonal correlation does not improve, or
the ridge probe loses its advantage over zero-delta and the trained head.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/supervised_horizon_frame.py`
- `test_code/test_world_model_evaluation.py`

Added context regularization controls:

- `--context_variance_weight`
- `--context_covariance_weight`
- `--context_variance_gamma`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/supervised_horizon_frame.py experiments/world/part1_jepa_latent/context_probe_audit.py`

Run:

```text
python experiments/world/part1_jepa_latent/supervised_horizon_frame.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_mode delta --frame_weight 0.25 \
  --retrieval_weight 0.05 --retrieval_temperature 0.1 \
  --context_variance_weight 0.05 \
  --context_covariance_weight 0.005 \
  --context_variance_gamma 0.1 \
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_contextreg_head011.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_contextreg_head011.pt
```

Audit:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_contextreg_head011.pt \
  --output_json results/world/part1_context_probe_audit_head011.json
```

## Result

Tests:

```text
12 passed in 0.73s
```

MRR-selected checkpoint:

```text
epoch 5
frame MSE      0.021399
frame MRR mean 0.058350
frame top1     0.010938
frame top5     0.083594
frame top10    0.135938
```

Raw persistence baseline:

```text
frame MSE      0.022376
frame MRR mean 0.052426
frame top1     0.000781
frame top5     0.086719
frame top10    0.135156
```

Context health after regularization:

```text
variance min       0.004570
variance mean      0.019640
effective rank     3.567140
participation      2.147186
offdiag abs mean   0.502522
offdiag abs max    0.972201
```

Frozen ridge probe after regularization:

```text
MSE        0.017674
MRR mean   0.108320
top1 mean  0.045313
top5 mean  0.140625
top10 mean 0.224219
```

Previous HEAD010 audit for comparison:

```text
effective rank   3.688405
offdiag abs mean 0.484346
ridge MSE        0.017107
ridge MRR mean   0.111537
```

## Mechanism Read

This trial is a mixed but mostly negative result. The regularized checkpoint
still beats raw persistence on frame MSE, MRR, top1, and top10, and the frozen
ridge probe still beats zero-delta by a wide margin. However, it does not solve
the representation-health bottleneck.

The regularizer increased context variance but made redundancy slightly worse:
effective rank decreased from `3.69` to `3.57`, and off-diagonal correlation
rose from `0.484` to `0.503`. The likely issue is that the VICReg-style
covariance term operates on covariance magnitude, so once the variance term
expands dominant axes it does not directly penalize high correlation strongly
enough.

## Decision

Continue JEPA-only.

Next step:

- do not treat simple variance/covariance regularization as sufficient;
- replace or supplement it with a correlation/whitening-style context
  regularizer that penalizes normalized off-diagonal correlation;
- keep the frozen context probe as the primary Part 1 gate.

