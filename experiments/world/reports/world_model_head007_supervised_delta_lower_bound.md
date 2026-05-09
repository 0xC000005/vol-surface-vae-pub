# World Model HEAD007: Supervised Delta Target Lower Bound

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Before more learned-target JEPA attempts, a fixed supervised target should show
whether the past 30-day IV window contains predictive information beyond raw
persistence. A horizon-delta target should be easier to learn than absolute
future frames because persistence is already a strong absolute-frame baseline.

Falsifier: a supervised horizon-delta predictor cannot beat raw persistence on
future-frame MSE, or delta-target retrieval remains at chance.

## Execution

Added:

- `experiments/world/part1_jepa_latent/supervised_horizon_frame.py`

Updated:

- `test_code/test_world_model_evaluation.py`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/supervised_horizon_frame.py`

Run:

```text
python experiments/world/part1_jepa_latent/supervised_horizon_frame.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_mode delta --frame_weight 0.25 \
  --output_json results/world/part1_supervised_horizon_delta_head007.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_head007.pt
```

## Result

Tests:

```text
8 passed in 0.72s
```

Best validation MSE epoch:

```text
epoch 21
val MSE        0.015180
val cosine     0.982725
val MRR mean   0.052198
val top1 mean  0.015625
val top5 mean  0.062500
val top10 mean 0.117969
```

Raw persistence baseline on the same windows:

```text
MSE mean        0.022376
cosine mean     0.974271
MRR mean        0.052426
top1 mean       0.000781
top5 mean       0.086719
top10 mean      0.135156
```

The supervised delta predictor beats persistence on frame MSE and top1, but not
top5/top10 frame retrieval.

Delta-target retrieval is much stronger than raw-frame retrieval:

```text
h1  top1 0.027344  top5 0.089844  top10 0.136719  MRR 0.068463
h5  top1 0.042969  top5 0.097656  top10 0.152344  MRR 0.084804
h10 top1 0.039062  top5 0.097656  top10 0.175781  MRR 0.089904
h20 top1 0.019531  top5 0.105469  top10 0.195312  MRR 0.080143
h30 top1 0.027344  top5 0.097656  top10 0.187500  MRR 0.087366
```

Chance for 256 candidates is top1 `0.003906`, top5 `0.019531`, and top10
`0.039062`.

## Mechanism Read

This is the first positive Part 1 signal:

- past windows contain learnable information about horizon deltas;
- fixed delta targets are much more discriminative than learned EMA targets;
- raw frame retrieval remains persistence-dominated at short horizons, so
  frame-space retrieval alone is a poor JEPA target-selection criterion.

The JEPA repair should use stable delta targets or delta embeddings first, not a
moving learned target encoder.

## Decision

Continue JEPA-only.

Next experiment:

- add an explicit delta-target retrieval / contrastive term to the supervised
  horizon-delta lower bound;
- evaluate both frame-space MSE and delta-space retrieval;
- if this remains stable, use fixed delta targets as the next JEPA target
  contract instead of EMA target embeddings.
