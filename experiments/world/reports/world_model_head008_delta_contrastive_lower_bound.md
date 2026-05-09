# World Model HEAD008: Delta-Target Contrastive Lower Bound

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Adding a contrastive retrieval term in fixed horizon-delta target space should
improve discriminative Part 1 signal while keeping the supervised delta model
better than raw persistence on frame MSE.

Falsifier: frame MSE no longer beats persistence, or delta-target retrieval does
not improve.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/supervised_horizon_frame.py`
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
  --retrieval_weight 0.05 --retrieval_temperature 0.1 \
  --output_json results/world/part1_supervised_horizon_delta_contrastive_head008.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_contrastive_head008.pt
```

## Result

Tests:

```text
8 passed in 0.70s
```

Best frame-MSE epoch:

```text
epoch 11
frame MSE      0.020757
frame MRR mean 0.053915
frame top1     0.013281
frame top5     0.071094
frame top10    0.129688
```

Raw persistence baseline:

```text
frame MSE      0.022376
frame MRR mean 0.052426
frame top1     0.000781
frame top5     0.086719
frame top10    0.135156
```

Best-MRR epoch from training history:

```text
epoch 5
frame MSE      0.021263
frame MRR mean 0.060965
frame top1     0.014844
frame top5     0.084375
```

Delta-target retrieval at the saved best-MSE checkpoint:

```text
h1  top1 0.011719  top5 0.078125  top10 0.167969  MRR 0.056569
h5  top1 0.023438  top5 0.109375  top10 0.210938  MRR 0.084870
h10 top1 0.019531  top5 0.128906  top10 0.261719  MRR 0.092585
h20 top1 0.027344  top5 0.140625  top10 0.257812  MRR 0.100753
h30 top1 0.035156  top5 0.171875  top10 0.308594  MRR 0.123473
```

## Mechanism Read

The contrastive term works in the right target space:

- delta retrieval improves strongly versus chance and versus the pure MSE run,
  especially at longer horizons;
- frame MSE still beats raw persistence at the best-MSE checkpoint;
- frame-space top5/top10 remains hard because raw persistence is a strong
  nearest-neighbor policy for absolute IV levels.

This makes fixed horizon-delta targets the current best Part 1 contract. The
remaining implementation issue is checkpoint selection: best MSE and best
retrieval are not the same epoch.

## Decision

Continue JEPA-only.

Next step:

- add selection by validation retrieval or a composite MSE/retrieval score;
- rerun the contrastive delta lower bound with retrieval-first checkpointing;
- then use the fixed delta target contract for the next JEPA representation
  architecture.
