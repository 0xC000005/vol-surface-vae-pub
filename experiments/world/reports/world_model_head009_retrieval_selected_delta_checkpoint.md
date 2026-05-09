# World Model HEAD009: Retrieval-Selected Delta Checkpoint

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

The fixed horizon-delta contrastive lower bound should save a more useful Part 1
checkpoint when selection uses validation retrieval rather than validation MSE.

Falsifier: retrieval-selected checkpointing fails to preserve the best MRR epoch
or loses the frame-MSE advantage over persistence.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/supervised_horizon_frame.py`
- `test_code/test_world_model_evaluation.py`

Added selection metrics:

- `mse`
- `mrr`
- `top5`

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
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_contrastive_mrr_head009.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_contrastive_mrr_head009.pt
```

## Result

Tests:

```text
9 passed in 0.71s
```

Selected checkpoint:

```text
epoch 5
selection metric mrr
frame MSE      0.021263
frame MRR mean 0.060965
frame top1     0.014844
frame top5     0.084375
frame top10    0.139844
```

Raw persistence baseline:

```text
frame MSE      0.022376
frame MRR mean 0.052426
frame top1     0.000781
frame top5     0.086719
frame top10    0.135156
```

Delta-target retrieval at the selected checkpoint:

```text
h1  top1 0.035156  top5 0.105469  top10 0.167969  MRR 0.082338
h5  top1 0.019531  top5 0.085938  top10 0.136719  MRR 0.069653
h10 top1 0.019531  top5 0.082031  top10 0.187500  MRR 0.073456
h20 top1 0.031250  top5 0.117188  top10 0.207031  MRR 0.089948
h30 top1 0.027344  top5 0.128906  top10 0.210938  MRR 0.093898
```

## Mechanism Read

Retrieval-selected checkpointing is the right default for this branch:

- the selected checkpoint beats persistence on MSE, MRR, top1, and top10;
- top5 is close to persistence and much better than chance;
- delta-target retrieval remains strong across horizons;
- the run now saves the checkpoint that matches the Part 1 discriminative
  objective rather than only the smoothest MSE epoch.

This is the strongest Part 1 result so far, but it is still a fixed-target
supervised lower bound. The next question is whether the learned context state
itself is a healthy JEPA-style representation, not merely whether the supervised
head predicts deltas.

## Decision

Continue JEPA-only.

Next step:

- audit context embeddings from the retrieval-selected fixed-delta checkpoint;
- measure context variance, effective rank, off-diagonal correlation, and frozen
  linear probes for horizon deltas;
- if context is healthy, promote fixed horizon deltas as the Part 1 target
  contract for the next JEPA model.
