# World Model HEAD022: Post-Hoc Epoch Selection

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Hypothesis

Auditing saved HEAD021 epoch checkpoints post-hoc should identify whether a
single saved epoch already balances MSE, probe quality, context health, and
top-k retrieval better than the scalar MRR-selected checkpoint.

Falsifier: no audited saved epoch improves the top-k-aware composite score over
the existing scalar MRR-selected epoch.

## Execution

No code changes.

Audited saved HEAD021 checkpoints:

```text
epoch_002.pt
epoch_003.pt
epoch_007.pt
epoch_008.pt
```

Each checkpoint was evaluated with:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint <epoch checkpoint> \
  --output_json results/world/part1_context_probe_audit_head022_epochXXX.json
```

Per-epoch train-score inputs were generated from the saved HEAD021 history plus
fresh frame-space validation metrics, then scored with:

```text
python experiments/world/part1_jepa_latent/score_context_runs.py \
  --entry epoch002 results/world/head022_epoch_score_inputs/epoch_002_train.json results/world/part1_context_probe_audit_head022_epoch002.json \
  --entry epoch003 results/world/head022_epoch_score_inputs/epoch_003_train.json results/world/part1_context_probe_audit_head022_epoch003.json \
  --entry epoch007 results/world/head022_epoch_score_inputs/epoch_007_train.json results/world/part1_context_probe_audit_head022_epoch007.json \
  --entry epoch008 results/world/head022_epoch_score_inputs/epoch_008_train.json results/world/part1_context_probe_audit_head022_epoch008.json \
  --output_json results/world/part1_context_composite_scores_head022_epochs.json
```

## Result

Post-hoc epoch ranking:

```text
epoch007 0.577865
epoch008 0.564362
epoch003 0.520371
epoch002 0.458707
```

Epoch 7:

```text
frame_mse_improvement 0.053350
frame MRR             0.058656
frame_top5_delta     -0.017969
frame_top10_delta    -0.000781
rank fraction         0.166353
decorrelation         0.647716
ridge MRR             0.110486
ridge MSE improvement 0.239285
```

Epoch 3:

```text
frame_mse_improvement 0.010343
frame MRR             0.056790
frame_top5_delta      0.003906
frame_top10_delta     0.004687
rank fraction         0.148390
ridge MRR             0.104012
```

Epoch 8:

```text
rank fraction         0.169888
decorrelation         0.654251
ridge MRR             0.109153
frame_top5_delta     -0.020312
frame_top10_delta    -0.011719
```

## Mechanism Read

Post-hoc selection does not reveal a better saved checkpoint. Epoch 7 remains
the best saved state by the top-k-aware composite score. Epoch 3 has positive
top5/top10 deltas but is too weak on MSE, rank, and frozen probes. Epoch 8 has
slightly better rank/decorrelation than epoch 7 but loses too much retrieval.

This means the bottleneck is now training dynamics/objective design, not only
checkpoint selection. The model passes the basic fixed-delta/persistence and
context-rank gates, but top-k neighborhood quality conflicts with later
MSE/probe maturation.

## Decision

Continue JEPA-only.

Next step:

- change the training objective or evaluation target for neighborhood retrieval
  rather than only changing checkpoint selection;
- a reasonable next candidate is a top-k preserving auxiliary objective or a
  smoother neighborhood target in delta space.

