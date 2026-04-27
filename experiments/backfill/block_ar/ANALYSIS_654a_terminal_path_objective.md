# 654a Terminal Path Objective on Shared-Source Multi-Head Path Flow

## Purpose

654a tests a loss-level change on the 652a shared-source multi-head backbone.
The architecture, stochastic source, and data coordinate are unchanged from the
reference-based 652a joint model. The only new mechanism is an optional terminal
path loss in mixed-coordinate space:

- Standard flow-matching MSE remains the main objective.
- The predicted velocity also implies a terminal future path estimate
  `x1_hat = x_t + (1 - t) * v_theta(x_t, t, context)`.
- A smooth-L1 terminal path loss compares `x1_hat` to the realized mixed
  coordinate path.
- A simple tail gate can upweight coordinates whose realized mixed-coordinate
  target exceeds a threshold.

The goal was to target realized future movement and tail allocation directly
without changing the model family.

## Command

```bash
python experiments/backfill/block_ar/train_647a_mixed_coordinate_path_flow.py \
  --state_scope joint38 \
  --head_mode multihead \
  --head_hidden 128 \
  --positive_level_policy reference_based \
  --terminal_path_loss_weight 0.25 \
  --terminal_tail_weight 1.0 \
  --terminal_tail_threshold 1.5 \
  --epochs 8 \
  --max_train_windows 2048 \
  --batch_size 32 \
  --memory_dim 128 \
  --memory_layers 3 \
  --memory_heads 4 \
  --memory_ff 256 \
  --token_dim 128 \
  --token_layers 3 \
  --token_heads 4 \
  --token_ff 256 \
  --flow_steps 16 \
  --prefix_feature_mode scale \
  --sample_count 4 \
  --sample_steps 8 \
  --chunk_size 2 \
  --seed 654 \
  --device cuda \
  --output_dir models/backfill/654a_joint38_multihead_terminal_path_e8_w2048_s654
```

## Results

IV full suite:

- `654a_joint38`: `4/11`.
- Failed suites: coverage, conditionality, time_series, regime_coverage,
  distributional_fidelity, mean_reversion, pathwise_jump_realism.
- Conditionality worsened: MAE reduction `1.8%`, and per-cell worst reduction
  `-13.1%`.
- Daily-change KS failed with `12/25` passing cells.
- Level KS failed with `8/25` passing cells.
- Tail-scale remained weak with `12/25` passing cells.
- Pathwise max-jump KS improved versus 652a (`0.570` vs `0.619`) but still
  failed the `<0.50` gate.
- Cross-cell correlation still passed with ratio `0.821`.
- Aggregate h1 mean reversion still passed with ratio `0.985`, but the
  full-horizon profile failed.

Native joint-panel audit:

- Factor daily-change KS mean: `0.0989`.
- Factor KS pass: `13/13`.
- Factor q99 tail pass: `13/13`.
- Factor-factor correlation similarity: `0.871`.
- IV-factor correlation similarity: `0.846`.

## Interpretation

The terminal path objective is clean and did move one intended metric:
pathwise max-jump KS improved from `0.619` to `0.570`. But that improvement was
not enough, and it came with weaker IV conditionality and daily-change
distributional fidelity.

This suggests that a pointwise terminal reconstruction-style loss is still too
close to supervised path fitting. It does not solve the conditional law problem:
the model needs better distributional allocation across plausible future paths,
not only a stronger pull toward the single realized validation future.

## Decision

Keep the terminal path objective in code as an optional disabled-by-default
training mechanism, but close this specific configuration as below the 652a
backbone for deployability.

The next route should avoid another pointwise loss. If continuing objective
research, it should use a true sampled distribution score, such as energy or
sliced-Wasserstein over multiple generated future paths, or return to the
stronger 510a/392a AR frontier for deployable IV-only quality while keeping 652a
as the clean native joint backbone.
