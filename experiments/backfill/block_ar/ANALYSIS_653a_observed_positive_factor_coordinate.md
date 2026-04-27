# 653a Observed-Positive Anchor-Factor Coordinate Policy

## Purpose

652a validated the shared-source multi-head architecture but exposed a data
coordinate problem: several anchor factors that are economically positive
levels, such as Treasury yields and credit spreads, were represented as
unconstrained difference-level variables because the source panel provided
`_diff` diagnostics instead of `_logret` diagnostics. The generated joint
samples could therefore produce negative yields or spreads.

653a tests a minimal data/preprocessing fix:

- Keep the 652a model architecture unchanged.
- Add `positive_level_policy=observed_positive`.
- If a non-return anchor-factor level is strictly positive after cleaning, use
  log-level coordinates even when no explicit `_logret` reference column exists.
- Keep variables with observed negative values, such as crude oil in this panel,
  in diff-level coordinates.

## Data Audit

Command:

```bash
python experiments/backfill/block_ar/audit_576a_unified_increment_panel.py \
  --history_len 30 \
  --future_lens 30 60 90 152 \
  --test_start 4511 \
  --val_size 441 \
  --iv_count 25 \
  --clean_nonpositive_log_levels \
  --positive_level_policy observed_positive \
  --output results/autoresearch/653a_observed_positive_panel_audit/unified_increment_audit.json
```

Findings:

- 30/60/90/152-day reconstruction max error remains `0.0`.
- No validation/test leakage was detected.
- Reference-increment max discrepancy drops to about `0.0377`.
- `us2y`, `us10y`, `aaa_oas`, and `bbb_oas` move from diff-level to log-level
  coordinates.
- Crude oil remains diff-level because the panel contains negative observations.

## Model Run

Command:

```bash
python experiments/backfill/block_ar/train_647a_mixed_coordinate_path_flow.py \
  --state_scope joint38 \
  --head_mode multihead \
  --head_hidden 128 \
  --positive_level_policy observed_positive \
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
  --seed 653 \
  --device cuda \
  --output_dir models/backfill/653a_joint38_multihead_observed_positive_e8_w2048_s653
```

Training smoke:

- Best validation loss: `1.0721`.
- Generated factor minimum: `0.656`, versus negative values in 652a for some
  positive-level anchors.
- Finite state and increment rates: `1.0`.

## Results

IV full suite:

- `653a_joint38`: `4/11`.
- Failed suites: coverage, conditionality, time_series, regime_coverage,
  distributional_fidelity, mean_reversion, pathwise_jump_realism.
- IV daily-change KS worsened to `13/25` passing cells.
- IV level KS worsened to `3/25` passing cells.
- Conditionality narrowly failed with `4.3%` MAE reduction.
- Cross-cell correlation still passed with ratio `0.797`.
- Aggregate h1 mean reversion still passed with ratio `0.994`.
- Pathwise max-jump KS worsened to `0.632`.

Native joint-panel audit:

- Factor daily-change KS mean: `0.1105`.
- Factor KS pass: `12/13`.
- Factor q99 tail pass: `13/13`.
- Factor-factor correlation similarity: `0.846`.
- IV-factor correlation similarity: `0.817`.
- Positive-level anchor support is cleaner: generated `us2y`, `us10y`,
  `aaa_oas`, and `bbb_oas` are no longer negative.

## Interpretation

The observed-positive coordinate policy is a defensible preprocessing option for
risk-manager support realism. It makes the data representation more consistent:
strictly positive levels are modeled multiplicatively, while variables that can
be negative remain additive.

It is not a performance breakthrough. The IV path law gets worse, and the joint
factor daily-change KS loses one passing factor. This indicates that the main
remaining problem is not factor support; it is still the IV future path
distribution objective.

## Decision

Keep `positive_level_policy=observed_positive` as an optional risk-control data
coordinate for joint anchor-factor deployments, but do not promote 653a over
652a as the active IV-quality backbone.

The next principled step is a path distribution objective on the shared-source
multi-head backbone. It should directly target realized future movement, tail
allocation, and long-horizon path geometry without adding separate IV/factor
model families or post-hoc glued scenarios.
