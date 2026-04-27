# 655a Sampled Path-Energy Objective on Shared-Source Multi-Head Path Flow

## Purpose

655a follows the 654a terminal-loss result. 654a showed that a pointwise
terminal path loss can move pathwise KS slightly but behaves too much like
supervised fitting to the single realized future. 655a therefore tests a true
sampled distribution score on the same shared-source multi-head backbone.

The model remains one native joint IV + anchor-factor path law. The new optional
objective samples multiple source paths per history inside training, predicts
their terminal mixed-coordinate paths at `t=0`, and applies an energy score
against the realized future path.

## Implementation

The sampled path-energy loss is disabled by default and controlled by:

- `path_energy_loss_weight`
- `path_energy_samples`
- `path_energy_tail_weight`
- `path_energy_tail_threshold`

For a batch with `K` sampled terminal path predictions:

- target distance: average L2 distance from each generated terminal path to the
  realized mixed-coordinate future;
- pair distance: average pairwise L2 distance among generated terminal paths;
- energy score: `target_distance - 0.5 * pair_distance`.

Tail-coordinate weighting uses the realized mixed-coordinate target magnitude as
a simple dimension weight.

## Command

```bash
python experiments/backfill/block_ar/train_647a_mixed_coordinate_path_flow.py \
  --state_scope joint38 \
  --head_mode multihead \
  --head_hidden 128 \
  --positive_level_policy reference_based \
  --path_energy_loss_weight 0.005 \
  --path_energy_samples 4 \
  --path_energy_tail_weight 1.0 \
  --path_energy_tail_threshold 1.5 \
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
  --seed 655 \
  --device cuda \
  --output_dir models/backfill/655a_joint38_multihead_path_energy_e8_w2048_s655
```

## Results

IV full suite:

- `655a_joint38`: `4/11`.
- Failed suites: coverage, conditionality, time_series, regime_coverage,
  distributional_fidelity, mean_reversion, pathwise_jump_realism.
- Pathwise max-jump KS improved to `0.556`, better than 652a `0.619` and 654a
  `0.570`, but still above the `<0.50` gate.
- Conditionality failed badly: MAE reduction `-0.3%`.
- Daily-change KS failed with `10/25` passing cells.
- Level KS failed with `5/25` passing cells.
- Tail-scale worsened to `9/25` passing cells.
- Cross-cell correlation still passed with ratio `0.830`.
- Aggregate h1 mean reversion still passed with ratio `0.802`, but full-horizon
  mean reversion failed.

Native joint-panel audit:

- Factor daily-change KS mean: `0.0979`.
- Factor KS pass: `13/13`.
- Factor q99 tail pass: `12/13`.
- Factor-factor correlation similarity: `0.859`.
- IV-factor correlation similarity: `0.834`.

## Interpretation

This confirms that sampled proper scoring is directionally capable of moving
pathwise jump shape, but this implementation and weight are too blunt. It
encourages broader/more energetic sampled paths in a way that worsens
conditionality, level occupancy, and per-cell tail allocation.

The result does not invalidate proper scoring as a research direction, but it
does falsify this naive global mixed-coordinate energy score as a deployable
next step.

## Decision

Keep the sampled path-energy objective as optional disabled-by-default code, but
close `655a` as below the 652a backbone.

The current clean frontier remains:

- IV-only learned frontier: historical 510a/392a family for strict IV quality.
- Native joint-law frontier: 652a shared-source multi-head path flow.
- Risk-support coordinate option: 653a `observed_positive`, if positive anchor
  support is more important than IV suite score.
