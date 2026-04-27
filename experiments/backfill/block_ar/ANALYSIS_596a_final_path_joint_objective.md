# 596a Final-Path Joint Objective

## Hypothesis

595a identified objective-level broad-frame calibration as the clean next direction. 596a kept the 510a AR flow architecture unchanged and finetuned the native model with a final-path joint-distribution objective:

- full-path energy score in empirical normal-score coordinates;
- batch-level sliced projection Wasserstein discrepancy over generated and realized future paths;
- deterministic horizon weighting toward later horizons;
- small FM anchor to limit transition-law drift.

This tests whether the broad-frame failures are caused by a training objective that is too local or transition-oriented.

## Implementation

- Script: `experiments/backfill/block_ar/train_596a_final_path_joint_objective.py`
- Test: `test_code/test_596a_joint_path_objective.py`
- Source checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- Output checkpoint: `models/backfill/596a_final_path_joint_objective_s596/best_model.pt`
- Evaluation output: `results/autoresearch/596a_final_path_joint_objective/full11.json`

The model architecture is unchanged. This is a native 510a finetune, not a wrapper.

## Commands

```bash
pytest test_code/test_596a_joint_path_objective.py -q
```

```bash
python experiments/backfill/block_ar/train_596a_final_path_joint_objective.py \
  --checkpoint models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt \
  --epochs 3 \
  --batch_size 8 \
  --train_sample_count 4 \
  --rollout_flow_steps 4 \
  --joint_weight 0.05 \
  --sw_weight 1.0 \
  --fm_anchor_weight 1.0 \
  --horizon_end_weight 2.0 \
  --n_projections 32 \
  --n_quantiles 16 \
  --lr 1e-5 \
  --output_dir models/backfill/596a_final_path_joint_objective_s596 \
  --seed 596 \
  --device cuda
```

```bash
python experiments/backfill/block_ar/evaluate_220h_full_multihorizon_v2_suite.py \
  --model_type 340c \
  --checkpoint models/backfill/596a_final_path_joint_objective_s596/best_model.pt \
  --max_windows 441 \
  --samples 48 \
  --batch_size 32 \
  --chunk_size 8 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --device cuda \
  --output_json results/autoresearch/596a_final_path_joint_objective/full11.json \
  --output_md results/autoresearch/596a_final_path_joint_objective/full11.md
```

## Result

Focused test passed: `2 passed in 1.44s`.

596a scored `5/11` on the 441-window broad frame, tying same-frame 510a on count but not improving the risk profile.

| Metric | Same-frame 510a | 596a |
| --- | ---: | ---: |
| Suite score | `5/11` | `5/11` |
| Coverage90 overall | `69.6%` | `65.7%` |
| Coverage90 h1 / h30 | `79.4% / 66.2%` | `78.7% / 61.2%` |
| Conditional MAE reduction | `7.61%` | `5.58%` |
| Worst-cell MAE reduction | `-13.1%` | `-18.1%` |
| Turbulent/calm width ratio | `1.093` | `1.078` |
| Kurtosis ratio | `0.618` | `0.584` |
| Cointegration worst-cell ratio | `0.283` | `0.264` |
| Regime layer2 | `0/8` | `0/8` |
| Persistent undercoverage | `16.2%` | `18.8%` |
| Daily-change KS cells | `25/25` | `25/25` |
| Level KS cells | `3/25` | `2/25` |
| Median-bias cells | `10/25` | `6/25` |
| Bad-window rate | `24.0%` | `29.7%` |
| Cross-cell corr / rank | `1.070 / 1.225` | `1.081 / 1.196` |
| Mean-reversion active corr | `0.531` | `0.581` |
| Pathwise max-jump KS | `0.463` | `0.456` |

## Mechanism Read

The final-path objective improves a few path-geometry quantities: pathwise max-jump KS and active-cell mean-reversion correlation move in the right direction. But it worsens the scenario deck as a conditional risk distribution. Coverage narrows, h30 coverage falls below the prior pass level, persistent undercoverage rises, level KS worsens, median-bias cells collapse, and aggregate kurtosis drops.

The clean causal read is that a one-realization final-path joint objective is still not enough to teach conditional uncertainty. It pulls generated paths toward observed realized paths and broad batch marginals, but does not add the missing conditional variance in the undercovered cells/windows.

## Decision

596a is not deployable and should not be weight-swept as the next default move. The architecture is clean and the objective is literature-aligned, but the result shows that objective-level final-path alignment alone tends to narrow the scenario distribution.

The next step should target the missing conditional uncertainty source directly while preserving the native model. A principled route is to diagnose whether the base AR flow is under-sampling its learned transition noise or whether the learned transition law itself is too narrow. That points to inference/training noise calibration diagnostics before another architecture change.
