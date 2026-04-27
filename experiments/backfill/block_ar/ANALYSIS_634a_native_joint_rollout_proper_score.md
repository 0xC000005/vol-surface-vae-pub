# 634a Native Joint Rollout Proper-Score Result

## Hypothesis

If 631a's main problem is one-step objective mismatch, then a small rollout-level proper score should improve 30-day scenario calibration while preserving the clean native joint architecture. The intended change was objective-only: keep the same state-conditioned encoded-increment transition model and add a low-weight full-path energy plus sliced-Wasserstein score over generated level and increment score paths.

## Execution

Implemented `train_634a_native_joint_rollout_proper_score.py` with:

- a differentiable rollout helper for `GenericStateConditionedIncrementFlowMatching`;
- an anchored loss `FM + 0.03 * (energy + 0.5 * sliced-Wasserstein)`;
- training over the same `joint38` panel and recent 2048-window frame as 631a;
- no IV/factor branch and no post-hoc deck gluing.

Focused verification passed:

```bash
pytest test_code/test_634a_native_joint_rollout_proper_score.py -q
```

Training command:

```bash
python experiments/backfill/block_ar/train_634a_native_joint_rollout_proper_score.py \
  --checkpoint models/backfill/631a_joint38_statecond_increment_scale_e8_w2048_s631/best_model.pt \
  --state_scope joint38 --epochs 2 --max_train_windows 2048 --batch_size 8 \
  --lr 1e-5 --weight_decay 1e-4 --train_sample_count 4 --rollout_flow_steps 4 \
  --rollout_weight 0.03 --sw_weight 0.5 --fm_anchor_weight 1.0 \
  --horizon_end_weight 1.5 --n_projections 16 --n_quantiles 16 \
  --max_train_batches 128 --max_val_batches 32 --seed 634 --device cuda \
  --output_dir models/backfill/634a_joint38_rollout_proper_e2_b128_s634
```

Best epoch was 1 with validation total `1.1283`.

## Result

The IV 11-suite fell to `3/11` versus 631a's `4/11`. 634a passed only `block_ar`, `cointegration`, and `cross_cell_correlation`.

Key IV metrics:

- surface explosion `54.7%`;
- overall 90% coverage `71.3%`;
- conditionality MAE reduction `4.4%`, with worst per-cell reduction `-24.7%`;
- turbulent/calm width ratio `1.014`;
- daily-change KS `24/25`;
- level KS `4/25`;
- median-bias pass `7/25`;
- cross-cell corr/rank ratios `0.928 / 1.170`;
- aggregate mean-reversion ratio `0.805`, but full-horizon active mean still failed;
- pathwise max-jump KS `0.549`, q90/q99 ratios `2.625 / 5.972`;
- q99 jump-scale pass `19/25`.

The joint-panel audit stayed structurally plausible:

- factor daily-change KS mean `0.103`, pass `12/13`;
- factor q99 abs-change median ratio `1.213`, pass `13/13`;
- factor-factor correlation shape `0.867`, generated mean absolute correlation `0.138` versus GT `0.225`;
- IV-factor correlation shape `0.880`, generated mean absolute correlation `0.130` versus GT `0.149`.

Artifacts:

- `models/backfill/634a_joint38_rollout_proper_e2_b128_s634/best_model.pt`
- `results/autoresearch/634a_joint38_rollout_proper_e2_b128_s634/full11.md`
- `results/autoresearch/634a_joint38_rollout_proper_e2_b128_s634/joint_panel.md`

## Mechanism Read

634a is a useful falsifier, not a promotion. The rollout score did not destroy the native joint factor/IV dependence, which supports the architecture and coordinate choice. But the high-dimensional full-path score did not solve risk-manager calibration. It made the generated IV deck less conditionally useful: worse coverage, weaker per-cell conditionality, worse level occupancy, and only marginally different jump realism.

The likely cause is that a global path energy/SW loss over all level and increment coordinates is too blunt. With one realized future per history, it can reward moving the sample cloud toward the realized path and batch-level marginal shape without enforcing the per-history conditional width, regime responsiveness, and cellwise occupancy that the risk manager needs.

## Decision

Reject 634a as the active model. Keep 631a as the active native joint learned base because it has better IV score and similarly clean joint mechanics.

The next step should not be another blind sweep of rollout weights. Run a focused post-experiment diagnostic/ideation pass: compare 631a vs 634a on conditional width, median path bias, tail scale, and regime stratification to design a more targeted but still generic conditional calibration objective.
