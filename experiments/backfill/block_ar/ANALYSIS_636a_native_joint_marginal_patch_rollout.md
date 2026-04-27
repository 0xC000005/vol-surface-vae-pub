# 636a Native Joint Marginal-Patch Rollout Result

## Hypothesis

635a showed that 634a's global full-path energy/SW objective was too blunt: it preserved broad joint geometry while damaging IV conditional calibration. 636a tested a more local proper-scoring objective using the same clean architecture and differentiable rollout helper:

- per-history, per-horizon, per-channel marginal CRPS/energy over level scores;
- marginal CRPS/energy over increment scores;
- a small short-patch multivariate energy anchor for dependence;
- no IV/factor branch and no post-hoc deck gluing.

## Execution

Implemented `train_636a_native_joint_marginal_patch_rollout.py` and focused tests:

```bash
pytest test_code/test_636a_native_joint_marginal_patch_rollout.py test_code/test_634a_native_joint_rollout_proper_score.py -q
```

Training command:

```bash
python experiments/backfill/block_ar/train_636a_native_joint_marginal_patch_rollout.py \
  --checkpoint models/backfill/631a_joint38_statecond_increment_scale_e8_w2048_s631/best_model.pt \
  --state_scope joint38 --epochs 2 --max_train_windows 2048 --batch_size 8 \
  --lr 1e-5 --weight_decay 1e-4 --train_sample_count 4 --rollout_flow_steps 4 \
  --marginal_weight 0.05 --increment_weight 0.5 --patch_weight 0.01 \
  --fm_anchor_weight 1.0 --patch_len 5 --max_train_batches 128 --max_val_batches 32 \
  --seed 636 --device cuda \
  --output_dir models/backfill/636a_joint38_marginal_patch_e2_b128_s636
```

Best epoch was 1 with validation total `1.1368`.

## Result

The IV full 11-suite fell to `2/11`. 636a passed only `block_ar` and `cross_cell_correlation`; `cointegration` failed on the worst-cell gate.

Key IV metrics:

- surface explosion `58.2%`;
- overall 90% coverage `67.5%`;
- conditionality MAE reduction `2.5%`, worst per-cell `-42.5%`;
- turbulent/calm width ratio `1.026`;
- daily-change KS `21/25`;
- level KS `2/25`;
- median-bias pass `4/25`;
- cross-cell corr/rank ratios `0.945 / 1.143`;
- mean-reversion ratio `0.771`, but full-horizon active mean still failed;
- pathwise max-jump KS `0.593`, q90/q99 ratios `2.755 / 6.559`;
- q99 jump-scale pass `19/25`.

The joint-panel audit still stayed structurally plausible:

- factor daily-change KS mean `0.109`, pass `12/13`;
- factor q99 abs-change median ratio `1.250`, pass `13/13`;
- factor-factor correlation shape `0.868`, generated mean absolute correlation `0.142` versus GT `0.225`;
- IV-factor correlation shape `0.879`, generated mean absolute correlation `0.137` versus GT `0.149`.

Artifacts:

- `models/backfill/636a_joint38_marginal_patch_e2_b128_s636/best_model.pt`
- `results/autoresearch/636a_joint38_marginal_patch_e2_b128_s636/full11.md`
- `results/autoresearch/636a_joint38_marginal_patch_e2_b128_s636/joint_panel.md`

## Mechanism Read

636a falsifies the immediate "make the rollout proper score more local" idea. It damaged the IV deck more than 634a: coverage fell, level KS and median-bias collapsed, conditionality worsened, and pathwise jump KS worsened. The joint-factor audit remained plausible, so the native 38-channel panel interface is still not the failure. The failure is the rollout fine-tuning route from 631a.

The shared pattern across 634a and 636a is terminal level-spread and occupancy damage. The training rollouts can match short-run score-space samples well enough to improve or preserve broad joint dependence, but they move the free-running IV level distribution away from the validation future. This indicates that the remaining issue is not the choice of global-vs-local proper score alone.

## Decision

Reject 636a. Keep 631a as the active native joint base.

Do not continue sweeping rollout-loss variants from 631a. The next principled move is a paradigm/coordinate diagnostic: address the state evolution itself. A candidate is a native 38-channel score-state transition that generates changes in empirical level-score coordinates rather than raw encoded log/diff increments, because the current increment integration accumulates level drift and explosions even when daily-change and joint-factor metrics are reasonable.
