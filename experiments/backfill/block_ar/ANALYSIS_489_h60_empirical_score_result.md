# 489 H60 Empirical-Score AR Result

## Context

488 found that a 60-day history frame adds generic state signal over the 30-day frame in aligned ridge probes. 489 tested the cleanest model-side falsifier: keep the empirical-normal-score causal-memory transition FM unchanged and train the same vanilla AR core with `history_len=60`.

## Execution

Command:

```bash
python experiments/backfill/block_ar/train_340a_empirical_normal_score_causal_memory_transition_flow.py \
  --history_len 60 \
  --future_len 30 \
  --epochs 30 \
  --batch_size 64 \
  --prefix_feature_mode basic \
  --output_dir models/backfill/489a_h60_empirical_score_transition_s42 \
  --seed 42 \
  --device cuda
```

Evaluation:

```bash
python experiments/backfill/block_ar/evaluate_220h_full_multihorizon_v2_suite.py \
  --model_type 340c \
  --checkpoint models/backfill/489a_h60_empirical_score_transition_s42/best_model.pt \
  --history_len 60 \
  --future_len 30 \
  --output_json results/block_ar/489a_h60_empirical_score_transition_s42/full11.json \
  --output_md results/block_ar/489a_h60_empirical_score_transition_s42/full11.md \
  --device cuda
```

## Result

489a scored `4/11`.

Passed:

- surface
- block_ar
- cross_cell_correlation
- mean_reversion

Failed:

- coverage
- conditionality
- time_series
- cointegration
- regime_coverage
- distributional_fidelity
- pathwise_jump_realism

Key metrics:

- best epoch: `12`, teacher-forced val loss `0.4492`
- coverage90: `0.8511`
- conditional MAE reduction: `2.63%`
- daily-change KS: `24/25`
- level KS: `6/25`
- median-bias fraction: `17/25`
- cointegration worst-cell ratio: `0.172`
- regime layer2: `1/8`
- pathwise per-cell q99: `19/25`

## Mechanism Read

Longer context alone is not enough. It improves neither conditionality nor level/regime allocation in rollout. It also weakens the structural margins that made 392a deployable: conditionality drops below gate, one cointegration cell fails, tail-scale/pathwise per-cell gates fail by one or more cells, and level KS remains far below the 15/25 gate.

This does not fully close H60 yet because the successful H30 frontier was not the raw 340c train-from-scratch checkpoint. The H30 path needed the 385 recent-quantile coordinate adaptation and then weak rollout-energy fine-tuning.

## Decision

Run one fair H60 analog of 385a next: recent-quantile FM adaptation from 489a using the same 441-window pre-validation block. If that cannot recover the structural passes and approach the 8/11 frontier, close H60 as a primary bottleneck.
