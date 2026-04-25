# 490 H60 Recent-Quantile Result

## Context

489a tested raw H60 training and scored only `4/11`, but the successful H30 frontier was not raw 340c. It passed through the 385a recent-quantile coordinate adaptation before weak rollout-energy fine-tuning. 490 therefore ran the fair H60 analogue of 385a: same H60 checkpoint, same architecture, same FM objective, and recent empirical-normal-score quantiles from the 441-window pre-validation block.

## Execution

Training:

```bash
python experiments/backfill/block_ar/train_377a_340c_recent_fm_adaptation.py \
  --checkpoint models/backfill/489a_h60_empirical_score_transition_s42/best_model.pt \
  --history_len 60 \
  --future_len 30 \
  --quantile_source recent \
  --adaptation_windows 441 \
  --epochs 8 \
  --batch_size 32 \
  --output_dir models/backfill/490a_h60_recent_quantiles_fm_s42 \
  --seed 42 \
  --device cuda
```

Evaluation:

```bash
python experiments/backfill/block_ar/evaluate_220h_full_multihorizon_v2_suite.py \
  --model_type 340c \
  --checkpoint models/backfill/490a_h60_recent_quantiles_fm_s42/best_model.pt \
  --history_len 60 \
  --future_len 30 \
  --output_json results/block_ar/490a_h60_recent_quantiles_fm_s42/full11.json \
  --output_md results/block_ar/490a_h60_recent_quantiles_fm_s42/full11.md \
  --device cuda
```

## Result

490a scored `7/11`.

Passed:

- surface
- time_series
- block_ar
- cointegration
- cross_cell_correlation
- mean_reversion
- pathwise_jump_realism

Failed:

- coverage
- conditionality
- regime_coverage
- distributional_fidelity

Key metrics:

- coverage90: `0.8440`
- conditional MAE reduction: `3.9%`
- daily-change KS: `25/25`
- level KS: `2/25`
- median-bias fraction: `15/25`
- cointegration worst-cell ratio: `0.303`
- regime layer2: `1/8`
- pathwise KS: `0.315`

## Mechanism Read

Recent quantiles recover the H60 structural suites, but H60 still sits below the H30 392a frontier. The decisive regression is not local transition shape; it is conditional state allocation and level occupancy. Conditionality remains below gate, level KS is much worse than 392a (`2/25` versus `10/25`), and median-bias fraction falls to `15/25`.

This makes an H60 weak-energy follow-up unattractive: the known energy direction tends to trade conditionality for level occupancy, and 490a already fails conditionality before energy is added.

## Decision

Close H60 as the primary bottleneck. The extra context contains some generic signal, but the current AR transition core does not convert it into a better deployable conditional scenario law. Next step should be a synthesis/paradigm decision rather than another H60 knob.
