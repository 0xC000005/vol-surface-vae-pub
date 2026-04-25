# 526a Factor-Conditioned Surface FM Full Evaluation

## Context
525a implemented the clean factor-conditioned surface-level mechanics. 526a ran the actual short adaptation and official full 11-suite evaluation.

## Run
Training:

```bash
python experiments/backfill/block_ar/train_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt \
  --adaptation_windows 441 \
  --epochs 4 \
  --batch_size 32 \
  --lr 5e-4 \
  --device cuda \
  --output_dir models/backfill/525a_factor_conditioned_surface_fm_e4_s525
```

Evaluation:

```bash
python experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/525a_factor_conditioned_surface_fm_e4_s525/best_model.pt \
  --max_windows 192 \
  --samples 48 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --batch_size 32 \
  --chunk_size 4 \
  --device cuda \
  --output_json results/autoresearch/526a_factor_conditioned_surface_fm_e4_s525/full11.json \
  --output_md results/autoresearch/526a_factor_conditioned_surface_fm_e4_s525/full11.md
```

## Result
Score: `7/11`.

Passed:

- `surface`
- `time_series`
- `block_ar`
- `cointegration`
- `cross_cell_correlation`
- `mean_reversion`
- `pathwise_jump_realism`

Failed:

- `coverage`
- `conditionality`
- `regime_coverage`
- `distributional_fidelity`

Key metrics versus frontier:

| metric | 392a | 510a | 526a |
|---|---:|---:|---:|
| score | `8/11` | `8/11` | `7/11` |
| cov90 overall | `0.868` | `0.873` | `0.865` |
| h30 cov90 | `0.885` | `0.891` | `0.878` |
| conditional MAE reduction | `5.14%` | `5.12%` | `4.74%` |
| turb/calm width ratio | `1.057` | `1.084` | `1.063` |
| cointegration gen/GT ratio | `0.700` | `0.595` | `0.726` |
| cointegration worst-cell ratio | `0.278` | `0.257` | `0.298` |
| regime layer2 | `0/8` | `0/8` | `0/8` |
| daily KS | `25/25` | `25/25` | `25/25` |
| level KS | `10/25` | `10/25` | `12/25` |
| median-bias cells | `20/25` | `20/25` | `20/25` |
| bias-magnitude cells | `25/25` | `25/25` | `25/25` |
| corr ratio | `0.963` | `0.968` | `0.958` |
| rank ratio | `1.495` | `1.462` | `1.498` |
| mean-reversion ratio | `1.024` | `0.986` | `1.015` |
| mean-reversion active pass | `0.833` | `0.833` | `0.833` |
| path max-jump KS | `0.373` | `0.361` | `0.374` |

## Mechanism Read
The factor side-channel is not a collapse. It preserves almost all 392a structural behavior and improves two important hard diagnostics: level KS and cointegration margin. It also eliminates the 522a daily-change-coordinate pathology.

However, it does not clear the acceptance gate because conditionality falls just below the hard threshold (`4.74%` vs `>5%`). Coverage is also not truly worse in a risk-manager sense: h1/h7/h14/h30 coverage all pass, worst-cell lower bounds pass, and the failure is a high h30 best-cell overcoverage (`97.4%`). Regime and distributional fidelity remain the same core frontier bottlenecks.

## Decision
Do not add a new loss or architecture knob yet. The next principled iteration is a repeatability audit of this same checkpoint/evaluator with a different sampling seed, because the lost pass is within sampling noise scale. If it remains `7/11`, the route is promising but below-frontier; if it reaches `8/11`, compare it to 392a/510a as a candidate frontier variant before any further modification.
