# 532a Coherent Gaussian Score-Path Density Result

## Context
531a selected the smallest non-redundant marginalization-consistent density falsifier: one explicit conditional Gaussian path law over the full 30x25 future IV-level path in empirical normal-score coordinates. This was meant to avoid the known `323`/`492` failure mode where a separately trained marginal map breaks path geometry.

## Implementation
Added:

- `diffusion/block_ar/coherent_gaussian_score_path_model.py`
- `experiments/backfill/block_ar/train_532a_coherent_gaussian_score_path.py`
- `experiments/backfill/block_ar/evaluate_532a_coherent_gaussian_score_path.py`
- `test_code/test_532a_coherent_gaussian_density.py`

The model uses:

- empirical normal-score transform fitted on training future levels;
- a GRU history encoder;
- future token embeddings for horizon/cell;
- direct heads for future-path `mu(H)` and `scale(H)`;
- a fixed full global residual Cholesky estimated from training future score paths;
- exact Gaussian NLL;
- one-shot deployable sampling from history plus Gaussian noise.

Verification:

```bash
pytest test_code/test_532a_coherent_gaussian_density.py -q
python -m py_compile diffusion/block_ar/coherent_gaussian_score_path_model.py experiments/backfill/block_ar/train_532a_coherent_gaussian_score_path.py experiments/backfill/block_ar/evaluate_532a_coherent_gaussian_score_path.py
```

Result: `3 passed`; compile passed.

## Run
Training:

```bash
python experiments/backfill/block_ar/train_532a_coherent_gaussian_score_path.py \
  --epochs 24 \
  --batch_size 64 \
  --lr 1e-3 \
  --cov_shrinkage 0.10 \
  --device cuda \
  --output_dir models/backfill/532a_coherent_gaussian_score_path_s532
```

Validation NLL selected epoch `1`; later epochs overfit monotonically.

Evaluation:

```bash
python experiments/backfill/block_ar/evaluate_532a_coherent_gaussian_score_path.py \
  --checkpoint models/backfill/532a_coherent_gaussian_score_path_s532/best_model.pt \
  --max_windows 192 \
  --samples 48 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --batch_size 32 \
  --chunk_size 8 \
  --seed 532 \
  --device cuda \
  --output_json results/autoresearch/532a_coherent_gaussian_score_path_s532/full11.json \
  --output_md results/autoresearch/532a_coherent_gaussian_score_path_s532/full11.md
```

## Result
Score: `3/11`.

Passed:

- `surface`
- `block_ar`
- `cointegration`

Failed:

- `coverage`
- `conditionality`
- `time_series`
- `regime_coverage`
- `distributional_fidelity`
- `cross_cell_correlation`
- `mean_reversion`
- `pathwise_jump_realism`

Key metrics:

| metric | 532a |
|---|---:|
| cov90 overall | `0.760` |
| h30 cov90 | `0.745` |
| conditional MAE reduction | `1.24%` |
| turb/calm width ratio | `1.143` |
| regime layer2 | `0/8` |
| daily-change KS pass cells | `10/25` |
| level KS pass cells | `2/25` |
| cross-cell corr ratio | `0.230` |
| effective-rank ratio | `3.895` |
| mean-reversion active pass | `0.083` |
| path max-jump KS | `0.748` |

## Mechanism Read
This cleanly falsifies the simple Gaussian marginalization-consistent route. The model is coherent as a density, but the family is too weak:

- Conditional dependence on history is too weak: MAE reduction is only `1.24%`.
- The global residual Cholesky does not preserve validation cross-cell structure: corr ratio falls to `0.230` and rank ratio rises to `3.895`.
- The one-shot mean head overuses current-level information and over-mean-reverts: aggregate ratio `2.361`, active-cell pass `0.083`.
- The empirical score transform does not by itself solve level occupancy: level KS is only `2/25`.
- Pathwise jumps are not realistic despite acceptable q90/q99 ratios: max-jump KS is `0.748`.

The important lesson is not that marginalization consistency is wrong. It is that a single global Gaussian residual copula plus per-token conditional mean/scale is not expressive enough for this IV scenario law. The remaining problem requires learned conditional dependence, not just coherent marginals.

## Decision
Close the simple coherent Gaussian density route. Do not add ad hoc state-dependent covariance bins, hand-set mean-reversion damping, or per-cell calibration to this model; that would recreate the old research-knob pathology.

The next principled step is a paradigm decision: either build a larger learned conditional dependence model with explicit likelihood, or return to the current deployable frontier and report the limitation honestly. A small local Gaussian-copula repair is no longer credible.
