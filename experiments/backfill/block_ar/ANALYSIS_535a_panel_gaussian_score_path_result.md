# 535a Panel Gaussian Score-Path Law Result

## Context
534a found that the local repository has a clean aligned SPX IV + 13-factor panel. 535a implemented the smallest genuine panel-law feasibility prototype: train one probabilistic model over the full aligned panel, then evaluate the generated IV subpanel with the unchanged 11-suite.

## Implementation
Added:

- `diffusion/block_ar/coherent_panel_score_path_model.py`
- `experiments/backfill/block_ar/_panel_law_535_utils.py`
- `experiments/backfill/block_ar/train_535a_panel_gaussian_score_path.py`
- `experiments/backfill/block_ar/evaluate_535a_panel_gaussian_score_path.py`
- `test_code/test_535a_panel_score_path_model.py`

The model trains on 51 raw panel variables:

- 25 IV level tokens;
- 13 factor level tokens;
- 13 factor return/diff tokens.

It uses empirical normal-score coordinates per variable, a GRU history encoder, future token heads for `mu(H)` and `scale(H)`, a fixed global residual Cholesky over the full future panel path, exact Gaussian NLL, and one-shot sampling. The evaluator uses a history-keyed wrapper so the official IV conditionality tests can still call the model with IV histories while the sampler internally conditions on full panel histories.

Verification:

```bash
pytest test_code/test_535a_panel_score_path_model.py test_code/test_532a_coherent_gaussian_density.py -q
python -m py_compile diffusion/block_ar/coherent_panel_score_path_model.py experiments/backfill/block_ar/_panel_law_535_utils.py experiments/backfill/block_ar/train_535a_panel_gaussian_score_path.py experiments/backfill/block_ar/evaluate_535a_panel_gaussian_score_path.py
```

Result: `5 passed`; compile passed.

## Run
Training:

```bash
python experiments/backfill/block_ar/train_535a_panel_gaussian_score_path.py \
  --epochs 12 \
  --batch_size 32 \
  --lr 8e-4 \
  --cov_shrinkage 0.10 \
  --device cuda \
  --output_dir models/backfill/535a_panel_gaussian_score_path_s535
```

Best checkpoint: epoch `2`, validation panel NLL `0.2363`.

Evaluation:

```bash
python experiments/backfill/block_ar/evaluate_535a_panel_gaussian_score_path.py \
  --checkpoint models/backfill/535a_panel_gaussian_score_path_s535/best_model.pt \
  --max_windows 192 \
  --samples 48 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --batch_size 32 \
  --chunk_size 8 \
  --seed 535 \
  --device cuda \
  --output_json results/autoresearch/535a_panel_gaussian_score_path_s535/full11.json \
  --output_md results/autoresearch/535a_panel_gaussian_score_path_s535/full11.md
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

| metric | 535a |
|---|---:|
| cov90 overall | `0.725` |
| h30 cov90 | `0.704` |
| conditional MAE reduction | `1.17%` |
| turb/calm width ratio | `1.057` |
| regime layer2 | `0/8` |
| daily-change KS pass cells | `10/25` |
| level KS pass cells | `1/25` |
| cross-cell corr ratio | `0.227` |
| effective-rank ratio | `3.903` |
| mean-reversion active pass | `0.125` |
| path max-jump KS | `0.636` |

## Mechanism Read
535a is a valid panel-law feasibility prototype, but it does not improve the IV scenario generator. It reproduces the 532 Gaussian-family failure:

- Conditioning is too weak: MAE reduction is only `1.17%`.
- Level occupancy is worse than the frontier: level KS is `1/25`.
- Cross-cell dependence is too diffuse: corr ratio `0.227`, rank ratio `3.903`.
- Mean reversion is too strong and too broad: ratio `2.436`, active pass `0.125`.
- Regime and per-cell coverage remain badly uneven.

Adding factor variables as targets does not fix the model family. The bottleneck is not merely that 532a was IV-only; it is that a global Gaussian score-path law is not expressive enough for conditional IV path dependence.

## Decision
Close the small panel Gaussian prototype. Keep the broader panel-law program alive only if the next step changes the sequence model class, not by adding panel-specific calibration or covariance bins.

The next principled options are:

- a genuinely autoregressive panel-token density/flow with learned conditional dependence; or
- a data acquisition/pretraining expansion beyond the single SPX IV panel and 13-factor local dataset.

Do not run another global-Gaussian panel variant.
