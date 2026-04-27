# 616a joint38 AR Student-t likelihood result

## Context

615a closed simple sampler-temperature calibration for the Gaussian likelihood model. The likelihood paradigm improved undercoverage, but Gaussian innovations had the wrong shape. 616a keeps the exact same state-panel architecture and NLL objective, replacing the Gaussian transition family with a fixed-degree-of-freedom multivariate Student-t transition.

## Change

Updated:

- `diffusion/block_ar/generic_gaussian_transition_law.py`
- `experiments/backfill/block_ar/train_614a_unified_ar_gaussian_likelihood.py`
- `test_code/test_614a_generic_gaussian_transition.py`

The model now supports:

- `distribution_family=gaussian`;
- `distribution_family=student_t`;
- fixed `student_t_df > 2`.

The Student-t branch uses the same full-covariance Cholesky head and exact multivariate Student-t log likelihood.

Verification:

```bash
python -m py_compile \
  diffusion/block_ar/generic_gaussian_transition_law.py \
  experiments/backfill/block_ar/train_614a_unified_ar_gaussian_likelihood.py \
  experiments/backfill/block_ar/evaluate_614a_unified_ar_gaussian_likelihood.py

pytest test_code/test_614a_generic_gaussian_transition.py -q
```

Result: `3 passed`.

## Run

Training:

- state scope: `joint38`;
- epochs: `8`;
- recent train windows: `2048`;
- distribution family: `student_t`;
- degrees of freedom: `5`;
- memory dim: `128`;
- memory layers: `3`;
- head hidden: `256`;
- seed: `616`.

Training result:

- best epoch: `4`;
- best validation NLL: `-9.594840`;
- final validation NLL: `-2.883219`;
- finite sample rate: `1.0`.

Compared with the Gaussian NLL run, validation behavior was less immediately degenerate: best epoch moved from `2` to `4`.

## Full Suite Result

Official IV bridge on 441 windows / 48 samples:

- score: `4/11`;
- passed: conditionality, block-AR, cointegration, cross-cell correlation;
- failed: surface, coverage, time-series properties, regime coverage, distributional fidelity, mean reversion, pathwise jump realism.

Key metrics:

- cov90 overall: `92.2%`;
- h1/h7/h14/h30 cov90: `88.3% / 92.5% / 92.6% / 92.2%`;
- conditional MAE reduction: `7.9%`;
- turbulent/calm width ratio: `0.901`;
- persistent severe undercoverage: `397/11025 = 3.6%`;
- regime layer2: `0/8`;
- daily-change KS cells: `14/25`;
- level KS cells: `0/25`;
- median-bias cells: `21/25`;
- bad coverage windows: `0/441`;
- ACF correlation: `0.978`;
- kurtosis ratio: `0.452`;
- cointegration gen/GT: `0.903`, worst-cell ratio `0.545`;
- cross-cell corr/rank: `0.747 / 1.833`;
- mean-reversion ratio: `0.427`, active pass `0/12`;
- pathwise max-jump KS: `0.403`;
- per-cell q99 jump-scale cells: `6/25`.

## Comparison to Gaussian NLL

| run | score | cov90 | cond MAE | cat. undercov | daily KS | level KS | bias cells | kurtosis | coin worst | MR ratio | path KS | q99 cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 614a Gaussian | 4/11 | 90.9% | 6.7% | 4.0% | 12/25 | 0/25 | 8/25 | 0.305 | 0.299 | 0.304 | 0.646 | 6/25 |
| 616a Student-t | 4/11 | 92.2% | 7.9% | 3.6% | 14/25 | 0/25 | 21/25 | 0.452 | 0.545 | 0.427 | 0.403 | 6/25 |

## Mechanism Read

Student-t is a better likelihood family than Gaussian, but still not deployable uncalibrated.

Improved:

- conditionality margin;
- persistent severe undercoverage;
- median-bias cells;
- cointegration worst-cell;
- mean-reversion ratio, though still failing;
- pathwise max-jump KS.

Still broken:

- overcoverage / per-cell coverage upper cap;
- surface calendar rate, likely from over-wide noisy samples;
- level KS remains `0/25`;
- per-cell q99 tail scale remains extremely uneven;
- aggregate kurtosis remains too low despite heavier tails;
- h1 mean reversion remains too weak.

This says the density-family shift is directionally useful, but the Student-t samples are still globally too wide and not state-placement accurate enough.

## Decision

Keep Student-t likelihood alive, but do not ship 616a.

The next controlled step should run a Student-t sampler-temperature diagnostic. Unlike Gaussian, Student-t improves pathwise max-jump structure and conditionality at temperature `1.0`, so a moderate down-temperature may reduce overcoverage/surface/tail-scale damage without fully losing the undercoverage gains.

If that fails, the next objective-level step should target the conditional mean/placement path directly rather than only innovation shape.

## Artifacts

- `models/backfill/616a_joint38_ar_studentt_nll_e8_w2048_s616/train_summary.json`
- `models/backfill/616a_joint38_ar_studentt_nll_e8_w2048_s616/training_history.json`
- `results/autoresearch/616a_joint38_ar_studentt_nll_e8_w2048/full11.json`
- `results/autoresearch/616a_joint38_ar_studentt_nll_e8_w2048/full11.md`
