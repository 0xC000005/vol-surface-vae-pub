# 614a joint38 AR Gaussian likelihood result

## Context

613a falsified the simple feature-framing hypothesis: adding generic scale-prefix features made the native AR flow worse. The recurring issue was not just missing volatility features; it was calibrated conditional probability mass. 614a therefore shifted the objective while preserving the clean state-panel framing.

614a uses:

- the same IV-only / joint38 state-scope preprocessing;
- empirical-score state transformation;
- one shared causal Transformer memory over the selected panel;
- one shared autoregressive transition law;
- exact conditional NLL for the next-day joint score increment.

The main architectural change is replacing flow-matching velocity MSE with a full-covariance Gaussian transition density.

## Implementation

Added:

- `diffusion/block_ar/generic_gaussian_transition_law.py`
- `experiments/backfill/block_ar/train_614a_unified_ar_gaussian_likelihood.py`
- `experiments/backfill/block_ar/evaluate_614a_unified_ar_gaussian_likelihood.py`
- `test_code/test_614a_generic_gaussian_transition.py`

Verification:

```bash
python -m py_compile \
  diffusion/block_ar/generic_gaussian_transition_law.py \
  experiments/backfill/block_ar/train_614a_unified_ar_gaussian_likelihood.py \
  experiments/backfill/block_ar/evaluate_614a_unified_ar_gaussian_likelihood.py

pytest test_code/test_614a_generic_gaussian_transition.py -q
```

Result: `2 passed`.

## Run

Training:

- state scope: `joint38`;
- epochs: `8`;
- recent train windows: `2048`;
- memory dim: `128`;
- memory layers: `3`;
- head hidden: `256`;
- diagonal floor/max: `0.03 / 3.0`;
- off-diagonal scale: `0.25`;
- seed: `614`.

Training result:

- best epoch: `2`;
- best validation NLL: `15.682419`;
- final validation NLL: `47.916576`;
- finite sample rate: `1.0`.

The NLL overfit rapidly after epoch 2, so the official evaluation used the best checkpoint.

## Full Suite Result

Official IV bridge on 441 windows / 48 samples:

- score: `4/11`;
- passed: conditionality, block-AR, cointegration, cross-cell correlation;
- failed: surface, coverage, time-series properties, regime coverage, distributional fidelity, mean reversion, pathwise jump realism.

Key metrics:

- cov90 overall: `90.9%`;
- h1/h7/h14/h30 cov90: `87.5% / 91.3% / 91.3% / 90.5%`;
- conditional MAE reduction: `6.7%`;
- turbulent/calm width ratio: `0.926`;
- persistent severe undercoverage: `445/11025 = 4.0%`;
- regime layer2: `0/8`;
- daily-change KS cells: `12/25`;
- level KS cells: `0/25`;
- median-bias cells: `8/25`;
- bad coverage windows: `0/441`;
- ACF correlation: `0.979`;
- kurtosis ratio: `0.305`;
- cointegration gen/GT: `0.766`, worst-cell ratio `0.299`;
- cross-cell corr/rank: `0.822 / 1.611`;
- mean-reversion ratio: `0.304`, active pass `0/12`;
- pathwise max-jump KS: `0.646`;
- per-cell q99 jump-scale cells: `6/25`.

## Mechanism Read

614a is not deployable, but it is informative because it changes the failure mode.

Positive evidence for the objective shift:

- aggregate coverage becomes high rather than low;
- persistent severe undercoverage finally passes (`4.0%`);
- no validation window has less than 50% coverage;
- conditionality passes;
- cointegration and cross-cell structure remain acceptable.

Negative evidence for the plain Gaussian transition:

- samples are too diffuse and too Gaussian-smoothed;
- surface validity barely fails on calendar arbitrage;
- level KS collapses to `0/25`;
- median-bias cells collapse to `8/25`;
- generated kurtosis is far too low at the aggregate level (`0.305`);
- per-cell tail scale is badly uneven;
- h1 mean reversion is too weak because broad Gaussian innovations wash out the conditional pull.

This supports the objective-level direction but falsifies the plain full-covariance Gaussian as the final likelihood family.

## Decision

Keep the likelihood-trained AR transition paradigm alive, but do not ship 614a.

The next clean diagnostic should calibrate the Gaussian sampler temperature downward before adding a richer density family. This is not a generic fix for the old flow path; it is a direct test of whether 614a's main pathology is simply over-diffuse likelihood sampling. If moderate temperature restores surface/time-series/distributional realism while preserving the coverage and persistent-undercoverage gains, the NLL path is viable. If not, the next density family should be heavier-tailed or mixture-based rather than Gaussian.

## Artifacts

- `models/backfill/614a_joint38_ar_gaussian_nll_e8_w2048_s614/train_summary.json`
- `models/backfill/614a_joint38_ar_gaussian_nll_e8_w2048_s614/training_history.json`
- `results/autoresearch/614a_joint38_ar_gaussian_nll_e8_w2048/full11.json`
- `results/autoresearch/614a_joint38_ar_gaussian_nll_e8_w2048/full11.md`
