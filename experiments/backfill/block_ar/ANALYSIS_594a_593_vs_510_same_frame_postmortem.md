# 594a Same-Frame Postmortem: 593a vs 510a

## Question

593a scored `4/11`, apparently much worse than the prior 510a frontier score of `8/11`. But the stored 510a frontier artifact used `192` validation windows, while 593a used `441` windows. 594a reran 510a on the same `441` validation-window frame before deciding whether the learned common-latent wrapper truly collapsed.

## Same-Frame 510a Command

```bash
python experiments/backfill/block_ar/evaluate_220h_full_multihorizon_v2_suite.py \
  --model_type 340c \
  --checkpoint models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt \
  --max_windows 441 \
  --samples 48 \
  --batch_size 32 \
  --chunk_size 8 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --device cuda \
  --output_json results/autoresearch/594a_510a_same441/full11.json \
  --output_md results/autoresearch/594a_510a_same441/full11.md
```

## Result

On the same 441-window frame:

| Metric | 510a same-frame | 593a wrapper |
| --- | ---: | ---: |
| Suite score | `5/11` | `4/11` |
| Failed suites | `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion` | `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion` |
| Coverage90 overall | `69.6%` | `69.7%` |
| Coverage90 h1 / h30 | `79.4% / 66.2%` | `79.4% / 66.8%` |
| Conditional MAE reduction | `7.61%` | `6.55%` |
| Worst-cell MAE reduction | `-13.1%` | `-12.8%` |
| Turbulent/calm width ratio | `1.093` | `1.077` |
| ACF correlation | `0.983` | `0.982` |
| Kurtosis ratio | `0.618` | `0.623` |
| Skewness ratio | `0.458` | `-0.133` |
| Tail-scale cells | `24/25` | `24/25` |
| Cointegration gen/GT ratio | `0.606` | `0.621` |
| Cointegration worst-cell ratio | `0.283` | `0.239` |
| Regime layer2 | `0/8` | `0/8` |
| Persistent undercoverage | `16.2%` | `16.2%` |
| Daily-change KS cells | `25/25` | `25/25` |
| Level KS cells | `3/25` | `3/25` |
| Median-bias cells | `10/25` | `11/25` |
| Bad-window rate | `24.0%` | `23.1%` |
| Cross-cell corr / rank | `1.070 / 1.225` | `1.062 / 1.229` |
| Mean-reversion ratio | `1.157` | `1.166` |
| Mean-reversion active pass / corr | `75.0% / 0.531` | `75.0% / 0.560` |
| Pathwise max-jump KS | `0.463` | `0.475` |

## Mechanism Read

The original `8/11` frontier score is frame-sensitive. On the broader 441-window frame, 510a itself drops to `5/11`; therefore 593a did not collapse from a robust `8/11` baseline. It mostly reproduces the same broader-frame failure profile.

The learned common latent is still not useful. It preserves most 510a behavior but does not fix the shared structural failures: long-horizon undercoverage, regime layer2 `0/8`, persistent undercoverage around `16%`, level KS `3/25`, bad-window coverage around `23-24%`, weak active-cell mean-reversion correlation, and insufficient aggregate kurtosis. It also costs one suite by pushing cointegration worst-cell ratio below the floor and flipping aggregate skewness negative.

## Decision

593a should be closed as an improvement branch. The wrapper is clean, but the evidence says it is a near-neutral perturbation of 510a with extra failure risk, not a mechanism that solves the hard requirements.

The more important 594a finding is that future claims should distinguish:

- `short-frame score`: the historical 192-window artifact where 510a/392a reached `8/11`;
- `broad-frame score`: the 441-window validation frame where 510a is only `5/11`.

The next research move should not be another small wrapper on 510a. The next move should target the common failure mechanism directly: conditional distribution calibration across the whole future path, especially per-window coverage floors, regime-cell coverage, level location, and active-cell mean-reversion geometry.
