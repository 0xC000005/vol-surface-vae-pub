# 597a Global Temperature Diagnostic

## Question

596a showed that final-path joint-distribution finetuning narrows the conditional scenario deck. The next clean diagnostic was to test whether broad-frame undercoverage is simply an inference-time sampling/noise calibration issue in the existing 510a AR flow.

If global temperature fixes coverage without breaking realism, the bottleneck is sampler calibration. If not, the learned transition law or available conditioning signal is the bottleneck.

## Commands

```bash
python experiments/backfill/block_ar/evaluate_220h_full_multihorizon_v2_suite.py \
  --model_type 340c \
  --checkpoint models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt \
  --sample_temperature_override 1.10 \
  --max_windows 441 \
  --samples 48 \
  --batch_size 32 \
  --chunk_size 8 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --device cuda \
  --output_json results/autoresearch/597b_510a_temp110_same441/full11.json \
  --output_md results/autoresearch/597b_510a_temp110_same441/full11.md
```

```bash
python experiments/backfill/block_ar/evaluate_220h_full_multihorizon_v2_suite.py \
  --model_type 340c \
  --checkpoint models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt \
  --sample_temperature_override 1.15 \
  --max_windows 441 \
  --samples 48 \
  --batch_size 32 \
  --chunk_size 8 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --device cuda \
  --output_json results/autoresearch/597a_510a_temp115_same441/full11.json \
  --output_md results/autoresearch/597a_510a_temp115_same441/full11.md
```

## Result

| Metric | 510a temp 1.00 | temp 1.10 | temp 1.15 |
| --- | ---: | ---: | ---: |
| Broad-frame score | `5/11` | `4/11` | `4/11` |
| Coverage90 overall | `69.6%` | `73.4%` | `75.0%` |
| h1/h7/h14/h30 coverage90 | `79.4/70.4/68.7/66.2` | `82.5/74.7/72.7/69.4` | `84.5/76.7/73.8/71.0` |
| Conditional MAE reduction | `7.61%` | `6.63%` | `6.79%` |
| Worst-cell MAE reduction | `-13.1%` | `-15.7%` | `-15.0%` |
| Kurtosis ratio | `0.618` | `0.515` | `0.467` |
| Skewness ratio | `0.458` | `0.275` | `0.125` |
| Tail-scale cells | `24/25` | `21/25` | `15/25` |
| Cointegration worst-cell ratio | `0.283` | `0.239` | `0.328` |
| Regime layer2 | `0/8` | `0/8` | `0/8` |
| Persistent undercoverage | `16.2%` | `14.5%` | `13.8%` |
| Daily-change KS cells | `25/25` | `24/25` | `17/25` |
| Level KS cells | `3/25` | `2/25` | `2/25` |
| Median-bias cells | `10/25` | `9/25` | `8/25` |
| Bad-window rate | `24.0%` | `18.6%` | `18.4%` |
| Cross-cell corr / rank | `1.070 / 1.225` | `0.932 / 1.509` | `0.862 / 1.668` |
| Mean-reversion active corr | `0.531` | `0.526` | `0.532` |
| Pathwise max-jump KS | `0.463` | `0.321` | `0.258` |

## Mechanism Read

Global temperature confirms that the base model is under-dispersed in an aggregate sense: increasing temperature improves horizon coverage, persistent undercoverage, bad-window rate, and pathwise max-jump KS.

But it is not a deployable fix. The same scalar widening worsens conditionality, level KS, median bias, time-series tail shape, tail-scale cells, and sometimes cointegration/pathwise subtests. Most importantly, regime layer2 stays `0/8` and per-cell undercoverage remains severe. A scalar temperature cannot allocate uncertainty to the right histories, cells, and horizons.

This also explains why prior source-scale branches were capped:

- `448a` unconstrained conditional scale collapsed noise and scored `4/11`.
- `449a` widening-only scale stayed at identity and scored `7/11`.
- `450a` joint conditional scale also stayed near identity and scored `7/11`.

The missing component is not "more global noise." It is conditional allocation and/or conditional location in sparse future regimes.

## Decision

Close global temperature as a solution route. It is useful as a diagnostic and possibly a risk-policy overlay, but it is not a learned conditional scenario generator.

The next research step should not be another source-noise knob. The evidence points to an IV-only information bottleneck or transition-law limitation: the model does not know when and where to put future level mass. The principled next move is a paradigm decision about adding genuinely informative state variables/factors to the same unified factor generator, not more IV-only sampler calibration.
