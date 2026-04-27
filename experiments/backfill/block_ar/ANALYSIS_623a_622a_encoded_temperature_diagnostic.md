# 623a Encoded-State Temperature Diagnostic

## Question

622a trained the native joint38 AR transition-flow in encoded log-level/diff-level coordinates. It restored mean reversion but collapsed coverage and produced medians above realized futures in most IV cells. 623a tested whether that failure is mainly stochastic width or deeper path-location/dependence bias.

## Run

Temperature `1.5` on the frozen 622a checkpoint:

```bash
python experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py \
  --checkpoint models/backfill/622a_joint38_ar_transition_encoded_e8_w2048_s622/best_model.pt \
  --state_scope joint38 \
  --value_coordinate encoded \
  --max_windows 441 \
  --samples 48 \
  --n_steps 30 \
  --batch_size 32 \
  --chunk_size 8 \
  --sample_temperature 1.5 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 62315 \
  --device cuda \
  --output_json results/autoresearch/623a_622a_encoded_temperature_diagnostic/temp150_full11.json \
  --output_md results/autoresearch/623a_622a_encoded_temperature_diagnostic/temp150_full11.md
```

## Result

Score: `2/11`.

Passed:

- block-AR;
- cointegration.

Failed:

- surface;
- coverage;
- conditionality;
- time-series;
- regime coverage;
- distributional fidelity;
- cross-cell correlation;
- mean reversion;
- pathwise jump realism.

Key metrics:

- cov90 overall: `77.1%` versus 622a `49.0%`;
- h1/h7/h14/h30 cov90: `90.6% / 83.1% / 76.1% / 70.1%`;
- conditionality MAE reduction: `-7.6%`;
- turbulent/calm width ratio: `1.002`;
- surface calendar arbitrage: `17.8%`, failing the `<15%` gate;
- kurtosis ratio: `0.202`;
- daily-change KS cells: `3/25`;
- level KS cells: `1/25`;
- median-bias cells: `1/25`;
- persistent severe undercoverage: `13.1%`;
- cointegration gen/GT: `0.813`, worst-cell `0.283`;
- cross-cell corr/rank: `0.244 / 3.281`;
- mean-reversion full-horizon active mean pass: `43.8%`;
- pathwise max-jump KS: `0.470`;
- per-cell q99 jump-scale cells: `3/25`.

## Mechanism Read

Temperature confirms that 622a is not merely too narrow.

What improves:

- aggregate and per-horizon coverage become much closer to usable;
- cointegration recovers.

What breaks:

- surface validity fails;
- cross-cell dependence collapses;
- daily-change and per-cell jump tails become extreme and uneven;
- full-horizon mean-reversion structure no longer passes;
- median/path-location bias remains almost unchanged.

The causal read is clean: scalar widening can inflate intervals around the wrong median path, but it cannot repair the conditional location law or dependence geometry. The encoded coordinate helps mean reversion at temperature 1.0, but it does not produce a risk-manager-deployable conditional scenario law.

## Decision

Close encoded-coordinate temperature as a deployability fix. Keep the code path because it is a useful general preprocessing option, but do not continue this branch through scalar widening or more local temperature sweeps.

The next model-side step should not be another one-step AR local adjustment. The evidence now points to a stochastic path law trained at the same granularity as the risk object: full generated future paths, but without the old one-shot MLP path-flow failure mode. The next ideation should specify a clean sequence-level stochastic objective that preserves AR/state feedback or introduces a minimal latent state-space path law while keeping one shared panel interface for IV-only and joint38.

