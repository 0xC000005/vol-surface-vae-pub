# 590a Latent-Source Temperature Diagnostic

## Hypothesis

589a passes official surface validity and cross-cell structure but is severely
under-dispersed. If the latent-bottleneck source has a viable spread frontier,
then a generic source-temperature increase should recover coverage and jump tails
without destroying surface and cross-cell passes.

This is a diagnostic, not a final deployable calibration claim.

## Change

Added a generic `source_temperature` hook to `UnifiedIncrementFlow` sampling:

- conditional-affine source: scales Gaussian noise around the conditional mean;
- conditional-latent source: scales the latent-decoded stochastic residual;
- path-Gaussian source: scales the Gaussian residual around the fitted mean.

Added `--source_temperature` to
`evaluate_588a_unified_flow_full11_bridge.py`.

## Runs

Focused tests:

```bash
pytest test_code/test_577a_unified_increment_flow.py \
  test_code/test_588a_unified_flow_full11_bridge.py -q
```

Result: `11 passed`.

Official temperature diagnostics:

```bash
python experiments/backfill/block_ar/evaluate_588a_unified_flow_full11_bridge.py \
  --checkpoint models/backfill/589a_conditional_latent_cumulative_flow_s589/best_model.pt \
  --max_windows 441 \
  --samples 48 \
  --sample_steps 16 \
  --source_temperature 2.0 \
  --batch_size 32 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 5902 \
  --device cuda \
  --output_json results/autoresearch/590a_latent_temperature_sweep/temp2_full11.json \
  --output_md results/autoresearch/590a_latent_temperature_sweep/temp2_full11.md
```

```bash
python experiments/backfill/block_ar/evaluate_588a_unified_flow_full11_bridge.py \
  --checkpoint models/backfill/589a_conditional_latent_cumulative_flow_s589/best_model.pt \
  --max_windows 441 \
  --samples 48 \
  --sample_steps 16 \
  --source_temperature 4.0 \
  --batch_size 32 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 5904 \
  --device cuda \
  --output_json results/autoresearch/590a_latent_temperature_sweep/temp4_full11.json \
  --output_md results/autoresearch/590a_latent_temperature_sweep/temp4_full11.md
```

## Result

| Metric | temp 1.0 / 589a | temp 2.0 | temp 4.0 |
|---|---:|---:|---:|
| Official score | `3/11` | `2/11` | `1/11` |
| Surface validity | PASS | PASS | FAIL |
| Cross-cell structure | PASS | FAIL | FAIL |
| Overall 90% coverage | `26.7%` | `47.3%` | `65.1%` |
| h30 90% coverage | `28.9%` | `50.2%` | `68.2%` |
| ACF | FAIL | PASS | PASS |
| Daily-change KS pass cells | `1/25` | `5/25` | `10/25` |
| Level KS pass cells | `2/25` | `3/25` | `1/25` |
| Cross-cell corr ratio | `0.578` | `0.487` | `0.370` |
| Rank ratio | `1.821` | `2.130` | `2.493` |
| Pathwise max-jump KS | `0.999` | `0.991` | `0.833` |
| q99 jump-scale pass cells | `1/25` | `5/25` | `11/25` |

## Mechanism Read

The temperature frontier is real but unfavorable:

- increasing temperature improves coverage and local move-size statistics;
- the same increase degrades cross-cell dependence and eventually surface
  validity;
- even at temp `4.0`, pathwise max-jump KS and regime coverage remain far from
  passing;
- no tested global temperature beats the `3/11` base latent model.

This means 589a's underdispersion is not a simple scalar amplitude problem. The
latent bottleneck recovers rank/correlation only while too narrow; broadening it
breaks the dependence structure before coverage/tails become acceptable.

## Decision

Close the unified MLP path-flow branch as a frontier route.

The useful lessons are:

- cumulative/final-series-oriented loss is necessary for level stability;
- a narrow stochastic bottleneck is necessary for cross-cell structure;
- a one-shot MLP path decoder cannot jointly preserve cross-cell dependence,
  local jump tails, coverage, and mean reversion under a global amplitude knob.

The next paradigm should not be another temperature, latent-dimension, or MLP
depth sweep. It should return to the historically stronger autoregressive
transition family or use a proper sequential/state-space model where local jumps,
mean reversion, and cross-cell common factors are learned stepwise while keeping
the latent common-noise lesson from 589a.
