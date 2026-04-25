# 474 Source-Transport Postmortem

## Context

The recent branch tested whether the 392a frontier could be improved without changing
the core conditional path model:

- `470a`: frozen 392a center plus iid-Gaussian residual flow.
- `471a`: frozen 392a residual samples as the source law, then learned residual transport.
- `472a`: global source residual temperature.
- `473a/b`: partial transport strength.

The baseline remains `392a`, the current deployable frontier at `8/11`.

## Result Comparison

| model | score | failed suites | key read |
| --- | ---: | --- | --- |
| `392a` | `8/11` | coverage, regime coverage, distributional fidelity | best deployable frontier; conditional structure is good |
| `470a` | `3/11` | coverage, conditionality, time-series, regime, distributional, correlation, MR, pathwise | iid residual source destroys geometry |
| `471a` | `6/11` | coverage, time-series, regime, distributional, MR | source geometry is restored, but level/allocation failures remain |
| `472a` | `4/11` | coverage, conditionality, time-series, cointegration, regime, distributional, MR | scalar source shrinkage damages conditional structure |
| `473a` | `7/11` | coverage, regime, distributional, MR | partial transport is better than full transport but still below 392a |

Important metrics:

- `392a`: coverage90 `0.8675`, level KS `10/25`, median-bias fraction `20/25`, MR active pass `83.3%`, corr ratio `0.963`, pathwise KS `0.373`.
- `471a`: coverage90 `0.9061`, level KS `10/25`, median-bias fraction `19/25`, MR active pass `54.2%`, corr ratio `0.943`, pathwise KS `0.243`.
- `473a`: coverage90 `0.8911`, level KS `11/25`, median-bias fraction `20/25`, MR active pass `62.5%`, corr ratio `0.954`, pathwise KS `0.310`.

## Mechanism

The source-transport branch answered one question cleanly:

Preserving the 392a residual source law is necessary. Replacing it with iid Gaussian
noise destroys cross-cell dependence and effective rank. This is why `470a` collapses.

But preserving source geometry is not sufficient. `471a` and `473a` keep the desirable
correlation/pathwise structure yet still fail the same hard suites:

- coverage cap failures are bidirectional by cell and horizon, not one scalar width error;
- regime layer-2 coverage has both undercovered and overcovered cells in the same regime;
- level KS remains around `10-13/25`, far below the `15/25` gate;
- h1 mean-reversion active-cell coverage is weakened by learned transport.

So the bottleneck is conditional level/allocation, not residual shape alone.

## Prior Negative Evidence

This postmortem is consistent with earlier branches:

- `444a` direct level-quantile and interval-pinball fine-tuning scored `5/11` and damaged conditionality, time-series, cointegration, and level KS.
- `446a` stationary history-marginal mapping scored `6/11` and damaged conditionality/cointegration.
- `448a-450a` conditional noise-scale heads either collapsed noise or stayed at identity, scoring `4/11` to `7/11`.
- `462a-465a` calibration variants improved pieces of coverage but stayed below the frontier and weakened conditional structure.

The repeated pattern is stable: wrappers or scalar calibration levers can move coverage,
but they do not learn the missing conditional level allocation without breaking other
required structure.

## Decision

Stop tuning scalar wrapper knobs around `392a` and stop the source-transport branch.

The next principled step is a paradigm-shift ideation cycle for a stronger learned core
that models the conditional future level law directly while preserving the path geometry
that 392a already gets right. The new candidate must not be a retrieval variant, a
validation-future oracle, a per-cell calibration table, or another scalar source/noise
temperature.

The current deployable frontier remains:

- model: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- result: `results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json`
- score: `8/11`
