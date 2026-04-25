# 512a Midpoint Checkpoint Result

## Context

511a selected one non-invasive checkpoint-trajectory audit: parameter-space
midpoint between the 392a frontier checkpoint and the 510a patch-energy final
checkpoint. This tested whether the path between the two `8/11` endpoints
contains a better full-suite tradeoff.

## Result

- Checkpoint: `models/backfill/512a_midpoint_392a_510a_s42/model.pt`
- Full suite: `results/block_ar/512a_midpoint_392a_510a_s42/full11.json`
- Score: `6/11`
- Failed: `coverage`, `conditionality`, `cointegration`,
  `regime_coverage`, `distributional_fidelity`

Key metrics:

- Coverage90: `0.8691`
- Per-cell coverage range: `0.703` to `0.984`
- Under-70 cells: `0`
- Over-95 cells: `9`
- Conditionality MAE reduction: `4.39%`
- Daily-change KS: `25/25`
- Level KS: `11/25`
- Median-bias fraction: `20/25`
- Bias magnitude: `25/25`
- Regime layer2: `0/8`
- Cointegration worst-cell ratio: `0.193`
- Cross-cell corr ratio: `0.960`
- Mean-reversion ratio: `0.993`
- Path max-jump KS: `0.377`

## Mechanism Read

The midpoint improves some coverage geometry and nudges level KS from `10/25` to
`11/25`, but it loses conditionality and worst-cell cointegration. This confirms
that the endpoint tradeoff is not hiding a simple parameter-space bridge to
`9/11+`.

## Decision

Close checkpoint interpolation. Keep the deployable frontier as the `8/11`
392a/510a tie, with 392a safer structurally and 510a useful as a risk-coverage
variant. Do not run interpolation-alpha sweeps.
