# 565a Softer 510a Stress-Selected Policy

## Context

564a produced the strongest risk-manager system prototype so far: `510a` base law plus severity-stratified selection from `192` candidate paths per history. It scored `7/11` and passed the risk-critical lower per-cell coverage, conditionality, time-series, mean-reversion, cross-cell, and pathwise suites.

Its main non-negotiable concern was cointegration worst-cell ratio `0.222`, slightly below the `0.25` gate. 565a tested one softer policy setting by reducing candidate count from `192` to `96`, keeping the same selected sample count (`48`) and selection rule.

## Result

Artifact:

- `results/autoresearch/565a_510a_stress_selected_policy_c96/full11.json`

Configuration:

- base checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- candidate paths per history: `96`
- selected scenarios per history: `48`

Score: `7/11`

Passed:

- `surface`
- `conditionality`
- `time_series`
- `block_ar`
- `cross_cell_correlation`
- `mean_reversion`
- `pathwise_jump_realism`

Failed:

- `coverage`
- `cointegration`
- `regime_coverage`
- `distributional_fidelity`

## Key Metrics

- coverage90: `0.898`
- h1/h7/h14/h30 worst-cell coverage: `0.714`, `0.755`, `0.781`, `0.740`
- conditionality MAE reduction: `6.6%`
- daily-change KS: `25/25`
- level KS: `1/25`
- median-bias pass cells: `20/25`
- cointegration gen/GT ratio: `0.574`
- cointegration worst-cell ratio: `0.193`
- regime layer2: `0/8`
- persistent severe undercoverage: `0.5%`
- cross-cell correlation ratio: `0.866`
- effective-rank ratio: `1.637`
- mean-reversion aggregate ratio: `0.980`
- pathwise max-jump KS: `0.482`

## Mechanism Read

The softer policy does not solve the non-negotiable concern. It preserves the good risk-system properties from 564a but worsens the two metrics that motivated the run:

- cointegration worst-cell ratio moves from `0.222` to `0.193`;
- regime layer2 moves from `1/8` to `0/8`.

The only notable improvement is a slightly better pathwise max-jump KS (`0.482` vs `0.488`), which is not enough to offset worse dependence/regime behavior.

## Decision

Do not promote 565a over 564a. Keep 564a as the current best risk-manager system prototype.

If continuing the policy branch, the only justified next single run is the opposite bracket: increase candidate count to `256` to test whether stronger candidate support improves the near-miss cointegration and regime cells. If that does not improve the non-negotiable dependence/regime concerns, stop policy-count tuning and package 564a with caveats.
