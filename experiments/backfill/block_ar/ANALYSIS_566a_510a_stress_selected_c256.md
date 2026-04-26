# 566a Stronger 510a Stress-Selected Policy

## Context

564a produced the current best risk-manager prototype: `510a` as the learned base law plus deterministic stress-scenario selection from `192` candidate paths per history. 565a tested the softer bracket (`96` candidates) and did not improve the remaining dependence/regime concerns.

566a tested the opposite bracket: increase candidate support to `256` candidates per validation history while keeping the same selected sample count (`48`) and the same severity-stratified policy.

## Result

Artifact:

- `results/autoresearch/566a_510a_stress_selected_policy_c256/full11.json`

Configuration:

- base checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- candidate paths per history: `256`
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

- coverage90: `0.900`
- h1/h7/h14/h30 worst-cell coverage: `0.724`, `0.740`, `0.740`, `0.760`
- conditionality MAE reduction: `5.6%`
- daily-change KS: `25/25`
- level KS: `1/25`
- median-bias pass cells: `20/25`
- cointegration gen/GT ratio: `0.627`
- cointegration worst-cell ratio: `0.222`
- regime layer2: `0/8`
- persistent severe undercoverage: `0.6%`
- cross-cell correlation ratio: `0.879`
- effective-rank ratio: `1.610`
- mean-reversion aggregate ratio: `0.986`
- pathwise max-jump KS: `0.492`

## Mechanism Read

The stronger candidate pool improves aggregate cointegration relative to 564a (`0.627` vs `0.568`) but does not move the decisive worst-cell ratio (`0.222`, still below the `0.25` gate). It also loses the small regime layer2 improvement seen in 564a (`0/8` vs `1/8`).

The risk-critical positive properties remain stable:

- surface validity remains clean;
- lower stress inclusion remains acceptable;
- conditionality remains above gate;
- daily-change realism, cross-cell correlation, mean reversion, and pathwise max-jump realism pass.

But the candidate-count axis is now exhausted as a principled single-knob explanation. The same failure set persists at `96`, `192`, and `256` candidates.

## Decision

Do not promote 566a over 564a. Keep 564a as the current best risk-manager system prototype because it has the best balance of dependence, regime, and path realism among the count checks.

Stop candidate-count tuning. The next principled step is to package 564a as a risk-manager stress scenario generator with explicit base-law/policy separation and caveats, rather than continuing a policy-knob sweep.
