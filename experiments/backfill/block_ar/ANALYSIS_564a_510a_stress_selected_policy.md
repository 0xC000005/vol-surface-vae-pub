# 564a 510a Stress-Selected Scenario Policy

## Context

563a closed the local TimePFN-style synthetic-prior branch. The best learned conditional law remains `510a`, but the user reframed the product as a risk-manager stress scenario generator rather than a calibrated conditional probability law.

564a therefore tested a separated risk-system policy:

1. keep `510a` as the base learned conditional law;
2. generate many authentic candidate paths from `510a`;
3. select a deterministic calm/central/stress-stratified scenario set by candidate severity;
4. report the result as conservative stress scenarios, not calibrated probabilities.

## Implementation

Added `evaluate_564a_stress_selected_510a.py` and focused selector tests.

Selection policy:

- sample `192` candidate paths per validation history;
- score each candidate by average future IV level;
- select `48` paths using low, central, and high severity bands;
- run the existing full 11-suite on the selected scenario set.

This is a policy overlay around the learned law, not a retrained generator.

## Verification

Focused tests:

- `pytest test_code/test_564a_stress_selection_policy.py -q`
- Result: `2 passed`.

Official-size evaluation:

- Artifact: `results/autoresearch/564a_510a_stress_selected_policy/full11.json`
- Base checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- Candidate paths per history: `192`
- Selected scenarios per history: `48`
- Score: `7/11`

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

- coverage90: `0.897`
- h1/h7/h14/h30 worst-cell coverage: `0.745`, `0.740`, `0.740`, `0.760`
- conditionality MAE reduction: `6.8%`
- daily-change KS: `25/25`
- level KS: `1/25`
- median-bias pass cells: `20/25`
- cointegration gen/GT ratio: `0.568`
- cointegration worst-cell ratio: `0.222`
- regime layer2: `1/8`
- persistent severe undercoverage: `0.6%`
- cross-cell correlation ratio: `0.878`
- effective-rank ratio: `1.604`
- mean-reversion aggregate ratio: `1.003`
- pathwise max-jump KS: `0.488`

## Risk-Manager Read

This is the strongest risk-system result so far under the revised product framing.

Compared with the raw `510a` prototype, 564a improves the properties a risk manager cares about most:

- lower per-cell coverage passes at all horizons;
- conditionality remains above gate;
- daily-change realism becomes perfect under the KS gate;
- time-series, mean-reversion, cross-cell dependence, and pathwise realism pass;
- persistent severe undercoverage remains low.

The remaining failures are interpretable under the stress-scenario framing:

- high-side overcoverage causes the formal coverage suite to fail, but that is acceptable for conservative stress sampling if disclosed;
- level KS is a probability-frequency warning, not a hard stress-scenario blocker;
- regime layer2 remains incomplete, but improves to `1/8` and no longer looks like persistent global stress collapse;
- cointegration is the main non-negotiable concern because the worst-cell ratio is slightly below gate (`0.222` vs `0.25`).

## Decision

Promote 564a as the current best risk-manager system prototype, not as a calibrated learned law.

Do not claim its selected scenario frequencies are probabilities. The system should be reported as:

- base learned law: `510a`;
- risk policy: severity-stratified stress scenario selection;
- intended use: conditional stress exploration and risk challenge;
- not intended use: calibrated probabilistic forecasting or capital model without separate validation.

The next clean step is one softer policy evaluation to test whether cointegration can be recovered while preserving lower stress inclusion: reduce candidate extremity by lowering candidate count or using a less extreme severity band. Avoid a broad policy-knob sweep.
