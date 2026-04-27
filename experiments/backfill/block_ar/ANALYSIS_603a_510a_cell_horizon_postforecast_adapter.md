# 603a 510a Cell/Horizon Postforecast Adapter

## Hypothesis

602a showed that state-only horizon scaling improves aggregate coverage but cannot target
the sparse bad cell/regime slices. 603a tests the next and likely final calibration-layer
falsifier: a bounded per-cell/per-horizon interval-scale table around the frozen 510a
generator.

The question is not whether this is an elegant base model. It is a deployability-layer
test: if targeted widening still cannot clear lower-only regime/cell inclusion without
damaging authenticity, then postforecast calibration is not enough.

## Method

603a reused `evaluate_405a_interval_scale_calibrated_system.py` on frozen 510a:

- base checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`;
- calibration block: 441 pre-validation windows, 48 samples per window;
- validation block: 441 broad-frame windows, 48 samples per window;
- calibration table: per-cell/per-horizon residual scale, no regime bins;
- policy: widening-only residual scaling around the sample median;
- target coverage: `0.95`;
- high-side overcoverage cap: disabled by setting `coverage_hi=1.0`;
- bounded candidate scale range: `[1.0, 2.0]`;
- fitted scale range: `1.075 / 1.475 / 1.900`.

## Result

603a scored `4/11`, the same full-suite count as 602a and below broad 510a's `5/11`.

Key metrics:

- overall 90% coverage: `84.5%`;
- h1/h7/h14/h30 coverage: `89.7% / 82.5% / 84.7% / 82.4%`;
- conditional MAE reduction: `7.09%`;
- pathwise max-jump KS: `0.202`;
- regime layer2: `0/8`;
- persistent severe undercoverage: `7.3%`;
- daily-change KS cells: `13/25`;
- level KS cells: `1/25`;
- pathwise per-cell extreme-jump scale cells: `14/25`;
- worst cell remained `(3,3)` at h7/h14/h30.

Risk-manager readiness comparison:

| Candidate | Full suite | Stress score | Stress pass | Worst lower-only cell | Worst regime cell |
| --- | ---: | ---: | --- | ---: | ---: |
| `510a_base_broad` | `5/11` | `1/4` | `False` | `0.478` | `0.213` |
| `602a_state_adapter` | `4/11` | `1/4` | `False` | `0.546` | `0.303` |
| `603a_cell_horizon_adapter` | `4/11` | `1/4` | `False` | `0.556` | `0.360` |

Artifacts:

- `results/autoresearch/603a_510a_cell_horizon_postforecast_adapter/full11.json`
- `results/autoresearch/603a_510a_cell_horizon_postforecast_adapter/full11.md`
- `results/autoresearch/603a_510a_cell_horizon_postforecast_adapter/risk_readiness.json`
- `results/autoresearch/603a_510a_cell_horizon_postforecast_adapter/risk_readiness.md`

## Mechanism Read

The calibration table can widen intervals enough to pass horizon-level coverage and reduce
bad-window floor failures, but it cannot solve the remaining sparse stress slices. The
hard failure is not just "cell X needs more width"; it is tied to regime, window, horizon,
and path direction. Scaling residuals around the sample median does not move probability
mass into the missing realized stress paths. It also distorts daily-change and extreme
jump distributions.

## Decision

Close the bounded postforecast widening route as a deployability solution. State-score
widening and cell/horizon widening both improve aggregate coverage but fail risk-manager
readiness and damage authenticity. The next step should not be another scale-table variant.
The remaining options are:

- design a fundamentally different sparse-stress allocation mechanism that preserves path
  authenticity;
- or document that the current observable conditioning set is insufficient and define the
  required additional state/data for deployment.
