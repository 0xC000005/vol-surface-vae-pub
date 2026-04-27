# 650a Mixed-Coordinate Path Flow With Horizon-Shared Source Affine

## Hypothesis

649a showed that a free per-horizon/per-channel source affine improves
conditionality but breaks shared IV geometry. 650a keeps the conditional-source
idea but constrains the source location and scale to be horizon-shared across
all channels. The intended effect is to let regime width vary without allowing
each cell to independently distort cross-cell correlation.

## Implementation

- Added `source_affine_mode` to the 647 path-flow model.
- Modes:
  - `full`: free per-horizon/per-channel source location and scale, matching
    649a.
  - `horizon_scalar`: one source location and scale per future horizon, shared
    by all channels.
  - `global_scalar`: one source location and scale for the whole path.
- 650a used `horizon_scalar`.
- No separate IV/factor treatment, no retrieval, no low-rank decoder, no
  post-hoc calibration.

## Results

Artifacts:

- model:
  `models/backfill/650a_joint38_mixed_path_horizon_source_e8_w2048_s650/best_model.pt`
- IV suite:
  `results/autoresearch/650a_joint38_mixed_path_horizon_source_e8_w2048_s650/full11.json`
- joint audit:
  `results/autoresearch/650a_joint38_mixed_path_horizon_source_e8_w2048_s650/joint_panel.json`

Focused tests passed:

```bash
pytest test_code/test_647a_mixed_coordinate_path_flow.py -q
```

IV 11-suite:

- score: 3/11
- coverage improved versus 647a and 649a: overall 90% CI 69.3%
- conditionality did not pass: MAE reduction 2.8%, per-cell worst -16.6%
- regime width signal existed but remained weak: turbulent/calm width ratios
  were 1.104/1.093/1.055/1.025 across h1/h7/h14/h30
- cross-cell correlation failed worse than 649a: ratio 0.403
- mean reversion failed: aggregate ratio 0.484
- daily-change KS pass stayed weak at 5/25
- pathwise max-jump KS 0.595 failed

Joint-panel audit:

- factor delta KS mean: 0.105
- factor KS pass: 13/13
- factor q99 pass: 11/13
- factor-factor correlation: 0.821
- IV-factor correlation: 0.791

## Mechanism Read

Horizon-sharing was not enough to preserve IV geometry. It improved coverage
relative to the free source affine but still fragmented the IV cross-cell
correlation structure. The likely issue is deeper than per-cell source freedom:
conditioning the source outside the flow lets the model explain conditional
variance through independent source amplitude instead of through a coherent
joint path transport.

## Decision

Close the conditional-source branch as informative but not baseline-improving.
The best clean native joint path baseline remains 647a, not 648a/649a/650a.
The next step should be post-experiment ideation, not another source-affine
variant.
