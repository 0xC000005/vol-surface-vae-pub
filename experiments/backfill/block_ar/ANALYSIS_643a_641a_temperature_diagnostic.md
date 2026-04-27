# 643a 641a Sample-Temperature Diagnostic

## Question

642a identified 641a as the clean native IV-plus-anchor-factor baseline but with weak IV coverage/location. 643a tests the smallest possible calibration hypothesis: is the support-valid mixed-coordinate model simply underdispersed at sampling time?

This is evaluation-only. It does not add architecture, retraining, IV-specific heads, or factor-specific treatment.

## Runs

Same checkpoint:

- `models/backfill/641a_joint38_mixedcoord_scale_e8_w2048_s641/best_model.pt`

Artifacts:

- baseline: `results/autoresearch/641a_joint38_mixedcoord_scale_e8_w2048_s641/full11.json`
- temp `1.10`: `results/autoresearch/643a_641a_temperature_diagnostic/temp110_full11.json`
- temp `1.25`: `results/autoresearch/643a_641a_temperature_diagnostic/temp125_full11.json`

## Result

| temperature | score | cov90 | cond MAE red. | turb/calm | daily KS | level KS | median pass | kurtosis ratio | q99 cells | cross corr | rank ratio | path KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 4/11 | 0.647 | 4.51% | 0.997 | 24/25 | 4/25 | 8/25 | 0.946 | 23/25 | 0.949 | 1.357 | 0.610 |
| 1.10 | 3/11 | 0.679 | 2.46% | 0.973 | 25/25 | 3/25 | 7/25 | 0.739 | 20/25 | 0.766 | 1.770 | 0.518 |
| 1.25 | 3/11 | 0.710 | -0.01% | 0.966 | 16/25 | 2/25 | 5/25 | 0.515 | 11/25 | 0.513 | 2.489 | 0.314 |

## Mechanism Read

Scalar temperature is not the missing deployability layer.

The positive signal is narrow:

- coverage increases from `64.7%` to `67.9%` and `71.0%`;
- pathwise max-jump KS improves from `0.610` to `0.518` and `0.314`.

The cost is broad and structural:

- score drops from `4/11` to `3/11`;
- conditional MAE reduction collapses from `4.51%` to `2.46%` and then `~0%`;
- turbulent/calm width remains flat or worse;
- global kurtosis falls from pass `0.946` to fail `0.739` and `0.515`;
- per-cell q99 tail-scale pass falls from `23/25` to `20/25` and `11/25`;
- cross-cell correlation/rank drift toward noisy, less coherent panels;
- level KS and median-bias both worsen.

Temperature turns undercoverage into over-noisy, poorly placed paths. That confirms the issue is not global spread. The missing piece is state/cell/horizon-specific conditional probability allocation.

## Decision

Close scalar sample temperature for 641a. Do not promote it as a risk-policy calibration.

The next step should not be another sampling knob. Since the generated coordinate is now clean and native-jointness is solved, the next research move must target the objective/calibration layer of the unified path law: improve conditional location/spread allocation without separating IV and anchor factors.
