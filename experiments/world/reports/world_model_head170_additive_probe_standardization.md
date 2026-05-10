# World Model HEAD170: Additive Probe Standardization

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_additive_signal_gate`; probe-hygiene diagnostic.

## Hypothesis

If the raw-plus-learned IV guardrail failure is mostly caused by feature
scale differences under one ridge penalty, then standardizing each
feature surface with train-split statistics should remove or materially
reduce the failure.

## Falsifier

The probe-scale artifact explanation is weak if standardization does not
fix the IV guardrail.

## Guardrail Comparison

| probe surface | raw-only IV MSE | raw+learned IV MSE | raw+learned/raw | status |
| --- | ---: | ---: | ---: | --- |
| unstandardized | 0.005630 | 0.005901 | 1.048173 | FAIL |
| standardized | 0.000262 | 0.000318 | 1.213267 | FAIL |

## Standardized Target Rows

| target | raw-only MSE | learned-only MSE | raw+learned MSE | raw+learned delta |
| --- | ---: | ---: | ---: | ---: |
| iv_surface | 0.000262 | 0.013528 | 0.000318 | 0.000056 |
| vol_side_channel | 1.353281 | 0.306299 | 0.262945 | -1.090336 |
| factor_level | 1.793016 | 0.587862 | 0.594086 | -1.198930 |
| factor_return | 1.125493 | 0.182881 | 0.166165 | -0.959327 |
| all_geometry | 0.821243 | 0.218278 | 0.206314 | -0.614929 |

## Decision

- Standardization fixes IV guardrail: `False`.
- Promotion decision: `DO_NOT_PROMOTE`.
- Part B blocked: `True`.

Feature standardization checks whether the raw-plus IV miss is mainly a ridge-probe scale artifact. This is evaluation hygiene, not a Part 1 objective change.
