# 599a Factor Signal Audit for IV Failure Windows

## Question

598a shifted away from blind IV-only tuning and proposed a signal audit before training another joint-factor generator.

The audit question:

Does factor history add out-of-sample information about the hard 510a broad-frame IV failure windows beyond IV history alone?

## Implementation

- Script: `experiments/backfill/block_ar/audit_599a_factor_signal_for_iv_failures.py`
- Test: `test_code/test_599a_factor_signal_audit.py`
- Output: `results/autoresearch/599a_factor_signal_for_iv_failures/audit.json`
- Report: `results/autoresearch/599a_factor_signal_for_iv_failures/audit.md`

The audit regenerates 510a broad-frame samples, builds failure targets, summarizes IV history and factor history separately, and compares simple time-ordered out-of-sample ridge models:

- IV-only features;
- factor-only features;
- IV plus factor features.

No neural model is trained.

## Targets

Target summary from 441 broad-frame windows:

- bad-window rate `<50% coverage`: mean `23.6%`;
- window coverage: mean `69.7%`;
- persistent-undercoverage count: mean `4.01` cells per window;
- median-bias fraction: mean `71.1%`;
- level absolute error: mean `2.83 IV points`.

## Result

Focused tests passed: `3 passed in 1.31s`.

Out-of-sample scores:

| Target | IV R2 / AUC | Factor R2 / AUC | IV+Factor R2 / AUC | Read |
| --- | ---: | ---: | ---: | --- |
| bad window `<50%` | `0.532 / 0.765` | `0.187 / 0.339` | `0.382 / 0.609` | factors hurt |
| window coverage | `0.784 / n/a` | `0.852 / n/a` | `0.841 / n/a` | small factor lift on smooth aggregate only |
| persistent-under count | `0.868 / n/a` | `0.295 / n/a` | `0.761 / n/a` | factors hurt |
| median-bias fraction | `-1.817 / n/a` | `-3.188 / n/a` | `0.042 / n/a` | IV+factor helps this one weak target |
| level absolute error | `0.575 / n/a` | `-1.022 / n/a` | `-0.021 / n/a` | factors hurt |

## Mechanism Read

The available factor panel does not explain the hard risk failures better than IV history. For bad-window detection, persistent undercoverage, and level-error magnitude, adding factors worsens out-of-sample predictive performance. IV+factor helps median-bias fraction, but that is not enough to justify a new joint generator because it does not improve the core undercoverage targets.

This supports the old Exp 100 result: some factor relationships are concurrent or already encoded in IV history, not reliably predictive of future IV failure windows.

## Decision

Do not train another native joint-factor generator merely because factors are available. The signal audit says the current factor panel is not the missing information source for the main 510a broad-frame failures.

The learned-law route is now constrained:

- IV-only architecture/objective/noise routes are capped.
- Available joint factors do not add enough signal for the hard IV failures.
- Prior panel models already showed poor IV law quality.

The honest next move is to separate the learned base model from a disclosed risk-policy overlay. That can be risk-manager deployable, but should not be claimed as a calibrated learned conditional probability law.
