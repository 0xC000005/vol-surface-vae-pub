# Autoresearch 466a: Calibration Branch Postmortem

## Context

The recent deployable calibration branch started from the strongest learned base
law, `392a` (`8/11`), and tested whether a separated risk calibration layer could
fix the remaining coverage, regime, and level-distribution failures without
damaging conditional structure.

## Evidence

| Run | Mechanism | Score | Main result |
| --- | --- | ---: | --- |
| `392a` | learned rollout-energy transition FM | `8/11` | best deployable frontier; fails coverage, regime, distributional fidelity |
| `456a` | marginal level quantile map, alpha 0.25 | `7/11` | weakens conditionality; level KS only `11/25` |
| `457a` | weaker marginal level quantile map | `6/11` | weaker alpha does not recover conditionality |
| `458a` | larger marginal calibration panel | `7/11` | same failure as 456a |
| `462a` | center-preserving residual scale | `6/11` | improves average coverage, loses conditionality/time-series |
| `463a` | regime-aware residual scale | `7/11` | best calibration variant; conditionality `4.86%`, level KS `12/25`, regime layer2 `1/8` |
| `464a` | wider-bound regime residual scale | `6/11` | level KS `13/25` but time-series fails |
| `465a` | residual magnitude quantile map | `6/11` | lowers overcoverage but creates undercoverage and cointegration damage |

## Mechanism Read

The branch is capped for a clean reason:

1. `392a` already learned useful conditional geometry.
2. Calibration tables can move width and some marginal level metrics.
3. But the remaining failures are coupled: fixing one cell/horizon/regime by
   marginal residual transformation tends to damage another gate, conditionality,
   time-series shape, or cointegration.
4. The calibration data has one realized future per history, so per-regime,
   per-horizon, per-cell empirical maps are noisy and behave like marginal
   corrections rather than learned conditional laws.

This is why the best calibration variants do not exceed the uncalibrated `392a`
frontier.

## Decision

Stop calibration-table variants as the main path. They are deployable, but not
publishably elegant enough to justify below-frontier scores.

The next principled path should be a learned-law/objective change around the
`392a` family, not another posthoc map. The likely next falsifier is a learned
conditional distribution critic:

- keep the `392a` generator architecture and sampler;
- train a critic on `(history, future path)` pairs to estimate joint-law
  discrepancy between generated and realized paths;
- fine-tune the generator with a small adversarial/critic loss anchored by the
  original FM loss;
- use no regime labels, validation oracle, retrieval, or calibration map.

This is more general than fixed MMD/CRPS because the critic can learn which
features distinguish generated from historical paths, including level shape,
regime coverage patterns, and path distributional defects. It is also cleaner
than hand-written gates because the inductive bias is just conditional two-sample
matching.

## Next Falsifier

Implement a small `467a` conditional critic fine-tune:

- initialize from `392a`;
- sample short free-running paths during training;
- train a compact critic to separate real `(H, Y)` from generated `(H, Y_hat)`;
- update the generator with FM anchor plus a small generator adversarial loss;
- evaluate with the unchanged 11-suite.

If this weakens conditionality or structural suites, abandon learned critics. If
it improves level/regime while preserving the `392a` passes, continue.
