# 631a Generic Volatility-State Prefix Features

## Hypothesis

629a's main remaining failure was state-dependent scale: undercoverage, weak
turbulent widening, and a few excessive jump-tail cells. 631a kept the same
state-conditioned encoded-increment law but changed the causal memory features
from:

`[encoded level score, encoded increment score]`

to:

`[encoded level score, encoded increment score, |increment score|, increment score^2]`

This is a generic volatility-state feature, not an IV-specific branch.

## Result

631a improves the IV full-suite score from `3/11` to `4/11`.

Passed:

- conditionality
- block-AR
- cointegration
- cross-cell correlation

Failed:

- surface validity
- coverage
- time-series properties
- regime coverage
- distributional fidelity
- mean reversion
- pathwise jump realism

Key comparison:

| metric | 629a basic prefix | 631a scale prefix |
| --- | ---: | ---: |
| IV suite score | 3/11 | 4/11 |
| surface explosion rate | 45.1% | 51.7% |
| overall 90% coverage | 76.7% | 78.1% |
| conditionality MAE reduction | 7.9% | 8.9% |
| daily-change KS pass | 25/25 | 24/25 |
| level KS pass | 5/25 | 8/25 |
| median-bias pass | 13/25 | 16/25 |
| cross-cell corr/rank ratio | 0.967 / 1.167 | 0.939 / 1.156 |
| h1 mean-reversion ratio | 0.976 | 0.815 |
| pathwise max-jump KS | 0.464 | 0.531 |
| per-cell q99 tail pass | 19/25 | 19/25 |
| turb/calm width ratio | 0.894 | 1.008 |

The joint-panel audit remains strong but slightly below 629a:

| metric | 629a basic prefix | 631a scale prefix |
| --- | ---: | ---: |
| factor KS mean | 0.095 | 0.115 |
| factor KS pass <0.20 | 12/13 | 12/13 |
| q99 ratio median | 1.164 | 1.235 |
| q99 pass [0.5,2.0] | 13/13 | 13/13 |
| factor-factor corr | 0.823 | 0.865 |
| IV-factor corr | 0.883 | 0.889 |

Artifacts:

- `models/backfill/631a_joint38_statecond_increment_scale_e8_w2048_s631/best_model.pt`
- `results/autoresearch/631a_joint38_statecond_increment_scale_e8_w2048/full11.md`
- `results/autoresearch/631a_joint38_statecond_increment_scale_e8_w2048/joint_panel.md`

## Mechanism Read

The volatility-state features help conditionality and level occupancy enough to
gain one suite pass. They also improve the turbulent/calm width ratio from
`0.894` to `1.008`, so the direction is right.

They do not solve regime scaling. Width is still nearly regime-invariant rather
than materially wider in turbulent histories, and pathwise tails remain too
large in a few cells. The model's learned conditional flow can represent
state-dependent scale, but the standard unit Gaussian source does not make that
scale control easy enough in free-running rollout.

## Decision

631a becomes the active native joint learned base by IV score and overall
risk-manager plausibility, while 629a remains slightly better on some joint-panel
anchor metrics and pathwise KS.

The next clean move is not another data-frame reset. Add a generic conditional
source-scale head to the state-conditioned increment model, initialized at unit
scale and bounded conservatively. This is a standard conditional-flow device for
heteroscedasticity, not an IV-specific policy rule. It directly targets the
remaining weak turbulent widening and conditional dispersion failures.
