# 632a Conditional Source-Scale Falsifier

## Hypothesis

631a still had nearly regime-invariant dispersion. 632a tested whether a standard
conditional-flow source-scale head could learn heteroscedasticity while keeping
the same state-conditioned encoded-increment architecture.

The head was initialized at unit scale and bounded to `[0.5, 2.0]`.

## Result

The IV full-suite score fell from `4/11` to `3/11`.

Passed:

- conditionality
- block-AR
- cross-cell correlation

Failed:

- surface validity
- coverage
- time-series properties
- cointegration
- regime coverage
- distributional fidelity
- mean reversion
- pathwise jump realism

Key comparison:

| metric | 631a scale-prefix | 632a + source scale |
| --- | ---: | ---: |
| IV suite score | 4/11 | 3/11 |
| surface explosion rate | 51.7% | 60.8% |
| overall 90% coverage | 78.1% | 80.8% |
| conditionality MAE reduction | 8.9% | 9.6% |
| turb/calm width ratio | 1.008 | 0.865 |
| daily-change KS pass | 24/25 | 25/25 |
| level KS pass | 8/25 | 6/25 |
| median-bias pass | 16/25 | 24/25 |
| cross-cell corr/rank ratio | 0.939 / 1.156 | 0.762 / 1.488 |
| pathwise max-jump KS | 0.531 | 0.607 |
| per-cell q99 tail pass | 19/25 | 19/25 |

The joint-panel audit also worsened:

| metric | 631a scale-prefix | 632a + source scale |
| --- | ---: | ---: |
| factor KS mean | 0.115 | 0.123 |
| factor KS pass <0.20 | 12/13 | 11/13 |
| q99 ratio median | 1.235 | 1.139 |
| q99 pass [0.5,2.0] | 13/13 | 13/13 |
| factor-factor corr | 0.865 | 0.827 |
| factor-factor gen/gt mean abs | 0.133 / 0.225 | 0.100 / 0.225 |
| IV-factor corr | 0.889 | 0.868 |
| IV-factor gen/gt mean abs | 0.119 / 0.149 | 0.081 / 0.149 |

The source scale collapsed:

- epoch 1 mean scale: `0.540`
- epoch 2 mean scale: `0.500`, std `0.000`
- best epoch 6 mean scale: `0.500`, std `0.0002`

## Mechanism Read

The conditional source-scale head repeats the earlier source-scale pathology in a
cleaner architecture. The flow-matching objective can reduce training loss by
shrinking the source distribution to the lower bound rather than learning useful
state-dependent widening. That improves some one-step/marginal metrics and
median-bias metrics, but it damages free-running dependence, regime width, and
pathwise realism.

The failure is not "heteroscedasticity is wrong." The failure is this
source-scale parameterization under this one-step flow objective.

## Decision

Reject conditional source scale for this branch. Keep 631a as the active native
joint learned base.

The next step should be analysis/ideation rather than another local knob. The
remaining problem likely needs a proper multi-step distributional objective or
rollout-level scoring rule that rewards calibrated scenario sets directly,
instead of a one-step source-scale shortcut.
