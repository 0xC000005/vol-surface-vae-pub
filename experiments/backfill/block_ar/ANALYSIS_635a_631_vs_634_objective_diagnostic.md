# 635a 631a vs 634a Objective Diagnostic

## Context

634a tested the cleanest objective-level fix after source-scale collapse: keep the 631a native joint architecture and add a low-weight global rollout proper score over generated level and increment score paths. It preserved joint mechanics but reduced the IV full-suite score from 4/11 to 3/11.

## Hypothesis

The failure is not that rollout-level training is intrinsically wrong. The failure is likely the specific shape of the global full-path objective: flattening all horizons and channels into one high-dimensional energy/SW score is too blunt for conditional risk scenarios. It does not explicitly protect per-history conditional width, cellwise level occupancy, or marginal calibration.

## Direct Comparison

| metric | 631a active base | 634a global rollout score | read |
| --- | ---: | ---: | --- |
| IV score | 4/11 | 3/11 | worse |
| surface explosion | 51.7% | 54.7% | worse |
| 90% coverage | 78.1% | 71.3% | materially worse |
| conditionality MAE reduction | 8.9% | 4.4% | lost gate |
| turbulent/calm width | 1.008 | 1.014 | no real change |
| daily-change KS pass | 24/25 | 24/25 | unchanged |
| level KS pass | 8/25 | 4/25 | worse |
| median-bias pass | 16/25 | 7/25 | materially worse |
| bad window coverage rate | 2.7% | 17.2% | materially worse |
| cross-cell corr ratio | 0.939 | 0.928 | similar |
| mean-reversion ratio | 0.815 | 0.805 | similar |
| pathwise max-jump KS | 0.531 | 0.549 | worse |
| q90 jump ratio | 2.519 | 2.625 | worse |
| q99 jump ratio | 6.007 | 5.972 | similar |

Joint-panel comparison:

| metric | 631a | 634a | read |
| --- | ---: | ---: | --- |
| factor delta KS mean | 0.115 | 0.103 | slightly better |
| factor KS pass | 12/13 | 12/13 | unchanged |
| factor q99 median ratio | 1.235 | 1.213 | slightly better |
| factor q99 pass | 13/13 | 13/13 | unchanged |
| factor-factor corr shape | 0.865 | 0.867 | unchanged |
| factor-factor generated mean abs corr | 0.133 | 0.138 | slightly better |
| IV-factor corr shape | 0.889 | 0.880 | similar |
| IV-factor generated mean abs corr | 0.119 | 0.130 | better |

## Mechanism Read

634a did not break the general native joint model. In fact, anchor-factor and IV-factor metrics were slightly better. The failure is specifically IV conditional calibration:

- the scenario cloud became less useful per history, with lower coverage and worse bad-window rate;
- level occupancy and median bias worsened sharply;
- turbulent/calm width remained nearly flat, so the loss did not learn regime-responsive dispersion;
- pathwise jump realism stayed too wide and slightly worsened by KS/q90.

This supports a narrower mechanism: a single global full-path energy/SW score can preserve broad joint geometry while failing the conditional marginal properties a risk manager sees in each history/cell/horizon. With one realized future path per history, the global score can over-reward batch-level shape and realized-path proximity instead of calibrated conditional scenario sets.

## Decision

Keep 631a as the active base and reject 634a.

The next experiment should still use the differentiable rollout machinery from 634a, but the proper scoring objective must be more local and conditional:

- use a per-history, per-horizon, per-channel marginal CRPS/energy score over generated level and increment scores;
- add only a small short-patch dependence anchor so cross-channel/path dependence does not collapse;
- train all 38 channels with the same objective and no IV/factor branch.

This is not a return to ad hoc post-hoc calibration. It is a standard probabilistic forecasting principle: first calibrate the conditional marginals the risk manager consumes, then anchor dependence with a small multivariate term.
