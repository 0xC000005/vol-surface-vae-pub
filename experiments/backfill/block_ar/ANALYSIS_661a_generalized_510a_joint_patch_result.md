# 661a Generalized-510a Joint AR Patch-Energy Result

## Question

661a tested whether the empirically strong 510a-style AR transition can be generalized cleanly to a single native `joint38` model: one checkpoint, one shared causal memory, one shared stochastic source, one transition flow, and no post-hoc IV/factor deck gluing.

The implementation used the generic empirical-score AR transition over the full 38-channel encoded state panel and added a small patch-energy free-rollout objective. This deliberately kept the architecture simple and reused the old 510a trunk rather than introducing separate IV and anchor-factor mechanisms.

## Main Result

| split | IV score | cov90 | cond MAE red. | turb/calm | daily KS | level KS | median bias | path max-jump KS | corr ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-tail | 6/11 | 0.786 | -1.347% | 1.152 | 24/25 | 22/25 | 25/25 | 0.171 | 0.806 |
| validation | 4/11 | 0.563 | 4.968% | 1.032 | 23/25 | 1/25 | 4/25 | 0.559 | 0.953 |

Train-tail failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage`.
Validation failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`.

## Joint-Panel Read

| split | factor KS mean | factor KS pass | factor q99 median | factor q99 pass | factor corr shape | factor abs corr GT/gen | IV-factor shape | IV-factor abs corr GT/gen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-tail | 0.118 | 12/13 | 1.720 | 10/13 | 0.761 | 0.200/0.064 | 0.801 | 0.168/0.093 |
| validation | 0.142 | 10/13 | 2.010 | 6/13 | 0.584 | 0.225/0.066 | 0.689 | 0.149/0.088 |

661a is acceptable as a mechanism check but not as the active deployable joint model. In train-tail it preserves many IV mechanics, but the joint-panel validation audit is weaker than the native mixed-coordinate baselines: factor q99 pass falls to 6/13, factor-factor correlation shape falls to 0.584, and generated absolute correlation magnitudes are strongly attenuated.

## Baseline Comparison

| model | IV score | cov90 | cond MAE red. | level KS | path KS | factor KS pass | factor q99 pass | factor corr shape | IV-factor shape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 641a mixed-coordinate AR | 4/11 | 0.647 | 4.509% | 4/25 | 0.610 | 12/13 | 13/13 | 0.866 | 0.852 |
| 647a mixed-coordinate path | 4/11 | 0.663 | -0.680% | 10/25 | 0.618 | 13/13 | 11/13 | 0.829 | 0.845 |
| 661a generalized-510a joint AR | 4/11 | 0.563 | 4.968% | 1/25 | 0.559 | 10/13 | 6/13 | 0.584 | 0.689 |

## Diagnosis

The 661a failure is not a shared-source failure. It is a coordinate failure. The empirical-score state decoder is fitted on the training support, so validation anchor factors that move beyond that support are clipped or pulled back toward the training range. The validation joint audit shows this directly: SPX and Nikkei generated maxima remain near the training-era range while the validation realized levels move higher.

This makes the absolute-level empirical-score coordinate a poor generic state variable for random-walk-like traded factors. It can preserve in-sample IV mechanics, but it is not robust for a general multivariate market panel where levels drift across regimes.

## Decision

Do not promote 661a as the active deployable model. Keep it as a negative-but-useful falsifier for the idea that the 510a empirical-score state transition can be directly lifted to a 38-channel level panel.

The best-supported native joint path remains the mixed-coordinate family: shared source and transition, but generated coordinates selected by data semantics. IV-like bounded mean-reverting surfaces need level-score style support control; anchor factors need movement-from-current coordinates so they can extrapolate with the observed market level.

The next implementation should not add another post-hoc calibration or separate factor deck. It should either return to the mixed-coordinate state-conditioned AR family or make the 510a trunk operate on movement coordinates with explicit state conditioning, not absolute encoded levels.
