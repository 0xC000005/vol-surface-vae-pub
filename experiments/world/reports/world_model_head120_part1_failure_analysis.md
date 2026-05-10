# World Model HEAD120: Part 1 Quality Gate Failure Analysis

Date: 2026-05-10

## Objective Family

`post_experiment_analysis` for the HEAD119 Part 1 quality-gate failure.

## Hypothesis

The quality gate failed because the current representation is not yet
proven as a market-state representation, not because the masked-view
embedding objective simply collapsed.

## Falsifier

The diagnosis would be wrong if the saved artifacts showed collapsed
representation health, no utility versus a mean baseline, or no
incremental value when combined with raw surface features.

## Summary

- Promotion decision: `DO_NOT_PROMOTE`.
- Quality gate passed: `False`.
- Representation health did not fail; the main blockers are baseline
  superiority, market-state probes, and scale/stability.

## Regression Failure Anatomy

| target | mean MSE | Barlow MSE | raw last MSE | raw flat MSE | raw+Barlow MSE | best feature | Barlow vs mean | Barlow vs best | raw+Barlow vs raw last |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| future_mean_delta | 0.013939 | 0.011635 | 0.006484 | 0.009746 | 0.006584 | `raw_surface_last` | -16.53% | 79.45% | 1.55% |
| future_range | 0.063851 | 0.047185 | 0.054625 | 0.050972 | 0.047705 | `barlow_clean_last` | -26.10% | 0.00% | -12.67% |
| future_terminal_delta | 0.032616 | 0.031830 | 0.019208 | 0.027846 | 0.021879 | `raw_surface_last` | -2.41% | 65.71% | 13.90% |
| future_max_abs_step | 0.048165 | 0.039526 | 0.041928 | 0.035980 | 0.040023 | `raw_surface_flat` | -17.94% | 9.86% | -4.54% |
| future_drawdown | 0.058070 | 0.042269 | 0.050058 | 0.044180 | 0.042762 | `barlow_clean_last` | -27.21% | 0.00% | -14.58% |

Interpretation:

- Barlow beats the mean baseline on `5/5` targets.
- Barlow is the best standalone feature on `2/5` targets.
- Adding Barlow to raw last-surface features improves `3/5` targets.
- The representation is not useless, but it is not yet superior to
  raw/simple baselines across the claimed target family.

## Regime Probe Anatomy

| feature | accuracy | majority | lift | macro recall |
| --- | ---: | ---: | ---: | ---: |
| barlow_clean_last | 0.109375 | 0.597656 | -0.488281 | 0.386260 |
| barlow_clean_mean | 0.050781 | 0.597656 | -0.546875 | 0.313900 |
| raw_surface_last | 0.554688 | 0.597656 | -0.042969 | 0.352704 |
| raw_surface_flat | 0.542969 | 0.597656 | -0.054688 | 0.330957 |
| raw_surface_last_plus_barlow_clean_last | 0.332031 | 0.597656 | -0.265625 | 0.331762 |

The regime layer is not solved by any current feature set. This is a
real gate failure, but it is also a probe-design warning: the current
regime label/probe cannot certify market-state quality.

## Diagnosis

- The gate did not fail because of representation collapse; representation health passed.
- The strongest empirical failure is baseline superiority: Barlow is useful versus a mean-target baseline but is not better than raw/simple features on enough downstream targets.
- Raw last-surface features dominate persistence-like mean and terminal targets; the Barlow representation appears more useful for path-shape and dispersion targets.
- Adding Barlow to raw last-surface features improves several path-shape targets, so the representation carries complementary signal, but not enough to certify a standalone market-state representation.
- The regime probe is not mature evidence: every feature set is below the majority baseline, so this layer currently indicates probe/label/baseline insufficiency as well as weak Barlow accuracy.
- The remaining failures are mostly missing evidence: factor-panel probes, IV-shape state probes, richer mask policies, seed sensitivity, and larger-scale training have not been run.

## Next Diagnostics

- Add frozen probes for present-state IV shape summaries and factor-panel summaries before changing the pretraining objective.
- Add PCA, persistence, and rolling-window statistics baselines for the same targets.
- Evaluate whether Barlow adds incremental value to raw features with consistent split-safe probes.
- Rerun mask robustness across seeds and richer held-out mask families.
- Only after those diagnostics, decide whether scale, architecture capacity, or objective changes are justified.
