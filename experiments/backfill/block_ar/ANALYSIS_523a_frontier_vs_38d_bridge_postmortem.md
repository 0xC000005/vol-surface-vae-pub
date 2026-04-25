# 523a Frontier vs 38-d Bridge Postmortem

## Context
522a falsified the idea that an existing joint IV+factor daily-change baseline is already close to the official full 11-suite frontier. This postmortem compares the active `8/11` frontier (`392a`, `510a`) against the official-aligned 38-d CSDI bridge (`522a`) to identify the mechanism rather than just the score gap.

## Metric Comparison

| metric | 392a | 510a | 522a 38-d CSDI |
|---|---:|---:|---:|
| score | `8/11` | `8/11` | `4/11` |
| failed suites | coverage, regime, fidelity | coverage, regime, fidelity | coverage, conditionality, cointegration, regime, fidelity, mean-reversion, pathwise |
| cov90 overall | `0.868` | `0.873` | `0.707` |
| h30 cov90 | `0.885` | `0.891` | `0.508` |
| conditional MAE reduction | `5.14%` | `5.12%` | `0.77%` |
| turb/calm width ratio | `1.057` | `1.084` | `1.009` |
| cointegration gen/GT ratio | `0.700` | `0.595` | `0.205` |
| cointegration worst-cell ratio | `0.278` | `0.257` | `0.079` |
| regime layer2 | `0/8` | `0/8` | `1/8` |
| daily-change KS | `25/25` | `25/25` | `24/25` |
| level KS | `10/25` | `10/25` | `0/25` |
| cross-cell corr ratio | `0.963` | `0.968` | `1.101` |
| rank ratio | `1.495` | `1.462` | `1.035` |
| mean-reversion ratio | `1.024` | `0.986` | `0.428` |
| mean-reversion active pass | `0.833` | `0.833` | `0.042` |
| path max-jump KS | `0.373` | `0.361` | `0.678` |
| path q99 ratio | `1.031` | `1.017` | `0.963` |
| floor / ceiling rate | `0.000 / 0.000` | `0.000 / 0.000` | `0.0348 / 0.0006` |

## Mechanism Read
The 522a result separates two notions that were getting conflated:

- Daily-change realism is not enough. 522a has daily-change KS `24/25`, cross-cell corr ratio `1.101`, and rank ratio `1.035`, yet it fails level KS `0/25`, h30 coverage, mean reversion, cointegration, and pathwise max-jump KS.
- The active frontier is not winning because it has broader market factors. 392a/510a are IV-only but pass conditionality, cointegration, mean-reversion, and pathwise realism because the generative law lives closer to the IV level path being evaluated.
- The critical inductive bias in 392a/510a is the empirical normal-score surface-level transition with causal memory. This is not a narrow human-engineered finance rule; it is a support and coordinate choice that keeps the learned law in the evaluation space and makes long-horizon level occupancy visible during training/sampling.
- The 38-d CSDI bridge models standardized daily changes and reconstructs levels by cumulative sum plus clipping. Small median and scale errors accumulate. The model can match local move magnitudes while putting the entire generated level distribution in the wrong region.

## Decision
Do not switch the main program to generic 38-d daily-change diffusion. Also do not abandon the clean learned-law direction because 522a failed: the failure is specifically about modeling the wrong object for this suite.

The most principled next route is a new single-stage learned core that keeps the model in future IV-level space, uses a proper distributional objective, and optionally conditions on broader factors without making daily changes the primary generated object. This preserves the bitter-lesson preference for a learned law while avoiding the 522a error of making the target coordinate misaligned with the risk-manager scenario requirements.

The immediate next iteration should be an executable design/prototype decision: either adapt the 340c empirical normal-score surface-level path law to include 38-d conditioning features, or build a minimal one-shot future-level density model whose acceptance gate is first to recover the 392a structural passes before trying to repair the remaining three failures.
