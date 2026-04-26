# 559a Post Hard-State Synthesis

## Context

558a falsified the cleanest hard-state replay experiment:

- it improved level KS to `11/25`;
- it preserved conditionality and pathwise realism;
- it broke cointegration worst-cell ratio to `0.139`;
- it remained below the `8/11` frontier and fell to risk-readiness `1/4`.

The immediate fallback from 557a was a small routed source-prior expert. Before implementing that, the existing log was checked for prior source/sampling-law experiments.

## Prior Source-Prior Evidence

The routed source-prior idea is not completely fresh in this repo:

- 448a-450a conditional noise-scale heads either collapsed noise or stayed near identity, scoring `4/11` to `7/11`.
- 507a shared-source direct-path flow scored `3/11`, preserving some correlation structure but losing conditional level law.
- 514a AR(1) temporal source prior scored `4/11` best and `5/11` final, badly over-broadening paths and damaging level/daily-change/pathwise geometry.

This does not prove that every possible routed source model is impossible, but it makes a small source-prior expert a low-priority next move. It would add deployed complexity in the same part of the model that has repeatedly damaged authenticity.

## Current Frontier

The honest frontier is still:

- `510a`: `8/11`, risk-readiness `3/4`, best current risk prototype.
- `555a`: `8/11`, risk-readiness `3/4`, useful factor-conditioning evidence but not better than 510a.
- `558a`: `7/11`, risk-readiness `1/4`, closed.

No current candidate is fully risk-manager acceptable because the lower-only regime gate still fails.

## Mechanism Read

The recurring failure is no longer a mystery:

1. The learned generator can produce locally realistic IV paths.
2. It can pass conditionality, dependence, mean reversion, and pathwise realism.
3. It cannot allocate enough probability mass to sparse future IV-level/regime cells without damaging another structural property.
4. Local objective pressure, wrappers, calibration policies, factor side-channels, and source-prior changes have all moved the tradeoff but not escaped it.

This is now an identifiability/product-boundary issue, not simply an optimizer issue.

## Decision

Do not run another immediate source-prior or hard-replay knob. The next principal move must be one of:

1. Product path: package 510a as an IV-only risk-scenario prototype with explicit regime-undercoverage warnings and separate base-law versus risk-policy metrics.
2. Data/model path: shift to a true joint multi-factor scenario generator with enough observable state and targets to learn the sparse stress regimes, not just condition IV on local factors.
3. Policy path: explicitly accept an auditable conservative stress overlay as policy calibration, reported separately from the learned conditional law.

For the original objective, the data/model path is the only one still aligned with a publishable learned generator. For near-term risk-manager presentation, the product path is the only honest deployable prototype.
