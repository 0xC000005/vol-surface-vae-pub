# 624a Conditional Transition-Flow Likelihood Paradigm

## Context

The 609-623 native joint AR line has now separated several failure modes:

- 610a raw-coordinate AR flow preserves surface, local path structure, cross-cell geometry, and pathwise jumps, but misses coverage/regime/level allocation.
- 612a conditional source scale reaches `6/11`, but the source-scale head collapses to the lower clamp, so it is not a clean learned uncertainty mechanism.
- 614-617 Gaussian/Student-t likelihood improves coverage and persistent undercoverage, but fixed elliptical increments do not place free-running multi-step levels or mean reversion correctly.
- 620a deterministic mean-rollout loss conflicts with stochastic likelihood and regresses to `3/11`.
- 622-623 encoded coordinates restore mean reversion but create high-biased undercoverage; temperature fixes average width while breaking dependence and tails.

The branch is still pointing to a specific missing object: not a bigger one-shot path decoder, not a scalar width layer, and not another deterministic center loss. We need a learned stochastic transition density that is flexible enough to represent skew, asymmetric tails, and non-elliptical cross-channel dependence while preserving the AR/state-feedback geometry that repeatedly performs best.

## Proposed Model

Use a conditional normalizing flow over the daily state increment:

```text
p(Y_1:T | H) = product_t p_theta(delta_t | H, Y_<t)
delta_t = Y_t - Y_{t-1}
```

Each one-step conditional density is an exact invertible flow:

```text
z_t ~ N(0, I)
delta_t = f_theta(z_t ; memory(H, Y_<t), current_state)
log p(delta_t | context) = log p(z_t) + log |det df^{-1}/d delta_t|
```

This is different from the previous branches:

- unlike 609/610 flow matching, training is exact conditional likelihood rather than velocity MSE;
- unlike 614-617 Gaussian/Student-t, the increment law is non-elliptical and learned;
- unlike 575-590 and 325/327 one-shot path laws, local AR state feedback remains explicit;
- unlike 620a, no deterministic mean path target is added;
- unlike 622/623, preprocessing is not the main mechanism.

## Clean Architecture

The first falsifier should stay minimal:

- one shared state panel for `iv_only` or `joint38`;
- same empirical-score coordinate and causal prefix memory as 614a;
- conditional affine coupling layers over the full state increment vector;
- fixed alternating masks;
- standard normal base;
- exact NLL summed over teacher-forced future steps;
- sample by drawing base noise and rolling generated states forward.

No separate IV/factor head, no low-rank readout, no post-hoc calibration, no retrieval, no regime labels, no bounded idiosyncratic path.

## Why This Is Worth Testing Despite Prior Failures

The previous likelihood family was too rigid. Student-t gained coverage but could not model the observed per-cell/tail/level geometry. The previous flexible flow family used flow matching and was vulnerable to source-scale degeneracy or source geometry mismatch. A conditional normalizing-flow likelihood combines the two useful lessons:

- exact density training from the likelihood path;
- flexible learned transport from the flow path.

The most important falsifier is whether a flexible transition likelihood improves the 617a/610a tradeoff without adding knobs:

- recover or exceed `5/11` on the broad 441-window bridge;
- preserve surface, block-AR, cointegration, cross-cell, and pathwise structure;
- improve at least one of coverage, level KS, mean reversion, regime layer2, or per-cell tail balance.

## Decision

Run `625a` as the minimal conditional RealNVP-style transition-flow likelihood over the same state-panel interface. Use `joint38` first because it is the target general model; use raw coordinates first because encoded coordinates introduced path-location bias. If 625a cannot beat the Student-t/flow frontier, close transition-density flexibility as insufficient and return to risk-policy product framing.

