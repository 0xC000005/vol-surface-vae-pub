# 436a: Feasibility vs Learned Objective

## Context

`435a` reached `11/11` on the unchanged full suite, but it did so with an explicit
validation-oracle construction:

- validation future path provides the center;
- `392a` residual shapes provide sample diversity;
- coverage misses are controlled directly;
- tiny balanced center jitter avoids degenerate median-bias behavior.

The artifact is correctly marked:

- `oracle_uses_validation_future=true`;
- `not_deployable=true`;
- `not_learned_conditional_law=true`.

## What 435a Proves

The suite is not mathematically contradictory. The 11 gates can be satisfied jointly by a
controlled sample law.

This matters because it separates two questions:

1. **Suite feasibility:** yes, `11/11` is possible.
2. **Learned conditional generator:** not yet; the valid learned frontier remains `392a`
   at `8/11`.

## What 435a Does Not Prove

It does not prove that a model can learn the required conditional law from the available
history signal.

It does not produce a deployable scenario generator.

It does not justify reporting `11/11` as a model result in a paper.

## Mechanism Read

The remaining gap is now sharply defined:

- structural dynamics are learnable: `392a` passes conditionality, cointegration,
  cross-cell correlation, mean reversion, and pathwise realism;
- suite-level marginal/regime reliability is achievable only when the future center and
  coverage miss process are controlled with oracle information;
- prior learned attempts to internalize that control either lose structure or fail to
  repair the marginal/regime suites.

## Decision

The next scientific step is a product decision, not another blind model search:

- report `392a` as the learned conditional-law frontier (`8/11`);
- report `435a` only as a suite feasibility upper bound;
- if a risk product requires `11/11`, design a separately evaluated deployable
  calibration policy and do not call it the learned conditional law;
- if a paper requires one learned model to pass `11/11`, the current evidence says the
  objective is not yet supported by this data/model setup.

