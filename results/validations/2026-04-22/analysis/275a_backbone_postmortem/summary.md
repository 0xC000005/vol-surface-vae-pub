# 275a Postmortem

## Result
- Full 11-suite score: 2/11
- Passes: block_ar, cross_cell_correlation
- Best epoch: 4

## What Matters For Stage A
`275a` is intentionally deterministic, so the stochastic suites are expected failures:
- coverage
- conditionality
- regime_coverage

The useful question is whether it improves the deterministic 8/11 backbone objective.

## Key Metrics
- surface:
  - explosion rate: 0.000
  - calendar arbitrage: 8.7%
  - butterfly arbitrage: 27.1%
  - failure is narrow: butterfly worst tenor = 53.0% just above the 50% gate
- time_series:
  - ACF corr: 0.525
  - kurtosis ratio: 8.471
  - pathwise q99 ratio: 0.102
- cointegration:
  - ratio: 0.274
  - worst-cell ratio: 0.013
- distributional_fidelity:
  - level KS pass: 19/25
  - median-bias pass: 24/25
  - bias-magnitude pass: 23/25
  - MAE pass: 24/25
  - change KS pass: 0/25
- cross_cell_correlation:
  - corr ratio: 1.023
  - rank ratio: 0.815
- mean_reversion:
  - aggregate ratio: 0.684
  - active cells: 1/24
  - active-cell corr: 0.594
- pathwise_jump_realism:
  - max-jump KS: 1.000
  - q90 ratio: 0.063
  - q99 ratio: 0.102

## Mechanism Read
- The simple deterministic autoregressive next-change backbone is good at:
  - staying on support
  - preserving cross-cell mean correlation and rank
  - preserving level marginals and per-cell MAE
- It is bad at:
  - generating realistic change distributions
  - sustaining active mean reversion across cells
  - generating jump magnitude and incidence
- So the current pathology is clean:
  - `275a` is too smooth and too shallow dynamically
  - the problem is not support or static cross-cell structure
  - the problem is dynamic richness in the deterministic state evolution

## Decision
- Keep the two-level reset.
- Close `275a` as the minimal deterministic baseline.
- Next step: `275b-v0`, a deterministic latent-state backbone with an explicit observation model and deterministic latent transition.
- Rationale:
  - reuse the strongest stage-A-relevant lesson from `272/273`: explicit observation models help manifold fidelity
  - remove the stochastic transition machinery that caused the earlier manifold-departure problems
  - keep the model deterministic and first-principles for the Stage A objective
