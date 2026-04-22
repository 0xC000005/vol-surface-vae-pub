# 275 Two-Level Reset

## Why Reset Again
The current single-model autoresearch line remains scientifically coherent, but it is not moving the frontier toward `11/11` fast enough. The repo now has enough evidence that the objective is structurally split:

- deterministic center-path realism accounts for the oracle-reachable `8/11`
- stochastic scenario realism accounts for the remaining `3/11`

The `11`-suite remains the final risk-manager acceptance test. The reset changes the research program, not the acceptance criteria.

## Program Structure

### Stage A: Deterministic Backbone
Goal:
- maximize the oracle-reachable deterministic portion of the suite
- treat `coverage`, `conditionality`, and `regime_coverage` as intentionally unsatisfied by design

Target suites:
- surface
- time_series
- block_ar
- cointegration
- distributional_fidelity
- cross_cell_correlation
- mean_reversion
- pathwise_jump_realism

Core doctrine:
- no diffusion
- no flow matching
- no posterior/prior latent machinery
- no hard low-rank decoder
- no bounded side paths
- no error-correction baseline
- no one-shot future generator

Minimal model story:
- autoregressive/state-space deterministic next-change predictor
- generated-state feedback
- simple learned recurrent hidden state
- only reusable lesson from the older tree: the asinh local-scale change coordinate

### Stage B: Residual Scenario Layer
Goal:
- preserve the Stage A center path
- learn the residual conditional scenario law around it
- target the remaining stochastic `3/11`

Target suites:
- coverage
- conditionality
- regime_coverage

Core doctrine:
- mean-preserving by construction
- conditional scenario allocation on top of the frozen Stage A backbone
- no change to the final deterministic center path

## Immediate Next Step
Implement `275a-v0` as the Stage A deterministic backbone.

## 275a-v0 Definition
- history encoder: GRU over normalized surfaces
- recurrent decoder/state update: GRUCell over current generated surface
- output head: next-step change in asinh local-scale coordinate
- autoregressive rollout during training and inference
- loss: deterministic next-level + next-change supervision over the full rollout

## Kill Criteria
- if `275a` cannot materially beat the recent strict-reset lines on the deterministic suites, stop and rethink the Stage A family instead of adding local knobs
- if support validity fails badly, revisit the deterministic coordinate/object choice, not side-path corrections
- do not add stochastic heads to `275a`; keep the Stage A / Stage B split clean
