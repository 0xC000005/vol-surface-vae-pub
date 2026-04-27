# 647a Mixed-Coordinate Path Flow

## Hypothesis

The one-step AR transition family may be misaligned with the evaluated object: a
30-day conditional path distribution. 647a tests the minimal alternative: one
shared model conditions on the 38-channel history and generates the full
future `[horizon, channel]` tensor jointly in mixed coordinates.

## Implementation

- IV channels use level-score deltas from the current state.
- Anchor-factor channels use encoded daily increment scores.
- One history transformer encodes level and increment scores.
- One future-path transformer denoises the full future tensor with flow
  matching.
- There is no retrieval, no separate IV/factor generator, no low-rank readout,
  and no post-hoc stress deck composition.

## Results

Artifacts:

- model: `models/backfill/647a_joint38_mixed_path_flow_e8_w2048_s647/best_model.pt`
- IV suite: `results/autoresearch/647a_joint38_mixed_path_flow_e8_w2048_s647/full11.json`
- joint audit: `results/autoresearch/647a_joint38_mixed_path_flow_e8_w2048_s647/joint_panel.json`

IV 11-suite:

- score: 4/11
- pass: surface validity, block boundary smoothness, IV-EWMA cointegration,
  cross-cell correlation
- fail: coverage, conditionality, time-series properties, regime coverage,
  distributional fidelity, mean reversion, pathwise jump realism
- coverage: overall 90% CI coverage was 66.3%; h1 passed, h7/h14/h30 failed
- conditionality: overall MAE reduction was -0.7%; h1/h7 were positive but
  h14/h30 regressed
- tails: generated kurtosis ratio was 0.337; per-cell q99 tail-scale pass was
  14/25
- pathwise max-jump KS was 0.618 versus the relaxed 0.50 gate

Joint-panel audit:

- finite rate: 1.0
- factor delta KS mean: 0.087
- factor delta KS pass under 0.20: 13/13
- factor q99 tail pass under [0.5, 2.0]: 11/13
- factor-factor correlation: 0.829
- IV-factor correlation: 0.845

## Mechanism Read

647a validates the unified native joint-panel direction: the anchor-factor
scenario law is coherent and no longer relies on rank-glued decks. It does not
solve the IV path-law bottleneck. The model still allocates too little
conditional probability to the realized 30-day IV path, weakens tail shape, and
does not condition width strongly on regimes.

The failure is not caused by post-hoc factor gluing, one-step rollout drift, or
invalid support. It is more likely an objective mismatch: plain flow-matching
MSE on a single future tensor learns a smooth average transport that preserves
support and correlation but does not directly optimize conditional coverage,
tail allocation, or realized path probability.

## Decision

Keep 647a as the clean unified joint baseline. Do not add scalar sampling
temperature, per-factor branches, or risk-policy stress overlays as the next
move. The next principled direction is to keep the same architecture and change
the training objective toward a proper sample-based conditional law objective
that directly sees multiple generated paths per history.
