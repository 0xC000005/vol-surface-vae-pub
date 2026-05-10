# World Model HEAD146: Context-Target Route Decision

Date: 2026-05-10

## Iteration Type

`paradigm_shift`

## Objective Family

Decision boundary for `context_to_target_jepa` versus the active
`masked_multiview_invariance` reference.

## Hypothesis

The minimal context-to-target branch should continue only if at least one
bounded variant beats the scaled Barlow candidate on exact-state probes or
representation health.

## Falsifier

If both tested variants are worse than scaled Barlow on current-IV retention and
rank, then the minimal route should be demoted rather than tuned.

## Evidence

| branch | target construction | current-IV MSE | effective rank | key failure |
| --- | --- | ---: | ---: | --- |
| HEAD127 scaled Barlow | two masked same-state views | 0.013756 | 18.761132 | best current Part 1 candidate, still not promoted |
| HEAD140 target-only context-to-target | target encoder sees target-only sparse values and target mask | 0.015289 | 11.592254 | target latent is mask-family heavy; predictor is low-rank |
| HEAD144 clean-target context-to-target | target encoder sees clean full window; select target rows from output | 0.015907 | 8.059559 | trains, but worsens rank and exact-state probes |
| raw surface baseline | raw last IV surface | 0.005630 | 3.241518 | exact-state baseline still strongest for current-IV retention |

HEAD142 explains why the target-only branch was misleading: predicted-to-target
cosine was high (`0.979068`) while row retrieval was poor (`top10=0.042969`).
HEAD145 shows the canonical clean-target correction does not recover the route.

## Decision

Demote the minimal context-to-target branch.

This does not mean canonical JEPA is invalid for time series. It means the local
minimal GRU row-level context-to-target diagnostic is not the principled next
use of effort. A stronger canonical JEPA attempt would need a deeper design:
token/geometry-level target representations, explicit target-position
conditioning, and health losses on the evaluated clean context surface. That is
a new architecture proposal, not a small patch.

For the current auto-research loop:

- keep HEAD127/HEAD130 scaled Barlow as the best known Part 1 candidate;
- keep Part 1 status `DO_NOT_PROMOTE`;
- do not tune context-to-target masks, EMA, hidden size, or predictor depth;
- do not start Part 2 decoder work from these candidates;
- next work should be bounded evidence consolidation or a new design gate, not
  an ad hoc model knob.
