# World Model HEAD148: Scaled Gate Reconciliation

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`part1_quality_gate_reconciliation`; no model change.

## Hypothesis

After demoting the minimal context-to-target branch, the formal Part 1 gate
should still point to the same active candidate and blockers.

## Falsifier

The reconciliation would fail if the context-to-target experiments changed the
promotion status, removed a failed gate layer, or created evidence that Part B
can start.

## Reconciled Gate

| layer | status | current interpretation |
| --- | --- | --- |
| package integrity | PASS | manifest, reports, and tracked guardrail docs verify with the package checker |
| representation health | PASS | scaled Barlow has high retrieval, rank `22.32` in training artifact and rank `18.76` on present-state probe features |
| corruption robustness | PASS | HEAD129/HEAD130 show no large mask-family or stratified-mask failure |
| state content | PARTIAL | scaled Barlow improves over earlier Barlow variants but still loses exact current-IV retention to raw surface |
| baseline superiority | FAIL | scaled Barlow helps path-shape/risk-width targets but loses persistence/exact-state targets |
| market-state regime probe | FAIL | accuracy remains below majority, though macro/rare-class recall has signal |
| scale and stability | PARTIAL | seed stability passed at smoke scale, but this is not full-data convergence |

## Context-Target Reconciliation

The context-to-target branch changes none of these gate statuses:

- HEAD140 target-only context-to-target is worse than scaled Barlow on exact
  IV retention and rank;
- HEAD142 shows its target latent is mask-family heavy and the predicted latent
  is low-rank despite high cosine;
- HEAD144/145 clean-target correction is also worse than scaled Barlow;
- HEAD146 demotes the minimal route;
- HEAD147 keeps scaled Barlow as the active learned candidate.

## Decision

Part 1 remains `DO_NOT_PROMOTE` and not ready for Part B.

The current active candidate is still HEAD127/HEAD130 scaled Barlow with
caveats. Any new Part 1 model proposal must target the exact-state/baseline
blocker directly and pass the literature/design gate first. Part 2 decoder work
remains blocked.
