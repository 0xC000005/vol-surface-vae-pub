# World Model HEAD160: Part 1 Gate Reconciliation After Surface-Local Demotion

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`part1_quality_gate_reconciliation`; no model change.

## Hypothesis

The HEAD154-157 surface-local branch should not change the formal Part 1 gate
unless it improves exact-state retention, baseline superiority, or
representation health relative to scaled Barlow and raw current-state baselines.

## Falsifier

This reconciliation would fail if the surface-local route created evidence that
Part 1 is ready for Part B, removed the exact-state/baseline blocker, or
supplanted scaled Barlow as the active learned candidate.

## Reconciled Gate

| layer | status | current interpretation |
| --- | --- | --- |
| package integrity | PASS | package checker verifies manifest, source reports, ignored artifact identities, and guardrail docs |
| representation health | PASS for scaled Barlow | scaled Barlow remains the only learned candidate with healthy rank/retrieval at smoke scale |
| corruption robustness | PASS/PARTIAL for scaled Barlow | HEAD129/130 remain the current mask-artifact and stratified-mask evidence |
| state content | PARTIAL | scaled Barlow improves over earlier learned variants but exact IV state remains below raw features |
| baseline superiority | FAIL | raw current-state features still beat learned embeddings on exact-state/persistence targets |
| market-state regime probe | FAIL | accuracy remains below majority; balanced recall signal is diagnostic only |
| scale and stability | PARTIAL | smoke-scale seed/rank stability is not full-data convergence |
| context-to-target alternatives | FAIL/DEMOTED | row-level, clean-target, and surface-local context-to-target routes are demoted as implemented |

## Surface-Local Reconciliation

The surface-local branch changes none of the promotion gates:

- HEAD151-152 prove data contract and target coverage, not representation
  quality.
- HEAD154 trains but has weak target-token retrieval and low predicted rank.
- HEAD155 shows the selected target latent is also low-rank and the predictor
  shrinks variance.
- HEAD156 shows the target latent is token/factor dominated rather than exact
  row/state dominated.
- HEAD157 demotes the current implementation and forbids small-knob tuning.

## Decision

Part 1 remains `DO_NOT_PROMOTE` and not ready for Part B. The active learned
candidate remains scaled Barlow with caveats. Any future Part 1 design must
target exact-state retention and baseline superiority directly, and any
context-to-target revival must first prove target latents carry state variation
before predictor training.
