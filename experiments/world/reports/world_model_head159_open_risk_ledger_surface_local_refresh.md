# World Model HEAD159: Open-Risk Ledger Surface-Local Refresh

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Open-risk ledger refresh for Part 1 after surface-local route demotion.

## Settled Claims

- Part 1 pretraining remains masked multiview market-state representation
  learning. Future/range/scenario tasks remain downstream probes or consumers.
- The active learned candidate is still scaled Barlow from HEAD127/HEAD130.
- Scaled Barlow is not promoted: exact-state retention and baseline superiority
  remain blocking failures.
- Raw current-state IV/surface features remain the exact-state floor to beat.
- Hard-mask Barlow is a negative diagnostic, not a route to continue by making
  masks more aggressive.
- Minimal row-level context-to-target JEPA is demoted.
- Clean-target row-level context-to-target JEPA is demoted.
- Surface-local token/geometry context-to-target JEPA is demoted as
  implemented.
- Surface-local target coverage is not the failure; the target latent geometry
  is low-rank and token/factor dominated.
- Part B decoder work remains blocked.

## Open Risks

- A learned Part 1 representation still has not beaten raw current-state
  features on exact IV state.
- Regime/state probes remain below promotion threshold, despite some balanced
  recall signal.
- Current learned representations help some path-shape/risk-width probes, but
  not enough to clear baseline superiority.
- The repo has no validated target-latent surface that carries state variation
  before predictor training.
- Full-data convergence remains unproven for the smoke-scale candidates.

## Forbidden Routine Continuations

Do not continue by:

- tuning context-to-target hidden size, predictor depth, EMA decay, epochs, or
  Barlow weights;
- increasing surface-local target coverage;
- making masks harder as a standalone fix;
- starting Part B decoder work;
- turning downstream future/range probes into pretraining losses;
- claiming ImageNet-level or canonical JEPA behavior.

## Allowed Next Work

Manual-stop continuation should use bounded, non-ad hoc work:

- gate reconciliation after the surface-local demotion;
- provenance/report/index/package checks;
- a new design gate only if it first defines how target latents will carry state
  variation before predictor training;
- exact-state blocker analysis that does not become a reconstruction auxiliary
  loss by default.

## Decision

The current Part 1 state is blocked but well-characterized. Continue only with
gate/risk reconciliation or a genuinely new design gate; do not resurrect
demoted context-to-target routes through small knobs.
