# 506a Post-MixLinear Architecture Decision

## Context

505a was the clean architecture-side test prompted by recent parameter-efficient
time-series work: add a very small MixLinear-style direct future-path flow and
evaluate it without calibration. It trained quickly with `67,849` parameters but
scored only `2/11`.

## Diagnosis

The failure is not generic undertraining. The model fits the flow objective
monotonically, but the official suite shows that the learned sample law is not a
coherent financial surface law:

- Cross-cell corr ratio is `0.376`, below the `[0.5, 2.0]` gate.
- Effective-rank ratio is `3.337`, above the `3.0` gate.
- Daily-change KS is only `10/25`.
- Level KS is only `2/25`.
- Path max-jump KS is `0.573`, above the relaxed `0.50` gate.
- Conditionality MAE reduction is only `2.29%`.

This points to excess independent local variation and weak shared surface
geometry. The small linear mixer has enough local capacity to generate plausible
values, but its base noise plus separable mixing does not force samples onto the
shared low-dimensional manifold that vol surfaces empirically occupy.

## Architecture Implication

The next clean change is not to make MixLinear wider or deeper. That would test
capacity, not the diagnosed pathology. The one allowed first-principles bias from
the reset doctrine is a narrow learned bottleneck/shared-factor geometry.

For a flow model, the least intrusive way to introduce this is at the source
distribution:

- keep the full future-path flow objective;
- keep the architecture otherwise unchanged;
- replace fully independent `x0` source noise with a mixed source:
  `sqrt(rho) * common_factor_noise + sqrt(1-rho) * local_noise`;
- use the same source law in training and sampling.

This is not post-hoc calibration and not an evaluator-specific correction. It is
part of the generative prior for the continuous path law. The falsifier is clear:
if a modest shared source factor does not improve cross-cell geometry and
pathwise realism without destroying level occupancy, direct-path new-core work
should be closed again.

## Decision

Run one source-geometry falsifier next: direct empirical normal-score path FM
with shared source noise. Use the existing axial direct-path core first, not the
failed MixLinear core, because 413a already showed that axial direct path can
learn level occupancy (`20/25`) while MixLinear could not.
