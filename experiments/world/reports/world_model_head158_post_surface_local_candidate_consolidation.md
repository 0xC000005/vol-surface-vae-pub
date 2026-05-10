# World Model HEAD158: Post Surface-Local Candidate Consolidation

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Part 1 candidate consolidation after `token_geometry_level_context_to_target_jepa`
demotion.

## Context

HEAD157 demoted the current surface-local context-to-target implementation. The
workflow needs a clean candidate table before any future Part 1 design or Part B
work.

## Candidate Table

| candidate | status | key evidence | decision |
| --- | --- | --- | --- |
| Raw current IV/state features | baseline floor | Best exact-state IV MSE remains `0.005630` | Not learned Part 1, but still the floor to beat |
| HEAD070 direct masked-multiview Barlow | packaged smoke reference | Strong same-state retrieval and non-collapse at smoke scale | Reference candidate with caveats |
| HEAD127/130 scaled Barlow | active learned candidate | Better rank and state content than HEAD070, but raw IV still wins exact-state probes | `DO_NOT_PROMOTE` |
| HEAD123 hard-mask Barlow | negative diagnostic | Harder masks worsened baseline superiority | Demoted |
| HEAD140 minimal context-to-target | negative diagnostic | Worse than scaled Barlow on current-IV state probes and rank | Demoted |
| HEAD144 clean-target correction | negative diagnostic | Clean target construction trained but worsened rank and exact-state probes | Demoted |
| HEAD154-157 surface-local token context-to-target | negative diagnostic | Runnable, but target latent is low-rank and token/factor dominated | Demoted as implemented |

## Gate Status

- Representation health: scaled Barlow passes.
- Corruption robustness: scaled Barlow passes/partial depending layer.
- State content: partial; exact IV state remains below raw surface baseline.
- Baseline superiority: fail.
- Regime/state probes: fail for promotion, with some balanced-recall signal.
- Part B decoder: blocked.

## Decision

The active learned candidate remains scaled Barlow, but it is still
`DO_NOT_PROMOTE` and not Part-B-ready. The context-to-target family has produced
negative diagnostics in both row-level and token/geometry-level forms as
implemented.

Next work should not be another small JEPA knob. It should be one of:

- a new design gate that first proves the target latent surface carries state
  variation before predictor training;
- a non-model quality-gate reconciliation that makes the remaining exact-state
  blocker explicit;
- an open-risk/provenance refresh so future resumes do not treat demoted routes
  as active.
