# World Model HEAD174: Temporal JEPA Route Decision

Date: 2026-05-11

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe_route_decision`; this is a provenance and route-control
iteration, not a new Part 1 objective and not Part B decoder work.

## Literature Status

`no_new_objective`; the decision inherits the HEAD173 frozen-evaluation rule:
representation claims require comparison against frozen probes and simple
controls, not pretraining loss alone.

## Hypothesis

If HEAD173 is a valid falsifier for the current temporal context-to-target
route, active restart documents should preserve it as a negative bakeoff and
should not point future runs into temporal knob tuning.

## Falsifier

The route decision fails if active package documents still frame HEAD172/173 as
positive promotion evidence, Part-B authorization, or permission to tune
temporal context-to-target knobs.

## Evidence

- HEAD173 current-IV raw+temporal/raw ratio: `0.837799`.
- HEAD173 current-IV raw+random/raw ratio: `0.817571`.
- HEAD173 temporal raw+future improvements: `1/5`.
- HEAD173 random raw+future improvements: `3/5`.
- HEAD173 scaled Barlow raw+future improvements: `3/5`.
- HEAD173 temporal raw+beats scaled Barlow raw+: `1/5`.
- Package checker after HEAD173: `ok`, `75` reports, `5` guardrail docs,
  `9` ignored artifacts.

## Route Decision

Demote the current temporal context-to-target route as implemented. HEAD172 was
a useful smoke signal, but HEAD173 shows the signal is not specific enough:
the current-IV gain is matched or beaten by random temporal features, and the
future-probe utility underperforms random and scaled Barlow controls.

The active learned candidate remains the scaled Barlow branch, still
`DO_NOT_PROMOTE`. Part B remains blocked.

## Allowed Next Work

- Provenance and package consistency checks.
- Gate reconciliation or risk-ledger updates that preserve exact-state
  retention and baseline superiority as active blockers.
- A genuinely new Part 1 design gate only if it explains how target latents
  will carry state variation before predictor training.

## Blocked Next Work

- Temporal context-to-target hidden-size, epoch, mask, target-len, EMA, or
  predictor-depth tuning.
- Treating raw+temporal current-IV improvement as promotion evidence.
- Starting Part B decoder work from HEAD172 or HEAD173.

## Decision

Promotion decision: `DO_NOT_PROMOTE`.

Next iteration should be non-modeling guardrail/provenance work or a new design
gate with a clearly different hypothesis. Do not continue by tuning the current
temporal context-to-target route.
