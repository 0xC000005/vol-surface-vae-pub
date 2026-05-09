# World Model HEAD047: Autoresearch Continuation Guardrail

Date: 2026-05-09

Iteration type: `paradigm_shift`

## Trigger

The user asked why the loop stopped after HEAD046. There was no
`autoresearch-session/WORLD_MODEL_STOP` file and `goal_reached` was false. The
stop happened because the workflow state said not to add more Part 1 knobs and
not to start decoder work unless explicitly requested. That was a constraint on
the next iteration, not a stop condition.

## Failure Class

Workflow-control failure: a modeling guardrail was incorrectly treated as a loop
stop condition.

## Fix

Updated `.agents/skills/world-model-autoresearch/SKILL.md` with a new section,
`When A Modeling Track Is Gated`.

The rule is now explicit:

- "no more Part 1 knobs" is not a stop condition;
- "do not start decoder work unless explicitly requested" is not a stop
  condition;
- if modeling tracks are gated, continue with bounded non-modeling work such as
  provenance checks, manifest/report reconciliation, handoff criteria, artifact
  inventory, and workflow guardrails.

## Decision

Continue autoresearch unless one of the actual stop conditions appears:

- `goal_reached` is true;
- `autoresearch-session/WORLD_MODEL_STOP` exists;
- a requested iteration budget is exhausted;
- runtime/tool limits make further work unreasonable.

Given the current state, the next iteration should remain non-modeling and
should not add Barlow Twins, retrieval/neighborhood objectives, target sweeps,
decoder work, or any new Part 1 knobs.

## Artifacts

- `.agents/skills/world-model-autoresearch/SKILL.md`
