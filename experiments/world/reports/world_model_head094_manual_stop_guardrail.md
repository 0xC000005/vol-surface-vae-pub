# World Model HEAD094: Manual-Stop Guardrail

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Workflow guardrail for manual-stop autoresearch.

## Hypothesis

The workflow should not allow a broad "practical runtime/tool limit" phrase to
be used as a discretionary pause reason in manual-stop mode.

## Falsifier

The iteration fails if the world-model skill or protocol still allows stopping
because bounded safe work is mostly exhausted, a cycle reached a clean
checkpoint, no model knob is justified, or the remaining work is
process-oriented.

## Execution

- Updated `.agents/skills/world-model-autoresearch/SKILL.md`.
- Updated `docs/research_protocols/world_model_autoresearch_plan.md`.
- Replaced broad practical runtime/tool-limit language with a narrower stop
  condition: an actual unrecoverable tool/platform failure that prevents further
  commands.
- Added explicit continuation guidance for gated or underdetermined work:
  provenance, manifest checks, metric reconciliation, report indexing, risk
  ledgers, and workflow guardrails.

## Decision

Manual-stop mode now forbids discretionary pauses from diminishing returns,
completed cycles, capped modeling branches, or process-only remaining work.

## Verification

- `rg` located the old stop-condition language before the edit.
- `git diff --check` passed after the edit.
