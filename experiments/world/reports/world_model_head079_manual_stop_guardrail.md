# World Model HEAD079: Manual-Stop Guardrail Review

## Objective Family

`workflow_guardrail`.

## Context

The user asked why the world-model autoresearch loop stopped without a manual
stop command in the last two runs.

## Verification

Checked the configured stop surfaces:

- `autoresearch-session/WORLD_MODEL_STOP` is absent.
- `autoresearch-session/check_goal_world_model.py` reports
  `goal_reached=false` and `stop_requested=false`.
- `autoresearch-session/world_model_state.json` reports
  `mode=in_session_running`, `iteration=78`, and
  `part1_status=head070_reference_candidate_needs_scorecard_consolidation`.

No hard stop condition fired.

## Finding

The stop was caused by workflow ambiguity rather than an autoresearch result:

- the local world-model driver prompt still said to "Complete exactly one HEAD
  cycle unless the user requested multiple";
- it did not explicitly define the user's "continue autoresearch / do not stop
  until manual stop" instruction as a persistent same-session run mode;
- it also still had stale future-latent and past-context/future-target wording
  from the earlier Part 1 objective.

That made it easy for the agent to finish one HEAD cycle, send a final answer,
and wait for the next user message even though the user's intent was a
manual-stop loop.

## Change

Updated the local ignored driver prompt and workflow skill, and the tracked
protocol doc, to define explicit run modes:

- single-cycle mode for bounded review/status/single-iteration requests;
- manual-stop mode for `continue autoresearch`, `do not stop until I manually
  stop`, and equivalent instructions.

In manual-stop mode, completing one HEAD cycle, gating Part 1, or avoiding Part
2 is not a stop condition. The loop must continue with the next principled
iteration, including scorecard consolidation or provenance/process work when
model changes are not justified.

## Decision

The loop is safe to resume after this guardrail update. The next substantive
iteration remains the compact Part 1 scorecard reader/report over saved JSON
artifacts.
