---
name: autoresearch-head-loop
description: Run a resumable in-session HEAD autoresearch loop toward 11/11 on the common full 11-suite for a generalizable conditional scenario generator. Use persistent state files, append results to the true end of RESEARCH_LOG.md via research-log-tail-append, and create one focused commit per iteration. Default mode is in-session; optional external-driver mode is secondary.
---

# Autoresearch HEAD Loop

This skill is for **persistent resumable autoresearch inside the current Codex session**.

It exists to make progress:

- resumable across turns,
- disciplined within a single long-running session,
- and optionally automatable by an outer driver later.

## Primary Mode: In-Session

Default mode is **in-session loop execution**.

That means:

1. The user starts or continues the loop from the current Codex session.
2. Codex runs one or more full HEAD iterations in sequence.
3. Persistent state is updated after each iteration.
4. The user can later say `continue autoresearch`, `run 2 iterations`, or `run until blocked`.

This is the primary workflow.

## Secondary Mode: External Driver

If the user later wants unattended execution outside the session, the same state files can be reused by an outer driver.

That mode is optional and secondary. Do not center the workflow around it unless the user explicitly asks.

## Required Companion Skill

Use `research-log-tail-append` for **every** append to `RESEARCH_LOG.md`.

Never append to the research log with a generic patch against `---`.

## Persistent Files

Read these first:

- `autoresearch-session/goal_11x11.json`
- `autoresearch-session/state_11x11.json`
- `autoresearch-session/driver_prompt_11x11.md`
- `autoresearch-session/check_goal_11x11.py`

## HEAD Meaning

- `H`: Hypothesis
- `E`: Execute
- `A`: Analyze
- `D`: Decide

Each iteration must complete one full HEAD cycle.

## Allowed Iteration Types

The next step is not always an experiment.

Choose exactly one of:

- `post_experiment_analysis`
- `research_ideation`
- `paradigm_shift`
- `experiment`

Choose the **most principled** next step based on current evidence.

## Decision Law

Use this order:

1. If there is a fresh result whose mechanism is not understood enough to pick the next move:
   choose `post_experiment_analysis`.
2. If the current family appears capped or the next step is underdetermined:
   choose `research_ideation`.
3. If the evidence says the current family should be abandoned:
   choose `paradigm_shift`.
4. If there is a clear decisive falsifier already identified:
   choose `experiment`.

Do not run another experiment just because “more experiments” sounds active.

## Clean Pathology Guard

The loop must stay scientifically legible.

If either of these becomes true:

- the current failure mechanism is no longer clean enough to describe in one or two concrete causal statements
- or the active family is accumulating too many special-case knobs, branches, losses, or flags

then do **not** keep experimenting inside the same branch by default.

Instead:

1. stop and choose `post_experiment_analysis` if the mechanism is not yet clear,
2. choose `research_ideation` if the mechanism is clear but the next move is underdetermined,
3. choose `paradigm_shift` if the active decomposition itself now looks wrong.

The goal is not only `11/11`. The goal is `11/11` with a model that remains elegant, publishable, and defensible as a generalizable conditional scenario generator.

## Single-Framework Candidate Gate

For this repository, a result is **not** a validated framework candidate unless the same frozen `framework_id` is evaluated on:

- `iv_only`
- `anchor_only`
- `joint`

The `framework_id` freezes the generated coordinate, normalization family, temporal factorization, backend/stochastic source, shared core, scalar loss terms, loss weights, sampler, and training protocol.

Allowed scope differences are only data-interface adaptations:

- input and output dimensionality
- support/coordinate transform implied by variable type
- input heads and decoder heads
- deterministic channel/group balancing from the same formula

Not allowed for a single framework candidate:

- IV-specific, anchor-specific, or joint-specific loss recipes
- different rollout-energy or contrast weights per scope
- scope-specific backend, prior, sampler, AR/one-shot choice, or calibration layer
- post-hoc gluing of separately sampled decks presented as one joint conditional law

If the best runs for IV-only, anchor-only, and joint use different recipes, label them as task-specialized frontiers and choose a framework-lock experiment or a post-experiment trade-off attribution before adding more knobs.

## Trade-Off Attribution Gate

When one frozen recipe works on one scope but fails on another, do not switch variants immediately. First classify the trade-off across:

- train versus validation behavior
- marginal realism
- path realism
- conditionality
- diversity
- dependency and co-movement
- data object, objective balancing, shared-core capacity, head interference, and distribution shift

If the trade-off mechanism is not clear enough to state in concrete causal terms, the next HEAD iteration must be `post_experiment_analysis` or `research_ideation`, not another model tweak.

## In-Session Workflow

When the user says things like:

- `start autoresearch`
- `continue autoresearch`
- `run 2 iterations`
- `run until blocked`

use this workflow:

1. Read the goal and current state.
2. Determine the requested run budget:
   - default for `continue autoresearch`: keep iterating within this session until manually stopped or a configured hard stop condition occurs
   - if user says `run N iterations`: use `N`
   - if user says `run until blocked`: keep iterating within this session until:
     - goal reached
     - `autoresearch-session/STOP` exists
     - configured hard iteration cap is reached
     - or session/runtime/tool limits make further work unreasonable
   - if the user explicitly asks for exactly one iteration, honor that
3. Before each iteration:
   - check stop condition
   - decide the iteration type using the decision law
4. Execute exactly one iteration.
5. Update `autoresearch-session/state_11x11.json`.
6. Append a concise entry to the true tail of `RESEARCH_LOG.md` using `research-log-tail-append`.
7. Make one focused git commit for that iteration.
8. If more iterations remain in the current in-session budget, continue immediately.

## Iteration Granularity

One iteration means one of:

- one post-experiment analysis cycle
- one research ideation cycle
- one paradigm-shift decision cycle
- one decisive experiment cycle

Do not bundle multiple unrelated hypotheses into one iteration just to look busy.

## State Update Requirements

After each iteration, update:

- `iteration`
- `last_iteration_type`
- `active_family`
- `current_bottleneck`
- `last_result_summary`
- `next_step_recommendation`
- `current_best_n_pass`
- `current_best_model`
- `current_best_note`
- `goal_reached`
- `last_commit`

## Commit Discipline

Every iteration must end with one focused commit.

Good commit pattern:

- `feat: autoresearch 254b temporal anti-collapse test`
- `docs: autoresearch postmortem for 254a`
- `refactor: autoresearch loop tooling`

Do not stage unrelated local state by default. Avoid committing:

- `.claude/`
- `.codex/`
- `logs/`
- tool cache / editor state
- unrelated experimental debris

unless the iteration is explicitly about those files.

## Research Log Entry Requirements

Every iteration’s appended log entry should include:

- context
- result
- mechanism read
- decision / next step
- artifact paths when applicable

Keep it concise, but enough to resume from the log alone.

## Stop Condition

The loop stops when either:

- `current_best_n_pass >= target_n_pass`
- `goal_reached == true`
- `autoresearch-session/STOP` exists
- the configured hard iteration cap is reached
- the user-specified in-session iteration budget is exhausted

Research or model blockers are **not** stop conditions by default. When a blocker appears,
the loop should react autonomously by choosing the most principled next iteration type:

- `post_experiment_analysis`
- `research_ideation`
- `paradigm_shift`

Only platform/runtime limits remain external stop conditions.

## Session Commands

Treat these as canonical user commands:

- `continue autoresearch`
  - keep iterating in this same session until manually stopped or a configured hard stop condition occurs
- `run 2 autoresearch iterations`
  - run exactly two iterations back-to-back
- `run autoresearch until blocked`
  - keep iterating in this session until a configured hard stop condition occurs

## Reality Constraint

Inside Codex, this still remains a **session-bounded** loop.

It persists across turns because state and logging are saved, but it does not become a background daemon. If the session ends, the loop must be resumed in a later turn with the saved state.
