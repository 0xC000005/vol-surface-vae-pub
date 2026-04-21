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

## In-Session Workflow

When the user says things like:

- `start autoresearch`
- `continue autoresearch`
- `run 2 iterations`
- `run until blocked`

use this workflow:

1. Read the goal and current state.
2. Determine the requested run budget:
   - default: `1` iteration
   - if user says `run N iterations`: use `N`
   - if user says `run until blocked`: keep iterating within this session until:
     - goal reached
     - `autoresearch-session/STOP` exists
     - a real blocker requires human input
     - or session/runtime/tool limits make further work unreasonable
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
- a hard blocker requires human input
- the user-specified in-session iteration budget is exhausted

## Session Commands

Treat these as canonical user commands:

- `continue autoresearch`
  - run one next principled iteration
- `run 2 autoresearch iterations`
  - run exactly two iterations back-to-back
- `run autoresearch until blocked`
  - keep iterating in this session until a real stop condition occurs

## Reality Constraint

Inside Codex, this still remains a **session-bounded** loop.

It persists across turns because state and logging are saved, but it does not become a background daemon. If the session ends, the loop must be resumed in a later turn with the saved state.
