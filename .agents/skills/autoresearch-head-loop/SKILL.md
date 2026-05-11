---
name: autoresearch-head-loop
description: Run a resumable in-session HEAD autoresearch loop toward 11/11 on the common full 11-suite for a generalizable conditional scenario generator. Use persistent state files, append results to the true end of RESEARCH_LOG.md via research-log-tail-append, and create workflow-scoped checkpoint commits. Default mode is in-session; optional external-driver mode is secondary.
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

Use `independent-verifier` before promoting a model, changing an incumbent,
making a paradigm-shift decision, claiming production readiness, or turning a
result into a paper-facing claim.

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

## Verification, Brainstorming, And Literature Gates

Invoke `independent-verifier` before:

- declaring a framework, model, decoder, or workflow promotable;
- claiming production readiness, paper-level evidence, or a new incumbent;
- making a paradigm-shift decision;
- scaling from a smoke/testflight result to a larger training or evaluation run;
- accepting a surprising result whose mechanism is not yet clear.

The verifier must inspect actual code, configs, artifacts, metrics, and
research-log context. If the verdict is mixed or weak, choose
`post_experiment_analysis` or `research_ideation` before running another model
change.

Use brainstorming before adding a new objective family, decoder family, major
evaluation protocol, product workflow, or other nontrivial design change. Record
the alternatives, recommendation, falsifier, and deliberate non-goals before
implementation.

Run online research when the iteration depends on external method claims,
current library/API behavior, deployment choices, UI best practices, or
related-work positioning. Use primary sources whenever possible and record
citations plus the design impact in `RESEARCH_LOG.md`.

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
- generic support-aware mixed discrete-continuous heads selected by one
  data-derived rule
- deterministic channel/group balancing from the same formula

Not allowed for a single framework candidate:

- IV-specific, anchor-specific, or joint-specific loss recipes
- different rollout-energy or contrast weights per scope
- scope-specific backend, prior, sampler, AR/one-shot choice, or calibration layer
- factor-name-specific sparse/sticky output heads
- post-hoc gluing of separately sampled decks presented as one joint conditional law

If the best runs for IV-only, anchor-only, and joint use different recipes, label them as task-specialized frontiers and choose a framework-lock experiment or a post-experiment trade-off attribution before adding more knobs.

## Incumbent And Promotion Gate

The loop must keep separate incumbent roles. Do not let a newer diagnostic
branch silently replace a deployable baseline.

- `deployable_tri_scope_incumbent`: the best risk-manager-presentable
  IV/anchor/joint framework. Current real-VIX incumbent: `734a/739a`.
- `iv_research_frontier`: the best IV-only strict-suite or diagnostic frontier.
  Current frontier: `755a`.
- `diagnostic_branch`: any experiment that explains a failure mechanism but has
  not passed tri-scope non-regression.
- `rejected_branch`: any branch that violates hard non-regression gates.

A model can be promoted over the deployable tri-scope incumbent only after it
runs `iv_only`, `anchor_only`, and `joint` under the same framework recipe and
passes the Tri-Scope Non-Regression Gate below. An IV-only improvement can be
promoted only to `iv_research_frontier`, never to deployable incumbent.

## Tri-Scope Non-Regression Gate

Every serious candidate experiment must evaluate all three scopes:

- `iv_only`
- `anchor_only`
- `joint`

Short single-scope probes are allowed only when explicitly labeled
`diagnostic_branch`; they are not promotable and must not be described as the
current model.

For promotion, compare against the relevant incumbents, not only against the
previous iteration:

- against `734a/739a` for deployable tri-scope quality;
- against `755a` for IV-only research-frontier quality when the experiment is
  an IV-side repair.

Hard non-regression blockers for deployable promotion:

- IV mean reversion remains acceptable. Losing IV mean reversion makes the
  candidate not risk-manager deployable, even if strict coverage or likelihood
  metrics improve.
- IV scenario realism remains acceptable: surface validity, time-series
  realism, cross-cell dependence, pathwise jump realism, daily-change
  distribution, level distribution, and risk-state uncertainty allocation.
- Anchor-only realism does not regress: finite paths, factor daily-change
  distribution, factor tail scale, factor-factor dependence, and conditional
  panel response.
- Native joint realism does not regress: IV slice realism, IV-factor
  co-movement, factor-factor dependence, and conditional panel response.
- Sticky/intermittent channels are monitored separately unless a generic
  support-aware observation adapter is being tested.

Do not chase a strictly proper conditional-law objective by sacrificing
risk-manager deployability. If a strict-law repair improves coverage, interval
score, CRPS, or old cointegration while breaking mean reversion or scenario
realism, classify it as a diagnostic branch or rejected branch.

Old path-prediction conditionality is deprecated for promotion decisions when
the population risk-state uncertainty allocation diagnostic passes. The old
IV-EWMA cointegration suite remains a monitoring diagnostic. The additive
`iv_ewma_economic_link` diagnostic is the promotion-facing IV/EWMA economic-link
gate when present, but it is not cointegration because it does not test
residual stationarity. Do not call cross-cell correlation or IV-factor
co-movement "new cointegration" unless a separate residual-stationarity gate is
explicitly specified.

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

## Support-Aware Output Law Gate

A mixed discrete-continuous sparse/sticky head is allowed only as a generic
output-support adapter. It must be selected by a data-derived no-change or
near-no-change statistic, not by factor name. It must preserve the same shared
encoder, stochastic source, generative core, scalar objective recipe, sampler,
and tri-scope framework ID.

Before adding this head, the loop must run a sticky-channel audit:

- no-change mass by channel on train and validation;
- tolerance sensitivity for the no-change statistic;
- baseline continuous-head error on move-event rate;
- baseline continuous-head error on nonzero jump tails;
- stress-state move frequency and cross-factor concurrence.

Reject the head if it requires credit-specific rules, scope-specific weights, or
post-hoc snapping. If the generic head fails, document sticky low-activity
channels as a limitation instead of stacking more sparse-channel knobs.

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
7. Apply the checkpoint commit policy.
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
- `pending_git_files`
- `pending_commit_reason`

## Checkpoint Commit Discipline

This workflow's commit-title scope is `sni-head`.

Every iteration must append to the research log, but Git commits should be
created at coherent checkpoints rather than after every micro-step. Do not make
single-file or log-only commits unless the file is a standalone protocol
decision, user-requested checkpoint, or urgent fix.

Create a checkpoint commit only when the staged files form one legible unit:

- one completed experiment/evaluator plus its validation evidence;
- one postmortem or protocol decision that changes future loop behavior;
- implementation plus tests;
- a research-log entry bundled with the code, report, or artifact it documents.

If the iteration is partial, cosmetic, or log-only, leave the changes pending
and record `last_commit: null`, `pending_git_files`, and
`pending_commit_reason` in state.

Every auto-research checkpoint commit title must include the workflow scope:

- `feat(sni-head): add temporal anti-collapse test`
- `eval(sni-head): compare sticky-channel generic head`
- `docs(sni-head): record postmortem for 254a`
- `refactor(sni-head): tighten autoresearch loop tooling`

Use a concise commit body for nontrivial checkpoints:

```text
Workflow: autoresearch-head-loop
HEAD: <post_experiment_analysis|research_ideation|paradigm_shift|experiment>
Objective: <one sentence>
Validation: <commands or artifact checks>
Artifacts: <paths, or n/a>
Research log: <date/title or line if known>
Next bottleneck: <one sentence>
```

Do not stage unrelated local state by default. Avoid committing:

- `.claude/`
- `.codex/`
- `logs/`
- tool cache / editor state
- unrelated experimental debris

unless the iteration is explicitly about those files.

Before committing, check:

- The title would still be meaningful six months later.
- The commit answers one clear HEAD question or checkpoint.
- Related code, tests, docs, and research-log evidence are bundled together.
- No unrelated dirty worktree changes are staged.

## Research Log Entry Requirements

Every iteration’s appended log entry should include:

- context
- result
- mechanism read
- decision / next step
- independent-verifier verdict when a verification gate was triggered
- brainstorming alternatives and literature citations when those gates were triggered
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
