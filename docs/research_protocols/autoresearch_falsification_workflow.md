# Autoresearch Falsification Workflow

## Purpose

Autoresearch is not a backend search loop. A bad score does not justify jumping
to the next architecture, sampler, loss, or calibration layer. Every failed
experiment must first identify what failed and rule out simpler explanations.

## Default Rule

Stay on the current methodology until diagnostics justify a switch.

Do not switch model family, backend, AR/one-shot structure, or architecture just
because:

- the 11-suite score is low;
- one metric regressed;
- validation looks worse than train-tail;
- a newer paper suggests a different backbone;
- another backend is easy to try.

## Failure Classification

Each failed iteration must assign exactly one primary failure class and may assign
secondary classes:

- `data_object`: the generated target is not stable, realistic, finite, or
  learnable after preprocessing.
- `output_support`: the generated target is stable, but the decoder/output law
  has the wrong support for the empirical channel type, such as forcing a
  purely continuous daily move law onto a channel with a large atom or near-atom
  at zero.
- `reconstruction`: true normalized/generated objects reconstruct correctly in
  theory but not in implementation or audit alignment.
- `train_fit`: the model does not learn the in-training/train-tail conditional
  law.
- `conditionality`: shuffled or altered histories produce similar distributions.
- `diversity`: same-history samples collapse or do not respond to stochastic
  source.
- `dependency`: marginal scenarios are plausible but cross-factor/cross-cell
  dependence fails.
- `calibration`: scenarios are realistic and conditional but width/location is
  miscalibrated.
- `distribution_shift`: train-tail works but validation fails materially.
- `backend`: object, reconstruction, train-fit, conditionality, diversity, and
  dependency are adequate, but the sampler/objective still cannot allocate path
  probability mass.
- `test_mismatch`: a metric is shown by oracle/split audit to be unstable,
  contradictory, or not aligned with the risk objective.

Only `backend` justifies changing flow/diffusion/copula/likelihood backend.
Only sustained `train_fit` or diagnosed representation limits justify changing
architecture.
Only `test_mismatch` justifies changing the test.

## Required Audits Before Switching

Before any backend or architecture switch, the iteration report must include:

- train-tail IV full-suite result;
- validation IV full-suite result;
- train-tail joint-panel audit when joint data are used;
- validation joint-panel audit when joint data are used;
- condition-shuffle audit;
- same-history diversity audit;
- data-object realism audit in the generated coordinate;
- reconstruction/alignment audit;
- comparison to the current incumbent on the same split and sample count.

If an audit is skipped, the report must state why and treat the conclusion as
provisional.

## Minimal-Fix Rule

After a failure, the next iteration should make the smallest intervention that
targets the diagnosed failure class.

Examples:

- `data_object`: change normalization/coordinate estimator, not backbone.
- `output_support`: change the support-aware output law or typed decoder head,
  not backend or factor-specific preprocessing.
- `reconstruction`: fix inverse transform/alignment, not model.
- `conditionality`: strengthen conditioning pathway or loss audit, not switch
  to diffusion by default.
- `diversity`: inspect source noise/objective/temperature before adding a new
  decoder.
- `dependency`: improve shared latent/dependency modeling before adding separate
  factor decks.
- `calibration`: use a principled calibration objective or diagnostic adapter,
  clearly separated from base-law metrics.
- `distribution_shift`: test adaptation/windowing/calibration after proving
  train-tail works.

## Research Axes

Treat each modeling choice as a separate research axis. Do not change multiple
axes in one iteration unless the failure diagnosis proves they are coupled.

- `data_object`: transforms, normalization, local scale/center estimator, and
  generated coordinate.
- `output_law`: support-aware decoder/output heads, such as continuous,
  bounded, positive-level, or mixed discrete-continuous sparse/sticky movement
  heads.
- `temporal_factorization`: autoregressive, one-shot, or hybrid path-latent plus
  autoregressive rollout.
- `backend`: flow matching, diffusion, copula, likelihood, energy score, or
  other conditional-law objective/sampler.
- `encoder`: Transformer, variable-token encoder, mixer, recurrent model, U-Net,
  or other conditioning architecture.
- `dependency_structure`: shared latent, attention/copula/dependency head,
  factor-token interaction, or correlation modeling.
- `calibration`: post-model risk-policy calibration, reported separately from
  the learned base law.

Example of an invalid next step after a failed run: changing EWMA normalization,
switching AR to one-shot, and replacing flow with diffusion in one experiment.
That result would be uninterpretable.

Example of a valid next step: if the failure class is `data_object`, compare
fixed EWMA/RMS scale to robust MAD/IQR scale while keeping AR, backend, encoder,
training budget, and evaluation fixed.

## Knob Ledger

Every new research knob must be logged with:

- name;
- default value;
- failure class it addresses;
- why it is not factor-specific overfitting;
- removal criterion;
- scopes it must be tested on (`iv_only`, `anchor_only`, `joint` where
  applicable).

If a knob cannot be assigned a failure class and removal criterion, do not add it.

## Support-Aware Output Law Gate

A mixed discrete-continuous output head is allowed when it is the statistically
correct support for the data, not a factor-name patch. This gate exists for
channels whose daily encoded increments have a large no-change or near-no-change
mass plus rare nonzero jumps.

Allowed form:

- the shared encoder, shared generative core, and stochastic source remain the
  same;
- the adapter is typed by a data-derived support statistic, not by a factor name;
- the head models both event probability and nonzero move size, for example
  `P(move | state)` and `p(move_size | move, state)`;
- the scalar training objective is one frozen framework recipe across
  `iv_only`, `anchor_only`, and `joint`, with channel-type terms produced by the
  same formula;
- the same channel-selection rule is used across scopes.

Required audits before adding this head:

- train and validation no-change mass by channel in the generated coordinate;
- sensitivity of the no-change statistic to the tolerance used for
  "near-zero";
- baseline continuous-head error on no-change mass, move-event rate, nonzero
  jump-size tails, and stress-state move frequency;
- check that the issue is not caused by data alignment, stale quotes, missing
  values, or reconstruction error.

Forbidden forms:

- selecting `aaa_oas` or `bbb_oas` by name without a data-derived rule;
- adding credit-spread-specific loss weights;
- changing backend, sampler, stochastic source, or calibration layer only for
  sparse channels;
- post-hoc snapping sampled paths to zero after generation and presenting that
  as a learned conditional law.

Acceptance criteria:

- no-change mass and move-event rate become realistic;
- nonzero jump-size tails remain realistic;
- stress-state widening remains possible and state-dependent;
- IV realism, anchor dependence, conditional panel response, and joint
  co-movement do not regress materially;
- if these criteria fail, document sticky low-activity channels as a limitation
  rather than stacking more sparse-channel knobs.

## Literature Search Gate

Online literature search is part of diagnosis, not random ideation. Use it when:

- the same failure class appears in two consecutive iterations without progress;
- a backend, encoder, temporal-factorization, or methodology switch is being
  considered;
- a data-object or normalization failure is not understood;
- the loop is running out of local first-principles fixes;
- a model limitation is about to be declared.

The search must be targeted. Each search should state:

- diagnostic question;
- venues or source types checked, prioritizing papers/docs over blog posts;
- what was learned;
- how it changes or does not change the next minimal fix.

Do not use literature search to justify hopping to a fashionable architecture
without a matching failure diagnosis.

## Generalization Guardrails

To avoid over-engineering for IV-only or joint-IV-anchor artifacts:

- the same framework should run on `iv_only`, `anchor_only`, and `joint` scopes;
- preprocessing rules must be support/coordinate based, not factor-name tuned;
- no per-factor hand-tuned constants unless explicitly documented as policy
  calibration rather than learned law;
- a mechanism promoted as general should not improve one scope by breaking
  another without an explanation;
- synthetic sanity panels should be used when possible: bounded mean-reverting,
  drifting log asset, spread-like spike process, and correlated multi-factor
  process.

## Single-Framework Candidate Gate

An experiment is not a framework candidate just because it is best-in-class for
one scope. A candidate must define a frozen `framework_id` and run the same
recipe on all required scopes:

- `iv_only`;
- `anchor_only`;
- `joint`.

The `framework_id` freezes:

- generated coordinate and normalization family;
- temporal factorization;
- backend and stochastic source;
- shared core architecture;
- scalar training objective, loss terms, loss weights, sampler, and training
  protocol;
- evaluation sample count and audit protocol used for candidate comparison.

Allowed scope differences are limited to data-interface adaptations:

- input dimension and output dimension;
- support/coordinate transform implied by variable type;
- input heads and decoder heads;
- support-aware mixed discrete-continuous heads selected by one data-derived
  rule;
- deterministic channel or group balancing computed by the same formula across
  scopes.

Not allowed in a framework candidate:

- IV-specific, anchor-specific, or joint-specific loss recipes;
- different rollout-energy or contrast weights per scope;
- scope-specific backend, AR/one-shot, prior, sampler, or calibration layer;
- post-hoc glue of separately sampled decks presented as one conditional law.

If the best IV-only, anchor-only, and joint results come from different recipes,
they must be labeled as task-specialized frontiers, not a validated framework.
The next iteration must then be a framework-lock experiment or a
post-experiment analysis explaining why one frozen recipe trades off across
scopes.

## General Acceptance Scorecard

For general conditional scenario-generator claims, the IV `11/11` suite is
necessary but not sufficient. Every deployable/general framework claim must run
the 712a acceptance scorecard:

- IV gate: existing IV full 11-suite, with the old conditionality failure
  replaceable by the population risk-state allocation diagnostic when that
  diagnostic passes;
- anchor gate: generic `anchor_only` panel realism, tail scale, dependence, and
  population conditional response;
- joint gate: native `joint` panel quality plus IV-factor co-movement;
- framework gate: one frozen core recipe across `iv_only`, `anchor_only`, and
  `joint`, with only allowed data-interface differences.

The scorecard output is the source of truth for statements like "general
deployable conditional law." IV-only stress prototypes must be labeled as such
when they do not pass the scorecard.

## Incumbent, Frontier, And Promotion Discipline

The loop must maintain separate labels so research branches cannot overwrite a
deployable model by accident:

- `deployable_tri_scope_incumbent`: best risk-manager-presentable framework
  evaluated on `iv_only`, `anchor_only`, and `joint`. Current real-VIX
  incumbent: `734a/739a`.
- `iv_research_frontier`: best IV-only research result. Current frontier:
  `755a`, because it improves train-tail strict-suite behavior but is not yet a
  validated tri-scope framework.
- `diagnostic_branch`: a run that explains a failure mechanism but cannot be
  promoted.
- `rejected_branch`: a run that violates hard non-regression gates.

Promotion to deployable incumbent requires tri-scope evidence. An IV-only run
may become the IV research frontier, but it cannot replace the deployable
tri-scope incumbent until the same recipe has been evaluated on all three
scopes and passes non-regression against `734a/739a`.

## Tri-Scope Non-Regression Gate

Every serious candidate model must run:

- `iv_only`;
- `anchor_only`;
- `joint`.

Single-scope experiments are allowed only as diagnostics. They must be labeled
as non-promotable and compared against the appropriate incumbent/frontier.

For deployable promotion, the candidate must not regress the risk-manager
properties that made the incumbent usable:

- IV mean reversion must remain acceptable. This is a hard gate.
- IV scenario realism must remain acceptable: surface validity, time-series
  realism, cross-cell dependence, daily-change and level distributions,
  pathwise jump realism, and population risk-state uncertainty allocation.
- Anchor-only realism must not regress: finite paths, factor daily-change
  realism, factor tail scale, factor-factor dependence, and conditional panel
  response.
- Native joint quality must not regress: IV slice realism, IV-factor
  co-movement, factor-factor dependence, and conditional panel response.
- Sticky low-activity channels such as OAS may remain monitored limitations
  unless the current iteration explicitly tests a generic support-aware
  observation adapter.

Strict conditional-law work is still allowed, but it cannot sacrifice
risk-manager deployability. If a proper-score, interval-score, temperature, or
likelihood-oriented repair improves one strict metric while losing IV mean
reversion or scenario realism, it is diagnostic or rejected, not promoted.

Old path-prediction conditionality is replaced by population risk-state
uncertainty allocation when that diagnostic passes. Old IV-EWMA cointegration
has no formal replacement yet; it remains a monitor unless a new
cointegration/dependence gate is explicitly defined. Do not rename cross-cell
correlation or IV-factor co-movement as "new cointegration" without defining
the gate.

## Trade-Off Attribution Gate

Before adding a new knob after a tri-scope mismatch, the loop must explain why
one frozen recipe fails differently on `iv_only`, `anchor_only`, and `joint`.
The report should compare:

- train versus validation behavior for each scope;
- marginal realism, path realism, conditionality, diversity, and dependency
  failures for each scope;
- whether the issue is caused by the data object, objective balancing, shared
  core capacity, input/output head interference, or distribution shift;
- whether a proposed change is a universal mechanism or a task-specific patch.

Only after this attribution can a new model change be made. If the attribution
is not clear, choose `post_experiment_analysis` or `research_ideation` rather
than another experiment.

## Iteration Report Template

Each autoresearch result should include:

1. Hypothesis.
2. Framework ID and Single-Framework Candidate Gate status.
3. Active research axis and axes held fixed.
4. Methodology and knobs added.
5. Required audits run.
6. Result summary.
7. Failure classification.
8. Trade-off attribution across scopes when applicable.
9. Literature searched, if required by the Literature Search Gate.
10. Minimal next fix.
11. Explicit statement: continue current methodology or justify switch.

## Switch Bar

A methodology switch is allowed only if:

- required audits have been run or their absence is justified;
- failure classification points to the component being switched;
- at least one minimal fix has been attempted or ruled out;
- literature search has been done when required by the Literature Search Gate;
- the next methodology is stated as a response to the diagnosis, not as a new
  guess.
