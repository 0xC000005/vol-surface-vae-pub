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

## Iteration Report Template

Each autoresearch result should include:

1. Hypothesis.
2. Active research axis and axes held fixed.
3. Methodology and knobs added.
4. Required audits run.
5. Result summary.
6. Failure classification.
7. Literature searched, if required by the Literature Search Gate.
8. Minimal next fix.
9. Explicit statement: continue current methodology or justify switch.

## Switch Bar

A methodology switch is allowed only if:

- required audits have been run or their absence is justified;
- failure classification points to the component being switched;
- at least one minimal fix has been attempted or ruled out;
- literature search has been done when required by the Literature Search Gate;
- the next methodology is stated as a response to the diagnosis, not as a new
  guess.
