# 712a General Conditional Scenario Acceptance Scorecard

## Purpose

The IV `11/11` suite remains useful, but it is not sufficient for the long-term
objective: a generic multivariate financial-factor conditional scenario
generator. This scorecard prevents two mistakes:

- treating IV-only success as joint-factor deployability;
- treating the old path-prediction conditionality gate as the only valid
  conditionality target when the validation data do not show that signal.

## Acceptance Gates

### IV Gate

Use the existing IV full 11-suite. The old `conditionality` failure may be
replaced by the population risk-state allocation diagnostic when that diagnostic
passes. Non-conditional failures are never hidden.

The IV gate fails if coverage, regime coverage, path realism, distributional
fidelity, mean reversion, surface validity, or dependence suites fail.

### Anchor Gate

Use the generic panel audit on `anchor_only` outputs. The anchor gate checks:

- finite generated samples;
- factor daily-change KS;
- factor tail scale;
- factor-factor dependence shape and magnitude;
- population conditional panel response.

This gate is intentionally factor-name agnostic. It reads only generic panel
metrics, not SPX-specific or credit-specific rules.

### Joint Gate

Use the generic panel audit on native `joint` outputs. The joint gate includes
the anchor gate plus IV-factor co-movement. It is not satisfied by post-hoc
pairing of independently sampled IV and factor decks.

### Framework Gate

A candidate is one framework only if `iv_only`, `anchor_only`, and `joint` share
one frozen core recipe. The frozen fields are:

- generated coordinate and normalization family;
- temporal factorization;
- backend and stochastic source;
- shared core architecture;
- scalar loss terms, loss weights, sampler, and training protocol.

Allowed differences are only data-interface differences:

- input/output dimensionality;
- support transform;
- input and decoder heads;
- deterministic group balancing from the same formula.

Disallowed differences include task-specific loss recipes, scope-specific
backend/sampler/calibration, and glued separately sampled decks.

## Interpretation

`overall_pass = IV gate AND anchor gate AND joint gate AND framework gate`.

The scorecard is stricter than risk-manager prototype readiness. A model may be
useful as an IV stress prototype while failing this scorecard. It should not be
described as a deployable generic conditional law until this scorecard passes.

## Current Implication

The active normalized-innovation family remains the main research path because
it is closest to the generic objective. The next research move should repair
lower-tail/regime under-inclusion inside the same frozen framework rather than
switching back to an IV-specific control.
