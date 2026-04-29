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

The old IV-EWMA `cointegration` suite is not a replacement for cross-market or
surface dependency quality. It remains a monitoring diagnostic until a formal
new cointegration/dependence gate is defined.

The IV gate fails if coverage, regime coverage, path realism, distributional
fidelity, mean reversion, surface validity, or dependency/co-movement suites
fail. For the current scorecard, the dependency suite is cross-cell correlation
and effective-rank preservation, not the old IV-EWMA cointegration monitor.

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

## Non-Regression Policy

The scorecard is also an incumbent-protection tool. A new candidate is not
promotable if it loses properties that made the current incumbent usable:

- IV mean reversion is a hard non-regression gate.
- IV scenario realism must remain acceptable: surface validity, time-series
  realism, cross-cell dependency, daily-change and level distributions,
  pathwise jump realism, and population risk-state allocation.
- Anchor-only and native-joint panel realism must be checked together with IV
  so improvements in one scope do not hide regressions in another.

Single-scope experiments may inform research, but they cannot establish a
general model. Promotion requires `iv_only`, `anchor_only`, and `joint`
evidence under one framework recipe with only allowed data-interface adapters.

## Current Implication

The active normalized-innovation family remains the main research path because
it is closest to the generic objective. The next research move should repair
lower-tail/regime under-inclusion inside the same frozen framework rather than
switching back to an IV-specific control.
