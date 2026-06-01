# Narrative-Conditioned Ensemble Calibration Intake

Date: 2026-05-25

Status: planned HEAD objective.

## Method Story

The incumbent narrative-conditioned system already has the right financial
backbone: a professional current-market narrative plus an approved starting
level selects a diverse, non-overlapping support mixture, and the frozen SNI
generator rolls out pathwise 30-day scenarios from that on-manifold support.
The remaining bottleneck is not support provenance. The bottleneck is that the
visible raw-level fan geometry is still dominated by the approved starting
level, while the narrative effect is clearer only after start normalization and
portfolio/factor diagnostics.

The next method therefore keeps the incumbent support-grounded SNI ensemble as
the base distribution and trains a small narrative/start-conditioned
calibration layer over the generated ensemble. The calibrator should adjust the
distribution in narrative-relevant channels, such as location, dispersion,
tail width, skew, cross-factor dependence, or portfolio-book exposure, while
remaining bounded around the frozen generator output.

This is closer to ensemble post-processing than to a new generator. The frozen
SNI generator remains responsible for realistic multivariate time-series
structure. The new layer learns how the professional narrative should tilt the
already-generated ensemble.

## Related-Work Basis

- Ensemble model output statistics and ensemble post-processing motivate
  calibrating a generated ensemble rather than replacing the physical or
  statistical simulator.
- Adapter and ControlNet-style methods motivate freezing a strong backbone and
  learning a small control branch instead of retraining the whole generator.
- Conditional normalizing-flow or residual-prior methods motivate a later
  bounded residual distribution if the first calibration TestFlight shows
  genuine signal.

## Why This Replaces Another Ranker-Only Probe

Prior response-aware support-weighting probes showed useful mechanism evidence,
but ranker-only changes repeatedly risked either smoothing away response or
overfitting one fixed-start diagnostic. The new objective asks a sharper
question:

```text
Given the same approved start and the same incumbent support-grounded SNI
ensemble, can a bounded narrative-conditioned calibrator increase
start-normalized narrative attribution and portfolio-tail separation while
preserving CRPS, energy, coverage, and auditability?
```

If yes, this gives the product a stronger narrative response without abandoning
the support mixture. If no, the limitation belongs to the frozen generator or
the support bank, not merely to the support scorer.

## Research Log Recap And Mistake Gates

Before launching this branch, the research log was checked for the previous
response-aware and fixed-start conditionality work. The important lessons are:

- 934a showed that cached/offline response reweighting can improve fixed-start
  conditionality diagnostics, but cached-rollout evidence is not enough for
  promotion.
- 934b showed that a train-only risk-book guard can over-broaden support and
  smooth away narrative response.
- 935/937 showed that live or larger response-preview gates can lose the small
  TestFlight gains, collide with direction checks, or underperform the
  incumbent on relevant-factor/path/portfolio metrics.
- 942d rejected response-preview methods as a stable default and restored the
  hard-direction component-preserving incumbent.
- 943a/942f verified that the incumbent beats a start-only null and remains the
  paper/demo default.
- 944a showed that the accepted start dominates raw-level geometry, while
  narrative is material but not dominant after start normalization.

The new branch must therefore pass these gates before promotion:

- no cached-only promotion;
- no ranker-only variant unless it is part of bounded ensemble calibration;
- no single-metric promotion from VaR range, target cosine, or retrieval rank;
- no method that makes the start-only null nonzero;
- no temperature, blend, or threshold sweep after a scaled rejection;
- require start-normalized attribution plus fixed-start qualitative plots before
  any paper/demo claim.

## Candidate Inputs

- Full professional narrative embedding and structured sidecar fields.
- Approved start state, preferably in normalized joint39 coordinates.
- Incumbent support weights and support diagnostics.
- Base generated ensemble summaries by factor, horizon, and portfolio book.
- Direction and warning checks from the grounding sidecar.

## Candidate Outputs

Start with bounded, interpretable calibration parameters:

- factor-group location tilt on normalized future deltas;
- factor-group dispersion/tail-width adjustment;
- optional rank-preserving quantile tilt for skew;
- optional low-rank cross-factor covariance adjustment;
- portfolio-book tail tilt for narrative-relevant risk books.

The first TestFlight should avoid a large neural architecture. A small MLP or
linear/monotone calibrator is enough to prove whether the signal exists.

## TestFlight Plan

1. Use the existing crossed fixed-start evidence as the baseline:
   `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_narrative_attribution_944a/start_narrative_attribution.json`.
2. Build an identity baseline equal to the incumbent support-grounded SNI
   ensemble.
3. Train or fit a small cross-fitted calibration model only on training
   historical backtest windows. Do not use realized future paths from the
   evaluated case.
4. Evaluate on held-out historical windows and on the six professional
   fixed-start narratives across starts 18, 22, and 40.
5. Compare against the incumbent and the start-only null.

## Promotion Metrics

The candidate can become a promotion candidate only if it improves narrative
conditionality without damaging distributional quality:

- start-normalized narrative plus interaction attribution share improves by at
  least five percentage points versus the 944a baseline, or reaches at least
  50%;
- fixed-start narrative factor KS and portfolio KS improve versus the incumbent
  while the start-only null remains flat;
- narrative-relevant factor panels and portfolio-tail plots show visible,
  explainable differences in the channels named by the narrative;
- held-out CRPS and energy remain within 2% of the incumbent or improve;
- 80% coverage does not regress by more than 0.03 absolute;
- direction checks, provenance, and warning/audit outputs remain intact.

## Kill Conditions

Stop or mark diagnostic if any of these occur:

- start-normalized narrative attribution does not improve over the incumbent;
- fixed-start factor/portfolio separation improves only by distorting coverage
  or energy;
- calibration moves paths in directions contradicted by the grounding checks;
- the method requires future-outcome language or realized future leakage;
- the added layer cannot be explained as bounded ensemble calibration over the
  frozen SNI support-grounded generator.

## First Implementation Target

Implement a small offline TestFlight before any paper/demo promotion:

```text
incumbent generated ensemble
+ narrative/start/support summary features
-> bounded calibration parameters
-> calibrated path ensemble
-> attribution, KS, CRPS, energy, coverage, portfolio-tail diagnostics
```

Do not change the production demo default until this passes the promotion
metrics and independent verification.
