# Grounding Reliability Audit Intake

Date: 2026-05-28

Workflow: `nl-prefix-latent-autoresearch`

Research lane: `production_demo`

Result status before run: `candidate`

Benchmark floor status before run: `not_applicable`

## Objective

Measure whether the narrative grounding sidecar is reliable enough to be a
paper/demo trust surface. The audit should test whether the LLM correctly
extracts current/recent market claims from professional risk-manager narratives,
keeps forward-looking language out of conditioning, and agrees with historical
market motion when the narrative is generated from a known prefix.

This is not a replacement for scenario-level backtesting. It is a separate
interpretation-layer audit. The scenario backtest asks whether generated paths
are useful. This audit asks whether the visible story-to-claims translation is
faithful and directionally usable.

## Method Story

The production pipeline uses the full narrative for support selection, while the
grounding sidecar provides checkable current/recent market implications,
warnings, unsupported claims, and future-language filters. Because grounding is
visible in the app and influences support checks, it should be measured like a
product contract rather than treated as an unverified LLM output.

The audit should score four surfaces:

1. **Narrative faithfulness.** Each extracted claim must be backed by an
   evidence phrase from the narrative.
2. **Temporal discipline.** Forward-looking or desired-future language must be
   warning-only and excluded from conditioning claims.
3. **Historical direction agreement.** For professional captions generated from
   known historical 30-day prefixes, extracted directions should agree with
   realized prefix motion in mapped factors such as SPX, VIX, credit spreads,
   rates, dollar, oil, gold, and IV.
4. **Support-direction consistency.** Selected historical support regimes and
   the top3/90 posterior support pool should satisfy high-confidence grounding
   directions, or emit visible warnings when they do not.

## Acceptance Metrics

Report these metrics overall and by narrative family:

- claim faithfulness rate;
- unsupported-claim rate;
- future-language detection rate;
- future-language leakage rate into conditioning claims;
- historical direction agreement rate;
- high-confidence direction agreement rate;
- support-direction mismatch rate;
- warning coverage for mismatches;
- qualitative examples of correct extraction, warning-only future language, and
  failed/ambiguous grounding.

## First TestFlight

Use a small professional-narrative set before scaling:

- the six default professional casebook narratives;
- a matched sample of known historical-prefix captions from the professional
  corpus;
- at least two hand-written hard cases with explicit future-looking language.

The TestFlight should write a JSON report and a compact markdown summary under
`experiments/backfill/block_ar/nl_scenario_demo_outputs/`.

## Scale Plan

If the TestFlight passes schema and sanity checks, scale to the full available
professional-caption corpus. Use cached grounding where available and call the
LLM only for missing current-format grounding outputs. The scaled report should
produce a paper-ready table and a short appendix example set.

## Promotion Rule

Grounding can be claimed as a measured audit layer only if:

- schema validation passes;
- future-looking language leakage is low and examples are documented;
- high-confidence current/recent directions agree with known historical motion
  at a materially high rate;
- mismatches are visible as warnings rather than silently entering the
  conditioning contract.

The audit does not need to beat a CRPS or energy-score floor, because it is not
a scenario-generation method. It must, however, avoid weakening the default
scenario workflow or hiding warnings.

## Kill Conditions

Stop and redesign the grounding prompt or validation if:

- future-looking language frequently enters conditioning claims;
- many high-confidence claims have no supporting evidence phrase;
- historical direction agreement is weak for high-confidence claims;
- support-direction mismatches are silent in the report or demo.

