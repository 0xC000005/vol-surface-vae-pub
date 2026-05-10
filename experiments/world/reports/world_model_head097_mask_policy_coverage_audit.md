# World Model HEAD097: Mask-Policy Coverage Audit

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Coverage audit for `masked_multiview_invariance`.

## Hypothesis

The HEAD070 package should state exactly which structured mask families were
trained/evaluated and which protocol ideas remain unvalidated coverage caveats.

## Falsifier

The audit fails if the package implies HEAD070 covers all mask families proposed
in HEAD063, including wing, ATM-strip, whole-surface day dropout, or
cross-family stress masks.

## Evidence

HEAD063 proposed a broad protocol, including:

- moneyness and maturity bands;
- local rectangles;
- wing masks;
- ATM-strip masks;
- whole-surface dropout on selected days;
- factor-family masks;
- sparse factor/day dropout;
- cross-family stress masks.

The implemented/default HEAD070 reference uses:

- `surface_maturity`;
- `surface_moneyness`;
- `surface_rectangle`;
- `vol_side_channel`;
- `factor_family`;
- `time_block`.

The code also implements `sparse`, but HEAD070 did not use it in the default
training/evaluation mask family tuple.

HEAD084 stratified the default six families and found no large stratified
failure, with minimum top10 retrieval about `0.826`.

## Decision

This is a coverage caveat, not a model failure. HEAD070 should be claimed as
validated only for the default six mask families above. Wing, ATM-strip,
whole-surface day dropout, sparse, and cross-family stress masks should remain
future diagnostics unless explicitly tested.

## Package Update

- Added `mask_policy_coverage` to
  `experiments/world/part1_jepa_latent/reference_manifest.json`.
- Added the same caveat to
  `experiments/world/part1_jepa_latent/package_summary.md`.

## Verification

- Read `experiments/world/evaluation/masked_multiview_data.py`.
- Read HEAD063 and HEAD084 reports.
- Parsed `results/world/masked_multiview_barlow_head070.json` visibility and
  mask-family counts.
