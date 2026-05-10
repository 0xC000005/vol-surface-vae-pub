# World Model HEAD098: Manifest Source-Report Update

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Package consistency for `masked_multiview_invariance`.

## Hypothesis

If HEAD097 adds a new manifest caveat, the reference manifest should list the
HEAD097 report as a source report so `reference_package_check.py` enforces that
the evidence file exists.

## Falsifier

The iteration fails if the manifest omits the HEAD097 mask-policy coverage
audit while carrying the new `mask_policy_coverage` field.

## Execution

- Added `experiments/world/reports/world_model_head097_mask_policy_coverage_audit.md`
  to `reference_manifest.json` source reports.
- Re-ran the reference package checker.

## Result

`reference_package_check.py` now verifies `13` source reports and `7`
artifacts.

## Decision

The package checker now covers the new mask-policy coverage caveat.
