# World Model HEAD101: Manifest Sample-Scale Source

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Package consistency for `masked_multiview_invariance`.

## Hypothesis

If HEAD100 adds a sample-scale caveat to the manifest, the manifest should list
the HEAD100 report as a source report so the package checker enforces that
evidence file.

## Falsifier

The iteration fails if `reference_manifest.json` carries `sample_scale_caveat`
but does not list `world_model_head100_sample_scale_caveat.md`.

## Execution

- Added `experiments/world/reports/world_model_head100_sample_scale_caveat.md`
  to `reference_manifest.json` source reports.
- Re-ran the reference package checker.

## Result

`reference_package_check.py` now verifies `14` source reports and `7`
artifacts.

## Decision

The package checker now covers the sample-scale caveat evidence.
