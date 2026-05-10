# World Model HEAD106: Manifest Target-Scope Source

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Package consistency for `masked_multiview_invariance`.

## Hypothesis

If HEAD105 adds a downstream target-scope caveat, the manifest should list the
HEAD105 report as a source report.

## Falsifier

The iteration fails if `reference_manifest.json` carries the IV-surface-only
downstream target caveat but does not list the HEAD105 report.

## Execution

- Added `experiments/world/reports/world_model_head105_downstream_target_scope_audit.md`
  to `reference_manifest.json` source reports.
- Re-ran the reference package checker.

## Result

`reference_package_check.py` now verifies `16` source reports and `7`
artifacts.

## Decision

The package checker now covers the downstream target-scope caveat evidence.
