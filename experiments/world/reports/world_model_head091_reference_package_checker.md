# World Model HEAD091: Reference Package Checker

Date: 2026-05-09

## Iteration Type

`experiment`

## Objective Family

Workflow/provenance guardrail for `masked_multiview_invariance`.

## Hypothesis

The HEAD090 one-off consistency checks should be reusable as a small local
checker so future runs can verify the Part 1 package before consuming ignored
checkpoint/result artifacts.

## Falsifier

The iteration fails if the checker cannot detect manifest source-report paths
and digest entries through a test, or if it fails on the current HEAD070
package.

## Execution

- Added `experiments/world/part1_jepa_latent/reference_package_check.py`.
- Added a focused unit test for report-path and digest validation.
- Added the checker command to the Part 1 README/restart checklist.

## Result

The checker validates the current package:

```json
{
  "ok": true,
  "missing_reports": [],
  "artifact_mismatches": [],
  "checked_reports": 12,
  "checked_artifacts": 7
}
```

## Verification

- First ran the focused test red; it failed because
  `reference_package_check` did not exist.
- `uv run pytest test_code/test_world_model_evaluation.py::test_check_reference_package_validates_reports_and_digest -q`
  passed.
- `python experiments/world/part1_jepa_latent/reference_package_check.py`
  passed.
- `uv run pytest test_code/test_world_model_evaluation.py -q` passed:
  `50 passed`.
- `git diff --check` passed.

## Decision

The package now has a reusable local consistency check. This does not justify a
new model knob or decoder work; it only lowers restart/provenance risk.
