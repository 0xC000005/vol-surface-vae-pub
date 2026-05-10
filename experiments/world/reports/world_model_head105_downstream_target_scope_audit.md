# World Model HEAD105: Downstream Target-Scope Audit

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Downstream-probe evidence audit.

## Hypothesis

The package should distinguish the representation input scope from the
downstream target scope: HEAD070 encodes the 58-token geometry panel, but HEAD085
probes evaluate IV-surface futures only.

## Falsifier

The audit fails if the package implies factor-panel future targets were
evaluated by the HEAD085 downstream probes.

## Evidence

- `build_masked_multiview_batch` builds a 58-token panel including IV surface,
  vol side channels, factor levels, and factor returns.
- `masked_multiview_downstream_probe_audit.py` builds targets through
  `build_iv_world_windows`.
- `build_iv_world_windows` uses `load_iv_surface_flat`, producing normalized
  25-cell IV-surface past/future windows.
- `make_extended_future_targets` operates on `past_surface` and
  `future_surface`.

## Interpretation

HEAD085 probes evaluate whether the frozen representation helps predict or
classify IV-surface future summaries and regime labels. They do not evaluate
future factor-panel levels/returns, cross-asset factor futures, or factor-panel
scenario targets.

## Package Update

- Added `target_scope:
  iv_surface_future_targets_only_no_factor_future_targets` to the manifest
  downstream caveat.
- Added the same caveat to the package summary.

## Decision

Keep the downstream claim scoped to IV-surface future targets. Multi-factor
future targets remain future downstream-probe work, not an established HEAD070
result.
