# World Model HEAD093: Part 1 Report Index

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Objective Family

Navigation guardrail for `masked_multiview_invariance`.

## Purpose

This index identifies the reports that matter for resuming the current HEAD070
masked-multiview Part 1 package without reading the full historical sequence.

## Core Protocol And Data Object

- HEAD063: `world_model_head063_geometry_masked_multiview_protocol.md`
  establishes the geometry-aware masked-multiview protocol.
- HEAD064: `world_model_head064_masked_multiview_data_builder.md` establishes
  the masked-view data builder.
- HEAD065: `world_model_head065_masked_multiview_diagnostics.md` establishes
  initial same-state diagnostics.

## Reference Candidate

- HEAD068: `world_model_head068_direct_barlow_smoke.md` shows the first direct
  Barlow same-state branch but has low-rank caveats.
- HEAD070: `world_model_head070_canonical_barlow_scaling.md` is the current
  checkpoint source for `masked_multiview_barlow_head070.pt`.
- HEAD071: `world_model_head071_canonical_barlow_reference_decision.md`
  records the first reference decision.

## Validation And Caveats

- HEAD080: `world_model_head080_part1_scorecard.md` consolidates candidate
  scorecards.
- HEAD082: `world_model_head082_part1_scorecard_health.md` adds variance,
  singular spectrum, and health-offdiag checks.
- HEAD083: `world_model_head083_mask_artifact_audit.md` audits synthetic-mask
  family leakage.
- HEAD084: `world_model_head084_stratified_mask_audit.md` audits mask-family
  strata.
- HEAD085: `world_model_head085_downstream_probe_coverage.md` expands frozen
  downstream probes.
- HEAD086: `world_model_head086_downstream_probe_interpretation.md` states the
  mixed downstream-probe caveat boundary.

## Packaging And Restart Guardrails

- HEAD087: `world_model_head087_part1_readiness_manifest.md` packages the
  HEAD070 reference candidate and updates manifest/digests/checklist files.
- HEAD088: `world_model_head088_stale_reference_guardrail.md` removes stale
  wording hazards.
- HEAD089: `world_model_head089_part1_open_risk_ledger.md` refreshes the
  current open-risk ledger.
- HEAD090: `world_model_head090_consistency_reconciliation.md` verifies package
  consistency.
- HEAD091: `world_model_head091_reference_package_checker.md` adds the reusable
  package checker.
- HEAD092: `world_model_head092_goal_state_reconciliation.md` reconciles local
  ignored goal/state files.

## Fast Resume Command

```bash
python experiments/world/part1_jepa_latent/reference_package_check.py
```

Then read:

1. `experiments/world/part1_jepa_latent/reference_manifest.json`
2. `experiments/world/part1_jepa_latent/restart_checklist.md`
3. `experiments/world/reports/world_model_head089_part1_open_risk_ledger.md`
4. `experiments/world/reports/world_model_head086_downstream_probe_interpretation.md`

## Decision

This index is a navigation artifact only. It does not change the model,
objective, or acceptance boundary.
