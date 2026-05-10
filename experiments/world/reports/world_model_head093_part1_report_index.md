# World Model HEAD093: Part 1 Report Index

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Navigation guardrail for `masked_multiview_invariance`.

## Purpose

This index identifies the reports that matter for resuming the current HEAD070
masked-multiview Part 1 package without reading the full historical sequence.
The authoritative source-report list is
`experiments/world/part1_jepa_latent/reference_manifest.json`; this file is a
human-oriented resume guide, not the source of truth.

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
- HEAD097: `world_model_head097_mask_policy_coverage_audit.md` states that
  HEAD070 is validated only for the default six mask families and treats
  sparse/wing/ATM/whole-surface/cross-family stress masks as future diagnostics.
- HEAD100: `world_model_head100_sample_scale_caveat.md` states that HEAD070 is
  a smoke-scale reference candidate, not a full-data convergence result.
- HEAD102: `world_model_head102_downstream_probe_reporting_audit.md` states
  that raw-surface-flat beats Barlow on max-absolute-step MSE, so downstream
  utility must not be overclaimed against all raw baselines.
- HEAD105: `world_model_head105_downstream_target_scope_audit.md` states that
  downstream probes target IV-surface futures only, not factor-panel future
  targets.

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
- HEAD094: `world_model_head094_manual_stop_guardrail.md` removes the vague
  discretionary pause path.
- HEAD095: `world_model_head095_target_stage_guardrail.md` prevents target-stage
  completion from stopping manual-stop mode unless `goal_reached` is explicit.
- HEAD096: `world_model_head096_stop_condition_verification.md` verifies the
  current hard-stop list.
- HEAD098: `world_model_head098_manifest_source_report_update.md` makes the
  package checker enforce the HEAD097 caveat report.
- HEAD101: `world_model_head101_manifest_sample_scale_source.md` makes the
  package checker enforce the HEAD100 sample-scale report.
- HEAD103: `world_model_head103_manifest_downstream_caveat_source.md` makes the
  package checker enforce the HEAD102 downstream caveat report.
- HEAD106: `world_model_head106_manifest_target_scope_source.md` makes the
  package checker enforce the HEAD105 target-scope report.
- HEAD107: `world_model_head107_manual_stop_runtime_guardrail.md` removes
  elapsed-time, turn-count, fatigue, diminishing-returns, and process-only-work
  pause interpretations from manual-stop mode.
- HEAD108: `world_model_head108_restart_checklist_caveat_sync.md` syncs the
  restart checklist with the current caveat boundary.
- HEAD109: `world_model_head109_active_readme_caveat_audit.md` syncs active
  README entry points with the current caveat boundary.
- HEAD110: `world_model_head110_package_checker_guardrail_docs.md` expands the
  package checker to enforce guardrail-doc caveat terms.

## Fast Resume Command

```bash
python experiments/world/part1_jepa_latent/reference_package_check.py
```

Then read:

1. `experiments/world/part1_jepa_latent/reference_manifest.json`
2. `experiments/world/part1_jepa_latent/restart_checklist.md`
3. `experiments/world/reports/world_model_head089_part1_open_risk_ledger.md`
4. `experiments/world/reports/world_model_head086_downstream_probe_interpretation.md`
5. `experiments/world/reports/world_model_head097_mask_policy_coverage_audit.md`
6. `experiments/world/reports/world_model_head096_stop_condition_verification.md`
7. `experiments/world/reports/world_model_head100_sample_scale_caveat.md`
8. `experiments/world/reports/world_model_head102_downstream_probe_reporting_audit.md`
9. `experiments/world/reports/world_model_head105_downstream_target_scope_audit.md`
10. `experiments/world/reports/world_model_head107_manual_stop_runtime_guardrail.md`
11. `experiments/world/reports/world_model_head110_package_checker_guardrail_docs.md`

## Decision

This index is a navigation artifact only. It does not change the model,
objective, or acceptance boundary.
