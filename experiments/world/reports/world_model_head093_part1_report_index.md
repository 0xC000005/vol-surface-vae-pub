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
- HEAD118: `world_model_head118_part1_literature_quality_gate.md` adds the
  literature-aligned Part 1 quality gate and blocks Part B until it passes.
- HEAD119: `world_model_head119_part1_quality_gate_assessment.md` runs the
  gate and returns `DO_NOT_PROMOTE`: representation health passes at smoke
  scale, but the full Part 1 quality gate does not.
- HEAD120: `world_model_head120_part1_failure_analysis.md` decomposes the
  failure: it is not collapse, but a baseline-superiority, probe-coverage, and
  scale/stability failure.
- HEAD121: `world_model_head121_jepa_fit_diagnosis.md` compares HEAD070 against
  JEPA literature and local mask difficulty; default validation masks hide only
  about `6.6-7.8%` per view and keep about `86.6%` visible in both views.
- HEAD122: `world_model_head122_hard_mask_preset.md` defines one hard-mask
  diagnostic preset that lowers validation both-visible overlap to about
  `60.9%` without starting a knob sweep.
- HEAD123: `world_model_head123_hard_mask_smoke.md` trains the same
  encoder/loss on that hard-mask preset and finds that mask aggression alone
  does not close the simple-baseline gap: standalone Barlow wins fall to `0/5`
  IV future targets versus best raw surface baselines.
- HEAD124: `world_model_head124_present_state_probe.md` audits frozen
  present-state information. It finds real factor-return signal, but also shows
  that the embedding loses exact IV/current-state geometry that simple raw
  baselines preserve.
- HEAD125: `world_model_head125_state_content_gate.md` turns HEAD124 into an
  explicit gate. Non-surface signal passes, but exact IV retention,
  factor-level fidelity, and hard-mask regression fail.
- HEAD126: `world_model_head126_grouped_geometry_state_probe.md` tests a
  grouped-geometry encoder. It improves retrieval but degrades rank and
  present-state probes, so higher retrieval is not enough for promotion.
- HEAD127: `world_model_head127_scale_state_probe.md` scales the HEAD070-style
  flat encoder to `1024` train windows and `256` validation windows. Rank,
  retrieval, and state-content probes improve, making it the next Part 1
  quality-gate candidate, but exact IV retention still loses to raw surface.
- HEAD128: `world_model_head128_scale_downstream_quality.md` audits HEAD127 on
  frozen downstream probes. Scale helps, but standalone Barlow still wins only
  `2/5` IV future targets and regime accuracy remains below majority.
- HEAD129: `world_model_head129_scale_mask_artifact_audit.md` and
  `world_model_head129_scale_stratified_mask_audit.md` audit the scaled HEAD127
  checkpoint for corruption robustness. They find no large mask-family leakage
  and no large stratified mask-family failure.
- HEAD130: `world_model_head130_scale_part1_quality_gate.md` runs the scaled
  candidate through the formal Part 1 gate. Representation health and
  corruption robustness pass, but state content and scale/stability remain
  partial while baseline superiority and market-state regime probes fail, so
  the decision remains `DO_NOT_PROMOTE`.
- HEAD131: `world_model_head131_scale_seed_stability.md` repeats the scaled
  flat Barlow setup for seeds `681` and `682`, compares them to HEAD127 seed
  `680`, and finds smoke-scale representation-health stability. This improves
  scale/stability evidence for representation health only; Part 1 still is not
  promoted.
- HEAD132: `world_model_head132_scale_exact_state_gap.md` decomposes the
  remaining present-state blocker. The scaled embedding is worse than raw
  last-surface features on `20/25` IV cells and has `2.44x` the raw IV
  reconstruction MSE, so the next blocker is exact-state retention and baseline
  certification rather than collapse or seed instability.
- HEAD133: `world_model_head133_scale_regime_probe_gap.md` decomposes the
  regime-probe blocker. The scaled embedding still fails majority accuracy, but
  it has better macro recall and rare class-4 recall than raw surface features,
  so the regime failure is partly an imbalanced-label diagnostic issue rather
  than pure no-signal.
- HEAD134: `world_model_head134_scale_baseline_target_taxonomy.md` decomposes
  baseline superiority by downstream target family. Scaled Barlow wins the
  path-shape/risk-width targets and often adds to raw-last features, but loses
  persistence/exact-state targets, so the blocker is target-family coverage.
- HEAD135: `world_model_head135_exact_state_retention_literature_gate.md`
  applies the literature gate before any exact-state objective change. It says
  to audit the frozen representation surface first, then consider same-window
  context-to-target JEPA if needed; do not add an ad hoc value-reconstruction
  auxiliary loss as the next default move.
- HEAD136: `world_model_head136_scale_representation_surface.md` runs that
  frozen representation-surface audit. Last-state scaled embeddings remain best
  for IV, side-channel, factor-level, and all-geometry probes, so the
  exact-state blocker is not solved by mean, last+mean, or flattened per-time
  readouts.

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
  discretionary stop path.
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
  stop interpretations from manual-stop mode.
- HEAD108: `world_model_head108_restart_checklist_caveat_sync.md` syncs the
  restart checklist with the current caveat boundary.
- HEAD109: `world_model_head109_active_readme_caveat_audit.md` syncs active
  README entry points with the current caveat boundary.
- HEAD110: `world_model_head110_package_checker_guardrail_docs.md` expands the
  package checker to enforce guardrail-doc caveat terms.
- HEAD112: `world_model_head112_open_risk_ledger_refresh.md` refreshes the
  open-risk ledger with current caveats and manual-stop constraints.
- HEAD113: `world_model_head113_stale_stop_wording_audit.md` audits active
  resume documents for stale discretionary-stop wording.
- HEAD114: `world_model_head114_score_summary_caveat_sync.md` syncs generated
  Part 1 score-summary caveats.
- HEAD115: `world_model_head115_package_checker_regression_test.md` adds focused
  pytest coverage for guardrail-doc package checking.
- HEAD116: `world_model_head116_goal_checker_manual_stop.md` documents the
  local goal-checker manual-stop guardrail.

## Fast Resume Command

```bash
python experiments/world/part1_jepa_latent/reference_package_check.py
```

Then read:

1. `experiments/world/part1_jepa_latent/reference_manifest.json`
2. `experiments/world/part1_jepa_latent/restart_checklist.md`
3. `experiments/world/part1_jepa_latent/part1_quality_gate.md`
4. `experiments/world/reports/world_model_head118_part1_literature_quality_gate.md`
5. `experiments/world/reports/world_model_head119_part1_quality_gate_assessment.md`
6. `experiments/world/reports/world_model_head120_part1_failure_analysis.md`
7. `experiments/world/reports/world_model_head121_jepa_fit_diagnosis.md`
8. `experiments/world/reports/world_model_head122_hard_mask_preset.md`
9. `experiments/world/reports/world_model_head123_hard_mask_smoke.md`
10. `experiments/world/reports/world_model_head124_present_state_probe.md`
11. `experiments/world/reports/world_model_head125_state_content_gate.md`
12. `experiments/world/reports/world_model_head126_grouped_geometry_state_probe.md`
13. `experiments/world/reports/world_model_head127_scale_state_probe.md`
14. `experiments/world/reports/world_model_head128_scale_downstream_quality.md`
15. `experiments/world/reports/world_model_head129_scale_mask_artifact_audit.md`
16. `experiments/world/reports/world_model_head129_scale_stratified_mask_audit.md`
17. `experiments/world/reports/world_model_head130_scale_part1_quality_gate.md`
18. `experiments/world/reports/world_model_head131_scale_seed_stability.md`
19. `experiments/world/reports/world_model_head132_scale_exact_state_gap.md`
20. `experiments/world/reports/world_model_head133_scale_regime_probe_gap.md`
21. `experiments/world/reports/world_model_head134_scale_baseline_target_taxonomy.md`
22. `experiments/world/reports/world_model_head135_exact_state_retention_literature_gate.md`
23. `experiments/world/reports/world_model_head136_scale_representation_surface.md`
24. `experiments/world/reports/world_model_head089_part1_open_risk_ledger.md`
25. `experiments/world/reports/world_model_head086_downstream_probe_interpretation.md`
26. `experiments/world/reports/world_model_head097_mask_policy_coverage_audit.md`
27. `experiments/world/reports/world_model_head096_stop_condition_verification.md`
28. `experiments/world/reports/world_model_head100_sample_scale_caveat.md`
29. `experiments/world/reports/world_model_head102_downstream_probe_reporting_audit.md`
30. `experiments/world/reports/world_model_head105_downstream_target_scope_audit.md`
31. `experiments/world/reports/world_model_head107_manual_stop_runtime_guardrail.md`
32. `experiments/world/reports/world_model_head110_package_checker_guardrail_docs.md`
33. `experiments/world/reports/world_model_head112_open_risk_ledger_refresh.md`
34. `experiments/world/reports/world_model_head116_goal_checker_manual_stop.md`

## Decision

This index is a navigation artifact only. It does not change the model,
objective, or acceptance boundary.
