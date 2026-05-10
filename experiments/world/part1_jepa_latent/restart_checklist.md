# World Model Part 1 Restart Checklist

Date: 2026-05-10

Use this before any future model change, decoder experiment, or downstream
probe that consumes the Part 1 world-model reference candidate.

## Read First

1. `docs/research_protocols/world_model_autoresearch_plan.md`
2. `experiments/world/part1_jepa_latent/reference_manifest.json`
3. `experiments/world/part1_jepa_latent/reference_artifact_digests.json`
4. `experiments/world/part1_jepa_latent/part1_quality_gate.md`
5. `experiments/world/reports/world_model_head118_part1_literature_quality_gate.md`
6. `experiments/world/reports/world_model_head119_part1_quality_gate_assessment.md`
7. `experiments/world/reports/world_model_head120_part1_failure_analysis.md`
8. `experiments/world/reports/world_model_head121_jepa_fit_diagnosis.md`
9. `experiments/world/reports/world_model_head122_hard_mask_preset.md`
10. `experiments/world/reports/world_model_head123_hard_mask_smoke.md`
11. `experiments/world/reports/world_model_head124_present_state_probe.md`
12. `experiments/world/reports/world_model_head125_state_content_gate.md`
13. `experiments/world/reports/world_model_head126_grouped_geometry_state_probe.md`
14. `experiments/world/reports/world_model_head127_scale_state_probe.md`
15. `experiments/world/reports/world_model_head128_scale_downstream_quality.md`
16. `experiments/world/reports/world_model_head129_scale_mask_artifact_audit.md`
17. `experiments/world/reports/world_model_head129_scale_stratified_mask_audit.md`
18. `experiments/world/reports/world_model_head130_scale_part1_quality_gate.md`
19. `experiments/world/reports/world_model_head131_scale_seed_stability.md`
20. `experiments/world/reports/world_model_head132_scale_exact_state_gap.md`
21. `experiments/world/reports/world_model_head133_scale_regime_probe_gap.md`
22. `experiments/world/reports/world_model_head134_scale_baseline_target_taxonomy.md`
23. `experiments/world/reports/world_model_head135_exact_state_retention_literature_gate.md`
24. `experiments/world/reports/world_model_head136_scale_representation_surface.md`
25. `experiments/world/reports/world_model_head082_part1_scorecard_health.md`
26. `experiments/world/reports/world_model_head083_mask_artifact_audit.md`
27. `experiments/world/reports/world_model_head084_stratified_mask_audit.md`
28. `experiments/world/reports/world_model_head086_downstream_probe_interpretation.md`
29. `experiments/world/reports/world_model_head097_mask_policy_coverage_audit.md`
30. `experiments/world/reports/world_model_head100_sample_scale_caveat.md`
31. `experiments/world/reports/world_model_head102_downstream_probe_reporting_audit.md`
32. `experiments/world/reports/world_model_head105_downstream_target_scope_audit.md`
33. latest tail of `RESEARCH_LOG.md`

## Fixed Reference Candidate

- Checkpoint:
  `models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt`.
- Training result:
  `results/world/masked_multiview_barlow_head070.json`.
- Data path: `data/vol_surface_with_ret.npz`.
- Split/window contract: history `30`, future `30`, normalized windows from
  `build_masked_multiview_batch`.
- Token dimension: `58`.
- Latent dimension: `64`.
- Reference scale: smoke-scale, `384` train windows, `128` validation windows,
  `8` epochs.
- Objective: direct masked-multiview same-state encoder alignment with
  Barlow-style redundancy control.

## Before Any Experiment

- Verify `autoresearch-session/WORLD_MODEL_STOP` is absent.
- Verify artifact digests if the experiment consumes ignored checkpoint,
  result, or data files, and verify guardrail-doc caveat terms are still
  present.
  Use:
  `python experiments/world/part1_jepa_latent/reference_package_check.py`.
- State whether the experiment is Part 1, Part 2, or a downstream probe.
- State the objective family: `masked_multiview_invariance`,
  `context_to_target_jepa`, or `downstream_probe`.
- Keep future prediction, range estimation, regime labels, and generation as
  downstream probes unless the workflow is explicitly changed.
- Copy the acceptance boundary from `reference_manifest.json`.
- Copy the caveats from `package_summary.md`: smoke-scale evidence only,
  validated default mask families only, mixed downstream utility, and
  IV-surface future targets only.
- Before Part B, copy the Part 1 quality gate and state which layers are
  already passed, not run, or failed.
- Treat HEAD119 as the current executed assessment: Part 1 is not ready for
  Part B. The embedding-learning signal passes at smoke scale, but baseline
  superiority, market-state probes, and scale/stability are not solved.
- Treat HEAD120 as the current failure analysis: this is not a collapse failure.
  The representation is useful versus trivial baselines and sometimes
  complementary to raw features, but it is not certified beyond raw/simple
  market-state baselines.
- Treat HEAD121 as the current JEPA-fit diagnosis: the current masks are mild
  and overlapping enough that direct two-view invariance may mostly preserve raw
  state identity. Before changing the objective or architecture, run one
  controlled hard-mask diagnostic with the same encoder/loss.
- Treat HEAD122 as the one allowed hard-mask diagnostic preset. It is not an
  active reference and should not start a knob sweep; use it to test the mask
  difficulty failure class.
- Treat HEAD123 as the current hard-mask outcome: mask aggression alone did not
  close baseline superiority and the hard checkpoint should not be promoted.
  Before tuning masks again, audit present-state and factor-panel information in
  the frozen embedding.
- Treat HEAD124 as the current present-state audit: the default embedding has
  factor-return signal but loses too much exact IV/current-state geometry. This
  explains the raw baseline gap and should guide the next Part 1 evidence work.
- Treat HEAD125 as the current state-content gate: non-surface signal passes,
  but exact IV-state retention and factor-level fidelity fail.
- Treat HEAD126 as the current grouped-geometry diagnostic: retrieval improves,
  but rank and state-content probes degrade, so shallow grouped pooling should
  not be promoted.
- Treat HEAD127 as the current scale diagnostic: the flat encoder improves
  rank, retrieval, and state-content probes at larger smoke scale, but it still
  needs the full Part 1 quality gate before any Part B work.
- Treat HEAD128 as the current downstream audit for HEAD127: scale helps, but
  baseline superiority and regime probes still block Part B.
- Treat HEAD129 as the current scaled corruption audit: the scaled candidate
  shows no large mask-family leakage and no large stratified mask-family
  failure.
- Treat HEAD130 as the current scaled Part 1 gate: representation health and
  corruption robustness pass, state content and scale/stability are partial,
  and baseline superiority plus market-state regime probes fail. The scaled
  checkpoint is the best candidate so far, but it is not promoted and is not
  Part-B-ready.
- Treat HEAD131 as the current scaled seed-stability smoke: representation
  health is stable across seeds `680`, `681`, and `682`, but this only upgrades
  the scale/stability evidence for representation health. It does not promote
  Part 1 or unblock Part B.
- Treat HEAD132 as the current exact-state gap diagnosis: scale does not fail
  through collapse or seed instability; it fails because raw current-state
  features still preserve IV-surface geometry better than the frozen embedding.
- Treat HEAD133 as the current regime-probe diagnosis: scaled Barlow still
  fails majority accuracy, but it has better macro recall and rare-class recall
  than raw surface features, so balanced metrics should inform future regime
  diagnostics.
- Treat HEAD134 as the current baseline-superiority diagnosis: scaled Barlow
  helps path-shape/risk-width targets and often adds to raw-last features, but
  loses persistence/exact-state targets, so broad Part 1 promotion remains
  blocked.
- Treat HEAD135 as the current exact-state-retention design gate: do not add an
  ad hoc value-reconstruction knob next; first audit whether per-time or
  flattened sequence embeddings preserve exact state better than the current
  last-state readout.
- Treat HEAD136 as the current representation-surface result: last-state
  scaled embeddings remain best for most present-state probes, and alternative
  pooling/flattening does not beat raw IV-surface exact-state baselines.

## Do Not Do Without Explicit Authorization

- Start Part 2 decoder training.
- Treat HEAD070 as Part-B-ready before the Part 1 quality gate passes.
- Add Part 1 model knobs or objective terms from HEAD085 alone.
- Reintroduce EMA/predictor same-state routing for two corrupted views.
- Treat fixed delta-PCA prediction as the current active reference.
- Promote regime classification as solved.
- Claim ImageNet-level JEPA behavior.
- Claim full-data convergence from the HEAD070 smoke-scale checkpoint.
- Claim the richer protocol mask ideas were validated by HEAD070 unless they
  have their own later report.
- Claim that harder masks solved Part 1; HEAD123 is a negative diagnostic for
  mask aggression alone.
- Convert present-state or future-probe tasks into Part 1 pretraining losses
  without an explicit objective-family change.
- Claim downstream factor-panel future target performance; HEAD085 targets are
  IV-surface futures only.
- Use generation metrics as evidence of Part 1 representation health.
- Add an exact-value reconstruction auxiliary loss before the HEAD135
  representation-surface audit is run and interpreted.

## Required Reporting

Every future authorized experiment should report:

- hypothesis and falsifier;
- objective family;
- literature status if the objective is nonstandard;
- checkpoint path and seed;
- exact split/data contract;
- whether Part 1 was frozen, semi-frozen, or modified;
- Part 1 representation metrics separately from Part 2 decoder metrics;
- explicit decision on whether the HEAD070 reference candidate remains valid.
