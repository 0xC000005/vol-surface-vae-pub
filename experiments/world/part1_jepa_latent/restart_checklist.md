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
25. `experiments/world/reports/world_model_head137_context_target_jepa_design.md`
26. `experiments/world/reports/world_model_head138_context_target_data_scaffold.md`
27. `experiments/world/reports/world_model_head139_context_target_model_scaffold.md`
28. `experiments/world/reports/world_model_head140_context_target_smoke.md`
29. `experiments/world/reports/world_model_head141_context_target_state_probe.md`
30. `experiments/world/reports/world_model_head142_context_target_latent_health.md`
31. `experiments/world/reports/world_model_head143_context_target_clean_target_gate.md`
32. `experiments/world/reports/world_model_head144_context_target_clean_smoke.md`
33. `experiments/world/reports/world_model_head145_context_target_clean_quality.md`
34. `experiments/world/reports/world_model_head146_context_target_route_decision.md`
35. `experiments/world/reports/world_model_head147_part1_candidate_decision_matrix.md`
36. `experiments/world/reports/world_model_head148_scaled_gate_reconciliation.md`
37. `experiments/world/reports/world_model_head149_scale_exact_state_topology.md`
38. `experiments/world/reports/world_model_head150_surface_local_jepa_design_gate.md`
39. `experiments/world/reports/world_model_head151_surface_local_data_contract.md`
40. `experiments/world/reports/world_model_head152_surface_local_target_coverage.md`
41. `experiments/world/reports/world_model_head153_surface_local_model_scaffold.md`
42. `experiments/world/reports/world_model_head154_surface_local_jepa_smoke.md`
43. `experiments/world/reports/world_model_head155_surface_local_smoke_failure.md`
44. `experiments/world/reports/world_model_head156_surface_local_target_geometry.md`
45. `experiments/world/reports/world_model_head157_surface_local_route_decision.md`
46. `experiments/world/reports/world_model_head158_post_surface_local_candidate_consolidation.md`
47. `experiments/world/reports/world_model_head159_open_risk_ledger_surface_local_refresh.md`
48. `experiments/world/reports/world_model_head160_part1_gate_reconciliation_after_surface_local.md`
49. `experiments/world/reports/world_model_head161_next_work_wording_refresh.md`
50. `experiments/world/reports/world_model_head162_active_doc_stale_route_scan.md`
51. `experiments/world/reports/world_model_head163_scale_stability_boundary.md`
52. `experiments/world/reports/world_model_head164_exact_state_literature_refresh.md`
53. `experiments/world/reports/world_model_head165_exact_state_conditioning_boundary.md`
54. `experiments/world/reports/world_model_head166_additive_signal_gate.md`
55. `experiments/world/reports/world_model_head167_additive_signal_gate_audit.md`
56. `experiments/world/reports/world_model_head168_additive_exact_state_guardrail.md`
57. `experiments/world/reports/world_model_head169_additive_exact_state_topology.md`
58. `experiments/world/reports/world_model_head170_additive_probe_standardization.md`
59. `experiments/world/reports/world_model_head171_additive_gate_reconciliation.md`
60. `experiments/world/reports/world_model_head172_temporal_block_jepa_smoke.md`
61. `experiments/world/reports/world_model_head173_temporal_jepa_bakeoff.md`
62. `experiments/world/reports/world_model_head174_temporal_route_decision.md`
63. `experiments/world/reports/world_model_head175_temporal_stale_route_scan.md`
64. `experiments/world/reports/world_model_head082_part1_scorecard_health.md`
65. `experiments/world/reports/world_model_head083_mask_artifact_audit.md`
66. `experiments/world/reports/world_model_head084_stratified_mask_audit.md`
67. `experiments/world/reports/world_model_head086_downstream_probe_interpretation.md`
68. `experiments/world/reports/world_model_head097_mask_policy_coverage_audit.md`
69. `experiments/world/reports/world_model_head100_sample_scale_caveat.md`
70. `experiments/world/reports/world_model_head102_downstream_probe_reporting_audit.md`
71. `experiments/world/reports/world_model_head105_downstream_target_scope_audit.md`
72. latest tail of `RESEARCH_LOG.md`

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
- Treat HEAD137 as the current context-to-target JEPA design boundary: any next
  model branch must be same-window masked current/history latent prediction,
  not future prediction, not value reconstruction, and not a silent mutation of
  the Barlow reference.
- Treat HEAD138 as the current context-to-target data surface: it is a scaffold
  for a future model smoke, not a promoted Part 1 branch.
- Treat HEAD139 as the current context-to-target model/loss surface: it is a
  scaffold for a future training smoke, not a trained or promoted branch.
- Treat HEAD140 as the current context-to-target training smoke: it trains, but
  representation health is weak, so it must not be promoted before frozen state
  probes and health correction.
- Treat HEAD141 as the current context-to-target state probe: the minimal
  context-to-target smoke is worse than scaled Barlow on current-IV exact-state
  probes and has weaker rank, so diagnose the branch before adding model knobs.
- Treat HEAD142 as the current context-to-target latent-health diagnosis: the
  target latent is mask-family heavy and the predictor is low-rank despite high
  cosine, so do not tune model knobs before fixing that diagnostic failure.
- Treat HEAD143 as the current context-to-target correction gate: one
  clean-target smoke is allowed, where the target encoder sees the clean full
  window and target rows are selected from its output.
- Treat HEAD144 as the current clean-target smoke result: it trains but worsens
  clean context rank versus HEAD140 and remains far below scaled Barlow.
- Treat HEAD145 as the current clean-target quality result: it is worse than
  target-only and scaled Barlow on current-IV exact-state probes and rank.
- Treat HEAD146 as the current context-to-target route decision: the minimal
  branch is demoted and should not be tuned with small knobs.
- Treat HEAD147 as the current Part 1 candidate matrix: scaled Barlow remains
  active but not promoted; minimal context-to-target variants are demoted.
- Treat HEAD148 as the current scaled gate reconciliation: Part 1 remains
  `DO_NOT_PROMOTE` and Part B remains blocked.
- Treat HEAD149 as the current exact-state topology audit: the raw-baseline gap
  is broad but especially concentrated in wing moneyness and edge maturities.
- Treat HEAD150 as a design gate only for token/geometry-level JEPA. It does not
  implement a model, promote Part 1, or unblock Part B.
- Treat HEAD151 as the token/geometry-level data contract scaffold only. Audit
  target coverage before adding an encoder or loss.
- Treat HEAD152 as the target coverage audit for that scaffold. It is not model
  evidence.
- Treat HEAD153 as the token/geometry-level model/loss scaffold. It is not a
  trained checkpoint, not a Part 1 promotion, and not permission to start Part B.
- Treat HEAD154 as the surface-local token JEPA smoke result: it trains, but
  target-token retrieval is weak and predicted rank is low, so diagnose before
  tuning or promoting the route.
- Treat HEAD155 as the current surface-local diagnosis: target coverage is not
  the failure; selected target latents are already low-rank and the predictor
  shrinks variance further.
- Treat HEAD156 as the current target-geometry audit: target latents cluster by
  token/factor and target family, while predictor retrieval remains poor for
  exact rows.
- Treat HEAD157 as the current route decision: the surface-local
  context-to-target implementation is demoted as implemented; do not tune small
  knobs on this route.
- Treat HEAD158 as the current candidate consolidation: scaled Barlow is still
  the active learned candidate but remains `DO_NOT_PROMOTE`; Part B remains
  blocked.
- Treat HEAD159 as the current open-risk ledger refresh after surface-local
  demotion; do not continue by tuning demoted context-to-target routes.
- Treat HEAD160 as the current gate reconciliation after surface-local demotion:
  Part 1 remains `DO_NOT_PROMOTE`, and Part B remains blocked.
- Treat HEAD161 as the next-work wording refresh: active docs no longer point
  future resumes into completed or demoted context-to-target tuning tasks.
- Treat HEAD162 as the active-doc stale-route scan: remaining active-doc hits
  are intended demotion/blocker statements.
- Treat HEAD163 as the scale-stability boundary: do not run more seed stability
  as the next default step; it does not address exact-state/baseline blockers.
- Treat HEAD164 as the exact-state literature refresh: do not add raw-value
  reconstruction under the JEPA label; any new JEPA target needs a state-variation
  gate first.
- Treat HEAD165 as the exact-state conditioning boundary: do not require one
  compact learned embedding to replace raw current-state identity information.
- Treat HEAD166 as the additive-signal quality gate: evaluate raw-only,
  learned-only, and raw-plus-learned frozen probes before treating learned
  embeddings as useful abstract market-state information.
- Treat HEAD167 as the current additive-signal audit: path-shape/risk-width
  additive signal passes, balanced-regime signal is partial, and exact-state
  plus persistence guardrails still block promotion.
- Treat HEAD168 as the current additive exact-state guardrail: raw-plus-learned
  still worsens IV exact-state slightly versus raw-only, so exact-state
  guardrail remains failed despite non-surface improvements.
- Treat HEAD169 as the current topology diagnosis for that guardrail:
  raw-plus-learned worsens `14/25` IV cells and improves `11/25`, so the
  exact-state miss is not only one pathological cell.
- Treat HEAD170 as the current probe-standardization diagnostic: feature
  standardization does not fix the raw-plus IV exact-state guardrail, so do not
  dismiss HEAD169 as only a scale artifact.
- Treat HEAD171 as the current additive-gate reconciliation: the embedding has
  additive abstract signal, but Part 1 remains blocked by exact-state and
  persistence guardrails.
- Treat HEAD172 as a separate temporal context-to-target JEPA smoke: it hides
  the last five days inside the history window and predicts latent target rows
  only, with no future-window target, decoder, or raw reconstruction. It shows
  mixed additive probe signal but weak retrieval/rank, so it is
  `SMOKE_ONLY_DO_NOT_PROMOTE`.
- Treat HEAD173 as the temporal JEPA frozen bakeoff: raw+temporal improves the
  current-IV guardrail, but raw+random does better on the same guardrail, while
  temporal raw+ improves only `1/5` future targets versus `3/5` for random raw+
  and `3/5` for scaled Barlow raw+. The temporal route is `DO_NOT_PROMOTE` and
  should not be tuned with small context-to-target knobs.
- Treat HEAD174 as the temporal route decision: the current temporal
  context-to-target route is demoted as implemented. Continue with provenance,
  gate reconciliation, or a genuinely new design gate; do not tune temporal
  hidden size, epochs, mask, target length, EMA, or predictor depth.
- Treat HEAD175 as the temporal stale-route scan: active docs are consistent
  after the demotion, and no temporal stale-route cleanup is currently needed.

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
- Implement context-to-target JEPA without comparing it against the scaled
  Barlow candidate and the formal Part 1 gate layers.
- Treat an additive-signal probe as decoder evidence or Part B authorization.
- Treat additive signal on path-shape/risk-width probes as sufficient Part 1
  promotion while exact-state and persistence guardrails remain failed or
  partial.
- Treat raw-plus-learned non-surface improvements as enough to ignore the IV
  exact-state guardrail failure.
- Treat the HEAD172 temporal smoke as Part-B-ready or as permission to tune
  context-to-target knobs; its own report marks it smoke-only.
- Treat the HEAD173 temporal frozen bakeoff as a positive temporal-JEPA result;
  its random-control and scaled-candidate comparisons are negative.
- Resume temporal context-to-target tuning without a new design gate; HEAD174
  explicitly demotes that route as implemented.

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
