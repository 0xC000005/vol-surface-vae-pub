# World Model Part 1 Package Summary

Date: 2026-05-10

## Status

The current Part 1 world-model reference candidate is the HEAD070 direct
masked-multiview Barlow representation. It replaces the earlier fixed
delta-PCA predictor package as the active objective after the masked-multiview
objective correction.

This package is not a decoder and not a future-prediction pretraining model. It
is a representation-learning package for same-market-state masked views.

## Reference

- Objective family: `masked_multiview_invariance`.
- Literature status:
  `supported_adjacent_direct_barlow_twins_for_same_state_masked_multiview`.
- Checkpoint:
  `models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt`.
- Training result:
  `results/world/masked_multiview_barlow_head070.json`.
- Manifest:
  `experiments/world/part1_jepa_latent/reference_manifest.json`.
- Artifact digests:
  `experiments/world/part1_jepa_latent/reference_artifact_digests.json`.
- Package checker:
  `experiments/world/part1_jepa_latent/reference_package_check.py`, covering
  source reports, guardrail-doc caveat terms, and ignored artifact identities.
- Restart checklist:
  `experiments/world/part1_jepa_latent/restart_checklist.md`.
- Part 1 quality gate:
  `experiments/world/part1_jepa_latent/part1_quality_gate.md`.
- Latest executed Part 1 quality-gate assessment:
  `experiments/world/reports/world_model_head119_part1_quality_gate_assessment.md`.
- Latest failure analysis:
  `experiments/world/reports/world_model_head120_part1_failure_analysis.md`.
- Latest JEPA-fit diagnosis:
  `experiments/world/reports/world_model_head121_jepa_fit_diagnosis.md`.
- Latest hard-mask diagnostic preset:
  `experiments/world/reports/world_model_head122_hard_mask_preset.md`.
- Latest hard-mask training smoke:
  `experiments/world/reports/world_model_head123_hard_mask_smoke.md`.
- Latest present-state information audit:
  `experiments/world/reports/world_model_head124_present_state_probe.md`.
- Latest state-content gate:
  `experiments/world/reports/world_model_head125_state_content_gate.md`.
- Latest grouped-geometry diagnostic:
  `experiments/world/reports/world_model_head126_grouped_geometry_state_probe.md`.
- Latest scale/state-content diagnostic:
  `experiments/world/reports/world_model_head127_scale_state_probe.md`.
- Latest scale downstream audit:
  `experiments/world/reports/world_model_head128_scale_downstream_quality.md`.
- Latest scale corruption audit:
  `experiments/world/reports/world_model_head129_scale_mask_artifact_audit.md`.
- Latest scale stratified-mask audit:
  `experiments/world/reports/world_model_head129_scale_stratified_mask_audit.md`.
- Latest scale Part 1 quality gate:
  `experiments/world/reports/world_model_head130_scale_part1_quality_gate.md`.
- Latest scale seed-stability smoke:
  `experiments/world/reports/world_model_head131_scale_seed_stability.md`.
- Latest scale exact-state gap diagnostic:
  `experiments/world/reports/world_model_head132_scale_exact_state_gap.md`.
- Latest scale regime-probe diagnostic:
  `experiments/world/reports/world_model_head133_scale_regime_probe_gap.md`.
- Latest scale baseline target-family diagnostic:
  `experiments/world/reports/world_model_head134_scale_baseline_target_taxonomy.md`.
- Latest exact-state retention literature gate:
  `experiments/world/reports/world_model_head135_exact_state_retention_literature_gate.md`.
- Latest scale representation-surface audit:
  `experiments/world/reports/world_model_head136_scale_representation_surface.md`.
- Latest context-to-target JEPA design:
  `experiments/world/reports/world_model_head137_context_target_jepa_design.md`.
- Latest context-target data scaffold:
  `experiments/world/reports/world_model_head138_context_target_data_scaffold.md`.
- Latest context-target model scaffold:
  `experiments/world/reports/world_model_head139_context_target_model_scaffold.md`.
- Latest context-target training smoke:
  `experiments/world/reports/world_model_head140_context_target_smoke.md`.
- Latest context-target present-state probe:
  `experiments/world/reports/world_model_head141_context_target_state_probe.md`.
- Latest context-target latent-health diagnosis:
  `experiments/world/reports/world_model_head142_context_target_latent_health.md`.
- Latest context-target clean-target design gate:
  `experiments/world/reports/world_model_head143_context_target_clean_target_gate.md`.
- Latest context-target clean-target smoke:
  `experiments/world/reports/world_model_head144_context_target_clean_smoke.md`.
- Latest context-target clean-target quality comparison:
  `experiments/world/reports/world_model_head145_context_target_clean_quality.md`.
- Latest context-target route decision:
  `experiments/world/reports/world_model_head146_context_target_route_decision.md`.
- Latest Part 1 candidate decision matrix:
  `experiments/world/reports/world_model_head147_part1_candidate_decision_matrix.md`.
- Latest scaled gate reconciliation:
  `experiments/world/reports/world_model_head148_scaled_gate_reconciliation.md`.
- Latest exact-state topology audit:
  `experiments/world/reports/world_model_head149_scale_exact_state_topology.md`.
- Latest surface-local JEPA design gate:
  `experiments/world/reports/world_model_head150_surface_local_jepa_design_gate.md`.
- Latest surface-local data contract:
  `experiments/world/reports/world_model_head151_surface_local_data_contract.md`.
- Latest surface-local target coverage audit:
  `experiments/world/reports/world_model_head152_surface_local_target_coverage.md`.
- Latest surface-local model scaffold:
  `experiments/world/reports/world_model_head153_surface_local_model_scaffold.md`.
- Latest surface-local training smoke:
  `experiments/world/reports/world_model_head154_surface_local_jepa_smoke.md`.
- Latest surface-local smoke failure diagnosis:
  `experiments/world/reports/world_model_head155_surface_local_smoke_failure.md`.
- Latest surface-local target-geometry audit:
  `experiments/world/reports/world_model_head156_surface_local_target_geometry.md`.
- Latest surface-local route decision:
  `experiments/world/reports/world_model_head157_surface_local_route_decision.md`.
- Latest post surface-local candidate consolidation:
  `experiments/world/reports/world_model_head158_post_surface_local_candidate_consolidation.md`.
- Latest open-risk ledger refresh:
  `experiments/world/reports/world_model_head159_open_risk_ledger_surface_local_refresh.md`.
- Latest Part 1 gate reconciliation:
  `experiments/world/reports/world_model_head160_part1_gate_reconciliation_after_surface_local.md`.
- Latest next-work wording refresh:
  `experiments/world/reports/world_model_head161_next_work_wording_refresh.md`.
- Latest active-doc stale-route scan:
  `experiments/world/reports/world_model_head162_active_doc_stale_route_scan.md`.
- Latest scale stability boundary:
  `experiments/world/reports/world_model_head163_scale_stability_boundary.md`.
- Latest exact-state literature refresh:
  `experiments/world/reports/world_model_head164_exact_state_literature_refresh.md`.
- Latest exact-state conditioning boundary:
  `experiments/world/reports/world_model_head165_exact_state_conditioning_boundary.md`.
- Latest additive-signal quality gate:
  `experiments/world/reports/world_model_head166_additive_signal_gate.md`.
- Latest additive-signal gate audit:
  `experiments/world/reports/world_model_head167_additive_signal_gate_audit.md`.
- Latest additive exact-state guardrail:
  `experiments/world/reports/world_model_head168_additive_exact_state_guardrail.md`.
- Latest additive exact-state topology:
  `experiments/world/reports/world_model_head169_additive_exact_state_topology.md`.
- Latest additive probe standardization diagnostic:
  `experiments/world/reports/world_model_head170_additive_probe_standardization.md`.
- Latest additive gate reconciliation:
  `experiments/world/reports/world_model_head171_additive_gate_reconciliation.md`.

## Fixed Contract

- Data: `data/vol_surface_with_ret.npz`.
- Windowing: history `30`, future `30`.
- Token dimension: `58`.
- Latent dimension: `64`.
- Reference training windows: `384`.
- Reference validation windows: `128`.
- Positive pair: same market window and same relative index under two
  structured synthetic masks.
- Loss: direct encoder-output Barlow alignment with canonical mean-scaled
  off-diagonal term.

## Evidence

- Same-state retrieval: top1/top5/top10
  `0.321354/0.662500/0.841927`.
- Rank health: effective rank about `14.5` for both views.
- Redundancy: health offdiag about `0.224`, substantially lower than the
  high-retrieval but low-rank HEAD068 branch.
- Mask-artifact audit: mask-family prediction is below majority baselines.
- Stratified audit: no mask family has top10 below `0.826`.
- Mask-policy coverage: HEAD070 trained and was audited on
  `surface_maturity`, `surface_moneyness`, `surface_rectangle`,
  `vol_side_channel`, `factor_family`, and `time_block`; richer protocol ideas
  such as wing, ATM-strip, whole-surface day dropout, and cross-family stress
  masks remain coverage caveats, not validated HEAD070 claims.
- Downstream probe caveat: HEAD070 helps some risk-width/path-shape probes
  relative to raw last-surface features, but raw last-surface features remain
  stronger for mean/terminal deltas and regime-label accuracy. The full-history
  raw surface baseline also beats Barlow on max-absolute-step MSE, so the
  downstream utility claim is not "Barlow beats all raw baselines." These
  downstream probes target IV-surface futures only; factor-panel future targets
  have not been evaluated.
- Executed quality-gate assessment: HEAD119 returns
  `DO_NOT_PROMOTE`. Package integrity and representation health pass, default
  corruption robustness is partial, baseline superiority fails, market-state
  linear probes fail, temporal utility remains partial, and scale/stability
  fails.
- Failure analysis: HEAD120 says the failure is not representation collapse.
  Barlow beats the mean baseline on `5/5` IV future targets and is the best
  standalone feature on `2/5`; adding Barlow to raw last-surface features
  improves `3/5` targets. The main empirical failure is that raw/simple features
  still dominate mean/terminal and max-step targets, while missing evidence
  remains for factor-panel probes, richer masks, and scale/stability.
- JEPA-fit diagnosis: HEAD121 says the current branch is a reasonable
  collapse-controlled smoke test but weaker than canonical JEPA as a
  market-state learning recipe. Validation masks hide only about `6.6-7.8%` per
  view and keep about `86.6%` visible in both views, so the task is likely too
  easy and too close to raw-state preservation.
- Hard-mask preset: HEAD122 defines one named diagnostic preset, not a new
  active reference. It raises validation hidden rates to about `22.6-23.6%` per
  view and lowers both-visible overlap to about `60.9%`, while preserving typed
  market geometry. Use it to test mask difficulty before objective/architecture
  changes.
- Hard-mask training smoke: HEAD123 says mask aggression alone does not close
  the simple-baseline gap. The hard-mask checkpoint still beats the raw
  masked-view baseline on retrieval, but validation top10 falls from `0.842` to
  `0.605`, effective rank falls from `14.5` to `11.8`, standalone Barlow
  baseline-superiority falls from `2/5` to `0/5` IV future targets, and regime
  accuracy improves but remains below both raw last-surface features and the
  majority baseline.
- Present-state probe: HEAD124 says the default Barlow embedding is not empty
  and is not ignoring the factor panel: it has strong factor-return signal
  (`R2=0.799`) and beats raw IV-surface-only features on non-surface MSE.
  However, it loses too much exact present-state geometry: raw IV-surface-only
  features beat it on current IV-surface reconstruction, and factor-level plus
  side-channel probes remain weak. This explains why simple raw market-state
  baselines remain strong on persistence-like future probes.
- State-content gate: HEAD125 formalizes HEAD124 as an evaluation layer. It
  passes non-surface signal, but fails exact IV-state retention, factor-level
  fidelity against the raw full-geometry upper bound, and mask-aggression
  regression. The gate verdict is `FAIL`.
- Grouped-geometry diagnostic: HEAD126 keeps geometry groups separate before
  daily fusion and improves same-state retrieval (`top10=0.846`, `top1=0.512`),
  but it is not a better market-state representation. Effective rank falls to
  `7.89`, present-state probe rank falls to `5.02`, factor-return signal
  weakens, and IV/factor-level state-content probes get worse versus HEAD070.
- Scale diagnostic: HEAD127 keeps the HEAD070 flat encoder and objective but
  trains on `1024` windows with `256` validation windows. It improves top10
  retrieval (`0.850`), effective rank (`22.32`), redundancy (`offdiag=0.168`),
  IV-surface MSE, factor-level MSE, and preserves factor-return signal. It still
  does not beat raw surface features on exact current-IV reconstruction, so it
  is a next quality-gate candidate, not Part-B-ready evidence.
- Scale downstream audit: HEAD128 improves most downstream diagnostics but does
  not clear Part 1. Standalone Barlow still beats the best raw surface baseline
  on only `2/5` IV future targets, raw-last+Barlow improves raw-last on `4/5`,
  and regime accuracy improves to `0.516` but remains below majority `0.598`.
- Scale corruption audits: HEAD129 says the scaled HEAD127 checkpoint does not
  show large synthetic-mask-family leakage and does not have a large stratified
  mask-family failure. Mask-family prediction lift is at most `0.016` above
  majority, and the weakest stratified same-state retrieval top10 is `0.804`.
- Scale Part 1 quality gate: HEAD130 returns `DO_NOT_PROMOTE`. Representation
  health and corruption robustness pass for the scaled candidate, state content
  and scale/stability are partial, and baseline superiority plus market-state
  regime probes fail. HEAD127 remains the best Part 1 candidate so far, but it
  is not Part-B-ready.
- Scale seed stability: HEAD131 repeats the same scaled flat Barlow setup for
  seeds `681` and `682` and compares them with HEAD127 seed `680`. Same-state
  top10 stays in `0.839-0.877`, effective rank stays in `22.18-22.35`, and
  offdiag stays in `0.160-0.164`. This is a representation-health stability
  smoke pass, not a Part 1 promotion signal.
- Scale exact-state gap: HEAD132 confirms that the remaining raw-baseline gap
  is mostly exact-state retention, not collapse or seed instability. The scaled
  embedding's current-IV reconstruction MSE is `0.013756` versus raw
  last-surface `0.005630`, a `2.44x` ratio, and it is worse on `20/25` IV
  surface cells. The largest gap is the `iv_m0_t0` corner cell.
- Scale regime-probe gap: HEAD133 says the regime accuracy gate still fails,
  but the failure is not pure no-signal. Class `3` is the majority class
  (`153/256` validation rows). Scaled Barlow accuracy is `0.516` versus
  majority `0.598`, but macro recall is `0.521` versus raw-last `0.353`, and
  class-4 recall is `0.727` versus raw-last `0.000`.
- Scale baseline target taxonomy: HEAD134 says the downstream baseline failure
  is structured by target family. Scaled Barlow wins `2/2` path-shape/risk-width
  targets and adds to raw-last on `4/5` targets, but wins `0/2`
  persistence/exact-state targets and remains `DO_NOT_PROMOTE`.
- Exact-state retention literature gate: HEAD135 says not to add an ad hoc
  exact-value auxiliary loss next. First audit the representation surface
  (per-time or flattened sequence embeddings). If that fails, the principled
  objective change is same-window context-to-target JEPA for masked
  current/history blocks, not future prediction.
- Scale representation-surface audit: HEAD136 tests last, mean, last+mean, and
  flattened per-time scaled embeddings. It does not fix the exact-state gap:
  last-state embeddings remain best for IV, side-channel, factor-level, and
  all-geometry probes, and the best scaled IV MSE remains `0.013756` versus raw
  last-surface `0.005630`.
- Context-to-target JEPA design: HEAD137 defines the principled fallback family
  after the invariance branch capped out. It is same-window missing-state latent
  prediction for masked current/history blocks, not future prediction and not a
  value-reconstruction decoder.
- Context-target data scaffold: HEAD138 adds the same-window data surface for
  the context-to-target diagnostic. It exposes context values, target-only
  values, observed/context/target masks, geometry metadata, and target family
  labels without introducing future targets.
- Context-target model scaffold: HEAD139 adds the minimal latent model and
  target-time masked loss. It has a trainable context encoder, frozen target
  encoder, and predictor, but no future targets and no value reconstruction.
- Context-target training smoke: HEAD140 runs the first same-window
  context-to-target training smoke. The loss decreases, but clean representation
  health is weak versus scaled Barlow: clean-last effective rank is `9.41` and
  offdiag is `0.304`. It is runnable, not promoted.
- Context-target present-state probe: HEAD141 compares the HEAD140 context
  encoder against scaled Barlow and raw state baselines on identical frozen
  present-state probes. It does not fix exact-state retention: context-target IV
  MSE is `0.015289` versus scaled Barlow `0.013756` and raw surface `0.005630`,
  and rank is `11.59` versus scaled Barlow `18.76`. It is not promoted.
- Context-target latent-health diagnosis: HEAD142 localizes the HEAD140 failure.
  The target latent is strongly target-family identifiable (`+0.468` accuracy
  lift), while the predicted target latent is low-rank (`5.91`). Predicted-to-
  target cosine is high (`0.979`), but row retrieval is poor (`top10=0.043`), so
  the low loss is not evidence of a useful market-state representation.
- Context-target clean-target gate: HEAD143 maps the HEAD142 failure back to
  canonical JEPA target construction. The next allowed diagnostic is a single
  clean-target smoke where the target encoder sees the clean full window and
  target rows are selected from its output. No future targets, value
  reconstruction, decoder loss, or knob sweep are authorized by this gate.
- Context-target clean-target smoke: HEAD144 implements the one authorized
  correction. It trains, but clean context rank falls to `7.32`, worse than
  HEAD140 target-only (`9.41`) and much worse than scaled Barlow (`22.32`).
  The correction is not promoted.
- Context-target clean-target quality comparison: HEAD145 confirms the
  correction does not improve exact-state probes. Clean-target IV MSE is
  `0.015907`, worse than HEAD140 target-only `0.015289`, scaled Barlow
  `0.013756`, and raw surface `0.005630`; rank is `8.06` versus scaled Barlow
  `18.76`.
- Context-target route decision: HEAD146 demotes the minimal GRU row-level
  context-to-target route. It does not invalidate canonical JEPA for time
  series, but says a stronger attempt would be a new token/geometry-level
  architecture proposal, not another small knob.
- Part 1 candidate decision matrix: HEAD147 consolidates active candidates.
  HEAD127 scaled Barlow remains the active learned candidate but is
  `DO_NOT_PROMOTE`; raw surface remains the exact-state floor; both minimal
  context-to-target variants are demoted.
- Scaled gate reconciliation: HEAD148 confirms context-to-target results do not
  change the formal Part 1 gate. Scaled Barlow remains active but not promoted;
  Part B remains blocked.
- Exact-state topology audit: HEAD149 shows the current-IV gap is broad
  (`20/25` IV cells worse than raw surface) but concentrated in wing moneyness
  and edge maturities. The largest gap is `iv_m0_t0`.
- Surface-local JEPA design gate: HEAD150 opens, but does not implement, a
  token/geometry-level context-to-target route. Any implementation must start
  with a TDD data contract for target-token identity and geometry coordinates.
- Surface-local data contract: HEAD151 adds the token/geometry target data
  scaffold and focused test. It exposes `(window, relative_time, token)` target
  positions and surface-local families, but does not train a model.
- Surface-local target coverage audit: HEAD152 verifies the data contract covers
  the HEAD149 problem regions. All IV cells are targeted and `iv_m0_t0` has
  `1410` validation target positions.

## Caveats

- This is not canonical ImageNet I-JEPA.
- The HEAD070 checkpoint is a smoke-scale reference candidate trained on `384`
  windows and validated on `128`; do not claim full-data convergence from it.
- Future prediction, range estimation, regime labels, and scenario generation
  remain downstream probes or consumers, not pretraining objectives.
- The representation should not be described as a general predictor.
- Regime classification is not ready as an acceptance criterion.
- Part 1 success does not prove Part 2 scenario-generation quality.
- Part B remains gated until the literature-aligned Part 1 quality gate passes:
  representation health, corruption robustness, baseline superiority,
  market-state frozen probes, temporal utility probes, and scale/stability.
- The latest executed assessment says Part 1 is not ready for Part B. The
  corruption-based embedding learning signal works at smoke scale, but it is
  not yet a certified joint market-state representation.
- Do not interpret the quality-gate failure as collapse. Interpret it as a
  baseline-superiority and evidence-coverage failure until stronger frozen
  probes and simple baselines are run.
- Do not jump straight to many new knobs. First run one controlled hard-mask
  diagnostic preset with the same encoder/loss to test whether mask difficulty,
  rather than objective family or architecture, is the next bottleneck.
- Treat the HEAD122 hard-mask preset as a diagnostic branch only until a
  training smoke and downstream/frozen-probe audit justify promoting it.
- Treat HEAD123 as a negative result for "just mask harder." Do not make masks
  more aggressive again before auditing whether the frozen embedding captures
  present-state and factor-panel geometry.
- Treat HEAD124 as evidence that the next Part 1 work should improve
  state-content retention and evaluation coverage, not future-prediction
  pretraining or mask-severity tuning by itself.
- Treat HEAD125 as the active explanation for why simple market-state baselines
  remain strong: the representation learns useful factors but is not certified
  to preserve exact current-state geometry.
- Treat HEAD126 as a negative result for shallow geometry grouping. Higher
  same-state retrieval is not sufficient if state-content gates degrade.
- Treat HEAD127 as the current best Part 1 candidate for the next gate, while
  preserving the caveat that exact current-IV retention still loses to raw
  surface features.
- Treat HEAD128 as evidence that scale helps but does not close the quality
  gate. Do not start Part B from HEAD127/128 without another explicit Part 1
  promotion decision.
- Treat HEAD129/HEAD130 as evidence that the scaled candidate clears the
  corruption-robustness layer, but still fails the formal Part 1 promotion gate.
  Do not promote it until baseline superiority, regime probes, and exact-state
  retention improve or are replaced by a better justified acceptance layer.
- Treat HEAD131 as evidence that the scaled representation-health metrics are
  not a one-seed accident at smoke scale. It does not solve baseline
  superiority, market-state regime probes, exact-state retention, or full-data
  stability.
- Treat HEAD132 as the active diagnosis of the exact-state blocker: the scaled
  embedding has broad market-state signal, but raw current-state features still
  preserve IV geometry that the frozen embedding compresses away.
- Treat HEAD133 as the active diagnosis of the regime blocker: accuracy remains
  a failed promotion layer, but balanced/class-specific recall should be tracked
  before concluding the embedding has no regime information.
- Treat HEAD134 as the active diagnosis of baseline superiority: report
  downstream utility by target family, because the scaled embedding helps
  path-shape/risk-width probes while losing exact-state/persistence probes.
- Treat HEAD135 as the guardrail for the next design move: run a
  representation-surface audit before changing the objective; keep MAE-style
  value reconstruction as a separate diagnostic branch, not the default JEPA
  route.
- Treat HEAD136 as the completed representation-surface audit: the exact-state
  blocker is probably in the learned representation/objective, not merely the
  downstream pooling surface.
- Treat HEAD137 as the design boundary for any next model code: implement a
  separate context-to-target diagnostic branch and compare it against scaled
  Barlow; do not silently mix it into the two-view invariance reference.
- Treat HEAD138 as data-surface scaffolding only. It does not promote the
  context-to-target branch and does not change the active Barlow reference.
- Treat HEAD139 as model/loss scaffolding only. It has not been trained or
  compared to scaled Barlow.
- Treat HEAD140 as a runnable negative/weak smoke until frozen exact-state and
  representation-health probes show otherwise.
- Treat HEAD141 as the frozen exact-state result for HEAD140: the minimal
  context-to-target branch is worse than scaled Barlow on current-IV state
  probes and has weaker rank, so diagnose the branch before adding knobs.
- Treat HEAD142 as the current context-to-target root-cause diagnosis: the
  supervised target latent is mask-family heavy and the predictor collapses to a
  low-rank surface with high cosine but poor retrieval. Do not tune knobs before
  fixing that target/predictor diagnostic.
- Treat HEAD143 as the current design gate for that fix: run one clean-target
  context-to-target smoke, matching canonical JEPA target construction more
  closely, before considering any broader architecture change.
- Treat HEAD144 as a negative result for the minimal clean-target correction:
  canonical target construction alone does not fix clean context rank.
- Treat HEAD145 as the current comparison result: both minimal context-to-target
  variants are worse than scaled Barlow on exact-state probes and rank.
- Treat HEAD146 as the current route decision: do not tune the minimal
  context-to-target branch further; HEAD127/HEAD130 scaled Barlow remains the
  best known Part 1 candidate but is still not promoted.
- Treat HEAD147 as the current candidate matrix: Part 1 is still blocked by
  exact-state retention and baseline superiority, not by lack of a runnable
  training script.
- Treat HEAD148 as the current quality-gate reconciliation: context-to-target
  negatives do not unblock Part B or promote scaled Barlow.
- Treat HEAD149 as the current exact-state topology audit: future design should
  target surface-local geometry, not only a global row-level objective.
- Treat HEAD150 as a design gate only. It does not authorize Part B and does
  not promote any candidate.
- Treat HEAD151 as data-contract scaffolding only. It is not model evidence and
  does not change Part 1 promotion status.
- Treat HEAD152 as data coverage evidence only. It says an encoder can be tested
  on the right target regions, not that the representation is improved.
- Treat HEAD153 as model/loss scaffolding only. It adds explicit
  token-position target selection and a clean target-encoder surface, but it is
  not trained model evidence and does not promote Part 1.
- Treat HEAD154 as a runnable but negative/weak surface-local smoke. Loss
  falls, but target-token retrieval and effective rank are too weak to promote
  the route.
- Treat HEAD155 as the current surface-local failure diagnosis: the issue is
  low-rank target latent plus predictor variance shrinkage, not target coverage.
- Treat HEAD156 as the current geometry/family audit: the target latent clusters
  strongly by factor/geometry and the predictor mostly recovers coarse labels,
  not exact rows.
- Treat HEAD157 as the current route decision: demote the current surface-local
  context-to-target implementation and do not tune it with small knobs.
- Treat HEAD158 as the current candidate consolidation: scaled Barlow remains
  the active learned candidate, but it is still `DO_NOT_PROMOTE`; all current
  context-to-target variants are demoted as implemented.
- Treat HEAD159 as the current open-risk ledger refresh after surface-local
  demotion.
- Treat HEAD160 as the current Part 1 gate reconciliation after surface-local
  demotion: Part 1 is still `DO_NOT_PROMOTE` and Part B remains blocked.
- Treat HEAD161/HEAD162 as active-doc guardrails: next-work wording has been
  refreshed and scanned for stale routes into demoted context-to-target tasks.
- Treat HEAD163 as the scale-stability boundary: same-objective seed stability
  is not the active blocker; exact-state retention and baseline superiority are.
- Treat HEAD164 as the exact-state literature refresh: any JEPA revival needs a
  target-latent state-variation gate, while raw-value reconstruction is a
  separate MAE-style diagnostic family.
- Treat HEAD165 as the exact-state conditioning boundary: raw exact state should
  remain an explicit conditioning channel; learned embeddings should prove
  additive abstract state rather than replace raw identity information.
- Treat HEAD166 as the additive-signal quality gate: evaluate raw-only,
  learned-only, and raw-plus-learned frozen probes before claiming the learned
  representation adds abstract market-state value.
- Treat HEAD167 as the current additive-signal audit: additive path-shape and
  balanced-regime signal is present, but exact-state and persistence
  guardrails still fail promotion.
- Treat HEAD168 as the current raw-plus-learned exact-state guardrail: adding
  the learned embedding to raw surface features still slightly worsens IV exact
  state (`1.048x` raw), though it helps non-surface geometry targets.
- Treat HEAD169 as the current topology of that failure: raw-plus-learned is
  worse on `14/25` IV cells and better on `11/25`; the degradation is modest
  on average but broad enough to keep the exact-state guardrail failed.
- Treat HEAD170 as the current probe-hygiene diagnostic: standardizing feature
  blocks does not fix the IV exact-state guardrail, so the failure is not only
  an unstandardized ridge-scale artifact.
- Treat HEAD171 as the current additive-gate reconciliation: additive abstract
  signal is present, but exact-state and persistence guardrails still block
  Part 1 promotion and Part B.

## Next Work Requires Direction

Future work should be one of:

- provenance, manifest, digest, package-checker, and report-index consistency
  checks;
- gate reconciliation or risk-ledger refreshes that keep exact-state retention
  and baseline superiority as the active blockers;
- bounded exact-state blocker analysis that does not silently add a raw-value
  reconstruction auxiliary loss;
- an additive-signal audit over raw-only, learned-only, and raw-plus-learned
  frozen probes, without changing Part 1 pretraining or starting Part B;
- a follow-up additive-signal coverage audit only if it expands beyond the
  current IV-surface future targets and regime diagnostics without changing
  the Part 1 objective;
- exact-state guardrail diagnostics that explain why raw-plus-learned slightly
  worsens IV exact-state probes while improving non-surface geometry, without
  adding a reconstruction loss;
- same-objective scale/stability evidence for the active scaled Barlow
  candidate, without changing the objective;
- a genuinely new Part 1 design gate only if it first explains how target
  latents will carry state variation before predictor training.

Do not resurrect demoted context-to-target routes through target coverage,
hidden-size, predictor-depth, EMA, epoch, mask-aggression, or Barlow-weight
tuning.
