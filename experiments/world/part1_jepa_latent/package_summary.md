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

## Next Work Requires Direction

Future work should be one of:

- multi-seed scale stability using the same objective and encoder family;
- a minimal same-window context-to-target JEPA model smoke using the HEAD138
  data surface;
- a documented Part 1 quality-gate diagnostic that does not add model knobs by
  default.
