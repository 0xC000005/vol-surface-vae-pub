# Narrative Prefix-Latent Current Truth

Last updated: 2026-05-11

This file is the tracked promotion index for the natural-language prefix-latent
scenario-generator workflow. Ignored artifacts remain the detailed evidence, but
this file states which claims are currently promoted and which are still only
diagnostic.

## Current Contract

The production default is:

```text
full risk-manager narrative
+ grounding sidecar for direction/warning checks
+ risk-manager-selected or user-supplied joint39 starting level
-> narrative-and-start-compatible soft support mixture
-> decoded/refined recent prefix
-> frozen joint39 SNI autoregressive rollout
-> 30-day scenario distribution
```

There is no hidden model-chosen starting level in the production default. A
narrative describes current or recent market conditions; forward-looking or
desired-future language is warning-only and must not become the future target.

## Current Incumbent

- Support prior: `soft_topk_narrative_start_checked`.
- Text bridge incumbent: `mlp_mse_contrastive + multi_caption_with_negatives`.
- Grounding role: sidecar audit/check, not a replacement for the full
  narrative.
- Retrieval role: provenance and support audit, not the publication target.
- Generator: frozen joint39 SNI checkpoint through the native autoregressive
  rollout path.
- Autoresearch orchestration: centralized HEAD loop with bounded multi-agent
  sidecars only at literature, critique, verification, report-audit, or
  disjoint-implementation gates.

## Current Autoresearch Policy

The NL prefix-latent workflow now separates research exploration from product
promotion:

- **Exploration lane:** cheap local tests, literature checks, and mechanism
  probes may regress below the simple-mixture floor. A bad result is acceptable
  when it teaches why a method should be promoted, modified, or killed.
- **Candidate lane:** a method with a plausible mechanism can tolerate moderate
  regression only if the expected trade-off was stated before the run.
- **Promotion lane:** a method must be competitive with the incumbent simple
  mixture on held-out scenario-level quality, fixed-start controls, support
  audits, and null/repeat checks before it changes defaults or paper claims.
- **Production/demo lane:** no drastic regression is allowed. Small regressions
  are acceptable only for clear trust, warning-quality, OOD-rejection,
  provenance, or fixed-start-stability gains.

Every future HEAD entry should name `research_lane`, `result_status`, and
`benchmark_floor_status`. The simple mixture remains the promotion floor and
production backbone, but it should not prevent low-cost exploration of learned
support rerankers, contrastive support alignment, prototype-aware mixture
weights, or bounded residual refinements.

Sophisticated narrative-to-mixture methods now require a **story gate** in
addition to the metric gate. Before a method can be promoted, the workflow must
record the method story, related-work basis, novelty claim, elegance check,
historical-backtest comparison, and kill condition. This is meant to keep the
research line publishable and product-legible: a new method should be
methodologically justified, not just more complicated, and it must still beat or
remain competitive with the simple mixture on historical backtesting before it
changes defaults or paper claims.

## Benchmark Floor For New Narrative Methods

The current long-term objective is to improve the narrative-to-mixture workflow,
not to remove the historical support mixture. New candidate methods must be
benchmarked against the working simple mixture before promotion.

Primary floor:

- incumbent: `soft_topk_narrative_start_checked` /
  `narrative_generator_topk`;
- representative artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_residual_scenario_eval_878a_full/scenario_level_eval_report.json`;
- direct text-predicted memory without support mixture:
  `+1.9%` ensemble CRPS, `+8.3%` energy, `0.390` 80% coverage versus
  persistence;
- narrative top-k support generator:
  `+17.2%` ensemble CRPS, `+21.6%` energy, `0.645` 80% coverage versus
  persistence;
- residual over top-k support at alpha `0.25`:
  `+17.4%` ensemble CRPS, `+21.6%` energy, `0.650` 80% coverage versus
  persistence.

A small distributional regression may be acceptable only when it buys clear
trust, stability, fixed-start conditionality, warning quality, or support
auditability. A method with drastic regression below the simple mixture remains
diagnostic even if it improves target cosine, retrieval rank, or an isolated
hard-case metric. Publishable novelty should come from learned support
reranking, contrastive support alignment, prototype-aware weighting,
text/start-conditioned mixture weights, or bounded residual refinement around
the mixture.

## Promoted Evidence

### Representative Bridge And Scenario Evaluation

- Bridge report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json`
- Scenario report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_scenario_level_eval_openai_schema_v2_representative_220/scenario_level_eval_report.json`

Current supported claim:

- held-out target cosine: `0.8597` mean, `0.8739` median;
- hard-negative gap: `0.8835`;
- recall@1 test-pool: `0.0794`;
- recall@3 test-pool: `0.1746`;
- narrative-generator energy improvement versus persistence: `+21.3%`;
- narrative-generator ensemble CRPS improvement versus persistence: `+17.4%`;
- mean-path and terminal point-error metrics remain worse than persistence.

Interpretation: the system is a distributional scenario generator with audit
support, not a point forecaster or exact historical-window retriever.

### Text-To-Memory Policy Tests

- Policy TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_850a_manifest_policy_testflight/bridge_architecture_bakeoff_report.json`
- Seed stability:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_851d_seed_stability/policy_stability_summary.json`
- CLIP MSE sweep:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_852d_clip_mse_sweep/clip_mse_sweep_summary.json`

Current supported claim:

- multi-caption hard-negative training is the incumbent;
- multi-caption without hard negatives improves target cosine but weakens
  directional separation;
- hard negatives restore and strongly improve directional separation while
  preserving target cosine;
- current CLIP/InfoNCE hybrid is not promoted because it trails the incumbent on
  generator-memory target cosine and hard-negative separation.

Diagnostic, not yet promoted:

- expanded OpenAI label manifest:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_expansion_878e_openai_full/narrative_pipeline_report.json`.
  The larger manifest produced `240` usable labeled windows, `2974` text
  examples, and one rejected label (`joint39_val_0240`) due an
  external-catalyst grounding violation. The expanded bridge report
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_expansion_878g/bridge_eval_report.json`
  keeps target cosine near the incumbent (`0.8590`) on `42` held-out windows,
  but hard-negative gap (`0.8252`) and recall@3 test-pool (`0.1326`) are weaker
  than the representative 182-window run. Scenario-level evaluation
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_scenario_level_eval_expansion_878h/scenario_level_eval_report.json`
  remains distributionally useful (`+16.5%` CRPS, `+20.6%` energy), and on the
  `28` overlapping old test windows it slightly improves CRPS and energy while
  lowering coverage. Treat this as evidence that label scaling is feasible but
  not by itself a solved text-to-latent upgrade.
- expanded caption hard-case audit:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_caption_grounding_audit_878i/caption_grounding_audit.json`.
  The current quality hard cases are narrow: one rejected label
  (`joint39_val_0240`) and five low hard-negative-margin windows
  (`joint39_val_0377`, `0379`, `0384`, `0386`, `0430`). The low-margin cases
  are mostly risk-on/reflation descriptions where positive and negative
  catalyst language can become too semantically close. This points to caption
  repair/hard-negative wording before another bridge architecture.
- hard-negative wording repair TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_hard_negative_repair_878j_hardcases/hard_negative_repair_report.json`.
  Re-embedding the artificial negative controls for the five low-margin
  hard-case windows did not fix separation. The follow-up audit
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_caption_grounding_audit_878j_repair_hardcases/caption_grounding_audit.json`
  still has five low-margin cases and increases low-gap cases from `1` to `3`.
  Do not scale this deterministic negative wrapper.
- low-margin embedding geometry:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_embedding_geometry_hardcases_878k/embedding_geometry_hardcases.json`.
  All five low-margin windows are already low-margin in raw OpenAI embedding
  space. Mean raw hard margin is `0.278`; the learned bridge improves the mean
  margin to `0.382` but does not fully repair it. The bottleneck is therefore
  primarily text-embedding/representation geometry for risk-on/reflation
  contrastive cases, not just the MLP adapter.
- embedding-model/representation ablation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_embedding_model_hardcase_ablation_878l_large_full/embedding_model_hardcase_ablation.json`
  and
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_embedding_model_hardcase_ablation_878l_small_factor_tokens/embedding_model_hardcase_ablation.json`.
  `text-embedding-3-large` on full text worsens mean hard margin to `0.226`,
  and factor-token-only text with `text-embedding-3-small` worsens it to
  `0.096`. Generic larger embeddings and stripping to factor tokens are not the
  current fix; the likely direction is a hybrid narrative embedding plus
  explicit directional structure.
- hybrid direction-feature bridge TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_hybrid_direction_bridge_878m/hybrid_direction_bridge_report.json`.
  Equal-norm concatenation of the full narrative embedding with explicit
  direction features is not promoted. It leaves target cosine essentially flat
  (`+0.00013`) and slightly improves recall@3 test-pool (`+0.00552`), but
  worsens hard-negative mean margin by `-0.11319` and hard-negative mean gap by
  `-0.02456`. The old worst case `joint39_val_0377` improves, but broader
  separation degrades. Treat direction features as likely auxiliary loss,
  calibration probe, or audit constraint candidates, not as a naive
  concatenated input default.
- bounded start-residual bridge:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_start_memory_stability_876c_start_residual/text_start_memory_stability.json`.
  Across seeds `775`, `776`, and `777`, the bounded residual preserves the
  text-only memory target (`-0.0007` mean cosine delta) and hard-negative
  separation (`-0.0065` mean gap delta, `-0.0081` mean margin delta), and
  improves recall@3 by `+0.0159`. It is not a default until downstream
  scenario-level evaluation shows that the preserved residual signal actually
  affects fixed-start narrative scenario distributions.
- downstream residual scenario evaluation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_residual_scenario_eval_878a_full/scenario_level_eval_report.json`.
  On the same 29 held-out representative windows, direct text-predicted memory
  remains too weak to replace support-conditioned rollout (`+1.9%` CRPS,
  `+8.3%` energy, `0.390` 80% coverage versus persistence). The best bounded
  residual-over-top-k variant, `narrative_residual_topk_a025`, is competitive
  with the current top-k incumbent (`+17.4%` CRPS, `+21.6%` energy, `0.650`
  80% coverage), but the gain is marginal and mixed across metrics. Treat it as
  a promising diagnostic, not a promoted production default.
- residual attribution diagnostic:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_residual_scenario_attribution_878b/residual_scenario_attribution.json`.
  Alpha `0.25` improves average CRPS by only `0.0014` z-score units over the
  incumbent and wins CRPS on `14/29` windows; alpha `0.10`, alpha `0.50`, and
  direct-memory replacement are worse. Residual gains are not positively
  associated with higher top-1 support similarity (`-0.248` correlation), so
  the current residual bridge does not yet show a clean support-quality
  mechanism.
- learned support-reranker TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_learned_support_reranker_879a_testflight/learned_support_reranker_report.json`.
  This exploration-lane branch trained a small ridge reranker on train-window
  query/candidate features with negative standardized replay loss as the label.
  It found support signal at the replay level: held-out replay CRPS improved by
  `+0.0218` z-score units and replay energy improved by `+0.0059`, while
  coverage fell by `-0.0135`. A same-seed 5-window frozen-generator smoke did
  not promote the idea: historical replay improved, but narrative-generator
  CRPS regressed by `-0.0031` and energy by `-0.0017` versus the original
  support ordering. The decomposition
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_learned_support_reranker_879d_support_quality_decomposition/support_quality_decomposition.json`
  labels the mechanism `support_replay_generator_mismatch`. Treat this as
  evidence that future support rerankers should optimize generator-calibrated
  rollout compatibility or include a frozen-generator-aware loss, not only
  historical replay closeness.
- replay/generator mismatch post-analysis:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_generator_mismatch_analysis_880a/support_generator_mismatch_analysis.json`.
  The learned reranker improves the support proxy stack on all `126` compared
  rows: mean replay loss delta is `-0.0532`, mean generator self-calibration
  energy delta is `-0.0331`, mean generator self-calibration CRPS delta is
  `-0.0297`, mean start-distance delta is `-1.1575`, and coverage proxy delta
  is `+0.0050`. But the actual same-seed rollout smoke still regresses
  (`-0.0031` CRPS and `-0.0017` energy improvement deltas). The current
  mechanism label is therefore `generator_proxy_false_positive`: train-window
  self-calibration and historical replay are insufficient proxy labels for a
  support reranker. The next candidate should use direct rollout-response
  labels or a stability-aware generator-level objective, not another replay or
  self-calibration ranking knob.
- rollout-response label TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_rollout_response_label_testflight_881a/rollout_response_label_summary.json`.
  The scenario evaluator now has an explicit duplicate-query mode for
  candidate-specific support labels. A small CUDA smoke evaluated `4` query
  windows by `3` candidate supports each (`12` candidate rows) with top-k `1`.
  In `3/4` queries, the best actual generator-response support was not the
  cosine top-1 support. The within-pool best-vs-top1 mean deltas were
  `-0.0527` generator energy and `-0.0434` generator CRPS, lower-is-better.
  This is not a promoted model result because it is tiny and low-sample, but it
  is a positive mechanism TestFlight: direct rollout-response labels contain
  support-ranking signal that cosine/replay/self-calibration proxies miss.
- scaled rollout-response upper-bound:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_rollout_response_label_testflight_882a_fullheldout/rollout_response_label_summary.json`.
  Scaling the candidate label bridge to all `29` held-out query windows with
  the top `5` candidate supports (`145` candidate rows) confirmed that direct
  rollout-response labels contain ranking signal: best actual generator-response
  support was not the cosine top-1 support in `22/29` cases. The within-pool
  best-vs-top1 deltas were `-0.0898` generator energy and `-0.0746` generator
  CRPS, lower-is-better. However, the best single support upper bound still did
  not beat the same-seed simple top-k mixture baseline:
  best-single energy/CRPS means were `1.1265`/`0.7868`, while the top-k3
  baseline was `0.9826`/`0.6814`. This means the next learned method should
  stay mixture-level; replacing the auditable mixture with a learned single
  support is the wrong direction.
- mixture-level rollout-response upper-bound:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_rollout_response_mixture_testflight_884a_fullheldout/rollout_response_label_summary.json`.
  Evaluating all top-3 support subsets from each held-out query's top-5 support
  pool (`29` queries, `290` mixture rows) produced a positive upper-bound
  result. The best generator-response mixture was not the default cosine top-3
  in `25/29` cases. The best-in-pool mixture improved over the default top-3 by
  `-0.0594` energy and `-0.0470` CRPS on average, and it beat the same-seed
  simple top-k3 baseline: best-mixture energy/CRPS means were
  `0.9118`/`0.6248` versus baseline `0.9895`/`0.6879`. This is still an oracle
  upper bound, not a deployable learned policy, but it is the first strong
  evidence that a more sophisticated mixture-level narrative-to-support policy
  could beat the simple mixture floor.
- learned generator-response mixture policy:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_learned_mixture_policy_886b_32train_to_fullheldout/learned_mixture_policy_report.json`.
  The non-leaky train-label pipeline now builds train-window query bridges from
  cached text/memory artifacts, scores candidate top-3 support mixtures through
  the frozen generator, and trains a linear policy on inference-available
  mixture features. The 32-query label set confirms strong oracle signal:
  best generator-response mixture is not the default top-3 in `29/32` train
  queries, with best-vs-default deltas of `-0.1115` energy and `-0.0950` CRPS.
  However, the first learned policy is not promoted. On the 29 held-out windows
  at the same seed, it slightly regresses versus the simple mixture on energy
  (`0.9915` vs `0.9907`) and CRPS (`0.6947` vs `0.6882`) while improving 80%
  coverage (`0.5796` vs `0.5651`). This means generator-response labels are a
  viable training target, but the current linear mixture policy is too weak or
  under-labeled to beat the simple baseline.

### Fixed-Start Conditionality

- Tracked case spec:
  `docs/research_protocols/nl_prefix_latent_promoted_specs/fixed_start_narrative_matrix_858b_cases.json`
- Current 192-path narrative bakeoff:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865a_full_narrative_s192/start_conditioned_bakeoff.json`
- Current symmetric 192-path control suite:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865d_full_s192_symmetric/fixed_start_control_suite.json`
- Current start reliability manifest:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_start_reliability_gate_865d_full_s192_symmetric/start_reliability_gate.json`
- Verifier report:
  `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-11_start_reliability_gate_865b.md`

Current supported claim:

- 36 fixed-start narrative runs: 6 narratives by 6 starts;
- fixed-start equality passes;
- support/direction consistency passes;
- direction status: `36/36` pass at 192 generated paths;
- operational validation: `30` pass, `6` warning;
- energy improvement versus persistence: `+18.5%`;
- CRPS improvement versus persistence: `+17.1%`;
- symmetric fixed-start controls at 192 paths:
  - observed narrative median gap: `0.749`;
  - no-narrative start-only median gap: `0.000`;
  - same-narrative repeat median gap: `0.337`, ratio `0.450`;
  - within-run bootstrap median gap: `0.339`, ratio `0.453`;
- starts `0`, `18`, `22`, `40`, and `77` pass current start reliability;
- start `178` remains `warn_high_instability` because its repeat ratio is
  `0.764`, just above the `0.75` threshold.

Interpretation: the narrative is not just selecting the initial level. With the
same start held fixed, different narratives can move the distribution. However,
start reliability is start-specific. The current demo manifest supports five
starts and keeps `178` as a hard-case warning, not a normal recommended start.

## Not Yet Promoted

- Arbitrary live user narrative plus arbitrary user-supplied joint39 start.
- Full all-training-window OpenAI labeling for every historical window.
- A true Sora/DALL-E-style text-plus-start latent prior.
- Direct text-plus-start concatenation into the generator-memory bridge. The
  cached diagnostics
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_start_memory_diagnostic_875a/text_start_memory_diagnostic.json`
  and
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_start_memory_diagnostic_875b_start_weight/text_start_memory_diagnostic.json`
  show that naive and weakly weighted start features degrade held-out
  target-memory cosine and hard-negative directional separation versus
  text-only.
- CLIP/InfoNCE hybrid bridge as default.
- Bounded residual latent refinement as a proven production mechanism.
- Exact historical-window retrieval as a success criterion.

## Required Promotion Additions

Before a future paper/demo/product claim is promoted, it must add:

1. verifier report saved under
   `docs/research_protocols/nl_prefix_latent_verifier_reports/`;
2. tracked case spec or manifest under
   `docs/research_protocols/nl_prefix_latent_promoted_specs/`;
3. artifact paths and exact commands;
4. per-start quality floors, not just mean quality;
5. null/repeat baselines:
   - same narrative repeated with different seeds;
   - shuffled narratives under the same starts;
   - memory-only or no-narrative baseline;
6. clear pass/warn/fail status for start damping.
