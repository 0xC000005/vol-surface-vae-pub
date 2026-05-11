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
