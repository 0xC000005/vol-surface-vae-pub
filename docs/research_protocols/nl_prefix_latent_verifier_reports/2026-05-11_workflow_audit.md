# Independent Verifier Report: NLP Prefix-Latent Workflow Audit

Date: 2026-05-11

Scope:

- `.agents/skills/nl-prefix-latent-autoresearch/SKILL.md`
- `docs/research_protocols/nl_prefix_latent_autoresearch_plan.md`
- `RESEARCH_LOG.md` entries around HEAD NLP Prefix-Latent 90-102
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_openai_schema_v2_representative_220/`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_850a_manifest_policy_testflight/`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_851*/`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_852*/`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_narrative_matrix_858b/`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_conditioning_gate_858b/`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_conditioning_analysis_859a/`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_start_damping_diagnostic_860a/`

## Verdict

**Partially working.**

The workflow is doing useful, evidence-backed research, but it is not yet a
production-ready or fully reliable autoresearch loop. The recent fixed-start
artifacts support a limited claim: with fixed historical starts and cached
narratives, narrative changes can move the generated distribution while the
start stays pinned. They do not yet prove robust live narrative conditioning for
arbitrary user-supplied starts.

## Confirmed Strengths

- The active contract is directionally correct: full narrative plus grounding
  sidecar, fixed start before mixture, no hidden model-chosen start in the
  production default.
- The bridge evidence is honest about winners and failures. The
  `mlp_mse_contrastive + multi_caption_with_negatives` bridge is supported;
  CLIP/SupCon-style variants are not promoted.
- The story-smoke path uses a decoded prefix and frozen SNI rollout rather than
  only a stale fixed memory vector.
- The 858b fixed-start gate is better than exact-retrieval metrics. It checks
  fixed-start equality, support/direction, persistence quality, operational
  status, and narrative separation.
- The 860a damping diagnostic is useful: `fixed_start_0` and
  `fixed_start_178` are identified as warning starts with plausible mechanisms.

## Methodological Weaknesses

- The main evidence is still diagnostic and small: 36 cached-condition runs, 6
  historical starts, 6 samples per run, no fresh live-story path, and no
  arbitrary user-supplied joint39 starts.
- The scenario quality gate is weak: it only requires mean improvement above
  zero versus persistence. It does not require per-start robustness, confidence
  intervals, repeated sampling seeds, or a shuffled-narrative/null baseline.
- Direction pass can be inflated by the selection mechanism.
  `soft_topk_narrative_start_checked` filters to direction-passing candidates
  before reporting `support_match=1.0`, so support/direction pass is partly a
  guardrail success, not independent proof of semantic conditioning.
- Pairwise standardized L2 gap is useful but not enough. Without same-narrative
  repeat runs or seed/bootstrap uncertainty, some narrative separation could be
  sampling noise or support-pool geometry.
- Exact retrieval remains weak. The bridge's recall@1 and recall@3 are low, so
  retrieval should remain audit/provenance, not a core success metric.
- The product target includes bounded residual/refinement, but recent evidence
  is still mostly deterministic text-memory bridge plus analogue mixture plus
  decoder. Residual refinement is not demonstrated.

## Workflow Weaknesses Found

- `autoresearch-session/nl_prefix_latent_goal.json` was stale and contradicted
  the current trust contract by referring to a model-chosen starting point.
- The 858b case spec lived only under ignored `autoresearch-session/`, so a
  tracked checkout could not reproduce the promoted matrix setup from tracked
  files alone.
- Verifier results were summarized in `RESEARCH_LOG.md`, but not saved as
  first-class verifier artifacts.
- There was no compact current-truth evidence index with commands, inputs,
  promoted/default status, and artifact pointers.
- Tests were mostly synthetic report-construction tests; they did not prove
  full fixed-start robustness across seeds, live inputs, or user-specified
  starts.

## Required Fixes

1. Reconcile the local goal JSON with the current no-hidden-start contract, or
   move a tracked goal spec into `docs/research_protocols/`.
2. Save verifier report artifacts for every promotion gate.
3. Track fixed-start case specs used for promoted matrices.
4. Add null/repeat baselines:
   - same narrative repeated;
   - shuffled narratives;
   - no-narrative or memory-only;
   - same-start different-seed runs.
5. Tighten gates with per-start quality floors, uncertainty, and visible
   operational warnings.
6. Expose start damping as a product trust warning before adding architecture
   knobs.
7. Keep `mlp_mse_contrastive + multi_caption_with_negatives` as incumbent until
   CLIP/SupCon beats target cosine and directional separation under explicit
   falsifiers.

## Follow-Up Actions Completed In Response

- Added tracked goal contract:
  `docs/research_protocols/nl_prefix_latent_goal.json`.
- Reconciled ignored local goal state from that tracked contract:
  `autoresearch-session/nl_prefix_latent_goal.json`.
- Added tracked current-truth index:
  `docs/research_protocols/nl_prefix_latent_current_truth.md`.
- Added tracked promoted case spec:
  `docs/research_protocols/nl_prefix_latent_promoted_specs/fixed_start_narrative_matrix_858b_cases.json`.
- Added this verifier report as a durable artifact.
