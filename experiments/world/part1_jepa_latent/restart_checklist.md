# World Model Part 1 Restart Checklist

Date: 2026-05-09

Use this before any future model change, decoder experiment, or downstream
probe that consumes the Part 1 world-model reference candidate.

## Read First

1. `docs/research_protocols/world_model_autoresearch_plan.md`
2. `experiments/world/part1_jepa_latent/reference_manifest.json`
3. `experiments/world/part1_jepa_latent/reference_artifact_digests.json`
4. `experiments/world/reports/world_model_head082_part1_scorecard_health.md`
5. `experiments/world/reports/world_model_head083_mask_artifact_audit.md`
6. `experiments/world/reports/world_model_head084_stratified_mask_audit.md`
7. `experiments/world/reports/world_model_head086_downstream_probe_interpretation.md`
8. latest tail of `RESEARCH_LOG.md`

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
- Objective: direct masked-multiview same-state encoder alignment with
  Barlow-style redundancy control.

## Before Any Experiment

- Verify `autoresearch-session/WORLD_MODEL_STOP` is absent.
- Verify artifact digests if the experiment consumes ignored checkpoint,
  result, or data files.
  Use:
  `python experiments/world/part1_jepa_latent/reference_package_check.py`.
- State whether the experiment is Part 1, Part 2, or a downstream probe.
- State the objective family: `masked_multiview_invariance`,
  `context_to_target_jepa`, or `downstream_probe`.
- Keep future prediction, range estimation, regime labels, and generation as
  downstream probes unless the workflow is explicitly changed.
- Copy the acceptance boundary from `reference_manifest.json`.

## Do Not Do Without Explicit Authorization

- Start Part 2 decoder training.
- Add Part 1 model knobs or objective terms from HEAD085 alone.
- Reintroduce EMA/predictor same-state routing for two corrupted views.
- Treat fixed delta-PCA prediction as the current active reference.
- Promote regime classification as solved.
- Claim ImageNet-level JEPA behavior.
- Use generation metrics as evidence of Part 1 representation health.

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
