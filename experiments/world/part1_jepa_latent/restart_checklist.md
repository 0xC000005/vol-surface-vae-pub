# World Model Part 1 Restart Checklist

Date: 2026-05-09

Use this before any future explicitly authorized model change that consumes or
modifies the Part 1 world-model reference.

## Read First

1. `docs/research_protocols/world_model_autoresearch_plan.md`
2. `experiments/world/part1_jepa_latent/reference_manifest.json`
3. `experiments/world/part1_jepa_latent/reference_artifact_digests.json`
4. `experiments/world/reports/world_model_head044_part1_reference_closeout.md`
5. `experiments/world/reports/world_model_head049_part1_handoff_readiness.md`
6. `experiments/world/reports/world_model_head051_part1_open_risk_ledger.md`
7. latest tail of `RESEARCH_LOG.md`

## Fixed Reference

- Primary checkpoint:
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt`.
- Support checkpoint:
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt`.
- Data path: `data/vol_surface_with_ret.npz`.
- Split/window contract: history `30`, future `30`, `test_start=4511`,
  `val_size=441`, normalized IV-surface coordinates.
- Target contract: fixed whitened train-fit horizon delta-PCA, horizons
  `(1, 5, 10, 20, 30)`, target dimension `8`.

## Before Any Experiment

- Verify `autoresearch-session/WORLD_MODEL_STOP` is absent.
- Verify `reference_artifact_digests.json` against local data/checkpoint/result
  files if the experiment consumes the saved artifacts.
- State whether the experiment consumes context, predicted fixed-PCA future
  latents, or both.
- State whether the experiment is Part 1, Part 2, or a downstream benchmark.
- Copy validation and test Part 1 metrics from `reference_manifest.json`.
- Preserve the caveat that raw-delta retrieval is not uniformly dominant versus
  older diagnostic retrieval contexts.

## Do Not Do Without Explicit Authorization

- Add Barlow Twins, VICReg, retrieval/neighborhood objectives, or other Part 1
  losses.
- Sweep target dimension, horizons, split contract, or history/future lengths.
- Start decoder training or conditional flow experiments.
- Refit fixed PCA targets on validation/test/downstream windows.
- Promote the reference to canonical ImageNet I-JEPA.
- Use generation metrics as evidence of Part 1 representation health.

## Required Reporting

Every future authorized experiment should report:

- hypothesis and falsifier;
- literature status if the objective is nonstandard;
- checkpoint path and seed;
- exact split/data contract;
- Part 1 and Part 2 metrics in separate sections;
- whether Part 1 was frozen, semi-frozen, or modified;
- explicit decision on whether the frozen reference remains valid.
