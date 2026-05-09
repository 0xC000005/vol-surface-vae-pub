# World Model Part 1 Frozen Package Summary

Date: 2026-05-09

## Status

The current Part 1 world-model reference is frozen for this workflow. It should
be consumed as a fixed upstream representation object unless a future Part 1
failure is explicitly documented and the literature gate is satisfied.

## Reference

- Model family: fused-context fixed delta-PCA predictor.
- Literature status: `supported_adjacent`.
- Primary seed/checkpoint:
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt`.
- Support seed/checkpoint:
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt`.
- Manifest:
  `experiments/world/part1_jepa_latent/reference_manifest.json`.
- Artifact digests:
  `experiments/world/part1_jepa_latent/reference_artifact_digests.json`.
- Restart checklist:
  `experiments/world/part1_jepa_latent/restart_checklist.md`.

## Fixed Contract

- Data: `data/vol_surface_with_ret.npz`.
- Windowing: history `30`, future `30`.
- Split: `test_start=4511`, `val_size=441`.
- Coordinates: normalized IV-surface coordinates.
- Target: fixed whitened train-fit horizon delta-PCA.
- Horizons: `(1, 5, 10, 20, 30)`.
- Target dimension: `8`.
- Loss: MSE to fixed future-latent targets.

## Evidence

- Validation and test metrics are recorded in
  `experiments/world/part1_jepa_latent/reference_manifest.json`.
- Provenance was checked in:
  `experiments/world/reports/world_model_head046_manifest_provenance_sanity.md`.
- Local artifact identity was recorded and verified in:
  `experiments/world/reports/world_model_head050_reference_artifact_digests.md`.
- Open risks and authorization boundaries were recorded in:
  `experiments/world/reports/world_model_head051_part1_open_risk_ledger.md`.

## Caveats

- This is JEPA-style time-series representation prediction, not canonical
  ImageNet I-JEPA.
- Barlow Twins is not the active objective.
- Raw-delta retrieval is not uniformly dominant versus older diagnostic
  retrieval contexts.
- Part 1 success does not prove Part 2 scenario-generation quality.

## Next Work Requires Direction

Future work should be one of:

- explicitly authorized Part 2 decoder work using the frozen reference;
- explicitly authorized Part 1 research after a new documented failure;
- downstream benchmark/probe work that consumes the frozen reference without
  mutating it.
