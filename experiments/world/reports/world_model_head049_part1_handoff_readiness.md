# World Model HEAD049: Part 1 Handoff Readiness

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Hypothesis / Falsifier

Hypothesis: the validated Part 1 reference can be handed to future downstream
forecasting, benchmark, or decoder work without reopening the JEPA objective if
the allowed inputs, forbidden mutations, and required caveats are explicit.

Falsifier: any ambiguity about which checkpoint, split contract, target
contract, or caveat a future run must use.

## Handoff Contract

Use `experiments/world/part1_jepa_latent/reference_manifest.json` as the source
of truth for the current Part 1 reference.

Allowed inputs:

- Primary checkpoint:
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt`.
- Supporting checkpoint:
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt`.
- Data object: `data/vol_surface_with_ret.npz`.
- Split/window contract:
  `experiments.world.evaluation.world_data.build_iv_world_windows` with
  history `30`, future `30`, `test_start=4511`, `val_size=441`, and normalized
  IV-surface coordinates.
- Target contract:
  `experiments.world.part1_jepa_latent.fixed_delta_pca_jepa` fixed whitened
  horizon delta-PCA targets over horizons `(1, 5, 10, 20, 30)`, target dim `8`,
  fit on the train split only.

Forbidden mutations:

- Do not refit the fixed PCA target on validation, test, or downstream
  evaluation windows.
- Do not change the split contract, history/future lengths, target dimension, or
  horizon set and still call the result the HEAD045/HEAD046 Part 1 reference.
- Do not add Barlow Twins, VICReg, retrieval/neighborhood objectives, target
  sweeps, or decoder feedback to Part 1 without a new documented Part 1
  failure.
- Do not use Part 2 generation metrics as evidence that Part 1 is healthy.
- Do not silently substitute the support seed for the primary seed; if the
  support checkpoint is used, say why and copy its metrics separately.

Required reporting fields for any future consumer:

- checkpoint path and seed;
- whether the consumer uses context only, predicted fixed-PCA future latents, or
  both;
- train/validation/test split used for any fitted downstream head;
- validation and held-out test Part 1 metrics copied from the reference
  manifest;
- caveat status for raw-delta retrieval versus coordinate/decoded-surface
  quality;
- statement that Barlow Twins is not the active objective.

## Decision

The current JEPA-style Part 1 reference is handoff-ready as a fixed upstream
conditioning object. Future work may consume it, but should not mutate the Part
1 training objective unless a new Part 1 failure is explicitly logged.

The next useful non-modeling step is to add a checksum/digest record for the
ignored local artifacts, so the exact data, checkpoint, and result JSON files
used by the manifest are identifiable without committing generated binaries.

## Artifacts

- `experiments/world/part1_jepa_latent/reference_manifest.json`
- `experiments/world/reports/world_model_head049_part1_handoff_readiness.md`
