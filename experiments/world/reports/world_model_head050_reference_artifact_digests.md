# World Model HEAD050: Reference Artifact Digests

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Hypothesis / Falsifier

Hypothesis: a tracked digest record can make the ignored local artifacts behind
the Part 1 reference identifiable without committing generated data,
checkpoints, or metric JSONs.

Falsifier: any referenced ignored artifact is missing, tracked unexpectedly, not
ignored by existing rules, or cannot be identified by a stable byte size and
SHA-256 digest.

## Checks

- Stop sentinel: absent.
- Ignored artifact count checked: `9`.
- Missing ignored artifacts: `0`.
- Unexpected tracked ignored artifacts: `0`.
- Unexpected non-ignored generated artifacts: `0`.
- Digest verification: recomputed `9` SHA-256 hashes from disk with `0`
  mismatches.

## Digest Record

Added `experiments/world/part1_jepa_latent/reference_artifact_digests.json`.

The digest record covers:

- `data/vol_surface_with_ret.npz`;
- the primary seed `7711` checkpoint and training/probe JSONs;
- the support seed `7710` checkpoint and training/probe JSONs.

## Decision

The Part 1 reference now has three separate handoff layers:

- tracked manifest for semantic contract and metrics;
- tracked report chain for rationale and caveats;
- tracked digest record for ignored local artifact identity.

This remains a provenance-only iteration. No JEPA objective, target, decoder,
retrieval/neighborhood loss, Barlow/VICReg term, or sweep was added.

The next useful non-modeling step is a compact open-risk ledger for the frozen
Part 1 reference: what is settled, what remains only caveated, and which future
work would require explicit user authorization.

## Artifacts

- `experiments/world/part1_jepa_latent/reference_artifact_digests.json`
- `experiments/world/reports/world_model_head050_reference_artifact_digests.md`
