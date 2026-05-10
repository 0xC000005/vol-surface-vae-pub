# World Model HEAD154: Surface-Local Token JEPA Smoke

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`token_geometry_level_context_to_target_jepa` smoke training.

## Hypothesis

A token/geometry-level context-to-target scaffold should train without
future targets or value reconstruction and should expose whether target-token
alignment is healthy or another low-rank/high-cosine shortcut appears.

## Run

- Train windows: `128`.
- Validation windows: `64`.
- Train target token rows: `14751`.
- Validation target token rows: `6962`.
- EMA decay: `0.990000`.

## Metrics

- Initial validation loss: `0.417678`.
- Final validation loss: `0.233348`.
- Validation alignment: `0.200579`.
- Validation cosine mean: `0.692233`.
- Validation retrieval top10 on subset: `0.054688`.
- Retrieval subset rows: `512`.
- Predicted effective rank: `3.945894`.
- Target effective rank: `4.750035`.
- Predicted offdiag abs mean: `0.415029`.
- Target offdiag abs mean: `0.402760`.

## Decision

Promotion decision: `SMOKE_ONLY_DO_NOT_PROMOTE`.

The loss is trainable, but target-token quality is weak: retrieval top10 is only
`0.054688` on a 512-row subset and predicted effective rank is `3.945894` in a
24-dimensional latent. This looks like another low-rank latent shortcut rather
than a solved Part 1 representation.

This is smoke evidence only. It does not certify Part 1, does not start
Part B, and should be followed by a diagnostic comparison against the scaled
Barlow candidate and raw exact-state baselines before any tuning.
