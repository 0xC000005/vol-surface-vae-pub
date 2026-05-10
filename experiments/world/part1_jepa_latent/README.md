# Part 1: JEPA Latent World Model

Purpose: learn a robust market-state representation from multivariate
time-series windows without hand-defining slow state variables.

Current default training form:

```text
same market window / same relative time position
-> structured masked view A + observed/synthetic mask channels
-> shared encoder -> z_a
same market window / same relative time position
-> structured masked view B + observed/synthetic mask channels
-> shared encoder -> z_b
```

Current default loss surface:

```text
L =
    L_barlow_style_same_state_alignment(z_a, z_b)
```

Variance/covariance/VICReg-style controls are allowed only as explicitly
justified representation-health controls on the evaluated embeddings. They are
not forecasting losses, and they should not be added as knobs unless a specific
Part 1 failure is documented.

Quality gates for this part should be representation-focused:

- same-state masked-view alignment,
- same-state retrieval top-k or MRR,
- Barlow/cross-correlation diagonal and off-diagonal terms,
- embedding variance and effective rank,
- off-diagonal covariance/correlation norm,
- mask-artifact diagnostics,
- frozen probes for future/range/state summaries after pretraining.

Do not align arbitrary time windows. The positive pair for the current branch is
two structured masked views of the same window at the same relative index. EMA
targets and predictor heads belong to a separately gated context-to-target JEPA
experiment, not the default masked-multiview objective.

## Current Smoke

- `jepa_smoke.py`: minimal IV-only GRU context encoder, EMA target encoder, and
  predictor. Use it as a Part 1 falsifier, not as the final architecture.
- `masked_multiview_barlow_smoke.py`: current direct two-view reference path:
  one shared encoder, two structured masked views, Barlow-style redundancy
  reduction applied directly to the evaluated embeddings.

Current package pointers:

- `reference_manifest.json`: HEAD070 masked-multiview reference candidate and
  caveats.
- `reference_artifact_digests.json`: local ignored artifact identities for the
  HEAD070 package.
- `reference_package_check.py`: verifies manifest report paths, guardrail-doc
  caveat terms, and local artifact byte/SHA-256 identities.
- `package_summary.md` and `restart_checklist.md`: restart/handoff guardrails.

Current caveat boundary:

- HEAD070 is a smoke-scale reference candidate: `384` train windows, `128`
  validation windows, `8` epochs.
- The validated mask-policy claim covers the default structured mask families
  only; richer wing/ATM/whole-surface/cross-family stress masks need separate
  evidence.
- Downstream probe evidence is mixed and targets IV-surface futures only.
- Do not claim ImageNet-level JEPA behavior, full-data convergence, solved
  regime classification, factor-panel future target performance, or Part 2
  scenario-generation quality from this Part 1 package.
