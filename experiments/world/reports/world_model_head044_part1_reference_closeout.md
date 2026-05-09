# World Model HEAD044: Part 1 Reference Closeout

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

After HEAD039-HEAD043, is the JEPA-style Part 1 representation sufficiently fixed to stop adding Part 1 knobs and treat the fused-context fixed delta-PCA predictor as the current reference?

## Reference Definition

Current Part 1 reference:

- Architecture: fused context encoder with GRU sequence branch plus direct flattened-past branch.
- Target contract: fixed whitened delta-PCA target over horizons `(1, 5, 10, 20, 30)`.
- Objective: MSE to fixed future-latent targets.
- Primary checkpoint: `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt`.
- Supporting checkpoint: `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt`.

This is a JEPA-style time-series model because it predicts future target representations from past context representations. It is not a canonical ImageNet I-JEPA reproduction because the target space is fixed PCA rather than an EMA target encoder. The proper label remains `supported_adjacent`.

## Part 1 Evidence

| split/seed | ctx rank | offdiag | trained decode MSE | trained fixed MRR/top5 | fixed ridge MRR/top5 | raw ridge MSE | raw ridge MRR/top5 | raw MSE improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val seed 7711 | 6.648142 | 0.323006 | 0.015176 | 0.103029/0.133594 | 0.103763/0.136719 | 0.015701 | 0.096147/0.128125 | 0.298315 |
| val seed 7710 | 6.551514 | 0.347958 | 0.015363 | 0.117733/0.157812 | 0.118667/0.157031 | 0.015880 | 0.106107/0.125000 | 0.290307 |
| test seed 7711 | 7.534067 | 0.310717 | 0.017769 | 0.119919/0.155469 | 0.116466/0.148438 | 0.017923 | 0.110315/0.142969 | 0.295762 |
| test seed 7710 | 7.508093 | 0.321090 | 0.017537 | 0.124176/0.175781 | 0.131317/0.191406 | 0.018021 | 0.115288/0.160156 | 0.291907 |

## What Is Fixed

- Non-collapse: context rank is stable across validation and test, and improves on held-out test.
- Fixed-target prediction: both trained heads and frozen linear probes recover fixed-PCA neighborhood structure.
- Decoded coordinate quality: trained fixed-PCA decoded-delta MSE remains consistently better than zero-delta/persistence-style baselines and stronger than earlier raw-delta supervised heads.
- Cross-seed robustness: both seeds pass validation and test probes.
- Split robustness: the primary and supporting seeds pass held-out test probing with similar raw-delta MSE improvement over zero baseline.

## Caveats

- The reference is fixed-target JEPA-style, not canonical EMA-target I-JEPA.
- Raw-delta retrieval is not uniformly best versus older diagnostic retrieval/contrastive contexts. Those older contexts remain useful retrieval baselines, but prior evidence says they lose forecast/decode gates or rely on noncanonical objectives.
- Barlow Twins is not the active method. Variance/covariance ideas are used as diagnostics and possible adjacent collapse-control framing, not as the current objective.
- Part 1 success does not prove Part 2 scenario-generation quality.

## Minimum Handoff Criteria For Later Decoder Work

Before starting a flow decoder, the handoff artifact should identify:

1. the exact checkpoint path and seed used for conditioning;
2. whether conditioning uses context only, predicted future fixed-PCA latents, or both;
3. the split and data window contract;
4. Part 1 metrics copied from this closeout table;
5. the caveat that raw-delta retrieval top-k is not uniformly dominant versus older retrieval diagnostics.

Decoder work should not modify the Part 1 objective unless a future Part 1 failure is discovered.

## Decision

Part 1 JEPA-style representation is fixed enough for the current workflow. Stop adding Part 1 losses, target sweeps, Barlow-style regularizers, or retrieval/neighborhood objectives. The next useful non-decoder step is to package this reference into a small manifest so later decoder or benchmark work cannot accidentally use the wrong checkpoint or metric contract.

## Artifacts

- `experiments/world/reports/world_model_head039_fused_context_probe_audit.md`
- `experiments/world/reports/world_model_head040_probe_baseline_comparison.md`
- `experiments/world/reports/world_model_head041_fused_context_probe_seed_check.md`
- `experiments/world/reports/world_model_head042_split_aware_probe_audit.md`
- `experiments/world/reports/world_model_head043_support_seed_test_probe.md`
