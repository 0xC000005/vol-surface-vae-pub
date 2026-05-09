# World Model HEAD046: Manifest Provenance Sanity

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: the Part 1 reference manifest created in HEAD045 should be internally valid, point to existing artifacts, and reproduce the key metrics from source JSON outputs within rounding tolerance.

Falsifier: the manifest is invalid JSON, references missing artifacts, or records metrics that do not match source results.

## Checks

- `python -m json.tool experiments/world/part1_jepa_latent/reference_manifest.json`
- `python -m json.tool autoresearch-session/world_model_state.json`
- Path-existence check over every checkpoint/result/report path referenced by the manifest.
- Metric reconciliation from source probe JSONs into the manifest summary table.

## Finding

The first reconciliation showed source metrics were consistent, but the manifest omitted `context_variance_mean`. That field is useful for the Part 1 non-collapse handoff, so the manifest was patched to include it for all four validation/test seed rows.

## Final Reconciliation

| label | checked fields | mismatches |
|---|---:|---:|
| validation seed 7711 | 12 | 0 |
| validation seed 7710 | 12 | 0 |
| test seed 7711 | 12 | 0 |
| test seed 7710 | 12 | 0 |

Checked fields:

- context effective rank;
- context off-diagonal absolute mean;
- context variance mean;
- trained decoded delta MSE;
- trained fixed-PCA MRR/top5;
- ridge fixed-PCA MRR/top5;
- ridge raw-delta MSE/MRR/top5;
- ridge raw-delta MSE improvement over zero baseline.

## Decision

The manifest is now provenance-consistent with the source artifacts. There is no evidence justifying more Part 1 modeling changes. The workflow should not add Barlow Twins, retrieval/neighborhood objectives, target sweeps, or decoder work unless explicitly requested or a new documented Part 1 failure appears.

## Artifacts

- `experiments/world/part1_jepa_latent/reference_manifest.json`
