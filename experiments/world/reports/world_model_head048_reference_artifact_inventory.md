# World Model HEAD048: Part 1 Reference Artifact Inventory

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Hypothesis / Falsifier

Hypothesis: the HEAD045/HEAD046 Part 1 reference manifest is usable as a
handoff contract because every referenced artifact exists locally, and future
work can tell which dependencies are versioned versus local generated outputs.

Falsifier: any missing manifest path, unclear tracked/ignored status, or an
artifact dependency that would make the validated Part 1 reference impossible to
reconstruct or audit from the repository plus local generated outputs.

## Inventory

| path | exists | tracked | ignored | bytes |
| --- | ---: | ---: | ---: | ---: |
| `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt` | true | false | true | 361367 |
| `results/world/part1_fused_context_delta_pca_head038_seed7711.json` | true | false | true | 49505 |
| `results/world/part1_fused_context_probe_audit_head039.json` | true | false | true | 30489 |
| `results/world/part1_fused_context_probe_audit_head042_test_seed7711.json` | true | false | true | 31990 |
| `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt` | true | false | true | 360960 |
| `results/world/part1_fused_context_delta_pca_head036.json` | true | false | true | 49472 |
| `results/world/part1_fused_context_probe_audit_head041_seed7710.json` | true | false | true | 30481 |
| `results/world/part1_fused_context_probe_audit_head043_test_seed7710.json` | true | false | true | 31978 |
| `experiments/world/reports/world_model_head039_fused_context_probe_audit.md` | true | true | false | 4359 |
| `experiments/world/reports/world_model_head040_probe_baseline_comparison.md` | true | true | false | 5132 |
| `experiments/world/reports/world_model_head041_fused_context_probe_seed_check.md` | true | true | false | 3289 |
| `experiments/world/reports/world_model_head042_split_aware_probe_audit.md` | true | true | false | 4248 |
| `experiments/world/reports/world_model_head043_support_seed_test_probe.md` | true | true | false | 3669 |
| `experiments/world/reports/world_model_head044_part1_reference_closeout.md` | true | true | false | 4446 |
| `experiments/world/part1_jepa_latent/reference_manifest.json` | true | true | false | 5438 |
| `experiments/world/part1_jepa_latent/fixed_delta_pca_jepa.py` | true | true | false | 16395 |
| `experiments/world/evaluation/world_data.py` | true | true | false | 5022 |
| `data/vol_surface_with_ret.npz` | true | false | true | 1398758 |

## Findings

- Missing artifact count: `0`.
- The audit trail is split cleanly: reports, source code, and the reference
  manifest are tracked; data, checkpoints, and generated metric JSONs are local
  generated artifacts covered by existing ignore rules.
- The current Part 1 handoff is therefore reproducible only on a workspace that
  has the local ignored artifacts, but it remains auditable from tracked reports
  and the manifest.
- No model objective, target, decoder, retrieval loss, Barlow/VICReg loss, or
  target sweep was introduced.

## Decision

Keep the fused-context fixed delta-PCA checkpoint as the current Part 1
reference, with the existing caveat that it is strongest on coordinate and
decoded-surface quality and not uniformly dominant on raw-delta top-k retrieval.

The next non-modeling step, if the loop continues without decoder authorization,
should be a handoff-readiness note: what a future decoder or benchmark run may
consume, what it must not mutate, and which caveats must appear in any Part 2
report.

## Artifacts

- `experiments/world/part1_jepa_latent/reference_manifest.json`
- `experiments/world/reports/world_model_head048_reference_artifact_inventory.md`
