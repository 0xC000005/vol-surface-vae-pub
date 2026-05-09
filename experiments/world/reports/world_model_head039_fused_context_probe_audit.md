# World Model HEAD039: Fused Context Probe Audit

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: the fused-context fixed delta-PCA model promoted in HEAD038 learns a reusable Part 1 context representation, not just a trained horizon head. A frozen linear probe from the context should recover useful fixed-PCA future targets and raw horizon deltas while the context remains non-collapsed.

Falsifier: frozen context probes cannot approach the trained head's retrieval/top-k signal, cannot beat a zero-delta baseline in raw delta space, or show collapsed/near-constant context health.

## Implementation

- Added `experiments/world/part1_jepa_latent/fused_context_probe_audit.py`.
- The audit loads the HEAD038 seed `7711` fused-context checkpoint.
- It freezes the context encoder and trains closed-form ridge probes from context to:
  - fixed delta-PCA future targets;
  - raw horizon delta targets.
- It reuses existing Part 1 health, prediction, retrieval, and decoded-delta metrics.
- It does not add a decoder, retrieval loss, neighborhood loss, target sweep, or new training objective.

## Validation

- Focused red test first failed with `ModuleNotFoundError` for the new audit module.
- Focused test passed: `pytest test_code/test_world_model_evaluation.py::test_fused_context_probe_audit_encodes_contexts -q`.
- Full validation passed: `pytest test_code/test_world_model_evaluation.py -q` returned `27 passed in 0.76s`.
- Compile check passed: `python -m py_compile experiments/world/part1_jepa_latent/fused_context_probe_audit.py test_code/test_world_model_evaluation.py`.

## Result

Run command:

```bash
python experiments/world/part1_jepa_latent/fused_context_probe_audit.py \
  --device cpu \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt \
  --max_train_windows 2048 --max_val_windows 256 \
  --batch_size 128 --ridge_alpha 1e-3 \
  --output_json results/world/part1_fused_context_probe_audit_head039.json
```

Context health on validation contexts:

- variance min `0.012638`, mean `0.153847`;
- effective rank `6.648142`, participation ratio `4.448062`;
- off-diagonal absolute mean `0.323006`, max `0.861041`.

Trained horizon head to fixed delta-PCA target:

- MSE `1.000193`, cosine `0.457203`;
- MRR `0.103029`, top1 `0.039062`, top5 `0.133594`, top10 `0.226562`;
- decoded delta MSE `0.015176`;
- predicted rank `4.585648`.

Frozen context ridge probe to fixed delta-PCA target:

- MSE `1.031914`, cosine `0.397176`;
- MRR `0.103763`, top1 `0.042187`, top5 `0.136719`, top10 `0.211719`;
- predicted rank `4.817703`.

Frozen context ridge probe to raw horizon deltas:

- MSE `0.015701`, cosine `0.403821`;
- MRR `0.096147`, top1 `0.037500`, top5 `0.128125`, top10 `0.196875`;
- predicted rank `3.016013`.

Zero-delta target baseline:

- MSE `0.022376`;
- MRR `0.023923`, top5 `0.019531`, top10 `0.039062`.

## Mechanism Read

- The context is not collapsed: the validation effective rank is `6.65` in a 32-dimensional context, with non-trivial per-dimension variance.
- A frozen linear probe from the fused context matches or slightly exceeds the trained head on fixed-PCA retrieval MRR/top5 (`0.103763`/`0.136719` versus `0.103029`/`0.133594`), so the context itself carries the neighborhood structure.
- The trained horizon head remains useful for coordinate accuracy: fixed-PCA MSE is better for the trained head (`1.000193`) than for the ridge probe (`1.031914`).
- The raw-delta probe beats the zero-delta baseline by a large margin on both MSE and retrieval, which supports future-state information in the representation beyond a degenerate persistence baseline.

## Decision / Next Step

- HEAD039 passes the frozen-probe audit as a Part 1 representation-quality check for the fused-context fixed-target model.
- Do not move to the decoder yet; keep the focus on JEPA Part 1.
- Next HEAD should be post-experiment analysis comparing HEAD039's probe results with earlier context-probe baselines and deciding whether the remaining Part 1 risk is test-split robustness, cross-seed probe robustness, or horizon-specific weakness.

## Artifacts

- `experiments/world/part1_jepa_latent/fused_context_probe_audit.py`
- `test_code/test_world_model_evaluation.py`
- Ignored output: `results/world/part1_fused_context_probe_audit_head039.json`
