# World Model HEAD042: Split-Aware Probe Audit

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: the fused-context fixed-target probe audit should be reusable on held-out test windows, training ridge probes only on train and evaluating on the requested split. The primary seed `7711` should retain non-collapse, fixed-PCA retrieval, and meaningful raw-delta improvement over a zero-delta baseline on `test`.

Falsifier: adding split-aware evaluation breaks the validation path, test contexts collapse, or test raw-delta probes fail to beat the zero-delta baseline.

## Implementation

- Added `--eval_split {val,test}` and `--max_eval_windows` to `experiments/world/part1_jepa_latent/fused_context_probe_audit.py`.
- Kept backward compatibility by preserving `val_shape` and `val_context_health` while adding `eval_shape`, `eval_context_health`, and `eval_split`.
- Added a focused parser test in `test_code/test_world_model_evaluation.py`.
- No model objective, target, decoder, retrieval loss, or Barlow/VICReg-style term was added.

## Validation

- Red test first failed with `ImportError: cannot import name 'build_arg_parser'`.
- Focused test passed: `pytest test_code/test_world_model_evaluation.py::test_fused_context_probe_audit_accepts_eval_split -q`.
- Full validation passed: `pytest test_code/test_world_model_evaluation.py -q` returned `28 passed in 0.75s`.
- Compile check passed for the audit script and test file.

## Test-Split Run

```bash
python experiments/world/part1_jepa_latent/fused_context_probe_audit.py \
  --device cpu \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head038_seed7711.pt \
  --max_train_windows 2048 \
  --eval_split test --max_eval_windows 256 \
  --batch_size 128 --ridge_alpha 1e-3 \
  --output_json results/world/part1_fused_context_probe_audit_head042_test_seed7711.json
```

Test context health:

- variance min `0.011115`, mean `0.121553`;
- effective rank `7.534067`, participation ratio `4.909533`;
- off-diagonal absolute mean `0.310717`.

Trained head to fixed delta-PCA on test:

- MSE `1.057194`;
- MRR `0.119919`, top5 `0.155469`, top10 `0.235937`;
- decoded delta MSE `0.017769`;
- predicted rank `4.735351`.

Frozen context ridge probe to fixed delta-PCA on test:

- MSE `1.078600`;
- MRR `0.116466`, top5 `0.148438`, top10 `0.220312`;
- predicted rank `5.045933`.

Frozen context ridge probe to raw horizon deltas on test:

- MSE `0.017923`;
- MRR `0.110315`, top5 `0.142969`, top10 `0.223438`;
- predicted rank `2.730868`.

Zero-delta test baseline:

- MSE `0.025450`;
- MRR `0.023923`, top5 `0.019531`, top10 `0.039062`.

## Validation/Test Comparison

| metric | val HEAD039 seed 7711 | test HEAD042 seed 7711 |
|---|---:|---:|
| context rank | 6.648142 | 7.534067 |
| context offdiag | 0.323006 | 0.310717 |
| trained decoded delta MSE | 0.015176 | 0.017769 |
| trained fixed-PCA MRR | 0.103029 | 0.119919 |
| fixed-PCA ridge MRR | 0.103763 | 0.116466 |
| raw-delta ridge MSE | 0.015701 | 0.017923 |
| raw-delta ridge MRR | 0.096147 | 0.110315 |
| raw-delta MSE improvement over zero | 0.298315 | 0.295762 |

## Interpretation

- Held-out test context health is at least as good as validation by rank/offdiag.
- Test decoded-delta and raw-delta MSE are worse in absolute terms, but the zero-delta baseline is also worse; relative raw-delta MSE improvement remains essentially unchanged.
- Test retrieval is stronger than validation on fixed-PCA and raw-delta MRR/top-k.
- This passes the primary seed test-split probe check.

## Decision / Next Step

- Primary seed test-split Part 1 probe passes.
- Do not move to the decoder yet.
- Next HEAD should run the same split-aware test audit on the supporting seed `7710`. If that also passes, the fused fixed-target JEPA-style representation can be treated as the current Part 1 reference for downstream decoder-conditioning experiments, with the caveat that raw-delta top5 is not uniformly best versus older diagnostic retrieval contexts.

## Artifacts

- `experiments/world/part1_jepa_latent/fused_context_probe_audit.py`
- `test_code/test_world_model_evaluation.py`
- Ignored output: `results/world/part1_fused_context_probe_audit_head042_test_seed7711.json`
