# World Model HEAD061: Joint Path-PCA Test Audit

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: the HEAD060 joint path-PCA candidate should keep its validation
advantage on the held-out test split when the checkpoint and train-fit PCA
target are reused, ridge probes are trained on train contexts only, and metrics
are evaluated on test windows.

Falsifier: test decoded-delta or raw-delta probe metrics fall behind the
existing HEAD042 primary reference test audit, even if validation metrics were
competitive.

## Implementation

Added `experiments/world/part1_jepa_latent/joint_path_pca_probe_audit.py`.

The audit:

- loads the HEAD060 checkpoint;
- reconstructs the saved train-fit joint path-PCA target from the checkpoint;
- evaluates the trained head on `test`;
- trains ridge probes on train contexts only;
- evaluates ridge probes on test path-PCA codes and raw horizon deltas;
- reports PCA-oracle and zero-delta baselines.

Added a focused parser test in `test_code/test_world_model_evaluation.py`.

No model objective, architecture, decoder, target sweep, retrieval loss, or
regularization knob was added.

## Validation

Focused checks:

```bash
pytest test_code/test_world_model_evaluation.py::test_joint_path_pca_target_roundtrips_and_predictor_shapes \
  test_code/test_world_model_evaluation.py::test_joint_path_pca_probe_audit_accepts_eval_split -q
```

Result: `2 passed in 0.75s`.

Compile check:

```bash
python -m py_compile experiments/world/part1_jepa_latent/joint_path_pca_probe_audit.py \
  test_code/test_world_model_evaluation.py
```

## Test-Split Run

```bash
python experiments/world/part1_jepa_latent/joint_path_pca_probe_audit.py \
  --device cpu \
  --checkpoint models/world/checkpoints/part1_jepa_latent/joint_path_pca_head060.pt \
  --max_train_windows 2048 \
  --eval_split test --max_eval_windows 256 \
  --batch_size 128 --ridge_alpha 1e-3 \
  --output_json results/world/part1_joint_path_pca_probe_audit_head061_test.json
```

Test context health:

- variance min `0.000906`, mean `0.044569`;
- effective rank `10.139321`, participation ratio `6.400367`;
- off-diagonal absolute mean `0.248893`.

Trained head on test:

- path-code MRR/top5/top10 `0.116251`/`0.160156`/`0.230469`;
- decoded-delta MSE/MRR/top5/top10
  `0.019414`/`0.081147`/`0.117969`/`0.185156`.

Frozen context ridge probes on test:

| probe | MSE | MRR | top5 | top10 |
| --- | ---: | ---: | ---: | ---: |
| joint path-PCA code | 1.133362 | 0.115078 | 0.132812 | 0.242188 |
| raw horizon deltas | 0.019124 | 0.101138 | 0.130469 | 0.205469 |

Baselines:

| baseline | MSE | MRR | top5 | top10 |
| --- | ---: | ---: | ---: | ---: |
| PCA oracle decoded deltas | 0.004093 | 0.399714 | 0.534375 | 0.646875 |
| zero deltas | 0.025450 | 0.023923 | 0.019531 | 0.039062 |

## Reference Comparison

| test gate | HEAD042 primary reference | HEAD061 joint path-PCA |
| --- | ---: | ---: |
| trained decoded-delta MSE | 0.017769 | 0.019414 |
| raw-delta ridge MSE | 0.017923 | 0.019124 |
| raw-delta ridge MRR | 0.110315 | 0.101138 |
| raw-delta ridge top5 | 0.142969 | 0.130469 |
| context effective rank | 7.534067 | 10.139321 |
| context offdiag abs mean | 0.310717 | 0.248893 |
| raw-delta MSE improvement over zero | 0.295762 | 0.248565 |

## Interpretation

HEAD061 rejects promotion of HEAD060 as the new Part 1 reference. The joint
path-PCA context remains healthier by rank/offdiag and still beats the
zero-delta baseline, but its test prediction and raw-delta retrieval gates are
weaker than the existing fixed-PCA reference.

The PCA oracle is strong on test, so the target object itself has enough
low-rank capacity. The gap is in learning a test-robust mapping from past
context to that target, not in the path-PCA decode ceiling.

This points to a split-generalization/selection issue rather than a collapse
issue.

## Decision / Next Step

Do not update `reference_manifest.json`.

Keep HEAD038/HEAD042 as the active Part 1 reference. Treat HEAD060/HEAD061 as a
useful diagnostic branch: joint path targets improve representation health and
validation raw-delta probes, but the gain does not transfer to held-out test.

Next iteration should be post-experiment analysis of the validation/test gap:
compare whether the failure is checkpoint selection by validation decoded MSE,
train-to-test regime drift, or the single global joint-PCA target. Do not add a
new objective until that failure class is understood.

## Artifacts

- `experiments/world/part1_jepa_latent/joint_path_pca_probe_audit.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/reports/world_model_head061_joint_path_pca_test_audit.md`
- `results/world/part1_joint_path_pca_probe_audit_head061_test.json`
