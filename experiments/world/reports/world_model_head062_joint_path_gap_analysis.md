# World Model HEAD062: Joint Path-PCA Validation/Test Gap Analysis

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

After HEAD061 rejected promotion of HEAD060 on held-out test, is the failure an
audit-script mismatch, target-capacity problem, context collapse, or a genuine
validation/test generalization gap in the learned past-to-target map?

## Evidence

Ran the same HEAD061 audit harness on validation:

```bash
python experiments/world/part1_jepa_latent/joint_path_pca_probe_audit.py \
  --device cpu \
  --checkpoint models/world/checkpoints/part1_jepa_latent/joint_path_pca_head060.pt \
  --max_train_windows 2048 \
  --eval_split val --max_eval_windows 256 \
  --batch_size 128 --ridge_alpha 1e-3 \
  --output_json results/world/part1_joint_path_pca_probe_audit_head062_val.json
```

The validation audit reproduced the HEAD060 metrics, so HEAD061 is not a
measurement mismatch.

| metric | val audit | test audit |
| --- | ---: | ---: |
| context rank | 9.204105 | 10.139321 |
| context offdiag abs mean | 0.282073 | 0.248893 |
| trained path-code MRR | 0.133841 | 0.116251 |
| trained path-code top5 | 0.175781 | 0.160156 |
| trained decoded-delta MSE | 0.015530 | 0.019414 |
| trained decoded-delta MRR | 0.094780 | 0.081147 |
| trained decoded-delta top5 | 0.131250 | 0.117969 |
| ridge path-code MRR | 0.141879 | 0.115078 |
| ridge path-code top5 | 0.207031 | 0.132812 |
| ridge raw-delta MSE | 0.015436 | 0.019124 |
| ridge raw-delta MRR | 0.101809 | 0.101138 |
| ridge raw-delta top5 | 0.126563 | 0.130469 |
| PCA-oracle decoded-delta MSE | 0.004067 | 0.004093 |
| zero-delta MSE | 0.022376 | 0.025450 |

Against the active fixed-PCA reference:

| test gate | active reference | joint path-PCA |
| --- | ---: | ---: |
| trained decoded-delta MSE | 0.017769 | 0.019414 |
| raw-delta ridge MSE | 0.017923 | 0.019124 |
| raw-delta ridge MRR | 0.110315 | 0.101138 |
| raw-delta ridge top5 | 0.142969 | 0.130469 |

Relative MSE improvement over zero:

| split | trained head | ridge raw-delta |
| --- | ---: | ---: |
| val | 0.305974 | 0.310153 |
| test | 0.237149 | 0.248567 |

## Mechanism Read

The target object is not the main problem. The PCA oracle decoded-delta MSE is
stable from validation to test (`0.004067` to `0.004093`) and far below both
learned models, so the joint target has enough low-rank test capacity.

The context is not collapsed. Test context rank/offdiag are healthier than
validation and healthier than the active reference's test context. The failure
is therefore not a simple variance/rank issue.

The weak link is the learned mapping from past context to the joint path target.
Both the trained head and the context ridge probe lose path-code retrieval on
test, and raw-delta MSE degrades more than the active fixed-PCA reference. The
global joint path-PCA target appears to produce a healthier latent geometry but
not a more robust predictive state under the current train/validation/test
split.

## Decision

Keep HEAD038/HEAD042 as the active Part 1 reference.

Do not add Barlow/VICReg, retrieval/neighborhood loss, decoder work, or another
target knob to rescue HEAD060. The evidence says the next improvement should
not be a loss patch; it should revisit the predictive target contract or
selection protocol with split-aware evidence from the start.

## Next Step

The next Part 1 iteration should be research ideation, not an experiment:

- define what a split-robust fixed target must satisfy before training;
- decide whether the target should remain per-horizon fixed-PCA, become a
  horizon-factorized path target, or use a train-only target with explicit
  horizon tokens;
- require the candidate report to include validation and test gates before any
  promotion language.

Until that analysis exists, the safest model-facing action is to keep using the
active fixed-PCA reference.

## Artifacts

- `experiments/world/reports/world_model_head062_joint_path_gap_analysis.md`
- `results/world/part1_joint_path_pca_probe_audit_head062_val.json`
- `results/world/part1_joint_path_pca_probe_audit_head061_test.json`
