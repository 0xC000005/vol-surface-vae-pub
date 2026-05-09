# World Model HEAD059: Part 1 Branch Decision

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

After HEAD056 and HEAD058, should Part 1 keep pursuing EMA target encoders, or
should it pivot to a more principled fixed target contract for world-state
learning?

## Evidence

EMA learned-target branch:

| run | target | MRR | top5 | pred rank | target rank |
| --- | --- | ---: | ---: | ---: | ---: |
| HEAD025 | delta frame, GRU context | 0.042356 | 0.043750 | 1.262683 | 2.077708 |
| HEAD056 | delta frame, fused relative context | 0.054649 | 0.057031 | 2.883115 | 3.806809 |
| HEAD058 | delta prefix, fused relative context | 0.060093 | 0.067188 | 3.776887 | 4.175854 |

Frozen-context probes:

| run | fixed-PCA MRR/top5 | raw-delta MSE/MRR/top5 |
| --- | ---: | ---: |
| HEAD056 | 0.069033 / 0.086719 | 0.019864 / 0.067651 / 0.074219 |
| HEAD058 | 0.065450 / 0.081250 | 0.019055 / 0.060700 / 0.069531 |
| fixed-PCA reference | 0.103763 / 0.136719 | 0.015701 / 0.096147 / 0.128125 |

## Mechanism Read

The EMA branch is improving monotonically in learned-target retrieval and rank,
but those gains are not translating into the frozen-context probes that matter
for a reusable world state. The target encoder learns a geometry that the
predictor can partially retrieve, but that geometry is still weaker than the
fixed-PCA future-delta contract for downstream fixed-PCA and raw-delta probes.

This is no longer just rank collapse. It is a target-geometry mismatch:

- isolated and prefix EMA targets are richer than earlier EMA runs;
- neither creates a context state that exposes future fixed-PCA/raw-delta
  structure at the current reference level;
- adding retrieval losses, variance/covariance weights, or target sweeps would
  be objective patching rather than a principled fix.

## Decision

Pause EMA target-encoder work. The next Part 1 experiment should stay
JEPA-style but pivot to a fixed target contract that better represents the
future world state:

```text
future horizon-delta path over horizons (1, 5, 10, 20, 30)
-> flatten full H x C path
-> train-fit whitened PCA path code
past -> fused context -> single future-path code
loss = MSE(z_pred_path, z_target_path)
```

This is not canonical ImageNet I-JEPA, but it is a more principled fixed target
than independent per-horizon PCA because it preserves cross-horizon/cross-cell
future path covariance in one target object.

## Falsifier For Next Experiment

The joint path-PCA target fails if:

- decoded horizon-delta MSE is not competitive with the current reference;
- path-code retrieval is weak;
- frozen context remains poor on raw-delta probes;
- context health collapses.

## Artifacts

- `experiments/world/reports/world_model_head059_part1_branch_decision.md`
