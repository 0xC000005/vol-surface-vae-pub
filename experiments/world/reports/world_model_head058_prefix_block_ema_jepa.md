# World Model HEAD058: Prefix-Block EMA JEPA

Date: 2026-05-09

Iteration type: `experiment`

## Literature Status

`canonical_jepa_adaptation`.

This is the HEAD057 target-block follow-up. It keeps the I-JEPA/A-JEPA-style
EMA target encoder and MSE context-to-target prediction objective, but changes
the time-series target block from an isolated future horizon-delta frame to the
future delta prefix ending at that horizon.

## Hypothesis / Falsifier

Hypothesis: future delta-prefix target blocks give the EMA target encoder a
richer and more stable target geometry than isolated horizon-delta frames,
improving learned-target retrieval/rank and frozen-context probe quality without
adding any new loss.

Falsifier:

- learned-target retrieval/top-k remains below HEAD056 or far below the
  fixed-PCA reference;
- predicted or target effective rank collapses below HEAD056;
- frozen-context fixed-PCA/raw-delta probes remain materially below the current
  fixed-PCA reference.

## Implementation

Updated `experiments/world/part1_jepa_latent/fused_context_ema_jepa.py` with a
single `target_block` option:

- `frame`: the HEAD056 target, `future[:, h-1:h] - past[:, -1:]`;
- `prefix`: the HEAD058 target, `future[:, :h] - past[:, -1:]`.

Added a focused test verifying that prefix targets use the full future block and
end at the same frame target.

No Barlow Twins, VICReg, retrieval/neighborhood loss, target sweep, decoder
component, or new scalar loss weight was added.

## Validation

```text
pytest test_code/test_world_model_evaluation.py::test_fused_context_ema_jepa_uses_relative_context_and_frozen_target \
  test_code/test_world_model_evaluation.py::test_fused_context_ema_jepa_prefix_targets_use_future_blocks -q

python -m py_compile experiments/world/part1_jepa_latent/fused_context_ema_jepa.py test_code/test_world_model_evaluation.py
```

Focused test result:

```text
2 passed in 0.75s
```

Real-data run:

```text
python experiments/world/part1_jepa_latent/fused_context_ema_jepa.py \
  --device cpu --epochs 20 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --hidden_dim 64 --latent_dim 16 --predictor_hidden_dim 128 \
  --ema_decay 0.99 --target_block prefix --seed 7713 \
  --output_json results/world/part1_fused_context_prefix_ema_jepa_head058.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_prefix_ema_jepa_head058.pt
```

## Result

Best learned-target epoch:

| metric | HEAD056 frame | HEAD058 prefix |
| --- | ---: | ---: |
| epoch | 15 | 19 |
| learned-target MSE | 0.097189 | 0.125657 |
| learned-target MRR | 0.054649 | 0.060093 |
| learned-target top5 | 0.057031 | 0.067188 |
| predicted effective rank | 2.883115 | 3.776887 |
| target effective rank | 3.806809 | 4.175854 |
| context effective rank | 4.441587 | 4.359951 |

Frozen-context probes:

| probe | HEAD056 frame | HEAD058 prefix | fixed-PCA reference |
| --- | ---: | ---: | ---: |
| fixed-PCA MSE | 1.240337 | 1.247527 | lower is better |
| fixed-PCA MRR | 0.069033 | 0.065450 | 0.103763 |
| fixed-PCA top5 | 0.086719 | 0.081250 | 0.136719 |
| raw-delta MSE | 0.019864 | 0.019055 | 0.015701 |
| raw-delta MRR | 0.067651 | 0.060700 | 0.096147 |
| raw-delta top5 | 0.074219 | 0.069531 | 0.128125 |

## Mechanism Read

The target-block hypothesis is partially right:

- prefix blocks improve learned-target MRR/top5 over HEAD056;
- prefix blocks improve predicted and target effective rank;
- raw-delta probe MSE improves slightly versus HEAD056.

But the improvement is not enough for the Part 1 gate:

- fixed-PCA and raw-delta retrieval probes both get worse versus HEAD056;
- all frozen-context probe retrieval metrics remain materially below the
  fixed-PCA reference;
- learned-target MRR/top5 are still far below what a useful downstream
  representation should show.

Primary failure class remains `latent_prediction`, now less clearly `collapse`
and more clearly target geometry / predictive information mismatch.

## Decision

Do not promote HEAD058. Keep the fixed-PCA fused-context reference as the active
Part 1 reference.

The canonical EMA branch has now improved from HEAD025 -> HEAD056 -> HEAD058,
but it remains below the fixed target reference. The next Part 1 move should be
post-experiment analysis, not another immediate target tweak: decide whether to
keep pursuing EMA target encoders or pivot to a more principled fixed target
contract for world-state learning, such as masked low-rank future path targets
with explicit frozen-probe gates.

## Artifacts

- `experiments/world/part1_jepa_latent/fused_context_ema_jepa.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/reports/world_model_head058_prefix_block_ema_jepa.md`
- `results/world/part1_fused_context_prefix_ema_jepa_head058.json`
- `models/world/checkpoints/part1_jepa_latent/fused_context_prefix_ema_jepa_head058.pt`
