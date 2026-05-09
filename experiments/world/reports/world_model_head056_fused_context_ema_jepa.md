# World Model HEAD056: Fused-Context EMA JEPA

Date: 2026-05-09

Iteration type: `experiment`

## Literature Status

`canonical_jepa_adaptation`.

Primary-source check:

- I-JEPA frames the central mechanism as predicting representations of target
  blocks from a context block, with target/context block design as a core
  modeling choice: <https://arxiv.org/abs/2301.08243>.
- A-JEPA applies the same JEPA pattern to audio and explicitly uses an EMA
  target encoder with time-frequency-aware masking: <https://arxiv.org/abs/2311.15830>.
- DMT-JEPA adds neighbor-derived discriminative targets:
  <https://arxiv.org/abs/2405.17995>. This was not used because it would reopen
  the retrieval/neighborhood objective path.

## Hypothesis / Falsifier

Hypothesis: the old EMA JEPA failures were partly a context bottleneck. If the
fused GRU/direct context from the current fixed-PCA reference is paired with a
canonical-style EMA target encoder over future horizon-delta frames, learned
target prediction should beat the old EMA runs and become competitive with the
fixed-PCA reference on retrieval and frozen-context probes.

Falsifier:

- learned-target MRR/top5 remains far below the fixed-PCA reference;
- frozen fused contexts do not linearly probe to fixed-PCA or raw-delta targets
  near the reference metrics;
- predicted or target ranks collapse toward the old EMA failure pattern.

## Implementation

Added `experiments/world/part1_jepa_latent/fused_context_ema_jepa.py`.

Contract:

```text
past -> relative_to_last(past)
relative past -> GRU branch + direct flattened branch -> fused context
fused context + horizon token -> z_pred
future horizon delta frame -> EMA target encoder -> stopgrad(z_target)
loss = MSE(z_pred, z_target)
```

No Barlow Twins, VICReg, retrieval/neighborhood loss, target sweep, decoder
component, or fixed-PCA prediction loss was added.

## Validation

```text
pytest test_code/test_world_model_evaluation.py::test_fused_context_ema_jepa_uses_relative_context_and_frozen_target -q
python -m py_compile experiments/world/part1_jepa_latent/fused_context_ema_jepa.py test_code/test_world_model_evaluation.py
```

Focused test result:

```text
1 passed in 0.74s
```

Real-data run:

```text
python experiments/world/part1_jepa_latent/fused_context_ema_jepa.py \
  --device cpu --epochs 20 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --hidden_dim 64 --latent_dim 16 --predictor_hidden_dim 128 \
  --ema_decay 0.99 --seed 7712 \
  --output_json results/world/part1_fused_context_ema_jepa_head056.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_ema_jepa_head056.pt
```

## Result

Best learned-target epoch:

| metric | value |
| --- | ---: |
| epoch | 15 |
| learned-target MSE | 0.097189 |
| learned-target MRR | 0.054649 |
| learned-target top5 | 0.057031 |
| predicted effective rank | 2.883115 |
| target effective rank | 3.806809 |
| context effective rank | 4.441587 |
| context offdiag abs mean | 0.394966 |
| context variance mean | 0.132657 |

Frozen-context probes from the selected checkpoint:

| probe | MSE | MRR | top5 |
| --- | ---: | ---: | ---: |
| fixed-PCA target | 1.240337 | 0.069033 | 0.086719 |
| raw horizon delta | 0.019864 | 0.067651 | 0.074219 |

Reference thresholds from the current primary fixed-PCA reference:

| metric | reference |
| --- | ---: |
| fixed-PCA ridge MRR | 0.103763 |
| fixed-PCA ridge top5 | 0.136719 |
| raw-delta ridge MSE | 0.015701 |
| raw-delta ridge MRR | 0.096147 |
| raw-delta ridge top5 | 0.128125 |

## Mechanism Read

The fused relative context improves over the old EMA JEPA family, where HEAD006
MRR was around `0.033-0.034` and predicted rank was roughly `1.7-2.2`.
HEAD056 reaches learned-target MRR `0.054649` and predicted rank `2.883115`.

That is still not enough. The learned EMA target path remains substantially
weaker than the fixed-PCA reference, and its context probes are worse on both
fixed-PCA and raw-delta targets. This says the old failure was not only a
context bottleneck. The EMA target geometry itself is still not discriminative
enough for the current IV-surface data object.

Primary failure class: `latent_prediction`, with residual `collapse` risk in
the learned target/predicted spaces.

## Decision

Do not promote HEAD056. Keep the fixed-PCA fused-context reference as the active
Part 1 reference.

The next Part 1 improvement should stay canonical but change target construction
rather than adding losses: test a masked full-window JEPA target where context
and target encoders see the same relative coordinate family, and future targets
are selected as blocks from the same 60-day relative path rather than isolated
horizon-delta frames.

## Artifacts

- `experiments/world/part1_jepa_latent/fused_context_ema_jepa.py`
- `test_code/test_world_model_evaluation.py`
- `results/world/part1_fused_context_ema_jepa_head056.json`
- `models/world/checkpoints/part1_jepa_latent/fused_context_ema_jepa_head056.pt`
