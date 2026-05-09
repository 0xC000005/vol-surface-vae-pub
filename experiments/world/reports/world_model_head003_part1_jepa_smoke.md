# World Model HEAD003: Minimal Part 1 JEPA Smoke

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

A minimal IV-only JEPA-style latent model can beat the raw last-frame /
mean-future placeholder on future-latent prediction and retrieval while keeping
basic variance/effective-rank diagnostics healthier than collapse.

Falsifier: validation retrieval remains at chance, or representation health is
low-rank/high-correlation even if prediction MSE improves.

## Execution

Added:

- `experiments/world/part1_jepa_latent/jepa_smoke.py`

The smoke model uses:

- GRU context encoder over the past 30-day IV window;
- EMA GRU target encoder over the future 30-day IV window;
- MLP predictor from context latent to target latent;
- prediction MSE plus variance and covariance health regularizers.

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/jepa_smoke.py`
- `python experiments/world/part1_jepa_latent/jepa_smoke.py --device cpu --epochs 8 --batch_size 128 --max_train_windows 1024 --max_val_windows 256 --output_json results/world/part1_jepa_smoke_head003.json --checkpoint models/world/checkpoints/part1_jepa_latent/jepa_smoke_head003.pt`

## Result

Tests:

```text
4 passed in 0.73s
```

Training loss:

```text
epoch 1 loss 0.272157 prediction 0.225814
epoch 8 loss 0.056699 prediction 0.009857
```

Validation shape:

```text
train past/future: (1024, 30, 25)
val past/future:   (256, 30, 25)
```

Validation JEPA metrics:

```text
prediction mse       0.0126758614
prediction rmse      0.1125871279
prediction cosine    0.9902829581
retrieval top1       0.00390625
retrieval top5       0.01953125
retrieval top10      0.04296875
retrieval mrr        0.0236203951
context eff rank     2.2630872353
context var mean     0.0056559027
context offdiag abs  0.5483659769
target eff rank      2.2553533675
target var mean      0.0069611738
target offdiag abs   0.4706197574
```

Raw placeholder baseline on the same 256 validation windows:

```text
prediction mse       0.0139568013
prediction rmse      0.1181389069
prediction cosine    0.9837976048
retrieval top1       0.00390625
retrieval top5       0.0390625
retrieval top10      0.06640625
retrieval mrr        0.0358554995
target eff rank      2.9557116003
target var mean      0.0034280295
target offdiag abs   0.6055170685
```

The model learned enough to reduce MSE and raise cosine versus the raw
placeholder, but retrieval is at chance and worse than the raw baseline top-k.
The learned target/context latents also remain low rank relative to the latent
dimension. This is not a passed Part 1 representation result.

## Mechanism Read

The failure is informative:

- Prediction MSE alone is too weak; the model can learn a smooth central future
  latent without preserving instance identity.
- The current variance regularizer is not strong enough to make the latent
  space use its available dimensions.
- EMA target encoding is mechanically working, but the target representation is
  still too compressive for retrieval to work.
- A direct decoder should not be attached to this latent yet, because that would
  risk hiding Part 1 failure behind Part 2 behavior.

Primary failure class: `latent_prediction`, with a secondary `collapse` risk.

## Decision

Run post-experiment analysis next before training more. The next iteration
should inspect the saved result and repair the Part 1 objective/data contract:

- add explicit multi-horizon targets or horizon tokens instead of one whole
  future latent;
- increase or redesign variance/covariance pressure;
- compare against a deterministic supervised future-summary probe;
- consider contrastive/retrieval auxiliary loss if prediction MSE continues to
  ignore instance identity.

Do not proceed to the flow decoder yet.
