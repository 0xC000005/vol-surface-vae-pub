# World Model HEAD034: Direct Delta-PCA Predictor Audit

Date: 2026-05-09

## Iteration Type

`experiment`

## Question

Is the fixed delta-PCA target space capped because the HEAD028 GRU context
predictor is too weak, or because the future-delta coordinates are only weakly
predictable from the available past window?

## Hypothesis

A direct flattened-past predictor can expose whether the past window contains
more discriminative signal than the GRU context predictor is using. If it cannot
beat HEAD028 on MSE, retrieval, decoded delta MSE, or rank, then the next move
should be context-object or data-object redesign. If it improves any important
gate, the bottleneck includes predictor/context architecture.

## Implementation

Added:

- `experiments/world/part1_jepa_latent/direct_delta_pca_predictor.py`
- focused test in `test_code/test_world_model_evaluation.py`

Contract:

```text
flatten(past window)
-> direct MLP trunk
-> horizon-conditioned predictor
-> fixed z_pca

loss = MSE(predicted z_pca, fixed z_pca)
```

No target encoder, retrieval/neighborhood loss, target sweep, or decoder was
used.

## Validation

Red test:

```text
pytest test_code/test_world_model_evaluation.py::test_direct_delta_pca_predictor_flattens_past_and_predicts_horizon_codes -q
```

Initially failed with:

```text
ModuleNotFoundError: No module named 'experiments.world.part1_jepa_latent.direct_delta_pca_predictor'
```

Post-implementation checks:

```text
pytest test_code/test_world_model_evaluation.py -q
25 passed in 0.76s

python -m py_compile experiments/world/part1_jepa_latent/direct_delta_pca_predictor.py test_code/test_world_model_evaluation.py
```

Run:

```text
python experiments/world/part1_jepa_latent/direct_delta_pca_predictor.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_dim 8 --hidden_dim 128 \
  --output_json results/world/part1_direct_delta_pca_predictor_head034.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/direct_delta_pca_predictor_head034.pt
```

## Result

MSE-selected checkpoint:

```text
epoch                  16
target-space MSE       1.093260
MRR                    0.098682
top1                   0.036719
top5                   0.132812
top10                  0.203125
decoded delta MSE      0.016982
pred effective rank    5.273018
context rank           11.586305
```

Best retrieval row:

```text
epoch                  15
target-space MSE       1.093676
MRR                    0.110811
top1                   0.049219
top5                   0.150781
top10                  0.221875
decoded delta MSE      0.016992
pred effective rank    5.180739
context rank           11.372300
```

Reference comparisons:

```text
HEAD028 fixed-PCA predictor:
MSE                    0.987130
MRR                    0.096374
top5                   0.132031
top10                  0.203125
decoded delta MSE      0.015369
pred effective rank    3.863753

HEAD032 frozen-target predictor:
MSE                    0.979270
MRR                    0.085140
top5                   0.112500
top10                  0.183594
decoded delta MSE      0.015502
pred effective rank    3.633565
```

Per-horizon metrics for the MSE-selected direct checkpoint:

```text
horizon  MSE       MRR       top5      top10     decoded-MSE  rank
1        0.696714  0.080469  0.109375  0.148438  0.010309     4.597642
5        0.937611  0.085710  0.125000  0.183594  0.014440     4.961278
10       1.132060  0.097516  0.132812  0.195312  0.017652     5.042494
20       1.318264  0.103106  0.125000  0.230469  0.020776     4.998804
30       1.381649  0.126611  0.171875  0.257812  0.021731     4.933492
```

## Mechanism Read

The direct predictor falsifies the strongest form of "the past contains no
retrievable signal." It beats HEAD028 on MRR in the best retrieval row
(`0.110811` vs `0.096374`) and improves top5/top10 (`0.150781`/`0.221875` vs
`0.132031`/`0.203125`).

It does not pass the full Part 1 gate:

- target-space MSE is worse than HEAD028;
- decoded delta MSE is worse than HEAD028;
- the retrieval win comes with much higher effective rank and a much larger
  context space, not with better coordinate accuracy.

This means the active bottleneck is not purely data-object unpredictability.
The past window contains more discriminative signal than the GRU predictor used
in HEAD028, but the direct MLP turns that into ranking structure rather than
accurate fixed-coordinate prediction.

## Decision

Do not promote HEAD034 as a model candidate. Treat it as positive evidence for
a predictor/context architecture bottleneck.

Next HEAD should be `post_experiment_analysis`:

- compare MSE-selected and retrieval-selected direct checkpoints against
  HEAD028;
- decide whether the next model change should be a context-architecture change
  or a principled selection/composite-gate change;
- avoid adding retrieval/neighborhood loss, target sweeps, or decoder work.

The clean next experiment, if the analysis supports it, should port only the
direct context/predictor architecture insight back into the fixed-PCA JEPA path
while keeping the objective as MSE to fixed targets.

## Artifacts

- `experiments/world/part1_jepa_latent/direct_delta_pca_predictor.py`
- `test_code/test_world_model_evaluation.py`
- Ignored outputs: `results/world/part1_direct_delta_pca_predictor_head034.json`,
  `models/world/checkpoints/part1_jepa_latent/direct_delta_pca_predictor_head034.pt`
