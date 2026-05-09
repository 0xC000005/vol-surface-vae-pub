# World Model HEAD036: Fused Context Delta-PCA Predictor

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

The HEAD034 direct predictor exposed useful retrieval signal but damaged
coordinate accuracy. A fused context architecture should preserve the GRU
branch's coordinate stability while borrowing the direct branch's discriminative
signal, with the objective still fixed-target MSE only.

Falsifier:

- the fused model does not beat HEAD028 on decoded delta MSE or fixed-target
  MSE;
- or it cannot preserve HEAD034's retrieval/top-k gain.

## Implementation

Added:

- `experiments/world/part1_jepa_latent/fused_context_delta_pca_predictor.py`
- focused test in `test_code/test_world_model_evaluation.py`

Architecture:

```text
past window
-> GRU sequence branch
-> direct flattened-past branch
-> fused context
-> horizon-conditioned predictor
-> fixed z_pca

loss = MSE(predicted z_pca, fixed z_pca)
```

No target encoder, retrieval/neighborhood loss, target sweep, or decoder was
used.

## Validation

Red test:

```text
pytest test_code/test_world_model_evaluation.py::test_fused_context_delta_pca_predictor_combines_sequence_and_direct_branches -q
```

Initially failed with:

```text
ModuleNotFoundError: No module named 'experiments.world.part1_jepa_latent.fused_context_delta_pca_predictor'
```

Post-implementation checks:

```text
pytest test_code/test_world_model_evaluation.py -q
26 passed in 0.79s

python -m py_compile experiments/world/part1_jepa_latent/fused_context_delta_pca_predictor.py test_code/test_world_model_evaluation.py
```

Run:

```text
python experiments/world/part1_jepa_latent/fused_context_delta_pca_predictor.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_dim 8 --hidden_dim 64 --context_dim 32 \
  --predictor_hidden_dim 128 \
  --output_json results/world/part1_fused_context_delta_pca_head036.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt
```

## Result

MSE-selected checkpoint:

```text
epoch                  16
target-space MSE       1.013283
MRR                    0.117733
top1                   0.052344
top5                   0.157812
top10                  0.242188
decoded delta MSE      0.015363
pred effective rank    4.595201
context rank           6.551514
```

Best retrieval row:

```text
epoch                  23
target-space MSE       1.022990
MRR                    0.117905
top5                   0.157812
top10                  0.231250
decoded delta MSE      0.016342
pred effective rank    4.762122
context rank           7.051774
```

Best top5 row:

```text
epoch                  15
target-space MSE       1.036251
MRR                    0.114566
top5                   0.158594
top10                  0.239063
decoded delta MSE      0.015562
pred effective rank    4.614262
context rank           6.585221
```

Reference comparisons:

```text
HEAD028 fixed-PCA:
MSE                    0.987130
MRR                    0.096374
top5                   0.132031
top10                  0.203125
decoded delta MSE      0.015369
pred effective rank    3.863753

HEAD034 direct best retrieval:
MSE                    1.093676
MRR                    0.110811
top5                   0.150781
top10                  0.221875
decoded delta MSE      0.016992
pred effective rank    5.180739
```

Per-horizon metrics for the MSE-selected fused checkpoint:

```text
horizon  MSE       MRR       top5      top10     decoded-MSE  rank
1        0.680695  0.093595  0.125000  0.179688  0.010050     4.286288
5        0.897788  0.099915  0.132812  0.214844  0.013495     4.342476
10       1.048558  0.123949  0.148438  0.242188  0.015879     4.370931
20       1.178550  0.126923  0.191406  0.265625  0.018213     4.529235
30       1.260822  0.144283  0.191406  0.308594  0.019178     4.541305
```

## Mechanism Read

The fused architecture does what HEAD035 asked for on retrieval and decoded
quality:

- MRR improves over HEAD028 and HEAD034;
- top5/top10 improve materially over HEAD028 and HEAD034;
- decoded delta MSE `0.015363` is effectively tied with, and slightly better
  than, HEAD028 `0.015369`;
- rank increases from HEAD028's `3.86` to `4.60`, without the severe decoded
  degradation seen in HEAD034.

The remaining failure is target-space coordinate MSE. HEAD036's best MSE
`1.013283` is worse than HEAD028's `0.987130`, even though decoded-frame MSE
and retrieval are better.

## Decision

HEAD036 is the strongest Part 1 candidate so far for retrieval plus decoded
delta quality, but it is not a complete pass because fixed-target MSE regresses.

Next HEAD should be `post_experiment_analysis`:

- decide whether the Part 1 gate should treat HEAD036 as the current reference
  despite the target-space MSE regression;
- compare whether the MSE regression is concentrated in low-variance or
  high-variance PCA directions;
- do not change the objective, add retrieval loss, add target sweeps, or move to
  the decoder before this is understood.

## Artifacts

- `experiments/world/part1_jepa_latent/fused_context_delta_pca_predictor.py`
- `test_code/test_world_model_evaluation.py`
- Ignored outputs: `results/world/part1_fused_context_delta_pca_head036.json`,
  `models/world/checkpoints/part1_jepa_latent/fused_context_delta_pca_head036.pt`
