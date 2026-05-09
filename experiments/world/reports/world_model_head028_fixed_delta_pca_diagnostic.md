# World Model HEAD028: Fixed Delta-PCA Diagnostic

Date: 2026-05-09

## Iteration Type

`experiment`

## Literature Status

`supported_adjacent`, not `canonical_jepa`.

This experiment uses a fixed PCA/whitened horizon-delta target as a diagnostic
contract. It is not a replacement for JEPA's learned target encoder. It tests
whether a stable future-delta target space exists before asking EMA JEPA to
learn that space.

Supporting families:

- I-JEPA/V-JEPA for context-to-target latent prediction and target construction.
- VICReg/Barlow Twins/VJ-VCR for variance, covariance, whitening, and
  redundancy-reduction motivation.

## Hypothesis

A fixed PCA/whitened horizon-delta target should avoid the target-space collapse
seen in HEAD025 and provide a usable target contract for Part 1.

Falsifier:

- predicted fixed-target retrieval does not beat HEAD025 MRR `0.042356`;
- predicted effective rank remains near `1`;
- decoded deltas do not improve over the raw persistence frame-MSE reference.

## Execution

Added:

- `experiments/world/part1_jepa_latent/fixed_delta_pca_jepa.py`
- focused tests in `test_code/test_world_model_evaluation.py`

Core target contract:

```text
delta_h = future[:, h - 1, :] - past[:, -1, :]
z_h = whitened PCA_8(delta_h), fitted on training deltas
past window + horizon token -> predicted z_h
loss = MSE(predicted z_h, z_h)
```

Run:

```text
python experiments/world/part1_jepa_latent/fixed_delta_pca_jepa.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_dim 8 \
  --hidden_dim 64 --context_dim 32 --predictor_hidden_dim 128 \
  --output_json results/world/part1_fixed_delta_pca_jepa_head028.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/fixed_delta_pca_jepa_head028.pt
```

## Result

Tests:

```text
21 passed in 0.75s
```

Best epoch:

```text
epoch 25
fixed-target MRR      0.096374
top1                  0.035156
top5                  0.132031
top10                 0.203125
decoded delta MSE     0.015369
pred effective rank   3.863753
target effective rank 3.327211
context rank          4.874274
```

Reference comparisons:

```text
HEAD025 learned delta EMA MRR       0.042356
raw persistence frame-space MRR     0.052426
raw persistence frame-space top5    0.086719
raw persistence frame-space top10   0.135156
raw persistence frame-space MSE     0.022376
HEAD028 decoded delta MSE           0.015369
```

Per-horizon fixed-target retrieval:

```text
h1  MRR 0.081196  top5 0.117188  top10 0.171875  pred rank 3.729984
h5  MRR 0.088257  top5 0.128906  top10 0.191406  pred rank 3.800845
h10 MRR 0.104040  top5 0.144531  top10 0.203125  pred rank 3.791976
h20 MRR 0.099184  top5 0.121094  top10 0.191406  pred rank 3.729061
h30 MRR 0.109192  top5 0.148438  top10 0.257812  pred rank 3.737418
```

## Mechanism Read

The fixed target contract passes the intended diagnostic:

- it beats HEAD025 learned-target MRR by a large margin;
- it beats raw persistence top-k retrieval in the fixed target space;
- it avoids the rank-1 collapse seen in HEAD025;
- its decoded delta MSE beats the raw persistence frame-MSE reference and is in
  the same direction as the supervised delta lower bound.

This does not prove a learned JEPA target encoder works. It proves that the
future-delta target space can be stable and predictive if the target contract is
fixed and whitened.

## Decision

Promote fixed delta-PCA as the current Part 1 target-space diagnostic contract,
not as the final model family.

Next step should be `post_experiment_analysis`:

- compare fixed-target retrieval with supervised delta-frame runs without
  mixing incompatible metrics;
- decide whether the next learned-target attempt should distill or reconstruct
  this fixed target contract;
- keep the decoder frozen out of scope.

## Artifacts

- `experiments/world/part1_jepa_latent/fixed_delta_pca_jepa.py`
- `test_code/test_world_model_evaluation.py`
- Ignored outputs: `results/world/part1_fixed_delta_pca_jepa_head028.json`,
  `models/world/checkpoints/part1_jepa_latent/fixed_delta_pca_jepa_head028.pt`
