# World Model HEAD006: Horizon-Specific JEPA Redesign

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

The previous JEPA smokes failed because one pooled 30-day target allowed a
smooth central future latent. A horizon-specific JEPA target should reduce that
smoothing by making the model predict target latents for explicit horizons
`{1, 5, 10, 20, 30}`.

Falsifier: horizon-specific targets still do not beat raw last-frame baselines
on retrieval, or predicted latents remain low-rank.

## Execution

Added:

- `experiments/world/part1_jepa_latent/horizon_jepa_smoke.py`

Updated:

- `test_code/test_world_model_evaluation.py`
- `experiments/world/part1_jepa_latent/jepa_smoke.py`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/horizon_jepa_smoke.py experiments/world/part1_jepa_latent/jepa_smoke.py`

Real-data runs:

```text
python experiments/world/part1_jepa_latent/horizon_jepa_smoke.py \
  --device cpu --epochs 12 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --variance_weight 0.2 --covariance_weight 0.005 \
  --retrieval_weight 0.1 --retrieval_temperature 0.1 \
  --output_json results/world/part1_horizon_jepa_head006.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/horizon_jepa_head006.pt

python experiments/world/part1_jepa_latent/horizon_jepa_smoke.py \
  --device cpu --epochs 12 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_mode frame \
  --variance_weight 0.2 --covariance_weight 0.005 \
  --retrieval_weight 0.1 --retrieval_temperature 0.1 \
  --output_json results/world/part1_horizon_frame_jepa_head006.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/horizon_frame_jepa_head006.pt

python experiments/world/part1_jepa_latent/horizon_jepa_smoke.py \
  --device cpu --epochs 12 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_mode frame --target_encoder_mode trainable \
  --variance_weight 0.2 --covariance_weight 0.005 \
  --retrieval_weight 0.1 --retrieval_temperature 0.1 \
  --output_json results/world/part1_horizon_frame_trainable_jepa_head006.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/horizon_frame_trainable_jepa_head006.pt
```

## Result

Tests:

```text
7 passed in 0.71s
```

Prefix-target EMA best epoch:

```text
epoch 10
MSE 0.088411
cosine 0.992570
MRR mean 0.034284
top1 mean 0.008594
top5 mean 0.035156
top10 mean 0.062500
predicted effective rank 2.207024
```

Frame-target EMA best epoch:

```text
epoch 9
MSE 0.065770
cosine 0.992969
MRR mean 0.033023
top1 mean 0.007031
top5 mean 0.032031
top10 mean 0.055469
predicted effective rank 1.715300
```

Frame-target trainable-target best/final behavior:

```text
best MRR epoch 12
MSE 0.541556
MRR mean 0.033148
top1 mean 0.007813
top5 mean 0.032813
predicted effective rank 1.510318
```

Raw horizon-frame baseline on the same validation windows:

```text
MSE mean 0.022376
cosine mean 0.974271
MRR mean 0.052426
top1 mean 0.000781
top5 mean 0.086719
top10 mean 0.135156
```

Per-horizon detail showed a limited useful signal: the prefix-target model beat
the raw baseline on MRR at horizon 5 and horizon 30, and on top5 at horizon 30.
It still failed the overall Part 1 representation gate because near-horizon
retrieval and raw-frame MSE were much worse than the raw persistence baseline.

## Mechanism Read

Horizon-specific targets helped diagnose the failure but did not fix it.

- Prefix targets smooth the future and remain low-rank.
- Frame targets reduce smoothing but make the predicted latent even lower-rank.
- Trainable target regularization destabilizes the target space; prediction
  quality becomes much worse without a retrieval win.

The main issue is now narrower: the learned target embedding is not yet a
trustworthy future representation. More decoder work would be premature, and
more blind JEPA weight sweeps are unlikely to fix this.

Primary failure class: `latent_prediction`, with persistent `collapse` risk.

## Decision

Keep the workflow focused entirely on JEPA/Part 1.

The next repair should establish a stable future-target lower bound before
another learned-target JEPA attempt:

- train a supervised horizon-frame predictor as a lower bound against raw
  persistence;
- add a fixed target embedding option, likely PCA or standardized raw horizon
  frames, so the context/predictor cannot chase a moving weak target;
- only reintroduce a learned target encoder after the fixed-target predictor can
  beat raw persistence on retrieval or calibrated probe metrics.
