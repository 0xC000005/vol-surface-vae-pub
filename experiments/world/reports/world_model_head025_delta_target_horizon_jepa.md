# World Model HEAD025: Delta-Target Horizon JEPA

Date: 2026-05-09

## Iteration Type

`experiment`

## Literature Status

`canonical_jepa`.

This experiment follows the literature gate from HEAD024: it changes target
construction for a context-to-target latent prediction model with an EMA target
encoder. It does not add a new ranking, neighborhood, decoder, or
collapse-prevention objective.

Relevant primary sources:

- I-JEPA: target-block representation prediction from context-block
  representations, with target/context construction as a core design choice:
  https://arxiv.org/abs/2301.08243
- V-JEPA: feature prediction as the stand-alone unsupervised objective, without
  negatives, reconstruction, text, or extra supervision:
  https://arxiv.org/abs/2404.08471

## Hypothesis

The HEAD006 learned-target JEPA failed partly because absolute future targets
are persistence-dominated. A canonical EMA horizon-JEPA using horizon-delta
targets should produce a more discriminative target latent without adding
ad-hoc retrieval or neighborhood objectives.

Falsifier: the delta-target EMA JEPA still produces low-rank predicted latents
and fails the retrieval/persistence gates.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/horizon_jepa_smoke.py`
- `test_code/test_world_model_evaluation.py`

Added:

- `select_horizon_delta_target`
- `select_horizon_encoder_input`
- `--target_coordinate {absolute,delta}`

TDD check:

```text
pytest test_code/test_world_model_evaluation.py::test_horizon_jepa_delta_target_subtracts_last_past_frame -q
```

Validation:

```text
pytest test_code/test_world_model_evaluation.py -q
python -m py_compile experiments/world/part1_jepa_latent/horizon_jepa_smoke.py test_code/test_world_model_evaluation.py
```

Run:

```text
python experiments/world/part1_jepa_latent/horizon_jepa_smoke.py \
  --device cpu --epochs 12 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_mode frame \
  --target_coordinate delta \
  --target_encoder_mode ema \
  --variance_weight 0.2 \
  --covariance_weight 0.005 \
  --retrieval_weight 0.0 \
  --output_json results/world/part1_horizon_delta_frame_jepa_head025.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/horizon_delta_frame_jepa_head025.pt
```

## Result

Tests:

```text
19 passed in 0.71s
```

Best MRR-selected epoch:

```text
epoch 10
val MRR mean 0.042356
val top1     0.003906
val top5     0.043750
val top10    0.091406
val MSE      0.263569
pred rank    1.262683
```

Target/predicted health at the selected epoch:

```text
target effective rank    2.077708
predicted effective rank 1.262683
predicted offdiag mean   0.787735
```

Raw horizon-frame persistence reference from the same validation windows:

```text
MRR mean 0.052426
top1     0.000781
top5     0.086719
top10    0.135156
MSE mean 0.022376
```

Training dynamics:

```text
epoch 4 pred rank 5.410567, MRR 0.024882
epoch 5 pred rank 4.140211, MRR 0.025631
epoch 10 pred rank 1.262683, MRR 0.042356
```

## Mechanism Read

The target-coordinate change is not enough. It improves retrieval relative to
the old HEAD006 learned-target JEPA runs, but it still loses to raw persistence
on MRR/top5/top10 and selects a collapsed predicted representation.

The failure is now more specific:

- the EMA target encoder's own delta-frame latent is low-rank;
- the predictor obtains its best MRR by collapsing toward an even lower-rank
  representation;
- early higher-rank predictions do not retrieve well.

This falsifies the first canonical delta-target JEPA attempt. The failure class
remains `latent_prediction`, with target-encoder `collapse` as the likely
proximal issue.

## Decision

Continue JEPA-only, but do not add another loss term or sweep a new weight.

The next HEAD should be `post_experiment_analysis`: audit target and predicted
latent spectra over epochs/horizons and decide whether to stabilize the target
encoder through a fixed target embedding contract or a separate target-space
pretraining step before returning to EMA JEPA.

## Artifacts

- `experiments/world/part1_jepa_latent/horizon_jepa_smoke.py`
- `test_code/test_world_model_evaluation.py`
- Ignored outputs: `results/world/part1_horizon_delta_frame_jepa_head025.json`,
  `models/world/checkpoints/part1_jepa_latent/horizon_delta_frame_jepa_head025.pt`
