# World Model HEAD005: Retrieval-Loss Part 1 Repair Smoke

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Adding an in-batch retrieval / InfoNCE auxiliary loss and stronger variance
pressure should move validation retrieval above chance and increase effective
rank without destroying prediction quality.

Falsifier: retrieval remains near chance, prediction quality collapses, or the
predicted latent stays low-rank.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/jepa_smoke.py`
- `test_code/test_world_model_evaluation.py`

New objective terms:

- `retrieval_contrastive_loss(predicted, target)`;
- `--retrieval_weight`;
- `--retrieval_temperature`;
- dynamic loss-component logging.

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/jepa_smoke.py`
- `python experiments/world/part1_jepa_latent/jepa_smoke.py --device cpu --epochs 12 --batch_size 128 --max_train_windows 2048 --max_val_windows 256 --variance_weight 0.5 --covariance_weight 0.01 --retrieval_weight 0.2 --retrieval_temperature 0.1 --output_json results/world/part1_jepa_retrieval_head005.json --checkpoint models/world/checkpoints/part1_jepa_latent/jepa_retrieval_head005.pt`

## Result

Tests:

```text
5 passed in 0.69s
```

Training:

```text
best loss epoch 8: loss 1.26749, prediction 0.11938, variance 0.43203, retrieval 4.56530
final epoch 12:    loss 1.35160, prediction 0.28474, variance 0.29679, retrieval 4.48958
```

Validation metrics:

```text
prediction mse       0.5007267533
prediction rmse      0.7076204868
prediction cosine    0.9172033036
retrieval top1       0.01171875
retrieval top5       0.03515625
retrieval top10      0.05078125
retrieval mrr        0.0362189969
context eff rank     3.0807123864
context var mean     0.6406426407
context offdiag abs  0.4830096061
predicted eff rank   1.9968460943
predicted var mean   0.0645630429
predicted offdiag    0.6674831478
```

Raw placeholder baseline on the same validation windows:

```text
prediction mse       0.0139568013
prediction cosine    0.9837976048
retrieval top1       0.00390625
retrieval top5       0.0390625
retrieval top10      0.06640625
retrieval mrr        0.0358554995
```

The repair did make variance pressure visible and improved top1 above chance,
but it destroyed prediction quality and did not beat the raw placeholder on
top5/top10 retrieval. Predicted-latent effective rank stayed near `2` despite a
16-dimensional latent. This is still a failed Part 1 representation result.

## Mechanism Read

The current repair shifts the model from smooth central prediction toward more
spread-out context states, but the predictor still fails to map that spread into
a discriminative future latent. The loss balance is not the only issue:

- context rank improved more than predicted rank;
- retrieval loss decreased slowly and remains close to batch-random InfoNCE
  scale;
- prediction MSE exploded, showing the objective terms are fighting rather than
  coordinating.

Primary failure class: `latent_prediction`, with persistent `collapse` risk in
the predicted latent.

## Decision

Do not attach the flow decoder.

The next research move should be a narrower Part 1 redesign, not another
blind weight sweep:

- make the future target easier and more explicit, such as horizon-specific
  future summary latents at `{1, 5, 10, 20, 30}`;
- evaluate supervised future-summary probes as a lower bound;
- consider a normalized projection head for retrieval while keeping a separate
  regression head for prediction;
- track best validation epoch instead of final epoch before making claims.
