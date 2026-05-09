# World Model HEAD012: Context Correlation Regularization

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

A normalized off-diagonal correlation penalty on context embeddings should
improve representation health more directly than covariance-magnitude
regularization while preserving the fixed-delta predictive signal.

Falsifier: context effective rank/off-diagonal correlation does not improve, or
the saved checkpoint loses the fixed-delta edge over persistence and zero-delta
baselines.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/supervised_horizon_frame.py`
- `test_code/test_world_model_evaluation.py`

Added:

- `context_correlation_loss`
- `--context_correlation_weight`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/supervised_horizon_frame.py experiments/world/part1_jepa_latent/context_probe_audit.py`

Run:

```text
python experiments/world/part1_jepa_latent/supervised_horizon_frame.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_mode delta --frame_weight 0.25 \
  --retrieval_weight 0.05 --retrieval_temperature 0.1 \
  --context_variance_weight 0.05 \
  --context_covariance_weight 0.0 \
  --context_correlation_weight 0.005 \
  --context_variance_gamma 0.1 \
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_corrreg_head012.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_head012.pt
```

Audit:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_head012.pt \
  --output_json results/world/part1_context_probe_audit_head012.json
```

## Result

Tests:

```text
13 passed in 0.71s
```

MRR-selected checkpoint:

```text
epoch 4
frame MSE      0.021497
frame MRR mean 0.058749
frame top1     0.012500
frame top5     0.087500
frame top10    0.135938
```

Raw persistence baseline:

```text
frame MSE      0.022376
frame MRR mean 0.052426
frame top1     0.000781
frame top5     0.086719
frame top10    0.135156
```

Context health:

```text
variance min       0.001531
variance mean      0.009405
effective rank     5.732591
participation      3.959500
offdiag abs mean   0.332259
offdiag abs max    0.895429
```

Frozen ridge probe:

```text
MSE        0.018033
MRR mean   0.106710
top1 mean  0.048438
top5 mean  0.136719
top10 mean 0.225000
```

Comparisons:

```text
HEAD010 context audit: rank 3.688405, offdiag 0.484346, ridge MSE 0.017107, ridge MRR 0.111537
HEAD011 cov trial:     rank 3.567140, offdiag 0.502522, ridge MSE 0.017674, ridge MRR 0.108320
HEAD012 corr trial:    rank 5.732591, offdiag 0.332259, ridge MSE 0.018033, ridge MRR 0.106710
```

## Mechanism Read

The normalized correlation penalty does what the covariance-magnitude penalty
did not: it improves the context rank gate and reduces normalized redundancy.
The selected checkpoint still beats raw persistence on frame MSE, MRR, top1,
top5, and top10, and the frozen ridge probe still beats the zero-delta target
baseline by a wide margin.

There is now a clear tradeoff. Correlation regularization makes the context
healthier, but the frozen probe is weaker than the unregularized HEAD010 audit.
This likely means the penalty is pushing useful predictive directions apart but
also shrinking or rotating some signal needed by a simple linear probe.

## Decision

Continue JEPA-only.

Next step:

- tune the rank/probe tradeoff using a lighter correlation weight or a composite
  checkpoint score;
- keep the context probe audit as the selection gate;
- do not move to a decoder until the context health improvement and probe
  advantage are both stable.

