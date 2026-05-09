# World Model HEAD010: Context Probe Audit

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

The retrieval-selected fixed-delta checkpoint should contain a context state that
is future-predictive under a frozen linear probe, not only through its trained
nonlinear horizon head.

Falsifier: frozen context embeddings are collapsed/near-constant or a linear
probe cannot beat the zero-delta baseline on horizon-delta targets.

## Execution

Added:

- `experiments/world/part1_jepa_latent/context_probe_audit.py`

Updated:

- `test_code/test_world_model_evaluation.py`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/context_probe_audit.py experiments/world/part1_jepa_latent/supervised_horizon_frame.py`

Run:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_contrastive_mrr_head009.pt \
  --output_json results/world/part1_context_probe_audit_head010.json
```

## Result

Tests:

```text
11 passed in 0.70s
```

Validation context health:

```text
variance min       0.003021
variance mean      0.014709
variance max       0.064836
effective rank     3.688405
participation      2.214723
offdiag abs mean   0.484346
offdiag abs max    0.969168
```

Frozen ridge probe on horizon-delta targets:

```text
MSE        0.017107
RMSE       0.130792
cosine     0.402427
MRR mean   0.111537
top1 mean  0.050000
top5 mean  0.147656
top10 mean 0.228906
```

Zero-delta target baseline:

```text
MSE        0.022376
MRR mean   0.023923
top1 mean  0.003906
top5 mean  0.019531
top10 mean 0.039063
```

Trained horizon head in target space:

```text
MSE        0.021263
MRR mean   0.081859
top1 mean  0.026563
top5 mean  0.103906
top10 mean 0.182031
```

Ridge probe by horizon:

```text
h1  MSE 0.010486  MRR 0.092428  top1 0.039063  top5 0.121094  top10 0.183594
h5  MSE 0.014109  MRR 0.101102  top1 0.042969  top5 0.128906  top10 0.210938
h10 MSE 0.019449  MRR 0.121001  top1 0.062500  top5 0.156250  top10 0.230469
h20 MSE 0.020619  MRR 0.121566  top1 0.050781  top5 0.175781  top10 0.269531
h30 MSE 0.020871  MRR 0.121590  top1 0.054688  top5 0.156250  top10 0.250000
```

## Mechanism Read

The context state is not collapsed: every dimension has nonzero variance and a
linear frozen probe strongly beats the zero-delta baseline. The probe also beats
the trained nonlinear horizon head in target-space MSE and retrieval, so the
encoder contains more usable future information than the current head extracts.

The representation is still redundant. Effective rank is only about `3.69` out
of `32`, participation ratio is about `2.21`, and off-diagonal correlations are
high. This explains why earlier learned-target JEPA variants looked low-rank:
the useful signal is concentrated in a few context directions unless redundancy
control is applied directly to the context state.

## Decision

Continue JEPA-only.

Next step:

- add a context variance/covariance regularizer to the supervised fixed-delta
  lower bound;
- use retrieval-selected checkpointing and the context probe audit as gates;
- falsify the change if it raises effective rank but loses the fixed-delta
  MSE/retrieval advantage over zero-delta and persistence baselines.

