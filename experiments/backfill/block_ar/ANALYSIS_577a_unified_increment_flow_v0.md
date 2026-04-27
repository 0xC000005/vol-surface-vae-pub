# 577a Unified Increment Flow V0

## Context

576a verified that IV and anchor factors can be represented as one reversible
future-increment tensor with 38 state variables. The next test was whether a
minimal shared conditional flow over that tensor is already usable.

## Implementation

Added:

- `experiments/backfill/block_ar/train_577a_unified_increment_flow.py`;
- `test_code/test_577a_unified_increment_flow.py`.

The model is deliberately minimal:

- one GRU over standardized 38-variable history;
- one flat conditional flow network over the full `(30, 38)` future-increment
  tensor;
- no separate IV head;
- no separate factor head;
- no stress selector;
- no low-rank readout.

## Runs

Short V0:

```bash
python experiments/backfill/block_ar/train_577a_unified_increment_flow.py \
  --epochs 8 \
  --batch_size 64 \
  --hidden_dim 256 \
  --depth 4 \
  --future_len 30 \
  --output_dir models/backfill/577a_unified_increment_flow_s577 \
  --seed 577 \
  --device cuda
```

Result:

- params: `1,084,628`;
- best epoch: `5`;
- best val loss: `2.6024347884`;
- finite sample increment rate: `1.0`;
- finite sample state rate: `1.0`;
- standardized increment std ratio: `1.1879`;
- reconstructed IV range: about `4.16e-09` to `3,526,132.5`;
- reconstructed factor range: about `-0.208` to `997,650.94`.

Longer/larger probe:

```bash
python experiments/backfill/block_ar/train_577a_unified_increment_flow.py \
  --epochs 50 \
  --batch_size 64 \
  --hidden_dim 384 \
  --depth 5 \
  --future_len 30 \
  --output_dir models/backfill/577b_unified_increment_flow_s578 \
  --seed 578 \
  --device cuda
```

Result:

- params: `2,117,716`;
- best epoch: `44`;
- best val loss: `2.5573100703`;
- finite sample increment rate: `1.0`;
- finite sample state rate: `1.0`;
- standardized increment std ratio: `1.2157`;
- reconstructed IV range: about `1.79e-08` to `134,412.89`;
- reconstructed factor range: about `-0.0968` to `1,663,435.75`.

## Mechanism Read

The V0 model trains and samples, but it is not risk-manager acceptable.

The failure is clean:

- the source distribution is independent standard Gaussian noise over all daily
  standardized increments;
- early in training, and even after 50 epochs, the model remains too close to
  that independent source;
- cumulative sums of roughly independent daily log-level increments create large
  terminal dispersion;
- exponentiating those cumulative log paths produces unrealistic IV and
  price-like factor levels.

This is not an IV-specific pathology and not an anchor-factor-specific pathology.
It is a path-source pathology in the unified increment coordinate.

## Decision

Do not add separate IV/factor corrections.

The next clean experiment should change the source distribution, not the
architecture:

- fit a full-path empirical Gaussian or shrinkage covariance over the standardized
  `(30, 38)` training increment tensor;
- draw flow source samples from that temporally and cross-sectionally correlated
  path law instead of independent daily Gaussian noise;
- keep the same shared conditional flow core.

This is aligned with the time-series literature signal from TSFlow/CW-Gen:
conditioning and source geometry matter for multivariate time-series generation.
