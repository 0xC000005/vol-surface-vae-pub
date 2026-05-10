# World Model HEAD063: Geometry-Aware Masked Multiview Protocol

Date: 2026-05-09

Iteration type: `research_ideation`

## Question

What should the first concrete Part 1 pretraining protocol be after reframing
the objective from future-latent prediction to masked multiview market-state
representation learning?

## Local Data Inventory

The first protocol should use local data already present in the repository.

| source | observed shape | geometry |
| --- | ---: | --- |
| `data/vol_surface_with_ret.npz:surface` | `5822 x 5 x 5` | IV surface grid |
| `data/vol_surface_with_ret.npz:ret` | `5822` | daily return side channel |
| `data/vol_surface_with_ret.npz:price` | `5822` | daily price side channel |
| `data/vol_surface_with_ret.npz:slopes/skews/levels` | `5822` each | surface summary side channels |
| `data/multi_factor_data.npz:levels` | `5825 x 14` | named factor panel |
| `data/multi_factor_data.npz:returns` | `5825 x 14` | named factor-change panel |
| `data/regime_labels.npz:labels` | `5763` | downstream state probe labels |

The multi-factor columns are:

```text
spx, usdcad, usdjpy, dxy, copper, wheat, crude_oil, us2y, us10y,
aaa_oas, bbb_oas, nikkei, gold, vix
```

## Token Schema

Use a typed token view rather than one flattened feature vector. A minimal
record for every value should include:

```text
value
observed_mask       # 1 if present in source data, 0 if real missing
synthetic_mask      # 1 if visible to the SSL view, 0 if hidden by corruption
absolute_index      # source-day index
relative_index      # position inside the 30-day window
geometry_id         # iv_surface, vol_side_channel, factor_level, factor_return
factor_id           # named feature id, or surface cell id
factor_family       # surface, vol_summary, equity, fx, commodity, rates, credit
geometry_coord      # e.g. moneyness/maturity for surface; factor slot otherwise
```

The model input can still be a dense tensor for the first smoke, but the view
builder must preserve this metadata so masks and diagnostics are geometry-aware.

## Positive Pair Rule

Positive pairs are not future pairs.

The first pretraining pair should be:

```text
same window
same absolute date
same relative index
same underlying panel state
different structured masks
```

Do not align the same absolute date across different relative indices in this
first protocol. A causal/contextual encoder may legitimately encode different
belief states when the same date appears with different preceding context.

## Mask Families

Each training sample should draw two independently corrupted views from a
bounded mixture of typed masks.

### IV Surface Masks

- maturity band: hide one or more full maturity rows/columns, depending on the
  repository's surface axis convention;
- moneyness band: hide one or more full moneyness rows/columns;
- local rectangle: hide a contiguous surface patch;
- wing mask: hide high/low moneyness edge points;
- ATM strip mask: hide central moneyness cells;
- whole-surface dropout for a small fraction of days only.

### Vol-Summary Side-Channel Masks

- hide `ret`, `price`, `slope`, `skew`, or `level` channels individually;
- hide all surface-summary side channels for selected contiguous days;
- never let a hidden zero be ambiguous with a real zero: the mask channel is
  mandatory.

### Multi-Factor Masks

Use named families:

| family | columns |
| --- | --- |
| equity/risk | `spx`, `nikkei`, `vix` |
| fx | `usdcad`, `usdjpy`, `dxy` |
| commodities | `copper`, `wheat`, `crude_oil`, `gold` |
| rates | `us2y`, `us10y` |
| credit | `aaa_oas`, `bbb_oas` |

Masks:

- individual factor history over a contiguous day block;
- whole family over a contiguous day block;
- sparse factor dropout on isolated days;
- cross-family stress mask, such as rates plus credit, only as a diagnostic
  mask family after the basic masks work.

### Time Masks

- contiguous day block inside the 30-day context;
- sparse days inside the same window;
- tail masks only if the positive-pair rule remains same-relative-index and no
  future information leaks into the online view.

## First Smoke Data Contract

The first implementation should produce a batch object shaped roughly as:

```text
view_a_values          (B, T, N, C_value_or_token)
view_b_values          (B, T, N, C_value_or_token)
observed_mask          (B, T, N)
synthetic_mask_a       (B, T, N)
synthetic_mask_b       (B, T, N)
absolute_index         (B, T)
relative_index         (T,)
geometry_id            (N,)
factor_id              (N,)
factor_family          (N,)
geometry_coord         (N, K)
positive_index         (B, T)
split                  train | val | test
```

For a simple dense smoke, `N` can be the concatenation of:

- 25 IV surface cells;
- 5 vol-surface side channels;
- 14 factor levels;
- 14 factor returns.

That gives `N = 58` observable tokens per day before any future expansion.

## Pretraining Loss

The first loss should be representation-only:

```text
z_a[t] = online_encoder(view_a)[t]
z_b[t] = stopgrad(target_encoder(view_b)[t])

L = alignment(z_a[t], z_b[t])
  + lambda_barlow * BarlowCrossCorrelation(z_a[t], z_b[t])
  + optional variance/covariance health terms
```

This is not a future-prediction loss. Forecast/range estimation is a downstream
probe after pretraining.

## Required Diagnostics

Part 1 smoke must report:

- same-state alignment MSE/cosine by geometry mask family;
- retrieval MRR/top-k from view A to matching view B among same-batch
  distractors;
- Barlow cross-correlation diagonal and off-diagonal terms;
- embedding variance, effective rank, and off-diagonal correlation;
- geometry-stratified failures: surface bands, factor families, time masks;
- mask-artifact check: a small classifier should not trivially infer the mask
  family from embeddings while alignment still looks good;
- downstream frozen probes only after the representation passes the above.

## Falsifiers

The protocol fails if:

- masked views align only for one geometry and fail for others;
- embeddings collapse or become low-rank under heavy masks;
- retrieval succeeds because the embedding encodes mask pattern rather than
  market state;
- a flat Bernoulli mask baseline performs as well as typed masks on
  geometry-stratified diagnostics;
- downstream probes improve only when future information leaks into the
  pretraining view.

## Decision

The next implementation iteration should build the reusable masked-view data
builder and focused tests, not train a model yet. The implementation should
start with IV surface plus side channels, then add multi-factor levels/returns
once the token schema and masks are verified.

## Next Step

Implement `experiments/world/evaluation/masked_multiview_data.py` with:

- token metadata construction for IV surface cells and side channels;
- deterministic typed mask sampling;
- separate `observed_mask` and `synthetic_mask`;
- same-window/same-relative-index positive metadata;
- focused tests in `test_code/test_world_model_evaluation.py`.
