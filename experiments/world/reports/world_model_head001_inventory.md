# World Model HEAD001: Data And Evaluation Inventory

Date: 2026-05-09

## Iteration Type

`research_ideation`

## Hypothesis

The repository already contains enough local IV/factor data and evaluation code
to define a smoke-scale JEPA latent world-model dataset plus separate Part 1 and
Part 2 metric contracts without adding new data or online dependencies.

## Execution

Commands and files inspected:

- `python` inventory over `data/vol_surface_with_ret.npz`,
  `data/multi_factor_data.npz`, `data/regime_labels.npz`, and
  `data/gt_cumulative_variance.npz`.
- `python -m json.tool results/baselines_current_panel/manifest.json`.
- `experiments/backfill/block_ar/train_289c_deterministic_obs_encoded_latent_world_model.py`.
- `experiments/backfill/block_ar/train_312a_unified_state_aware_future_logit_path_flow_matching.py`.
- `experiments/backfill/block_ar/train_oneshot_flow.py`.
- `experiments/backfill/block_ar/train_169c_shape_scale_student_t.py`.
- `experiments/backfill/block_ar/evaluate_220h_full_multihorizon_v2_suite.py`.
- `experiments/backfill/block_ar/test_block_ar_requirements_v2.py`.

No training was run in this iteration.

## Data Inventory

`data/vol_surface_with_ret.npz` is the right first smoke object:

- `surface`: `(5822, 5, 5)`, finite `145550/145550`, range
  `[0.0100046, 0.99572]`.
- `ret`, `price`, `slopes`, `skews`, `levels`: each `(5822,)`, fully finite.
- It matches the existing IV surface target shape used by Block-AR,
  deterministic world-model, and path-flow scripts.

`data/multi_factor_data.npz` should be second-stage, not first-stage:

- `dates`: `(5825,)`, `2000-01-03` through `2023-02-27`.
- `levels`: `(5825, 14)`, finite `80859/81550`.
- `returns`: `(5825, 14)`, finite `80624/81550`.
- The missing values and transform policy make it valuable for generalization
  and joint-panel conditioning after the IV-only harness is stable.

`data/regime_labels.npz` is immediately useful for probes/buckets:

- `labels`: `(5763,)`, five regimes.
- `history_len=30`, `future_len=30`, aligned to the same local window scale.
- `features`: `(5763, 5)`, fully finite.

`data/gt_cumulative_variance.npz` is useful for Part 1 frozen probes:

- horizons are `[4, 9, 19, 29]`.
- each horizon matrix is `(5, 5)`.

`results/baselines_current_panel/manifest.json` confirms the current-panel
joint state has 53 columns:

- 25 IV cells.
- 14 factor levels.
- 14 factor increments/log returns/diffs.
- train windows `0..4009`, validation windows `4010..4450`.

## Existing Split Convention

The reusable IV split is:

- `history_len=30`.
- `future_len=30`.
- `test_start=4511`.
- `val_size=441`.
- `max_train_idx = test_start - history_len - future_len = 4451`.
- train indices: `0..4009`, `4010` windows.
- validation indices: `4010..4450`, `441` windows.

This matches the baseline manifest and the common 289c/312a training pattern.
The older one-shot flow script uses a nearby standalone convention
(`train_end=4040`, `3981` train futures, `500` validation futures), but the
world-model smoke should use the manifest-aligned 30/30 split above.

## Minimum Smoke Dataset Contract

First dataset builder target:

```text
past_window:   (B, 30, 25)
future_window: (B, 30, 25)
start_index:   (B,)
split:         train | val
regime_label:  optional aligned label
```

Coordinate choice for the first smoke:

- load `surface` from `data/vol_surface_with_ret.npz`;
- convert `(T, 5, 5)` to `(T, 25)`;
- reuse `normalize_iv(surface)` from
  `experiments/backfill/block_ar/train_169a_transformed_student_t.py` for a
  first normalized `[-1, 1]` smoke;
- keep bounded-logit / current-panel state transforms for the second harness
  step, once the IV-only contract is verified.

Part 1 target shape:

- context encoder sees `past_window`;
- target/EMA encoder sees `future_window` or horizon-specific future patches;
- predictor emits horizon-conditioned future latents for horizons
  `{1, 5, 10, 20, 30}`.

Part 2 decoder target shape:

- flow decoder generates `future_window` as `(B, 30, 25)`;
- base noise and vector field live in the same flattened `750`-dimensional path
  coordinate for the first smoke;
- later versions can condition on both `z_context` and predicted
  `z_future_hat`.

## Reusable Metrics

Part 1 needs new lightweight wrappers under `experiments/world/evaluation/`:

- latent prediction MSE/cosine by horizon;
- retrieval top-k and MRR against validation future-latent distractors;
- per-dimension variance;
- effective rank / participation ratio;
- singular-value spectrum;
- off-diagonal covariance/correlation norm;
- frozen probes for future variance/regime/jump/tail summaries.

Part 2 can reuse or port existing Block-AR pieces:

- `evaluate_220h_full_multihorizon_v2_suite.py` provides the full multi-horizon
  scenario-evaluation structure.
- `test_block_ar_requirements_v2.py` provides surface validity, coverage,
  conditionality, time-series, block-AR, cointegration, regime, distributional,
  cross-cell correlation, mean-reversion, and jump-realism tests.
- `train_oneshot_flow.py` already has a compact sample-quality smoke:
  effective rank, PC alignment, Frobenius correlation error, KS pass count,
  kurtosis ratio, and spread growth.

For the first flow-decoder smoke, use the compact one-shot metrics first and
only graduate to the full 11-suite once the decoder produces nontrivial samples.

## Mechanism Read

The first bottleneck is not architecture choice. It is a stable local harness:

- the data object is clear enough for an IV-only JEPA smoke;
- the split convention is clear enough and matches existing current-panel
  artifacts;
- Part 1 metrics do not yet exist in world-model-specific reusable form;
- Part 2 metrics exist in richer Block-AR form, but need a small wrapper for
  direct `(B, S, 30, 5, 5)` decoder samples.

The modeling language should stay careful: fixed parametric decoders are weak
baselines here because they are too restrictive for the conditional,
high-dimensional scenario distribution. That is not evidence that mixture heads
never work; it is evidence that the primary branch should be conditional flow
matching.

## Decision

Proceed to implementation next. The next HEAD iteration should build the smoke
dataset and metric harness, not train the full model yet.

Target next artifacts:

- `experiments/world/evaluation/world_data.py`: manifest-aligned 30/30 window
  builder for IV surfaces.
- `experiments/world/evaluation/part1_metrics.py`: representation diagnostics.
- `experiments/world/evaluation/part2_metrics.py`: compact decoder sample
  diagnostics.
- A tiny smoke command that loads the dataset and emits JSON shape/stat checks.

Primary failure class for this iteration: none. The data/evaluation contract is
defined enough to continue.
