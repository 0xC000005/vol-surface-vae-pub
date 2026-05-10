# World Model HEAD064: Masked Multiview Data Builder

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: the HEAD063 protocol can be implemented as a reusable data builder
that produces geometry-aware masked multiview batches with separate real
observedness and synthetic SSL masks, without starting model training.

Falsifier: the builder flattens geometry away, conflates real missingness with
synthetic masks, cannot deterministically sample typed masks, or fails to
produce same-window/same-relative-index positive metadata.

## Implementation

Added `experiments/world/evaluation/masked_multiview_data.py`.

The module provides:

- `GeometryTokenMetadata`;
- `MaskedMultiviewBatch`;
- `load_geometry_panel_values`;
- `sample_typed_synthetic_mask`;
- `apply_synthetic_mask`;
- `build_masked_multiview_batch`.

The dense smoke token schema is:

| geometry | tokens |
| --- | ---: |
| IV surface cells | 25 |
| vol side channels | 5 |
| factor levels | 14 |
| factor returns | 14 |
| total | 58 |

The builder keeps:

- `observed_mask`: real source-data availability;
- `synthetic_mask_a/b`: SSL corruption visibility for each view;
- `absolute_index` and `relative_index`;
- `positive_index` for same-position view alignment;
- `geometry_id`, `factor_id`, `factor_family`, and `geometry_coord`.

Typed masks implemented:

- `surface_maturity`;
- `surface_moneyness`;
- `surface_rectangle`;
- `vol_side_channel`;
- `factor_family`;
- `time_block`;
- `sparse`.

## Validation

Focused tests:

```bash
pytest test_code/test_world_model_evaluation.py::test_masked_multiview_batch_preserves_geometry_and_masks \
  test_code/test_world_model_evaluation.py::test_masked_multiview_keeps_real_missingness_separate -q
```

Result: `2 passed in 0.77s`.

Full world-model evaluation test slice:

```bash
pytest test_code/test_world_model_evaluation.py -q
```

Result: `34 passed in 0.81s`.

Compile check:

```bash
python -m py_compile experiments/world/evaluation/masked_multiview_data.py \
  test_code/test_world_model_evaluation.py
```

## Real-Data Smoke

Command:

```bash
python - <<'PY'
from experiments.world.evaluation.masked_multiview_data import build_masked_multiview_batch
b = build_masked_multiview_batch(split='train', max_windows=8, seed=640, normalize=True)
print(b.clean_values.shape, b.observed_mask.mean(), b.synthetic_mask_a.mean())
PY
```

Saved summary: `results/world/masked_multiview_head064_smoke.json`.

Key output:

| metric | value |
| --- | ---: |
| clean shape | `8 x 30 x 58` |
| observed rate | 0.896552 |
| view A visible rate | 0.858477 |
| view B visible rate | 0.927874 |
| IV surface tokens | 25 |
| vol side-channel tokens | 5 |
| factor level tokens | 14 |
| factor return tokens | 14 |

The real-data observed rate below `1.0` confirms that real missingness exists
and is represented separately from synthetic masking.

## Decision / Next Step

The masked multiview data contract is now concrete enough for the next Part 1
iteration.

Do not train a model yet. The next iteration should implement the representation
loss/diagnostic harness for pretraining batches:

- same-state alignment metrics;
- Barlow cross-correlation diagonal/off-diagonal terms;
- same-state retrieval;
- geometry-stratified summaries by mask family.

## Artifacts

- `experiments/world/evaluation/masked_multiview_data.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/evaluation/README.md`
- `experiments/world/reports/world_model_head064_masked_multiview_data_builder.md`
- `results/world/masked_multiview_head064_smoke.json`
