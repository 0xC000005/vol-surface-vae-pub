# World Model HEAD074: Probe Complementarity Check

Date: 2026-05-09

Iteration type: `experiment`

## Hypothesis / Falsifier

Hypothesis: HEAD070 latents and raw surface features carry complementary
information, so concatenating them should improve frozen downstream probes.

Falsifier: combined features fail to improve over the best single-source
feature for either target.

## Implementation

Updated `masked_multiview_barlow_probe_audit.py` with combined feature sets:

- `raw_surface_last_plus_barlow_clean_last`;
- `raw_surface_flat_plus_barlow_clean_last`;
- `raw_geometry_last_plus_barlow_clean_last`.

Added `concatenate_feature_blocks` and a focused test for row-count validation.

## Validation

Focused test:

```bash
pytest test_code/test_world_model_evaluation.py::test_probe_feature_block_concatenation_checks_rows -q
```

Result: `1 passed in 0.75s`.

Audit command:

```bash
python experiments/world/part1_jepa_latent/masked_multiview_barlow_probe_audit.py \
  --device cpu \
  --output_json results/world/masked_multiview_barlow_probe_head074.json
```

Saved result: `results/world/masked_multiview_barlow_probe_head074.json`.

## Result

Validation, ridge alpha `10.0`:

| feature | mean-delta MSE | mean-delta R2 | range MSE | range R2 |
| --- | ---: | ---: | ---: | ---: |
| raw surface last | 0.006484 | 0.533258 | 0.054625 | -2.969679 |
| HEAD070 clean last latent | 0.011635 | 0.162420 | 0.047185 | -2.428997 |
| raw surface last + HEAD070 last | 0.006584 | 0.526010 | 0.047705 | -2.466780 |
| raw surface flat | 0.009746 | 0.298396 | 0.050972 | -2.704215 |
| raw surface flat + HEAD070 last | 0.009258 | 0.333521 | 0.053574 | -2.893356 |
| raw geometry last | 0.013460 | 0.031052 | 0.064093 | -3.657776 |
| raw geometry last + HEAD070 last | 0.012985 | 0.065231 | 0.060980 | -3.431566 |

## Decision

The falsifier fired for the main combined feature:

- raw surface last remains best for mean delta;
- HEAD070 clean last remains best for range;
- concatenation gives small gains over some weaker baselines, but does not beat
  the best single-source feature on either target.

This suggests HEAD070 is a useful representation, but a linear concatenation
probe does not establish broad downstream complementarity. The likely next
Part 1 issue is encoder geometry: the current daily flattened GRU may discard
surface-local level information that raw surface last keeps.

## Next Step

Post-experiment analysis should decide whether to move from daily flattened GRU
encoding to a geometry-aware token encoder for pretraining. That would be a
principled architecture change, not an extra objective knob.

## Artifacts

- `experiments/world/part1_jepa_latent/masked_multiview_barlow_probe_audit.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/reports/world_model_head074_probe_complementarity.md`
- `results/world/masked_multiview_barlow_probe_head074.json`
