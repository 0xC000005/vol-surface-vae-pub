# World Model HEAD075: Geometry-Aware Encoder Decision

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

After HEAD074, should the next Part 1 change be another objective/probe tweak,
or a geometry-aware encoder?

## Evidence

The objective and data pipeline are now geometry-aware:

- masks are structured by surface maturity, moneyness, rectangles, factor
  families, side channels, and time blocks;
- token metadata contains geometry id, factor id, family, and coordinates;
- diagnostics are stratified by geometry and family.

But the current model architecture is not geometry-aware:

- it concatenates all token values and mask channels into one daily vector;
- the GRU sees token identity only through fixed column position;
- IV surface locality, factor families, and side-channel structure are not
  modeled explicitly.

The probe evidence points in the same direction:

- raw surface last-day features dominate future mean-delta;
- HEAD070 latents dominate future range;
- linear concatenation does not beat the best single feature on either target;
- full geometry flattening can overfit.

This suggests the model learns useful invariant regime/range information, but
does not preserve enough local surface-level state for simple delta probes.

## Decision

The next principled Part 1 change is a geometry-aware encoder, not another
objective knob.

The pretraining target should remain canonical direct Barlow masked-multiview.
The architecture should change so the model can use the geometry metadata
already carried by the data builder.

## Proposed Minimal Experiment

Implement a small token-aware daily encoder:

- per-token input: value, observed mask, synthetic mask, geometry coordinates,
  and compact one-hot geometry/family descriptors;
- shared token MLP;
- daily pooling over token embeddings;
- temporal GRU over daily pooled states;
- direct canonical Barlow loss on per-time-row embeddings;
- same masks, same train/validation windows, same diagnostics, same frozen probe
  protocol.

This is intentionally not a Transformer yet and not a new objective. It tests
whether giving the encoder the geometry that the masking policy already uses
helps Part 1.

## Falsifier

The geometry-aware encoder is not justified if it fails to improve at least one
of these without damaging the others:

- same-state retrieval/rank compared with HEAD070;
- off-diagonal redundancy compared with HEAD070;
- frozen probes, especially mean-delta versus raw surface last and future range
  versus HEAD070.

## Artifacts

- `experiments/world/evaluation/masked_multiview_data.py`
- `experiments/world/part1_jepa_latent/masked_multiview_barlow_smoke.py`
- `experiments/world/part1_jepa_latent/masked_multiview_barlow_probe_audit.py`
- `experiments/world/reports/world_model_head075_geometry_encoder_decision.md`
