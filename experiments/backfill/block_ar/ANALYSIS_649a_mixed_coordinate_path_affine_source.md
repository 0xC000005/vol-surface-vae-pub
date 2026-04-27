# 649a Mixed-Coordinate Path Flow With Conditional Source Affine

## Hypothesis

648a showed that pushing spread through a full-path energy score creates level
bias and worsens coverage. 649a tests a cleaner generative alternative: keep
the 647 path-flow architecture, but let the base noise distribution have a
history-conditioned location and scale before the flow transport.

This is not a separate deterministic center/residual branch. It is a standard
conditional-flow source distribution:

```text
x0 = mu(history) + sigma(history) * epsilon
x1 = generated mixed-coordinate future path
```

## Implementation

- Added optional `conditional_source_affine` to
  `GenericMixedCoordinatePathFlowMatching`.
- Source location/scale is produced once from the shared history context and
  covers all future horizons/channels.
- Final source affine layer is zero-initialized, so the model starts at the
  original standard-normal source.
- Training script support was added to
  `train_647a_mixed_coordinate_path_flow.py`.
- No factor-specific branch, no retrieval, no low-rank decoder, no post-hoc
  stress deck.

## Results

Artifacts:

- model:
  `models/backfill/649a_joint38_mixed_path_affine_source_e8_w2048_s649/best_model.pt`
- IV suite:
  `results/autoresearch/649a_joint38_mixed_path_affine_source_e8_w2048_s649/full11.json`
- joint audit:
  `results/autoresearch/649a_joint38_mixed_path_affine_source_e8_w2048_s649/joint_panel.json`

Focused tests passed:

```bash
pytest test_code/test_647a_mixed_coordinate_path_flow.py test_code/test_648a_mixed_coordinate_path_energy_finetune.py -q
```

IV 11-suite:

- score: 3/11, worse than 647a and 648a
- conditionality improved materially: overall MAE reduction -0.7% -> 3.3%;
  per-cell conditionality passed with worst cell -4.7%
- regime width signal improved: turbulent/calm width at h1/h7/h14/h30 became
  1.116/1.094/1.112/1.063 instead of near-flat or inverted
- cross-cell correlation failed: ratio 0.464 versus gate [0.5, 2.0]
- coverage remained low: overall 90% CI 63.6%
- daily-change KS worsened: pass 5/25
- tail scale remained over-allocated in many cells: q99 pass 9/25
- long-horizon mean reversion weakened: h7/h14/h30 profile failed

Joint-panel audit:

- factor delta KS mean: 0.113
- factor KS pass: 13/13
- factor q99 pass: 9/13
- factor-factor correlation: 0.822
- IV-factor correlation: 0.811

## Mechanism Read

The conditional source affine is the first 647-family move that clearly improves
conditionality and regime-dependent width. That suggests the bottleneck really
does involve conditional source/location-scale allocation. But the source
affine is too unconstrained: it reduces shared cross-cell factor structure,
over-amplifies some cell/factor tails, and weakens long-horizon mean reversion.

The failure is therefore not "conditional source is wrong"; it is that a fully
free horizon-channel affine source learns conditional spread independently
across cells and breaks the shared geometry that 647a preserved.

## Decision

Do not keep 649a as the baseline. The next clean direction should keep the
source-conditioning insight but preserve shared geometry, for example by using
a low-dimensional shared source-scale token or correlation-preserving global
source scale rather than a free per-horizon/per-channel affine source.
