# 255a-v0 Postmortem

## Result

- suite score: `1/11`
- pass: `block_ar`
- fails include deterministic target suites:
  - `surface`
  - `time_series`
  - `cointegration`
  - `distributional_fidelity`
  - `cross_cell_correlation`
  - `mean_reversion`
  - `pathwise_jump_realism`

Artifacts:

- eval: `results/block_ar/255a_v0_M16_s42/full11.json`
- spec: `results/validations/2026-04-21/analysis/255a_design/spec.md`
- checkpoint: `models/backfill/255a_v0_M16_s42/best_model.pt`

## High-Signal Metrics

- `change KS`: `2/25`
- `level KS`: `1/25`
- `corr_ratio`: `0.492`
- `rank_ratio`: `1.943`
- `mr_gt_ratio`: `1.426`
- `mr_h30`: `0.882`
- `acf_corr`: `0.881`
- `kurtosis_ratio`: `0.440`
- `cointegration_ratio`: `0.827`
- `max-jump KS`: `0.823`

## Mechanism Read

The motif-routing paradigm failed through motif collapse rather than through a dead
residual adapter alone.

### Routing collapse

- validation route entropy mean: `0.329`
- validation top-1 routing weight mean: `0.913`
- active top-1 motifs on 192 windows: `1/16`
- all 192 validation windows routed to motif index `3` as the top choice

### Motif bank collapse

- motif-bank effective rank: `2.06`
- motif-bank PC1 share: `0.834`

So the model did not learn a diverse reusable motif library. It learned a mostly
single-mode motif basis and then routed nearly every window to the same mode.

### Residual adapter stayed secondary

- motif share RMS: `0.948`
- residual share RMS: `0.108`
- residual budget mean: `0.0278`

That means the small bounded residual head did not silently take over. The failure
is the paradigm itself as implemented here: a collapsed motif bank feeding a nearly
single-motif router.

### Output structure

- output effective rank: `7.13`
- output PC1 share: `0.521`

This is not the `254`-style over-shared rank-1 common-mode collapse. It is a
different failure: one dominant routed motif plus insufficient adaptive capacity to
recover the missing temporal and cross-cell structure.

## Conclusion

`255a-v0` cleanly failed to recover the `4/11` frontier and does not justify another
fixed-motif follow-up.

The next principled move is a new paradigm:

- move from fixed sparse motif routing
- to a **hierarchical latent-token future representation**

The next family should keep non-AR fixed-horizon generation, but represent future
structure with multiple conditional latent tokens rather than one continuous common
path or one collapsed motif bank.
