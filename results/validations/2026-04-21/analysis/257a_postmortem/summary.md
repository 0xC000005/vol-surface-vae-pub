# 257a-v0 Postmortem

## Result

- suite score: `3/11`
- passes:
  - `surface`
  - `block_ar`
  - `cross_cell_correlation`

This is still below the `4/11` frontier, but it is the first family in the loop to
produce **non-zero stochastic coverage** while also passing cross-cell structure.

Artifacts:

- eval: `results/block_ar/257a_v0_K4_s42/full11.json`
- spec: `results/validations/2026-04-21/analysis/257a_design/spec.md`
- checkpoint: `models/backfill/257a_v0_K4_s42/best_model.pt`

## High-Signal Metrics

- coverage 90% overall: `20.5%`
- h30 coverage 90%: `11.5%`
- `corr_ratio`: `0.604`
- `rank_ratio`: `0.598`
- `change KS`: `2/25`
- `level KS`: `0/25`
- `mr_gt_ratio`: `1.782`
- `cointegration_ratio`: `1.387` overall, but worst cell fails
- `max-jump KS`: `0.460`

## Mechanism Read

`257a` is not a dead stochastic model. The latent channel is active, but still too
weak to fix the conditional mean path.

### What improved

- coverage moved from deterministic `0%` to `20.5%`
- persistent severe undercoverage improved from `100%` to `69.9%`
- cross-cell correlation structure now passes:
  - `corr_ratio=0.604`
  - `rank_ratio=0.598`
- token attention is selective rather than dead:
  - attention entropy mean: `1.197`
  - top-1 token weight mean: `0.433`

### Why it still fails

- KL mean on validation is only `0.0028`
- prior std mean: `0.450`
- posterior std mean: `0.462`
- sample path std mean: `0.0062`
- terminal std mean: `0.0040`

This says the latent mechanism is alive, but weak. It adds some spread and preserves
joint structure, but not enough to solve:

- conditionality
- fidelity / KS
- mean reversion
- jump realism

The decoder still produces the wrong mean path while the latent variables mainly add
modest dispersion around it.

## Conclusion

`257a-v0` is the first **promising stochastic** family in this loop, but it is still a
partial success rather than a breakthrough.

Most likely next step:

- stay inside the `257` family
- do a focused analysis iteration first
- then strengthen the latent channel rather than abandoning the paradigm immediately
