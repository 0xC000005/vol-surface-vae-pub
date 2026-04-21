# 257b Postmortem

## Result

- suite score: `2/11`
- passes:
  - `surface`
  - `block_ar`

Relative to `257a`, `257b` improved stochastic coverage but regressed overall.

Artifacts:

- eval: `results/block_ar/257b_v0_K4_s42/full11.json`
- spec: `results/validations/2026-04-21/analysis/257b_design/spec.md`
- checkpoint: `models/backfill/257b_v0_K4_s42/best_model.pt`

## High-Signal Metrics

- coverage 90% overall: `25.0%`
- h30 coverage 90%: `22.7%`
- `corr_ratio`: `0.648`
- `rank_ratio`: `0.439`
- `change KS`: `3/25`
- `level KS`: `1/25`
- `mr_gt_ratio`: `1.407`
- `cointegration_ratio`: `0.329`
- `max-jump KS`: `0.819`

## Mechanism Read

`257b` succeeded in increasing latent pressure and coverage, but it still did not
make sample identity matter enough.

- attention entropy mean: `1.194`
- attention top-1 mean: `0.450`
- `post_vs_zero_mae = 0.1306`
- `prior_mean_vs_zero_mae = 0.1278`
- `post_vs_prior_mean_mae = 0.0067`
- sample std mean: `0.0075`

Interpretation:

- token attention is alive
- token-conditioned structure matters more than in `257a`
- but changing from the token mean to a sampled latent realization still barely
  changes the decoded future

So `257b` bought more spread, not enough sample-dependent scenario realism.

## Conclusion

`257b` is not a dead end for the stochastic family, but it is not the right direct
upgrade either. The next step should target **sample-dependent scenario training**
rather than only stronger latent pressure.
