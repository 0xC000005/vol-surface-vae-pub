# 492a Rank-Copula Conditional Marginal Result

## Hypothesis

Use the current best 392a model only as a conditional rank-copula/path-geometry
source, then learn a lightweight conditional marginal quantile law
`Q_j(u | history)` for every future horizon/cell. At inference, map each 392a
sample rank through the learned conditional quantile function.

This tests whether 392a's remaining failures are primarily marginal occupancy
errors rather than dependency/path-law errors.

## Result

Artifacts:

- Model: `models/backfill/492a_392a_rank_copula_cond_marginals_s42/best_model.pt`
- Evaluation: `results/block_ar/492a_392a_rank_copula_cond_marginals_s42/full11.json`
- Markdown: `results/block_ar/492a_392a_rank_copula_cond_marginals_s42/full11.md`

Score: `3/11`.

Passed:

- surface
- block_ar
- cross_cell_correlation

Failed:

- coverage
- conditionality
- time_series
- cointegration
- regime_coverage
- distributional_fidelity
- mean_reversion
- pathwise_jump_realism

Key comparison to 392a:

- 392a: `8/11`; 492a: `3/11`
- coverage90: `0.8675 -> 0.8723`
- conditional MAE reduction: `5.14% -> 3.00%`
- daily-change KS cells: `25/25 -> 16/25`
- level KS cells: `10/25 -> 1/25`
- median-bias cells: `20/25 -> 18/25`
- bias-magnitude cells: `25/25 -> 23/25`
- cointegration worst-cell ratio: `0.278 -> 0.125`
- mean-reversion ratio: `1.024 -> 1.362`
- mean-reversion active pass rate: `83.3% -> 58.3%`
- pathwise max-jump KS: `0.373 -> 0.551`
- pathwise per-cell q99 pass: `22/25 -> 10/25`

## Mechanism Read

The separable marginal map did not simply repair level occupancy. It changed the
geometry of realized paths after the rank draw:

- Per-token marginal quantiles improved aggregate coverage only slightly, but made
  per-cell coverage more uneven. Several cells became badly undercovered while
  others became effectively overcovered.
- Mapping ranks through independently trained horizon/cell quantiles broke the
  daily-change and extreme-jump profiles that 392a already handled well.
- Mean reversion became too strong at h1, and cointegration worst-cell behavior
  weakened, indicating that the marginal map is not path-law neutral.
- Level KS fell to `1/25`, so the learned conditional marginal model did not
  learn the required unconditional occupancy either.

## Decision

Close 492a as a decisive falsifier of this specific factorization. The evidence
does not support a salvage pass that adds more marginal/cell knobs, because the
failure is structural: a separately trained marginal quantile layer is not
compatible with the already-good 392a path law.

Next step should return to the 392a frontier and look for a path-law-preserving
way to address the three remaining failures, or do a fresh post-experiment
analysis if the next move is not clean enough to state in one sentence.
