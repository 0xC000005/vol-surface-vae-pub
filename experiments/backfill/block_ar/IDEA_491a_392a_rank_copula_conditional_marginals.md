# 491a 392a Rank-Copula Conditional Marginals

## Context

The post-487 and H60 evidence closes the current local routes:

- More history is not the missing ingredient. H60 has generic state signal, but 489a scored `4/11` and 490a only recovered to `7/11`, below 392a.
- More objective pressure around 392a is capped. Energy, PIT, CRPS, MMD, critic, density-ratio, source-transport, and calibration variants move average coverage or level KS but damage conditionality, cointegration, or path geometry.
- The persistent failure is now clean: 392a has the best learned path/rank geometry, but its conditional marginal level occupancy and per-cell/regime interval allocation are wrong.

## Paradigm

Use the canonical conditional distribution factorization:

```text
p(Y | H) = copula(U | H) + conditional marginals F_j(Y_j | H)
```

where each future variable `j = (horizon, cell)`.

The deployable sampler is:

1. Sample paths from frozen 392a.
2. Convert each 392a sample path into per-variable ranks `u_j` across the sample ensemble.
3. Train a conditional quantile model `Q_j(u | H)` from rolling historical `(history, realized_future)` pairs.
4. Map each frozen 392a rank through the learned marginal quantile:

```text
Y'_j = Q_j(u_j | H)
```

This preserves the 392a rank-copula/path geometry samplewise while replacing the weak marginal level law with a learned conditional marginal law.

## Why This Is Not The Old 323 Branch

323 was a valid Sklar-factorization idea, but it used a weak fixed scalar-AR copula (`321c`) and never had 392a-level path geometry. The failure mechanism was a brittle interface around a poor copula: marginal transports distorted path geometry and still could not make the weak sampler deployable.

491a is a different falsifier because the copula supplier is the current deployable frontier:

- 392a already passes daily-change shape, cross-cell correlation, cointegration, mean reversion, pathwise jumps, and conditionality.
- Density-ratio resampling showed that 392a support contains useful missing level outcomes, but probability reallocation alone is capped.
- The remaining target is specifically the conditional marginal `F_j(. | H)`, not a new path generator.

## Methodological Status

This is theory-backed rather than a research knob:

- Sklar factorization is a first-principles decomposition of a joint law into marginals and dependence.
- The generative core stays deployable: history plus frozen 392a noise paths plus a learned quantile readout.
- No regime labels, low-rank decoder, bounded residual path, validation oracle, retrieval bank, or evaluator-specific gates are used.
- If it succeeds, report base 392a metrics and rank-copula/marginal-system metrics separately.

## Decisive Falsifier

Implement `492a`:

- Train a lightweight conditional quantile marginal model on train windows only.
- Use frozen 392a samples as the rank-copula supplier at inference.
- Evaluate the unchanged full 11-suite.

Success means exceeding the 392a `8/11` frontier without losing structural passes. Failure means the remaining issue is not separable conditional marginals around 392a's copula, and the next paradigm must learn marginals and dependence jointly in one model rather than factorizing them.
