# 409a Free-Running Marginal CRPS Fine-Tune

## Context

The current learned frontier is `392a` at `8/11`, failing:

- coverage
- regime_coverage
- distributional_fidelity

The calibrated-system branch is now capped as a primary path:

- `403a` marginal quantile calibration damaged time/path and level geometry.
- `405a` target-90 interval scaling over-widened already-safe cells.
- `407a` deadband interval scaling was cleaner but still scored `7/11`, losing conditionality while leaving level/regime gaps.

So the next move should return the pressure to the learned generator, not add post-hoc cell/regime rules.

## Hypothesis

The previous free-running path-energy objective was directionally right but too blunt:

- it improved level occupancy from `385a` to `392a`;
- stronger weights improved some level metrics further;
- but scalar energy strength traded away conditionality, coverage, or cointegration before level KS reached the gate.

Energy score is a multivariate path score. Its gradients emphasize whole-path Euclidean geometry and can improve average path distance without precisely repairing per-cell/per-horizon marginal level occupancy.

A cleaner next objective is free-running marginal CRPS:

```text
L = L_FM_anchor + lambda_crps * mean_{t,cell} CRPS({generated score paths}, target score)
```

This remains a proper scoring rule, but it directly targets the marginal distributions that drive:

- per-cell coverage,
- level KS,
- median-bias fraction,
- and regime-cell coverage indirectly through conditional marginal spread.

## Why This Is Principled

- It keeps the generative core unchanged: same 392a causal-memory transition FM.
- It adds no new architecture, decoder path, bounded branch, low-rank readout, retrieval table, or hand-coded regime correction.
- It is trained in free-running mode, matching the deployment/evaluation mode.
- It is not an evaluator hack: CRPS is a standard proper scoring rule over the full future grid and horizon.
- It is more targeted than energy score while still general across factor panels because every future variable/time receives the same rule.

## Minimal Experiment

Implement `410a` as a fine-tune from:

```text
models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt
```

Use:

- teacher-forced FM anchor retained;
- free-running rollout samples with gradient;
- marginal CRPS over normal-score future paths;
- small CRPS weight first, to avoid repeating the energy-strength shrinkage pattern;
- official full-11 evaluation after training.

## Falsifier

This route is falsified if marginal CRPS:

- fails to improve level KS or coverage geometry relative to 392a, or
- improves them only by losing conditionality, cointegration, or time-series realism.

If falsified, the issue is likely not a simple proper-score choice. The next paradigm should change the base likelihood/representation rather than keep adding fine-tune losses.
