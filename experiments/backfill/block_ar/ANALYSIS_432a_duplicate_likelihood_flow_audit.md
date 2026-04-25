# 432a: Duplicate Likelihood-Flow Audit

## Context

431a selected a minimal exact-likelihood joint future path flow as the next clean
paradigm. Before implementing it, the codebase audit found that this model already
exists and was already evaluated.

The prior implementation is:

- model: `diffusion/block_ar/empirical_normal_score_path_coupling_density.py`;
- training script:
  `experiments/backfill/block_ar/train_346a_empirical_normal_score_path_coupling_density.py`;
- result: `results/block_ar/346a_v0_s42/full11.json`.

It is materially the same idea as 431a: empirical normal-score future path, conditional
affine coupling flow, exact likelihood, one-shot future sampling.

## Prior Result

`346a` scored `3/11`.

Key metrics:

- coverage failed: overall 90% coverage `0.734`, worst cells as low as `0.068` to
  `0.141` across horizons, best cells near `1.0`;
- conditionality failed: MAE reduction `0.19%`, turb/calm width ratio `0.915`;
- time-series failed;
- regime layer2 `0/8`;
- distributional fidelity failed: daily KS `3/25`, level KS `3/25`;
- cross-cell correlation collapsed: corr ratio `0.003`, rank ratio `4.382`;
- mean reversion failed: active-cell pass rate `12.5%`;
- pathwise jump realism failed with max-jump KS near the gate.

The historical postmortem also noted that `347a`, the transition-level exact likelihood
variant, restored cross-cell geometry but still scored only `4/11` and failed
coverage/conditionality/time-series/regime/distribution/mean-reversion/pathwise suites.

## Mechanism Read

The exact-likelihood idea was already falsified in both obvious placements:

- full future path likelihood (`346a`) destroys stochastic geometry;
- transition-level likelihood (`347a`) preserves geometry better but does not solve
  level/regime/mean-reversion failures.

Re-implementing 431a as 432a would duplicate a closed experiment and violate the loop's
anti-knob discipline.

## Decision

Do not implement the duplicate 431a/346a flow.

Next iteration should be a broader evidence audit over the post-340 frontier, specifically
to identify any genuinely untested mechanism rather than reselecting a previously closed
paradigm from memory. The active best remains `392a` at `8/11`.

