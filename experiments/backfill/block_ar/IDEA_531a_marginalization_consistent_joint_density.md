# 531a Marginalization-Consistent Joint Density Ideation

## Context
530a closed the factor-conditioned side-channel branch. The current learned frontier remains `392a`/`510a` at `8/11`, with the persistent deployable failures concentrated in coverage, regime layer2, and future IV-level distributional fidelity.

The local repair history is also clear:

- `410a`/`487a` marginal CRPS moved average calibration but did not solve conditional level allocation.
- `455a` joint MMD and later sliced/energy variants could move geometry but damaged conditionality or stayed below frontier.
- `323a-d` and `492a` separable marginal/copula maps collapsed because independently repaired marginals were not path-law neutral.
- `525a-530a` broader factor conditioning improved conditionality but did not repair the future IV-level law.

## Literature Check
Recent ICLR 2026 time-series work gives three useful signals:

- MMPD argues that MSE-style deterministic forecasting misses multi-modal future distributions and proposes a patch diffusion loss for patch-based backbones: https://openreview.net/forum?id=NEUgHT8dvH
- MixLinear shows that clean segmented/time-frequency linear structure can be highly parameter-efficient for long-horizon forecasting, but it is primarily a deterministic forecasting backbone, not a full scenario law: https://openreview.net/forum?id=QUj0KuCumD
- MOSES / marginalization-consistent flows argues that probabilistic forecasters can have unreliable marginal predictions when the modeled joint law is not marginalization-consistent; its solution is a mixture of separable flows over latent Gaussian processes with analytically valid marginals: https://openreview.net/forum?id=awWi4hJI7O

The strongest match to the current pathology is the third point. Our failing suites are mostly marginal or low-dimensional queries of a generated joint future path: per-cell coverage, per-regime per-cell coverage, and per-cell IV-level KS. A model can look plausible as a path sampler while still being unreliable under these marginal queries.

## Non-Redundant Direction
Do not repeat `323` or `492`: those used a fixed external copula/path supplier plus a separately trained marginal map. The failure mode was exactly that the marginal repair changed path geometry and did not produce a coherent learned joint law.

The next clean falsifier should instead train one coherent conditional density:

```text
Y_future_score | H ~ N(mu(H), diag(s(H)) R diag(s(H)))
```

where:

- `Y_future_score` is the 30x25 future IV-level path in empirical normal-score coordinates;
- `mu(H)` and `s(H)` are predicted directly from history by one encoder;
- `R` is a single learned or estimated full residual correlation over all future horizon/cell tokens;
- training uses exact Gaussian negative log likelihood;
- sampling is one-shot and deployable from history plus Gaussian noise.

This is not a low-rank decoder, not a bounded residual path, not a retrieval copula, not a calibration wrapper, and not an evaluator loss. It is the simplest marginalization-consistent conditional joint density we can test locally.

## Why It Is Worth Testing
The current AR flow samples realistic daily changes but drifts in future IV-level occupancy. A one-shot conditional density over future levels attacks the same variable the suite asks about. Because the same Gaussian law defines the full path and every marginal subset, marginal failures become model-likelihood failures rather than an after-the-fact rollout artifact.

This also gives a direct falsifier of whether a clean explicit likelihood can beat the 392a frontier without the complexity of a full diffusion Transformer or another calibrated policy layer.

## First Falsifier
Implement `532a` as a small coherent Gaussian-copula path model:

- empirical normal-score transform fitted on train future levels;
- history encoder over normal-score history levels;
- heads for future-path `mu` and `log_s`;
- full global residual correlation/Cholesky;
- exact NLL training;
- one-shot sampling and official 11-suite evaluation.

Success criterion: beat or tie `8/11` while preserving conditionality, time-series, mean-reversion, cross-cell correlation, and pathwise realism. If it scores far below frontier, close this explicit Gaussian-density route and move to a larger learned-law program rather than adding marginal/copula knobs.
