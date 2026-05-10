# World Model Experiment Track

This folder is for testing a JEPA-style latent time-series world model with a
conditional flow decoder for scenario generation.

Workflow protocol: `docs/research_protocols/world_model_autoresearch_plan.md`.
Local resumability state: `autoresearch-session/world_model_state.json`.

The research-log context is important:

- Earlier flow-matching probes showed that CFM can learn ground-truth-aligned
  diversity and cross-cell structure in IV-surface settings.
- Naive high-dimensional one-shot MLP CFM failed, but factored temporal/spatial
  flow architectures worked much better.
- Deterministic AR flow matching had underdispersion and rollout spread issues,
  so this track should evaluate the latent world model and decoder separately.
- Fixed parametric decoders are too restrictive for the conditional,
  high-dimensional scenario distribution targeted here. Use conditional flow
  matching as the primary decoder path.

## Directory Layout

- `part1_jepa_latent/`: self-supervised latent world-model experiments.
- `part2_flow_decoder/`: conditional flow decoder experiments attached to the
  learned latent state.
- `evaluation/`: scripts that score Part 1 and Part 2 separately.
- `configs/`: small config files for reproducible smoke runs and ablations.
- `reports/`: local analysis notes for experiment outcomes.

## Part 1: Latent World Model Quality

Part 1 should answer whether the encoder learns a useful market-state
representation from two structured masked views of the same window and same
relative time position, independent of generated scenario quality.

Primary checks:

- Same-state masked-view alignment.
- Same-state retrieval top-k accuracy or MRR.
- Barlow/cross-correlation diagonal and off-diagonal checks.
- Embedding variance per dimension.
- Effective rank and singular-value spectrum.
- Off-diagonal covariance/correlation norm.
- Mask-artifact diagnostics.
- Frozen probes for future volatility, jumps, drawdown, correlation, or
  downstream forecasting/classification tasks.

The core objective should be:

```text
L_part1 =
    L_masked_multiview_invariance
  + lambda_redundancy L_barlow_or_vicreg
  + lambda_var L_variance
  + lambda_cov L_covariance
```

The default implementation is one shared encoder over two corrupted views. EMA
targets and predictor heads are reserved for explicitly gated
context-to-target JEPA experiments, not for the current two-corruption
same-state objective. Future prediction and range estimation are downstream
probes after pretraining.

## Part 2: Conditional Flow Decoder Quality

Part 2 should answer whether the learned latent state can condition a calibrated,
diverse, realistic scenario distribution.

Primary checks:

- CRPS or quantile/pinball scores for marginal calibration.
- Energy Score and Variogram Score for multivariate scenario quality.
- Empirical coverage by horizon and by cell.
- Sample variance ratio and pairwise scenario distance.
- Correlation matrix error and PCA/eigenvalue spectrum error.
- Tail metrics such as extreme quantile coverage, max-jump KS, or drawdown
  distribution distance.
- Conditionality by regime/context bucket.

The default decoder should be conditional flow matching:

```text
epsilon ~ base_noise
future_path = x_1
x_tau = (1 - tau) * epsilon + tau * x_1
v_theta(x_tau, tau | z_past, z_future_hat, tokens, past) -> x_1 - epsilon
```

At inference, sample base noise and integrate the learned vector field to obtain
multi-day scenario paths.

## First Implementation Direction

Start with a smoke-scale path:

1. Train Part 1 on normalized/state-mapped windows using structured masked
   multiview positives and direct encoder-output redundancy control.
2. Freeze or semi-freeze the Part 1 encoder.
3. Score frozen probes for future/range/state tasks before any decoder claim.
4. Train Part 2 as a conditional flow path decoder only after Part 1 gates are
   understood.
5. Report Part 1 and Part 2 metrics in separate tables before combining claims.

Do not use reconstruction or scenario generation as the main proof of Part 1.
Use them as probes or downstream evaluations after collapse diagnostics pass.
