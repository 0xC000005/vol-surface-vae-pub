# 422a: Persistent Source-Noise Flow Matching

## Context

After 421a, three clean non-latent routes are capped:

- `392a` recursive AR transition FM preserves structure but underlearns full 30-day level/regime occupancy.
- `413/417` direct future-level path FM can improve occupancy but loses structural coupling.
- `421a` joint transition-path FM preserves daily changes and cointegration but collapses cross-cell stochastic geometry and remains overbroad.

The prior `307` path-latent branch should not be copied directly: it used a posterior/prior latent plus a diagonal likelihood, and the diagonal likelihood collapsed cross-cell dependence. The useful lesson is narrower: persistent path-level uncertainty matters, but it should not replace the strong 392a token-flow transition law.

## Hypothesis

Use the same 392a AR transition-flow architecture, but change the flow source distribution from IID per-step Gaussian noise to a simple persistent source process:

```text
x0_t = sqrt(rho) * g_path + sqrt(1-rho) * eps_t
```

where `g_path` is one Gaussian 25-cell source vector shared across the 30-day scenario, and `eps_t` is local transition noise.

This adds a path-level stochastic state without:

- a posterior encoder,
- a VAE-style learned prior,
- a low-rank decoder,
- a bounded side path,
- a calibration table,
- a regime-specific rule.

It is still vanilla flow matching: only the base source distribution changes. The model learns the transport from a source with persistent scenario-level variation to the empirical transition law.

## Why This Is Principled

The suite failures repeatedly show that independent daily source noise is not enough to produce coherent 30-day level/regime occupancy, while direct full-path modeling loses the local AR geometry.

Persistent source noise is the minimal bridge:

- keep AR state propagation and token transition velocity from 392a;
- preserve one-step transition modeling;
- add a coherent shared path driver at the source-noise level;
- let the learned flow decide how that persistent source maps into future transitions.

This is closer to a world-model latent than to post-hoc calibration, but it avoids the posterior/prior language the user objected to. The latent is just part of the generative base process.

## Decisive Test

Run one conservative fine-tune from 392a:

- add `path_source_corr` to the 340/392 empirical normal-score AR transition FM config;
- train/evaluate with a fixed moderate `rho`, not a sweep;
- retain the same teacher-forced FM objective and recent-window adaptation framing;
- evaluate unchanged on the official full 11-suite.

Success criterion: improve level KS/regime coverage or coverage without losing 392a's conditionality, cointegration, cross-cell correlation, mean reversion, and pathwise realism. If it only behaves like another width actuator, close this route.
