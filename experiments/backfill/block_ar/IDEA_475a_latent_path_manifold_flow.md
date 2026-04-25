# 475a: Learned Future-Path Latent Manifold Flow

## Context

The active deployable frontier remains `392a` at `8/11`. It passes local and
structural scenario requirements, but still fails coverage, regime coverage, and
distributional fidelity. The last branch (`470a`-`473b`) showed that preserving
`392a` residual geometry is necessary, but wrapper transports and scalar source
knobs cannot learn the missing conditional level allocation.

Prior direct full-path attempts (`339`, `345`, `354`, `413`, `417`, `421`, `460`)
mostly failed because a high-dimensional future path was modeled directly from
noise or with a simple token likelihood. They either lost cross-cell stochastic
geometry, overbroadened the path law, or underlearned mean-reversion and level
occupancy. The repeated failure does not imply that a joint future law is wrong;
it implies that modeling the full `30 x 25` path in raw observed coordinates is a
poor statistical object for this data scale.

## Hypothesis

Use the only core architectural bias the reset doctrine allows: a narrow learned
encoder-decoder bottleneck.

Train a deterministic future-path autoencoder in empirical normal-score
coordinates:

```text
future path z_1:T -> compact latent u -> reconstructed future path z_hat_1:T
```

Then train a vanilla conditional flow matching model:

```text
history -> flow from Gaussian noise to u
```

Sampling is deployable:

```text
history -> sample latent u -> decode full 30-day future path -> empirical-quantile decode to IV
```

This is a learned latent path-manifold model, not a calibration wrapper around
`392a`. It does not require a validation-future oracle, retrieval, low-rank
readout, bounded idiosyncratic path, regime table, or per-cell posthoc map.

## Why This Is Different

The target of the generative core is the learned path coordinate `u`, not raw
future levels and not one-day recursive transitions. The autoencoder is trained to
learn the path manifold from data; the conditional flow only has to model the
conditional distribution over that compact path coordinate.

This is closer to the Bitter Lesson than hand-engineered residual calibration:
the model learns the representation and the conditional law from data, while the
human design choice is limited to compression. It is also cleaner than the old
posterior/prior latent branches because the first falsifier can use a
deterministic encoder plus a standard latent flow, avoiding VAE-specific
posterior language and diagonal-likelihood shortcuts.

## Expected Failure Modes

This paradigm should be killed quickly if either happens:

- the autoencoder reconstruction itself cannot preserve daily-change, level,
  mean-reversion, and cross-cell geometry on validation futures;
- the conditional latent flow samples plausible latents by reconstruction metrics
  but still collapses conditionality or cross-cell structure in the full 11-suite.

If reconstruction is strong but conditional sampling is weak, the bottleneck is
conditional latent-law learning. If reconstruction is weak, the bottleneck size or
decoder is insufficient, and the branch should not be judged by the scenario
suite yet.

## First Falsifier

Implement one minimal `475a/476a` baseline:

- empirical normal-score coordinate shared with the `340/392` family;
- deterministic future-path encoder-decoder with a compact latent;
- conditional latent rectified flow from history features to latent;
- no 392a wrapper, no calibration table, no regime labels, no low-rank/bounded
  side path;
- unchanged official full 11-suite evaluation.

The first success criterion is not instant `11/11`. The branch stays alive only if
it beats old direct one-shot/full-path attempts while preserving the structural
suites that made `392a` deployable. A plausible first target is at least `6/11`
with credible level-KS or coverage movement and no cross-cell/pathwise collapse.
