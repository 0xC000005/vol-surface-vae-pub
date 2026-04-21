# 257b Spec

Date: 2026-04-21

## Goal

Strengthen latent usage inside the `257` latent-token VAE family after analysis
showed that `257a-v0` uses the history-conditioned token mean much more than the
sampled latent variation.

## Changes vs 257a

1. reduce direct history bypass in the decoder:
   - remove history input from query, factor, and residual heads
2. make loadings token-dependent:
   - sampled token summary perturbs the loading matrix
3. add explicit KL-floor pressure:
   - a penalty if KL falls below a target floor

## Intended Mechanism

- force the decoder to rely more on sampled latent tokens
- make sampled token variation affect the mean path, not only a small residual spread
- keep the same non-AR fixed-horizon scenario generation interface

## Kill Criteria

1. beat `257a` on either `n_pass` or a combination of:
   - higher coverage
   - better conditionality
   - no loss of cross-cell structure
2. keep token attention selective
3. materially raise effective latent usage:
   - higher KL and/or stronger sample sensitivity
