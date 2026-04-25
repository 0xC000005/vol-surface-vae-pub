# Autoresearch 459a: conditional future-token density model

## Context

The recent branches are now cleanly falsified:

- Scaling the vanilla teacher-forced AR empirical-score transition core improved
  validation FM loss but worsened scenario quality.
- Rollout score variants around 392a are capped: score-energy preserves
  conditionality but misses level/regime; marginal CRPS improves calibration but
  loses conditionality; joint-MMD improves coverage shape but remains below gate.
- Deployable quantile calibration is capped: alpha and calibration-window tuning
  do not solve conditionality, regime layer-2, or level KS.

The common limitation is that the current core is not trained as an explicit
conditional density over the future path. Flow matching gives samples, but model
selection has repeatedly relied on indirect rollout scores and calibration layers.

## Paradigm

Replace the current flow-matching core with an explicit conditional future-token
density:

```text
p(Y_1, ..., Y_T | H) = product_k p(Y_k | H, Y_<k)
```

where tokens are future time/cell coordinates in a generic transformed coordinate
such as empirical normal score or logit-IV.

This keeps the architecture first-principled:

- history encoder,
- future-token causal decoder,
- simple parametric output density per token,
- exact conditional likelihood training,
- ancestral sampling at inference.

No low-rank readout, bounded idio path, regime labels, retrieval, validation
oracle, or posthoc correction is required.

## Why This Is Different From Prior One-Shot Flow Attempts

Prior one-shot path flows tried to transport a full future path from noise in one
global operation and repeatedly lost serial geometry. A token-density model is
still a joint future-path model, but it factorizes the joint law with causal
conditioning over future tokens. That gives the model a direct likelihood target
and lets it learn dependency structure through the decoder rather than relying on
rollout calibration after a transition FM model.

## Proposed First Falsifier

Implement a small deployable baseline:

- history GRU/Transformer encoder,
- causal Transformer decoder over future `(time, cell)` tokens,
- Gaussian or Student-t output density in empirical normal-score coordinates,
- teacher-forced NLL training,
- ancestral sampling for the existing 11-suite.

Success criterion for keeping the branch alive is not immediate 11/11. The first
falsifier should beat old one-shot path flows and approach the 392a geometry
without calibration. If it cannot preserve time-series, mean-reversion, and
pathwise realism, abandon or revise the factorization.
