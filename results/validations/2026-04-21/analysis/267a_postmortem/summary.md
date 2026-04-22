# 267a Postmortem

- checkpoint: `models/backfill/267a_v0_s42/best_model.pt`
- best epoch: `19`
- suite score: `3/11`
- passes: `surface`, `block_ar`, `cross_cell_correlation`

## Headline Read

`267a-v0` made the right paradigm shift: scenario variability was finally part of the model specification.

But the plain ELBO implementation collapsed immediately.

The evidence is direct:
- KL stayed at `0.0`
- prior std stayed at `~0.998`
- posterior std stayed at `~0.998`

So the decoder ignored the latent token path and the sampled model became another near-deterministic forecaster, just inside a formally probabilistic wrapper.

## Full Sample Metrics

- coverage90: `0.003`
- calibration error: `0.499`
- turb/calm width ratio: `1.468`
- ACF corr: `0.878`
- kurtosis ratio: `0.788`
- corr ratio: `1.205`
- rank ratio: `0.719`
- cointegration ratio: `0.568`
- MR ratio: `2.051`
- MR h30 ratio: `0.879`
- max-jump KS: `1.000`
- max-jump q99 ratio: `0.011`
- very-small-move ratio: `1.724`
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`

## Mechanism Conclusion

This is a clean result.

The probabilistic latent-token idea is **not** falsified.
The plain implementation is.

What failed is:
- the decoder had too easy a bypass around the latent
- the history-conditioned mapping was strong enough that the latent token path was ignored
- the model therefore collapsed to near-deterministic behavior

That is why:
- coverage stayed near zero
- jump incidence stayed near zero
- move-size profile stayed heavily over-smoothed

## What Improved

Even under collapse, `267a` still did something important:
- cross-cell correlation structure passed
- rank structure passed
- cointegration stayed above the aggregate gate

So the probabilistic shift did not destroy the cleaner structural gains from late `266`.

## Decision

Stay in the `267` family, but do **not** add KL tricks or side heads first.

The next principled move should be a **latent-usage** change, not a heuristic regularization change.

## Next Question

What is the smallest first-principles anti-collapse change that makes the latent token path necessary?

The leading hypothesis is:
- reduce the decoder's direct history bypass
- keep the latent-token prior/posterior clean
- avoid free-bits, KL annealing, or auxiliary losses unless a later review forces them
