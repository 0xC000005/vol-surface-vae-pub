# 267b Postmortem

- checkpoint: `models/backfill/267b_v0_s42/best_model.pt`
- best epoch: `19`
- suite score: `2/11`
- passes: `surface`, `block_ar`

## Headline Read

`267b-v0` made the exact anti-collapse change from the memo:
- reduce the decoder's direct history bypass
- keep the same probabilistic latent-token family
- keep the same plain ELBO objective

It did **not** solve posterior collapse.

The decisive evidence stayed unchanged:
- KL remained at `0.0`
- prior std stayed at `~0.998`
- posterior std stayed at `~0.998`

So decoder-bypass reduction alone is insufficient to make the latent token path necessary.

## Full Sample Metrics

- coverage90: `0.026`
- calibration error: `0.488`
- turb/calm width ratio: `1.799`
- ACF corr: `0.830`
- kurtosis ratio: `3.013`
- corr ratio: `0.441`
- rank ratio: `1.050`
- cointegration ratio: `0.197`
- MR ratio: `2.055`
- MR h30 ratio: `0.920`
- max-jump KS: `1.000`
- max-jump q99 ratio: `0.028`
- very-small-move ratio: `1.717`
- daily-change KS pass cells: `0/25`
- level KS pass cells: `2/25`

## Relative To 267a

Relative to `267a-v0`:
- coverage improved slightly: `0.003 -> 0.026`
- regime-width ratio improved: `1.468 -> 1.799`
- rank improved: `0.719 -> 1.050`

But:
- corr ratio fell below the lower gate: `1.205 -> 0.441`
- cointegration collapsed: `0.568 -> 0.197`
- jump realism stayed effectively dead
- KL still stayed exactly collapsed

So the change altered the deterministic geometry of the sampled law, but it did not make the latent variable active.

## Mechanism Conclusion

This is a clean negative result.

`267a` said:
- plain ELBO collapses

`267b` now says:
- reducing decoder history bypass alone does not fix that collapse

That means the plain latent-variable bottleneck line is no longer the best active path if we want to stay:
- elegant
- first-principles
- and free of heuristic KL rescue machinery

## Decision

Do not keep stacking anti-collapse patches inside the same plain-ELBO latent family.

The next principled move is a paradigm shift toward a **direct conditional scenario generator** that models the future path itself, rather than routing stochasticity through a latent bottleneck that the model keeps ignoring.
