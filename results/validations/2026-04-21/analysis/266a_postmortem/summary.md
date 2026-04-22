# 266a Postmortem

- checkpoint: `models/backfill/266a_v0_s42/best_model.pt`
- best epoch: `28`
- suite score: `2/11`
- passes: `surface`, `block_ar`

## Headline Read

`266a-v0` is a valid clean first-principles baseline, but it is not competitive yet.

It learns:
- valid support
- nontrivial stochastic spread
- strong aggregate coverage

It does not learn:
- per-cell coverage allocation
- regime-sensitive width
- realistic jump incidence / jump scale
- cointegration
- balanced cross-cell rank structure
- correct short-horizon mean-reversion magnitude

## Full Sample Metrics

- coverage90: `0.881`
- calibration error: `0.042`
- turb/calm width ratio: `0.911`
- ACF corr: `0.786`
- kurtosis ratio: `3.155`
- corr ratio: `1.871`
- rank ratio: `0.346`
- cointegration ratio: `0.174`
- MR ratio: `2.111`
- MR h30 ratio: `0.872`
- max-jump KS: `1.000`
- max-jump q99 ratio: `0.104`

## Decoder vs Prior Probe

To separate the bottleneck/decoder from the latent prior, a reconstruction probe was compared against the sampled generator.

### Reconstruction Probe

- ACF corr: `0.703`
- kurtosis ratio: `0.537`
- corr ratio: `1.069`
- rank ratio: `0.569`
- cointegration ratio: `0.319`
- MR ratio: `2.057`
- max-jump KS: `1.000`
- max-jump q99 ratio: `0.010`

### Interpretation

The decoder/bottleneck already has two strong pathologies:

1. it over-smooths the path and suppresses jump amplitude almost completely
2. it overstates short-horizon mean reversion

The latent diffusion prior then adds a second pathology:

1. it pushes the sampled joint law toward stronger common-mode collapse
2. it reduces effective rank and weakens cointegration relative to reconstruction
3. it restores some tail activity relative to reconstruction, but in the wrong way, leaving jump realism dead

## Mechanism Conclusion

This is still a clean failure.

`266a-v0` does **not** fail because the reset was too weak or because the model needed low-rank / bounded-side-path corrections to even function.

It fails because:
- the single-vector bottleneck + direct decoder learns an over-smoothed future representation
- the latent diffusion prior over that bottleneck does not preserve the richer joint structure needed for realistic conditional scenarios

## Next Question

The next principled question is:

what is the smallest first-principles extension that increases temporal/joint flexibility beyond a single compressed future code, without reintroducing hard low-rank structure or bounded correction paths?
