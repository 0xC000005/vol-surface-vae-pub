# 266c Postmortem

- checkpoint: `models/backfill/266c_v0_s42/best_model.pt`
- best epoch: `28`
- suite score: `2/11`
- passes: `surface`, `block_ar`

## Headline Read

`266c-v0` kept the first-principles `266` family clean and changed exactly one thing from `266b`: the latent diffusion prior became sequence-aware over the bottleneck token path.

That change was real, but it did **not** move the actual scenario-generation bottleneck.

The model improved sampled temporal moments, especially aggregate kurtosis, yet it still failed badly on:
- coverage and calibration
- rank preservation
- cointegration
- level fidelity
- jump realism

So the active issue is no longer that the prior flattens token order. The active issue is the **latent diffusion prior itself** on this bottlenecked latent path.

## Full Sample Metrics

- coverage90: `0.204`
- calibration error: `0.405`
- turb/calm width ratio: `0.980`
- ACF corr: `0.718`
- kurtosis ratio: `1.020`
- corr ratio: `2.113`
- rank ratio: `0.242`
- cointegration ratio: `0.318`
- MR ratio: `1.877`
- MR h30 ratio: `1.094`
- max-jump KS: `0.989`
- max-jump q99 ratio: `0.439`
- very-small-move ratio: `1.312`
- small-move ratio: `1.187`
- daily-change KS pass cells: `2/25`
- level KS pass cells: `0/25`

## Relative To 266b

Relative to `266b-v0`:
- ACF improved: `0.637 -> 0.718`
- aggregate kurtosis improved sharply: `0.263 -> 1.020`
- corr ratio stayed bad: `2.165 -> 2.113`
- rank ratio stayed bad: `0.229 -> 0.242`
- cointegration got slightly worse: `0.371 -> 0.318`
- coverage worsened: `0.304 -> 0.204`
- calibration worsened: `0.346 -> 0.405`
- jump q99 stayed subcritical: `0.484 -> 0.439`
- tiny-move oversmoothing got worse: `1.062 -> 1.312`

So the sequence-aware prior changed the sampled law, but mostly by shifting temporal moments, not by preserving the latent representation where the risk-manager suites are failing.

## Reconstruction vs Sampling

### Reconstruction Probe

- ACF corr: `0.608`
- kurtosis ratio: `0.687`
- corr ratio: `0.592`
- rank ratio: `0.788`
- cointegration ratio: `0.656`
- MR ratio: `1.891`
- max-jump KS: `1.000`
- pathwise q99 ratio: `0.008`

### Sample Probe

- ACF corr: `0.713`
- kurtosis ratio: `1.073`
- corr ratio: `2.110`
- rank ratio: `0.243`
- cointegration ratio: `0.374`
- MR ratio: `1.877`
- max-jump KS: `0.992`
- pathwise q99 ratio: `0.416`

## Interpretation

The key split is now clear:

1. The **representation** is not dead.
- posterior reconstruction preserves rank much better than the sampled generator
- posterior reconstruction preserves cointegration much better than the sampled generator

2. The **sampled prior** is still wrong.
- it pushes the decoded future toward excessive common-mode coupling
- it collapses effective rank
- it weakens cointegration
- it still fails to generate realistic jump incidence

3. The sequence-aware prior did help one thing:
- it made sampled temporal tails less Gaussian and fixed aggregate kurtosis

But that is not enough for the actual conditional scenario objective, because the sampled joint structure is still wrong.

## Mechanism Conclusion

`266c-v0` is a clean negative result.

It says:
- the temporal bottleneck remains a live first-principles representation
- but the **latent diffusion prior is now the bottleneck**, even after making it sequence-aware

So the next step should **not** be more denoiser micro-tuning inside diffusion.

## Next Question

What is the smallest first-principles generative-core change that:
- keeps the `266` temporal bottleneck,
- keeps the decoder,
- keeps the architecture elegant,
- but replaces the latent diffusion prior with a more direct conditional generator over latent token paths?
