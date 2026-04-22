# 266b Postmortem

- checkpoint: `models/backfill/266b_v0_s42/best_model.pt`
- best epoch: `28`
- suite score: `2/11`
- passes: `surface`, `block_ar`

## Headline Read

`266b-v0` did not improve the frontier, but it produced a more informative failure than `266a`.

The temporally structured bottleneck is **not** the problem.
The active problem is now the **latent prior over the token path**.

## Full Sample Metrics

- coverage90: `0.304`
- calibration error: `0.346`
- turb/calm width ratio: `1.056`
- ACF corr: `0.637`
- kurtosis ratio: `0.263`
- corr ratio: `2.165`
- rank ratio: `0.229`
- cointegration ratio: `0.371`
- MR ratio: `1.889`
- max-jump KS: `0.979`
- pathwise q99 ratio: `0.484`

Relative to `266a`:
- move-size profile improved materially
- jump q99 improved materially
- cointegration improved materially
- but coverage, calibration, and cross-cell common-mode collapse became much worse

## Reconstruction vs Sampling

### Reconstruction Probe

- ACF corr: `0.417`
- kurtosis ratio: `0.266`
- corr ratio: `0.759`
- rank ratio: `0.747`
- cointegration ratio: `0.706`
- MR ratio: `1.902`
- max-jump KS: `1.000`
- pathwise q99 ratio: `0.005`

### Interpretation

The temporal bottleneck representation itself is useful:
- reconstruction preserves rank structure much better than `266a`
- reconstruction preserves cointegration much better than `266a`

But reconstruction still has a serious pathology:
- jump scale is still almost absent
- short-horizon MR is still too strong

The sampling prior then introduces a second pathology:
- it injects more jump amplitude
- but does so with strong positive bias, severe undercoverage, and excessive common-mode collapse

## Mechanism Conclusion

`266b-v0` says something important:

- moving from a single future code to a short latent token path was a **real representational improvement**
- but the current flattened latent diffusion prior is not the right way to generate that token path

So the `266` family is still alive, but the active bottleneck has shifted:

**representation improved; prior geometry is now the problem**

## Next Question

The next principled question is:

what is the smallest first-principles change to the latent token-path prior that preserves the improved token representation at sampling time, without reintroducing low-rank heads, bounded side paths, or bespoke correction mechanisms?
