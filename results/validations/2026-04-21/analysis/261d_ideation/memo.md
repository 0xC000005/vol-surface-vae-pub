# 261d Ideation — Scale-Calibrated Latent-Factor Residual FM

## Starting Point
`261c` restored latent residual amplitude and kept the cleaner factor-space decomposition alive.

What it fixed versus `261b`:
- latent factor std restored to target scale
- coverage recovered materially
- cointegration recovered to pass
- level KS improved
- window-floor pathology mostly disappeared

What it still misses:
- regime-sensitive width is still too weak
- per-regime cell coverage still fails
- jump incidence and pathwise max-jump realism are still too low
- aggregate MR still too weak

## Principle
Do not add panel-space residual hacks.
Do not reopen the deterministic center-path search.

The next residual change should be the smallest elegant one that targets the actual `261c` miss:
- the family has enough average residual amplitude
- but that amplitude is not allocated correctly across regimes / windows / horizons

## Recommended 261d Design
Keep the exact `261c` latent-factor residual FM, but factorize the residual into:
- standardized latent residual path
- times a small positive **conditional scale profile**

### Concretely
1. Freeze `260e`
2. Keep the `261c` factor-space residual FM backbone
3. Add one small scale head from frozen history context
4. Output either:
   - a per-window scalar scale, or
   - a per-window per-horizon scalar profile
5. Multiply sampled latent residual paths by that scale before factor-space centering / decoding

## Why This Is Principled
- stays entirely in factor space
- keeps the FM core vanilla
- adds one interpretable heteroskedasticity mechanism instead of another branch
- directly targets the 261c pathology:
  - wrong residual allocation, not zero total amplitude

## Training Signal
Supervise the scale head using target factor-residual magnitude:
- target factor residuals already exist in `261b/261c`
- use their per-window or per-horizon std as a direct supervised scale target

This is cleaner than trying to get regime sensitivity indirectly from the FM loss alone.

## Recommendation
Run `261d-v0` next:
- same `261c` backbone
- add one conditional scale head
- supervise the scale profile
- keep exact factor-space centering at inference

If that fails, then the next re-think should be about the residual process family itself, not more low-level tuning.
