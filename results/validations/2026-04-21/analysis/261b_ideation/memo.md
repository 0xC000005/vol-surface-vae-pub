# 261b Ideation — Latent Factor Residual FM on Top of Frozen 260e

## Why 261a Is Not the Right Residual Space
`261a` proved that a direct panel-space residual layer is not a clean decomposition.

What it got right:
- calibration improved
- regime-sensitive width moved in the right direction
- level KS improved
- cross-cell structure improved

What it got wrong:
- cointegration regressed
- mean reversion regressed
- jump realism regressed
- exact zero-mean centering in raw residuals still did **not** preserve the mean level path after integration and support/clamping

So the next residual family should not add more knobs to panel-space residuals.

## Principle
Keep the residual uncertainty inside the same low-rank structural subspace that already defines the frozen deterministic core.

That means:
- freeze `260e`
- freeze its low-rank readout/loadings
- model stochastic residuals in **latent factor space**
- decode them back through the frozen loadings
- keep idio residuals off in `v0`

## Recommended 261b Design
### Decomposition
1. Freeze `260e`
2. For each window, compute frozen loadings and the frozen center path
3. Define the transformed panel residual target
4. Project that residual target into factor coordinates under the frozen loadings
5. Train a small residual FM model on the factor-residual path
6. At inference:
   - sample factor residual paths
   - exact-center them in factor space
   - decode through frozen loadings
   - add to the frozen center path

### Why This Is Cleaner
- residual uncertainty stays inside the same low-rank structure as the deterministic core
- fewer degrees of freedom than 261a
- better protection against destroying cointegration / MR / cross-cell structure
- avoids reopening the deterministic architecture
- still uses vanilla FM as the generative core

## 261b-v0
- shallow temporal backbone over factor residuals
- no panel-space residual head
- no new anchor branch
- no motif/token machinery
- no additional deterministic losses

## Kill Criteria
Reject `261b-v0` if:
- it still materially shifts the frozen `260e` mean path in level space
- it fails to beat `261a` on at least one of:
  - conditionality
  - regime coverage
  - jump realism
while preserving or improving:
  - cross-cell structure
  - cointegration
  - MR relative to 261a

## Recommendation
Run `261b-v0` next.

This is the most elegant next family because it fixes the live `261a` pathology without adding another bespoke branch or reopening the deterministic search.
