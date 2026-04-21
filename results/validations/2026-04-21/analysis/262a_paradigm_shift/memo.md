# 262a Paradigm Shift — Joint Probabilistic Latent-Factor FM

## Why The 260e -> 261* Decomposition Is Now Capped
The `260e -> 261a/b/c/d` line taught three things:

1. `260e` is a strong deterministic center-path core.
2. Residual uncertainty is easier to keep clean in factor space than in panel space.
3. The residual targets induced by freezing `260e` are **misaligned** with the regime-sensitive uncertainty objective.

`261d` made the key point explicit:
- the scale head learned the target residual scale reasonably
- but the target factor-residual scale itself was anti-correlated with vol-of-vol

So another residual-layer tweak is unlikely to fix the real problem.

## New Principle
Stop forcing uncertainty to be learned as a residual around a frozen deterministic model.

Instead, learn:
- the conditional center path
- and the conditional uncertainty path

**jointly**, inside one explicit low-rank latent-factor model.

## Recommended 262a Family
### Object
A joint probabilistic latent-factor FM model with:
- history encoder
- explicit low-rank loadings
- latent factor center path
- latent factor scale path
- vanilla FM on standardized latent innovations

### Decomposition
For each future step:
- latent factor mean path: `mu_t`
- latent factor positive scale path: `sigma_t`
- standardized latent innovation path: `eps_t ~ FM(...)`
- latent factor path: `z_t = mu_t + sigma_t * eps_t`
- panel change path: `Delta x_t = Lambda * z_t`

Optional bounded idio path remains secondary, not primary.

## Why This Is More Elegant
- one coherent probabilistic model instead of a frozen deterministic core plus correction layer
- explicit low-rank structure still preserved
- stochastic engine still vanilla FM
- uncertainty calibration no longer depends on a residual target that may be misaligned with the risk objective
- easier to justify than stacking more residual heuristics

## What To Keep
- `asinh` local-scale change coordinate from the restart line
- explicit low-rank loadings
- bounded idio budget if needed
- minimal temporal backbone

## What To Drop
- frozen `260e` residualization as the main path
- panel-space residual scenario layers
- more residual-only scale / jump knobs on top of `260e`

## First Prototype: 262a-v0
- no frozen deterministic core
- one latent factor mean path
- one latent factor scale path
- standardized latent innovation FM
- decode through explicit loadings
- no extra jump head
- no regime router
- no token hierarchy

## Success Criteria
The first sign that 262a is the right family is not 11/11 immediately.
It is:
- recover at least `4/11`
- while improving one of the uncertainty suites without destroying the deterministic structure
- and without the decomposition misalignment that capped `261d`

## Recommendation
Move to `262a-v0` next.

This is the cleanest paradigm shift available from the current evidence.
