# 480a: Endpoint-Conditioned AR Bridge

## Context

The current frontier and failed branches now give a clean map:

- `392a` is the deployable learned frontier at `8/11`; it learns local AR
  transition geometry but misses long-horizon level/regime occupancy.
- Wrapper, calibration, and source-transport variants around `392a` can move
  coverage but cannot learn conditional level allocation without breaking
  structure.
- Deterministic future-path bottlenecks (`476`-`478`, consistent with older
  `327/328`) preserve some shared geometry but smooth away validation
  level/jump detail.
- Prior coarse support/scaffold lines (`293/294/315`) showed that explicit
  coarse path objects are directionally useful but locally capped when used as
  side scaffolds for a fixed joint-token decoder.

So the next candidate must not be another wrapper, another full-path bottleneck,
or another side scaffold.

## Hypothesis

Use an exact chain-rule decomposition that makes long-horizon level occupancy
native while preserving local AR path evolution:

```text
p(Y_1:T | H) = p(Y_T | H) * product_t p(Y_t | H, Y_<t, Y_T)
```

In words:

1. sample a stochastic terminal surface from a learned conditional endpoint law;
2. sample the path as an endpoint-conditioned AR bridge.

This is not a low-rank assumption, bounded side path, retrieval bank, or
calibration table. It is a probability identity. The inductive bias is only the
factorization order: terminal level first, local bridge dynamics second.

## Why This Is Different From Prior Coarse-Knot Work

The old `293/294/315` support-object branches used coarse states as scaffolds or
conditioning aids inside a fixed-horizon token model. They did not make a sampled
terminal level a native random variable with an AR transition law conditioned on
the remaining terminal gap at every step.

The bridge formulation changes the learned object:

- the endpoint factor directly owns the 30-day level distribution;
- the bridge factor owns daily changes, jumps, mean reversion, and surface path
  geometry;
- the current generated level remains part of the transition state, avoiding the
  level-anchor loss seen in transition-only models.

## Minimal Model

A first falsifier should remain small:

### Endpoint Law

- coordinate: empirical normal-score surface, same as `340/392`;
- target: terminal score surface `S_T`;
- model: vanilla conditional rectified flow from Gaussian noise to `S_T` given
  history scores.

### Bridge Law

- coordinate: empirical normal-score transition `S_t - S_{t-1}`;
- condition: history encoding, current score surface, sampled endpoint score
  surface, remaining normalized time, and endpoint gap `(S_T - S_{t-1})`;
- model: vanilla one-step transition rectified flow;
- training: teacher-forced real prefixes and real endpoints first.

### Sampling

```text
endpoint ~ p(S_T | H)
prefix = history
for t = 1..T:
    sample delta_t ~ p(delta_t | H, prefix, endpoint, remaining_gap)
    S_t = S_{t-1} + delta_t
```

The first implementation should not force the final generated point to equal the
sampled endpoint. The endpoint is conditioning information, not a hard projection,
because hard projection would create a bridge-specific jump artifact. If the
sampled path ignores the endpoint, that falsifies the bridge-conditioning
strength.

## Success / Kill Criteria

Keep the branch alive only if it beats old direct/scaffold models and preserves
the 392a structural passes:

- daily-change KS near `392a` levels;
- cross-cell correlation and rank pass;
- mean reversion pass or near-pass;
- level KS, coverage allocation, or regime layer-2 improves materially over
  `392a`/`471a` wrappers.

Kill quickly if:

- endpoint sampling is weak and level KS does not improve;
- bridge conditioning damages daily moves, mean reversion, or pathwise jumps;
- the model behaves like another width actuator.

## Decision

Run one endpoint-conditioned AR bridge falsifier next. Do not add coarse-knot
sequences, support codebooks, calibration layers, or endpoint hard projection in
the first implementation.
