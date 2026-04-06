# 186a Hard-Slice / Tail-Aware Objective Spec

## Goal

Keep the `183c` model class fixed and change only the training signal.

The repeated failure pattern after `183c -> 185b` is:

- broad structure is already good
- the model knows roughly where difficult slices are
- but training still rewards diffuse allocation too much
- the same cluster survives:
  - `S3` local conditional width
  - `S7` regime-by-cell hard slices
  - tightened `S4` kurtosis / quiet-shoulder-extreme mix

So `186a` tests the simpler hypothesis:

> the bottleneck is objective weighting and curriculum, not missing architecture.

## Principle

No architecture change.

Keep:

- explicit mean branch
- explicit covariance branch
- `183c` state-metric residual transport

Change:

- sampling curriculum
- elementwise flow-matching weights
- local/band supervision weights
- direct quiet / shoulder / extreme spectrum pressure

## Training Changes

### 1. Rare-window sampling

Use a weighted sampler over training windows based on realized future difficulty:

- average change scale
- late-horizon change scale
- max change scale

This is generic and data-agnostic: it oversamples rare difficult windows without any IV-specific labels.

### 2. Hard-slice local weighting

Upweight local control errors where the teacher target says concentration matters most:

- larger `|target_local_log|`
- later horizons
- harder windows

This should push uncertainty toward the truly difficult `horizon x cell` slices instead of letting it spread diffusely.

### 3. Tail-aware flow-matching weights

Weight the flow-matching loss by target residual magnitude regime:

- quiet region
- shoulder region
- extreme region

with extra emphasis on:

- quiet mass
- extremes

relative to shoulders.

This directly targets the tightened `S4` failure mode:

- too little quiet mass
- too much shoulder mass

### 4. Differentiable spectrum loss

Add a soft quiet / shoulder / extreme mass loss on `|pred_v|` vs `|target_v|`.

This keeps the objective aligned with the benchmark concern rather than relying on aggregate kurtosis alone.

## Why This Is Principled

This is a general training-objective change, not an IV-specific rule.

It does not hard-code:

- turbulence labels
- specific cells
- specific maturities

It only says:

- rare conditional failures should matter more in training
- the residual law should match quiet / shoulder / extreme mass more faithfully

That is generic across financial factor systems.

## Expected Readout

If the hypothesis is right, `186a` should improve the stricter anchor on:

- `S3`
- `S4`
- `S7`

without giving back:

- `S2`
- `S8`
- `S10`
- `S11`

If it fails, that is evidence that objective weighting alone is not enough and we should revisit model class again.
