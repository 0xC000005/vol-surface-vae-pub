# 254b Anti-Collapse Ideation Memo

Date: 2026-04-21

## Context

`254a-v0` was the first post-`253` deterministic family. It changed the backbone in the
right direction but failed by collapsing too aggressively into a shared common mode:

- `corr_ratio = 2.233`
- `rank_ratio = 0.201`
- `mr_gt_ratio = 2.724`
- `change KS = 0/25`
- `loading_top1_share = 0.739`

The key read is not "dual-timescale is wrong". The read is:

- the backbone remained active,
- but the low-rank readout / common-path parameterization had no strong guard against
  **PC1 domination / rank collapse**,
- so the model solved the task by pushing nearly everything through one dominant
  shared mode.

This is a different failure from:

- `252a`: too much deterministic idio leakage
- `253b`: dormant new branch
- `253c`: active temporal pulse branch but unresolved MR vs fidelity tradeoff

So the most principled next move is **anti-collapse within the 254 family**, not an
immediate family abandonment.

## Candidate Fixes

### Candidate 1: Harder idio floor

Force a larger idiosyncratic contribution so the model cannot collapse entirely into
common mode.

Why reject it:

- this pushes in the wrong direction relative to the `253` lesson
- it risks recreating `252a` / early-`253` style per-cell mean patching
- the missing flexibility should come from a richer **common subspace**, not more idio

### Candidate 2: Dense temporal backbone with low-rank head

Move immediately to a stronger generic dense backbone and hope output constraints are
enough.

Why reject it for now:

- too many new assumptions at once
- too easy to silently relearn dense coupling
- weaker causal link to the specific `254a` failure

### Candidate 3: Anti-collapse dual-timescale readout (`254b`)

Keep the same broad family but change the readout so the common path can express
multiple cross-sectional modes without collapsing into PC1.

This is the recommended next experiment.

## Recommended Design: `254b`

### Core idea

Keep:

- deterministic fixed-horizon center-path
- dual-timescale slow/fast backbone
- low-rank semantics
- bounded idio path
- bounded EC path

Change:

1. replace the single static loading matrix `Lambda(h0)` with a **base + dynamic residual**
   loading decomposition
2. add an explicit **anti-PC1 / anti-rank-collapse regularizer**
3. give the fast branch its own low-rank cross-sectional modulation path instead of
   forcing all temporal diversity through a nearly fixed loading map

### Proposed readout

```text
Lambda_base(h0)        : B x D x L
Lambda_slow_delta(t)   : B x T x D x L   [small amplitude, smooth]
Lambda_fast_delta(t)   : B x T x D x L   [gated, localized]

Lambda_t = Lambda_base
         + alpha_slow * Lambda_slow_delta(t)
         + alpha_fast * gate_fast(t) * Lambda_fast_delta(t)

common_t = Lambda_t @ latent_t
```

Key difference vs `254a-v0`:

- `254a-v0` made temporal variation mostly live in latent amplitudes through a nearly
  fixed cross-sectional map
- `254b` lets the **cross-sectional mode itself move modestly over time**

This should reduce the incentive to collapse everything into one static dominant PC.

### Anti-collapse regularization

Add two targeted penalties:

1. **Top1-share penalty**
   - estimate singular values of the batch-mean loading matrix
   - penalize top-1 variance share above a threshold, e.g. `> 0.55`

2. **Effective-rank floor penalty**
   - penalize effective rank below a floor, e.g. `< 3.0`

These are architecture-level structural penalties, not evaluator-specific hacks.
They directly target the observed failure mechanism.

### Why this is still principled

- it preserves explicit low-rank semantics
- it preserves generalizability across panels `(B,T,D)`
- it does not introduce grid-specific templates
- it attacks the exact observed failure rather than tuning unrelated losses

## Kill Criteria

Mechanism:

1. `loading_top1_share <= 0.60`
2. `loading_eff_rank >= 3.0`
3. `corr_ratio <= 1.6`
4. `rank_ratio >= 0.45`

Outcome:

1. recover at least the `4/11` frontier
2. improve `change KS` materially above `254a-v0` (`0/25`)
3. reduce `max-jump KS` materially below `0.94`
4. avoid reverting to large idio leakage

Decision:

- if `254b` still collapses or stays below the `4/11` frontier, treat the `254`
  dual-timescale family as likely capped and switch to the next paradigm
- if it restores `4/11` with better balance, continue once more inside `254`

## Conclusion

The most principled next experiment is:

## `254b = anti-collapse dual-timescale backbone with dynamic loading modulation`

This is the cleanest test of whether `254a-v0` failed because the family is wrong, or
because the readout collapsed the family into a near rank-1 shared mode.
