# 267b Targeted Ideation Memo

## Context

`267a-v0` established two things at once:

1. The probabilistic latent-token paradigm is alive.
2. The plain ELBO implementation collapses immediately because the decoder can ignore the latent path.

The evidence for collapse was clean:
- KL stayed at `0.0`
- prior/posterior std stayed identical
- sampled coverage remained near zero
- jump incidence remained near zero

So the next step should not be KL heuristics first.

## Candidate Directions

### Option A: Reduce Decoder History Bypass

Keep the `267a` family intact, but remove the strongest latent-ignoring path:
- do not concatenate the full history context to every future decoding step
- let history context set the prior and the decoder initial state
- let the latent token path carry the future-specific information

Why this is the cleanest next move:
- changes exactly one mechanism
- directly targets the diagnosed collapse path
- stays first-principles and architecture-level
- avoids objective hacks

### Option B: KL Warmup / Free Bits

Why it is not the next step:
- heuristic
- easy to use as a rescue knob
- less publishable as the first anti-collapse response

### Option C: Weaker Decoder Capacity Everywhere

Why it is not the next step:
- too blunt
- confounds latent usage with general underfitting

## Recommended Next Family

**267b-v0: probabilistic latent-token model with reduced decoder history bypass**

Keep:
- same prior head
- same posterior head
- same latent token bottleneck
- same probabilistic objective

Change only:
- decoder sees latent token path and positional inputs directly
- history context is used only through the initial hidden state (and prior/posterior networks), not repeated at every decoding step

## Why 267b Is The Most Principled Next Step

This is the smallest first-principles anti-collapse change.

If `267a` collapsed because the decoder did not need the latent path, then the cleanest falsifier is:

**make the latent path more necessary without adding training heuristics.**

## Pre-Registered Question

Does reducing the decoder's direct history bypass:
- increase KL above zero
- increase sample diversity and coverage
- preserve or improve cross-cell structure

without introducing more architectural machinery?

## Kill Criteria

Close `267b-v0` quickly if:
- KL still stays near zero
- coverage remains near zero
- or structural metrics collapse while trying to force latent usage

## Decision

Proceed with `267b-v0`.

Do not:
- add free bits first
- add KL annealing first
- add auxiliary losses or side heads first

The clean next test is decoder-bypass reduction.
