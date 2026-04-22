# 268a Paradigm Shift Memo

## Context

The clean reset lines have now falsified two broad strategies:

1. **deterministic encoded future targets** (`266a`-`266d`)
2. **plain probabilistic latent-token bottlenecks** (`267a`-`267b`)

The first family failed because the future-token target was the wrong latent object.
The second family failed because the latent path kept collapsing under the plain non-heuristic ELBO setup.

So the next principled move is not another latent patch.

## New Family

**268a-v0: direct conditional future-path flow matching**

Core idea:
- model the future path itself as the stochastic object
- condition only on history
- use a vanilla conditional FM core
- avoid latent bottlenecks entirely

Minimal formulation:
- history encoder -> context vector
- future path `x_t` at interpolation time
- sequence-aware velocity network over future path
- no latent posterior
- no KL
- no low-rank head
- no bounded side paths

## Why This Is The Most Principled Next Step

This is the cleanest Bitter-Lesson-aligned model remaining:
- generic
- high-capacity
- direct
- no handcrafted factorization in the core

It also directly matches the risk-manager object:
- learn the conditional law of the future path itself

## Why Not Other Options

### More latent anti-collapse work
No longer justified without heuristics.

### Return to DiT-style chunk modeling
Too broad and already previously negative in older lines.

### Copula-specific decomposition
Adds explicit structure assumptions the current reset is trying to avoid.

## Proposed 268a-v0

Smallest clean version:
- history GRU encoder
- future-path velocity network:
  - per-step feature projection
  - temporal conv stack over horizon
  - output velocity in future-path space
- Euler sampler
- standard FM objective

## Pre-Registered Question

Can a direct conditional path-space FM model:
- preserve the structural gains seen in late reset lines
- while recovering nontrivial scenario spread
- without latent collapse or deterministic-target mismatch?

## Kill Criteria

Close `268a-v0` quickly if:
- coverage still stays effectively zero
- scenario diversity remains near deterministic
- or cross-cell structure collapses back toward old direct-generator failures

## Decision

Proceed with `268a-v0`.

This is the simplest direct conditional scenario generator still aligned with:
- first principles
- methodological elegance
- Bitter Lesson discipline
- and the long-term multi-factor objective.
