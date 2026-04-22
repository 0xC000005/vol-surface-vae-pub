## 289d Stage A World-Model Follow-Up

### Context
`289c` confirmed the world-model line is still alive:
- cross-cell structure is now back in gate
- mean-reversion materially improved
- jump scale improved versus `289b`

But the same clean bottleneck remains:
- single latent state rollout is still too low-variance
- change KS is still `0/25`
- cointegration is still below gate
- jump realism is still far too weak

### Decision
Next step: `289d-v0`

### Family
Deterministic latent **token-state** world model.

### Core idea
Keep:
- Stage A as a learned parametric world model
- deterministic autoregressive rollout
- normalized-change target
- no retrieval bank
- no hard low-rank head
- no bounded side paths

Change:
- replace the single latent vector with a small latent token/state set
- update that token set with a sequence-aware latent transition
- decode next normalized change from the updated latent token set plus the current observation embedding

### Minimal 289d-v0 design
- history encoder: map past panel level/change sequence to `K` latent tokens
- observation encoder: current panel level/change -> observation token
- transition: small Transformer or GRU-style block over latent tokens conditioned on the observation token
- decoder: cross-attend or pool latent tokens + observation token to predict next normalized change
- deterministic rollout

### Why this is still first-principles
This is still a learned world model:
- learned state
- learned transition
- learned observation update
- learned decoder

No retrieval memory, no support engineering, no explicit factor restriction.

### Why this is the smallest justified move
`289b -> 289c` showed that:
- improving observation encoding helps
- but a single latent vector still compresses rollout too aggressively

The smallest next capacity increase that keeps the same paradigm is a small latent token/state set.

### Preferred first implementation
- `K = 4` latent tokens
- small Transformer-style token update
- keep widths modest to avoid turning the test into a compute sweep

### Kill criteria
`289d` is only alive if it improves the rollout-capacity symptoms together:
- change KS strictly above `0/25`
- cointegration ratio materially above `0.292`
- jump q90/q99 ratios materially above `0.094 / 0.131`
- cross-cell correlation remains in gate

If those do not move together, the Stage A world-model line is likely capped by deterministic next-change supervision itself and needs a broader formulation rethink.
