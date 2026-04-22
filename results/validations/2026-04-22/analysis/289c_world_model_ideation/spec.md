## 289c Stage A World-Model Follow-Up

### Context
`289b` kept the world-model paradigm alive, but its rollout collapsed:
- one-step latent prediction is usable
- autoregressive rollout variance collapses
- common structure improves, but sharp change-law / jump behavior disappears

So the next issue is not retrieval, not support engineering, and not direct observation-space collapse.
It is generated-state state-update quality.

### Decision
Next step: `289c-v0`

### Family
Deterministic latent world model with an explicit observation encoder in the recurrent update.

### Core idea
Keep:
- Stage A as a learned parametric world model
- autoregressive generated-state rollout
- normalized-change target
- no retrieval bank
- no hard low-rank head
- no bounded side paths

Change:
- replace the shallow `[curr_level, prev_change] -> latent input` update with a richer learned observation encoder
- feed that observation embedding into the recurrent latent transition at every step
- decode next normalized change from latent state plus the observation embedding, not latent state alone

### Minimal 289c-v0 design
- history encoder: GRU over past panel level/change tokens to initialize latent state
- observation encoder: small MLP from current panel level/change token to observation embedding
- latent transition: GRUCell from observation embedding to latent state
- decoder: MLP from `[latent, obs_embedding]` to next normalized change
- deterministic rollout
- same teacher-forced + rollout loss structure as `289b`

### Why this is still first-principles
This is still a compact learned world model:
- learned state
- learned transition
- learned observation update
- learned decoder

No retrieval bank, no hand-crafted support object, no explicit factor restriction.

### Why this is the smallest justified move
`289b` already showed the latent-state idea helps.
The clean failure is that rollout loses variance and sharpness even when one-step prediction is not dead.

So the smallest justified next move is to improve how generated panel states are encoded back into the latent recurrence, not to add a new family.

### Kill criteria
`289c` is only alive if it materially improves the rollout-collapse symptoms versus `289b`:
- corr ratio higher than `0.44`
- cointegration ratio materially above `0.34`
- jump q90/q99 ratios materially above `0.045 / 0.070`
- deterministic change KS strictly above `0/25`

If those do not move together, the single-state latent world-model family is likely capped and Stage A needs a broader sequence-state rethink.
