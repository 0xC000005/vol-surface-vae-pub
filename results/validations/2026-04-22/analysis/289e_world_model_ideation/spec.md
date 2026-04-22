## 289e Stage A World-Model Follow-Up

### Context
`289d` falsified the naive token-state capacity increase:
- aggregate cointegration improved
- but common cross-cell structure and level realism regressed sharply
- the teacher-forced path itself became too smooth

That means the next bottleneck is not simply “more latent slots.”

The cleaner interpretation is:
- `289c` compressed history too aggressively into one recurrent state
- `289d` loosened that compression by fragmenting the state, and lost shared global structure

### Decision
Next step: `289e-v0`

### Family
Deterministic world model with:
- one global latent recurrent state
- explicit history-memory access during rollout

### Core idea
Keep:
- deterministic Stage A world-model framing
- normalized-change target
- autoregressive generated-state rollout
- no retrieval bank
- no hand-crafted factor head
- no bounded side paths

Change:
- preserve the stronger `289c` global latent state
- keep the explicit observation encoder
- add a read-only encoded history memory that the model can attend to at every generated step

### Minimal 289e-v0 design
- history encoder:
  - encode past panel level/change tokens into a sequence of history memory vectors
  - also produce the initial recurrent latent state
- observation encoder:
  - current panel level/change -> observation embedding
- history read:
  - use the current latent state and/or observation embedding to attend over the encoded history memory
- latent transition:
  - recurrent state update from `[obs_embedding, history_context]`
- decoder:
  - predict next normalized change from `[latent_state, obs_embedding, history_context]`

### Why this is still first-principles
This remains a learned parametric world model:
- learned state
- learned history representation
- learned transition
- learned decoder

No retrieval support object, no explicit factor hand-constraints, no side channels.

### Why this is the smallest justified move
It directly addresses the current evidence:
- `289c` kept global structure but forgot too much under rollout
- `289d` added state capacity but destroyed the global structure

So the next clean hypothesis is:
- keep the global state
- give it access to persistent history memory instead of multiplying free latent slots

### Kill criteria
`289e` is only alive if it keeps `289c`-level common structure and improves at least one of the still-dead deterministic suites:
- change KS > `0/25`
- cointegration > `0.292` while corr ratio stays in gate
- jump q90/q99 > `0.094 / 0.131`

If not, the Stage A world-model line is likely capped by deterministic next-change supervision itself rather than by state architecture alone.
