## 289b Stage A World-Model Follow-Up

### Context
`289a` falsified the direct observation-space deterministic world model.

The failure was clean:
- no explosion
- no instability
- instead: strong oversmoothing and decorrelation

This suggests the world-model paradigm is still viable, but the state representation is wrong.

### Decision
Next step: `289b-v0`

### Family
Deterministic **latent** world model.

### Core idea
Keep:
- Stage A as a learned parametric world model
- autoregressive generated-state rollout
- no retrieval bank
- no low-rank hand-constraint
- no bounded side paths

Change:
- add a learned observation bottleneck

That means:
- encode panel state to latent state
- model latent transition autoregressively
- decode latent state back to panel level / change

### Why This Is Still First-Principles
This is not a return to retrieval or handcrafted support objects.
It is still a learned parametric transition model.

The bottleneck is justified because:
- direct observation-space transition was too hard
- a learned latent state is the minimal way to let the model discover a compact system state without hard-coding low-rank structure

### Minimal 289b-v0 Design
- panel encoder: MLP or small Transformer encoder from current normalized panel state to latent state
- latent transition: causal GRU or Transformer over latent states
- panel decoder: latent state back to next normalized change or next normalized level
- deterministic rollout

### Preference
For the first follow-up:
- use a **GRU latent transition**
- keep the encoder/decoder small

Reason:
- this tests the latent-state hypothesis directly
- with less compute and less architectural noise than another Transformer stack

### Kill Criteria
`289b` is only alive if it materially beats `289a` on the deterministic structural suites:
- cross-cell structure must recover into gate
- cointegration must recover into gate
- jump realism must recover materially above the `289a` collapse

If not, the problem is deeper than observation-space vs latent-space parameterization, and Stage A may need a broader rethink again.
