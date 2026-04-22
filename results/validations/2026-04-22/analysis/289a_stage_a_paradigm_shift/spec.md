## 289a Stage A Paradigm Shift

### Decision
Retire the deterministic Stage A retrieval implementation as the active search line.

Keep:
- the **hierarchical decomposition**
  - Stage A: deterministic backbone targeting the oracle-reachable `8/11`
  - Stage B: stochastic/distributional layer for the final `3/11`

Replace:
- the current Stage A **retrieval-and-replay** implementation

### Why the Retrieval Line Is Capped
The current deterministic retrieval bracket now has a clean failure signature:

- `277d`
  - best deterministic frontier in this reset line: `5/11`
  - strong aggregate MR and active-cell support
  - but weak level KS / local fidelity
- `287e`
  - recovers strong local change law and better level KS
  - keeps cointegration and cross-cell structure in gate
  - but active-cell MR support collapses
- `288a`
  - soft replay improves level-side support further
  - but destroys change KS and jump realism

This means the local retrieval program is stuck in a structural tradeoff:
- sharp replay preserves change law but is too brittle
- soft replay improves level fit but oversmooths the dynamics

That is a support-mechanism bottleneck, not just a tuning issue.

### New Stage A Family
`289a-v0`: **deterministic causal world-model backbone**

This is a learned parametric model, not a retrieval model.

#### Core idea
- ingest recent panel history
- maintain a learned hidden state of the system
- predict the next normalized panel change jointly
- feed generated state forward autoregressively

#### Minimal architecture
- causal Transformer first
- fallback later if needed: GRU / SSM

#### First-principles constraints
- no retrieval bank
- no nearest-neighbor replay
- no support transport
- no low-rank output head
- no bounded idio side path
- no bounded EC baseline in the core model
- no handcrafted Stage A residual bank

### Proposed 289a-v0 Design

#### Input
- recent history of the full panel
- use a normalized-change representation as the main prediction target
- keep the representation joint across all cells

#### Backbone
- causal sequence model over the history and generated future state
- hidden state is learned end to end
- generated future states feed back into later predictions

#### Output
- deterministic next normalized change for the whole panel jointly
- decode autoregressively over the horizon

### Why This Is the Right Reset
- it keeps the hierarchical `8 + 3` program
- it removes the retrieval-bank bottleneck completely
- it is more Bitter-Lesson-aligned:
  - fewer support heuristics
  - more learned capacity
  - less hand-engineered replay logic
- it scales more naturally to longer horizons and other factor panels

### Kill Criteria
`289a` is only worth keeping if it can beat the retrieval cap on the deterministic suites.

Minimum bar:
- match or beat `277d` on mean-reversion structure
- keep cointegration and cross-cell structure in gate
- improve level-side fidelity beyond `277d`

If it cannot do that, the next rethink is at the deterministic Stage A world-model family level, not another retrieval-style detour.

### Immediate Next Action
Implement `289a-v0` in fresh files as:
- deterministic causal Transformer backbone
- normalized-change target
- autoregressive generated-state rollout
- direct Stage A evaluation on the common 11-suite
