## 288a Support-Use Ideation

### Context
The `287c / 287d / 287e` bracket isolated the query-key family cleanly:

- `287c`: globally anchored query key
  - better aggregate MR and slow structure
  - weaker local level fidelity
- `287d`: fully local-history query key
  - better local level KS
  - mean-reversion structure collapses
- `287e`: mixed query key
  - recovers aggregate MR and active-cell correlation
  - keeps strong local change law and better level KS than `287c`
  - but still stays at `4/11`

So the live bottleneck is no longer the query key.

### Decision
Close the `287` query-key branch as locally exhausted.

Next family: `288a-v0`

### Core Idea
Keep fixed:
- deterministic Stage A retrieval program
- normalized-change replay support primitive
- `287e` mixed query key

Change only the support use:
- from single-nearest replay
- to deterministic top-k / soft retrieval replay

### Why This Is The Smallest Principled Step
The `287` bracket already says the embedding can express the relevant tradeoff.
What it cannot do with one nearest neighbor is:
- preserve local fidelity
- while activating the right set of strong MR cells

A soft top-k replay is the smallest way to test whether the cap is in:
- **which** support example is used
- rather than **how** it is embedded

### Minimal 288a-v0 Design
- retrieve top-k futures under the fixed `287e` score
- convert scores to deterministic weights
  - simple softmax over top-k scores
- replay the weighted average of normalized future changes
- no stochasticity yet
- no new support object
- no new side path

### Kill Criteria
`288a` is worth keeping only if it improves at least one structural suite without giving back the others:
- `mean_reversion` active-pass behavior must improve materially over `287e`
- `level KS` must stay at least near the `287e` range
- `cointegration` and `cross_cell_correlation` must remain in gate

If not, the deterministic Stage A retrieval line is likely capped and the next move should be a broader Stage A reset rather than more local smoothing.
