## 296a paradigm shift

### Context
The fixed-horizon one-stage joint-law line (`293` through `295`) has now delivered a
clear, internally consistent result:

- it can learn local stochastic law
- it can keep non-degenerate cross-cell structure alive
- `295a` showed it can also allocate conditional spread well

But it still cannot learn the center-path dynamics well enough:
- level-law fidelity stays weak
- mean reversion stays effectively dead
- worst-cell cointegration robustness stays below gate

This means the one-stage fixed-horizon paradigm is learning the **distributional shell**
better than the **structural center path**.

### Decision
Retire the fixed-horizon one-stage joint-law line as the active mainline.

Next active family: `296a`

### New program
Move to a **hybrid two-part scenario generator**:

1. **Structural center-path backbone**
- reuse or rebuild from the strongest deterministic structural line
- objective: mean/path/cointegration/MR suites

2. **Fixed-horizon stochastic shell**
- use the strongest lesson from `293`-`295`
- objective: coverage / conditionality / jump / regime-width suites

The shell should be conditioned on:
- history
- and the backbone path or backbone hidden states

### Why this is the most principled shift
This is not a retreat to the old Stage-B weighting bank.

It keeps what the fixed-horizon one-stage line actually learned:
- joint stochastic law
- spread allocation
- cross-cell dependence

And it stops asking that same family to also solve the structural center path,
which the evidence says it does poorly.

### Why not continue the current paradigm
`295a` was the last justified in-paradigm mechanism-class shift.

It restored:
- horizon-wide coverage
- good calibration
- preserved change-law fidelity

But even then:
- MR collapsed to zero
- level KS stayed weak
- worst-cell cointegration still failed

That is enough evidence to call the current paradigm **capped for the full objective**.

### Concrete next step
Next experiment family: `296a-v0`

Start from:
- the strongest structural backbone available in the archive/frontier
- plus a stochastic shell informed by `295a`

Design rule:
- do not use old retrieval-weighting Stage B
- do not ask the shell to predict the center path
- do not ask the backbone to generate diversity

### Success criterion
The new family is justified only if the split is explicit and testable:
- backbone improves structural deterministic suites
- shell improves stochastic suites on top of that

If that hybrid line also stalls, then the next reset should reconsider the evaluation
decomposition itself, not just the architecture.
