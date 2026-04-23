## 296a ideation

### Context
The previous fixed-horizon one-stage joint-law line (`293` through `295`) produced a
clear split:

- it learned stochastic shell behavior well
  - coverage
  - calibration
  - cross-cell dependence
- it did **not** learn the structural center path well
  - level-law fidelity
  - mean reversion
  - worst-cell cointegration robustness

Separately, the earlier deterministic Stage A line showed that `277d` is still the
strongest structural backbone in the reset program:
- `5/11`
- passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`,
  `mean_reversion`
- strong aggregate and active-cell MR support

So the clean next step is not another all-in-one model.
It is a hybrid that lets each component do the job it already proved it can do.

### Alternatives considered
1. **Reuse `277d` as frozen center path and add a learned stochastic shell**
- strongest structural evidence
- cleanest first `296a` baseline
- recommended

2. **Use a learned world-model backbone such as `289e` plus stochastic shell**
- cleaner and more parametric
- but materially weaker structural evidence than `277d`
- good fallback if `277d`-based hybrid stalls

3. **Revive the old retrieval-weighting Stage B**
- rejected
- old objective collapsed or distorted the scenario law
- not the right shell mechanism to reuse

### Decision
Next step: `296a-v0`

Use:
- **frozen `277d` structural backbone**
- **learned residual stochastic shell**

### Concrete architecture
1. **Backbone**
- load the best `277d` checkpoint
- for each history window, produce the deterministic center path
- keep backbone frozen for `v0`

2. **Shell conditioning**
- history
- deterministic future path from `277d`
- deterministic future daily change coordinates implied by that path

3. **Shell target**
- residual future normalized-change coordinates relative to the `277d` center path

4. **Shell output**
- stochastic daily residual token law
- sampled future path is:
  - `backbone daily change coord + sampled residual coord`

### Why this is the cleanest hybrid
It is **not** the old two-level retrieval weighting bank:
- no top-k residual weighting objective
- no expected-distance collapse objective
- no reweighting over stored scenario candidates

It is:
- one explicit structural backbone
- one learned stochastic residual shell

That keeps the split legible:
- backbone owns center path
- shell owns spread and deviations

### Why this matches the evidence
`277d` already solved the exact structural suites the one-stage line could not:
- MR
- cointegration
- structural cross-cell law

`295a` already showed the shell family can solve:
- short-horizon and long-horizon coverage
- calibration
- conditional spread allocation

The missing test is whether combining those strengths in one explicit split can avoid
their previous failures:
- retrieval-weighting Stage B collapse
- one-stage fixed-horizon center-path failure

### Expected effect
If this is right, `296a` should improve on `277d` by restoring:
- coverage
- calibration
- regime width

while preserving most of:
- mean reversion
- cointegration
- cross-cell structure

### Kill criteria
`296a` is alive only if it preserves at least the structural backbone gains of `277d`
while materially improving stochastic suites.

Specifically, it should keep:
- cointegration pass
- mean reversion pass
- cross-cell correlation pass

and improve at least one of:
- coverage
- conditionality
- regime coverage
- jump realism

If it cannot do that, then the hybrid split itself is weaker than hoped and the next
reset should reconsider the decomposition, not just the implementation.
