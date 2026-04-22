## 293d / 293e / 293f comparison

### Context
The `293` family moved from latent-conditioning variants to explicit path support because the earlier latent scaffolds never became real path-shape controllers.

The explicit-support bracket is now:
- `293d`: one monolithic coarse path anchor code
- `293e`: compositional knot-token support
- `293f`: monolithic anchor plus light knotwise refinement

The question is whether the branch still needs a better support **representation**, or whether the real bottleneck has moved to support **use** inside the daily joint-law decoder.

### Score trajectory
- `293d`: `4/11`
- `293e`: `3/11`
- `293f`: `4/11`

So the hybrid did not break the frontier. It landed back on the anchored side of the tradeoff.

### What remained alive across all three
The family still has robust local-law signal:
- coverage is alive
  - `0.909 -> 0.901 -> 0.913`
- change KS is alive
  - `25/25 -> 24/25 -> 25/25`
- cross-cell structure is alive
  - corr ratio: `1.301 -> 1.215 -> 1.278`
  - rank ratio: `0.849 -> 0.914 -> 0.863`

That is not the bottleneck anymore.

### What stayed dead across all three
The hard path-shape suites still did not come alive:
- level KS: `1/25 -> 3/25 -> 2/25`
- regime-sensitive width timing: `0.984 -> 1.029 -> 1.049`
- max-jump KS: `0.647 -> 0.625 -> 0.660`
- aggregate MR ratio stayed far below gate:
  - `0.093 -> 0.075 -> 0.098`

So explicit support still is not acting as a decisive low-frequency trajectory controller.

### What changed when the support object changed
`293d`:
- strongest structural anchor
- best active-cell MR correlation: `0.773`
- first support variant to recover cointegration pass and strong daily change KS together

`293e`:
- easiest support object to learn
- best softer path-shape proxies:
  - level KS: `3/25`
  - calibration: `0.038`
  - max-jump KS: `0.625`
- but weaker structural anchor

`293f`:
- preserved the anchored side better than `293e`
- improved coverage and cointegration robustness over `293d`
- but did not keep enough of `293e`'s softer gains

So the branch no longer looks like:
- monolithic support is wrong
- compositional support is right

Instead it looks like:
- both support forms carry useful information
- neither becomes a strong controller of the daily path law

### Mechanism conclusion
The bottleneck has shifted from support **representation** to support **use**.

Why:
- `293d` proved that a hard global anchor can recover some structure even with weak coarse-code accuracy
- `293e` proved that a softer support object improves learnability and softer level/jump proxies
- `293f` proved that combining the two support representations still does not make the scaffold govern the dead suites

That points at the daily decoder:
- it is reading the scaffold as extra context
- it is not being forced to model the future as a residual or refinement around that scaffold

So more support-object surgery is likely to keep rediscovering the same tradeoff.

### Decision
Do **research ideation** next, but not for another support representation.

The next clean mechanism test should be:
- keep the explicit coarse support object
- change the daily joint-law head so it models **residual next-change law around the scaffold**, not the absolute next-change law

That is the cleanest way to test whether the branch is capped because:
- the scaffold is too weak
- or because the model is not using it structurally enough

### Bottom line
`293` is still alive.

But the explicit support-object subfamily is near a local cap **as a representation search**.

The next principled step is to test **support-conditioned residual modeling**, not another coarse support variant.
