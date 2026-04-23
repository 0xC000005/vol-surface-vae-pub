## 277d vs 295a vs 296a

### Context
`296a-v0` was meant to combine:
- `277d` structural center-path validity
- `295a` stochastic-law coverage/calibration

This comparison checks whether the hybrid decomposition itself failed, or whether the
specific daily residual-shell interface failed.

### Headline comparison
| model | score | main passes | main failure mode |
| --- | --- | --- | --- |
| `277d` | `5/11` | surface, block_ar, cointegration, cross_cell_correlation, mean_reversion | deterministic undercoverage, weak level KS, brittle top-1 replay |
| `295a` | `3/11` | surface, block_ar, cross_cell_correlation | good stochastic width, but dead structural center path |
| `296a` | `2/11` | block_ar, cross_cell_correlation | neither structure nor stochastic adequacy survived |

### High-signal metrics
| metric | `277d` | `295a` | `296a` |
| --- | ---: | ---: | ---: |
| score | `5/11` | `3/11` | `2/11` |
| coverage90 | `0.000` | `0.910` | `0.483` |
| calibration error | large deterministic miss | `0.044` | `0.299` |
| change KS pass | `23/25` | `25/25` | `1/25` |
| level KS pass | `0/25` | `2/25` | `0/25` |
| corr ratio | `1.058` | `1.213` | `0.900` |
| cointegration ratio | `0.766` | `0.681` | `0.440` |
| MR ratio | `1.065` | `-0.005` | `1.605` |
| max-jump KS | `0.391` | `0.620` | `0.723` |

### What this proves
`296a` is **not** a blended version of the two parents.

If the hybrid split were wrong at the conceptual level, we would expect `296a` to
look like one parent dominating the other. It does not.

Instead, `296a` is worse than both parents in the exact place where the shell touched
the backbone:
- structural validity was no longer preserved
- stochastic spread was still too narrow and badly calibrated

So the comparison points to a narrower diagnosis:
- the **hybrid decomposition is still plausible**
- the **daily residual-token shell interface is wrong**

### Mechanism read
The failure comes from two coupled design mistakes in `296a`:

1. **Direct daily residual correction is too free**
- the shell can change the daily path locally at every step
- that lets it drag the `277d` center path away from its structural validity

2. **Exact token reconstruction is too deterministic**
- the shell is trained to match one realized residual path
- token entropy collapsed to `~0.13`
- so the shell becomes a narrow corrective median, not a usable scenario law

This is why `296a` lost both:
- the deterministic center path
- and the stochastic shell

### Decision
Do **not** abandon the hybrid program yet.

Do **not** continue the `296a` interface either.

The next step should be `296b` ideation with two explicit constraints:
- the shell must be **mean-preserving** relative to the frozen backbone
- the shell must operate at a **coarser pathwise granularity** than direct daily token
  correction

That is the cleanest way to test whether the hybrid split is still alive without
repeating the `296a` failure mode.
