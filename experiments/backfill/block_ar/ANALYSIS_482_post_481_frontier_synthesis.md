# 482 Post-481 Frontier Synthesis

## Context

`481a` tested a clean endpoint-conditioned AR bridge:

```text
p(Y_1:T | H) = p(Y_T | H) * prod_t p(Y_t | H, Y_<t, Y_T)
```

The factorization is principled, deployable, and free of calibration tables, regime
labels, retrieval, low-rank readouts, or validation-future information. Its failure
therefore carries useful evidence: the issue is not merely that previous wrappers
were inelegant; clean learned path-law replacements are also struggling to preserve
the geometry that the frontier model already gets right.

## Frontier Comparison

| model | score | key read |
| --- | ---: | --- |
| `340c_rev` | `7/11` | strong time-series/path geometry but weaker conditionality |
| `392a` | `8/11` | current deployable frontier; local AR plus weak rollout energy preserves the most gates |
| `471a` | `6/11` | preserving 392a source geometry helps, but learned transport weakens MR/level allocation |
| `473a` | `7/11` | partial transport interpolates cleanly but cannot cross level/regime gates |
| `476a` | `4/11` | latent full-path bottleneck smooths away level/jump geometry |
| `481a` | `4/11` | endpoint bridge preserves validity/coverage but loses local path allocation |

Important metrics:

| model | cov90 | cond MAE red. | daily KS | level KS | median | corr ratio | MR active | path KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `392a` | `0.8675` | `5.14%` | `25/25` | `10/25` | `20/25` | `0.963` | `86.8%` | `0.373` |
| `471a` | `0.9061` | `5.63%` | `25/25` | `10/25` | `19/25` | `0.943` | `79.5%` | `0.243` |
| `473a` | `0.8911` | `5.27%` | `25/25` | `11/25` | `20/25` | `0.954` | `81.6%` | `0.310` |
| `476a` | `0.6803` | `-4.23%` | `14/25` | `2/25` | `11/25` | `1.453` | `62.1%` | `0.905` |
| `481a` | `0.8982` | `0.66%` | `7/25` | `1/25` | `19/25` | `0.525` | `47.5%` | `0.784` |

## Mechanism

The evidence separates the problem into two pieces.

First, the model family must preserve the stochastic path geometry already learned
by `392a`: daily-change shape, cross-cell correlation, active mean reversion, and
pathwise jump distribution. Attempts that replace that geometry with iid residual
noise, deterministic latent compression, or a new endpoint bridge collapse several
of these gates.

Second, the remaining frontier failures are mostly probability-allocation failures:
per-cell coverage, regime layer-2 coverage, and unconditional level occupancy. These
are not solved by scalar width, scalar transport strength, or global critic pressure.
Those levers move average calibration but either leave level KS stuck around
`10-13/25` or damage conditionality/cointegration.

The key post-481 read is therefore:

```text
do not rewrite the path law again until we test whether 392a already has enough
candidate support and only assigns the wrong probabilities to candidate paths.
```

## Next Principled Falsifier

Train a frozen-proposal conditional density-ratio resampler:

```text
q_392(Y | H) = frozen 392a proposal
w_theta(H, Y) ~= p_data(Y | H) / q_392(Y | H)
final samples = resample candidates from q_392 using softmax(log w_theta)
```

This is a learned conditional law correction, not a validation oracle and not a
per-cell calibration table:

- `392a` remains frozen, so path geometry is preserved by construction.
- The learned component only reallocates probability mass among deployable proposal
  paths.
- The objective can be binary density-ratio estimation between realized futures and
  proposal futures for the same history.
- If `392a` lacks support for the missing level/regime outcomes, the method will fail
  cleanly and prove that new support generation is required.

## Design Guardrails

The first implementation should stay minimal:

- no regime labels;
- no hand-written per-cell gates;
- no validation-future information;
- no generator fine-tuning;
- no transport ODE;
- no scalar temperature sweep before the base result is known.

The only unavoidable deployment cost is candidate oversampling: generate `K` frozen
392a candidates, score them, and resample the requested number of scenarios.

## Decision

Run one minimal `483a` density-ratio resampling falsifier. This is the cleanest next
step because it directly tests the remaining uncertainty: support versus allocation.
If it fails without improving level/regime coverage, then the frontier evidence will
point away from proposal reweighting and back toward a genuinely stronger support
generator.
