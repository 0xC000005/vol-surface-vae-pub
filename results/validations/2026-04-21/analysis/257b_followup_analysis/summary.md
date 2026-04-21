# 257b Follow-Up Analysis

## Question

`257b` increased coverage relative to `257a` but fell to `2/11`. Is that still a
useful direction, or is the `257` family already capped?

## Core Finding

`257b` strengthened the **history-conditioned token mean** more than the **sampled
latent variation**.

So it bought more spread, but not enough genuinely sample-dependent structure to
improve the overall suite.

## Evidence

### 1. Token selectivity stayed alive

- attention entropy mean: `1.194`
- attention top-1 mean: `0.450`

So this is not a dead-attention failure.

### 2. Token mean influence increased

- `post_vs_zero_mae = 0.1306`
- `prior_mean_vs_zero_mae = 0.1278`

Compared with `257a`, the decoder now depends more strongly on token-conditioned
structure overall.

### 3. Sample-specific influence is still tiny

- `post_vs_prior_mean_mae = 0.0067`
- sample std mean: `0.0075`
- ensemble mean abs error: `0.8328`
- std / mean-error ratio: `0.0091`

Interpretation:

- the decoder still responds mainly to the **token mean**
- changing from the token mean to a sampled latent realization barely changes the
  output
- stochastic variation is still tiny relative to the mean-path error

### 4. Practical effect in the suite

What improved:

- coverage `20.5% -> 25.0%`
- h30 coverage `11.5% -> 22.7%`
- persistent severe undercoverage `69.9% -> 62.3%`

What worsened:

- suite score `3/11 -> 2/11`
- cointegration overall failed
- cross-cell rank failed
- jump realism worsened

## Mechanistic Conclusion

`257b` proves that simply increasing latent pressure inside the same VAE is not
enough.

The family is still alive, but the next improvement must specifically target
**sample-dependent scenario realism**, not just stronger token-mean influence.

## Most Principled Next Step

Do a short **research ideation** iteration before another run.

The leading follow-up should be one of:

1. `257c`: multi-sample scenario training objective
   - e.g. small ensemble reconstruction / CRPS-style term in latent-sample space
2. `258a`: richer latent prior in token space
   - e.g. latent flow/diffusion prior while keeping the decoder family

The key is that the next design must make *sample identity* matter, not only latent
means.
