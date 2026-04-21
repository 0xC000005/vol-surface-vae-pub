# 257a Follow-Up Analysis

## Question

After `257a-v0` reached `3/11`, the next decision is whether to stay in the `257`
family and strengthen latent usage, or abandon the paradigm.

## Core Finding

The latent-token VAE is **alive but weak**.

It is not a dead-token failure like `256a`, but the sampled latent variation is far
too small relative to the model's mean-path error.

## Evidence

### 1. Decoder uses token means more than token samples

- `post_vs_zero_mae = 0.0965`
- `prior_mean_vs_zero_mae = 0.0979`
- `post_vs_prior_mean_mae = 0.0057`

Interpretation:

- the decoder meaningfully uses the **history-conditioned token mean**
- but changing from the prior mean to a posterior sample barely moves the output

So the sampled stochastic component is weak relative to the deterministic token mean.

### 2. Sample spread is tiny relative to path error

- ensemble mean abs error: `0.8328`
- sample std mean: `0.0061`
- std / mean-error ratio: `0.0073`

Interpretation:

- stochastic spread is present
- but it is tiny compared with how wrong the center path still is

That explains the outcome:

- coverage improves from `0%` to `20.5%`
- but deterministic suites still fail badly

### 3. KL is present but very small

- validation KL mean: `0.0028`
- prior std mean: `0.450`
- posterior std mean: `0.462`

Interpretation:

- posterior collapse is not total
- but the latent channel is not carrying much information beyond the history-driven
  prior mean

## Mechanistic Conclusion

`257a-v0` is not the wrong paradigm.

The main issue is **underpowered latent influence**:

1. too much direct history bypass in the decoder
2. too little pressure for sampled latents to matter
3. too much of the future explained by the history-conditioned token mean alone

## Most Principled Next Step

Stay in the `257` family and make one focused follow-up:

- `257b`: stronger latent-usage VAE

Targeted changes should be:

- reduce direct history bypass in the decoder
- add a **free-bits / minimum-KL** style constraint instead of only warmup
- make loadings or factor state more token-dependent so token samples affect the mean
- optionally train with a small multi-sample reconstruction/scoring term so scenario
  spread matters in the objective

## Decision

The next iteration should be an **experiment**, not another paradigm shift.

`257a` is the first family with:

- non-zero coverage
- passing cross-cell structure
- non-dead token attention

That is enough evidence to justify one focused `257b` follow-up.
