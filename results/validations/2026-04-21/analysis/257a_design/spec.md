# 257a-v0 Spec

Date: 2026-04-21

## Goal

Run the first stochastic latent-token prototype after deterministic center-path and
deterministic token families were judged exhausted.

## Hypothesis

The future should be represented by a **sampled latent token set** rather than a
deterministic basis selected from history.

## Architecture

1. `history -> encoder -> h`
2. `future (train only) -> posterior encoder -> q(z_tokens | history, future)`
3. `history -> prior head -> p(z_tokens | history)`
4. sample latent token set
5. decoder attends to sampled tokens over time
6. low-rank readout + bounded residual path produce the future trajectory

Default prototype:

- `n_tokens = 4`
- `token_dim = 64`
- `latent_dim = 8`

## Losses

- weighted SmoothL1 on levels
- weighted SmoothL1 on changes
- pathwise max-jump loss
- terminal loss
- residual-budget penalty
- KL regularization with warmup

## Kill Criteria

1. recover at least the `4/11` frontier
2. meaningfully beat deterministic `0%` coverage
3. avoid posterior collapse:
   - KL stays materially non-zero
   - token std stays non-trivial
4. preserve or improve at least one deterministic temporal-law suite

## Decision Rule

- if `257a-v0` beats the deterministic frontier and shows real stochastic coverage,
  continue in latent-token generative families
- if it collapses, escalate to richer latent priors (flow/diffusion) in token space
