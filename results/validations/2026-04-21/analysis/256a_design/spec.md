# 256a-v0 Spec

Date: 2026-04-21

## Goal

Run the first hierarchical latent-token prototype after the continuous common-path
families (`253/254`) and fixed sparse motif routing (`255a`) were judged capped.

## Hypothesis

The deterministic future path should be represented by a **small adaptive set of
history-conditioned tokens**, not by:

- one continuous common path
- or one fixed motif bank

## Architecture

1. `history -> GRU encoder -> h`
2. `h -> one slow token`
3. `h -> K fast tokens`
4. per-horizon temporal queries attend to the token set
5. attention context updates a dynamic latent state
6. state + token context -> factor scores
7. static low-rank loadings from `h` map factor scores to `delta_t`
8. small bounded residual adapter refines the delta
9. integrate to levels

Default prototype:

- `K_fast = 4`
- `token_dim = 64`
- `latent_dim = 8`

## Why This Is Different

- unlike `253/254`, the future is not forced into one dominant continuous basis
- unlike `255a`, the basis is generated per history window rather than chosen from a
  fixed motif bank

## Losses

- weighted SmoothL1 on levels
- weighted SmoothL1 on changes
- pathwise max-jump loss
- terminal loss
- small residual-budget penalty
- attention entropy-floor penalty
- token-utilization entropy-floor penalty

## Kill Criteria

1. recover at least the `4/11` deterministic frontier
2. improve at least one of:
   - `change KS`
   - `max-jump KS`
   - `distributional_fidelity`
3. avoid token collapse:
   - more than one token materially used
4. avoid `254`-style over-shared collapse:
   - no severe PC1 / common-mode domination

## Decision Rule

- if `256a-v0` recovers the frontier and improves temporal-law metrics, continue in
  the token family once more
- if it fails below `4/11` or collapses to a one-token future representation, leave
  deterministic token families and shift to a richer latent generative family
