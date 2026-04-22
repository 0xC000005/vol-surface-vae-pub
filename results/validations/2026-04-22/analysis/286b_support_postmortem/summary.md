# 286b Support Postmortem

## Result
- Full 11-suite score: `4/11`
- Passes: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`

## What Was Tested
`286b` bracketed `286a` by removing the scale part of the new history-coordinate
support family.

Instead of the full affine transport
- `future = mu_q + sigma_q * ((future_lib - mu_lib) / sigma_lib)`

it used only a history-mean translation
- `future = future_lib + (mu_q - mu_lib)`

Everything else stayed fixed:
- same `283a` reweighting checkpoint
- same candidate bank
- same top-k learned retrieval scores

## Mechanism Read
This does **not** rescue the history-coordinate family.

Relative to `286a`, mean-only transport:
- does not improve level KS at all (`0/25`)
- weakens coverage and calibration
- weakens regime allocation
- worsens jump realism
- still fails short-horizon active mean reversion

So the scale term was not the main culprit. The deeper problem is upstream:
- the current retrieval coordinate and candidate support bank do not contain the right
  central level law for this hierarchy
- post-hoc transport of the same retrieved futures is no longer enough

## Decision
- Close the local `286a` / `286b` history-coordinate transport bracket.
- Do not continue with more support transports on top of the current `277d` retrieval
  bank.
- Next step should be research ideation for a new **Stage A retrieval coordinate /
  support bank**, ideally in the same local-history coordinate that the support
  experiments just made interpretable.
