# 547a Paradigm Decision After Raw-IV Energy Tradeoff

## Evidence

544a and 546a isolate the current tradeoff:

- 544a local-shift AR flow: `5/11`
  - strong support and path realism,
  - weak raw level occupancy and mean-reversion strength.
- 546a paired raw-IV path energy fine-tune: `4/11`
  - level KS improved from `4/25` to `14/25`,
  - mean-reversion ratio improved from `0.360` to `0.453`,
  - but coverage fell to `0.721` and pathwise max-jump KS worsened to `0.772`.

The paired objective is the wrong pressure: with only one realized future per condition,
matching each conditional sample cloud to that one path contracts support and destroys jump
diversity.

## Prior Related Result

504a already tested joint sliced-Wasserstein fine-tuning around the 392a frontier and scored
`7/11`. It was below frontier but informative:

- it preserved conditionality,
- kept daily KS at `25/25`,
- improved level KS only slightly to `11/25`,
- lost worst-cell cointegration.

So repeating 504a around 392a is not justified.

## Decision

Open exactly one new falsifier, 548a, because the mechanism is different from both 504a
and 546a:

- start from 544a local-shift AR flow,
- keep the local-score FM anchor,
- replace paired raw-IV path energy with **unpaired batch-level raw-IV joint-law alignment**,
- align the empirical law of generated `(history, future)` raw-IV paths against the empirical
  law of observed `(history, future)` paths in each batch.

This targets level occupancy without telling the generator that every conditional scenario
must collapse toward the single realized future.

## Acceptance Gate

548a is accepted only if it recovers at least `8/11` and improves level KS relative to 544a
without losing these 544a structural wins:

- daily-change KS pass,
- cross-cell correlation pass,
- cointegration pass,
- pathwise jump realism pass.

Do not sweep SW weights, projection counts, adaptation windows, or checkpoint epochs. If the
first clean unpaired objective is below frontier, close this local objective family.
