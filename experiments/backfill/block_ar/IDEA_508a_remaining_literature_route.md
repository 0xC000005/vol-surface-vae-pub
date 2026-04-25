# 508a Remaining Literature Route

## Context

The post-fetch experiments have now covered the two user-identified literature
directions in minimal local form:

- DistDF-style joint distribution alignment: 504a joint sliced-Wasserstein
  around 392a scored `7/11`, below the `8/11` frontier.
- MixLinear/Minkowski-linear parameter-efficient architecture: 505a scored
  `2/11`.
- Shared-source direct path geometry: 507a repaired cross-cell geometry but
  scored `3/11` and was slow.

The older log also already covers the obvious bridge attempts:

- 421a joint transition-path flow did not combine AR structure with level
  occupancy.
- 423a persistent source-noise AR fine-tune scored `6/11`.
- 415a AR/direct mixture did not beat the `8/11` frontier.

## Remaining Non-Redundant Idea

The only recent-literature idea not yet directly represented is the MMPD-style
patch view: train the generator against distributions of local future patches
rather than the whole path at once or horizon/cell marginals independently.

This is different from prior losses:

- Full-path energy pressures global path geometry and can trade off
  conditionality.
- Marginal CRPS/interval/PIT pressures individual cells and horizons and can
  miss joint local path modes.
- Joint MMD/SW pressures unconditional joint occupancy and can damage structure.
- Patch energy pressures intermediate objects: contiguous multi-day future
  patches across all cells.

The hypothesis is narrow: 392a may already have one-day transition structure but
miss multi-day patch distribution, especially regime-local coverage and
distributional fidelity. A patch loss could be more local than full-path
alignment but more joint than per-cell losses.

## Falsifier

Run one 392a fine-tune with:

- unchanged 392a AR transition core;
- FM anchor retained;
- free-running rollout samples;
- energy score on overlapping `5`-day future patches across all cells;
- one small fixed patch-energy weight;
- official 11-suite evaluation.

This is still a local objective test, so it must be one-shot. If it fails below
frontier, close the MMPD-inspired local objective route and do not sweep patch
lengths or weights.

## Decision

Run 509a patch-energy fine-tune from the 392a frontier. This is the last clean
literature-derived local learned-law falsifier before returning to the product
conclusion or a genuinely larger new-core program.
