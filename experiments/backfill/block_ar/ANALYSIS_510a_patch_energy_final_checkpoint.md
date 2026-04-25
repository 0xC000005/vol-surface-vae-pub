# 510a Patch Energy Final Checkpoint Audit

## Context

509a best-by-validation checkpoint scored `6/11`, below the 392a frontier. Since
this repo has repeatedly shown that internal validation loss can disagree with
the official 11-suite, 510a evaluated the already-trained 509a final checkpoint.

This is a checkpoint audit, not a new training knob.

## Result

- Model: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- Full suite: `results/block_ar/510a_509a_patch_energy_final_checkpoint/full11.json`
- Score: `8/11`
- Failed: `coverage`, `regime_coverage`, `distributional_fidelity`

Key metrics:

- Coverage90: `0.8732`
- Per-cell coverage range: `0.714` to `0.979`
- Conditionality MAE reduction: `5.12%`
- Daily-change KS: `25/25`
- Level KS: `10/25`
- Median-bias fraction: `20/25`
- Bias magnitude: `25/25`
- Regime layer2: `0/8`
- Cointegration worst-cell ratio: `0.257`
- Cross-cell corr ratio: `0.968`
- Rank ratio: `1.462`
- Mean-reversion ratio: `0.986`
- Path max-jump KS: `0.361`

## Mechanism Read

The final checkpoint recovers the same `8/11` failure set as 392a, while changing
the shape of the frontier:

- Better aggregate coverage than 392a-style weak energy, and fewer severe
  per-cell coverage problems.
- Same level-KS bottleneck at `10/25`.
- Same regime layer2 bottleneck at `0/8`.
- Conditionality passes, but with a thin `5.12%` margin.
- Cointegration passes, but with a thinner worst-cell margin than 392a.

This means the MMPD-inspired patch objective is not dead, but it is still not a
path to 11/11 by itself. It can trade coverage geometry against structural
margins while leaving level/regime occupancy unchanged.

## Decision

Promote 510a as an active `8/11` frontier tie, not a strict replacement for 392a.
The next step should be post-experiment comparison of 392a versus 510a to decide
whether the improved coverage geometry is a usable anchor or whether the thinner
cointegration/conditionality margins make 392a safer.
