# Post-W3 Autoresearch Results Log

**Session**: Post-W3 stacking compass
**Branch**: diffusion-poc-v1
**Started**: 2026-04-21
**Baseline**: 251h (4/11) — W3 clamp removal, co-champion with 250ac
**Target**: cross 5/11+, continue until all 11 suites pass OR user stops

## Iteration #0 — Baseline

- Checkpoint: `models/backfill/251h_noclamp_from251b_s42/best_model.pt`
- n_pass: 4/11
- Passing suites: surface, block_ar, cross_cell_correlation, distributional_fidelity
- Failing suites: coverage, conditionality, time_series, cointegration, regime_coverage, mean_reversion, pathwise_jump_realism
- Key sub-metrics:
  - conditionality.width_turb_calm_ratio: ~1.00 (gate >= 1.15)
  - regime_coverage.layer2/3: FAIL
  - mean_reversion.mr_gt_ratio: 0.511 (gate >= 0.70)
  - time_series.kurtosis_ratio: 2.49 (gate [0.8, 1.25])
  - pathwise_jump_realism.ks_stat: 0.836 (gate <= 0.20)

## Theory queue (initial)

1. NW2 (W3+H1) — regime modulation on unclamped baseline [priority 1]
2. Constraint audit A (idio_scale_clip) [priority 2]
3. NW3 (triple stack) [priority 3, queued]
4. Constraint audit B (log_sigma_z) [priority 4, queued]
5. NW4 (kurtosis investigation) [priority 5, queued]

---
