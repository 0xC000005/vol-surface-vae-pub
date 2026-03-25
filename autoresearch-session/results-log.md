# RC17 Autoresearch Results Log

**Session**: RC17 — Principled CI Calibration via Proper Scoring Rules + CLN Noise
**Branch**: autoresearch-session-rc17
**Started**: 2026-03-25
**Baseline**: 154b (CI worst=0.544, uncond residual FM + CFM loss)
**Target**: CI worst_cell >= 0.80 while corr_ratio > 0.80 and kurtosis 0.5-2.0

**Key literature backing**:
- H1r/H2r: AIFS-CRPS (2412.15832), CRPS-LAM (2510.09484), Lakatos (2509.02784)
- H3: AIFS-CRPS exact implementation (Anemoi source), ConditionalLayerNorm zero-init
- H5: Latte (2401.03048, TMLR 2025), FGN (2506.10772)

**Mandatory post-experiment investigations**: A (spread), B (quality), C (loss), D (noise), E (comparison), F (WHY)

---

| # | Exp ID | Direction | Metric | Decision |
|---|--------|-----------|--------|----------|
| 1 | 155a | H1r: Single-pass residual MLP + afCRPS | CI=0.668 (ep80), corr=1.010 (ep200), KS=24/25 | VALUABLE FAILURE — afCRPS improves CI +23% vs CFM but MLP can't resist spread contraction. |
| 2 | 155b | H2r: Single-pass residual MLP + afCRPS + VS | CI=0.750 (ep40 peak), corr=1.107, KS=9/25 | VALUABLE FAILURE — VS pushes wider spread but collapses faster. KS degraded. |
| 3 | 155c | H3: CLN velocity net + afCRPS (within FM) | CI=0.128 (ep40), corr=0.467, CLN_scale=1.15 | VALUABLE FAILURE — ODE contracts CLN diversity. CLN disrupts spatial correlation. |
| 4 | 155d | H5: CLN transformer + afCRPS (NO ODE) | CI=0.745, KS=25/25, kurt=1.09, corr=0.895, SS=1.08 | **BREAKTHROUGH** — All metrics improve monotonically. No spread contraction. Build on this. |

