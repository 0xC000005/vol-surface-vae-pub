# RC14 Autoresearch Results Log

**Session**: RC14 — Structural Diversity + Calibrated Spread
**Branch**: autoresearch-session-rc14
**Started**: 2026-03-23
**Baseline**: 146b (5/9, score 69.14)
**Best reference**: 150b s42 (7/9, seed-specific, epoch 8)

## Baseline (146b)
- Suites: 5/9 [1,3,4,5,6] — fails [2,7,8,9]
- CI_90: 77%, rank_ratio: 0.45, KS: 21/25
- Two independent problems: P1 (spread+regime), P2 (rank)

---

| # | Exp ID | Direction | Suites | Decision |
|---|--------|-----------|--------|----------|
| 1 | 151a | H0: Gaussian copula ceiling | 4/9 (PASS: 1,3,5,6+9 — FAIL: 2,4,7,8) | INFORMATIVE: Suite 9 fixable, but post-hoc breaks S4/S8. Proceed H1a/H1b. |
| 2 | 151b | H1a-S1: Detach skip (MLP decoder, bug) | 4/9 (PASS: 1,3,5,6) | BUG: used MLP decoder not transformer. Skip was zero. Invalid. |
| 2b | 151b_v3 | H1a-S1: Detach skip (transformer) | 4/9 (PASS: 1,3,5,9) | PARTIAL: Suite 9 PASS (rank 0.89)! But kurtosis 0.35, lost coint. |

### Iteration 1: Exp 151a — H0 Gaussian Copula Ceiling
- **Hypothesis**: ECC reordering with GT Gaussian copula. Ceiling for Suite 9.
- **Result**: Suite 9 PASS (rank_ratio 0.45→1.44). But Suite 4 FAIL (kurtosis 1.21→0.38) and Suite 8 FAIL (KS 21→6).
- **KEY FINDING**: Cross-cell correlation IS the Suite 9 bottleneck (not marginals).
  Post-hoc reordering breaks temporal coherence → must inject at generation time.
- **MECHANISM**: Per-timestep independent copula reordering assigns different ranks
  to member k's cell j at consecutive timesteps, creating artificial jumps in daily
  changes. This destroys kurtosis (heavy tails) and KS daily (distribution shape).
- **Decision gate**: Suite 9 passes with copula → **H1a/H1b is the right direction**.
  Marginals are sufficient. Correlation structure needs fixing at generation time.

