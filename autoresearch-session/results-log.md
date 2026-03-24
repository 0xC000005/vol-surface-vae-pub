# RC15 Autoresearch Results Log

**Session**: RC15 — Conditional One-Shot Flow Matching (Literature-Informed)
**Branch**: autoresearch-session-rc15
**Started**: 2026-03-24
**Baseline**: 152e unconditional (PC1=0.989, kurt=1.215, KS 25/25)
**Best conditional reference**: 152b AR flow (5/9 [1,4,5,6,9])
**Literature**: FMAP (2504.03463), Lim et al. (2410.03229), Lim & Erichson (2602.08318)

## Baseline (152e unconditional)
- eff_rank: 5.99 (GT 7.61), PC1: 0.989, PC2: 0.965
- KS daily: 25/25, kurtosis: 1.215, Frobenius: 3.225
- Spread flat (h1=0.041, h30=0.039)
- PC3-5 weak (0.43, 0.23, 0.59)
- UNCONDITIONAL — no conditioning, no suite evaluation possible

---

| # | Exp ID | Direction | Metric | Decision |
|---|--------|-----------|--------|----------|
| 1 | 152f | RC15-H2: Data-dependent source (persistence) | eff_rank=3.41 (152e=6.04), PC1=0.967, PC2=0.928, KS=17/25, kurt=0.852, frob=8.11 | VALUABLE FAILURE — Lim vel-var prediction confirmed (0.50 vs 0.96) but overfits at 4K scale. Use N(0,I) for H1. |
| 2 | 153a | RC15-H1-S1: Conditional + concatenation | eff_rank=6.36, PC1=0.999, PC2=0.996, KS=25/25, kurt=1.113, frob=1.30, spread=0.070 (GT=0.072) | BUILD ON THIS — every metric improved over 152e. Turb/calm=1.138 (weak). Shuffled cond ≈ real cond — population prior, not window-specific. |

