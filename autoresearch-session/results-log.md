# RC22 Autoresearch Results Log

**Session**: RC22 — Factorized Decoder & Factor Structure
**Branch**: autoresearch-session-rc22
**Started**: 2026-04-02
**Previous**: RC21 → 8 H1 experiments exhausted, Pareto frontier proved (6/9 three ways)

## Baseline (from RC20.6, trained with IS=0.05 VS=0.5)

| Metric | Softplus best (ep11) | Target |
|--------|---------------------|--------|
| Suites | 6/9 | 7+/9 |
| S2 coverage | FAIL | PASS |
| Noise eff_rank | ~2 (PC1=74%) | >3 (toward GT PC1=52%) |
| Gradient budget | CRPS 29%, VS 5%, IS 66% | Fixed in 166a |

## Key Evidence
- IS at lambda=0.05 was 66% of gradient (10x miscalibrated) — primary distortion
- CRPS gives ~2.9x more spread than centering gradient (qualitatively robust)
- Decoder attention compresses 27 noise dims to PC1=74% (GT: PC1=52%, 5 factors)
- Model achieves GT factor structure at ep10, CRPS destroys by ep40
- noise_dim reduction falsified (bottleneck is decoder attention, not noise input)
- Oracle debiasing → 100% coverage on ALL models (spread adequate, centering wrong)
- Individual authenticity: conditional mean emerges from realistic scenarios, no aux MSE
- Encoder 98/2 split: NOT collapse, downstream conditioning pathway is the issue

## Iterations

| # | Exp ID | Direction | Key Metrics | Decision |
|---|--------|-----------|-------------|----------|
| 1 | 166a | Loss rebalance: IS=0.005 VS=1.0 | 5/9 best (S4=1.000, S8 bias PASS) S2 FAIL | EXHAUSTED — loss tuning ceiling reached |
| — | 167a | Factorized decoder + split conditioning | Target: eff_rank>3, S2 PASS, ≥7/9 | NEXT EXPERIMENT |

## RC22 v2 Key Finding (from 166a + 3 Codex reviews)
- The architecture traces a Pareto frontier (baseline/V2/166a all 6/9 different compositions)
- Loss tuning cannot escape this frontier — architectural change required
- Factorized output (base_head + load_head) separates drift from factor structure post-attention
- Split conditioning (cond_resid via FiLM for load_head) addresses 98/2 encoder dominance
- Codex corrections: small random init for load_head, detached EMA for cond_ref, under-reversion kill check
