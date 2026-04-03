# RC22 Autoresearch Results Log

**Session**: RC22 — Loss Rebalancing & Factor Structure Preservation
**Branch**: autoresearch-session-rc22
**Started**: 2026-04-02
**Previous**: RC21 → 7 H1 experiments exhausted, best 6/9 (V2 mean head, S2 PASS)

## Baseline (from RC20.6, trained with IS=0.05 VS=0.5)

| Metric | Softplus best (ep11) | Target |
|--------|---------------------|--------|
| Suites | 6/9 | 7+/9 |
| Gradient budget | CRPS 29%, VS 5%, IS 66% | CRPS 63%, VS 23%, IS 14% |
| Noise eff_rank | ~2 (PC1=74%) | 3+ (toward GT PC1=52%) |
| S2 coverage | FAIL | PASS |
| S3 turb/calm | 1.185 PASS | >1.15 |
| S9 cross_cell | PASS | preserved |

## Key Evidence (from RC21 + gradient decomposition)
- IS at lambda=0.05 was 66% of gradient (10x miscalibrated)
- VS at lambda=0.5 was only 5% of gradient (too weak for cross-cell structure)
- Decoder achieves GT factor structure at epoch 10 (corr=0.389, eff_rank=2.35)
  then CRPS destroys it by epoch 40 (corr=0.756, eff_rank=1.41)
- noise_dim reduction falsified (bottleneck is decoder attention, not noise input)
- Conditional mean should emerge from individual scenario authenticity, not auxiliary losses

## Iterations

| # | Exp ID | Direction | Key Metrics | Decision |
|---|--------|-----------|-------------|----------|
| — | 166a | Loss rebalance: IS=0.005 VS=1.0 | Target: eff_rank 3+, S2 PASS | NEXT EXPERIMENT |
