# RC17 Validation Audit — 2026-03-25

## Scope
Audited all 10 RC17 experiments (155a through 155e_v2).
Ran comprehensive evaluation on 4 key models + long-horizon test.

## Cross-Model Comparison (val split, 441 windows, 50 samples)

| Model | sw | λ_vs | CI worst | CI mean | KS daily | Kurt | Corr | SS | Grow Unc | Suites |
|-------|----|------|---------|---------|----------|------|------|------|----------|--------|
| 155a (MLP) | 0.5 | 0 | 0.655 | - | 22/25 | 0.681 | 0.991 | 0.747 | 0.97 | - |
| **155d** | **0.5** | **0** | **0.748** | **0.828** | **25/25** | **1.166** | **0.910** | **1.078** | **1.00** | **5/6** |
| 155d_v5 | 0.55 | 0 | 0.786 | 0.865 | 24/25 | 1.514 | 0.764 | 3.326 | 1.00 | 4/6 |
| 155e | 0.5 | 0.01 | 0.775 | 0.873 | 25/25 | 1.756 | 1.565 | 1.498 | 1.00 | 4/6 |

## Key Findings

### 155d is the best model (5/6 suites)
- **Only fails S2 (CI worst_cell=0.748, threshold 0.80)**
- Passes S1 (no explosions), S4 (kurtosis 1.17), S5 (growing unc 1.00), S8 (KS 25/25), S9 (corr 0.910)
- Spread-skill ratio 1.078 — near-perfect calibration
- Worst cells: column 4 (deep OTM) — cells (3,4), (0,4), (1,4), (4,4)

### Corrections to prior inline eval claims
- 155d inline eval (ep200): CI=0.745, corr=0.895 → Full val eval: CI=0.748, corr=0.910
  Slight improvement — inline eval used 160 windows, full uses 441. Consistent.
- 155d_v5 inline eval: CI=0.796 → Full val eval: CI=0.786
  Slight decrease — more conservative on full data. Consistent direction.
- 155e inline eval: CI=0.775 → Full val eval: CI=0.775
  Exact match.

### Growing uncertainty PASSES for all CLN models
The concern that "no AR structure → flat spread" was wrong. Temporal attention
with position encodings learns horizon-dependent spread. All models show 1.00
monotonicity ratio.

### Long-horizon 252d test: PASSES
- 155d: zero explosions over 10 windows × 20 samples × 252 days
- Spread grows 1.82x from h=30 to h=252 (healthy growth)
- No compounding errors despite chained 30-frame blocks

### Turb/calm conditionality
- 155d: turb/calm = 0.900 (calm slightly wider — inverted from expected)
- 155e: turb/calm = 0.721 (calm significantly wider — VS pushes this wrong direction)
- This confirms conditionality is NOT learned by the current setup

### Bottleneck cells
Column 4 (deep OTM) is consistently the worst across all models.
Cell (3,4) and (0,4) are the binding constraint for CI passage.

## Outstanding Items
- [ ] Full test split eval (4540+) — needs base prediction generation (~30 min)
- [ ] Multi-seed verification for 155d (seeds 43, 44)
- [ ] 155d_v2 (300ep) best checkpoint at ep120 — evaluate that specific epoch
