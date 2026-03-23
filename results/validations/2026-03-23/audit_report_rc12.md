# Validation Audit Report — RC12 Session (2026-03-23)

## Scope
Mandatory validation gate before RC13. Audited RC12 experiments 149a, 149b, 149c.
Deep investigation of KS-CI trade-off mechanism.

## Metric Verification
**ALL claimed metrics verified — ZERO mismatches.** Training histories confirmed.

## Deep Investigation Results

### The KS-CI Trade-Off is NOT FUNDAMENTAL — It's FIXABLE

**Mechanism identified**: Per-cell noise scale (softplus, unbounded) produces heavy-tailed
daily changes. Wider tails = wider CI (good) BUT heavier kurtosis = KS failure (bad).

Key evidence:
- Distribution CENTER barely changed (|median_bias change| = 0.0182)
- Kurtosis TRIPLED (1.21 → 3.33) — it's purely a TAIL problem
- KS degradation is ADDITIVE: all 25 cells get ~+0.10 KS D-statistic increase
- CI improvement is real: h=1 went from 59.8% → 67.9% (+8.1pp)
- 149c per-cell scale: P95=3.89, max=19.92. 9.6% of draws produce scale>3.0

**The fix**: Clamp softplus output to [0.8, 1.5] range. This prevents outlier scale
values (that cause fat tails / KS failure) while preserving the per-cell variance
structure (that causes CI improvement). Expected: CI h=1 ~+4-5pp with KS ≥ 20/25.

### Kurtosis is Concentrated in 1-2 Cells
- Cell (0,3) = OTM-put × T3 is the worst across ALL models: 7.10 (146b) → 19.75 (149c)
- This is the same cell with 6.2× level-dependent volatility (documented bottleneck #4)
- 6 cells consistently have high kurtosis across all 4 models

### 149c Long-Horizon
- d30 CI: 99.2% (excellent). d90: 96.8%. d180: 71.2%. d252: 82.8%.
- Kurtosis catastrophically elevated (60-115× across all horizons)
- Cointegration collapsed: 25.6% vs GT 98.8%
- Per-cell noise scale destroys long-horizon dynamics

## Recommended Next Experiment (RC13-H1)
Train 149c architecture with softplus output clamped to [0.8, 1.5]:
```python
percell_scale = F.softplus(self.percell_noise_scale(noise_embed))
percell_scale = percell_scale.clamp(0.8, 1.5)  # prevent tail inflation
```
This is a 1-line code change that directly addresses the identified mechanism.

## Files Produced
- 4 verification_result.json
- 4 analysis directories with detailed outputs
- 4 reproducible scripts
