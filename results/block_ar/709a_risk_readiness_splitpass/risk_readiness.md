# 552a Risk-Manager Readiness Audit

## Framing

This audit treats the generator as a risk stress scenario system, not a strictly calibrated conditional law.
High-side overcoverage is not a failure here. Under-inclusion, weak conditionality, unrealistic paths, broken dependence, and missing factor linkage remain failures.

## Candidate Ranking

| Candidate | Original | Stress score | Stress pass | Key warnings |
| --- | ---: | ---: | --- | --- |
| `510a_splitpass` | `7/11` | `2/4` | `False` | level_ks_warning, regime_undercoverage_warning |
| `392a_splitpass` | `6/11` | `1/4` | `False` | level_ks_warning, regime_undercoverage_warning, coverage_underinclusion_warning |

## Best Current Candidate

- best candidate: `510a_splitpass`
- stress pass: `False`
- lower-only coverage pass: `True` (cov90 `0.871`, worst cell `0.708`)
- lower-only regime pass: `False` (worst regime cell `0.538`)
- conditionality pass: `True` (MAE reduction `5.456%`)
- risk-state allocation pass: `True` (observable `True`, oracle future `False`)
- scenario authenticity pass: `False`

## Factor Readiness

- available non-IV factors: `ret, price, slopes, skews, levels`
- current generator scope: `iv_surface_only`
- multifactor ready: `False`

## Decision

No current candidate is fully presentable as a risk-manager stress system. The closest candidate can be shown as a prototype, but the next research step must target regime under-inclusion and scenario authenticity, not overcoverage.
