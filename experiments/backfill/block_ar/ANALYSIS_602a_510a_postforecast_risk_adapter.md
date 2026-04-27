# 602a 510a Postforecast Risk Adapter

## Hypothesis

601a selected a frozen-postforecast adapter as the next deployability-layer test. The
question was whether the existing 510a broad-frame generator can be made risk-manager
usable by a disclosed, bounded, widening-only calibration layer without retraining the
learned core.

## Method

602a reused the existing 551a state-score interval adapter on the frozen 510a checkpoint:

- base checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`;
- calibration block: 441 pre-validation windows, 48 samples per window;
- validation block: 441 broad-frame windows, 48 samples per window;
- state feature: `history_abs_move_q90`;
- policy: widening-only residual scaling around the sample median;
- target coverage: `0.95`;
- high-side overcoverage cap: disabled by setting `coverage_hi=1.0`;
- bounded scale range: `[1.0, 1.8]`;
- fitted scale range: `1.125 / 1.450 / 1.800`.

Existing adapter and risk-readiness tests passed before the run:

```text
pytest test_code/test_551a_state_score_interval_scale.py test_code/test_552a_risk_readiness_audit.py -q
8 passed in 1.34s
```

## Result

602a scored `4/11` on the full broad-frame v2 suite.

Key metrics:

- overall 90% coverage improved from broad-frame 510a `69.6%` to `84.0%`;
- h1/h7/h14/h30 coverage became `90.5% / 83.8% / 83.4% / 81.2%`;
- conditional MAE reduction improved to `8.35%`;
- pathwise max-jump KS improved to `0.191`;
- regime layer2 stayed `0/8`;
- level KS fell to `1/25`;
- daily-change KS fell to `13/25`;
- pathwise per-cell extreme jump scale fell to `15/25`;
- persistent undercoverage remained above gate at `7.9%`;
- the hard undercovered cell `(3,3)` remained bad at h7/h14/h30.

Risk-manager readiness comparison:

| Candidate | Full suite | Stress score | Stress pass | Notes |
| --- | ---: | ---: | --- | --- |
| `510a_base_broad` | `5/11` | `1/4` | `False` | Best of the two, but fails lower-only coverage and regime inclusion |
| `602a_postforecast_adapter` | `4/11` | `1/4` | `False` | Better aggregate coverage, worse authenticity |

Risk-readiness details for 602a:

- lower-only coverage: fail, cov90 `0.840`, worst cell `0.546`;
- lower-only regime: fail, worst regime cell `0.303`;
- conditionality: pass, MAE reduction `8.351%`;
- scenario authenticity: fail.

Artifacts:

- `results/autoresearch/602a_510a_postforecast_risk_adapter/full11.json`
- `results/autoresearch/602a_510a_postforecast_risk_adapter/full11.md`
- `results/autoresearch/602a_510a_postforecast_risk_adapter/risk_readiness.json`
- `results/autoresearch/602a_510a_postforecast_risk_adapter/risk_readiness.md`

## Mechanism Read

The adapter solves the easy aggregate undercoverage direction but not the actual sparse
regime/cell allocation problem. Widening around the sample median raises broad coverage,
but the remaining bad cells/regimes are spatially and temporally specific. A state-only
horizon scale cannot direct probability mass into those slices. Because the adapter scales
residuals globally within each state/horizon bin, it also distorts daily-change and extreme
jump scale authenticity.

## Decision

Close this frozen state-score widening adapter as not deployable. It is useful evidence:
overcoverage is not the real blocker, and aggregate widening is insufficient. The next
research step must directly target sparse cell/regime under-inclusion while preserving path
authenticity, or conclude that the currently available conditioning data lacks the signal
required for risk-manager deployability.
