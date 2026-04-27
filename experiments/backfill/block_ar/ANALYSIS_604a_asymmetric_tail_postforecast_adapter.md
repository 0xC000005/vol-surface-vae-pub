# 604a Asymmetric Tail Postforecast Adapter

## Hypothesis

603a showed that symmetric per-cell/per-horizon widening cannot solve sparse
cell/regime under-inclusion. The suite output showed many regime misses with bias marked
`UP`, meaning the realized future often sat below the lower band. 604a therefore tested a
directional postforecast adapter: fit separate lower-tail and upper-tail widening scales
around the frozen 510a sample median.

This was the minimal directional alternative to another symmetric scale table. If the
issue were one-sided tail allocation, asymmetric scaling should improve lower-only stress
readiness without needing a new learned core.

## Implementation

Added `experiments/backfill/block_ar/evaluate_604a_asymmetric_tail_postforecast_adapter.py`
and focused tests in `test_code/test_604a_asymmetric_tail_adapter.py`.

The adapter is monotone and median-preserving:

- samples below the sample median are scaled by a learned lower-tail scale;
- samples above the sample median are scaled by a learned upper-tail scale;
- scales are fitted per horizon/cell on a pre-validation calibration block;
- the frozen 510a learned core is not retrained.

Focused tests passed:

```text
pytest test_code/test_604a_asymmetric_tail_adapter.py -q
3 passed in 1.29s
```

Run settings:

- base checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`;
- calibration block: 441 pre-validation windows, 48 samples per window;
- validation block: 441 broad-frame windows, 48 samples per window;
- lower/upper miss targets: `0.025 / 0.025`;
- scale range: `[1.0, 2.5]`;
- fitted lower scale range: `1.000 / 1.150 / 1.750`;
- fitted upper scale range: `1.050 / 1.450 / 2.150`.

## Result

604a scored `4/11`, still below broad 510a's `5/11` and not better than 602a/603a.

Key metrics:

- overall 90% coverage: `76.6%`;
- h1/h7/h14/h30 coverage: `88.4% / 77.0% / 76.1% / 73.3%`;
- conditional MAE reduction: `7.20%`;
- regime layer2: `0/8`;
- turbulent h7/h14/h30 layer-1 coverage failed;
- persistent severe undercoverage worsened to `13.2%`;
- daily-change KS improved to `19/25`;
- level KS remained weak at `3/25`;
- pathwise max-jump KS improved to `0.139`;
- per-cell extreme jump scale remained weak at `14/25`.

Risk-manager readiness:

| Candidate | Full suite | Stress score | Stress pass | Worst lower-only cell | Worst regime cell |
| --- | ---: | ---: | --- | ---: | ---: |
| `510a_base_broad` | `5/11` | `1/4` | `False` | `0.478` | `0.213` |
| `602a_state_adapter` | `4/11` | `1/4` | `False` | `0.546` | `0.303` |
| `603a_cell_horizon_adapter` | `4/11` | `1/4` | `False` | `0.556` | `0.360` |
| `604a_asymmetric_tail_adapter` | `4/11` | `1/4` | `False` | `0.512` | `0.270` |

Artifacts:

- `results/autoresearch/604a_510a_asymmetric_tail_postforecast_adapter/full11.json`
- `results/autoresearch/604a_510a_asymmetric_tail_postforecast_adapter/full11.md`
- `results/autoresearch/604a_510a_asymmetric_tail_postforecast_adapter/risk_readiness.json`
- `results/autoresearch/604a_510a_asymmetric_tail_postforecast_adapter/risk_readiness.md`

## Mechanism Read

The calibration block did not learn the missing stress direction. The fitted lower-tail
scales were modest while upper-tail scales were larger, and validation undercoverage
worsened in turbulent regimes. This means the bad validation slices are not corrected by a
stationary per-cell asymmetric residual table. The missing mass is tied to conditional
state and path direction, not just lower-vs-upper residual width.

## Decision

Close asymmetric postforecast tail calibration. The postforecast overlay route now has
three negative variants:

- 602a state/horizon symmetric scaling;
- 603a per-cell/horizon symmetric scaling;
- 604a per-cell/horizon asymmetric tail scaling.

The next step should not be another interval-scale adapter. The remaining hard problem is
conditional sparse stress-path allocation, and the evidence increasingly points to either
a new path-location mechanism or insufficient available conditioning signal.
