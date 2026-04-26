# 570a Long-Horizon Rollout Smoke

## Context

569a showed that factor-panel data framing is mechanically ready for local and broad factor stacks. 570a tests the other requested extension: can the current `510a/564a/568a` IV stress-deck system generate beyond the native 30-day horizon?

This is a mechanical rollout smoke, not a statistical validation.

## Initial Finding

The first direct 60-day run failed:

```text
ValueError: Expected n_steps in [1,30], got 60
```

The underlying `340c/510a` sampler is capped by its trained `future_len=30`. So the existing 568a CLI was not long-horizon extensible as written.

## Implementation

Added a `BlockwiseLongHorizonModel` wrapper inside:

- `experiments/backfill/block_ar/generate_568a_risk_scenario_deck.py`

The wrapper:

- keeps the trained 30-day kernel unchanged;
- if requested horizon is above the native cap, generates repeated blocks of at most `30` days;
- updates each candidate path's rolling 30-day history after every block;
- returns one continuous scenario tensor for the requested horizon.

Also added manifest diagnostics:

- finite rate;
- min IV;
- max IV;
- terminal mean IV.

## Verification

Focused tests:

- Red check: failed because `BlockwiseLongHorizonModel` did not exist.
- Final command: `pytest test_code/test_568a_risk_scenario_deck_cli.py test_code/test_570a_long_horizon_smoke.py -q`
- Result: `5 passed`.

Smoke generation artifacts:

- `results/autoresearch/570a_long_horizon_rollout_smoke/h60_manifest.json`
- `results/autoresearch/570a_long_horizon_rollout_smoke/h90_manifest.json`
- `results/autoresearch/570a_long_horizon_rollout_smoke/h252_manifest.json`

## Result

| horizon | scenario shape | bucket counts | finite rate | min IV | max IV | terminal mean IV |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| `60` | `(6, 60, 5, 5)` | `2/2/2` calm/central/stress | `1.0` | `0.010087` | `0.907568` | `0.195214` |
| `90` | `(6, 90, 5, 5)` | `2/2/2` calm/central/stress | `1.0` | `0.010195` | `0.812864` | `0.184023` |
| `252` | `(3, 252, 5, 5)` | `1/1/1` calm/central/stress | `1.0` | `0.010087` | `0.902535` | `0.201870` |

## Mechanism Read

The current system can be operationally extended beyond 30 days by blockwise AR rollout. The resulting paths are finite and remain within the IV support. This is enough to show that the 568a risk-deck generator can produce longer-dated artifacts for desk review.

However, this is horizon extrapolation:

- the learned kernel was trained/evaluated primarily for 30-day rollouts;
- the stress-selection severity score still uses average future IV, now over a longer path;
- no 60/90/252-day statistical suite has been validated;
- cross-block source/noise persistence is not guaranteed to match a model natively trained at those horizons.

## Decision

The long-horizon deployment path is mechanically feasible but not yet scientifically validated.

For near-term risk review, the CLI can now produce 60/90/252-day IV stress decks with explicit extrapolation caveats. For a defensible long-horizon model, the next principled step is not more wrapper tuning; it is training/evaluating a model with native long-horizon objectives or a joint multi-factor panel law whose metrics are defined at those horizons.
