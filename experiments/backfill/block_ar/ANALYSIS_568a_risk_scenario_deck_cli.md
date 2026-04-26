# 568a Risk Scenario Deck CLI

## Context

567a packaged `564a` as the current constrained risk-manager deployable prototype, but the package was still a report. A risk user needs a direct artifact generator, not only an evaluator.

568a adds a minimal CLI that produces a stress scenario deck and manifest from the `510a` learned base law using the same severity-stratified policy as 564a.

## Implementation

Added:

- `experiments/backfill/block_ar/generate_568a_risk_scenario_deck.py`
- `test_code/test_568a_risk_scenario_deck_cli.py`

The CLI:

- loads the `510a` checkpoint through the existing `340c` loader;
- uses the latest 30-day IV history by default, or a supplied `history_end_index`;
- generates candidate paths from the learned conditional law;
- applies the 564a calm/central/stress severity-stratified selection policy;
- writes a compressed `.npz` deck containing `history`, `scenarios`, `bucket_labels`, and `path_mean_iv`;
- writes a JSON manifest that explicitly states the selected deck is not calibrated probabilities.

## Verification

Red/green tests:

- Initial test run failed because `generate_568a_risk_scenario_deck.py` did not exist.
- Final focused regression command: `pytest test_code/test_564a_stress_selection_policy.py test_code/test_568a_risk_scenario_deck_cli.py -q`
- Result: `4 passed`.

Smoke generation:

```bash
python experiments/backfill/block_ar/generate_568a_risk_scenario_deck.py \
  --samples 6 \
  --candidate_count 18 \
  --chunk_size 6 \
  --device cuda \
  --output_npz results/autoresearch/568a_564a_risk_scenario_deck/smoke_scenario_deck.npz \
  --output_manifest results/autoresearch/568a_564a_risk_scenario_deck/smoke_manifest.json
```

Smoke manifest summary:

- scenario shape: `(6, 30, 5, 5)`
- bucket counts: `2` calm, `2` central, `2` stress
- latest-history slice: `[5792, 5822)`
- path mean IV range: `0.193325` to `0.221426`

## Decision

568a makes the current risk-manager prototype operationally usable: it can now generate a scenario deck artifact and manifest from the existing learned law.

This does not change the scientific status of the model. The underlying caveats remain:

- IV-only scope;
- non-calibrated selected frequencies;
- weak regime layer2;
- near-miss cointegration worst-cell ratio;
- poor level-frequency matching under stress selection.

But under the constrained risk-manager framing, the system is now both presentable and runnable.
