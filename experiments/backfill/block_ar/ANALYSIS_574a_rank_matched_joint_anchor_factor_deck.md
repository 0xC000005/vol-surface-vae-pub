# 574a Rank-Matched Joint Anchor-Factor Deck

## Context

573a made IV plus anchor-factor stress decks operationally acceptable, but the factor overlay was only severity-aligned by broad calm/central/stress order. The user asked to improve anchor factors toward the same quality level as the IV scenarios.

The cleanest next improvement is pairing, not a new architecture:

- keep the accepted `510a/568a` IV stress deck;
- keep `537a` factor increments with deterministic factor-level reconstruction;
- match each selected factor overlay to the actual selected IV candidate severity quantile.

This makes the factor overlay scenario-by-scenario consistent with the IV stress selection policy.

## Implementation

Updated:

- `experiments/backfill/block_ar/generate_573a_joint_anchor_factor_deck.py`;
- `test_code/test_573a_joint_anchor_factor_deck.py`.

Added:

- `quantile_matched_indices`;
- `rank_matched_indices` unit coverage;
- factor selected severity quantiles in the `.npz` artifact;
- IV selected severity quantiles in the `.npz` artifact;
- manifest pairing diagnostics.

The generator now samples IV candidates directly, applies the same severity-stratified selection, records the selected IV candidate severity quantiles, and selects factor candidates at those same internal-panel severity quantiles.

## Generated Artifact

Command:

```bash
python experiments/backfill/block_ar/generate_573a_joint_anchor_factor_deck.py \
  --samples 48 \
  --iv_candidate_count 192 \
  --factor_candidate_count 192 \
  --chunk_size 8 \
  --device cuda \
  --seed 574 \
  --output_npz results/autoresearch/574a_rank_matched_joint_anchor_factor_deck/joint_anchor_factor_deck.npz \
  --output_manifest results/autoresearch/574a_rank_matched_joint_anchor_factor_deck/manifest.json
```

Artifact:

- `results/autoresearch/574a_rank_matched_joint_anchor_factor_deck/joint_anchor_factor_deck.npz`;
- `results/autoresearch/574a_rank_matched_joint_anchor_factor_deck/manifest.json`.

Shapes and finiteness:

- `iv_scenarios`: `(48, 30, 5, 5)`, finite rate `1.0`;
- `factor_scenarios`: `(48, 30, 26)`, finite rate `1.0`;
- `iv_selected_severity_quantiles`: `(48,)`, finite rate `1.0`;
- `factor_selected_severity_quantiles`: `(48,)`, finite rate `1.0`.

Severity buckets:

- `16` calm;
- `16` central;
- `16` stress.

Pairing diagnostics:

- factor pairing policy: `candidate_quantile_matched_to_selected_iv_severity`;
- mean absolute quantile error: `0.0`;
- max absolute quantile error: `0.0`;
- IV selected quantile range: `0.0` to `1.0`;
- factor selected quantile range: `0.0` to `1.0`.

## Verification

Focused tests:

- `pytest test_code/test_573a_joint_anchor_factor_deck.py -q` -> `7 passed`.

Final regression set:

- `pytest test_code/test_573a_joint_anchor_factor_deck.py test_code/test_572a_joint_panel_quality_audit.py test_code/test_568a_risk_scenario_deck_cli.py -q` -> expected verification target for commit.

## Decision

574a improves the anchor-factor overlay from broad bucket alignment to exact selected-candidate severity-quantile matching. This is closer to the IV deck's scenario construction quality because every factor overlay now corresponds to the same stress-selection quantile as its paired IV scenario.

The claim remains bounded:

- acceptable as a risk-manager stress scenario deck;
- not a calibrated joint probability law;
- factor levels remain deterministic reconstructions from generated increments;
- future publication-grade work should train a native joint increment-law model rather than relying on a policy-composed deck.
