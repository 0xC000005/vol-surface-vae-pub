# 573a Joint Anchor-Factor Stress Deck

## Context

572 established two facts:

- raw `571` joint checkpoints are not risk-manager acceptable;
- the older full `537a` panel model becomes a usable anchor-factor prototype if factor levels are reconstructed deterministically from generated returns/diffs.

The next principled step was not another architecture tweak. It was to package the currently strongest components under a transparent risk-manager stress-deck contract:

- IV scenarios from the accepted `510a/564a/568a` IV stress system;
- anchor-factor overlays from `537a` factor increments with deterministic factor-level reconstruction.

## Implementation

Added:

- `experiments/backfill/block_ar/generate_573a_joint_anchor_factor_deck.py`;
- `test_code/test_573a_joint_anchor_factor_deck.py`.

The generator:

- loads the `510a` IV base law through the existing `568a` stress-deck path;
- applies the same severity-stratified calm/central/stress selection policy as `564a`;
- loads the `537a` 51-channel panel model for factor candidates;
- reconstructs factor levels from generated factor returns/diffs;
- selects factor overlays by the panel model's internal IV severity ranks, so factor overlays are calm/central/stress aligned;
- writes one `.npz` deck containing IV history, IV scenarios, factor history, factor scenarios, factor columns, bucket labels, selected factor indices, and diagnostics;
- writes a manifest with separate IV and joint-anchor-factor risk contracts.

This is intentionally not presented as a calibrated probability law.

## Verification

Unit tests:

- command: `pytest test_code/test_573a_joint_anchor_factor_deck.py -q`;
- result: `3 passed`.

Generated deck:

```bash
python experiments/backfill/block_ar/generate_573a_joint_anchor_factor_deck.py \
  --samples 48 \
  --iv_candidate_count 192 \
  --factor_candidate_count 192 \
  --chunk_size 8 \
  --device cuda \
  --seed 573 \
  --output_npz results/autoresearch/573a_joint_anchor_factor_deck/joint_anchor_factor_deck.npz \
  --output_manifest results/autoresearch/573a_joint_anchor_factor_deck/manifest.json
```

Artifact sanity check:

- `iv_history`: `(30, 5, 5)`, finite rate `1.0`;
- `iv_scenarios`: `(48, 30, 5, 5)`, finite rate `1.0`;
- `factor_history`: `(30, 26)`, finite rate `1.0`;
- `factor_scenarios`: `(48, 30, 26)`, finite rate `1.0`;
- bucket counts: `16` calm, `16` central, `16` stress;
- IV range: `0.010207` to `0.907568`;
- factor range: `-0.25` to `33798.507812`.

## Anchor Factor List

The generated factor deck contains 26 anchor-factor channels:

- levels: `spx`, `usdcad`, `usdjpy`, `dxy`, `copper`, `wheat`, `crude_oil`, `us2y`, `us10y`, `aaa_oas`, `bbb_oas`, `nikkei`, `gold`;
- increments: `spx_logret`, `usdcad_logret`, `usdjpy_logret`, `dxy_logret`, `copper_logret`, `wheat_logret`, `crude_oil_logret`, `us2y_diff`, `us10y_diff`, `aaa_oas_diff`, `bbb_oas_diff`, `nikkei_logret`, `gold_logret`.

## Acceptance Contract

The manifest reports:

- `iv_risk_contract.risk_manager_acceptable = true`;
- `joint_anchor_factor_contract.risk_manager_acceptable = true`.

The IV contract cites:

- `experiments/backfill/block_ar/REPORT_567a_564a_risk_manager_deployability_package.md`.

The factor contract cites:

- `results/autoresearch/572e_537a_joint_panel_reconstructed_quality/quality.json`.

The acceptance is valid only under the stress-deck framing:

- selected scenario frequencies are not probabilities;
- level-frequency KS remains a warning, not a stress blocker;
- regime layer2 and cointegration caveats remain disclosed;
- factor overlays are severity-aligned anchor scenarios, not a fully calibrated cross-asset joint probability law.

## Decision

For the user's stated risk-manager objective, `573a` is the current acceptable joint stress scenario product:

- IV-only scenario deck: acceptable under the existing `568a`/`567a` stress contract;
- IV plus anchor-factor list: acceptable under the new `573a` stress overlay contract.

This does not supersede the scientific model-quality frontier:

- `510a` remains the best learned IV law;
- `537a` plus reconstruction remains the best factor-overlay prototype;
- a future publishable joint conditional law should train factor increments as first-class targets rather than independently generating duplicate factor levels.
