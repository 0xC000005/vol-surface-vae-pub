# 572a Joint Panel Scenario Quality Audit

## Context

571 proved that a 51-channel IV-plus-factor model can train and sample at h30 and h90. It did not prove scenario quality. This audit checks whether the generated joint scenarios are acceptable for risk-manager use.

## Tests Added

Added a reusable joint-panel quality audit:

- script: `experiments/backfill/block_ar/audit_572a_joint_panel_quality.py`;
- tests: `test_code/test_572a_joint_panel_quality_audit.py`.

The audit checks:

- finite generated panel rate;
- factor level and factor return KS statistics;
- factor level move-scale and return-scale q99 ratios;
- IV-factor end-to-end co-movement correlation error;
- optional integration with the official IV 11-suite result.

It also adds a deterministic factor-level reconstruction option:

- generate factor returns/diffs as stochastic outputs;
- derive factor levels from the last observed level plus cumulative generated returns/diffs;
- keep IV channels unchanged.

This is a data-framing/accounting fix, not a learned calibration knob.

## Raw 571 h30 Result

Official IV suite:

- artifact: `results/autoresearch/571c_joint_panel_h30_full11/full11.json`;
- score: `3/11`;
- failed suites: coverage, conditionality, time-series, regime coverage, distributional fidelity, cross-cell correlation, mean reversion, pathwise jump realism;
- conditionality MAE reduction: `1.8%`;
- cross-cell correlation ratio: `0.152`;
- pathwise max-jump KS: `0.579`.

Joint-panel audit:

- artifact: `results/autoresearch/572a_joint_panel_h30_quality/quality.json`;
- risk-manager acceptable: `false`;
- failed checks: factor level marginals, factor level move scale, IV conditionality, IV cross-cell structure, IV pathwise realism;
- factor return KS median/worst: `0.144` / `0.274`;
- factor level KS median/worst: `0.577` / `0.807`;
- factor level delta q99 worst-fold error: `19.058`.

Conclusion: raw 571 h30 is not acceptable.

## Raw 571 h90 Result

Joint-panel audit:

- artifact: `results/autoresearch/572b_joint_panel_h90_quality/quality.json`;
- risk-manager acceptable: `false`;
- failed checks: factor level marginals, factor level move scale;
- factor return KS median/worst: `0.106` / `0.158`;
- factor level KS median/worst: `0.598` / `0.877`;
- factor level delta q99 worst-fold error: `18.239`.

Conclusion: raw 571 h90 is not acceptable as a joint factor scenario generator.

## Reconstruction Fix

After deriving factor levels from generated return/diff channels:

h30 reconstructed audit:

- artifact: `results/autoresearch/572c_joint_panel_h30_reconstructed_quality/quality.json`;
- risk-manager acceptable: `false`;
- failed checks: factor level marginals, IV conditionality, IV cross-cell structure, IV pathwise realism;
- factor level KS median/worst: `0.349` / `0.486`;
- factor return KS median/worst: `0.144` / `0.269`;
- factor level delta q99 worst-fold error: `2.137`;
- factor return q99 worst-fold error: `1.463`.

h90 reconstructed audit:

- artifact: `results/autoresearch/572d_joint_panel_h90_reconstructed_quality/quality.json`;
- risk-manager acceptable: `false`;
- failed checks: factor level marginals;
- factor level KS median/worst: `0.368` / `0.551`;
- factor return KS median/worst: `0.106` / `0.156`;
- factor level delta q99 worst-fold error: `1.505`;
- factor return q99 worst-fold error: `1.626`.

Conclusion: reconstruction fixes the incoherent factor-level move scale but not the level-occupancy mismatch. For h30, the IV side remains the dominant blocker.

## Best Existing Joint Candidate

The older full-sized `537a` checkpoint was also audited with factor-level reconstruction:

- checkpoint: `models/backfill/537a_panel_daily_cholesky_transition_s537/best_model.pt`;
- audit artifact: `results/autoresearch/572e_537a_joint_panel_reconstructed_quality/quality.json`;
- risk-manager acceptable: `false`;
- failed checks: IV conditionality, IV pathwise realism.

What passes:

- finite panel;
- factor level marginals;
- factor return marginals;
- factor level move scale;
- factor return scale;
- IV-factor co-movement;
- IV surface validity;
- IV lower coverage;
- IV cross-cell structure.

Key factor metrics:

- factor level KS median/worst: `0.183` / `0.490`;
- factor return KS median/worst: `0.082` / `0.156`;
- factor level delta q99 worst-fold error: `2.987`;
- factor return q99 worst-fold error: `2.959`;
- IV-factor median abs correlation error: `0.113`.

Remaining IV issues:

- conditionality MAE reduction: `3.826%`, below the `5%` gate;
- h1 and h7 conditionality are useful, but h14 and h30 are weak;
- aggregate pathwise max-jump passes under the relaxed `0.50` gate, but per-cell q99 jump scale remains uneven.

## Verdict

The raw 571 joint model is not risk-manager acceptable.

The strongest current joint candidate is not 571. It is `537a` plus deterministic factor-level reconstruction. That candidate is reviewable as a joint stress-scenario prototype, but it is still not acceptable as a full conditional joint scenario generator because long-horizon IV conditionality and per-cell IV jump realism remain weak.

## Decision

Do not deploy the raw 571 joint checkpoints.

If the objective is a risk-manager review deck, the least-bad route is:

- use `537a` with factor levels reconstructed from generated returns/diffs for joint factor paths;
- label it as a stress-review prototype, not a validated conditional law;
- keep the current `510a/568a` IV deck as the stronger IV-only production candidate;
- next iteration should either combine the stronger IV deck with reconstructed factor overlays, or train a native joint model where factor levels are not independent generated targets.
