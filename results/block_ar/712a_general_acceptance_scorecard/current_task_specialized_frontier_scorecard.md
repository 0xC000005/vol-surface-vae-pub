# 712a General Conditional Scenario Acceptance Scorecard

- overall pass: `False`

| Gate | Pass | Notes |
| --- | --- | --- |
| `iv` | `False` | effective_failed=['coverage', 'regime_coverage', 'distributional_fidelity'] |
| `anchor` | `False` | failed_checks=['factor_delta_ks', 'conditional_panel'] |
| `joint` | `False` | failed_checks=['factor_delta_ks', 'conditional_panel'] |
| `framework` | `False` | disallowed_diffs=['generated_coordinate', 'loss_weights', 'scalar_loss_terms', 'training_protocol'], missing_scopes=[] |
