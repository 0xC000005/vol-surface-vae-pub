### Context

718a is the required post-experiment analysis after 717a. Since both innovation-score coordinates and observed-positive log-level support failed to produce a frozen tri-scope candidate, the loop paused model changes and audited the root cause of the repeated AAA/BBB OAS failures and the IV coverage regression.

### Findings

Data-object audit:

- AAA OAS has high daily no-change mass: train `0.435`, validation `0.474`.
- BBB OAS also has high no-change mass: train `0.265`, validation `0.383`.
- Validation OAS tails are calmer, not more extreme: AAA val/train raw q99 ratio `0.222`; BBB `0.417`.
- Train/validation raw-delta KS is moderate: AAA `0.058`, BBB `0.136`; normalized-increment KS is also moderate: AAA about `0.09`, BBB about `0.134`.
- The same OAS no-change structure exists under both `reference_based` and `observed_positive`; log-level support changes the coordinate but not the underlying sticky data object.

Model audit:

- The repeated anchor/joint blockers are still AAA/BBB OAS across 688/676 baseline, 714 marginal-CRPS, 716 innovation-score, and 717 observed-positive.
- Observed-positive support worsened excursions: 717 anchor generated AAA up to `6.135` versus GT max `0.99`, and BBB up to `6.988` versus GT max `3.03`.
- Innovation-score helped BBB in anchor-only (`0.194` KS) but did not solve the frozen tri-scope problem and badly regressed IV.
- IV failures are a separate long-horizon/regime undercoverage problem: 674/711/714/716/717 all miss coverage/regime/level fidelity, and 717 is worse than the incumbent on cov90 and h30 worst-cell coverage.

Artifacts:

- `experiments/backfill/block_ar/analyze_718a_oas_sticky_iv_coverage_attribution.py`
- `results/block_ar/718a_oas_sticky_iv_coverage_attribution/analysis.json`
- `results/block_ar/718a_oas_sticky_iv_coverage_attribution/analysis.md`

### Mechanism Read

OAS failure is not primarily train-validation shift and not simple positive-support mismatch. The more precise failure class is sticky/zero-inflated spread dynamics inside a continuous-only normalized-innovation law: the data contains a large no-change atom and state-dependent jump behavior, while the continuous flow smears that atom into small moves and occasional excessive spread excursions.

### Decision

The next decisive experiment should be a generic sticky or mixed discrete-continuous innovation coordinate for channels with empirical no-change atoms. This must be framed as a variable-type data coordinate and frozen across IV-only, anchor-only, and joint scopes, not as an OAS-specific special case or a scope-specific loss.
