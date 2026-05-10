# World Model HEAD119: Part 1 Quality Gate Assessment

Date: 2026-05-10

## Objective Family

`post_experiment_analysis` for the frozen `masked_multiview_invariance` Part 1 reference.

## Execution

- Ran the package-integrity checker.
- Read the saved HEAD070, HEAD082, HEAD083, HEAD084, and HEAD085 artifacts.
- Scored the literature-aligned Part 1 quality-gate layers without changing the pretraining objective.

## Verdict

- Quality gate passed: `False`.
- Promotion decision: `DO_NOT_PROMOTE`.
- Part 1 ready for Part B: `False`.
- Corruption-based embedding learning signal: `PASS` at smoke scale.

## Layer Results

| layer | status | evidence | decision |
| --- | --- | --- | --- |
| 1. Package Integrity | `PASS` | package_check ok=True; reports=18; guardrail_docs=5; artifacts=7 | Reference package is internally consistent. |
| 2. Representation Health | `PASS` | top10=0.841927 vs raw=0.373177; mrr=0.478177 vs raw=0.149323; effective_rank=14.501/14.594; variance_min=0.020293/0.020221; singular_top1_share=0.091/0.090 | Smoke-scale same-state masked-view embedding learning is real. |
| 3. Corruption Robustness | `PARTIAL` | mask-family probes below majority; view_a=0.179688/0.210938; view_b=0.203125/0.257812; default-family min_top10=0.826190 | Default structured masks look robust, but richer held-out mask families and seed stability are not yet validated. |
| 4. Baseline Superiority | `FAIL` | barlow_clean_last beats best raw baseline on 2/5 IV future targets; best raw baseline wins 3/5. | Broad baseline superiority is not established; the representation wins some path-shape targets but loses mean/terminal and max-step claims against raw features. |
| 5. Market-State Linear Probes | `FAIL` | regime probe accuracy=0.109375 vs majority=0.597656; factor-panel state probes and IV-shape state probes are not complete. | Current frozen probes do not certify market-state information. |
| 6. Temporal Utility Probes | `PARTIAL` | IV-surface future probes exist, but performance is mixed and factor-panel future targets, horizon sensitivity, and held-out time-split robustness are not complete. | Temporal utility is diagnostic-only, not promotion evidence yet. |
| 7. Scale And Stability | `FAIL` | reference checkpoint uses 384 train windows, 128 validation windows, 8 epochs, one seed, CPU smoke scale. | No full-data, multi-seed, or horizon-stability claim is supported. |

## Downstream Raw-Baseline Check

| target | Barlow MSE | best raw feature | best raw MSE | Barlow beats best raw |
| --- | ---: | --- | ---: | --- |
| future_mean_delta | 0.011635 | `raw_surface_last` | 0.006484 | `False` |
| future_range | 0.047185 | `raw_surface_flat` | 0.050972 | `True` |
| future_terminal_delta | 0.031830 | `raw_surface_last` | 0.019208 | `False` |
| future_max_abs_step | 0.039526 | `raw_surface_flat` | 0.035980 | `False` |
| future_drawdown | 0.042269 | `raw_surface_flat` | 0.044180 | `True` |

## Interpretation

The corruption-based masked-multiview Barlow representation works as a smoke-scale embedding-learning signal, but the current Part 1 package is not good enough to promote to a certified joint market-state representation for Part B.

The current result is useful for diagnostics and for designing the next
frozen probes. It should not be used to start Part B as if Part 1 were
already certified.

## Next Required Evidence

- frozen market-state probes that beat raw/simple baselines
- factor-panel future target probes before joint-factor claims
- simple PCA, persistence, and rolling-window baseline comparisons
- held-out mask-family and mask-seed robustness
- larger-scale and multi-seed stability without adding objective knobs
