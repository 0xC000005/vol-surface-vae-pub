# 608a Support Path Fallback Fraction Audit

## Context

608a is the single controlled follow-up to 607a. It keeps the same frozen 340c/510a
backbone, the same train-support trigger, and the same q80 stressed historical
increment pool, changing only the fallback fraction from 0.50 to 0.75.

This tests whether the 607a path-location signal was under-applied, or whether the
fallback mechanism is already capped by a conditionality/authenticity trade-off.

## Result

- Full 11-suite score: 4/11.
- Failed suites: coverage, conditionality, time_series, regime_coverage,
  distributional_fidelity, mean_reversion, pathwise_jump_realism.
- Fallback trigger: 127/441 validation windows at train support q95 threshold 8.244.
- 90% coverage: 80.2% overall; h1/h7/h14/h30 = 86.5% / 82.0% / 80.2% / 76.3%.
- Conditional MAE reduction: 4.98%, just below the risk-readiness cutoff.
- Turbulent/calm width ratio: 1.171, still directionally correct.
- Regime layer2: 0/8, worse than 607a's 1/8.
- Distributional fidelity: daily-change KS 25/25, level KS 3/25, median-bias cells 16/25.
- Pathwise max-jump KS: 0.430 passes the relaxed aggregate gate, but per-cell q99 passes only 14/25.
- Persistent severe undercoverage: 811/11025 cells = 7.4%.
- Mean reversion: aggregate/core behavior remains directionally right, but h30 aggregate ratio is 0.689, just below the 0.70 gate.

Risk-readiness audit ranked 608a below 607a because the stronger fallback made
conditionality borderline/failing while not clearing coverage or regime inclusion:

- 510a base broad: stress score 1/4, worst lower-only cell 0.478, worst regime cell 0.213.
- 607a: stress score 1/4, worst lower-only cell 0.644, worst regime cell 0.472, conditionality passes the risk cutoff.
- 608a: stress score 0/4, worst lower-only cell 0.646, worst regime cell 0.472, conditionality falls to 4.98%.

## Mechanism Read

The support-aware path-location fallback is pointing at a real pathology: sparse
out-of-support validation windows need larger and differently located paths than
the frozen learned law usually emits. However, the current fallback is a blunt
replacement rule. Increasing the fraction mostly swaps conditional model behavior
for stressed historical increments. That marginally lifts some coverage and median
bias counts, but it does not solve regime-cell inclusion and it weakens the
conditionality/authenticity argument.

The result falsifies broad fallback-fraction tuning as the next path. The useful
lesson is not "add more fallback"; it is that path location must be learned or
conditioned in a support-aware way rather than patched by a larger replacement deck.

## Decision

Close the aggressive path-location fallback branch as non-deployable. Keep 607a as
evidence that support-aware path location is the right missing axis, but do not
continue sweeping fallback fractions or adding more replacement knobs.

If research resumes, the next principled move is either:

- a learned conditional path-location mechanism that keeps one unified generative
  law while responding to support distance, or
- a documented data/signal requirement if the validation sparse-regime support is
  not learnable from the available training histories.

The in-session loop stops here because the user explicitly requested stopping
after this active iteration.

## Artifacts

- `results/autoresearch/608a_support_path_fallback_frac075/full11.json`
- `results/autoresearch/608a_support_path_fallback_frac075/full11.md`
- `results/autoresearch/608a_support_path_fallback_frac075/risk_readiness.json`
- `results/autoresearch/608a_support_path_fallback_frac075/risk_readiness.md`
