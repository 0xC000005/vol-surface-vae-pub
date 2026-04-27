# 626a Native Joint Learned-Law Synthesis

## Context

610-625 tested the clean native joint path the user requested: one model, one state panel, one stochastic rollout, with IV-only available as a state-scope special case and joint38 available by changing the input panel/preprocessing.

This branch deliberately avoided separate IV/factor decks. It tested whether a learned joint conditional law could become risk-manager deployable by changing only principled architecture/objective/data-coordinate choices.

## Evidence Summary

| Run | Core Idea | Score | Useful Result | Failure |
| --- | --- | ---: | --- | --- |
| 610a | raw joint38 AR flow matching | `5/11` | first clean native joint learned law; surface, block-AR, cointegration, cross-cell, pathwise pass | undercoverage, weak conditionality, level/regime allocation |
| 612a | conditional source scale | `6/11` | conditionality, time-series, daily KS, level KS improved | source-scale collapsed to lower clamp; not clean uncertainty learning |
| 614a | Gaussian transition likelihood | `4/11` | exact likelihood improved persistent undercoverage | too rigid; level KS and mean reversion failed |
| 617a | Student-t likelihood temperature | `5/11` | best likelihood tradeoff; coverage and undercoverage much better | level occupancy, mean reversion, tails, regime still failed |
| 620a | recursive mean-rollout loss | `3/11` | falsified deterministic multi-step mean placement | damaged conditionality and surface; no level/MR fix |
| 622a | encoded log/diff state coordinate | `4/11` | mean reversion restored cleanly | high-biased, under-dispersed, cointegration/local KS broke |
| 623a | encoded temperature `1.5` | `2/11` | proved width can be inflated | broke surface/dependence/tails; path-location bias remained |
| 625a | exact RealNVP transition likelihood | `3/11` | average coverage and daily-change KS improved | cross-cell dependence, h1 MR, level/regime allocation failed |

## Mechanism

The branch has a stable causal map:

1. AR/state-feedback geometry is necessary. It is the only route that repeatedly preserves surface validity, block smoothness, cointegration, cross-cell structure, and local daily-change shape.
2. Local one-step objectives are not sufficient. Flow matching, Gaussian/Student-t NLL, and RealNVP NLL can each improve some local increment or coverage statistics, but free-running multi-step paths still miss level occupancy and regime/cell allocation.
3. Scalar or deterministic fixes are not sufficient. Temperature, conditional source scale, and mean-rollout loss each improve one symptom while breaking another structural gate.
4. State-coordinate changes are useful but not decisive. Encoded log/diff coordinates restore mean reversion, but introduce strong path-location bias.
5. The remaining failure is not a single architecture bug. It is weakly identified conditional allocation of probability mass over 30-day multi-cell paths under one realized future per history.

## Risk-Manager Read

For a risk manager, the most dangerous failures are not level-KS aesthetics. The dangerous failures are:

- lower-side sparse-window undercoverage;
- regime/cell coverage gaps;
- broken cross-asset/cross-cell dependence;
- unrealistic jump placement;
- path medians that sit on the wrong side of realized futures for many windows.

No current native learned joint law clears all of those. 610a is the cleanest native joint learned law structurally. 612a has the best broad-frame count but has an unclean source-scale collapse. 625a is a useful likelihood baseline but fails joint dependence too severely for a joint risk product.

## Decision

Do not keep adding learned-law knobs locally. The next principled step is not another neural module. Because the user objective now explicitly includes 25+13 joint scenarios, the immediate gap is that the official bridge still scores only IV behavior.

Before choosing a risk-manager-deployable base or policy layer, run a generic joint-panel scenario audit:

- anchor-factor marginal change realism;
- anchor-factor level/change autocorrelation;
- factor-factor correlation;
- IV-factor cross-correlation;
- decoded factor range/finite sanity;
- scenario-level dependence preservation.

This should evaluate at least the clean native joint candidates:

- 610a: cleanest native AR joint law;
- 612a: best count but unclean source-scale collapse;
- 625a: flexible likelihood baseline.

If none has acceptable anchor-factor/co-movement realism, the next path is data/preprocessing and joint-panel evaluation, not more IV-only full-suite optimization. If one is acceptable, then the risk-manager-deployable product should be framed around that base law with base metrics reported separately from any disclosed scenario-set/risk-policy calibration.

