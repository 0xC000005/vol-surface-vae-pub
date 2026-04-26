# TimePFN-Style Synthetic Prior Pretraining Pilot

## Objective

Test whether the TimePFN/Prior-data Fitted Network idea helps the current IV-surface scenario generator by adding broad synthetic-prior pretraining before real SPX adaptation.

The pilot should answer one question: does synthetic exposure to many plausible calm, stressed, and anomalous IV-surface regimes improve stress support and regime inclusion without damaging the properties that make `510a` risk-manager presentable?

## Product Framing

The target product is a conditional stress scenario generator, not a fully calibrated conditional probability law.

This means the pilot should prioritize:

- conditionality on the observed IV history;
- lower stress-regime inclusion;
- coherent IV-surface paths;
- credible cross-cell dependence, cointegration/correlation, and mean reversion;
- no explosions or malformed surfaces;
- enough scenario diversity.

The pilot should not optimize primarily for exact marginal level-KS matching or high-side overcoverage caps. Those remain diagnostics, not primary blockers, under the risk-manager framing.

## Recommended Approach

Use the existing empirical-normal-score causal-memory AR flow backbone from the `340c/392a/510a` family and add a synthetic-prior pretraining stage.

This isolates the TimePFN-style hypothesis while keeping the architecture, sampler, and evaluation path comparable to the current frontier. A full new transformer/PFN architecture is intentionally out of scope for the first falsifier because it would change too many variables at once.

## Components

### Synthetic IV Sequence Generator

Create a reusable generator that emits synthetic daily IV-surface panels with the same grid shape and approximate normalized range as the local SPX data.

Each generated sequence should combine:

- latent level factor with mean reversion;
- slope and skew factors with correlated dynamics;
- term-structure and moneyness loadings;
- calm/stress regime switching;
- occasional jumps and volatility-of-volatility bursts;
- idiosyncratic cell noise that is bounded and smoothed across the surface.

The generator should produce `(history, future)` windows compatible with the existing `history_len=30`, `future_len=30`, and `5x5` grid convention.

### Synthetic Pretraining

Train the existing empirical-normal-score AR flow on synthetic windows first.

The pretraining objective should remain the model's standard flow-matching objective. Do not add new calibration or policy losses in the first pilot. The goal is to test whether the prior-data distribution improves learned support, not whether extra evaluator-aware losses can tune metrics.

### Real-Data Adaptation

Fine-tune the pretrained model on the same real-data adaptation window used by the current frontier family.

The first real-data adaptation should be conservative:

- initialize from the synthetic-pretrained checkpoint;
- use the same empirical quantile machinery as the current backbone;
- use the same recent-window SPX adaptation framing as the `392a/510a` line;
- keep the patch-energy or rollout-energy step only if needed to match the current frontier's final adaptation recipe.

### Evaluation

Evaluate using the existing full 11-suite and the relaxed risk-manager readout.

The primary comparison set is:

- `510a` as the current most shippable risk-manager prototype;
- `555a` as the clean local-factor-conditioned tie;
- `392a` as the structural anchor.

## Success Criteria

The pilot is useful only if it beats `510a` on risk-manager deployability, not just on one marginal metric.

Minimum success:

- preserves lower-only coverage pass;
- preserves conditionality pass;
- preserves scenario authenticity, including path realism and dependence structure;
- improves lower stress-regime inclusion versus `510a`, especially the worst regime-cell margin;
- does not introduce explosions or malformed surfaces.

Strong success:

- reaches stress-readiness `4/4` under the relaxed risk-manager audit;
- keeps original full-suite score at least `8/11`;
- improves regime lower-inclusion without worsening conditionality or cointegration margins.

Failure:

- worse than `510a` on conditionality or scenario authenticity;
- improves level or coverage metrics only by broadening paths into unrealistic stress noise;
- cannot improve regime lower-inclusion after real-data adaptation.

## Implementation Boundaries

Keep this pilot small and falsifiable.

Do not implement a full TimePFN transformer yet. Do not add external data. Do not add post-hoc per-cell evaluator tables. Do not introduce a risk-policy overlay into the model training loop. Those may be separate future branches, but this pilot tests synthetic prior pretraining only.

## Testing Plan

Add focused tests for:

- synthetic generator output shape and bounded values;
- deterministic generation under fixed seed;
- presence of calm and stress regimes in synthetic batches;
- compatibility of synthetic windows with the existing empirical-score AR flow training path;
- no-lookahead separation between synthetic pretraining and real validation evaluation.

Then run:

- a smoke synthetic pretraining job;
- a short real-data adaptation job;
- the full 11-suite for the resulting checkpoint;
- the risk-manager readiness audit against `510a`.

## Expected Interpretation

If this works, the bottleneck was at least partly sparse support exposure: the local SPX history did not provide enough stress transitions for the current AR flow to learn robust regime inclusion.

If this fails while preserving implementation correctness, it is evidence that a simple synthetic prior is not enough. The next principled move would then be either a richer multi-underlying data program or a true PFN/transformer architecture trained on a much broader synthetic prior.
