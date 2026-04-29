Use the `autoresearch-head-loop` skill and the `research-log-tail-append` skill.

Continue the persistent autoresearch loop toward `11/11` on the common full 11-suite for a generalizable conditional scenario generator over general financial factors.

Honor the active restart phase in `state_11x11.json`. If the state says the old
`205-265` program is archived, do not resume `258b`, `265a`, or any other old-family follow-up.
Treat that tree as baselines and falsification history only.

Rules for this invocation:

1. Read:
   - `autoresearch-session/goal_11x11.json`
   - `autoresearch-session/state_11x11.json`
   - the tail of `RESEARCH_LOG.md`
2. Decide the single most principled next step:
   - post-experiment analysis
   - research ideation
   - paradigm shift
   - experiment
3. Keep executing full HEAD iterations in this same session:
   - Hypothesis
   - Execute
   - Analyze
   - Decide
4. After each iteration:
   - update `autoresearch-session/state_11x11.json`
   - append a concise result entry to the actual end of `RESEARCH_LOG.md` using `research-log-tail-append`
   - create one focused git commit for that iteration
5. If the pathology becomes unclear or the active line starts accumulating too many knobs, do not keep stacking local experiments:
   - switch to post-experiment analysis, research ideation, or paradigm shift as appropriate
6. Enforce the single-framework candidate gate:
   - a model is not a framework candidate unless the same frozen recipe is run on IV-only, anchor-only, and joint scopes
   - allowed differences are input/output dimensions, support transforms, input heads, decoder heads, generic support-aware mixed discrete-continuous heads selected by one data-derived rule, and deterministic channel/group balancing
   - forbidden differences are scope-specific losses, loss weights, backend, prior, sampler, calibration layer, or post-hoc glued decks
   - if the best scope-specific runs use different recipes, run framework-lock analysis/experiments before adding another knob
7. Enforce incumbent and non-regression discipline:
   - keep `734a/739a` as the deployable real-VIX tri-scope incumbent until a new recipe beats it on IV-only, anchor-only, and joint
   - keep `755a` as the IV-only research frontier until a better IV-only run beats it without losing hard IV realism gates
   - label every run as deployable_incumbent, iv_research_frontier, diagnostic_branch, or rejected_branch
   - every serious candidate must run IV-only, anchor-only, and joint; single-scope probes are diagnostic only and cannot be promoted
   - IV mean reversion is a hard non-regression gate; losing it makes the model not risk-manager deployable even if coverage, CRPS, interval score, likelihood, or old cointegration improves
   - preserve IV scenario realism, risk-state uncertainty allocation, pathwise jumps, cross-cell dependency, anchor realism, joint IV-factor co-movement, and conditional panel response
   - old path-prediction conditionality is deprecated when risk-state uncertainty allocation passes
   - old IV-EWMA cointegration is kept as a monitoring diagnostic
   - when present, use `iv_ewma_economic_link` as the IV/EWMA economic-link gate for promotion; it is not cointegration because it does not test residual stationarity
   - do not call co-movement "new cointegration" unless a residual-stationarity gate is specified
8. Continue until stopped only by:
   - the goal being reached
   - `autoresearch-session/STOP` existing
   - the configured hard iteration cap being reached
   - or session/runtime/tool limits making further work unreasonable

Do not stop for research/model blockers.
If a blocker appears, convert it into:
- post-experiment analysis
- research ideation
- or paradigm shift
and continue.

Current reset doctrine:
- the only core architectural bias that should be assumed by default is a narrow encoder-decoder bottleneck
- keep the generative core vanilla
- do not assume hard low-rank structure in the core spec
- do not assume bounded idio or EC side paths in the core spec
- if those return later, they must return only as explicit post-failure ablations

Current active research direction:
- use the real-VIX 14-anchor panel documented in `docs/research_protocols/733_real_vix_tri_scope_handoff.md`
- `vix` is observed Yahoo Finance `^VIX`; do not use or reintroduce the IV-derived `vix_proxy`
- treat `719a_zero_center_normalized_innovation_sticky_zero_readout_frozen` as the pre-real-VIX methodology/risk-manager baseline family, not as a current real-VIX champion checkpoint
- preserve the single generation mechanism: AR normalized-innovation flow, one stochastic source, one frozen tri-scope framework unless diagnostics justify a switch
- the next framework candidate must run one frozen recipe on `iv_only`, `anchor_only`, and `joint` with only allowed data-interface differences
- the next promotable candidate must compare directly against `734a/739a` and, for IV-side changes, `755a`
- do not promote a model that regresses IV mean reversion or risk-manager scenario realism
- known pathology: AAA/BBB OAS credit spreads are sticky and may require either a documented limitation or one generic support-aware mixed discrete-continuous low-activity-channel adapter
- before adding that adapter, run a sticky-channel audit: no-change mass, move-event rate, nonzero jump tails, and stress-state move frequency across anchor channels
- do not add credit-specific or scope-specific knobs unless the failure diagnosis and literature/first-principles argument justify the mechanism as generic

This invocation should leave the repository in a resumable state after every iteration.
