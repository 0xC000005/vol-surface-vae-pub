# 733 Real-VIX Tri-Scope Handoff

## Objective

The active objective is one coherent conditional scenario generator for:

- IV-only;
- observed-anchor-only;
- joint IV plus observed anchors.

Single model means one stochastic source and one generative core. It does not
require one checkpoint for all dimensionalities. Different input/output heads,
decoder heads, and support-aware coordinate transforms are allowed as data
adapters.

## Active Data

The anchor panel now contains 14 observed factor levels and 14 corresponding
increments:

- `spx`
- `usdcad`
- `usdjpy`
- `dxy`
- `copper`
- `wheat`
- `crude_oil`
- `us2y`
- `us10y`
- `aaa_oas`
- `bbb_oas`
- `nikkei`
- `gold`
- `vix`

`vix` is observed Yahoo Finance `^VIX`. The IV-derived `vix_proxy` is not part
of the active data or model path.

## Current Methodology

Continue the state-aware normalized-innovation conditional-law program:

`p(z_future | encoded history state, recent normalized innovations, local scale)`

Flow matching and AR rollout are implementation choices, not doctrine. Do not
switch backend, temporal factorization, or architecture until diagnostics show
that the current component is the bottleneck.

## Candidate Gate

A publishable framework candidate must freeze one recipe and evaluate it on all
three scopes:

- `iv_only`
- `anchor_only`
- `joint`

Allowed differences are only data-interface adaptations:

- input/output dimensionality;
- support or coordinate transform implied by variable type;
- input heads and decoder heads;
- generic support-aware mixed discrete-continuous heads selected by one
  data-derived no-change rule;
- deterministic channel/group balancing from the same formula.

Forbidden differences:

- scope-specific loss recipes or loss weights;
- scope-specific backend, sampler, or calibration layer;
- separate IV and anchor decks glued after sampling;
- IV-derived VIX proxy.

## Known Pathology

AAA/BBB OAS credit spreads are sticky: day-to-day increments are often exactly
or near zero. Continuous generators can over-move these channels even when other
factor realism is acceptable.

Do not add a credit-specific hack by default. If needed, the only acceptable
repair direction is a generic sparse/sticky low-activity-channel adapter that is
triggered by data statistics and can apply to any channel with similar behavior.
If this does not work cleanly, document sticky spreads as a limitation and keep
the main framework intact.

The adapter is allowed only if it models the correct support:
`P(move | state)` plus `p(move_size | move, state)`. It must preserve the same
shared stochastic source and generative core, and it must be selected by one
declared no-change statistic rather than by factor name.

## Next HEAD Iteration

Run the first real-VIX tri-scope framework-lock iteration:

1. Pick the 719a-style normalized-innovation recipe as the baseline recipe.
2. Train/evaluate the same frozen recipe on `iv_only`, `anchor_only`, and
   `joint` with the updated 14-anchor observed panel.
3. Report IV 11-suite, anchor panel realism/dependence/conditional response,
   joint panel realism/dependence/conditional response, and IV-factor
   co-movement.
4. Include a sticky-channel audit: no-change mass, move-event rate, nonzero
   jump tails, and stress-state move frequency for all anchor channels.
5. Classify failures before adding any new knob. If sticky low-activity
   channels are the only serious defect, the next minimal fix may be one
   generic support-aware mixed head, tested as a frozen tri-scope framework.
