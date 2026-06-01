# Response-Aware Support Weighting Intake

## Candidate Name

Short name: `response_aware_support_weighting_next`

Workflow lane: `candidate`

Iteration type: `research_ideation -> experiment`

## Local Bottleneck

Because the 933a fixed-start stress test shows nonzero narrative
conditionality, but the raw fan response can still look too similar in generic
factor panels, the current pipeline fails when semantically plausible supports
do not produce enough risk-manager-visible response in the channels the
narrative is actually about. This candidate should help because it learns or
calibrates support weights from generator-response evidence, not merely because
it adds a more complex text embedding or ranker.

Required mechanism:

```text
same approved start
+ professional narrative
+ candidate support pool
-> response-aware support weights
-> component-preserving frozen SNI rollout
-> stronger narrative-relevant factor and portfolio-tail separation
   while preserving CRPS, energy, coverage, provenance, and direction checks
```

## Method Story

The risk-manager narrative remains the primary input. Grounded market
implications remain a sidecar for direction checks and warnings. The starting
level is fixed before support weighting. The candidate does not let the LLM
invent a future distribution and does not replace the frozen SNI generator.

The method scores candidate support components or support sets by how well
their generated response matches the risk channel implied by the narrative. For
example, commodity-inflation narratives should be evaluated in crude, rates,
equity, and volatility channels; dollar-liquidity narratives in dollar, credit,
volatility, and equity channels; safe-haven narratives in gold, rates,
volatility, and equity channels. The output is still an auditable weighted
support mixture over historical prefixes or prototypes.

## Related Work Basis

- Learning-to-rank supports listwise support scoring within a query. The
  transfer is the query-local support list; the label source is not clicks, but
  historical generator-response quality.
- Retrieval-augmented generation supports explicit memory and provenance. The
  transfer is auditable support; the numerical generator remains the frozen SNI
  model rather than an LLM.
- Ensemble postprocessing and copula-style scenario methods support preserving
  dependence structures while changing weights or calibration. The transfer is
  component-preserving path distributions; the financial dependence source is
  the SNI rollout over selected recent prefixes.
- CLIP-style contrastive alignment supports using language to select compatible
  latent objects. The transfer is narrative/support compatibility; promotion
  still requires scenario-level backtests rather than embedding similarity.

## Novelty Claim

This is not just semantic search over history. The local contribution is a
response-aware support prior: the narrative and fixed start select an auditable
support distribution, and support weights are trained or calibrated by the
future scenario behavior induced through the frozen financial generator.

## Elegance Check

- It keeps the existing historical support prior and frozen SNI rollout.
- It adds one learned/calibrated weighting layer over candidate supports.
- It avoids hidden start selection and direct LLM-generated scenarios.
- It replaces generic similarity-only weighting with a response-aware objective.
- It is interpretable if it fails: either the response labels are noisy, the
  candidate support pool is insufficient, or the frozen generator does not
  amplify the selected support differences enough.

New knob:

- response objective family: narrative-relevant factor response,
  portfolio-tail response, or a calibrated blend.

Out of scope for this candidate:

- direct text-to-scenario generation;
- support-free text latent conditioning;
- decoder fine-tuning;
- hand-tuned story-specific trading rules.

## Training And Inference Contract

Training data: historical windows with professional or cached narratives,
structured sidecar fields, fixed starts, candidate support pools, frozen-SNI
rollouts, and realized next-30-day paths.

Inference inputs: professional narrative, grounding sidecar, approved joint39
start, candidate support pool, and precomputed support features available
before the future is observed.

Output object: a weighted support mixture with component weights, support
provenance, response diagnostics, direction checks, and warning fields.

Leakage guard: generator-response labels may use realized futures only for
historical training/evaluation labels. Inference must not access future paths.

No hidden future information: required.

No hidden model-chosen start: required.

## Baselines

Required baselines:

- current component-preserving hard-direction start-aware support mixture;
- hard-direction narrative-first support mixture;
- start-only null;
- incumbent simple mixture floor from the 878a representative evaluation;
- previous portfolio-response quality-guard candidate if comparable artifacts
  are available.

## Backtest And Conditionality Gate

Promotion requires both scenario quality and narrative conditionality:

- held-out CRPS, energy, and 80% coverage competitive with the incumbent
  mixture floor;
- same-start, multi-narrative support overlap below the start-only null;
- relevant-factor terminal KS and path energy above repeat/bootstrap noise;
- portfolio VaR/ES or expected-shortfall-style tail separation above
  start-only and repeat controls;
- qualitative raw-level fans, contrast panels, and portfolio-tail plots showing
  the response in narrative-relevant channels;
- provenance and direction checks remain inspectable for every run.

## Kill Condition

Kill or keep diagnostic if:

- support weights change but factor/path/portfolio conditionality does not
  improve versus the current component-preserving baseline;
- CRPS and energy both regress without a measured trust or conditionality gain;
- gains appear only in a single hand-picked factor or exposure book;
- the method relies on future information, hidden start selection, or
  story-specific hand rules;
- the method cannot be explained as one coherent support-weighting mechanism.

## Independent Verification Trigger

Use independent verifier before:

- changing a demo or paper default;
- claiming the conditionality problem is solved;
- claiming production readiness;
- promoting a paper-facing result;
- scaling expensive labeling or rollout jobs based on this method.

Verifier artifact path:
`docs/research_protocols/nl_prefix_latent_verifier_reports/response_aware_support_weighting_next.md`

## Decision

Decision: `run_testflight`

Decision rationale: the current evidence shows conditionality exists, but the
product needs stronger narrative-relevant response. Response-aware support
weighting is the smallest coherent next method because it improves the support
distribution that already drives the frozen generator, rather than adding a new
text model or replacing the scenario backbone.

## Initial TestFlight Evidence

Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_response_aware_support_weighting_934a/response_aware_support_weighting_report.json`

Implementation:
`experiments/backfill/block_ar/nl_response_aware_support_weighting_testflight.py`

Status: `candidate_testflight_promising_not_promoted`

The 934a TestFlight reweights cached 932a component-preserving rollout samples
using the generator response in the grounded narrative channels. It makes no
OpenAI calls and no new generator calls. The operational contract is bounded:
same accepted start, same support components, same frozen SNI rollout samples,
new response-aware support weights.

Main alpha `0.75` result:

- current component baseline: relevant-factor KS `0.1775`, path energy
  `0.0431`, portfolio KS `0.1179`, VaR95 range `10.339`;
- response-aware candidate: relevant-factor KS `0.2127`, path energy
  `0.0489`, portfolio KS `0.1462`, VaR95 range `10.658`;
- start-only null: relevant-factor KS `0.0000`, path energy `0.0000`,
  portfolio KS `0.0000`, VaR95 range `0.000`.

Bounded sensitivity:

- alpha `0.35`: relevant-factor KS `0.2042`, path energy `0.0480`,
  portfolio KS `0.1448`, VaR95 range `8.771`;
- alpha `1.25`: relevant-factor KS `0.2205`, path energy `0.0528`,
  portfolio KS `0.1502`, VaR95 range `10.924`.

Interpretation: response-aware weighting strengthens the fixed-start
conditionality diagnostics and keeps the start-only null flat. It is not
promoted yet because this is an offline cached-rollout TestFlight, not a
held-out scenario-quality backtest and not yet an operational pre-rollout
support scorer.

Next evidence required:

1. train or calibrate a pre-rollout support scorer that approximates the 934a
   response-aware weights from support/narrative/start features available
   before final generation;
2. run held-out CRPS, energy, coverage, relevant-factor, and portfolio-tail
   comparisons against the current component-preserving baseline;
3. run independent verification before any paper/demo default promotion.

## Operational Pre-Rollout Probe

Artifacts:

- rollout root:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_932a_s384`
- conditionality stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_934b_response_book_guard/conditionality_stress_test.json`

Implementation:

- `narrative_book_quality_guard_926b` in
  `experiments/backfill/block_ar/nl_prefix_latent_analogue_mixture_prior.py`
- fixed-start policy `narrative_book_response_guard_gap30` in
  `experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py`

Status: `candidate_rejected_as_default`

This pre-rollout candidate blends train-only per-book portfolio-response priors
according to the grounded narrative risk channels, then selects support
mixtures before frozen SNI rollout. It is a useful diagnostic because it tests
whether the response-aware idea can be approximated without looking at
generated rollout responses for the current narrative.

Result versus the current component-preserving baseline:

- current component baseline: `6/6` conditionality gates,
  relevant-factor KS `0.1775`, path energy `0.0431`, portfolio KS `0.1179`,
  VaR95 range `10.339`;
- pre-rollout narrative-book guard: `3/6` conditionality gates,
  relevant-factor KS `0.0985`, path energy `0.0085`, portfolio KS `0.0536`,
  VaR95 range `1.479`;
- start-only null: `1/6` conditionality gates and zero separation.

Interpretation: the operational pre-rollout book guard is not the promotion
path. It often broadens or marginalizes the support pool too much, which
smooths away the narrative-specific response. One defensive-risk-off run also
failed the final direction check. This means "choose a risk book, then use
train-only support reliability priors" is too blunt for the product
conditionality goal.

## Deployable Feature-Sufficiency Refresh

Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_feature_sufficiency_936a_refresh/portfolio_response_feature_sufficiency.json`

Implementation:
`experiments/backfill/block_ar/nl_portfolio_response_feature_sufficiency.py`

Status: `deployable_feature_signal_found`

The 936a refresh tests whether support-level reliability and response features
available before final scenario generation contain enough signal to justify a
deployable response-aware support scorer. The strongest deployable scorer is
`support_reliability_prior`. It beats the rank floor on the test split:
pairwise accuracy `0.5717` versus `0.5253`, and mean selection regret
`0.0540` versus `0.0804`.

Interpretation: this is evidence that response-aware support scoring is not
purely a post-hoc explanation. There is a usable pre-rollout support signal.
However, this is still a label-surface diagnostic; it must pass fixed-start
rollout and conditionality stress gates before it can become a workflow
default.

## Direction-Safe Portfolio Quality-Guard Gate

Artifacts:

- rollout root:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_936d_portfolio_quality_guard_direction_safe_s64`
- conditionality stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_936d_portfolio_quality_guard_direction_safe_s64/conditionality_stress_test.json`

Implementation:

- fixed-start policy `portfolio_quality_guard_gap30` in
  `experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py`
- direction-safe fallback for `portfolio_quality_guard_924e` and
  `narrative_book_quality_guard_926b` in
  `experiments/backfill/block_ar/nl_prefix_latent_analogue_mixture_prior.py`

Status: `candidate_promising_not_promoted`

The first 936b rollout found that the generic portfolio quality guard could
weaken conditionality and produced one final mixed-prefix direction rejection.
The direction-safe 936d update now falls back to the base direction-checked
support prior when a response-aware mixture fails the final direction audit.
With that guard in place, the 64-sample fixed-start gate reports:

- incumbent current component policy: `6/6` direction passes, relevant-factor
  KS `0.1805`, path energy `0.0237`, portfolio KS `0.1302`, VaR95 range
  `4.3177`;
- portfolio quality support guard: `6/6` direction passes, relevant-factor KS
  `0.1831`, path energy `0.0247`, portfolio KS `0.1385`, VaR95 range
  `4.1152`;
- start-only null: `1/6` gates, zero factor/path/portfolio separation.

Interpretation: the new direction-safe guard is an improvement over the
earlier failed pre-rollout probe and demonstrates conditionality above the
start-only null. It is not promoted yet because it is only a 64-sample
fixed-start gate and has not passed a broader held-out scenario-quality run.
The next evidence target is to scale the same direction-safe response-aware
scorer to a larger sample size and then run held-out CRPS, energy, coverage,
and qualitative raw-level factor/portfolio plots.

Next method direction: keep the successful 934a mechanism but make it
operational. Instead of using a generic train-only book prior, the next
candidate should learn or calibrate a response-aware scorer closer to the
actual generator-response surface: support features plus narrative channels
should predict which support components produce useful relevant-factor and
portfolio-tail response after frozen SNI rollout, without leaking realized
future paths.

## Larger Fixed-Start Follow-Up And Direction-Safe Candidate Recovery

Artifacts:

- 384-sample fixed-start rollout:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_937a_portfolio_quality_guard_direction_safe_s384/fixed_start_rollout_policy_comparison.json`
- 384-sample conditionality stress:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_937a_portfolio_quality_guard_direction_safe_s384/conditionality_stress_test.json`
- fallback-reason breadth diagnostic:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_quality_guard_live_breadth_937d_start18_supportmax_off_reasons/portfolio_quality_guard_live_breadth.json`
- direction-safe candidate recovery diagnostic:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_quality_guard_live_breadth_937e_start18_direction_safe_candidate/portfolio_quality_guard_live_breadth.json`
- support-max-off rollout:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_937f_qg_direction_candidate_supportmax_off_s64/conditionality_stress_test.json`
- default-guard rollout after candidate recovery:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_937g_qg_direction_candidate_default_s64/conditionality_stress_test.json`

Implementation update:

- `portfolio_quality_guard_924e` now exposes the candidate-entropy and
  support-max quality-guard thresholds as research controls while preserving
  prior defaults.
- If the marginal response-weighted support set fails the final mixed-prefix
  direction check, the selector can now recover the first high-scoring
  candidate mixture that passes the final direction check before falling back
  to the base support prior.

Status: `candidate_not_promoted_larger_gate_mixed`

The 384-sample 937a gate confirmed that both the incumbent and the portfolio
quality guard produce nonzero conditionality versus the start-only null. The
portfolio guard passes all six gates and slightly improves mean portfolio
terminal KS (`0.1198` versus `0.1137`), but relevant-factor terminal KS and
path energy are slightly weaker than the incumbent (`0.1305` versus `0.1331`,
and `0.0234` versus `0.0239`). This is not a clean improvement.

The fallback diagnostics show the mechanism clearly. Candidate breadth is not
the primary blocker. Most failed activations fall back because the final
mixed-prefix direction check rejects the response-weighted mixture. Disabling
the support-max concentration guard does not solve this; it worsens
fixed-start conditionality in the 937f stress test. The new direction-safe
candidate recovery is useful as a safety improvement, but only changed one of
six fixed-start live cases in the diagnostic and did not outperform the
incumbent in the 937g default-guard stress:

- incumbent: `6/6` gates, relevant terminal KS `0.1884`, path energy `0.0283`,
  portfolio KS `0.1625`, VaR95 range `4.5227`;
- portfolio quality guard: `6/6` gates, relevant terminal KS `0.1677`, path
  energy `0.0215`, portfolio KS `0.1354`, VaR95 range `2.2341`;
- start-only null: `1/6` gates and zero separation.

Decision: reject the current direction-safe portfolio quality guard as a
default/promotion candidate. Keep the implementation because it improves
auditable failure handling and preserves defaults, but treat it as diagnostic.
The next principled method is not another threshold sweep. It should change the
candidate-selection mechanism so final direction compatibility is enforced
before response-quality marginalization, or replace the generic portfolio
response score with a learned narrative-channel-specific response scorer whose
candidate sets pass both direction and scenario-quality gates.

## Preview-Split Response TestFlight

Artifacts:

- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_response_preview_support_weighting_934c_p8/response_aware_support_weighting_report.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_response_preview_support_weighting_934c_p16/response_aware_support_weighting_report.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_response_preview_support_weighting_934c_p32/response_aware_support_weighting_report.json`

Status: `candidate_mechanism_found_not_promoted`

The 934c preview-split TestFlight is closer to a deployable response-aware
support scorer than 934a. It still uses cached component-preserving rollouts,
but it separates the scoring samples from the final pooling samples. For each
support component, the response score is estimated from a small shuffled
preview subset; the final scenario deck is resampled from the remaining
component samples when possible. This emulates a two-stage operational path:
run a small frozen-generator preview, score support components in the
narrative's risk channels, then run or pool the final scenario distribution.
It uses no realized future paths and no OpenAI calls.

Sensitivity results versus the current component-preserving baseline:

| Pilot samples/component | Gates | Relevant KS delta | Path-energy delta | Portfolio KS delta | VaR95 range delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | `6/6` | `+0.0352` | `+0.0095` | `+0.0467` | `+4.4488` |
| 16 | `6/6` | `+0.0359` | `+0.0076` | `+0.0439` | `+0.4514` |
| 32 | `6/6` | `+0.0327` | `+0.0065` | `+0.0437` | `+3.9899` |

Interpretation: unlike the train-only narrative-book guard, the preview-split
response scorer preserves and strengthens fixed-start narrative conditionality
across multiple preview sizes. This suggests the useful signal is in the
frozen generator's local response surface, not in a generic train-only
portfolio-book reliability prior. It is still not promoted as production
default because the current evidence is an artifact-level split over cached
rollouts. The next step is to wire this as a true two-stage story-smoke mode:
small preview rollouts per support component, response-aware weight update,
then final component-preserving rollout with held-out scenario-quality checks.

## Live Two-Stage Story-Smoke Probe

Artifacts:

- 64-sample live rollout root:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_935a_response_preview_s64`
- 64-sample stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_935a_response_preview_s64/conditionality_stress_test.json`
- 384-sample full-preview comparison:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_932a_s384`
- 384-sample full-preview stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_935b_response_preview_s384/conditionality_stress_test.json`
- 384-sample bounded-preview comparison:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_935c_response_preview_blend035_s384`
- 384-sample bounded-preview stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_935c_response_preview_blend035_s384/conditionality_stress_test.json`

Status: `candidate_mechanism_not_promoted_live_probe_negative_at_scale`

The live story-smoke implementation applies the preview idea before final
rollout: build the incumbent diverse, direction-checked support set; run a
small preview rollout for each support component; score those components in the
grounded narrative channels; update support weights; then reset the RNG seed
and run the final component-preserving scenario deck from the reweighted
support mixture. This is operationally closer to production than the cached
934c split because the preview samples are separate generator calls, not a
post-hoc split of one already generated deck.

The small 64-sample smoke was encouraging:

| Policy | Gates | Relevant KS | Path energy | Portfolio KS | VaR95 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| current component baseline | `6/6` | `0.1714` | `0.0162` | `0.1354` | `3.587` |
| response preview | `6/6` | `0.1894` | `0.0222` | `0.1479` | `4.686` |
| start-only null | `1/6` | `0.0000` | `0.0000` | `0.0000` | `0.000` |

The larger 384-sample gate did not confirm that gain. With full preview
reweighting, response preview stayed direction-consistent but weakened all
main separation metrics versus the incumbent:

| Policy | Gates | Relevant KS | Path energy | Portfolio KS | VaR95 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| current component baseline | `6/6` | `0.1775` | `0.0431` | `0.1179` | `10.339` |
| response preview, full | `6/6` | `0.1574` | `0.0272` | `0.1071` | `5.935` |
| start-only null | `1/6` | `0.0000` | `0.0000` | `0.0000` | `0.000` |

A bounded blend of preview weights with incumbent weights reduced weight
concentration, but it still underperformed:

| Policy | Gates | Relevant KS | Path energy | Portfolio KS | VaR95 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| response preview, `0.35` blend | `6/6` | `0.1422` | `0.0215` | `0.0847` | `5.559` |

Mechanism read:

- The full preview version moves support weights materially and sometimes
  concentrates them too much, reducing effective support breadth.
- The bounded blend preserves breadth better, but the preview response signal
  is still too noisy or too local to improve the final 384-sample distribution.
- The cached 934c improvement was therefore not sufficient promotion evidence;
  it was a useful mechanism probe, not a deployable default.

Decision: keep the live preview implementation as a diagnostic tool, not the
production path. The next principled method should learn or calibrate a
deployable response-aware support scorer from historical backtest labels and
preview-response labels, then test that scorer against the current component
baseline. Do not add another preview temperature/blend sweep unless the new
experiment has a sharper causal question and a held-out quality gate.
