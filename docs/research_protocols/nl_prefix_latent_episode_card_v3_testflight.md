# NL EpisodeCardV3 TestFlight Protocol

## Purpose

This protocol defines the next isolated TestFlight before any full corpus
regeneration. The goal is to test whether richer, multi-view and multi-angle
historical episode narratives improve narrative-to-support retrieval and
fixed-start conditionality.

The current paper/demo default remains unchanged:

```text
professional narrative
+ grounding sidecar
+ fixed selected start
-> nearest-similar top3/90 support posterior
-> component-preserving frozen SNI rollout
```

EpisodeCardV3 is a research branch. It must not replace the incumbent default
until the TestFlight passes and the user explicitly approves full regeneration.

## Core Design

Each historical 30-day prefix is represented as an episode. Each episode can
produce multiple supported economic angles, and each angle is written through
multiple narrative views.

```text
30-day historical prefix
-> deterministic prefix facts
-> supported economic angles
-> view x angle narrative records
-> retrieval records grouped back to one support episode
```

The corpus size is therefore:

```text
episodes x valid_angles_per_episode x views_per_angle
```

Not every episode receives every angle. Unsupported angles become rejection
examples or hard negatives, not positive retrieval documents.

## Windowing Policy

Rich institutional narrative records are expensive and redundant under daily
windows because adjacent 30-day prefixes share 29 observations. The TestFlight
therefore uses:

- rich institutional episode cards on a 15-day stride / half-overlap;
- short sparse user-like narratives on daily windows inside the same TestFlight
  slice;
- the full daily support bank remains available for final start-fit and SNI
  memory compatibility checks.

This gives the retriever realistic sparse user prompts without flooding the
rich institutional corpus with near-duplicates.

## TestFlight Scale

Start with a bounded Codex/GPT-authored TestFlight:

- 50-100 historical 30-day prefixes for rich institutional records;
- daily sparse user-like records inside the same date span;
- no paper/demo default changes;
- no retrieval/backtest claims from locally generated prose.

If the TestFlight passes, full regeneration can use the same direct-authored
schema and validators with a larger 15-day rich stride plus daily sparse-user
records.

## Codex-Authored Narrative Review Gate

Before full regeneration, run a small Codex-authored narrative regeneration
TestFlight:

```text
broad support-card inventory
-> temporally diverse historical windows
-> strict RiskManagerCaptionV2 Codex prompt
-> validated Codex-authored captions
-> human review markdown packet
-> user approval before full regeneration
```

This gate exists because local deterministic/template prose is banned for
narrative artifacts, including smoke tests and mechanism tests. Local code may
compute structured facts, support metadata, supported-angle labels, validation
status, and leakage status. The Codex-authored review packet must include the
selected historical evidence, raw generated captions, evidence used, ambiguity
flags, no-forecast caveats, hard negatives, validation status, and leakage
status.

The earlier `974a` review packet is not review-ready for the multi-format
question because it generated one caption per window and then rendered several
local views from that same caption. The `977a` TestFlight fixed the request-like
sparse prompt bug. The `978a` TestFlight added an explicit bank/institution
stress-scenario field, but exposed residual quality warnings where weekly
monitor fields were still too factor-list-like. The `979a` packet passed those
checks but still used bank-stress wording for non-stress episodes and could
inherit over-broad rough labels. The current corrected multi-format TestFlight
is `980a`; it asks Codex to author the mechanical baseline, sparse declarative
user narrative, weekly monitor, institutional risk-committee note,
mechanism-first memo, and full risk-manager memo as separate fields under one
strict schema, with scenario titles selected from the dominant evidence. The
`975a`, `976a`, `977a`, `978a`, and `979a` packets are superseded for full
regeneration review. The later local 981 corpus is also retracted because its
searchable views were locally generated rather than directly Codex/GPT-authored.

The current corrected TestFlight artifacts are:

- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/review_ready_single_period.md`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_review_ready.md`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_codex_report.json`

Do not start full corpus regeneration until the user reviews this packet and
explicitly approves the schema and prose standard.

## Narrative Views

Each supported angle should be written in several source-inspired styles. These
styles are structural templates, not copied text.

Required views:

1. `risk_manager_memo`: portfolio-facing risk-manager language, including
   mechanism, transmission, cross-asset evidence, ambiguity, and no-forecast
   caveat.
2. `fed_current_conditions`: central-bank style current/recent conditions,
   using measured language about what has changed and what uncertainty remains.
3. `institutional_risk_committee_note`: committee-level risk note with
   severity, affected exposures, transmission channels, and caveats; for relief
   regimes it must state that the episode is not an acute stress state.
4. `macro_outlook_newsletter`: economist or institutional outlook style,
   emphasizing macro backdrop, policy/liquidity channel, and uncertainty.
5. `weekly_risk_monitor`: concise institutional market-monitor style, focused
   on what changed, what risk channel is active, and which markets confirm it.
6. `technical_factor_evidence`: compact factor-evidence language for
   deterministic audit and lexical retrieval baselines.
7. `sparse_user_query`: short realistic user input that mentions only one or
   two channels and leaves cross-asset implications for the system to infer.

Optional views after the first pass:

- `one_factor_user_query`;
- `two_factor_user_query`;
- `ambiguous_user_query`;
- `hard_negative_near_miss`.

## Supported Angles

The first TestFlight should include the tightened taxonomy from the 972b
support-card work:

- `Fragile risk-on rebound`;
- `Defensive risk-off shock`;
- `Commodity-inflation pressure`;
- `Dollar-liquidity squeeze`;
- `Rates-led tightening pressure`;
- `Classic safe-haven gold risk-off`;
- `Safe-haven gold bid`;
- `Gold-duration bid without risk-off confirmation`;
- `Gold up in risk-on relief`;
- `Gold up mixed defensive`;
- `Dollar/rates defensive regime`;
- `Dollar strength in risk-on relief`;
- `Mixed cross-asset regime`.

Angle assignment must be evidence-based. For example:

- Classic Safe-haven Gold requires Gold up, US10Y down, VIX up, and SPX down.
- Medium-confidence Safe-haven Gold requires Gold up and US10Y down plus at
  least one stress confirmation.
- Gold up with SPX up and VIX down is not classic safe-haven risk-off; it is a
  risk-on relief or mixed-duration Gold case.
- Commodity Inflation must include crude or commodity evidence.
- Dollar Liquidity must include DXY, FX, funding, or dollar-pressure evidence.

## EpisodeCardV3 Record Schema

Each searchable narrative record should include:

- `episode_id`;
- `window_index`;
- `history_start`;
- `history_end`;
- `window_stride_type`: `rich_15d`, `sparse_daily`, or `nearby_daily_refine`;
- `view_name`;
- `angle_name`;
- `confidence`: `high`, `medium`, `low`, or `reject`;
- `narrative_text`;
- `short_query_variant`;
- `structured_claims`;
- `anchor_markets`;
- `supporting_evidence`;
- `contradictions`;
- `inferred_channels`;
- `source_style`: `fed_like`, `stress_test_like`, `weekly_monitor_like`,
  `macro_outlook_like`, `risk_manager_like`, `technical_like`, or
  `sparse_user_like`;
- `hard_negative`;
- `leakage_status`;
- `timestamp_safety_status`;
- `factor_prefix_summary`;
- `sni_memory_key_ref`;
- `support_prefix_ref`.

The retrieval index may contain many records per episode, but final support
selection must aggregate records back to `episode_id` so one episode is not
over-counted merely because it has more narrative variants.

## Leakage Rules

Narratives describe only the current/recent 30-day prefix.

Forbidden in retrieval text:

- realized next-30-day path;
- generated scenario outcome;
- terminal value;
- VaR or ES outcome;
- target PnL;
- explicit prediction that a factor will move after the conditioning window.

Allowed:

- current/recent prefix evidence;
- ambiguity and uncertainty;
- warning-only future-risk caveat, explicitly marked as not a conditioning
  target.

## Retrieval TestFlight

The TestFlight should compare:

1. incumbent projected-memory selector;
2. 971/972 hybrid start-aware narrative selector;
3. EpisodeCardV3 dense text-to-text selector;
4. EpisodeCardV3 hybrid selector with start-fit and grounded-claim checks;
5. optional Codex/agentic shortlist rerank if local retrieval is semantically
   weak.

Required controls:

- start-only/no-narrative support selection;
- shuffled-narrative null;
- hard-negative query retrieval;
- temporal-gap support diversity;
- grouped-by-episode retrieval so narrative-record count does not bias support
  selection.

## Evaluation Gates

Primary product gate:

```text
same accepted start
+ different user/professional narratives
-> different support episodes
-> different generated scenario distributions
```

Required metrics:

- support overlap versus start-only;
- title/archetype match for intended angles;
- grounded-claim support pass rate;
- narrative-vs-start-only factor KS and path energy distance;
- start-normalized narrative attribution;
- qualitative fan charts in narrative-relevant factors;
- held-out CRPS, Energy Score, and 80% coverage as guardrails.

Promotion requires narrative lift without unacceptable historical-quality
regression. If conditionality improves but CRPS/Energy degrade too much, the
TestFlight should stop at research-candidate status and not trigger full
regeneration.

## TestFlight Output Contract

The TestFlight must produce:

- `episode_card_v3_schema.json`;
- `episode_card_v3_testflight_cards.jsonl`;
- `episode_card_v3_quality_report.json`;
- `episode_card_v3_retrieval_report.json`;
- `episode_card_v3_scenario_eval_report.json`;
- `episode_card_v3_conditionality_lift_report.json`;
- a research-log entry summarizing results and whether full regeneration is
  recommended.

## Decision Rule

After the Codex/GPT-authored TestFlight, stop and ask for explicit user
approval before full regeneration.

Recommended outcomes:

- `go_full_regeneration`: passes direct-authored narrative quality, leakage,
  and guardrail checks.
- `revise_schema`: useful signal but weak view/angle quality or missing
  support coverage.
- `do_not_regenerate`: no material improvement over 971/972 hybrid or start-only
  controls.
