# NL Episode-Level Narrative Retrieval Plan

## Purpose

This is a new isolated research lane for improving how a professional
risk-manager narrative retrieves historical support episodes.

The incumbent paper/demo method remains unchanged:

```text
professional narrative
+ grounding sidecar
+ fixed selected start
-> nearest-similar, direction-checked, non-overlapping support regimes
-> main-regime top3/90 posterior view
-> component-preserving frozen SNI rollout
```

This branch asks whether support selection improves when historical 30-day
episodes are searched as rich text objects rather than only through a projected
128-dimensional SNI final-hidden memory vector.

The primary differentiator for this branch is **fixed-start conditionality over
the start-only baseline**. Held-out CRPS, Energy Score, and coverage remain
guardrails, but they are not the main product claim. The core product question
is:

```text
same accepted start
+ narrative-conditioned episode retrieval
vs.
same accepted start
+ start-only support retrieval
-> does the narrative add visible and economically meaningful scenario
   distribution movement?
```

## Core Hypothesis

The current support selector can lose information because it compresses the
input narrative into an embedding and then into a single projected memory
vector. A historical 30-day episode may match a narrative because of its full
path shape, sequencing, cross-asset mechanism, and risk-manager interpretation.

A better selector should compare:

```text
user narrative <-> historical episode narrative card
```

and then retain the existing numerical guardrails:

```text
grounded claim agreement
+ fixed-start compatibility
+ temporal diversity
+ SNI memory compatibility
+ top3/90 component-preserving rollout
```

## Corpus-Quality Finding

The current caption inventory is useful but not sufficient by itself for the
final text-to-text retrieval product. The six paper/demo narratives are
professional and readable, but they are deliberately fully specified across many
factors. The larger self-supervised caption corpus contains structured fields
such as trigger, transmission, sequence, and portfolio implication, but the
`training_caption` text is often short and factor-list-like. A real risk
manager may describe only one or two visible channels and expect the system to
infer the remaining cross-asset configuration.

Therefore this branch must begin with corpus enrichment before retrieval. The
goal is not to invent unsupported causes. The goal is to create multiple
timestamp-safe, leakage-free narrative views of each historical 30-day prefix
so sparse user narratives can retrieve matching episodes by mechanism,
sequencing, and risk channel, not only by explicit factor names.

## Non-Negotiable Guardrails

- Do not modify the existing paper/demo default.
- Do not replace the incumbent top3/90 support-posterior method.
- Do not edit the existing caption generator, cookbook, or self-supervised
  narrative files in place.
- If an existing script is needed as a starting point, copy it to a new file
  with an `episode_narrative_retrieval` name and modify only the copy.
- Do not upload private paper text, private notebooks, or under-review material
  to external services.
- Do not let realized future paths leak into training captions or retrieval
  episode cards.
- Do not promote this branch unless it beats or is clearly competitive with
  the incumbent on historical quality and shows stronger fixed-start
  conditionality.

## Required Starting Point

Start with a **Phase 0c EpisodeCardV3 TestFlight**, documented in:

`docs/research_protocols/nl_prefix_latent_episode_card_v3_testflight.md`.

Rationale:

- Current captions are useful but too often read like factor move summaries.
- Real risk-manager queries may be sparse, mechanism-led, and qualitative.
- A text-to-text support retriever needs historical cards that can be searched
  by mechanism, transmission, sequencing, and partial evidence.
- This protects tomorrow's presentation by leaving the current loop untouched.

Online material hunting is now a required later phase, but only after a
Codex/GPT-authored episode-card inventory exists and leakage controls are in
place.

The corpus uses 15-day stride / half-overlap windows for rich
institutional-style episode narratives, while allowing short sparse user-like
query variants on daily windows. Rich views should imitate the structure of
central-bank, stress-test, macro-outlook, weekly market-monitor, and
risk-manager writing without copying external text. Sparse user-like views
should be short, incomplete, and realistic: they may mention only one or two
channels and require the system to infer the remaining cross-asset regime.
Every searchable view must be directly Codex/GPT-authored or supplied by a
trusted human/source document.

Local deterministic/template prose is banned for all narrative artifacts,
including smoke tests and mechanism tests. Local code may still compute
structured facts, supported-angle labels, validation fields, and numerical
scenario metrics.

Correction from the 974a-981 runs: a local template-rendered V3 corpus is
invalid, a one-caption-per-window Codex packet is not enough to assess distinct
narrative formats, and sparse fields must be declarative market narratives
rather than assistant requests or internal-system references. The current
corrected review packets are:

- one-page single-period review:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/review_ready_single_period.md`
- all-window review:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_review_ready.md`
- report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_codex_report.json`

## Candidate Methods

### 1. Incumbent Memory Retrieval Baseline

Use the current projected text-to-SNI-memory retrieval plus grounding and
top3/90 rollout. This is the benchmark floor.

### 2. Narrative-to-Narrative Dense Retrieval

Embed the user narrative and each historical episode narrative card. Retrieve
support episodes by text-to-text similarity, then apply direction, start, and
temporal-diversity guardrails.

### 3. Hybrid Episode Retrieval

Score each candidate support episode with a weighted combination:

```text
episode_score =
  text_dense_similarity
+ lexical_or_claim_overlap
+ grounded_direction_agreement
+ SNI_memory_compatibility
- fixed_start_gap_penalty
- temporal_overlap_penalty
```

This is the recommended first candidate because it keeps both rich narrative
matching and numerical market-state discipline.

### 4. Hybrid Retrieval Plus Qualitative Rerank

Use Codex or an LLM only on the top retrieved candidates. The reranker asks
whether the full 30-day episode qualitatively matches the user narrative's
regime, mechanism, sequencing, and cross-asset evidence. The reranker is not
the final judge; downstream historical backtests and deterministic direction
checks remain final.

If pure semantic similarity is not strong enough, Codex/agentic adjudication is
allowed as a bounded rerank layer over a shortlist. It must emit an auditable
match rationale, missing/contradictory channel notes, and structured scores. It
must not be used as an untraceable oracle over the full support bank.

### 5. Online-Enriched Episode Retrieval

Use Codex-assisted online search to collect timestamp-safe public macro/risk
context for selected historical windows. This can include official documents,
public market commentaries, and auditable macro-risk summaries. The online text
is used to enrich episode cards and create more realistic sparse narrative
views. It must not import future outcomes or unverified claims into the
conditioning text.

## Episode Card Schema

Each historical 30-day episode card should contain:

- `window_id`
- `history_start`
- `history_end`
- `split`
- `training_caption`
- `scenario_title`
- `archetype`
- `mechanical_summary`
- `trigger`
- `transmission`
- `cross_asset_reaction`
- `sequence`
- `portfolio_vulnerability`
- `risk_manager_implication`
- `evidence_used`
- `ambiguity`
- `no_forecast_caveat`
- `view_full_professional`
- `view_sparse_user_query`
- `view_mechanism_first`
- `view_one_or_two_factor_headline`
- `view_factor_list_baseline`
- `hard_negative_views`
- optional timestamp-safe online context snippets and source URLs
- structured grounded claims from the caption
- factor-prefix summary for deterministic direction checks
- SNI final-hidden memory key
- full support prefix pointer for rollout

The card describes only the current/recent 30-day prefix. Realized future
returns, generated scenario outcomes, terminal values, VaR, ES, and target PnL
are forbidden in retrieval text fields.

## Implementation Shape

Use new files only:

- `experiments/backfill/block_ar/nl_episode_narrative_cards.py`
- `experiments/backfill/block_ar/nl_episode_narrative_retrieval.py`
- `experiments/backfill/block_ar/nl_episode_narrative_retrieval_testflight.py`
- `experiments/backfill/block_ar/nl_episode_narrative_conditionality_lift.py`
- `test_code/test_nl_episode_narrative_retrieval.py`

Do not modify these incumbent files for the first branch:

- `experiments/backfill/block_ar/nl_risk_manager_caption_v2.py`
- `experiments/backfill/block_ar/nl_codex_caption_batch.py`
- `experiments/backfill/block_ar/nl_narrative_grounded_scenario_pipeline.py`
- `experiments/backfill/block_ar/nl_support_component_posterior_bakeoff.py`
- paper/demo files and saved presentation artifacts

Import reusable helpers from incumbent modules when stable; copy small
experiment-specific logic only when isolation is safer.

## Evaluation Plan

Run the same user-facing fixed-start narratives and the same historical
quality gates against five support selectors:

1. incumbent projected-memory selector;
2. dense narrative-to-narrative selector;
3. hybrid episode selector;
4. hybrid selector plus Codex/LLM qualitative rerank on top candidates;
5. online-enriched hybrid selector.

Primary checks:

- fixed-start narrative-vs-start-only support overlap;
- fixed-start narrative-vs-start-only distributional lift;
- start-only and shuffled-narrative null controls;
- grounded-claim support pass rate;
- temporal diversity and support-family cohesion;
- narrative-relevant factor KS and path-energy separation;
- portfolio VaR/ES or tail-channel separation;
- held-out CRPS, Energy Score, and 80% coverage as guardrails;
- qualitative fan-chart and support-provenance review.

## Current Two-Stage Text-Space Retrieval Objective

After the six-case support review, the active branch is no longer pure
semantic narrative-to-narrative matching. The finding was that generic text
similarity can match wording such as "gold", "duration", or "liquidity" while
missing the cross-asset market state. The new objective keeps retrieval in
OpenAI embedding space but trains/evaluates the metric with financial
supervision.

Stage 1 is a multi-view contrastive retriever:

- input representation: `text-embedding-3-large` over the existing
  direct Codex/GPT-authored 982g narrative corpus;
- positives: different narrative views for the same historical 30-day prefix;
- hard negatives: narratives that sound similar but fail grounded market
  compatibility or are far in historical/SNI support evidence;
- output: a text-space retrieval metric or adapter that scores narrative
  compatibility without projecting the query into 128-dimensional SNI memory.

Stage 2 is a frozen-SNI backtest preference reranker:

- retrieve a candidate support pool from Stage 1;
- keep the existing support object, top3/90 component-preserving assembly, and
  frozen SNI rollout unchanged;
- compare each candidate/support set with the realized held-out next-30-day
  path using existing historical backtest metrics such as CRPS and Energy
  Score;
- train/evaluate pairwise preferences that rank better supports above worse
  supports.

Baselines for this branch are raw OpenAI embedding retrieval, projected-memory
plus grounding, start-only terminal-state retrieval, and the current verified
top3/90 paper/demo candidate. Promotion requires improved support coherence and
fixed-start narrative conditionality without unacceptable regression in CRPS,
Energy Score, coverage, temporal diversity, or grounded-claim pass rate.

## Scale Plan

### Phase 0: Corpus-Quality And Multi-View Episode Cards

Audit the existing caption inventory and build richer historical episode cards
without editing the incumbent caption files. Each card should include the
existing structured caption fields plus multiple narrative views:

- risk-manager memo;
- central-bank/Fed-like current-conditions language;
- institutional risk-committee language;
- macro/economics outlook newsletter language;
- weekly risk-monitor language;
- technical factor-evidence language;
- sparse user-query style;
- hard-negative and near-miss views.

This phase must also report corpus coverage, length distribution, factor-list
bias, causal/mechanism coverage, and leakage checks.

For EpisodeCardV3, each searchable record is `view x supported_angle`.
Unsupported angles must be recorded as rejection examples or hard negatives,
not positive retrieval records. Retrieval must aggregate records back to the
historical `episode_id` before support selection so an episode is not
over-counted merely because it has more text variants.

### Phase 1: Local Retrieval Mechanism

Use existing captions and a bounded support-bank sample. No online enrichment.
Target: prove whether text-to-text retrieval changes support selection in the
right direction when the query is either fully specified or sparse.

### Phase 2: Scenario-Level TestFlight

Pipe selected supports through the existing top3/90 component-preserving frozen
SNI rollout. Compare against the incumbent with matched starts, narratives,
seeds, and sample counts.

**Current 2026-06-01 status.** Phase 0b/1/2 now have an isolated local
TestFlight:

- broad raw-history support-card inventory:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_support_bank_cards_970c_train/support_card_quality_report.json`;
- all-window support cards for decoder-test queries:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_support_bank_cards_all_970f/support_card_quality_report.json`;
- gap-30 broad retrieval screen:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_retrieval_phase1_broad_support_gap30_970e/local_retrieval_report.json`;
- 66-window bridge-style report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_bridge_report_970g_gap30_full66/episode_retrieval_bridge_report.json`;
- 66-window scenario TestFlight with true-history oracle:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_scenario_eval_970i_top3_full66_s4_oracle/scenario_level_eval_report.json`;
- compact evidence summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_evidence_summary_970j/episode_narrative_evidence_summary.json`.

The result is `mechanism_found_not_promoted`: episode-level text retrieval over
the broad support inventory improves 66-window CRPS by `13.3%` and energy by
`15.4%` versus persistence, close to the same-split true-history SNI oracle
(`15.2%` CRPS and `16.2%` energy). The branch is still not a default because
the local raw-history cards are rule-based and show weak semantic specificity
for rates/commodity narratives.

### Phase 3: Candidate Confirmation

If phase 2 passes, expand to more windows/starts and add repeat/null controls.
Trigger independent verification before any paper/demo claim.

**Current 2026-06-01 Phase 3 result.** A start-only bridge report and matched
66-window frozen-SNI rollout were added for the same decoder-test starts:

- start-only bridge report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_start_only_bridge_report_971a_gap30_full66/start_only_bridge_report.json`;
- start-only scenario evaluation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_start_only_scenario_eval_971b_top3_full66_s4/scenario_level_eval_report.json`;
- narrative-vs-start-only lift audit:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_conditionality_lift_971c_vs_start_only/conditionality_lift_report.json`.

Result: `conditionality_lift_detected_quality_warning`. Episode-level narrative
retrieval changes supports almost completely relative to start-only retrieval
(`mean support Jaccard 0.006`) and creates material generated-distribution
separation (`mean terminal factor KS 0.365`, path energy distance 0.340,
mean terminal mean shift 0.360 standardized units). However, the pure local
episode-text retriever is weaker than start-only on historical fidelity:
CRPS `0.579` vs start-only `0.527`, Energy `0.776` vs start-only `0.693`.

Interpretation: conditionality is present, but the current pure text-to-text
selector is not yet promotable. The next method should combine start-fit,
narrative episode match, grounded direction checks, and optional Codex/agentic
reranking so narrative adds scenario lift without discarding too much
state-level information.

### Phase 4: Codex-Assisted Online Search And Episode Enrichment

Run a bounded online-enrichment branch using Codex-assisted search or equivalent
auditable online lookup. Enrich selected historical cards with timestamp-safe
public macro context from official or auditable sources such as FOMC materials,
FRED observations, IMF/central-bank publications, public market commentaries,
and licensed/internal market commentary where permitted.

This phase is required for the overall branch. It may remain diagnostic if it
adds noise, leakage risk, or weak scenario-level performance.

**Current 2026-06-01 Phase 4 local hybrid result.** Before invoking Codex
reranking or online enrichment, a deterministic start-aware narrative reranker
was added:

```text
local narrative-to-narrative recall
-> combined score = text_weight * narrative_match
                  + start_weight * terminal_start_match
-> temporal-gap top-k support set
-> same top3/90 frozen-SNI scenario evaluation
```

Artifacts:

- hybrid bridge reports:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_bridge_971d_t25_s75_gap30_full66`,
  `.../episode_narrative_hybrid_start_text_bridge_971e_t50_s50_gap30_full66`,
  and
  `.../episode_narrative_hybrid_start_text_bridge_971f_t75_s25_gap30_full66`;
- scenario evaluations:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_eval_971h_t25_s75_top3_full66_s4`,
  `.../episode_narrative_hybrid_start_text_eval_971g_t50_s50_top3_full66_s4`,
  and
  `.../episode_narrative_hybrid_start_text_eval_971i_t75_s25_top3_full66_s4`;
- selection summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_selection_971m/hybrid_start_text_selection_report.json`.

Result: all three hybrid weights pass `conditionality_lift_detected`. The
recommended next candidate is `text_weight=0.25`, `start_weight=0.75` because
it has the best CRPS/Energy among the hybrid candidates while preserving
narrative-vs-start-only lift:

- CRPS `0.541` versus start-only `0.527`;
- Energy `0.710` versus start-only `0.693`;
- mean support Jaccard versus start-only `0.042`;
- mean terminal factor KS versus start-only `0.292`;
- path energy distance versus start-only `0.133`.

Interpretation: the first deterministic hybrid fixes the pure-text failure
mode. It preserves enough state compatibility to keep historical quality close
to start-only, while still giving measurable scenario movement from the
narrative. Codex/agentic reranking remains allowed, but it is no longer the
immediate next step unless the higher-sample confirmation exposes a semantic
failure that the deterministic hybrid cannot resolve.

**Higher-sample confirmation.** The recommended `text_weight=0.25`,
`start_weight=0.75` candidate was rerun with 16 samples per support component
against the matched start-only baseline:

- start-only high-sample evaluation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_start_only_scenario_eval_971n_top3_full66_s16/scenario_level_eval_report.json`;
- hybrid high-sample evaluation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_eval_971o_t25_s75_top3_full66_s16/scenario_level_eval_report.json`;
- high-sample lift audit:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_lift_971p_t25_s75_s16_vs_start_only/conditionality_lift_report.json`;
- confirmation summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_confirmation_971q/hybrid_start_text_confirmation_report.json`.

Result: `conditionality_lift_detected` persists at higher sample count. The
hybrid remains close to start-only on quality while adding narrative
conditionality:

- hybrid CRPS `0.519` versus start-only `0.506`;
- hybrid Energy `0.675` versus start-only `0.660`;
- mean support Jaccard versus start-only `0.042`;
- mean terminal factor KS versus start-only `0.213`;
- path energy distance versus start-only `0.075`.

Decision: keep `text_weight=0.25`, `start_weight=0.75` as the current
episode-retrieval research candidate. It should receive verifier review and
optional semantic spot checks before any public default or paper/demo change.

### Phase 5: Product Readout And Promotion Review

If any selector beats or is competitive with the incumbent, build a product
readout that explains:

- which historical episode narratives matched the user narrative;
- which factors were explicit versus inferred;
- which grounding claims passed support checks;
- how the selected support posterior changes the 30-day distribution;
- where the method improves fixed-start conditionality and where it does not.

Do not change demo or paper defaults until the independent verifier agrees.

## Full Regeneration Prep

The full-regeneration prep packet is now documented at:

`docs/research_protocols/nl_prefix_latent_episode_full_regeneration_prep.md`.

Use that document as the runbook when the user sets the next persistent goal.
It records the approved `980a` narrative standard, mixed 15-day rich / daily
sparse windowing policy, exact command skeletons, comparison methods, product
gates, and kill conditions.

The key implementation decision is to keep the existing local lexical /
hashed-vector retriever as a cheap retrieval-control over direct Codex/GPT
cards, then add true semantic narrative-to-narrative retrieval after full
caption regeneration. The final comparison must include the incumbent
projected-memory selector, start-only terminal-state selector, dense text
retrieval, hybrid text/start retrieval, and only then optional Codex/agentic
reranking over a shortlist.

## Kill Conditions

Stop or demote the branch if:

- text-to-text retrieval selects supports that pass semantic similarity but
  fail grounded direction checks;
- conditionality improves only because start-only or shuffled-narrative nulls
  also move;
- CRPS and energy materially regress without a measured trust or
  conditionality gain;
- Codex/LLM reranking changes outputs but cannot be interpreted or reproduced;
- online enrichment creates leakage or untraceable narrative claims.

## Activation Plan

When this branch is activated, set the autoresearch goal to:

```text
Run the isolated EpisodeCardV3 full-regeneration and retrieval-evaluation
branch without changing the incumbent paper/demo defaults. Regenerate the
historical episode narrative corpus using direct Codex/GPT authoring under the
980a-approved professional multi-format standard, with rich institutional
narratives on 15-trading-day stride and sparse user-like narratives on daily
windows. Do not use local deterministic/template prose for any narrative
artifact, including smoke or mechanism tests. Build local-control and
true-embedding narrative-to-narrative retrieval indexes, compare them
against the incumbent projected-memory selector and the start-only selector,
then evaluate hybrid narrative/start support selection through the existing
top3/90 component-preserving frozen SNI rollout. The main success criterion is
fixed-start narrative conditionality over the start-only baseline; CRPS, Energy
Score, and coverage are guardrails. If local or embedding retrieval is not
semantically strong enough, add a bounded Codex/agentic rerank over a shortlist
and/or timestamp-safe online enrichment. Document every phase in the research
log, preserve leakage controls, keep existing cookbook/demo/paper files
untouched until verifier-backed promotion, and stop only after producing a
promotion/non-promotion recommendation with qualitative and quantitative
evidence.
```

The first HEAD iteration should run the prep checks in the full-regeneration
runbook, then begin corpus generation only after the user explicitly sets the
goal. It should not touch the production demo or paper defaults.
