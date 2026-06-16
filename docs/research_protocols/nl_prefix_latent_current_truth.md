# Narrative Prefix-Latent Current Truth

Last updated: 2026-06-16

This file is the tracked promotion index for the natural-language prefix-latent
scenario-generator workflow. Ignored artifacts remain the detailed evidence, but
this file states which claims are currently promoted and which are still only
diagnostic.


## 2026-06-16 — T7 reweighter: distinguishable but NOT steered; conditioning ceiling ACCEPTED (parked)

Built T7, a training-free narrative reweighter `score += beta * match(narrative_emphasis,
analogue_factor_profile)` applied before `_apply_top3_90` over the frozen joint39 SNI (beta=0 exact
no-op; 13 unit tests). Extended the #48 causal-gap fix to the embedding-bridge retrievers (they used
mutual-diversity-only filtering) and regenerated a clean 66-query deck
(`..._embedding_grounded_top3_90_66q_clean48/`, 0/528 causal-gap violations). Findings (frozen 734a,
field_weight, 48 samples, CRN-by-query):

- Responsiveness is near-tautological (a different analogue trivially gives a different rollout) and
  fidelity-neutral (CRPS median 0.515 unchanged, coverage 0.714->0.701). Not evidence of conditioning.
- **Directional steering — DECISIVE NEGATIVE.** Selected-analogue HISTORY matches narrative claims
  **0.849**, but the generated forward SCENARIO matches only **0.428 (<=chance)**; beta=0.25 vs beta=0
  is **+0.000** on movers. Fix-pool-vary-emphasis (12 pools x 4 contrasting synthetic emphases):
  cross-emphasis separation **3.46** (~5x reseed noise) = scenarios ARE distinguishable, but
  directional hit-rate **0.544** vs 0.467 control vs 0.5 chance (n=180, SE~0.037 -> not significant).
- **Root cause:** the frozen SNI washes out the seed analogue's direction (history 0.849 -> forward
  ~chance). This is the THIRD convergent ceiling alongside the risk_context oracle-injection probe
  (generator insensitive to injected conditioning) and the T4 retriever fit-ceiling.

**Conclusion (owner-accepted 2026-06-16): outside-in narrative *directional* steering is not
achievable on the frozen generator.** "Weak conditionality" = separation-without-directional-control;
prior `conditionality_lift_detected` results were measuring separation, not steering. **No method
promoted; start-only remains the absolute-fidelity leader.** T7 retains demo value (distinguishable,
fidelity-safe scenarios) but is **beta=0 for steering**. The conditionable-generator pivot (retrain
the SNI with a conditioning channel that propagates forward) is **PARKED, not killed** — owner accepts
the ceiling "for now"; the probe warns even injected conditioning is washed out, so it is an
architectural research question, not a tuning fix. Evidence:
`nl_prefix_latent_verifier_reports/2026-06-16_t7_reweighter_directional.md` (+ `_riskslot_oracle_probe.md`);
scripts `nl_t7_select_beta.py` / `nl_t7_beta_sensitivity_gate.py` / `nl_t7_directional_hitrate.py` /
`nl_t7_fixpool_conditionality.py`. No independent verifier was run (no promotion / default change).


## 2026-06-11 CONTAMINATION NOTICE — joint39 factor-mapping bug (READ FIRST)

**STATUS: RESOLVED 2026-06-12** — see recovery summary below and verifier report
`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-06-15_contamination_recovery_and_clean_rebuild.md`.
Clean 14x14 hard-negative bank rebuild COMPLETE 2026-06-15 (802/802 pass, status=pass on disk;
the earlier "IN PROGRESS" note was stale — verified 2026-06-16). No retrieval method is promoted;
the clean-restamp 14x14 trained bridges (both/projected_memory seed0-2) FAIL the fit gate
(recall@10 ~0.007 vs 0.10), reproducing the 992b information ceiling on CLEAN data. The improved
NV-Retriever objective is PENDING (not trained). The demo's wireable clean backend = the
training-free 939a numeric support-bank top3/90 (already its live default).

Narrative-facing factor extraction mapped AAA_OAS->col36 (NIKKEI) and USDJPY->col29 (COPPER)
in the untracked support-card builder (`nl_episode_narrative_support_cards.py` MARKETS); fixed
2026-06-11 ~14:11 EDT (owner Codex session). Fix committed 2c0aa31d; canonical column map now
derived at runtime from `data/multi_factor_data.npz` level_columns+25 via
`nl_joint39_anchor_map.py`, grounding gate `test_nl_joint39_anchor_map_grounding.py` PASSES.

Verified contaminated TEXT artifacts (pre-2026-06-12): 970c/970f/972b support banks; the FULL
daily 982g corpus (factor claims for USDJPY/AAA_OAS described copper/nikkei); 988b/990a 14+14
stride-5 corpus; 995a/995c val-frame corpus; demo saved packets; casebook DISPLAYED analogue
narratives. Text-method eval numbers (984a/episode_text/embedding variants, 990g, 991a/992a,
996a/b, 993a text columns) are measurements of contaminated inputs — geometric/structural
conclusions stand per the 2026-06-11 RESEARCH_LOG audit entry; exact numbers require
restamping after regeneration.

CLEAN artifacts (regenerated 2026-06-12, grounding gate passes):
- **R1 support cards**: 970c/970f/972b — regenerated clean (commit f018dc38)
- **R3 val corpus**: 995c — 89 val-frame cards, clean
- **Clean stride-5 rich narratives**: 982g — 802 cards, clean
  (`episode_card_v3_codex_multiformat_982g_clean_stride5_20260612/`)

STILL CONTAMINATED until full rebuild completes: 988b/990a 14+14 paired hard-negative bank;
text_hash_digest `c1586f2f...` in the 990a training manifest points to contaminated 988b cards.
Clean rebuild COMPLETE 2026-06-15 (verified 2026-06-16: status=pass, 802/802):
`experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_fourteen_view_bank_clean_20260615/`.
This bank is the prerequisite corpus for FUTURE retriever training; the clean-restamp bridges
trained on it already failed the fit gate (see 2026-06-16 entry above).

CLEAN: the production grounding/query lane (name-based factor lookups), all numeric machinery
and results (734a/739a, 939a, 994a/b harness+oracle, 995b chassis, 995d labels, 992b ceiling,
start-only baselines). Supersedes any statement below that implies the pre-2026-06-12 982g
corpus or 14+14 corpus "remains usable": those artifacts are AUTHORING-ROUTE PROOF ONLY and
are superseded by the clean 2026-06-12 regenerations listed above.

## Current Contract

The current paper/demo production candidate is:

```text
full risk-manager narrative
+ grounding sidecar for direction/warning checks
+ risk-manager-selected or user-supplied joint39 starting level
-> nearest-similar, direction-checked, non-overlapping support regimes
-> main-regime component-posterior view: top 3 / 90% for presentation
   (top 2 / 80% remains the sharper KS-oriented variant;
    all selected regimes remain the calibration diagnostic)
-> component-preserving decoded/refined recent prefixes
-> frozen joint39 SNI autoregressive rollout
-> 30-day scenario distribution
```

There is no hidden model-chosen starting level in the production default. A
narrative describes current or recent market conditions; forward-looking or
desired-future language is warning-only and must not become the future target.

## Active HEAD Objective: Hard-Negative Corpus Gate

The active autoresearch goal is now an explicit hard-negative corpus audit,
regeneration, and validation gate. This gate exists because the paper and
presentation describe explicit hard-negative contrastive training, but the
current 982g bridge evidence does not yet prove that matched hard-negative
narratives linked to incompatible historical memories were actually used.

Current factual state:

- the 982g positive corpus is direct Codex/GPT-authored; the contaminated daily 4010-card
  build is SUPERSEDED by the clean stride-5 802-card rebuild (2026-06-12); "remains usable"
  as written above is stale and no longer applies to the pre-2026-06-12 daily corpus;
- stored `hard_negative_views` exist, but are short rejection labels;
- those stored negatives are not yet linked to real incompatible historical
  support windows;
- current text-memory bridge training uses positive views plus in-batch target
  memory negatives, not explicit stored hard-negative narrative rows;
- current text-space contrastive work uses same-window positives and derived
  incompatibility signals, but does not yet consume a validated matched
  hard-negative narrative bank.

The required next step is:

```text
audit 982g hard negatives
-> regenerate matched hard-negative narratives with Codex/GPT only
-> link negatives to real incompatible historical windows
-> validate coverage / contradiction / leakage / split safety
-> retrain projected-memory and text-space methods with explicit negative rows
```

Until this gate passes, no paper/demo method claim should say that the current
reported bridge was trained with the full explicit hard-negative method.

Protocol:

`docs/research_protocols/nl_prefix_latent_hard_negative_corpus_plan.md`.

Audit artifact:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/hard_negative_corpus_audit_985a/hard_negative_corpus_audit.json`.

Audit result: `fail_needs_regeneration`.

- `4010` cards audited.
- Stored hard-negative text coverage is only `3` or `4` short strings per
  card.
- `0` cards meet the current minimum of `8` matched hard-negative texts.
- `0` cards have linked negative-window metadata.
- Current audited bridge/retriever reports do not use explicit stored
  hard-negative rows.

Generation manifest:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/hard_negative_bank_regeneration_985b_manifest/hard_negative_generation_manifest.jsonl`.

Manifest status: `ok_manifest_ready_for_codex_gpt_authoring`.

- `32080` rows = `4010` windows x `8` positive training views.
- Each row links a target positive view to a real incompatible historical
  window and includes contradiction channels.
- Same-window negative links: `0`.
- Duplicate target/view pairs: `0`.
- Local generated negative prose: `False`; the hard-negative text field remains
  empty until Codex/GPT authors it.

Codex/GPT authoring smoke:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/hard_negative_bank_regeneration_985c_codex_batch/hard_negative_bank.jsonl`.

Result: partial batch pass.

- `12433` generated hard-negative rows: `1554` full target windows across the
  `8` training views, plus one valid pre-existing sparse-user view for
  `joint39_train_1554`.
- `12433` rows passed independent validation.
- Remaining rows: `19647`.
- Distinct linked negative windows: `1978`.
- Same-window negative links: `0`.
- Validation errors: `0`.
- Validation warnings: `65`; these are high token-overlap warnings on
  mechanical/factor-list rows where the factor names overlap but the directions
  are contradictory. The two new warnings are the `technical_factor_evidence`
  and `factor_list_baseline` rows for `joint39_train_1462`; no new warnings
  were introduced in the latest tranche.
- Latest tranche note: a serialized `--batch-size 8 --max-new 256` tranche
  accepted all `256` requested rows, continued the normal scale-tranche path,
  extended contiguous full eight-view coverage through `joint39_train_1553`,
  and left the first missing manifest row at
  `joint39_train_1554__weekly_risk_monitor`.
- Validator hardening: same-window negative links and exact positive-text
  copies are now explicit validation errors.
- Codex errors: `0`.
- Local generated prose: `False`.

This proves the generation and validation route. It does not satisfy the full
objective yet because the remaining `19647` rows have not been authored and
validated.

### 2026-06-10 owner decision: corpus criterion re-anchored to stride-5 14+14

The full daily `32080`-row bank is de-scoped by project-owner decision. The
accepted matched hard-negative corpus surface is the stride-5 14+14 lane:

- `988b` stride-5 fourteen-view bank: `802/802` targets pass (14 positive
  views + 14 matched, window-linked hard-negative narratives per target),
  spotcheck `989a` PASS, `0` validation errors;
- `990a` training manifest: `11228` positives + `11228` matched hard
  negatives, `11228` pair rows, `0` validation errors;
- `990e` retrieval training consumed all `11228` pair rows for both the
  text-space and projected-memory methods (explicit source and reciprocal
  hard-negative memory margins).

`985c` authoring stays paused at `12433` rows (artifact kept; no further
authoring planned). The corpus-generation half of this gate is therefore
satisfied. The retraining half remains OPEN: `990e` was pilot-scale (80
optimizer steps, single seed, no held-out text split) and the matched
top3/90 scenario eval (`990g`) ranked both methods behind start-only —
logged "Do not promote". The ban below therefore still stands: the bridge
metrics currently reported in the paper/presentation come from the older
in-batch-negative training, and no paper/demo claim may state the reported
bridge was trained with the full explicit hard-negative method until a
stride-5-trained method passes the downstream backtest gate and an
independent verifier agrees.

### 2026-06-15 status: clean 14x14 bank rebuild in progress

Owner decision 2026-06-15 ("invest smart"): rebuild the clean 14x14 hard-negative bank ONCE
from clean 972b support cards, then train an IMPROVED retriever (NV-Retriever false-negative
filtering + locality-soft/posterior P5 target) AND restamp the old text_space/projected_memory
results as an honest baseline on clean data. Gate via validation framework v1 + independent
Codex verifier after retrain. The 992b information-ceiling finding (exact-window retrieval
objective is information-limited in 128-dim SNI space; oracle adjacent-window rank 208/4010,
recall@10 0.112) falsifies the exact-window training OBJECTIVE, not retrieval-conditioning in
general, and motivates the improved target. No retrieval method is promoted. The contaminated
988b/990a corpus is NOT to be used for any future retrieval training.

Clean rebuild artifact (in progress):
`experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_fourteen_view_bank_clean_20260615/`

## Previous Active Objective: Two-Stage Text-Space Narrative Retriever

The active autoresearch goal is now an isolated two-stage text-space narrative
retrieval branch. The current paper/demo default remains the verified
nearest-similar top3/90 support-posterior workflow, but the active research
question has narrowed: can we keep retrieval in OpenAI narrative-embedding space
while teaching the metric that "similar" means financially compatible
30-trading-day market condition, not merely similar prose?

This objective explicitly includes:

1. a raw `text-embedding-3-large` narrative-to-narrative baseline over the
   direct Codex/GPT-authored 982g corpus;
2. Stage 1 multi-view contrastive text-space retrieval: same historical prefix
   across narrative styles is positive, while semantically similar but
   directionally incompatible prefixes are hard negatives;
3. Stage 2 frozen-SNI historical-backtest preference reranking: keep the
   retrieved supports and current top3/90 assembly unchanged, run frozen SNI,
   and learn/evaluate which supports produce better held-out historical rollout
   quality;
4. side-by-side comparison with projected-memory plus grounding, start-only
   terminal-state retrieval, and the current top3/90 paper/demo candidate;
5. fixed-start narrative conditionality, support coherence, grounded-claim pass
   rate, CRPS, Energy Score, and coverage guardrails before any promotion.

This branch must not regenerate narratives, template narratives, change the
current top3/90 assembly, or update the frozen SNI encoder/decoder. SNI is used
as a teacher/evaluator through historical prefix identity, SNI neighborhood
checks, and decoder/backtest preference; the retrieval representation itself
stays in text-embedding space.

The 982g branch has completed full direct Codex/GPT-authored multi-format
corpus regeneration and the first retrieval/backtest/conditionality evaluation.
Local deterministic/template prose remains banned for narrative artifacts,
including smoke tests and mechanism tests. Local code may compute structured
market facts, supported-angle metadata, leakage checks, support metadata, and
scenario metrics, but every searchable narrative view must be directly authored
by Codex/GPT or supplied by a trusted human/source document.

The protocol is documented in:

`docs/research_protocols/nl_prefix_latent_episode_card_v3_testflight.md`.

The corpus-generation step should build a multi-view, multi-angle corpus:

- rich institutional narratives on 15-day stride / half-overlap windows;
- short sparse user-like narratives on daily windows inside the same
  TestFlight slice;
- source-inspired views that follow central-bank current-conditions,
  stress-test, macro-outlook, weekly risk-monitor, risk-manager memo,
  technical-evidence, and sparse-user-query styles;
- supported economic angles from the tightened 972b taxonomy;
- hard negatives and rejection examples for unsupported or contradictory
  angles;
- grouped-by-episode retrieval so an episode is not over-counted because it has
  more narrative records.

The next trustworthy evaluation must use this Codex/GPT-authored corpus and the
two-stage text-space retrieval objective before any retrieval/backtest or
conditionality claim.

### Retracted Local EpisodeCardV3 TestFlight Result

Main artifacts:

- V3 card corpus:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_testflight_973a/episode_card_v3_testflight_cards.jsonl`
- V3 quality report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_testflight_973a/episode_card_v3_quality_report.json`
- V3 scenario eval:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_scenario_eval_973e_t25_s75_top3_full66_s4/scenario_level_eval_report.json`
- Start-only matched eval:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_start_only_eval_973f_top3_full66_s4/scenario_level_eval_report.json`
- Tightened 972b matched eval:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_legacy_972b_eval_973g_t25_s75_top3_full66_s4/scenario_level_eval_report.json`
- V3 conditionality lift:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_conditionality_lift_973h_vs_start_only/conditionality_lift_report.json`
- TestFlight summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_testflight_973a/episode_card_v3_testflight_summary.json`

Corpus quality:

- `4010` grouped episode cards;
- `27754` narrative records;
- `100` rich 15-day institutional cards;
- `3910` sparse daily cards;
- `4010` hard-negative records;
- leakage pass share `1.0`;
- mean positive angles per card `2.795`.

Matched 66-window scenario results at 4 samples/support component:

- V3 CRPS `0.557`, Energy `0.736`, coverage `0.573`;
- start-only CRPS `0.534`, Energy `0.704`, coverage `0.592`;
- tightened 972b CRPS `0.550`, Energy `0.721`, coverage `0.568`.

V3 has a small quality regression versus start-only and tightened 972b, but
stronger fixed-start narrative conditionality than tightened 972b:

- V3 mean terminal factor KS versus start-only `0.310`;
- 972b mean terminal factor KS versus start-only `0.283`;
- V3 path energy distance versus start-only `0.180`;
- 972b path energy distance versus start-only `0.130`;
- V3 support Jaccard distance versus start-only `0.959`.

Decision: 973a-973i is retracted as a valid narrative-regeneration or
retrieval/conditionality TestFlight. It did not call Codex/GPT for the
searchable views; it tested locally rendered template records. These artifacts
may be read only as historical evidence of the banned failure mode. They must
not seed training, retrieval, smoke tests, mechanism tests, paper claims, demo
claims, or selector promotion.

### EpisodeCardV3 Codex-Authored Narrative TestFlight Result

The 974a review packet is superseded for the multi-format narrative question.
It generated one `RiskManagerCaptionV2` per window, then rendered multiple
local views from that same caption. That made the one-page review look like
several independent formats even though the prose was not independently
authored.

The corrected 980a TestFlight generated a small reviewable set of independent
Codex-authored multi-format narratives from the broad 4,010-window historical
support-card bank. Each window now has separate generated fields for the
mechanical baseline, sparse user prompt, weekly risk monitor, institutional
risk-committee note, mechanism memo, and full risk-manager memo.

The 975a, 976a, 977a, 978a, and 979a packets are superseded: 975a fixed independent
multi-format generation but still allowed request-style sparse prompts such as
"give me a read"; 976a removed request-style prompts but still allowed one
internal "support system" reference; 977a fixed those hard failures but did not
include the explicit bank/institution note; 978a added the bank note but showed
weekly-monitor factor-list warnings; 979a used bank-stress wording for non-stress
episodes and could inherit rough source labels. The 980a validator blocks
request/internal language, passes the tightened sparse/weekly quality checks,
and generated evidence-dominant titles on the current four-window verification
packet.

Main artifacts:

- selected historical cases:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/selected_support_cases.md`
- multi-format Codex report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_codex_report.json`
- all-window human review packet:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_review_ready.md`
- one-page review packet:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/review_ready_single_period.md`
- generated narratives:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_narratives.jsonl`

Result:

- selected cases: `4`, using a `180`-window minimum gap from the broad support
  card inventory;
- Codex model: `gpt-5.5` with `xhigh` reasoning effort;
- multi-format narratives generated: `4 / 4`;
- validation errors: `0`;
- validation warnings: `0`;
- Codex errors: `0`;

Interpretation: 980a fixes the user-identified bugs. The sparse user prompts now
intentionally mention only one or two channels, are declarative market
conditions rather than instructions to the assistant, and avoid internal system
language. The weekly monitor is less factor-list-like, the institutional
risk-committee note now works for both stress and relief regimes, and the
professional fields are longer, mechanism-first, explicit about ambiguity, and
bounded by no-forecast caveats. The mechanical baseline remains intentionally
factor-specific so reviewers can compare it against the richer institutional
formats. The 980a titles also avoid overstating gold safe-haven support and
avoid over-weighting commodity/rates support when credit-beta relief is the
dominant evidence.

Decision: the 980a packet is the approved narrative-quality standard for the
next full-regeneration branch, but it is not by itself a selector promotion.
Do not promote a paper/demo default from this narrative-quality TestFlight
alone. The full corpus is still not regenerated in a trustworthy form.

Full-regeneration prep is documented here:

`docs/research_protocols/nl_prefix_latent_episode_full_regeneration_prep.md`.

The next goal should regenerate EpisodeCardV3 records under the 980a standard,
using direct Codex/GPT authoring for every searchable narrative view. Use mixed
15-day rich / daily sparse windowing, build local-control and true semantic
text-to-text retrieval indexes, compare against the incumbent projected-memory
selector and start-only selector, and evaluate fixed-start narrative lift
through the existing top3/90 frozen-SNI rollout. The production paper/demo
default remains unchanged until a verifier-backed promotion.

Tracked plan:
`docs/research_protocols/nl_prefix_latent_episode_narrative_retrieval_plan.md`.

Active tracked goal:
`docs/research_protocols/nl_prefix_latent_episode_narrative_retrieval_goal.json`.

Method intake:
`docs/research_protocols/nl_prefix_latent_episode_narrative_retrieval_method_intake.md`.

The branch must preserve the incumbent nearest-similar top3/90 workflow for
tomorrow's presentation and for current paper/demo artifacts. It may create new
`nl_episode_narrative_*` scripts and tests, but it must not edit the existing
caption generator, Codex caption batcher, narrative pipeline,
component-posterior bakeoff, paper, or demo defaults until a promotion gate and
independent verifier support the change.

### Full Direct-Codex 982g Corpus and Retrieval Result

Main artifacts:

- Final Codex corpus report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_multiformat_982g_sharded/final/final_codex_corpus_report.json`
- Final retrieval cards:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_multiformat_982g_sharded/final/multiformat_episode_cards.jsonl`
- Local text-to-text retrieval screen:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_multiformat_982g_local_retrieval/local_retrieval_report.json`
- OpenAI embedding hybrid bridge:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_embedding_bridge_hybrid_66q/hybrid_embedding_start_bridge_report.json`
- OpenAI embedding hybrid scenario eval:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_scenario_embedding_hybrid_66q_s16/scenario_level_eval_report.json`
- OpenAI embedding hybrid conditionality lift:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_lift_embedding_hybrid_vs_start/conditionality_lift_report.json`

Corpus result:

- expected windows: `4010`;
- generated narrative records/cards: `4010 / 4010`;
- missing windows: `0`;
- validation errors: `0`;
- invalid cards: `0`;
- validation warnings: `21`;
- retrieval guard passed: `true`;
- authoring route: `direct_codex_multiformat_gpt_5_5_xhigh`;
- local template prose used: `false`.

Matched 66-window frozen-SNI scenario results at 16 samples/support component:

- start-only: CRPS `0.506351`, Energy `0.660344`, coverage `0.672391`;
- pure local episode-text retrieval: CRPS `0.549540`, Energy `0.717667`,
  coverage `0.679461`;
- local 25/75 hybrid: CRPS `0.528387`, Energy `0.685665`, coverage `0.656773`;
- OpenAI `text-embedding-3-large` 25/75 hybrid: CRPS `0.525272`, Energy
  `0.682662`, coverage `0.653212`.

Conditionality lift versus start-only:

- pure episode-text: `conditionality_lift_detected`, terminal factor KS
  `0.305742`, path energy distance `0.221377`, CRPS delta versus start-only
  `0.043189`;
- local 25/75 hybrid: `conditionality_lift_detected`, terminal factor KS
  `0.194558`, path energy distance `0.090675`, CRPS delta `0.022036`;
- OpenAI 25/75 hybrid: `conditionality_lift_detected`, terminal factor KS
  `0.189483`, path energy distance `0.089874`, CRPS delta `0.018921`.

Stage-2 grounded text-preference reranker `984a`:

- Reranker report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_grounded_text_preference_reranker_984a/grounded_text_preference_reranker_report.json`
- Reranked bridge report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_grounded_text_preference_reranker_984a/grounded_text_preference_bridge_report.json`
- Scenario eval:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_scenario_grounded_text_preference_984a_s16/scenario_level_eval_report.json`
- Conditionality lift:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_lift_grounded_text_preference_984a_vs_start/conditionality_lift_report.json`

Result: `984a` is the strongest current text-space research candidate, but not
yet a paper/demo default. It keeps the OpenAI text-embedding retrieval pool and
grounding checks, then learns a support-preference reranker from historical
next-30-day replay closeness before applying the same top3/90 assembly. The
replay screen found a positive mechanism over 1,024 train queries / 8,078
candidate rows: CRPS delta `+0.028768` and Energy delta `+0.040435` versus the
unreranked grounded text pool, with coverage delta `-0.008379`.

At 66 heldout windows and 16 samples/support component, the frozen-SNI rollout
has CRPS improvement versus persistence `+0.219440`, Energy improvement
`+0.251990`, and coverage `0.649767`. This beats the prior grounded text pool
on CRPS/Energy (`+0.186184` / `+0.214236`) and projected-memory+grounding on
CRPS/Energy (`+0.202889` / `+0.241549`). It is slightly better than the
OpenAI 25/75 hybrid on CRPS (`+0.219440` versus `+0.214008`) and slightly
worse on Energy (`+0.251990` versus `+0.256225`) and coverage (`0.649767`
versus `0.653212`).

Conditionality versus start-only is stronger than the OpenAI 25/75 hybrid:
`conditionality_lift_detected`, terminal factor KS `0.274038`, path energy
distance `0.181869`, support Jaccard distance `0.990909`, and mean terminal
shift `0.271840` standardized units. Absolute historical quality is still
slightly worse than start-only on this split: CRPS delta `+0.015291` and Energy
delta `+0.026206` versus start-only. This remains within the branch guardrail
but means `984a` should be treated as a research candidate pending a verifier
and higher-sample/multistart confirmation, not silently promoted.

Decision: 982g proves that direct Codex-authored narrative-to-narrative support
retrieval adds measurable fixed-start conditionality beyond start-only support
selection. It does not yet justify changing the paper/demo default. Start-only
and the incumbent top3/90 workflow remain stronger on historical fidelity.
Episode-level text retrieval should continue as a research branch for improving
support provenance and conditionality, not as the silent production default.

### Current Episode-Retrieval Evidence

The 970 Phase 0-2 local episode-retrieval TestFlight found a viable mechanism
but did not promote a new default. Broad raw-history episode cards gave 3,944
train support cards and a 66-window decoder-test bridge report. The local
hybrid text-to-text retriever produced historically reasonable frozen-SNI
scenario distributions: CRPS improved `13.3%` and Energy improved `15.4%`
versus persistence, close to the same-split true-history SNI oracle.

The 971 Phase 3 audit sharpened the objective around the user's product
question: does the narrative add scenario movement beyond the same accepted
start? The matched start-only baseline selects supports only by terminal-state
compatibility with the accepted start and uses no narrative text.

Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_conditionality_lift_971c_vs_start_only/conditionality_lift_report.json`.

Result: `conditionality_lift_detected_quality_warning`. The episode-level
narrative condition changes support selection almost completely relative to
start-only (`mean support Jaccard 0.006`) and produces material generated-path
separation (`mean terminal factor KS 0.365`, path energy distance 0.340, mean
terminal mean shift 0.360 standardized units). This answers the immediate
conditionality question: the narrative is not invisible after fixing the start.

However, the pure local text-to-text selector is not promotable because its
historical quality guardrail is weaker than start-only on the same 66 windows:
CRPS `0.579` versus start-only `0.527`, and Energy `0.776` versus start-only
`0.693`. The next method should therefore be a hybrid or agentic-reranked
selector that preserves start-level compatibility while adding narrative lift.
Codex/agentic qualitative adjudication is allowed over a shortlist if pure
semantic similarity is not strong enough, but it must emit auditable structured
match rationales and remain subject to deterministic support checks and
scenario guardrails.

The 971 deterministic hybrid follow-up adds start-fit reranking after local
narrative-to-narrative recall. It uses:

```text
combined score = text_weight * narrative_match
               + start_weight * terminal_start_match
```

Grid artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_selection_971m/hybrid_start_text_selection_report.json`.

Result: all tested hybrid balances pass `conditionality_lift_detected` against
the same start-only baseline. The recommended research candidate is
`text_weight=0.25`, `start_weight=0.75`: CRPS `0.541`, Energy `0.710`, mean
support Jaccard versus start-only `0.042`, mean terminal factor KS `0.292`, and
path energy distance `0.133`. This is materially better than pure text
retrieval on quality while still adding fixed-start narrative lift. It is a
candidate for higher-sample confirmation, not a promoted demo/paper default.

Higher-sample confirmation artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_confirmation_971q/hybrid_start_text_confirmation_report.json`.

Result: the same `text_weight=0.25`, `start_weight=0.75` candidate remains
`conditionality_lift_detected` at 16 samples per support component. Hybrid
quality is close to start-only but not identical: CRPS `0.519` versus start-only
`0.506`, Energy `0.675` versus start-only `0.660`, and coverage `0.635` versus
start-only `0.672`. Narrative lift remains measurable: mean support Jaccard
versus start-only `0.042`, mean terminal factor KS `0.213`, and path energy
distance `0.075`. This confirms the deterministic hybrid as the current
episode-retrieval research candidate pending verifier review and optional
semantic spot checks. It still does not change the paper/demo default.

Verifier:
`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-06-01_episode_retrieval_hybrid_971.md`.
Verdict `PARTIAL`: the narrow lift claim is supported, but public-default
promotion is not. Start-only remains better on CRPS, Energy, and coverage, and
the next gate should inspect qualitative support-match correctness before using
Codex/agentic reranking or changing public artifacts.

Qualitative support-match spot check:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_support_spotcheck_971r/support_match_spotcheck.json`.
Across 66 decoder-test queries, the hybrid selector has title-match top1 share
`0.894`, title-match top3 share `0.970`, archetype-match top1 share `0.924`,
and archetype-match top3 share `0.985`. This is semantically plausible enough
to continue deterministic development without immediate full online enrichment.
Codex/agentic reranking remains useful for borderline cases, such as a
defensive risk-off query selecting a safe-haven financial-accident support.

Retracted 981 local EpisodeCardV3 regeneration and true-embedding comparison:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_retrieval_evaluation_981q/episode_card_v3_full_retrieval_evaluation_summary.json`.

Result: the isolated 981 branch generated `4,010` EpisodeCardV3 cards with
`30,034` narrative records: `268` rich institutional 15-trading-day stride
cards and `3,742` sparse daily cards. Leakage pass share is `1.0`. The
corrected scenario-level evaluator uses the full 4,010-window train support
bank through `--eval_split train --val_size 441 --test_start 4511
--max_windows 4010`.

Retraction: these cards were produced by local deterministic/template-style
EpisodeCardV3 prose generation, not by direct Codex/GPT authoring of each
searchable narrative view. Therefore all 981 retrieval, embedding, and
scenario-level conclusions are invalid for final training/retrieval
assessment, smoke/mechanism evidence, paper claims, demo claims, or default
promotion. The results may only be used as negative evidence for why the
guardrail exists.

The local `text_weight=0.25`, `start_weight=0.75` hybrid appeared to pass
`conditionality_lift_detected` versus the same start-only baseline with mean
terminal factor KS `0.276`, path energy distance `0.158`, and mean terminal
mean shift `0.266`, while keeping guardrail regression modest: CRPS `0.528`
versus start-only `0.508`, Energy `0.689` versus start-only `0.664`, and 80%
coverage `0.670` versus start-only `0.677`. This is now historical invalid
evidence because the underlying narrative corpus was not valid.

The same-index projected-memory adapter baseline also passes lift, with
terminal KS `0.324` and path energy `0.226`, but it is a weaker product
candidate: CRPS `0.541`, Energy `0.699`, 80% coverage `0.656`, title top1
support match only `0.106`, and title top3 support match `0.424`. This supports
the original concern that compressing the narrative into a single projected
memory can move distributions while selecting semantically weak supports.

Pure local text retrieval, projected-memory adapter, and true OpenAI
`text-embedding-3-large` comparisons from 981 are also invalid as final
evidence for the same reason: they consumed the local deterministic/template
EpisodeCardV3 corpus.

Decision: do not change paper/demo defaults, and do not keep a 981 branch
incumbent. Redo the full corpus using direct Codex/GPT-authored multi-format
narratives before running retrieval, true embeddings, historical backtests, or
fixed-start conditionality audits.

## Previous HEAD Objective: Public Paper Cleanup

The previous autoresearch goal revised the natural-language conditioned
scenario-generator manuscript into a concise public technical paper while
preserving the nearest-similar top3/90 support-posterior workflow.

Tracked plan:
`docs/research_protocols/nl_prefix_latent_public_paper_cleanup_plan.md`.

## Previous HEAD Objective: Safe-Haven Gold Mechanism Audit

The previous autoresearch goal was to explain why the Safe-haven gold narrative
can pass the Gold-up grounding/support checks while the day-30 Gold terminal
distribution remains close to the same-start baseline. The mechanism audit
reused the current professional top3/90 artifacts for starts 18, 22, and 40:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_966b_professional_start18_s384_d400`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_966a_professional_start22_s384_d400`,
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_966c_professional_start40_s384_d400`.

Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/safe_haven_gold_mechanism_audit_966d/safe_haven_gold_mechanism_audit.json`.

Result: the weak Gold terminal response is not a table/index bug, not unique to
start 22, and not simply a failure to select Gold-up prefixes. In the broad
support bank, 564 historical windows satisfy the Safe-haven prefix pattern
`Gold up, US10Y down, VIX up, SPX down`, but their next-30-day Gold terminal
distribution is only mildly positive: mean about `+5.62` Gold points, median
about `+4.30`, with only `53.9%` up futures. Across all 4,010 support-bank
windows, Gold prefix delta and next-30-day Gold terminal delta have weak
negative correlation around `-0.17`; within the Safe-haven prefix subset the
correlation is around `-0.37`.

The selected Safe-haven and start-only support components also show the same
mechanism. Across 18 selected top3 components from starts 18, 22, and 40,
`94.4%` have Gold-up prefixes, but the realized next-30-day Gold future is
mixed (`61.1%` up, median about `+5.8`, p10/p90 about `-63.9`/`+160`). The
frozen SNI rollout shrinks these component-specific historical outcomes toward
a much tighter generated terminal Gold distribution: generated component means
have median about `+3.6` and p10/p90 about `-2.2`/`+8.3`. Realized support
future Gold deltas and generated component Gold means have low correlation
around `0.16`.

Interpretation: the current grounding/support check is doing what it is meant
to do: it verifies that the current/recent conditioning prefix has the
Safe-haven Gold pattern. It does not impose a terminal Gold forecast. The
start-only baseline often already selects Gold-up or safe-haven-like prefixes,
and historical Gold-up safe-haven prefixes are not a strong predictor of
further Gold upside. Public-facing claims should therefore describe Safe-haven
Gold as prefix-supported and baseline-relative, not as a guarantee that Gold
itself must separate at day 30. A stronger Gold-facing product claim would
require an explicitly tested response-aware selector/readout for the hedge
channel, with CRPS/energy and calibration guardrails.

## Previous HEAD Objective: Safe-Haven Gold Start Sensitivity Audit

The previous autoresearch goal was to diagnose whether the weak terminal Gold
response in the Safe-haven gold narrative is caused by the accepted starting
level. The audit reuses the current professional top3/90 artifacts for starts
18, 22, and 40.

Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/safe_haven_gold_start_sensitivity_966c/safe_haven_gold_start_sensitivity_audit.json`.

Result: the weak Gold terminal response is not unique to start 22. Across starts
18, 22, and 40, the narrative-vs-start-only Gold mean-delta difference is small:
about `-0.80`, `-0.10`, and `-0.35` Gold points, respectively. The maximum
absolute path-up probability difference is only about `2.1` percentage points.
The narrative and baseline support pools both often contain Gold-up prefixes,
so the Safe-haven narrative can pass the current/recent Gold support check while
adding little incremental day-30 Gold terminal response above the same-start
baseline.

Interpretation: changing the accepted starting level within the current
professional top3/90 artifact set does not solve the Gold terminal-neutrality
issue. This points to the current frozen-generator/support response, not a
single bad start level. Public-facing claims should continue to say that
Safe-haven Gold is prefix-supported and baseline-relative, while avoiding the
claim that Gold itself must separate at terminal horizon.

## Previous HEAD Objective: Safe-Haven Gold Channel Audit

The previous autoresearch goal was a narrow factor-channel audit prompted by
the fixed-start appendix table for Safe-haven gold. The question is why the
story says gold is supported while the day-30 Gold row is effectively unchanged
relative to the start-only baseline. The underlying production candidate
remains nearest-similar top3/90 with component-preserving frozen SNI rollout.
This is a reproduction and interpretation audit; it must not change support
selection, scenario generation, the top3/90 posterior ensemble, or the running
demo.

Reproduction artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/safe_haven_gold_channel_audit_966b/safe_haven_gold_channel_audit.json`.
Verifier:
`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-29_safe_haven_gold_channel_audit_966b.md`.

Result: no evidence of a table, factor-index, or top3/90 selection bug. The
Safe-haven narrative extracts a high-confidence `GOLD up` current/recent claim
and the selected support prefixes pass the Gold-up direction check. However,
the frozen SNI rollout from the same accepted start produces a day-30 Gold
terminal distribution that is baseline-neutral: narrative top3/90 Gold mean
delta is about `+4.98` points with `57.7%` up paths, while the start-only
top3/90 baseline is about `+5.08` points with `59.3%` up paths.

Interpretation: grounding/support checks validate the current/recent prefix,
not a requested terminal Gold forecast. For this fixed start, Gold is
prefix-supported but terminal-neutral relative to the start-only baseline. The
paper/demo should describe this case as a defensive support regime whose
baseline-relative impact appears more clearly in equity, credit, FX, or
portfolio channels, not as evidence that every named factor must move in the
story direction at day 30.

## Previous HEAD Objective: Fixed-Start Casebook Appendix

The previous autoresearch goal was to make the paper's fixed-start casebook
match the demo's baseline-vs-narrative readout. The underlying production
candidate remained nearest-similar top3/90 with component-preserving frozen SNI
rollout. The paper should show, in appendix form, one factor-vertical table per
professional narrative at the same accepted start:

```text
baseline view / path count / mean move
vs.
narrative view / path count / mean move
plus 30d change versus the start-only baseline
```

This is a presentation and reproducibility task. It must not change support
selection, scenario generation, the top3/90 posterior ensemble, or the running
demo. The tables should be generated from the saved professional start-22
top3/90 artifacts:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_966a_professional_start22_s384_d400`.

The previous confidence-calibration task produced the demo table format now
used here: `Baseline View`, `Baseline Path Count`, `Baseline Mean Move`,
`Narrative View`, `Narrative Path Count`, `Narrative Mean Move`, and `30d
Change vs Baseline`.

## Previous HEAD Objective: Grounding Reliability Audit

The previous autoresearch goal was to measure the grounding sidecar as a
product trust surface. Grounding is not trained by historical backtest and is
not the scenario generator. It is an LLM interpretation layer that extracts
current/recent market implications, warning-only future language, unsupported
claims, and evidence snippets from the narrative. Because this table is visible
in the demo and used for support-direction checks, it needs its own audit.

The audit should measure:

- narrative faithfulness: extracted claims are supported by quoted/evidenced
  narrative snippets;
- temporal discipline: future-looking language is warning-only and does not
  enter current/recent conditioning claims;
- historical direction agreement: for professional captions generated from
  known historical prefixes, grounded directions agree with observed prefix
  motion in mapped markets;
- support-direction consistency: selected support regimes and the top3/90
  posterior support pool satisfy high-confidence grounding directions, or emit
  visible warnings when they do not.

Tracked intake:
`docs/research_protocols/nl_prefix_latent_grounding_reliability_audit_intake.md`.

The grounding audit is separate from CRPS/energy scenario-quality backtesting.
Scenario backtests validate the generated distribution. Grounding reliability
validates the story-to-claims translation and warning discipline.

Latest audit artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_grounding_reliability_audit_965a_66case/grounding_reliability_audit.json`.
The 965a audit combines six live professional narratives with support-direction
checks and 66 historical-prefix captions with known realized prefix directions:
72 cases, 651 extracted claims, 100.0% claim faithfulness, 100.0%
future-language detection, 0.0% future-language leakage, 100.0% historical
direction agreement over 607 checked directions, 94.0% historical direction
coverage, and 100.0% support-direction pass rate. Independent verifier:
`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-28_grounding_reliability_audit_965a.md`.
Interpretation: this is a structured reliability sanity check for the visible
grounding layer, not a human-labeled semantic benchmark and not a scenario
quality test.

Scenario-to-text training captions now have a separate risk-manager quality
gate. Every narrative-generation or captioning change must check both specialist
Word documents under `research/narrative_specialist/`, include their paths and
hashes in artifacts, and pass a small TestFlight before scale:

- `research/narrative_specialist/quant generated scenarios story narrative.docx`
- `research/narrative_specialist/quant generated scenarios story narrative 2.docx`

The required caption shape is: scenario title, mechanical summary,
archetype/regime, trigger, transmission channel, cross-asset reaction,
sequencing, portfolio/risk implication, evidence used, ambiguity, contrastive
hard negatives, and a no-forecast caveat. Conditioning captions must describe
only the current/recent historical prefix. Realized future paths, generated
scenario outputs, terminal values, VaR/ES, target P&L, and post-horizon facts
are leakage.

## Current Promotion Status

The active posterior-ensemble family is **Nearest similar regimes / Main-regime
view**. The current promoted candidate is the three-start high-sample
confirmation of **Main-regime view: top 3 / 90%**:

```text
cohesive_topk_narrative_start_checked
-> nearest-similar, direction-checked support regimes
-> component posterior = top3_90 for product-facing presentation
-> component posterior = top2_80 as the sharper KS-oriented variant
-> component-preserving frozen SNI rollout
```

This posterior family remains useful because it exposes the observed tradeoff
between risk-manager-visible conditionality and historical distributional
quality:

- Selection report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_selection_961b_29w/posterior_ensemble_selection_report.json`.
- 29-window historical backtest for the selected candidate: CRPS improvement
  `+0.172`, energy improvement `+0.269`, and 80% coverage `0.746` versus
  persistence.
- Fixed-start conditionality for the selected candidate: factor KS `0.524`,
  portfolio KS `0.572`, path energy `66.244`, and VaR95 range `5.275`.
- The same candidate ranked first in the 24-window screen and again in the
  29-window comparable selection run.
- High-sample confirmation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962c_selection_confirmation/posterior_ensemble_selection_report.json`.
  With 384 fixed-start presentation samples and 64 held-out backtest samples,
  the low-sample top3/90 conditionality estimate was not confirmed. The
  confirmation ranks nearest-similar top2/80 first: CRPS improvement `+0.189`,
  energy improvement `+0.281`, coverage `0.778`, factor KS `0.187`, and
  portfolio KS `0.177`. Top3/90 remains better calibrated than top2/80, with
  CRPS improvement `+0.207`, energy improvement `+0.295`, coverage `0.807`,
  factor KS `0.165`, and portfolio KS `0.149`.
- Three-start high-sample confirmation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963c_multistart_confirmation/posterior_ensemble_multistart_confirmation.json`.
  Across starts `18`, `22`, and `40`, the current weighted tradeoff again ranks
  nearest-similar top3/90 first: CRPS improvement `+0.207`, energy improvement
  `+0.295`, coverage `0.807`, mean factor KS `0.163`, mean portfolio KS
  `0.134`, mean path energy `10.109`, mean VaR95 range `3.046`, and score
  `0.800`. Top2/80 is second: CRPS improvement `+0.189`, energy improvement
  `+0.281`, coverage `0.778`, mean factor KS `0.177`, mean portfolio KS
  `0.147`, mean path energy `10.566`, mean VaR95 range `2.288`, and score
  `0.753`.

All-regime views remain stronger on pure calibration metrics, but they visibly
smooth away narrative-specific scenario families. Therefore all-regime pooling
is now a calibration diagnostic. The paper/demo presentation should report the
top3/90 versus top2/80 tradeoff explicitly: top3/90 is the current multistart
tradeoff candidate because it has stronger calibration and larger portfolio-tail
spread, while top2/80 remains the sharper KS-oriented variant.

Independent verifiers:

- `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-27_posterior_ensemble_selection_961b.md`
- `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-27_posterior_ensemble_confirmation_962c.md`
- `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-27_posterior_ensemble_multistart_confirmation_963c.md`

Verifier verdict for the earlier 961b selector: `PARTIAL / PROCEED AS CURRENT
PRODUCTION CANDIDATE`. That verdict is now superseded by the 962 high-sample
confirmation and the 962c verifier. The posterior-ensemble family remains valid,
but the single-start exact top3/90 promotion claim was not supported as the
single paper/demo default. The 963c multistart confirmation is the current
reversal evidence. The 963c verifier verdict is `AGREE`: promote top3/90 as
the current paper/demo candidate under the explicit tradeoff framing, keep
top2/80 as the sharper KS-oriented variant, and keep all-regime pooling as the
calibration diagnostic. This is not a production-ready or conditionality-solved
claim.

The older broad diverse support mixture, response-preview methods, and
support-gated calibration branches remain diagnostics or historical baselines
unless a later verifier promotes them again.

Superseded broad-support evidence: the 947b matched broad-support calibration
deck remains useful as historical calibration evidence, but it is no longer the
public-facing paper/demo default. Paper and demo surfaces should now present the
verified nearest-similar top3/90 candidate. Broad all-regime or all-pooling
language is allowed only when explicitly labeled as a calibration diagnostic,
appendix tradeoff, or superseded historical baseline.

Live demo calibration wiring: the Gradio story demo now applies the 947b-style
bounded support-gated directional calibration after the support-grounded frozen
SNI rollout, only when rollout arrays are present, the direction support gate
passes, and the grounding sidecar contains active current/recent directional
claims. The layer is deliberately report-visible rather than hidden:
`generation.narrative_ensemble_calibration` records whether calibration was
applied, the effective beta, the support gate, the operational variant index,
and the summary scope; the markdown report receives a `Live Demo Narrative
Calibration` section. This is a demo/paper candidate over the incumbent
support-grounded rollout, not a silent production default. Fresh live smoke:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_story_gradio_demo/prefix_latent_live_smoke/condition_only_run/prefix_latent_story_smoke_report.json`.
Cached-condition markdown smoke:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_story_gradio_demo/prefix_latent_live_smoke/cached_casebook_run/condition_only_report/prefix_latent_story_smoke_report.md`.
Fixed-start live story-deck sweep:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/fixed_start22_calibrated_story_deck_conditionality_summary.json`.
This run sends the six default professional narratives through the public
Gradio API route at the same explicit historical start `22`; all `6/6` cases
passed, calibration applied in `6/6`, the minimum calibration support gate was
`1.0`, per-case report/array snapshots were preserved, and pairwise support
Jaccard was `0.0`. This strengthens the live-demo support-conditionality and
auditability claim, but it remains a demo-path validation rather than a full
production promotion. Reproducible analysis script:
`experiments/backfill/block_ar/nl_live_story_deck_analysis.py`. Terminal-delta
panel:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/fixed_start22_terminal_mean_deltas.png`.

Combined production-readiness audit:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_production_readiness_audit_949a/production_readiness_audit.json`.
This audit combines the 947b held-out/five-start calibration evidence with the
948d live fixed-start story-deck evidence. The result is
`paper_demo_candidate`, with no failed hard gates and `goal_complete=false`.
Headline metrics: CRPS improvement vs persistence `0.2169`, energy improvement
`0.3062`, 80% coverage `0.8226`, start-normalized narrative plus interaction
`0.5239`, fixed-start factor KS `0.3283`, fixed-start portfolio KS `0.4110`,
and live max support Jaccard `0.0`. The only warning gate is
`production_default`, because the method remains a bounded post-rollout
support-gated calibration layer and needs broader UX/product validation before
being treated as a silent production default.

Broad live multi-start UX sweep:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_multistart_950c_5start_default_deck_warning_aware/multi_start_live_story_deck_summary.json`.
This run sends the six default professional narratives through the public
Gradio API route at starts `0`, `18`, `22`, `40`, and `77`. All `30/30` cases
passed, calibration applied in `30/30`, the minimum calibration support gate was
`1.0`, and the maximum within-start pairwise support Jaccard was `0.0` for all
five starts. Start `0` produced six selected-start warnings because its fixed
level is far from the selected support, but those warnings were recorded rather
than hidden. This strengthens the live UX evidence that narrative support
selection does not collapse to one start-driven support set, while preserving
the current caveat that raw-level path geometry is still strongly shaped by the
accepted starting level.

Latest support-coherence/component-posterior bakeoff:

- Fixed-start smoke root:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/support_cohesion_component_posterior_bakeoff_960a_smoke_s8_start22`.
  The start-only null stayed flat under all posterior readouts. Full pooled
  incumbent separation was factor KS `0.263`, portfolio KS `0.225`, and VaR95
  range `2.473`. Cluster-family full pooling increased portfolio KS to `0.350`
  and VaR95 range to `13.017`; sparse component-posterior readouts exposed
  substantially stronger separation, with cluster-family `top3_90` at factor KS
  `0.425` and portfolio KS `0.424`.
- Component-posterior plot:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/support_cohesion_component_posterior_bakeoff_960a_smoke_s8_start22/component_posterior_factor_fans.png`.
  This is diagnostic/product evidence for component-aware views, not a silent
  default replacement.
- 24-window guardrails:
  incumbent CRPS/energy improvements `+0.214250` / `+0.304095`; cohesive
  `+0.213287` / `+0.303971`; cluster-family `+0.212316` / `+0.305110`;
  low-temperature kernel `+0.215874` / `+0.306683`.
- Comparable 29-scored-window cluster-family run:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/support_cohesion_component_posterior_bakeoff_960c_guardrail_cluster_66req_s32_d200/component_backtest_report.json`.
  It remains close to the existing incumbent but is a small regression on CRPS,
  coverage, and terminal MAE. Therefore support-coherence selectors are
  candidates, not promoted defaults.

Current decision from the support-coherence bakeoff and high-sample
confirmation: keep **Nearest similar regimes / Main-regime view** as the active
posterior-ensemble family and treat top3/90 as the current multistart
high-sample paper/demo candidate. All-regime pooling remains the calibration
diagnostic, and top2/80 remains the sharper KS-oriented variant.

Immediate paper/demo consistency objective: update the narrative-conditioned
scenario paper and Gradio demo so public-facing defaults consistently use the
nearest-similar top3/90 posterior-ensemble candidate. The paper should present
top3/90 as the current product-facing candidate, with top2/80 and all-regime
pooling only as clearly labeled diagnostics or appendix tradeoffs. The demo
should use the same candidate and explain, in plain language, the professional
narrative, selected starting level, selected historical support regimes, the
top3/90 ensemble rule, and the resulting 30-day scenario fan chart. Remove or
relabel stale all-pooling and old weak-conditionality language from public
surfaces.

Refreshed attribution artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_964a_start_narrative_attribution_top3_90/start_narrative_attribution.json`.
For nearest-similar top3/90 across starts `18`, `22`, and `40`, raw terminal
levels attribute `83.1%` of standardized distribution-summary variation to the
accepted start, `11.1%` to narrative, and `5.7%` to interaction. After
start-normalization, the start share falls to `58.1%`, narrative rises to
`31.8%`, and interaction is `10.1%`. The start-only null remains flat across
narratives under the same start, with same-start factor and portfolio KS both
`0.000`.

Latest demo/paper default cleanup:

- Default fixed-start rollout and paper-facing qualitative evidence now use
  only the incumbent `current_start_checked_gap30` policy and the
  `start_only_topk` null control. Response-preview policies remain available
  in code as diagnostics but are not selected by default.
- Public narrative case ids are clean (`fragile_risk_on`,
  `defensive_risk_off`, `commodity_inflation`, `dollar_liquidity`,
  `rates_selloff`, and `safe_haven_gold`). Legacy condition-report directories
  whose names include an old start id are accepted only as cache aliases; new
  rollout artifacts are written under the clean public ids.
- Clean fixed-start comparison artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_943a_start22_incumbent_clean_s64_d400/fixed_start_rollout_policy_comparison.json`.
  The incumbent has direction checks `6/6`, mean support Jaccard `0.033`,
  mean factor terminal KS `0.169`, mean portfolio terminal KS `0.124`, and
  VaR95 loss range `2.290`. The start-only null has support Jaccard `1.000`
  with zero factor, portfolio, and VaR separation.
- Clean stress-test artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_943a_start22_incumbent_clean/conditionality_stress_test.json`.
  The incumbent passes `6/6` gates; the start-only null passes `1/6`.
- Paper-facing figures now use the clean incumbent artifacts:
  `paper/narrative_grounded_scenarios/figures/fixed_start_rollout_policy_s64_incumbent_factor_fans.png`,
  `paper/narrative_grounded_scenarios/figures/narrative_relevant_factor_panels_943a.png`,
  `paper/narrative_grounded_scenarios/figures/narrative_reference_contrast_panels_943a.png`,
  `paper/narrative_grounded_scenarios/figures/narrative_casebook_fixed_start.png`,
  `paper/narrative_grounded_scenarios/figures/narrative_portfolio_impact_fixed_start.png`,
  `paper/narrative_grounded_scenarios/figures/narrative_portfolio_tail_deltas_943a.png`, and
  `paper/narrative_grounded_scenarios/figures/fixed_start_rollout_policy_s64_incumbent_portfolio_tail.png`.
- Cleanup verifier:
  `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-22_incumbent_cleanup_943a.md`.
  Verdict: `PARTIAL`; cleanup/default/paper-facing evidence is verified, but
  this does not by itself promote the whole product as production-ready.

## Product Conditionality Definition

For a fixed approved starting level, a professional current-market narrative is
conditionally useful only if it changes the auditable support mixture and
produces a distinguishable future risk distribution in the risk channels
implied by that narrative, above repeat, bootstrap, and start-only controls.

The working product gate is:

1. semantic narrative representation: preserve the risk-manager story, not only
   direction labels;
2. support mixture conditionality: different narratives select different
   diverse support mixtures under the same start;
3. decoded-prefix conditionality: those supports become different recent-prefix
   objects before rollout;
4. factor-distribution conditionality: generated factor paths differ beyond
   repeat/bootstrap/start-only controls;
5. portfolio-tail conditionality: VaR/ES-style tails and portfolio P&L views
   change in the expected risk channels;
6. auditability/provenance: reports expose narrative, sidecar, start, support,
   weights, warnings, controls, and artifacts.

Earlier product-conditionality audit:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_product_conditionality_contract_919a/product_conditionality_contract_audit.json`

Verdict: `product_conditionality_partially_supported_with_warnings`. Semantic,
support, decoded-prefix, and auditability gates pass. The remaining warning
gates are `factor_distribution_conditionality` and
`portfolio_tail_conditionality`: within-run bootstrap path energy remains close
to the observed cross-narrative effect, and VaR95-style portfolio-tail
separation is below the same-narrative repeat threshold. That warning drove the
component-preserving rollout and fixed-start stress-test work.

Latest fixed-start stress-test evidence:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_943a_start22_incumbent_clean/conditionality_stress_test.json`

## Previous HEAD Objective: Support-Coherence Component-Posterior Bakeoff

The latest component-aware audit shows that professional narratives select
different support components and that component-level path separation is much
larger than broad pooled fan separation:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/qualified_narrative_broad_support_multistart_956c/start_0_18_22_40_77_samples8/component_aware_scenario_view_audit/component_aware_scenario_view_audit.json`

Promoted fact from the audit:

- broad-support professional deck: 5 starts, 6 narratives per start, 48 support
  components per start;
- max support Jaccard across starts: `0.231`;
- mean component-to-pooled path-energy ratio: `31.09`;
- sparse component-family policies increase terminal p50 separation relative to
  full pooling: top-1 `2.98x`, top-2/80% `1.98x`, top-3/90% `1.74x`.

This changed the previous research target. That loop tested whether weak visual
fan conditionality was caused by pooling heterogeneous support components. The
method intake is:

`docs/research_protocols/nl_prefix_latent_support_cohesion_component_posterior_intake.md`

The required bakeoff crosses support-selection policy with distribution policy:

1. support selection:
   - current broad diverse support;
   - most-similar cohesive support around the top candidate;
   - cluster/family support;
   - low-temperature similarity-kernel support;
2. distribution policy:
   - full pooled distribution;
   - top-1 component posterior;
   - top-2 or 80% sparse posterior;
   - top-3 or 90% sparse posterior.

Promotion requires both sides of the evidence:

- stronger same-start, professional-narrative conditionality in visual fans,
  narrative-relevant factor KS/path energy, and portfolio VaR/ES readouts;
- held-out CRPS, energy, 80% coverage, terminal MAE, direction checks, support
  provenance, and start-only/null controls that remain competitive with the
  incumbent broad pooled support mixture.

Verdict: `fixed_start_component_preserving_conditionality_supported`. The
selected hard-direction start-aware policy passes all six gates: same start,
direction consistency, low support overlap, relevant-factor response,
path-distribution response, and portfolio-tail response. The start-only null
passes only one gate and has zero factor KS, zero path energy, zero portfolio
KS, and zero VaR95 range. Therefore the active bottleneck is no longer "is
there any conditionality?" but "can we make the conditionality stronger and
more useful in the narrative's risk channels without damaging scenario
quality?"

Start-versus-narrative attribution diagnostic:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_narrative_attribution_944a/start_narrative_attribution.json`

Verifier:

`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-25_start_narrative_attribution.md`

Verdict: `factorial_attribution_supported_for_current_evidence`. The crossed
diagnostic uses starts 18, 22, and 40 crossed with the six public professional
narratives. On raw terminal-level distribution features, the accepted start
dominates the visible market geometry: start share `81.3%`, narrative share
`7.8%`, and interaction `10.9%`. On start-normalized terminal-move features,
the narrative channel is material but not dominant: start share `57.6%`,
narrative share `22.3%`, and interaction `20.1%`. Pairwise distribution
distances agree with this interpretation: at a fixed start, narrative changes
produce mean factor KS `0.176` and portfolio KS `0.149`; across starts under
the same narrative, start changes produce larger factor KS `0.566` and
portfolio KS `0.486`. The start-only null assigns `100%` of standardized
feature variation to start and `0%` to narrative, with zero same-start
narrative KS. This confirms that the paper's fixed-start conditionality is not
a start artifact, while also making clear that accepted level selection remains
the larger source of raw-level scenario variation.

Previous method branch: **narrative-conditioned ensemble calibration** over the
incumbent support-grounded SNI ensemble. This remains useful background but is
not the active HEAD objective after the component-aware pooling audit. Its
objective was to keep the same fixed start, incumbent diverse support mixture,
and frozen SNI rollout, then learn a small bounded calibration layer that makes
the generated ensemble respond more clearly in the narrative's relevant factor
and portfolio-tail channels while remaining competitive with the incumbent on
CRPS, energy, coverage, provenance, and direction checks.

Previous intake:

`docs/research_protocols/nl_prefix_latent_narrative_ensemble_calibration_intake.md`

The baseline target from 944a was start-normalized narrative plus interaction
share `42.4%`, fixed-start factor KS `0.176`, and fixed-start portfolio KS
`0.149`. A promotion-quality candidate should improve these start-normalized
and fixed-start narrative-response metrics without making the start-only null
nonzero or degrading held-out scenario quality beyond the stated floor.

Initial narrative-conditioned ensemble-calibration TestFlight:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945a/narrative_ensemble_calibration_report.json`

Verdict: `candidate_not_promoted_initial_testflight`. The first bounded
calibrator keeps support selection and frozen SNI rollout unchanged, then fits a
single narrative-direction beta on a split of incumbent historical backtest
rows. It selected beta `0.25`. On the 14-row evaluation split, calibrated CRPS
and energy were slightly better than identity (`-0.00048` CRPS and `-0.00064`
energy, lower is better), while 80% coverage changed by only `-0.00073`.
Fixed-start attribution improved materially: start-normalized narrative plus
interaction share increased from `42.4%` to `66.4%`, mean factor KS from
`0.176` to `0.326`, and portfolio KS from `0.149` to `0.402`. This is the
first positive candidate for the new objective, but it is not yet a production
default. Promotion still requires repeat/split robustness, independent
verification, and qualitative plot review.

Qualitative TestFlight plot:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945a/fixed_start_calibrated_factor_fans.png`

Robustness follow-up:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945b_odd_even/narrative_ensemble_calibration_report.json`
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945c_chronological/narrative_ensemble_calibration_report.json`

The odd/even split repeats the 945a result: beta `0.25`, CRPS `-0.00056`,
energy `-0.00114`, coverage `-0.00068`, start-normalized narrative plus
interaction share `66.4%`, factor KS `0.326`, and portfolio KS `0.402`. The
chronological split is still quality-positive but more conservative: beta
`0.05`, CRPS `-0.00028`, energy `-0.00048`, coverage `+0.00011`, narrative plus
interaction share `44.5%`, factor KS `0.193`, and portfolio KS `0.167`. This
means the candidate is robust enough to continue, but not strong enough to
promote as a default. The next HEAD task should treat beta selection as a
quality-constrained conditionality objective rather than pure CRPS selection,
then rerun the same split/fixed-start gates.

Quality-constrained response selection:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945e_qcr_even_odd/narrative_ensemble_calibration_report.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945f_qcr_odd_even/narrative_ensemble_calibration_report.json`,
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945g_qcr_chronological/narrative_ensemble_calibration_report.json`

The quality-constrained response rule chooses the largest bounded beta that
stays within the historical quality floors instead of minimizing calibration
CRPS alone. This makes the selection objective match the product objective:
preserve broad scenario quality while increasing narrative response. All three
splits select effective beta `0.25` and pass quality gates. The chronological
split, which had been the weak case, now improves held-out CRPS by `-0.00116`,
energy by `-0.00208`, and changes coverage by only `-0.00040`; fixed-start
start-normalized narrative plus interaction share is `66.4%`, factor KS
`0.326`, and portfolio KS `0.402`. This is now the strongest candidate in the
ensemble-calibration family. It is still not a production default until an
independent verifier audits the method, the artifact paths, and the claim that
this bounded calibration does not turn current/recent narrative implications
into an impermissible future prescription.

Verifier:

`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-25_narrative_ensemble_calibration_945efg.md`

Verifier verdict: `PARTIAL`. The verifier agrees that the 945e/f/g artifacts
support a stronger candidate than pure CRPS beta selection. It does not support
promotion yet. Remaining requirements are start-only/null controls under the
calibration path, repeat or shuffled-narrative controls if available, beta-bound
sensitivity, and qualitative raw-level fan inspection.

Support-gated calibration follow-up:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945k_qcr_support_gated_even_odd/narrative_ensemble_calibration_report.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945l_qcr_support_gated_odd_even/narrative_ensemble_calibration_report.json`,
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945j_qcr_support_gated_current_chronological/narrative_ensemble_calibration_report.json`

The support-gated variant keeps the quality-constrained response selector but
blocks calibration when the support prior is the explicit start-only mode or
when the support direction check rejects the narrative. This directly addresses
the verifier's product-contract warning: calibration is now conditional on
accepted narrative support, not merely on the presence of a directional phrase.
Across even/odd, odd/even, and chronological splits it keeps beta `0.25`,
passes quality gates, and retains fixed-start metrics: narrative plus
interaction share `66.4%`, factor KS `0.326`, portfolio KS `0.402`, and support
Jaccard `0.032`. The explicit start-only null artifact
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945i_qcr_support_gated_start_only_null/narrative_ensemble_calibration_report.json`
is flat again: narrative plus interaction share approximately `0`, factor KS
`0`, portfolio KS `0`, and support Jaccard `1.000`.

Beta-bound sensitivity on the chronological split:

- beta bound `0.15`:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945m_qcr_support_gated_bound015/narrative_ensemble_calibration_report.json`
  gives CRPS `-0.00077`, energy `-0.00134`, coverage `+0.00006`,
  narrative plus interaction share `56.6%`, factor KS `0.263`, portfolio KS
  `0.294`.
- beta bound `0.25`: current conservative candidate, CRPS `-0.00116`, energy
  `-0.00208`, coverage `-0.00040`, narrative plus interaction share `66.4%`,
  factor KS `0.326`, portfolio KS `0.402`.
- beta bound `0.35`:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945n_qcr_support_gated_bound035/narrative_ensemble_calibration_report.json`
  gives CRPS `-0.00145`, energy `-0.00269`, coverage `-0.00028`,
  narrative plus interaction share `72.6%`, factor KS `0.374`, portfolio KS
  `0.482`.

The higher bound is stronger on the current diagnostics, but the current truth
should keep `0.25` as the conservative candidate until qualitative review and a
second verifier pass confirm that the stronger response is still risk-manager
plausible rather than visually over-directed.

Support-gated qualitative review and second verifier:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/narrative_ensemble_calibration_report.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/support_gated_qualitative_review.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/support_gated_narrative_relevant_raw_panels.png`,
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/support_gated_start_only_null_contrasts.png`

The qualitative review keeps beta `0.25` and the chronological held-out quality
gains from 945j: CRPS `-0.00116`, energy `-0.00208`, and coverage `-0.00040`
versus the identity incumbent. It adds raw-level narrative-relevant panels and
a start-only null contrast. At the same approved start, the candidate produces
visible terminal-median responses in the markets each story names: fragile
risk-on SPX `+18.74` and VIX `-1.73` versus start-only null, defensive risk-off
SPX `-8.67`, VIX `+1.20`, BBB OAS `+0.16`, commodity inflation crude `+1.25`
and US10Y `+0.11`, dollar liquidity DXY `+0.92`, BBB OAS `+0.14`, VIX `+1.23`,
rates selloff US10Y `+0.10`, and safe-haven gold gold `+9.27`, US10Y `-0.06`,
VIX `+1.39`. The same-start fixed-start attribution remains materially above
the start-only null: narrative plus interaction `66.4%`, factor KS `0.326`,
portfolio KS `0.402`, and support Jaccard `0.032`.

Verifier artifact:

`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-25_support_gated_calibration_945o.md`

Verifier verdict: `PARTIAL`. The verifier accepts `945o` as a legitimate
paper/demo candidate for risk-manager-visible support-gated narrative response,
but not yet as a silent production default. The claim must stay scoped as a
bounded calibration layer over accepted historical support, not a solved direct
text-to-latent bridge and not an LLM forecast. Required before default
promotion: broader replication or explicit product-owner promotion under the
same support-gated null-control contract.

Per-start promotion-gate addendum:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945q_per_start_promotion_gates/narrative_ensemble_calibration_report.json`

The 945q rerun adds explicit per-start narrative gates to avoid hiding a weak
start inside an aggregate attribution score. The current support-gated
candidate passes all added gates. Across the three accepted starts, the minimum
same-start narrative factor KS is `0.316`, the minimum same-start narrative
portfolio KS is `0.393`, and the maximum same-start support Jaccard is only
`0.033`. Per-start rows are: start18 factor KS `0.320`, portfolio KS `0.398`;
start22 factor KS `0.316`, portfolio KS `0.393`; start40 factor KS `0.342`,
portfolio KS `0.416`. This makes the candidate stronger than the prior
candidate-only state, but the default-promotion decision should still remain
explicit because the mechanism is a bounded post-rollout calibration overlay.

Broad-support calibration-fit replication:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945r_broad_support_backtest/narrative_ensemble_calibration_report.json`

The 945r addendum ran the same support-gated quality-constrained calibration
fit/evaluation on the broad-support 940d component-backtest artifact. It again
selects beta `0.25` and improves held-out quality versus identity: CRPS
`-0.00115`, energy `-0.00209`, and coverage `+0.00040`. This supports the
calibration quality side of the candidate under a broader support report. It
does not add a new broad-support fixed-start visual deck; the qualitative
raw-level panels were still the 945q fixed-start review at that point. This
surface is superseded by the 946d matched full-window broad-support evidence
below.

Verifier addendum:

`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-25_support_gated_calibration_945q_r_addendum.md`

Verdict: `PARTIAL`. The addendum accepted 945q-r as paper/demo candidate
evidence under careful framing at that time. It warned that the 940d `66w`
reports contain `29` usable `window_scores`, so do not call that artifact a
66-window calibration evaluation. This caveat is superseded for the current
paper/demo surface by 946d, which uses a true `66/66` full906b held-out
component backtest.

Paper/demo evidence refresh:

`paper/narrative_grounded_scenarios/generated_tables/table_support_gated_calibration.tex`

`paper/narrative_grounded_scenarios/figures/support_gated_narrative_relevant_raw_panels_947b.png`

`paper/narrative_grounded_scenarios/figures/support_gated_start_only_null_contrasts_947b.png`

The paper surface now uses the 947b matched five-start broad-support evidence
rather than the older weak-conditionality panels, the mixed 945q-r surface, the
smaller 946a matched deck, or the three-start 946d surface. 946d first rebuilt
the broad component backtest on the full906b bridge-evaluation manifest,
scoring `66/66` held-out anchors. 947b keeps that same held-out quality report
and reruns the support-gated calibration against matched broad-support
fixed-start decks for starts 0, 18, 22, 40, and 77 under one seed/settings deck.
It selects beta `0.25`; held-out deltas versus identity are CRPS `-0.00181`,
energy `-0.00289`, and coverage `+0.00101`; matched broad fixed-start
attribution is factor KS `0.328`, portfolio KS `0.411`, and support Jaccard
`0.030`; the weakest per-start floor is factor KS `0.311`, portfolio KS
`0.373`, and support Jaccard `0.032`. This strengthens the paper/demo claim
because calibration and visual support no longer come from different support
universes, the held-out backtest uses all 66 requested full906b windows, and
the fixed-start acceptance deck now spans five starts. It remains a paper/demo
candidate surface, not a silent production default, because the method is still
a bounded post-rollout support-gated calibration layer over the frozen
support-grounded SNI ensemble.

Verifier:

`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-25_matched_broad_support_calibration_947b_5start.md`

Verdict: `PARTIAL`. The verifier accepts 947b as the current paper/demo
candidate because it preserves the 66/66 full906b held-out quality check while
extending the matched fixed-start evidence to five starts. It keeps the same
non-production caveat: this is support-gated calibration over the frozen
support-grounded SNI ensemble, not a direct text-to-scenario model and not a
silent production default.

Previous verifier:

`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-25_matched_broad_support_calibration_946d.md`

Verdict: `PARTIAL`. The verifier accepted 946d as the current paper/demo
candidate because it fixes the mixed-evidence issue in 945q-r, removes the
smaller 29-window broad-report caveat from 946a, and supports visible
fixed-start narrative response under matched broad support. The same verifier
keeps the non-production caveat: this is a bounded post-rollout support-gated
calibration layer over the frozen support-grounded SNI ensemble. This verifier
is superseded by the 947b five-start verifier above.

Initial response-aware TestFlight:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_response_aware_support_weighting_934a/response_aware_support_weighting_report.json`

Verdict: `candidate_testflight_promising_not_promoted`. The offline 934a
candidate reweights cached component-preserving rollout samples by the
generator response in narrative-relevant channels. It improves the fixed-start
conditionality scorecard versus the current component baseline: relevant-factor
KS rises from `0.1775` to `0.2127`, path energy from `0.0431` to `0.0489`,
portfolio KS from `0.1179` to `0.1462`, and VaR95 range from `10.339` to
`10.658`; the start-only null remains exactly flat. This is useful mechanism
evidence, but not yet a promoted method because it is an offline cached-rollout
reweighting test. The next promotion-quality task is to train or calibrate a
pre-rollout response-aware support scorer and run held-out CRPS/energy/coverage
plus fixed-start conditionality gates.

Operational pre-rollout probe:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_934b_response_book_guard/conditionality_stress_test.json`

Verdict: `candidate_rejected_as_default`. The
`narrative_book_response_guard_gap30` policy wires the response-aware idea into
the pre-rollout support selector by blending train-only per-book
portfolio-response priors using the grounded narrative risk channels. It
generates successfully but is weaker than the incumbent component baseline:
conditionality gates fall to `3/6`, relevant-factor KS falls to `0.0985`, path
energy to `0.0085`, portfolio KS to `0.0536`, and VaR95 range to `1.479`.
The incumbent remains `6/6`, with relevant-factor KS `0.1775`, path energy
`0.0431`, portfolio KS `0.1179`, and VaR95 range `10.339`. The failure mode is
useful: a train-only risk-book prior can over-broaden or marginalize support
mixtures, smoothing away narrative response. The next method should learn a
closer approximation to the 934a generator-response surface rather than
promoting the book-guard prior.

Preview-split response TestFlight:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_response_preview_support_weighting_934c_p16/response_aware_support_weighting_report.json`

Verdict: `candidate_mechanism_found_not_promoted`. The 934c candidate was the
strongest response-aware mechanism probe. It separates response scoring from
final pooling inside cached component-preserving rollout artifacts: for each
support component, a small preview subset estimates response in the
narrative-relevant channels, and the final scenario deck is sampled from the
remaining component paths when possible. This is closer to the intended
two-stage production mechanism than the rejected train-only book guard:
small frozen-generator preview, response-aware support reweighting, then final
component-preserving rollout. It uses no realized future paths and no OpenAI
calls.

Sensitivity against the same current component baseline is stable across pilot
sizes:

| Preview samples/component | Gates | Relevant KS delta | Path-energy delta | Portfolio KS delta | VaR95 range delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `6/6` | `+0.0352` | `+0.0095` | `+0.0467` | `+4.4488` |
| `16` | `6/6` | `+0.0359` | `+0.0076` | `+0.0439` | `+0.4514` |
| `32` | `6/6` | `+0.0327` | `+0.0065` | `+0.0437` | `+3.9899` |

This demonstrates materially stronger fixed-start conditionality than the
incumbent in narrative-relevant factor and portfolio-tail readouts while the
start-only null remains flat. It is still not a production/default promotion
because the evidence is artifact-level over cached rollouts.

Live two-stage response-preview probe:

- 64-sample live rollout:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_935a_response_preview_s64`
- 64-sample stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_935a_response_preview_s64/conditionality_stress_test.json`
- 384-sample full-preview stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_935b_response_preview_s384/conditionality_stress_test.json`
- 384-sample bounded-preview stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_935c_response_preview_blend035_s384/conditionality_stress_test.json`

Verdict: `candidate_mechanism_not_promoted_live_probe_negative_at_scale`. The
64-sample live two-stage smoke improved relevant-factor KS (`0.1714` to
`0.1894`), path energy (`0.0162` to `0.0222`), portfolio KS (`0.1354` to
`0.1479`), and VaR95 range (`3.587` to `4.686`) versus the current component
baseline. The 384-sample gate did not confirm the improvement. Full response
preview weakened the incumbent metrics: relevant-factor KS `0.1574` versus
`0.1775`, path energy `0.0272` versus `0.0431`, portfolio KS `0.1071` versus
`0.1179`, and VaR95 range `5.935` versus `10.339`. A bounded `0.35` blend
reduced weight concentration but still underperformed: relevant-factor KS
`0.1422`, path energy `0.0215`, portfolio KS `0.0847`, and VaR95 range
`5.559`.

Decision: keep live response preview as a diagnostic, not a default. The next
promotion-quality task is to learn or calibrate a deployable response-aware
support scorer from historical backtest labels and preview-response labels,
then compare it against the current component-preserving baseline on held-out
CRPS/energy/coverage and fixed-start conditionality gates. Do not continue by
only sweeping preview temperature or blend; the current mechanism needs a
learned response surface, not another operational knob.

Deployable support-reliability refresh and direction-safe quality guard:

- feature-sufficiency diagnostic:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_feature_sufficiency_936a_refresh/portfolio_response_feature_sufficiency.json`
- direction-safe fixed-start rollout:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_936d_portfolio_quality_guard_direction_safe_s64`
- direction-safe stress test:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_936d_portfolio_quality_guard_direction_safe_s64/conditionality_stress_test.json`

Verdict: `candidate_promising_not_promoted`. The 936a refresh found deployable
support-reliability signal: test pairwise accuracy `0.5717` versus a rank floor
of `0.5253`, and mean selection regret `0.0540` versus `0.0804`. The first live
quality-guard rollout exposed an important safety failure: one response-aware
mixture failed the final mixed-prefix direction check. The 936d update keeps
the response-aware scorer but adds direction-safe fallback to the base support
prior whenever the final mixed prefix rejects.

The 64-sample direction-safe gate now passes all six conditionality gates. The
portfolio quality support guard is slightly stronger than the incumbent on the
main separation metrics: relevant-factor KS `0.1831` versus `0.1805`, path
energy `0.0247` versus `0.0237`, and portfolio KS `0.1385` versus `0.1302`.
The VaR95 range is slightly lower (`4.1152` versus `4.3177`), and the start-only
null still fails with zero factor, path, and portfolio separation.

Decision: keep the direction-safe portfolio quality guard as the active
response-aware candidate, but do not promote it to demo or paper default yet.
It needs a larger-sample rollout and a held-out scenario-quality check
including CRPS, energy, coverage, qualitative raw-level fans, and
portfolio-tail plots.

937 follow-up verdict: `candidate_not_promoted_larger_gate_mixed`. The
larger/default follow-up kept the conditionality signal above the start-only
null, but did not beat the incumbent component-preserving mixture. In the
384-sample 937a gate, the portfolio quality guard passed `6/6` gates and
slightly improved mean portfolio terminal KS (`0.1198` versus `0.1137`), but
relevant-factor terminal KS and path energy were slightly weaker than the
incumbent (`0.1305` versus `0.1331`, and `0.0234` versus `0.0239`). A
direction-safe candidate-recovery implementation was then added to improve
fallback behavior when marginal response weighting fails the final mixed-prefix
direction check. That recovery is useful for auditability, but it did not
produce a promotable default: the 937g default-guard stress reports incumbent
relevant terminal KS `0.1884`, path energy `0.0283`, portfolio KS `0.1625`, and
VaR95 range `4.5227`, versus portfolio quality guard `0.1677`, `0.0215`,
`0.1354`, and `2.2341`. The support-max-off branch was worse and is rejected.
Therefore the response-aware quality guard remains diagnostic, not a product
default.

Current bottleneck: response-aware scoring is colliding with the final
direction-safe support contract. Candidate breadth exists, but many
response-weighted mixtures fail the final direction check or smooth away the
response. The next principled step is not another threshold sweep. It is to
make candidate generation and response scoring jointly direction-aware before
marginalization, or to learn a narrative-channel-specific response scorer whose
candidate sets pass both final direction and scenario-quality gates.

## Current Method Identity Before Episode-Retrieval Branch

The prior objective was **support-coherence and component-posterior selection
inside support-grounded latent scenario generation**, not a generic
text-to-time-series generator and not an agentic LLM distribution forecaster.
The full narrative and fixed starting level first define an auditable support
distribution over historical prefixes or learned support prototypes. The frozen
SNI rollout then turns that support-grounded condition into an ensemble of
30-day paths. The next research step is to test whether internally coherent
support families and sparse component-posterior distributions preserve more
narrative signal than broad heterogeneous pooling, without replacing the SNI
generator or hiding support provenance.

The latest conditionality audit found that an averaged-prefix implementation is
not sufficient for the product claim. It selected different support pools, but
then averaged the support memory before rollout, producing fan charts that were
mostly shared-shape distributions with small mean/width changes. The current
candidate fix is **component-preserving support-mixture rollout**: keep the same
selected start, decode each supported component separately, run the frozen SNI
rollout per component, and pool samples by support weight. This keeps mixture
shape operational instead of averaging it away before generation.

This resolves the latest objective drift risk. The historical mixture is not a
fallback, a weakness, or a disposable interpretability layer. It is the
financial inductive bias that keeps language-conditioned generation on the
learned market manifold. Future sophistication should improve the generated
support-grounded distribution itself: better support-family coherence,
component-posterior formation, warning quality, or directional audits around
support. A stronger generic text embedding or a more complex ranker is not
enough unless it improves held-out scenario quality, start-normalized narrative
attribution, or trust under this support-grounded contract.

## Current Incumbent

- Support prior: `diverse_topk_narrative_start_checked` with a direction gate
  and a demo/product temporal gap of `30` bridge-local windows. The selector
  uses fewer than the requested top-k supports if only fewer distinct,
  non-overlapping supports pass.
- Portfolio-response support policy: `portfolio_quality_guard_924e` is now
  available as an opt-in story-smoke memory-prior mode. It reuses the verified
  train-only portfolio-plus-quality scoring rule and the train-derived
  support-max gate. If the support-max gate fails, the mode falls back to the
  base `diverse_topk_narrative_start_checked` support prior. This is product
  wiring for the current default candidate, not a user-facing demo default yet.
  Contract smokes:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_story_smoke_925b_portfolio_quality_guard_skip/prefix_latent_story_smoke_report.json`
  and
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_story_smoke_925c_portfolio_quality_guard_rollout_smoke/prefix_latent_story_smoke_report.json`.
  A cached three-case comparison (`925d`) against the base diverse support
  prior is mixed: the commodity case selected a materially different support
  set, the defensive case fell back, and the fragile risk-on case had only one
  admissible quality-guard candidate. Summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_casebook_compare_925d_summary/casebook_quality_guard_comparison.json`.
  Keep this mode opt-in until a broader paired comparison shows adequate live
  candidate breadth and no regression against the simple support-mixture floor.
  Follow-up `925e`/`925f` added explicit candidate-breadth controls and a
  minimum-candidate fallback: low-breadth rows no longer activate the live
  quality guard. The post-gate 36-row diagnostic reports changed support
  `15/36`, active low-breadth rows `0/36`, fallback rows `10/36`, and status
  `fallback_warning`. Artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_quality_guard_live_breadth_925f_minbreadth/portfolio_quality_guard_live_breadth.json`.
  A story-smoke CLI check records fallback reason
  `insufficient_live_candidate_mixtures` for a one-candidate fragile risk-on
  case:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_story_smoke_925f_qg_minbreadth_skip/prefix_latent_story_smoke_report.json`.
  Active-row paired rollout pilots (`925g`/`925h`/`925i`) show that the opt-in
  guard can change generated distributions, not just support tables: mean
  support Jaccard was `0.051` across three cases, SPX terminal means shifted
  lower in all three, and VIX distribution width increased in all three.
  Aggregate:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_quality_guard_paired_rollout_925ghi_summary/paired_rollout_pilot_summary.json`.
  This is candidate evidence only; quality-guard story-smoke runs still carry
  warnings and need a batched/sequential paired-rollout evaluator before any
  default-promotion decision.
  `925j` adds that sequential evaluator:
  `experiments/backfill/block_ar/nl_portfolio_quality_guard_paired_rollout_runner.py`.
  A one-pair smoke completed at
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_quality_guard_paired_rollout_runner_925j_smoke/paired_rollout_runner_summary.json`.
  The scaled active-row pass completed in `925l` over all `23` active
  adequate-breadth rows:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_quality_guard_paired_rollout_runner_925l_active23/paired_rollout_runner_summary.json`.
  The guard stayed active with no fallback and materially changed support
  selection (`mean_support_jaccard=0.0557`), but the generated distribution
  response is still mixed: commodity rows mostly lower SPX and raise VIX,
  defensive rows raise VIX and worsen equity-beta/carry path loss, but SPX and
  portfolio responses are split by starting level and all quality-guard runs
  remain warning-level. Therefore `portfolio_quality_guard_924e` remains an
  opt-in diagnostic candidate, not the default product/demo path. The next
  method work should improve response-aligned support scoring or readout
  targets so promotion is based on risk-manager-visible distribution behavior,
  not merely low support overlap.
- Text bridge incumbent: `mlp_mse_contrastive + multi_caption_with_negatives`.
- Scenario-to-text caption incumbent: `risk_manager_caption_v2_2026_05_15`
  structured output with both quant-specialist Word documents loaded and hashed
  per run.
- Optional premium caption lane: Codex CLI `codex exec -m gpt-5.5 -c
  model_reasoning_effort='xhigh' --output-schema ...`, using a strict
  all-fields-required `RiskManagerCaptionV2` schema and the same leakage
  validator. This is currently a gold-caption/evaluator route for users with a
  Codex subscription, not a wholesale replacement for API captioning until
  downstream embedding/support/scenario tests show it improves conditioning.
- Caption approval status: current `risk_manager_caption_v2` outputs are
  **risk-manager-document compliant**, not literally signed off by a human risk
  manager. The evidence is the two specialist-document checks, strict schema,
  leakage validator, and 3-case Codex/API provider comparison. Human review can
  still be used as a final gold-standard calibration step before any larger
  labeling run.
- Grounding role: sidecar audit/check, not a replacement for the full
  narrative.
- Retrieval role: operational support prior, provenance, and support audit; not
  single-neighbor replay and not a support-free text latent.
- Rollout mixture role: `component_prefix_mixture` is the candidate product
  path; `averaged_prefix` is now a diagnostic baseline unless future held-out
  evidence shows it has enough risk-manager-visible conditionality.
- Generator: frozen joint39 SNI checkpoint through the native autoregressive
  rollout path.
- Autoresearch orchestration: centralized HEAD loop with bounded multi-agent
  sidecars only at literature, critique, verification, report-audit, or
  disjoint-implementation gates.

## Current Autoresearch Policy

The NL prefix-latent workflow now separates research exploration from product
promotion:

- **Exploration lane:** cheap local tests, literature checks, and mechanism
  probes may regress below the simple-mixture floor. A bad result is acceptable
  when it teaches why a method should be promoted, modified, or killed.
- **Candidate lane:** a method with a plausible mechanism can tolerate moderate
  regression only if the expected trade-off was stated before the run.
- **Promotion lane:** a method must be competitive with the incumbent simple
  mixture on held-out scenario-level quality, fixed-start controls, support
  audits, and null/repeat checks before it changes defaults or paper claims.
- **Production/demo lane:** no drastic regression is allowed. Small regressions
  are acceptable only for clear trust, warning-quality, OOD-rejection,
  provenance, or fixed-start-stability gains.

Every future HEAD entry should name `research_lane`, `result_status`, and
`benchmark_floor_status`. The simple mixture remains the promotion floor and
production backbone, but it should not prevent low-cost exploration of learned
support rerankers, contrastive support alignment, prototype-aware mixture
weights, or bounded residual refinements.

Sophisticated narrative-to-mixture methods now require a **story gate** in
addition to the metric gate. Before a method can be promoted, the workflow must
record the method story, related-work basis, novelty claim, elegance check,
historical-backtest comparison, and kill condition. This is meant to keep the
research line publishable and product-legible: a new method should be
methodologically justified, not just more complicated, and it must still beat or
remain competitive with the simple mixture on historical backtesting before it
changes defaults or paper claims.

The tracked method-intake template is:

`docs/research_protocols/nl_prefix_latent_method_intake_template.md`

Use it before a sophisticated candidate leaves ideation. A candidate that lacks
an intake artifact is not eligible for promotion, even if it improves a narrow
metric. The intake must explain why the method is the smallest justified change,
which related work supports it, what the local novelty is, what held-out
backtest will decide it, and what result kills it.

Current readout candidate intake:

`docs/research_protocols/nl_prefix_latent_conditionality_aware_readout_intake.md`

This candidate targets the latest bottleneck: preserving narrative-conditioned
path dependence while calibrating spread/readout quality. It is inspired by
ensemble copula coupling and related multivariate ensemble postprocessing, not
by a new text embedding model.

Current active support-weighting candidate intake:

`docs/research_protocols/nl_prefix_latent_response_aware_support_weighting_intake.md`

This is now the main next objective. It generalizes the previous
portfolio-response experiments into one response-aware support-weighting program:
learn or calibrate support weights using generator-response evidence, then
prove the method with fixed-start controls, start-only nulls,
narrative-relevant factor panels, portfolio-tail readouts, and held-out
CRPS/energy/coverage checks. It should strengthen conditionality in the
narrative's risk channels without abandoning the support prior or frozen SNI
rollout.

Current portfolio-response support-policy intake:

`docs/research_protocols/nl_prefix_latent_portfolio_response_support_policy_intake.md`

The first TestFlight, `portfolio_response_kernel_listwise_921a`, builds
portfolio-risk-response labels from existing train candidate rollouts and trains
the same compact kernel-listwise support policy used by the previous
generator-response experiment. The candidate improves the targeted reliable
portfolio response score on the 66-window held-out comparison, but it is not
promoted: ensemble CRPS is essentially flat/slightly better, coverage is
slightly better, and energy regresses slightly. Current status:
`candidate_tradeoff_not_promoted`, benchmark floor status `competitive`.
Artifacts:

- label report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_policy_921a/portfolio_response_label_scenario_report.json`
- learned policy:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_kernel_listwise_921a/learned_mixture_policy_report.json`
- held-out evaluation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_kernel_listwise_921a_scenario_eval/scenario_level_eval_report.json`
- comparison:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_policy_921a/portfolio_response_support_policy_comparison.json`

Do not change the demo or paper default to this policy unless a later
experiment removes the energy trade-off or demonstrates a larger
portfolio-tail/conditionality gain under fixed-start controls.

Follow-up oracle gate:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_oracle_soft_support_921b_comparison_seed921/portfolio_response_support_policy_comparison.json`

This leakage-only oracle used the same held-out candidate support mixtures, but
weighted them using the realized portfolio-response labels. It beats the equal
floor on the same-seed comparison: CRPS delta `-0.000246`, energy delta
`-0.000165`, and reliable portfolio path-score delta `-0.014482` (about `+1.82%`
relative reduction). This says the support store and portfolio-response label
have a real upper-bound signal. The bottleneck is learning that response surface
from deployable narrative/start/support features, not abandoning historical
support mixture.

Deployable follow-ups:

- `nl_portfolio_response_kernel_listwise_feature_lift_921c` adds signed
  recent-prefix portfolio features for the reliable books. It remains
  diagnostic: reliable portfolio path-score improves by `-0.002967`, but CRPS
  and energy regress slightly versus the equal floor.
- `nl_portfolio_response_support_set_921d` uses the existing set-based support
  scorer with the same labels/features. It is rejected: CRPS delta `+0.036520`,
  energy delta `+0.050584`, coverage delta `-0.060956`, and reliable portfolio
  path-score delta `+0.065194`.

The next support-policy work should not add more capacity blindly. It should
first explain the gap between the positive oracle and weak deployable policies,
for example by analyzing whether the candidate features fail to predict
portfolio labels out-of-sample, whether labels are too sample-noisy at
`samples=2`, or whether candidate support generation needs richer candidates.

## Immediate Reverse-Direction Gate

The next caption-related step is a controlled reverse-direction TestFlight, not
a full relabeling run:

```text
scenario -> risk-manager-document-compliant caption
caption + structured sidecar -> text representation
text representation + fixed start -> support mixture
component-preserving support rollout -> held-out scenario metrics and audits
```

The TestFlight must compare incumbent API captions against the Codex gold
caption lane on the same windows and starts. It should test both
`text-embedding-3-small` and `text-embedding-3-large`, because embedding cost is
small relative to caption generation and the older `text-embedding-3-large`
failure was measured on weaker caption text. Do not promote a larger embedding
model unless it improves at least one product-facing downstream gate: support
selection quality, hard-negative separation, direction/support audit quality,
fixed-start conditionality, or held-out CRPS/energy/coverage.

The bridge target remains the SNI encoder's 128-dimensional terminal memory
state for support ranking. The production generator should still use
component-preserving support prefixes for rollout. The terminal-memory bridge is
a retrieval/support-prior index, not sufficient evidence for direct text-only
generation.

First reverse-direction TestFlight result:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_reverse_ab_916e_openai_small_large/caption_reverse_ab_report.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_reverse_ab_916e_openai_small_large/caption_reverse_ab_rollout_group_summary.md`

This 10-window, 66-text-variant A/B tested legacy/simple text, the old generic
fragile-risk-on demo narrative, `risk_manager_caption_v2` API captions, Codex
gold captions, and both `text-embedding-3-small` and `text-embedding-3-large`.
It supports a guarded claim: richer risk-manager captions materially improve
bridge/support alignment and beat the old generic demo narrative in rollout,
but they do not yet beat the strongest simple fact-token baseline in
scenario-level rollout. For `text-embedding-3-small`, rich API captions improve
target-memory cosine from `0.373` to `0.680` versus the simple group and improve
support cosine from `0.656` to `0.838`; rollout CRPS improves over the generic
demo story (`0.873` versus `0.909`) but remains behind the simple group
(`0.847`). `text-embedding-3-large` gives the rich API caption the best target
cosine (`0.722`) but does not solve the rollout gap. Do not sell the richer
caption as a proven scenario-quality win over direct fact-token conditioning
yet; sell it as better risk-manager language and better support alignment, with
the next research step focused on representation fusion and support weighting.

Latest Codex caption scaling TestFlight:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_v2_codex_batch_917a_pilot1/codex_caption_batch_report.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_reverse_ab_917c_codex_batch_20_union/caption_reverse_ab_report.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_reverse_ab_917c_codex_batch_20_union/small_rollout/caption_rollout_group_summary.md`

The workflow now has a resumable Codex CLI corpus-caption runner with strict
`RiskManagerCaptionV2` schema validation and both ordered full-corpus selection
and split-balanced representative pilot selection. A 20-window Codex `gpt-5.5` xhigh
batch passed after tightening the prompt to keep leakage-exclusion wording out
of `training_caption` and to require the exact phrase `not a forecast` in the
caveat. This is not the full 380-window corpus yet; it is the safe scaling gate
before launching the multi-hour full-corpus Codex job. On the 20-window union
A/B, `text-embedding-3-small` remains better than `text-embedding-3-large` for
the Codex reverse path. Fused Codex text improves target-memory cosine to
`0.694` versus simple text at `0.373` and support cosine to `0.813` versus
`0.645`. In the low-budget scenario rollout, fused Codex gives CRPS/energy
improvements of `0.207`/`0.216` versus persistence, slightly better than the
simple group at `0.169`/`0.185`, but 80% coverage remains similar and terminal
MAE is not improved. Treat this as a candidate-level signal that professional
Codex captions plus explicit fact tokens can help scenario quality; do not
promote it until the full corpus and held-out backtest confirm the gain.

Balanced-80 split-balanced Codex caption extension:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_v2_codex_batch_917e_balanced80/codex_caption_batch_report.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_reverse_ab_917e_balanced80/caption_reverse_ab_report.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_reverse_ab_917e_balanced80/small_rollout/caption_rollout_group_summary.md`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_caption_reverse_ab_917e_balanced80/large_rollout/caption_rollout_group_summary.md`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/caption_conditionality_refresh_917f_balanced80/caption_conditionality_refresh_summary.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_conditionality_strength_benchmark_917f_balanced80_caption_quality/conditionality_strength_benchmark.json`

The 80-window split-balanced Codex batch is the first representative caption
scale-up across train/validation/test. It passed strict schema/leakage
validation for all 80 requested windows, with 0 Codex execution errors and 0
caption validation errors. Logged Codex CLI usage across the 80 event files was
approximately 2.83M input tokens, 1.18M cached input tokens, 135.8k output
tokens, and 43.6k reasoning output tokens.

The stronger caption text clearly improves the reverse-direction
text-to-support signal. With `text-embedding-3-small`, fused Codex captions
improve mean target-memory cosine from `0.367` for simple text to `0.692`, and
mean support cosine from `0.646` to `0.829`. With `text-embedding-3-large`,
fused Codex captions improve target-memory cosine from `0.577` to `0.751`, and
support cosine from `0.686` to `0.885`.

Scenario rollout confirms the improvement direction, but it does not yet justify
promoting a new default. For `text-embedding-3-small`, fused Codex improves CRPS
and energy by `0.139`/`0.155` versus persistence, ahead of the simple group at
`0.085`/`0.110`. For `text-embedding-3-large`, fused Codex improves CRPS and
energy by `0.124`/`0.142`, ahead of the simple group at `0.107`/`0.125`, but
less decisively than the bridge-space gain. Overall large-embedding top-k
rollout improves CRPS/energy by `0.111`/`0.129` versus persistence with 80%
coverage of `0.506`; small improves `0.104`/`0.125` with 80% coverage `0.504`.

Decision: keep Codex professional captions plus explicit fact tokens as a
candidate text representation. The evidence is now stronger than the 20-window
pilot, but the generator/readout still attenuates some caption-quality gains.
Do not promote the default until a larger full-corpus or true held-out
non-train gate shows that the fused representation remains competitive with the
simple fact-token floor while improving risk-manager narrative quality,
directional checks, and qualitative conditionality.

Caption-side conditionality refresh: because the scenario-level gain persisted
under both embedding models, the paper figures now include a quantitative
balanced-80 caption gate and a raw-level same-window qualitative case. The
selected qualitative test window `joint39_val_0378` is intentionally not used
as a new aggregate metric: it shows the mechanism. Simple fact tokens selected
support windows `0079`, `0243`, and `0197`; Codex professional captions
selected `0046`, `0219`, and `0085`; fused Codex+facts selected `0252`,
`0040`, and `0219`. The simple-vs-fused support Jaccard was `0.0`, and fused
Codex reduced that window's CRPS by `0.509` and energy by `0.543` versus
simple fact tokens. The refreshed fixed-start benchmark remains
`conditionality_partially_supported_with_warnings`: support diversity and
repeat/start-only controls remain favorable, but bootstrap path noise and
portfolio tail separation still limit a clean no-warning conditionality claim.

Conditionality transmission audit:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_conditionality_transmission_audit_918a_balanced80/conditionality_transmission_audit.json`

The follow-up transmission audit was run only after confirming that the
balanced-80 scenario-level gain persisted. Its verdict is
`support_and_prefix_preserved_rollout_tail_bottleneck`. The support layer is
not the immediate bottleneck: observed cross-narrative support TV distance is
`1.000`, while same-narrative repeat and start-only ratios are `0.000`. The
decoded-prefix layer also preserves the narrative difference: observed
decoded-prefix RMSE is `0.710`, with same-narrative repeat and start-only
ratios again `0.000`. The final rollout layer is where the signal is limited:
observed rollout path energy is `0.055`, repeat/observed is `0.133`, but
within-run bootstrap/observed is `0.938`. The portfolio audit also keeps the
tail warning because VaR95 observed/repeat is only `0.735`. Diagnosis: the
current narrative conditioning does enter through different support pools and
decoded prefixes; the next method work should focus on generator-response-aware
support weighting or path/dependence-aware readout so that the frozen SNI
rollout transmits those support differences into portfolio-risk distributions
above bootstrap noise.

Current readout-frontier diagnostic:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_readout_frontier_905e_alpha_global/readout_frontier_report.json`

The global mean-preserving fan-scale readout is not sufficient as a promoted
production display fix. Across alpha `2.5`, `3.0`, and `3.5`, no candidate met
the retention gate. The best alpha retained only `0.314` of uncalibrated
full-path energy signal and had bootstrap-to-observed path energy of `1.894`.
This supports the current diagnosis that the narrative mixture is not the main
failure; the display/readout layer needs a conditionality-aware calibration
that preserves path ranks/dependence while improving marginal spread.

Rank-preserving marginal readout TestFlight:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_rank_preserving_readout_905f/rank_preserving_readout_report.json`

This data-fitted readout is more promising for realized historical
distribution quality but still not production-ready. It fits a positive
horizon/factor alpha map on calibration backtest rows and preserves
per-variable sample ranks. On evaluation rows it improves 80% coverage by
`+0.359`, CRPS by `-0.078`, and energy by `-0.152` versus the uncalibrated
component mixture. However, fixed-start conditionality still fails: same-story
repeat controls are too close to observed narrative effects, and bootstrap
noise remains high. Treat this as evidence that marginal readout calibration
can improve backtest quality, but conditionality preservation needs a stronger
path/dependence-aware design before promotion.

Repeat-noise diagnostic:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_repeat_noise_diagnostic_905g/repeat_noise_diagnostic.json`

Same-story repeats reuse identical support weights (`1.0` weighted overlap),
so support selection is not the repeat-noise source. The decoded prefix is not
stable enough across seeds: repeat decoded-prefix L2 is `0.651` of observed
cross-narrative decoded-prefix L2. Uncalibrated generated path energy for
repeats is much smaller (`0.223` of observed), but the rank-preserving readout
amplifies repeat/control differences until they become too close to the
observed narrative effect. The next candidate should therefore stabilize or
cache the prefix decoder/readout path before adding a more aggressive
calibration layer.

Fixed-decoder seed TestFlight:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_repeat_noise_fixed_decoder_905h/repeat_noise_diagnostic.json`

The story-smoke path now supports separate decoder and rollout seeds. Holding
the decoder seed fixed while varying rollout seeds eliminates decoded-prefix
repeat instability: repeat-to-observed decoded-prefix L2 falls from `0.651` to
`0.0`. Same-story repeat generated path energy is then `0.687` of observed
narrative energy, below the current `0.75` repeat gate. This supports a
production rule: the memory+start prefix decoder should be treated as a frozen
or cached model component, not retrained with each rollout/sample seed.
Remaining noise is downstream rollout/bootstrap noise, not support or prefix
selection instability.

Promoted diverse-support fixed-start redo:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_mixture_fixed_start_906a_diverse_gap30_s384/`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_decoder_controls_906a_diverse_gap30_s384/component_fixed_start_controls.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_906a_diverse_gap30_s384/fixed_start_shape_audit.json`

`paper/narrative_grounded_scenarios/figures/narrative_qualitative_casebook_summary.json`

The paper-facing conditionality analysis was rerun under the promoted
`diverse_topk_narrative_start_checked` selector with a `30` bridge-local
non-overlap gap and `384` generated paths per narrative. This redo supersedes
the earlier 905i/904k paper evidence for current paper claims because those
artifacts did not use the formal non-overlap selector. The new result is
guarded: support sets are distinct and start-only controls stay at `0.0`, but
within-run bootstrap noise is still close to the observed narrative effect.
The component control status is `warning` with
bootstrap-to-observed median ratio `1.184`, repeat-to-observed median ratio
`0.137`, and start-only ratio `0.0`. The full path audit is also `warning`
with repeat-to-observed path energy/variance/Wasserstein
`0.089` / `0.395` / `0.388`, bootstrap-to-observed
`1.556` / `0.716` / `1.047`, and start-only `0.0` / `0.0` / `0.0`.
Conclusion: the current paper should claim auditable, partial fixed-start
conditionality with explicit bootstrap/readout warnings, not a clean uniform
promotion pass.

Conservative global readout calibration:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_905j_alpha1p05/component_global_calibration_report.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_905j_fixed_decoder_global1p05/fixed_start_shape_audit.json`

A very small global fan scale, `alpha=1.05`, is the first nontrivial readout
candidate that passes the matched fixed-decoder path gate with no warnings.
On held-out evaluation rows it improves 80% coverage by `+0.0147`, ensemble
CRPS by `-0.00418`, and energy by `-0.00738` versus the uncalibrated component
mixture. The gain is modest, so this is a candidate display/readout option,
not a new narrative-conditioning method. More aggressive global scales
(`1.10`, `1.25`, `1.50`, `2.00`) improve backtest quality more but trigger
bootstrap-noise warnings; marginal rank-preserving readout improves historical
backtest quality substantially but still collapses fixed-start conditionality.

Readout gate selector:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_readout_gate_selection_905j/readout_gate_selection.json`

The readout selector combines held-out quality deltas with fixed-start path
audit status and chooses a calibration only when both gates agree. On the
current candidate set it selects `alpha1p05` and rejects `alpha1p10`,
`alpha1p25`, and `alpha1p50` because their path audits warn. This is now the
preferred governance rule for readout changes: historical CRPS/energy gains
alone are not enough if the calibration weakens narrative conditionality
relative to bootstrap/repeat controls.

Start-22 fixed-decoder robustness check:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start22_fixed_decoder_controls_905k_s384/component_fixed_start_controls.json`

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start22_path_audit_905k_fixed_decoder_s384/fixed_start_shape_audit.json`

After fixing the arbitrary-start loaders, start `22` was re-audited at `384`
samples with a fixed/cached decoder seed. The path audit remains `warning`, but
the failure mode is now clearer: repeat-to-observed path energy is `0.419`,
path variance is `0.273`, and path Wasserstein is `0.575`, so same-story
repeat path noise is below the hard gate. The warning is driven by
within-run bootstrap noise (`1.521` path energy, `0.974` path Wasserstein)
being too close to or larger than observed cross-narrative differences. The
terminal-gap control is stricter and fails because repeat terminal gap is
`0.788` of observed and bootstrap terminal gap is `1.895`. This supports the
current diagnosis: for some starting levels, the narrative support channel
selects distinct supports, but the frozen rollout/readout does not amplify
those differences enough above bootstrap noise.

Refreshed start-level stratification:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_start_stratification_905k_fixed_decoder_refresh/component_start_stratification.json`

The refreshed five-audit stratification is: one `pass` (`start 18`), three
`warning` cases driven by bootstrap/readout noise (`starts 22`, `40`, and
`77`), and one `fail` from start incompatibility (`start 0`). This is the
current broad production statement. The method is usable as a guarded
workflow, but the app/paper should not claim uniform fixed-start
conditionality across arbitrary starts yet.

## Benchmark Floor For New Narrative Methods

The current long-term objective is to improve the narrative-to-mixture workflow,
not to remove the historical support mixture. New candidate methods must be
benchmarked against the working simple-mixture floor and the current
direction-checked diverse support path before promotion.

Primary floor:

- simple floor: `soft_topk_narrative_start_checked` /
  `narrative_generator_topk`;
- representative artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_residual_scenario_eval_878a_full/scenario_level_eval_report.json`;
- direct text-predicted memory without support mixture:
  `+1.9%` ensemble CRPS, `+8.3%` energy, `0.390` 80% coverage versus
  persistence;
- narrative top-k support generator:
  `+17.2%` ensemble CRPS, `+21.6%` energy, `0.645` 80% coverage versus
  persistence;
- residual over top-k support at alpha `0.25`:
  `+17.4%` ensemble CRPS, `+21.6%` energy, `0.650` 80% coverage versus
  persistence.

A small distributional regression may be acceptable only when it buys clear
trust, stability, fixed-start conditionality, warning quality, or support
auditability. A method with drastic regression below the simple mixture remains
diagnostic even if it improves target cosine, retrieval rank, or an isolated
hard-case metric. Publishable novelty should come from learned support
reranking, contrastive support alignment, prototype-aware weighting,
text/start-conditioned mixture weights, or bounded residual refinement around
the mixture.

Fixed-start narrative methods also need a **conditionality usefulness gate**.
The gate should measure whether different narratives under the same selected
start produce materially different distributions, not only shifted or stretched
copies. Required diagnostics include support-overlap, standardized
cross-narrative sample correlation, normalized quantile-shape distance,
terminal KS, width-ratio changes, start-only controls, and same-narrative repeat
controls.

## Promoted Evidence

### Representative Bridge And Scenario Evaluation

- Bridge report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json`
- Scenario report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_scenario_level_eval_openai_schema_v2_representative_220/scenario_level_eval_report.json`

Current supported claim:

- held-out target cosine: `0.8597` mean, `0.8739` median;
- hard-negative gap: `0.8835`;
- recall@1 test-pool: `0.0794`;
- recall@3 test-pool: `0.1746`;
- narrative-generator energy improvement versus persistence: `+21.3%`;
- narrative-generator ensemble CRPS improvement versus persistence: `+17.4%`;
- mean-path and terminal point-error metrics remain worse than persistence.

Interpretation: the system is a distributional scenario generator with audit
support, not a point forecaster or exact historical-window retriever.

### Text-To-Memory Policy Tests

- Policy TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_850a_manifest_policy_testflight/bridge_architecture_bakeoff_report.json`
- Seed stability:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_851d_seed_stability/policy_stability_summary.json`
- CLIP MSE sweep:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_latent_bakeoff_852d_clip_mse_sweep/clip_mse_sweep_summary.json`

Current supported claim:

- multi-caption hard-negative training is the incumbent;
- multi-caption without hard negatives improves target cosine but weakens
  directional separation;
- hard negatives restore and strongly improve directional separation while
  preserving target cosine;
- current CLIP/InfoNCE hybrid is not promoted because it trails the incumbent on
  generator-memory target cosine and hard-negative separation.

Diagnostic, not yet promoted:

- expanded OpenAI label manifest:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_expansion_878e_openai_full/narrative_pipeline_report.json`.
  The larger manifest produced `240` usable labeled windows, `2974` text
  examples, and one rejected label (`joint39_val_0240`) due an
  external-catalyst grounding violation. The expanded bridge report
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_expansion_878g/bridge_eval_report.json`
  keeps target cosine near the incumbent (`0.8590`) on `42` held-out windows,
  but hard-negative gap (`0.8252`) and recall@3 test-pool (`0.1326`) are weaker
  than the representative 182-window run. Scenario-level evaluation
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_scenario_level_eval_expansion_878h/scenario_level_eval_report.json`
  remains distributionally useful (`+16.5%` CRPS, `+20.6%` energy), and on the
  `28` overlapping old test windows it slightly improves CRPS and energy while
  lowering coverage. Treat this as evidence that label scaling is feasible but
  not by itself a solved text-to-latent upgrade.
- all-window OpenAI label expansion:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_openai_full_906b_all_windows/narrative_pipeline_report.json`.
  This run directly tested whether weak narrative conditionality was caused by
  a limited cached label pool. It used the full `441` available-window manifest
  with a `279` / `36` / `66` train/validation/test split after embargo,
  reused existing labels, and called OpenAI for the `141` missing labels. The
  final validated set has `380` windows and `4663` text examples. Scenario
  quality stays competitive with the representative baseline: the direct-memory
  scenario ablation remains weak (`+3.29%` CRPS, `+8.37%` energy, `0.399`
  coverage), while narrative top-k support mixture remains strong (`+17.29%`
  CRPS, `+21.28%` energy, `0.623` coverage). Bridge exact retrieval still
  weakens as the candidate pool grows (`0.063` recall@3 in the held-out test
  pool), although hard-negative separation recovers to `0.878`. Conclusion:
  limited cache size is not the main blocker; the production path should keep
  the support mixture and improve support ranking/readout/generator-response
  calibration instead of assuming more labels alone will fix text-to-memory
  conditioning.
- expanded caption hard-case audit:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_caption_grounding_audit_878i/caption_grounding_audit.json`.
  The current quality hard cases are narrow: one rejected label
  (`joint39_val_0240`) and five low hard-negative-margin windows
  (`joint39_val_0377`, `0379`, `0384`, `0386`, `0430`). The low-margin cases
  are mostly risk-on/reflation descriptions where positive and negative
  catalyst language can become too semantically close. This points to caption
  repair/hard-negative wording before another bridge architecture.
- hard-negative wording repair TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_hard_negative_repair_878j_hardcases/hard_negative_repair_report.json`.
  Re-embedding the artificial negative controls for the five low-margin
  hard-case windows did not fix separation. The follow-up audit
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_caption_grounding_audit_878j_repair_hardcases/caption_grounding_audit.json`
  still has five low-margin cases and increases low-gap cases from `1` to `3`.
  Do not scale this deterministic negative wrapper.
- low-margin embedding geometry:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_embedding_geometry_hardcases_878k/embedding_geometry_hardcases.json`.
  All five low-margin windows are already low-margin in raw OpenAI embedding
  space. Mean raw hard margin is `0.278`; the learned bridge improves the mean
  margin to `0.382` but does not fully repair it. The bottleneck is therefore
  primarily text-embedding/representation geometry for risk-on/reflation
  contrastive cases, not just the MLP adapter.
- embedding-model/representation ablation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_embedding_model_hardcase_ablation_878l_large_full/embedding_model_hardcase_ablation.json`
  and
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_embedding_model_hardcase_ablation_878l_small_factor_tokens/embedding_model_hardcase_ablation.json`.
  `text-embedding-3-large` on full text worsens mean hard margin to `0.226`,
  and factor-token-only text with `text-embedding-3-small` worsens it to
  `0.096`. Generic larger embeddings and stripping to factor tokens are not the
  current fix; the likely direction is a hybrid narrative embedding plus
  explicit directional structure.
- hybrid direction-feature bridge TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_hybrid_direction_bridge_878m/hybrid_direction_bridge_report.json`.
  Equal-norm concatenation of the full narrative embedding with explicit
  direction features is not promoted. It leaves target cosine essentially flat
  (`+0.00013`) and slightly improves recall@3 test-pool (`+0.00552`), but
  worsens hard-negative mean margin by `-0.11319` and hard-negative mean gap by
  `-0.02456`. The old worst case `joint39_val_0377` improves, but broader
  separation degrades. Treat direction features as likely auxiliary loss,
  calibration probe, or audit constraint candidates, not as a naive
  concatenated input default.
- bounded start-residual bridge:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_start_memory_stability_876c_start_residual/text_start_memory_stability.json`.
  Across seeds `775`, `776`, and `777`, the bounded residual preserves the
  text-only memory target (`-0.0007` mean cosine delta) and hard-negative
  separation (`-0.0065` mean gap delta, `-0.0081` mean margin delta), and
  improves recall@3 by `+0.0159`. It is not a default until downstream
  scenario-level evaluation shows that the preserved residual signal actually
  affects fixed-start narrative scenario distributions.
- downstream residual scenario evaluation:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start_residual_scenario_eval_878a_full/scenario_level_eval_report.json`.
  On the same 29 held-out representative windows, direct text-predicted memory
  remains too weak to replace support-conditioned rollout (`+1.9%` CRPS,
  `+8.3%` energy, `0.390` 80% coverage versus persistence). The best bounded
  residual-over-top-k variant, `narrative_residual_topk_a025`, is competitive
  with the current top-k incumbent (`+17.4%` CRPS, `+21.6%` energy, `0.650`
  80% coverage), but the gain is marginal and mixed across metrics. Treat it as
  a promising diagnostic, not a promoted production default.
- residual attribution diagnostic:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_residual_scenario_attribution_878b/residual_scenario_attribution.json`.
  Alpha `0.25` improves average CRPS by only `0.0014` z-score units over the
  incumbent and wins CRPS on `14/29` windows; alpha `0.10`, alpha `0.50`, and
  direct-memory replacement are worse. Residual gains are not positively
  associated with higher top-1 support similarity (`-0.248` correlation), so
  the current residual bridge does not yet show a clean support-quality
  mechanism.
- learned support-reranker TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_learned_support_reranker_879a_testflight/learned_support_reranker_report.json`.
  This exploration-lane branch trained a small ridge reranker on train-window
  query/candidate features with negative standardized replay loss as the label.
  It found support signal at the replay level: held-out replay CRPS improved by
  `+0.0218` z-score units and replay energy improved by `+0.0059`, while
  coverage fell by `-0.0135`. A same-seed 5-window frozen-generator smoke did
  not promote the idea: historical replay improved, but narrative-generator
  CRPS regressed by `-0.0031` and energy by `-0.0017` versus the original
  support ordering. The decomposition
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_learned_support_reranker_879d_support_quality_decomposition/support_quality_decomposition.json`
  labels the mechanism `support_replay_generator_mismatch`. Treat this as
  evidence that future support rerankers should optimize generator-calibrated
  rollout compatibility or include a frozen-generator-aware loss, not only
  historical replay closeness.
- replay/generator mismatch post-analysis:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_generator_mismatch_analysis_880a/support_generator_mismatch_analysis.json`.
  The learned reranker improves the support proxy stack on all `126` compared
  rows: mean replay loss delta is `-0.0532`, mean generator self-calibration
  energy delta is `-0.0331`, mean generator self-calibration CRPS delta is
  `-0.0297`, mean start-distance delta is `-1.1575`, and coverage proxy delta
  is `+0.0050`. But the actual same-seed rollout smoke still regresses
  (`-0.0031` CRPS and `-0.0017` energy improvement deltas). The current
  mechanism label is therefore `generator_proxy_false_positive`: train-window
  self-calibration and historical replay are insufficient proxy labels for a
  support reranker. The next candidate should use direct rollout-response
  labels or a stability-aware generator-level objective, not another replay or
  self-calibration ranking knob.
- rollout-response label TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_rollout_response_label_testflight_881a/rollout_response_label_summary.json`.
  The scenario evaluator now has an explicit duplicate-query mode for
  candidate-specific support labels. A small CUDA smoke evaluated `4` query
  windows by `3` candidate supports each (`12` candidate rows) with top-k `1`.
  In `3/4` queries, the best actual generator-response support was not the
  cosine top-1 support. The within-pool best-vs-top1 mean deltas were
  `-0.0527` generator energy and `-0.0434` generator CRPS, lower-is-better.
  This is not a promoted model result because it is tiny and low-sample, but it
  is a positive mechanism TestFlight: direct rollout-response labels contain
  support-ranking signal that cosine/replay/self-calibration proxies miss.
- scaled rollout-response upper-bound:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_rollout_response_label_testflight_882a_fullheldout/rollout_response_label_summary.json`.
  Scaling the candidate label bridge to all `29` held-out query windows with
  the top `5` candidate supports (`145` candidate rows) confirmed that direct
  rollout-response labels contain ranking signal: best actual generator-response
  support was not the cosine top-1 support in `22/29` cases. The within-pool
  best-vs-top1 deltas were `-0.0898` generator energy and `-0.0746` generator
  CRPS, lower-is-better. However, the best single support upper bound still did
  not beat the same-seed simple top-k mixture baseline:
  best-single energy/CRPS means were `1.1265`/`0.7868`, while the top-k3
  baseline was `0.9826`/`0.6814`. This means the next learned method should
  stay mixture-level; replacing the auditable mixture with a learned single
  support is the wrong direction.
- mixture-level rollout-response upper-bound:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_rollout_response_mixture_testflight_884a_fullheldout/rollout_response_label_summary.json`.
  Evaluating all top-3 support subsets from each held-out query's top-5 support
  pool (`29` queries, `290` mixture rows) produced a positive upper-bound
  result. The best generator-response mixture was not the default cosine top-3
  in `25/29` cases. The best-in-pool mixture improved over the default top-3 by
  `-0.0594` energy and `-0.0470` CRPS on average, and it beat the same-seed
  simple top-k3 baseline: best-mixture energy/CRPS means were
  `0.9118`/`0.6248` versus baseline `0.9895`/`0.6879`. This is still an oracle
  upper bound, not a deployable learned policy, but it is the first strong
  evidence that a more sophisticated mixture-level narrative-to-support policy
  could beat the simple mixture floor.
- learned generator-response mixture policy:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_learned_mixture_policy_886b_32train_to_fullheldout/learned_mixture_policy_report.json`.
  The non-leaky train-label pipeline now builds train-window query bridges from
  cached text/memory artifacts, scores candidate top-3 support mixtures through
  the frozen generator, and trains a linear policy on inference-available
  mixture features. The 32-query label set confirms strong oracle signal:
  best generator-response mixture is not the default top-3 in `29/32` train
  queries, with best-vs-default deltas of `-0.1115` energy and `-0.0950` CRPS.
  However, the first learned policy is not promoted. On the 29 held-out windows
  at the same seed, it slightly regresses versus the simple mixture on energy
  (`0.9915` vs `0.9907`) and CRPS (`0.6947` vs `0.6882`) while improving 80%
  coverage (`0.5796` vs `0.5651`). This means generator-response labels are a
  viable training target, but the current linear mixture policy is too weak or
  under-labeled to beat the simple baseline.
- linear mixture policy post-analysis:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_learned_mixture_policy_887a_post_analysis/linear_policy_post_analysis.json`.
  On the held-out candidate-mixture pool, the linear policy score has weak
  out-of-sample correlation with actual generator-response labels
  (`0.095` versus negative energy, `0.173` versus negative CRPS). It chooses
  the exact energy-oracle mixture in only `2/29` windows and makes the
  candidate-pool result worse than default on average (`+0.0185` energy,
  `+0.0235` CRPS versus the default candidate mixture). The failure mechanism
  is now `linear_policy_underfits_generator_response_surface`, not lack of
  oracle signal.
- current next-candidate method story:
  `docs/research_protocols/nl_prefix_latent_mixture_policy_method_story.md`.
  The ideation candidate is query-relative pairwise/listwise support-mixture
  ranking. It is related-work-supported by learning-to-rank for within-query
  candidate comparison and permutation-invariant set models for support-mixture
  inputs. It is not promoted. Its required next gate is a no-OpenAI TestFlight
  that trains on existing generator-response mixture labels and compares
  same-seed held-out CRPS, energy, coverage, and support audits against the
  simple mixture floor.
- pairwise mixture-ranker TestFlight:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_pairwise_mixture_policy_887e_post_analysis/pairwise_policy_post_analysis.json`.
  This is not promoted. It used existing generator-response mixture labels
  without OpenAI calls and kept the support mixture as the output object. The
  pairwise policy improves over the previous linear policy on held-out scenario
  CRPS (`0.6909` versus `0.6947`) but remains worse than the same-seed simple
  mixture (`0.6882`). It also trails the simple mixture on energy (`0.9936`
  versus `0.9907`) while improving coverage (`0.5748` versus `0.5651`). The
  held-out candidate-pool audit shows weak energy alignment (`0.049`
  correlation with negative energy), moderate CRPS alignment (`0.234`
  correlation with negative CRPS), exact energy-oracle selection of only
  `2/29`, and selected mixtures worse than the default candidate mixture by
  `+0.0105` energy and `+0.0052` CRPS. The mechanism read is
  `pairwise_ranker_improves_some_crps_alignment_but_not_generator_quality`.
- mixture-ranker target/data-scale analysis:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_mixture_ranker_utility_analysis_887f/mixture_ranker_utility_analysis.json`
  and
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_pairwise_mixture_policy_888d_128train_post_analysis/pairwise_policy_128train_post_analysis.json`.
  Energy and CRPS candidate oracles are highly aligned within held-out
  candidate pools: mean within-query correlation is `0.917`, and the same
  candidate is both the energy and CRPS oracle in `22/29` held-out queries.
  Therefore the pairwise failure is not mainly a metric-conflict problem.
  Scaling generator-response labels from `32` train queries (`320` candidate
  mixtures) to `128` train queries (`1280` mixtures) also does not fix the
  current pooled-feature pairwise model. The 128-query ranker has only `0.153`
  training correlation with negative energy, `0.077` held-out correlation with
  negative energy, and selects mixtures worse than the default candidate
  mixture by `+0.0155` energy and `+0.0150` CRPS. The current bottleneck is
  therefore representation/model structure for support-mixture scoring, not
  merely metric scalarization or too few train labels.
- next method-intake candidate:
  `docs/research_protocols/nl_prefix_latent_set_ranker_method_intake.md`.
  The next proposed exploration candidate is `support_set_item_ranker`: a
  DeepSets-style support-set item encoder that scores candidate mixtures from
  per-support item features plus query/start context. The 128-query TestFlight
  is implemented and not promoted. Artifacts:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889b_128train_to_fullheldout/learned_mixture_policy_report.json`,
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889c_128train_fullheldout_scenario_eval/scenario_level_eval_report.json`,
  and
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889d_128train_post_analysis/support_set_ranker_post_analysis.json`.
  The support-set ranker slightly improves held-out energy versus the same-seed
  simple mixture (`0.9896` versus `0.9907`) and improves 80% coverage (`0.580`
  versus `0.565`), but worsens CRPS (`0.6902` versus `0.6882`). More
  importantly, its held-out candidate-pool scores are negatively correlated
  with actual generator-response energy and CRPS (`-0.120` and `-0.063`), and
  the selected mixtures are worse than the default candidate mixture by
  `+0.0147` energy and `+0.0197` CRPS. The current mechanism read is
  `set_ranker_adds_capacity_but_still_misses_candidate_pool_oracle`. A follow-up
  overfit stress check,
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889e_failure_analysis/support_set_ranker_failure_analysis.json`,
  shows that a larger no-regularization item ranker improves train correlation
  to `0.279` and train pairwise accuracy to `0.693`, but held-out correlation
  remains near zero (`-0.011`) and held-out pairwise accuracy falls below
  random (`0.452`). The failure is therefore not solved by simply adding set
  encoder capacity.
- next method-intake candidate after the support-set ranker failure:
  `docs/research_protocols/nl_prefix_latent_soft_listwise_mixture_policy_intake.md`.
  The new candidate is `soft_listwise_mixture_policy`: learn a listwise
  distribution over candidate support mixtures from generator-response labels,
  marginalize that distribution into auditable support-window weights, and make
  the scenario evaluator honor those weights instead of treating all selected
  analogues equally. This is not implemented or promoted. The reason to try it
  is specific: the prior hard-selection rankers are brittle, while the product
  already needs support weights to be more than display-only provenance. The
  weighted-sampler implementation is now TDD-covered and diagnostic only:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_weighted_sampler_889i_baseline_analysis/weighted_sampler_baseline_analysis.json`.
  Equal sampling remains the default. Softmax-cosine weighting at temperature
  `1.0` is a no-op because all `29` held-out windows allocate `[2,2,2]`
  samples, while a sharper diagnostic temperature `0.02` worsens CRPS
  (`0.6894` versus `0.6882`) and energy (`0.9921` versus `0.9907`) while only
  improving coverage (`0.570` versus `0.565`). Naive similarity sharpening is
  therefore not promoted; learned listwise weights still need a separate
  TestFlight. The first minimal listwise TestFlight is also diagnostic, not
  promoted:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_listwise_mixture_policy_889k_candidate_analysis/listwise_policy_candidate_analysis.json`.
  It improves the sign of held-out candidate-pool alignment versus the
  support-set ranker (`0.093` energy correlation and `0.133` CRPS correlation),
  but exact energy-oracle selection is only `3/29`, selected mixtures are still
  worse than default by `+0.0048` energy and `+0.0022` CRPS, and the average
  support weights are nearly uniform (`effective_n` about `4.99`). Do not run
  this policy as a scenario-level candidate until the listwise weights become
  materially more discriminative. Follow-up analysis
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_listwise_mixture_policy_889l_target_diffuseness/listwise_target_diffuseness_analysis.json`
  shows why: the target distribution is moderately sharp (`target_effective_n`
  about `5.36` out of `10`, max probability about `0.33`), but the model
  predicts almost uniform candidate probabilities (`pred_effective_n` about
  `9.89`, max probability about `0.116`). The bottleneck is not that the target
  is fully flat; the current linear features/model fail to recover the sharper
  generator-response mixture distribution.
- oracle soft-support upper bound:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_oracle_soft_support_weights_906c_fullheldout/oracle_soft_support_comparison.json`.
  This leakage diagnostic asks whether better support selection/weights could
  move the frozen generator if they were known with hindsight. The answer is
  yes. On the same-seed 29-window held-out representative split, oracle top-3
  support selection improves CRPS/energy/coverage versus equal top-3
  (`0.6763` / `0.9742` / `0.573` versus `0.6879` / `0.9895` / `0.558`).
  Top-5 oracle field-weight sampling also improves versus equal top-5
  (`0.6419` / `0.9295` / `0.633` versus `0.6452` / `0.9348` / `0.632`) and
  produces non-uniform sample allocation in `25/29` held-out windows. This is
  not deployable because it uses realized future labels. It does show that the
  support-prior/weighting lever is worth improving; prior learned policies
  failed because they did not learn the generator-response weighting surface,
  not because support weighting has no signal.
- query-relative kernel listwise support policy:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_kernel_listwise_mixture_policy_906d_128train_to_fullheldout/kernel_listwise_policy_comparison.json`.
  This is the first deployable learned support-weight policy in this branch
  that slightly beats the same-seed equal top-5 mixture on all three aggregate
  held-out diagnostics, while using only train-window generator-response labels
  for training and no held-out future labels at inference. It trains on `128`
  train queries / `1280` candidate mixtures, augments candidate features with
  within-query standardized features, and uses kernel regression over train
  candidate prototypes to score held-out candidate mixtures. The best bounded
  probability temperature is `0.20`: CRPS mean `0.6441` versus equal top-5
  `0.6452`, energy mean `0.9316` versus `0.9348`, and coverage `0.636` versus
  `0.632`; it creates non-uniform support allocation in `24/29` held-out
  windows. This remains diagnostic, not promoted: the gains are modest and
  capture only a small part of the oracle soft-support upper bound.
- scaled query-relative kernel/listwise support policy:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_kernel_listwise_full_906e_278train_to_66test_temp0p20/full_kernel_policy_comparison.json`.
  This repeats the deployable support-weight experiment on the full available
  OpenAI-labeled manifest split: `278` train query windows / `2780` train
  candidate mixtures, evaluated on the untouched `66` held-out test windows.
  The `66` count is the current test split size in the `380` usable-window
  manifest, not a new cap. Compared with same-seed equal top-5 support sampling,
  the learned kernel/listwise policy slightly improves held-out CRPS and
  coverage but slightly worsens energy: CRPS `0.6679` versus `0.6690`, energy
  `0.9737` versus `0.9722`, and coverage `0.610` versus `0.608`. It creates
  non-uniform support allocation in `46/66` held-out windows. This is
  competitive but not promoted: scaling the labels did not turn the method into
  a clean aggregate improvement over the simple equal-weight mixture.

### Fixed-Start Conditionality

- Tracked case spec:
  `docs/research_protocols/nl_prefix_latent_promoted_specs/fixed_start_narrative_matrix_858b_cases.json`
- Current 192-path narrative bakeoff:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865a_full_narrative_s192/start_conditioned_bakeoff.json`
- Current symmetric 192-path control suite:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865d_full_s192_symmetric/fixed_start_control_suite.json`
- Current start reliability manifest:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_start_reliability_gate_865d_full_s192_symmetric/start_reliability_gate.json`
- Verifier report:
  `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-11_start_reliability_gate_865b.md`

Current supported claim:

- 36 fixed-start narrative runs: 6 narratives by 6 starts;
- fixed-start equality passes;
- support/direction consistency passes;
- direction status: `36/36` pass at 192 generated paths;
- operational validation: `30` pass, `6` warning;
- energy improvement versus persistence: `+18.5%`;
- CRPS improvement versus persistence: `+17.1%`;
- symmetric fixed-start controls at 192 paths:
  - observed narrative median gap: `0.749`;
  - no-narrative start-only median gap: `0.000`;
  - same-narrative repeat median gap: `0.337`, ratio `0.450`;
  - within-run bootstrap median gap: `0.339`, ratio `0.453`;
- starts `0`, `18`, `22`, `40`, and `77` pass current start reliability;
- start `178` remains `warn_high_instability` because its repeat ratio is
  `0.764`, just above the `0.75` threshold.

Interpretation: the narrative is not just selecting the initial level. With the
same start held fixed, different narratives can move the distribution. However,
start reliability is start-specific. The current demo manifest supports five
starts and keeps `178` as a hard-case warning, not a normal recommended start.

Current support-selection rule for the demo/product path:

- support is selected after the user-provided historical/current start is
  fixed;
- the promoted mode is `diverse_topk_narrative_start_checked`;
- high-confidence current/recent grounding directions are used as a support
  gate when available;
- the selected support must be latent-diverse and temporally distinct, with the
  demo using a bridge-local minimum index gap of `30` windows to avoid counting
  overlapping rolling prefixes as separate evidence;
- if fewer distinct supports satisfy the rule, the workflow uses fewer support
  components rather than padding with adjacent overlapping windows;
- this is part of the formal conditioning path, not only a display rule for the
  support table.

### Component-Preserving Support Mixture Candidate

Latest candidate artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_backtest_heldout_29w_s96/component_backtest_report.json`

Independent verifier report:
`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-12_component_preserving_rollout_backtest.md`

The earlier averaged-prefix path can hide narrative conditionality because it
collapses the support pool into one memory before rollout. The current candidate
production path keeps support components separate through generation: decode and
roll out each supported component under the same fixed start, then pool samples
by support weight.

Current paper-facing raw-level diagnostics:

- fixed-start narrative factor panels:
  `paper/narrative_grounded_scenarios/figures/narrative_relevant_factor_panels_943a.png`;
- reference-contrast panels:
  `paper/narrative_grounded_scenarios/figures/narrative_reference_contrast_panels_943a.png`;
- qualitative fixed-start casebook:
  `paper/narrative_grounded_scenarios/figures/narrative_casebook_fixed_start.png`;
- portfolio-impact readout:
  `paper/narrative_grounded_scenarios/figures/narrative_portfolio_impact_fixed_start.png`.

Paper-facing regenerated figures now use the clean incumbent support cases
from:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_943a_start22_incumbent_clean_s64_d400/`

Current evidence:

- six fixed-start narratives, same start `22`, 64 samples per run for the
  clean paper-facing comparison;
- support prior: `diverse_topk_narrative_start_checked`, direction-checked,
  latent-diverse, and non-overlapping at a `30` bridge-local window gap;
- fixed-start max absolute difference: `0.0`;
- pairwise support Jaccard versus the fragile-risk-on reference in the
  qualitative casebook: `0.0` for all five non-reference narratives;
- qualitative raw-level casebook summary:
  `paper/narrative_grounded_scenarios/figures/narrative_qualitative_casebook_summary.json`;
- representative terminal raw-level medians at the same start:
  - fragile risk-on: SPX `2054`, VIX `14.01`, BBB OAS `2.23`,
    1Y ATM IV `0.172`;
  - commodity inflation: SPX `2078`, VIX `13.29`, BBB OAS `2.39`,
    1Y ATM IV `0.166`;
  - dollar liquidity: SPX `2053`, VIX `13.79`, BBB OAS `2.29`,
    1Y ATM IV `0.170`;
- the qualitative casebook is read together with the stress-test scorecard
  above; the paper-facing claim should use the persisted 943a stress metrics
  rather than older single-figure shape diagnostics.
- full held-out cached-narrative backtest over `29` windows at `96` samples:
  - averaged-prefix CRPS improvement: `+12.83%`;
  - component-prefix-mixture CRPS improvement: `+13.15%`;
  - averaged-prefix energy improvement: `+15.60%`;
  - component-prefix-mixture energy improvement: `+15.80%`;
  - component 80% coverage: `0.2957`;
  - averaged 80% coverage: `0.2896`.
- calibration/control gates:
  - fixed-start control report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_decoder_controls_906a_diverse_gap30_s384/component_fixed_start_controls.json`;
  - path audit:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_906a_diverse_gap30_s384/fixed_start_shape_audit.json`;
  - control status: `warning`, because bootstrap noise is close to observed
    narrative differences;
  - same-narrative repeat ratio to observed: `0.137`;
  - start-only null ratio to observed: `0.000`;
  - bootstrap ratio to observed: `1.184`;
  - full path audit status: `warning`, no failures.
- global fan-width calibration candidate:
  - calibration script:
    `experiments/backfill/block_ar/nl_prefix_latent_component_global_calibration.py`;
  - chronological split report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902a_29w_s96/component_global_calibration_report.json`;
  - reverse split report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902b_reverse_29w_s96/component_global_calibration_report.json`;
  - even/odd split report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902c_evenodd_29w_s96/component_global_calibration_report.json`;
  - all three splits selected `alpha=3.5`;
  - evaluation-row improvements versus uncalibrated component:
    chronological coverage `+0.4771`, CRPS `-0.0750`, energy `-0.1910`;
    reverse coverage `+0.4915`, CRPS `-0.0789`, energy `-0.1058`;
    even/odd coverage `+0.4936`, CRPS `-0.0787`, energy `-0.1471`;
  - full 29-window calibrated coverage at `alpha=3.5`: `0.7802`;
  - full 29-window calibrated CRPS improvement versus persistence: `+23.06%`;
  - full 29-window calibrated energy improvement versus persistence: `+28.25%`.
- calibrated demo wiring:
  - verifier report:
    `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-12_global_fan_calibration_wiring.md`;
  - story-smoke option: `--rollout-fan-scale`;
  - app demo default: `rollout_fan_scale=3.5`;
  - cached smoke:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_calibrated_app_smoke_902d_s4/prefix_latent_story_smoke_report.json`;
  - cached smoke generated shape: `[2, 4, 30, 39]`;
  - calibrated and uncalibrated sample mean max difference:
    `1.9073486328125e-06`.
- fixed-start full path-distribution audit:
  - verifier report:
    `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-12_fixed_start_full_path_distribution_audit.md`;
  - audit script:
    `experiments/backfill/block_ar/nl_prefix_latent_fixed_start_shape_audit.py`;
  - uncalibrated `192`-sample report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904f_s192_uncalibrated/fixed_start_shape_audit.json`;
  - calibrated-from-uncalibrated `192`-sample report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904g_s192_calibrated_from_uncalibrated/fixed_start_shape_audit.json`;
  - supporting fixed-start control report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start_controls_904f_s192_uncalibrated/component_fixed_start_controls.json`;
  - broader repeat-control hardening report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start_controls_904i_s192_repeat6x3_uncalibrated/component_fixed_start_controls.json`;
  - broader-repeat uncalibrated path audit:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904i_s192_repeat6x3_uncalibrated/fixed_start_shape_audit.json`;
  - broader-repeat calibrated path audit:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904j_s192_repeat6x3_calibrated/fixed_start_shape_audit.json`;
  - `384`-sample fixed-start control report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start_controls_904k_s384_uncalibrated/component_fixed_start_controls.json`;
  - `384`-sample uncalibrated path audit:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904k_s384_uncalibrated/fixed_start_shape_audit.json`;
  - `384`-sample calibrated path audit:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904l_s384_calibrated/fixed_start_shape_audit.json`;
  - calibration-aware conditionality analysis:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_calibration_conditionality_904m_s384/calibration_conditionality_report.json`;
  - alpha `2.5` calibration-aware conditionality analysis:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_calibration_conditionality_904n_s384_alpha2p5/calibration_conditionality_report.json`;
  - alpha `3.0` calibration-aware conditionality analysis:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_calibration_conditionality_904o_s384_alpha3p0/calibration_conditionality_report.json`;
  - broad cached fixed-start manifest audit for the older soft-topk narrative
    path:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_manifest_audit_904p_s192_uncalibrated/fixed_start_manifest_audit.json`;
  - broad cached fixed-start manifest audit with recovered start-only nulls:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_manifest_audit_904q_s192_with_start_only/fixed_start_manifest_audit.json`;
  - broad cached manifest verifier:
    `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-12_broad_fixed_start_manifest_audit.md`;
  - component broad-manifest TestFlight:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_broad_manifest_testflight_904r_s8/start_conditioned_bakeoff.json`;
  - component start-only broad-manifest TestFlight:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_broad_manifest_start_only_testflight_904r_s8/start_conditioned_bakeoff.json`;
  - component fixed-start `0` bounded audit at `16` samples:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start0_manifest_audit_904s_s16/fixed_start_manifest_audit.json`;
  - component fixed-start `0` bounded audit at `64` samples:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start0_manifest_audit_904t_s64/fixed_start_manifest_audit.json`;
  - component fixed-start `0` bounded audit with start-reliability overlay:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start0_manifest_audit_904u_s64_start_gate/fixed_start_manifest_audit.json`;
  - component fixed-start `22` bounded audit with start-reliability overlay:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start22_manifest_audit_904v_s64_full6_start_gate/fixed_start_manifest_audit.json`;
  - component fixed-start `18` bounded audit at `64` samples with
    start-reliability overlay:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start18_manifest_audit_904w_s64_full6_start_gate/fixed_start_manifest_audit.json`;
  - component fixed-start `22` `192`-sample attribution audit with
    start-reliability overlay:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start22_manifest_audit_904x_s192_full6_start_gate/fixed_start_manifest_audit.json`;
  - component fixed-start `22` `384`-sample attribution audit with
    start-reliability overlay:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start22_manifest_audit_904y_s384_full6_start_gate/fixed_start_manifest_audit.json`;
  - component fixed-start `22` `384`-sample readout audit with
    sample-size-matched observed diagnostics:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start22_manifest_audit_904z_s384_full6_readout/fixed_start_manifest_audit.json`;
  - component fixed-start `40` `192`-sample broadening pilot with
    sample-size-matched observed diagnostics:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start40_manifest_audit_905a_s192_full6_readout/fixed_start_manifest_audit.json`;
  - component fixed-start `77` `192`-sample broadening pilot with
    sample-size-matched observed diagnostics:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start77_manifest_audit_905b_s192_full6_readout/fixed_start_manifest_audit.json`;
  - current component fixed-start stratification artifact:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_start_stratification_905c/component_start_stratification.json`;
  - component support-overlap diagnostics:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_support_diagnostics_905d/component_support_diagnostics.json`;
  - component support-temperature TestFlight for start `22`:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_support_temperature_start22_905e_s96/start_conditioned_bakeoff.json`;
  - component generator-temperature TestFlight and low-temperature audit for
    start `22`:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_generator_temperature_start22_audit_905g_s96_gen0p25/fixed_start_manifest_audit.json`;
  - path metric plot:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904f_s192_uncalibrated/fixed_start_narrative_path_metric_summary.png`;
  - calibrated path metric plot:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904g_s192_calibrated_from_uncalibrated/fixed_start_narrative_path_metric_summary.png`;
  - older terminal-shape-only diagnostic report:
    `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_shape_audit_903a_s96_calibrated/fixed_start_shape_audit.json`;
  - full path-distribution status: `warning`, no failures;
  - terminal shape-only diagnostic status: `fail`;
  - fixed-start max absolute difference: `0.0`;
  - fixed-start control report status: `pass`, with repeat median ratio
    `0.423`, bootstrap median ratio `0.619`, and start-only ratio `0.0`;
  - broader repeat-control hardening status: `pass`, with repeat median ratio
    `0.219` over `18` same-narrative repeat pairs, bootstrap median ratio
    `0.619`, and start-only ratio `0.0`;
  - uncalibrated full-path ratios:
    - repeat-to-observed path energy `0.238`;
    - repeat-to-observed path variance `0.338`;
    - repeat-to-observed path Wasserstein `0.621`;
    - bootstrap-to-observed path energy `0.583`;
    - bootstrap-to-observed path variance `0.643`;
    - bootstrap-to-observed path Wasserstein `0.978`;
    - start-only path energy, variance, and Wasserstein all `0.0`;
  - calibrated full-path ratios:
    - repeat-to-observed path energy `0.405`;
    - repeat-to-observed path variance `0.338`;
    - repeat-to-observed path Wasserstein `0.600`;
    - bootstrap-to-observed path energy `2.856`;
    - bootstrap-to-observed path variance `0.643`;
    - bootstrap-to-observed path Wasserstein `1.331`;
    - start-only path energy, variance, and Wasserstein all `0.0`;
  - broader-repeat uncalibrated full-path ratios:
    - repeat-to-observed path energy `0.148`;
    - repeat-to-observed path variance `0.624`;
    - repeat-to-observed path Wasserstein `0.605`;
    - bootstrap-to-observed path energy `0.583`;
    - bootstrap-to-observed path variance `0.643`;
    - bootstrap-to-observed path Wasserstein `0.978`;
    - start-only path energy, variance, and Wasserstein all `0.0`;
  - broader-repeat calibrated full-path ratios:
    - repeat-to-observed path energy `0.398`;
    - repeat-to-observed path variance `0.624`;
    - repeat-to-observed path Wasserstein `0.632`;
    - bootstrap-to-observed path energy `2.856`;
    - bootstrap-to-observed path variance `0.643`;
    - bootstrap-to-observed path Wasserstein `1.331`;
    - start-only path energy, variance, and Wasserstein all `0.0`;
  - `384`-sample fixed-start control status: `warning`, because terminal
    bootstrap-to-observed median ratio is `0.751`; repeat-to-observed median
    ratio is `0.419`, and start-only ratio is `0.0`;
  - `384`-sample uncalibrated path-audit status: `pass`, with no warnings or
    failures:
    - repeat-to-observed path energy `0.223`;
    - repeat-to-observed path variance `0.304`;
    - repeat-to-observed path Wasserstein `0.533`;
    - bootstrap-to-observed path energy `0.595`;
    - bootstrap-to-observed path variance `0.587`;
    - bootstrap-to-observed path Wasserstein `0.729`;
    - start-only path energy, variance, and Wasserstein all `0.0`;
  - `384`-sample calibrated path-audit status: `warning`, with no failures:
    - repeat-to-observed path energy `0.251`;
    - repeat-to-observed path variance `0.304`;
    - repeat-to-observed path Wasserstein `0.536`;
    - bootstrap-to-observed path energy `2.241`;
    - bootstrap-to-observed path variance `0.587`;
    - bootstrap-to-observed path Wasserstein `1.181`;
    - start-only path energy, variance, and Wasserstein all `0.0`;
  - calibration-aware analysis status: `warning`, decision
    `base_conditioning_passes_calibrated_display_warns`;
  - calibrated observed signal retention versus uncalibrated:
    - path energy `0.266`;
    - path Wasserstein `0.617`;
    - path variance `1.000`;
    - drawdown probability `0.882`;
  - calibrated bootstrap retention versus uncalibrated:
    - path energy `1.000`;
    - path Wasserstein `1.000`;
    - path variance `1.000`;
    - drawdown probability `1.028`;
  - narrow alpha diagnostic, using the existing global calibration report for
    coverage/score context and the `384`-sample fixed-start path gate for
    conditionality:
    - alpha `1.0`: path gate `pass`, coverage `0.286`;
    - alpha `2.5`: path gate `warning`, coverage `0.625`;
    - alpha `3.0`: path gate `warning`, coverage `0.709`;
    - alpha `3.5`: path gate `warning`, coverage `0.777`;
  - broad cached soft-topk manifest audit with recovered start-only nulls:
    status `pass`, with no warnings or failures, over `36` observed cases (`6`
    narratives by `6` fixed starts), `72` repeat cases, `36` start-only cases,
    `90` same-start cross-narrative pairs, `36` same-narrative repeat pairs,
    `36` within-run bootstrap pairs, and `90` start-only pairs:
    - repeat-to-observed path energy `0.381`;
    - repeat-to-observed path variance `0.533`;
    - repeat-to-observed path Wasserstein `0.641`;
    - bootstrap-to-observed path energy `0.294`;
    - bootstrap-to-observed path variance `0.417`;
    - bootstrap-to-observed path Wasserstein `0.671`;
    - start-only-to-observed path energy `0.0`;
    - start-only-to-observed path variance `0.0`;
    - start-only-to-observed path Wasserstein `0.0`;
    - max per-start absolute start difference `0.0`;
    - verifier verdict `PARTIAL`, because this broad suite uses the older
      `decoder_soft_topk_narrative_start_checked_gen_temp_0p50` path, not the
      latest component-preserving rollout candidate;
  - component broad-manifest wiring TestFlight: `2/2` narrative runs and `2/2`
    matching start-only runs completed with status `pass`, using explicit
    `rollout_mixture_mode=component_prefix_mixture`;
  - component fixed-start `0` bounded audit status: `fail` at both `16` and
    `64` samples. At `64` samples, start-only ratios remain `0.0`, but
    repeat-to-observed ratios remain above threshold:
    - repeat-to-observed path energy `0.860`;
    - repeat-to-observed path variance `0.755`;
    - repeat-to-observed path Wasserstein `0.980`;
    - bootstrap-to-observed path energy `3.726`;
    - bootstrap-to-observed path variance `1.007`;
    - bootstrap-to-observed path Wasserstein `1.748`;
  - remaining warning: bootstrap path noise is below threshold for the
    uncalibrated `384`-sample path gate but still exceeds threshold after the
    global fan-width calibration layer; calibrated conditionality should
    therefore remain warning-level until calibration-aware controls are better
    understood.

Interpretation: the component-preserving mixture fixes the visible
conditionality failure without sacrificing held-out distributional quality in the
current 29-window cached-narrative backtest. Independent verification returned
`PARTIAL`: keep this as the candidate production path for narrative-visible
conditionality. The immediate null/repeat control gate now passes. The first
order calibration gap appears to be under-dispersion: a single global
fan-width scale, applied around each ensemble mean, greatly improves coverage
and distributional scores without moving the mean path. The global `alpha=3.5`
layer is now wired into the story-smoke runner and Gradio path as a demo
candidate, with verifier status `PARTIAL`. It is not a change to the
narrative-conditioning mechanism and should not be called a fully promoted
production default until checked on a larger/newer manifest or factor-family
calibration split. The product-facing conditionality gate is now the full
30-day path distribution under an exactly fixed start, not terminal shape-only.
At `192` samples, observed narrative differences exceed same-narrative repeat
and start-only controls on path energy, horizon-wise path variance, path
Wasserstein, and drawdown/rally path metrics. The broader six-case, three-seed
repeat-control hardening strengthens the repeat-control result, especially for
path energy. The `384`-sample uncalibrated audit now passes the full-path gate,
which suggests the earlier bootstrap warning was partly a sampling-budget
artifact. The broad cached manifest audit extends this beyond the handpicked
start-18 case for the older soft-topk narrative path: across all six fixed
starts, same-start cross-narrative path differences remain larger than repeat,
bootstrap, and start-only controls on path energy, variance, and Wasserstein.
Independent verification marked that claim `PARTIAL`: the broad result is
real, but it is not yet the same as a broad promotion for the current
component-preserving rollout candidate. The global fan-width calibrated audit
still warns because calibration amplifies bootstrap path energy/Wasserstein;
treat this as a calibration/readout issue, not proof that the
narrative-conditioning mechanism failed. Terminal shape-only remains a negative
diagnostic: after removing each narrative distribution's own location and
scale, observed terminal shape gaps are not larger than repeat/bootstrap noise.
So the system can currently claim support-grounded uncalibrated full-path
distributional response at fixed start, calibrated fan width with warning-level
conditionality uncertainty, and broad soft-topk manifest evidence that
narrative conditioning is not only a one-case artifact. It still cannot claim
robust terminal shape-only conditionality or fully promoted broad fixed-start
conditionality for the latest component-preserving path until matching broad
component-preserving observed/repeat/start-only controls are run.
The start-conditioned bakeoff runner now makes rollout mode explicit in the
variant contract: legacy soft-topk variants are `averaged_prefix`, while
component variants opt into `component_prefix_mixture`. This prevents future
broad-manifest evidence from silently mixing old and new rollout semantics.
The first current-path broadening attempt found a real hard case: fixed start
`0` fails the component-preserving path-distribution gate at both `16` and `64`
samples. The start-only null is still clean, so the problem is not that the
initial level alone creates the effect. The problem is that narrative separation
under this start is too small relative to repeat/bootstrap path noise. Do not
scale the full component broad grid until this start-specific failure is
explained or the gate is revised with a principled per-start reliability policy.
Initial post-analysis points to start compatibility as the likely mechanism:
all six start-0 component observed runs have operational validation status
`warning` due `large_start_distance` (`start_distance_z` about `22.08`), while
the passing start-18 component runs have operational status `pass`. Direction
checks and support match rates still pass, so direction grounding is not the
obvious failure layer. The next decision should separate "narrative
conditionality failed" from "the selected start is too far from the narrative
query/support manifold for component rollout to be stable."
The manifest audit now exposes a separate `promotion_status` and
start-reliability summary. When `--enforce-start-reliability` is used, observed
operational start warnings become promotion warnings even if the raw path metric
status is otherwise pass. This is a guardrail for production claims: a
conditionality fan can be shown with warnings, but it should not become a
promotion case if the selected start violates the operational start-distance
gate.
However, start compatibility alone is not enough. A second current-path check
on fixed start `22` has pass-level operational start reliability
(`start_distance_z` about `14.40`, no start warnings) but still fails the
`64`-sample path gate: repeat-to-observed path energy `0.772`, variance
`0.700`, Wasserstein `0.899`, and bootstrap-to-observed path energy `3.196`.
The matching fixed-start `18` check also fails at `64` samples despite
pass-level operational start reliability, while the older `384`-sample
start-18 path audit passes. This shows that `64` samples is too small for the
component fixed-start path gate and should not be used for promotion-level
conditionality claims.
Scaling fixed start `22` to `192` and `384` samples removes the repeat-control
failures but leaves bootstrap warnings. At `384` samples, repeat-to-observed
ratios improve to path energy `0.225`, path variance `0.400`, and path
Wasserstein `0.496`; start-only ratios remain exactly `0.0`; bootstrap ratios
remain warning-level with path energy `1.521` and path Wasserstein `0.974`.
Therefore the current component path has real narrative response versus
repeat/null controls for start `22`, but it is still not a clean promotion case
under the within-run bootstrap gate.
The readout diagnostic in `904z` adds sample-size-matched observed pairs
(`384`-sample runs split into `192`-sample halves). This reduces the bootstrap
ratios but does not eliminate the warning: bootstrap-to-sample-matched-observed
ratios are path energy `0.980`, path variance `0.439`, and path Wasserstein
`0.813`. That means the warning is not purely an unfair full-sample versus
half-sample comparison. It is evidence that start `22` has weaker narrative
separation than start `18`, although repeat and start-only controls still show
that the narrative channel is doing something real.
The first additional current-path broadening pilot, fixed start `40` at `192`
samples, also lands at `warning`: repeat-to-observed ratios are path energy
`0.398`, path variance `0.603`, and path Wasserstein `0.729`; start-only ratios
are `0.0`; bootstrap-to-sample-matched-observed ratios are path energy `1.209`,
path variance `0.828`, and path Wasserstein `1.008`. This is not a total
narrative-conditioning failure, but it is another reason not to claim broad
component-path promotion without pass/warning stratification.
The next pilot, fixed start `77` at `192` samples, repeats the same pattern:
repeat-to-observed ratios are path energy `0.437`, path variance `0.378`, and
path Wasserstein `0.671`; start-only ratios are `0.0`; bootstrap-to-sample-
matched-observed ratios are path energy `1.000`, path variance `0.559`, and
path Wasserstein `0.979`. This strengthens the conclusion that the component
path is not dead, but broad production language needs start-level
pass/warning stratification and probably a clearer bootstrap/readout policy.
The current stratification artifact summarizes five available current-path
audits as: one `pass` (`start 18`, `384` samples), three `warning` cases
(`starts 22`, `40`, and `77`, driven by bootstrap/readout warnings), and one
`fail` case (`start 0`, start-incompatibility). This is the clearest current
production statement: the component-preserving support mixture produces
fixed-start narrative response, but the broad claim is not yet uniformly clean
across accepted starts.
Support-overlap diagnostics show that the warning starts are not warning
because every narrative chooses the same support pool. For starts `18`, `22`,
`40`, and `77`, the median weighted support overlap across narrative pairs is
`0.0`, median support-set Jaccard is `0.0`, and the same-top-window rate is
`0.0`. The narrative channel is selecting distinct support components. The
remaining bottleneck is downstream: for some fixed starts, distinct support
components and decoded prefixes do not produce rollout path distributions far
enough above bootstrap/readout noise to clear a promotion gate. The
prefix-to-rollout diagnostic sharpens this: median decoded-prefix L2 is not
smaller for warning starts (`0.706` for start `22`, `0.534` for start `40`,
`0.466` for start `77`) than for the pass start (`0.482` for start `18`), but
generated path energy is much lower for warning starts (`0.019` to `0.021`)
than for start `18` (`0.076`). The generated-energy-per-prefix ratio is about
`0.158` for start `18` but only `0.028` to `0.041` for starts `22`, `40`, and
`77`. That points to fixed-start rollout attenuation/readout sensitivity rather
than text or support selection collapse.
The start-22 support-temperature TestFlight confirms that simple mixture
sharpening is not enough. Lowering support temperature from `0.20` to `0.05`
reduces median effective support size from `7.34` to `3.72`, but median
generated path energy only rises from `0.026` to `0.031`, and
generated-energy-per-prefix only rises from `0.043` to `0.058`. This is a
mechanistic improvement, but it remains far below the clean start-18 ratio
around `0.158`; do not promote support sharpening as a production fix without
full controls and a much larger effect.
Lowering generator temperature is more directly aligned with the readout-noise
diagnosis, and it helps more than support sharpening in the small probe:
start-22 generated-energy-per-prefix rises from `0.042` at generator
temperature `0.50` to `0.074` at `0.25`. But the controlled `96`-sample
gen-temp-`0.25` audit remains `warning`: repeat-to-observed path energy
`0.391`, start-only ratios `0.0`, but bootstrap-to-sample-matched-observed path
energy `1.328` and path Wasserstein `0.952`. Lower rollout temperature is a
promising diagnostic lever, not a clean production fix.
This means the next broadening step should not add a new language model or
support-ranker knob. It should either build a broader current-path component
manifest at production sample budget and report pass/warning stratification, or
improve the bootstrap/readout gate so that promotion distinguishes weak
narrative separation from harmless finite-sample path-resampling noise.
The calibration-aware analysis explains the warning: the global fan-width layer
preserves means but widens sample deviations, retaining only about `26.6%` of
observed path-energy signal and `61.7%` of observed path-Wasserstein signal in
standardized units, while bootstrap path-energy/Wasserstein retention stays
approximately `1.0`. This is a display/readout signal-to-width issue, not a
new reason to abandon the support-grounded mixture.
The narrow alpha check shows the trade-off is structural for this global
mean-preserving fan scale: alpha `1.0` preserves conditionality but undercovers,
while alphas that materially improve coverage (`2.5` to `3.5`) still trigger
calibrated conditionality warnings. A production calibration layer likely needs
to be conditionality-aware, factor-family-aware, or display both base and
calibrated views instead of relying on one global scale as the only product
view.
The generator-response support-weighting frontier should not be promoted yet.
The non-deployable oracle soft-top-5 upper bound improves same-seed equal top-5
on all three metrics over `29` held-out windows: CRPS delta `-0.0033`, energy
delta `-0.0052`, and coverage delta `+0.0011`. The deployable full-train
kernel/listwise policy over `66` held-out windows is only competitive: CRPS
delta `-0.0011`, coverage delta `+0.0024`, but energy delta `+0.0015`.
Decision artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_generator_response_weighting_frontier_918b/generator_response_weighting_frontier.json`.
Interpretation: support weighting has real upper-bound signal, but the current
learned generator-response surface is not good enough to replace the simple
mixture. Continue this branch only with better regime/prototype-aware or
portfolio-risk-aware response labels, while treating rollout/readout noise as
the current product bottleneck.

The first start-aware readout gate prevents the narrow `alpha1p05` readout
from being over-promoted. It joins the existing readout selector with
fixed-start path audits for starts `18`, `22`, `40`, and `77`. The result is
`warning`, with recommendation
`keep_selected_readout_as_local_candidate_only`: start `18` passes, while
starts `22`, `40`, and `77` remain bootstrap/readout warnings under the same
selected readout. The new artifact is
`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_start_aware_readout_gate_919b/start_aware_readout_gate.json`.
This confirms that global fan scaling is not the production fix. The next
candidate should either adapt readout to start-level signal-to-noise or shift
the response target to portfolio risk where the user value is clearer.

The first portfolio-risk response-label audit shifts the product diagnostic into
risk-manager-facing books instead of a single generic portfolio or individual
factor fans. It uses the full906b fixed-start component cases and six normalized
portfolio books: equity beta/carry, credit+duration, dollar-liquidity carry,
commodity inflation, safe-haven hedge, and short volatility. The result is
`warning` with three `pass` books and three `warning` books. Equity beta/carry,
dollar-liquidity carry, and short-volatility books show cross-narrative response
above repeat/bootstrap controls and VaR/ES repeat controls. Credit+duration,
commodity inflation, and safe-haven hedge still show tail-noise limitations.
Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_risk_response_label_audit_920a/portfolio_risk_response_label_audit.json`.
Interpretation: portfolio-risk labels are a better product target than generic
SPX fan-shape changes, but they are not yet a promoted support policy. The next
candidate should test whether a compact portfolio-response-aware support utility
can improve useful conditionality without regressing held-out CRPS/energy or
overfitting a single exposure book.

The portfolio-response support-policy postmortem separates three failure modes:
candidate headroom, label noise, and deployable feature predictability. On the
original `samples=2` per analogue / `6` total samples per candidate held-out
candidate set, the oracle support list still has headroom
(`median top1 minus oracle = 0.1834` reliable portfolio path-score z), but
half-sample label agreement is weak (`Pearson = 0.4363`) and the learned
candidate scorer is indistinguishable from rank order (`pairwise = 0.4943`
versus rank baseline `0.4983`). A local CUDA high-sample rerun at `samples=8`
per analogue / `24` total samples per candidate over the same `660` rows improves
label stability (`Pearson = 0.6447`, `Spearman = 0.6961`) and improves aggregate
scenario quality (`+18.8%` CRPS, `+22.8%` energy versus persistence), but the
learned deployable scorer still does not beat the rank baseline
(`pairwise = 0.5222` versus `0.5242`). Artifacts:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_policy_postmortem_922a/portfolio_response_policy_postmortem.json`
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_policy_postmortem_922c_highsample_full/portfolio_response_policy_postmortem.json`.
Interpretation: do not train another support policy on low-sample labels.
Higher sample count reduces rollout-label noise, but the current feature surface
still cannot reliably predict which candidate mixture will improve the realized
portfolio-response label. The next candidate should either use common-random-
number candidate labeling or learn a richer response surface/prototype label
only after the label-stability gate clears.

The first common-random-number candidate-labeling gate is promising but not yet
a promoted policy. The evaluator now supports
`--common-random-numbers-by-query`, which resets the rollout RNG by query window
so duplicate candidate mixtures for the same query are compared under the same
random stream. On the full `660` held-out candidate rows at `samples=8` per
analogue, half-label Pearson improves to `0.7380`, half-label Spearman improves
to `0.7914`, and the diagnostic learned scorer finally beats rank order on this
test label surface (`pairwise = 0.5481` versus rank baseline `0.5253`; mean
selection regret `0.0713` versus rank `0.0804`). The artifact is
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_policy_postmortem_922e_crn_full/portfolio_response_policy_postmortem.json`.
Interpretation: CRN is the correct next labeling protocol candidate. The next
experiment should regenerate the training candidate-label surface with CRN and
then rerun the learned support-policy scenario evaluation. Do not claim a
production support policy until that train-on-CRN/evaluate-on-CRN path beats the
simple mixture floor.

The CRN-trained support-policy bakeoff does not yet promote a new default. The
CRN training label surface covers `2780` candidate rows across `278` train
queries at `samples=8` per analogue and improves the train candidate-surface
scenario metrics versus persistence (`+24.6%` CRPS, `+25.2%` energy). A compact
kernel-listwise policy trained on those CRN labels barely beats the equal top-5
floor on one same-seed held-out comparison (`seed=922`: CRPS delta `-0.000554`,
energy delta `-0.000619`, reliable portfolio path-score delta `-0.002219`), but
does not survive the immediate repeat (`seed=923`: energy delta `+0.000070`,
reliable portfolio path-score delta `+0.000774`, coverage delta `-0.000427`).
Artifacts:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_train_mixture_labels_922f_crn_full/portfolio_response_label_scenario_report.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_kernel_listwise_922g_crn_train_to_test66/learned_mixture_policy_report.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_kernel_listwise_922h_crn_policy_comparison/portfolio_response_support_policy_comparison.json`,
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_kernel_listwise_922i_crn_policy_comparison_seed923/portfolio_response_support_policy_comparison.json`.
Interpretation: keep CRN labeling as an evaluation protocol improvement, but
do not promote the learned portfolio-response support policy. The remaining
bottleneck is deployable response-feature predictability and seed-stable
support weighting, not the equal support-mixture backbone.

A direct seed-stability diagnostic on the held-out candidate label surface
confirms that the CRN labels are not the main remaining blocker. Re-running the
same `660` held-out candidate mixtures under a second CRN base seed and matching
rows by `query_id` gives global Pearson `0.9157`, Spearman `0.9184`, weighted
within-query pairwise agreement `0.8835`, top-1 match rate `0.6970`, and median
first-to-second top-1 regret `0.0`. Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_label_seed_stability_922j/portfolio_response_label_seed_stability.json`.
Interpretation: the label protocol is good enough to keep using. The reason the
learned policy is still not promoted is more likely the deployable feature
surface and support scorer, or the tiny effect size relative to the equal top-5
floor.

The first support-reliability-prior diagnostic finds a deployable signal that
the generic feature scorer missed. A train-only per-support reliability prior,
computed from query-standardized CRN portfolio-response labels, beats candidate
rank order on the held-out label surface: pairwise `0.5717` versus rank
`0.5253`, and mean selection regret `0.0540` versus rank `0.0804`. Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_feature_sufficiency_923a/portfolio_response_feature_sufficiency.json`.
This says support identity/reliability is a useful response feature.

The corresponding scenario-level policy is competitive but not yet promoted.
The softmax reliability bridge at temperature `1.0` is too weak: it is almost
equal-weighted and fails seed `923` on portfolio response. Hard best-candidate
selection is too strong: it improves reliable portfolio path score by about
`2.17%` on seed `923`, but regresses broad scenario CRPS by `+0.0127` and energy
by `+0.0177`. The calibrated middle point, `candidate_softmax` with temperature
`0.25`, improves reliable portfolio path score on both tested seeds
(`+0.80%` relative reduction on seed `922`, `+0.53%` on seed `923`) and improves
CRPS on both seeds (`-0.000735`, `-0.000478`), but seed `923` has a tiny energy
regression (`+0.000051`) and coverage reduction (`-0.00236`). Artifacts:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_reliability_923d_softmax_t025_comparison_seed922/portfolio_response_support_policy_comparison.json`
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_reliability_923d_softmax_t025_comparison_seed923/portfolio_response_support_policy_comparison.json`.
The next damping check, temperature `0.5`, is worse on seed `923`: it gives only
`+0.03%` reliable portfolio path-score reduction, regresses CRPS by `+0.000080`
and energy by `+0.000385`, and remains `candidate_tradeoff_not_promoted`.
Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_reliability_923e_softmax_t05_comparison_seed923/portfolio_response_support_policy_comparison.json`.
Interpretation: this is the best current candidate lever for making narrative
conditioning more useful in portfolio-risk terms, but it remains
`candidate_tradeoff_not_promoted` until a quality guard, damping rule, or
additional seed check removes the residual energy/coverage trade-off.

An entropy-gated version improves the trade-off but still is not a clean
promotion. The gate uses train-set candidate-probability entropy at the `0.75`
quantile: apply reliability weighting only when the support-reliability
candidate distribution is decisive enough, otherwise leave the equal support
pool unchanged. It activates `47/66` held-out windows and falls back on `19`.
On seeds `922` and `923`, it improves reliable portfolio path score by about
`0.68%` and `0.79%`, and improves CRPS on both. Seed `922` beats the floor
cleanly, while seed `923` remains competitive due to a tiny energy regression
(`+0.000031`). Seed `924` under ordinary non-CRN comparison fails the portfolio
delta, but a paired CRN comparison still improves reliable portfolio path score
by `-0.004764` and CRPS by `-0.000099`, with energy `+0.000487`. Artifacts:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_reliability_923g_t025_entropyq75_comparison_seed922/portfolio_response_support_policy_comparison.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_reliability_923g_t025_entropyq75_comparison_seed923/portfolio_response_support_policy_comparison.json`,
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_reliability_923g_t025_entropyq75_comparison_seed924/portfolio_response_support_policy_comparison.json`,
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_support_reliability_923g_t025_entropyq75_comparison_crn925/portfolio_response_support_policy_comparison.json`.
Interpretation: the reliability prior is a useful portfolio-risk overlay and a
promising route to stronger narrative conditionality, but the production default
should remain equal support mixture until the broad scenario energy/coverage
trade-off is resolved or explicitly exposed as an alternate portfolio-risk view.

The next quality-guarded portfolio-response overlay resolves most of that
trade-off but is still not a clean default. Added
`experiments/backfill/block_ar/nl_portfolio_response_quality_guard_policy.py`
and `test_code/test_924a_nl_portfolio_response_quality_guard_policy.py`. The
policy learns three train-only support priors from historical candidate-mixture
labels: reliable portfolio path utility, CRPS utility, and energy utility. It
scores held-out candidate support mixtures by portfolio utility plus a one-sided
penalty when train-derived CRPS or energy support utility is negative, then
marginalizes candidate probabilities into auditable support weights. No OpenAI
calls and no generator calls are made while building the bridge.

Paired CRN held-out results versus equal support mixture:

- CRN `925`: `candidate_beats_equal_floor`; CRPS `-0.001326`, energy
  `-0.000250`, coverage `+0.001813`, reliable portfolio path score
  `-0.006648`.
- CRN `926`: `candidate_beats_equal_floor`; CRPS `-0.001420`, energy
  `-0.001494`, coverage `+0.004416`, reliable portfolio path score
  `-0.007780`.
- CRN `927`: `candidate_tradeoff_not_promoted`; CRPS `-0.000267`, energy
  `+0.000221`, coverage `+0.001813`, reliable portfolio path score
  `-0.004537`.

An entropy-gated quality guard at train candidate-entropy quantile `0.75`
activated `46/66` held-out windows and fell back on `20/66`, but did not fix the
weak seed: on CRN `927` it improved portfolio path score by `-0.002287`, CRPS by
`-0.000218`, and coverage by `+0.002253`, while energy regressed by
`+0.000434`. Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_quality_guard_924b_t025_entropyq75_comparison_crn927/portfolio_response_support_policy_comparison.json`.

Independent verification agreed that the CRN `925`/`926` evidence supports
candidate-overlay promotion, not production-default promotion. A post-verifier
third seed confirms the same operational conclusion: this is now the strongest
portfolio-risk overlay for narrative conditionality, but equal support mixture
remains the default broad scenario policy until energy stability is cleaner.
Verifier artifact:
`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-20_quality_guard_portfolio_overlay.md`.

Added `experiments/backfill/block_ar/nl_portfolio_response_overlay_stability_gate.py`
and `test_code/test_924c_nl_portfolio_response_overlay_stability_gate.py` to
make this decision reproducible instead of prose-only. On the three CRN
comparison reports (`925`, `926`, `927`), the gate returns
`overlay_portfolio_candidate`: clean seeds `2/3`, portfolio-useful seeds `3/3`,
mean CRPS delta `-0.001004`, mean energy delta `-0.000508`, mean coverage delta
`+0.002681`, mean reliable portfolio path delta `-0.006322`, and max energy
delta `+0.000221`. Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_overlay_stability_gate_924c_t025_3seed/portfolio_response_overlay_stability_gate.json`.

The support-concentration gated quality guard, `924e`, is stronger. It keeps the
same train-only quality-guard score and adds a train-derived final support
concentration gate: if the final support distribution's max weight is at or
below the train `0.25` quantile, fall back to equal support. This gate is
motivated by the held-out postmortem but its threshold is computed only from
train candidate groups. It activates `49/66` held-out windows and falls back on
`17/66`.

Paired CRN results for `924e`:

- CRN `925`: `candidate_beats_equal_floor`; CRPS `-0.000796`, energy
  `-0.000186`, coverage `+0.000648`, reliable portfolio path score
  `-0.004546`.
- CRN `926`: `candidate_beats_equal_floor`; CRPS `-0.001403`, energy
  `-0.001431`, coverage `+0.004170`, reliable portfolio path score
  `-0.005937`.
- CRN `927`: `candidate_beats_equal_floor`; CRPS `-0.000338`, energy
  `-0.000069`, coverage `+0.002461`, reliable portfolio path score
  `-0.003440`.

The stability gate returns `overlay_default_candidate`: clean seeds `3/3`, mean
CRPS delta `-0.000846`, mean energy delta `-0.000562`, mean coverage delta
`+0.002426`, and mean reliable portfolio path-score delta `-0.004641`.
Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_overlay_stability_gate_924e_supportmaxq25_3seed/portfolio_response_overlay_stability_gate.json`.

Independent verification agreed that `924e` is a legitimate default-candidate
support policy under the current three-seed paired-CRN gate, while warning not
to call it the final production default until broader split/time-block
confirmation is done. Verifier artifact:
`docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-20_supportmax_quality_guard_default_candidate.md`.

Initial broader confirmation on the previously excluded middle time block also
passes. I rebuilt cached query/candidate bridges for excluded indices `278-313`
using train-only support, yielding `36` excluded query windows and `360`
candidate mixtures. The same train-derived `924e` policy activates `28/36`
excluded windows and falls back on `8/36`. Two paired CRN comparisons versus
equal support both beat the floor:

- excluded CRN `928`: CRPS `-0.000846`, energy `-0.000700`, coverage
  `+0.002991`, reliable portfolio path score `-0.004410`.
- excluded CRN `929`: CRPS `-0.000997`, energy `-0.001116`, coverage
  `+0.003181`, reliable portfolio path score `-0.000414`.

The excluded-block stability gate returns `overlay_default_candidate`.
Artifacts:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_quality_guard_924f_excluded_t025_supportmaxq25/quality_guard_policy_bridge_report.json`
and
`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_portfolio_response_overlay_stability_gate_924f_excluded_supportmaxq25_2seed/portfolio_response_overlay_stability_gate.json`.

## Not Yet Promoted

- Arbitrary live user narrative plus arbitrary user-supplied joint39 start.
- Full all-training-window OpenAI labeling for every historical window.
- A true Sora/DALL-E-style text-plus-start latent prior.
- Direct text-plus-start concatenation into the generator-memory bridge. The
  cached diagnostics
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_start_memory_diagnostic_875a/text_start_memory_diagnostic.json`
  and
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_text_start_memory_diagnostic_875b_start_weight/text_start_memory_diagnostic.json`
  show that naive and weakly weighted start features degrade held-out
  target-memory cosine and hard-negative directional separation versus
  text-only.
- CLIP/InfoNCE hybrid bridge as default.
- Bounded residual latent refinement as a proven production mechanism.
- Exact historical-window retrieval as a success criterion.
- Episode-level narrative-to-narrative support retrieval as a paper/demo
  default. The isolated 970 branch shows a positive mechanism, but it remains
  exploratory: broad support-bank episode cards improve 66-window CRPS by
  `13.3%` and energy by `15.4%` versus persistence, close to the same-split
  true-history SNI oracle, while local raw-history cards are still rule-based
  and weak for rates/commodity semantic specificity. Evidence summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_evidence_summary_970j/episode_narrative_evidence_summary.json`.

## Required Promotion Additions

Before a future paper/demo/product claim is promoted, it must add:

1. verifier report saved under
   `docs/research_protocols/nl_prefix_latent_verifier_reports/`;
2. tracked case spec or manifest under
   `docs/research_protocols/nl_prefix_latent_promoted_specs/`;
3. artifact paths and exact commands;
4. per-start quality floors, not just mean quality;
5. null/repeat baselines:
   - same narrative repeated with different seeds;
   - shuffled narratives under the same starts;
   - memory-only or no-narrative baseline;
6. clear pass/warn/fail status for start damping.
