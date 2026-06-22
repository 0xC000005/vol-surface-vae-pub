# NL EpisodeCardV3 Full Regeneration Prep

Last updated: 2026-06-01

## Status

Prep is partially reset after the 981 retraction. Do not run retrieval,
backtests, or conditionality claims from locally generated EpisodeCardV3 prose.
The next valid regeneration must use direct Codex/GPT authoring for every
searchable narrative view.

The current paper/demo default remains unchanged:

```text
professional narrative
+ grounding sidecar
+ fixed accepted start
-> nearest-similar top3/90 support posterior
-> component-preserving frozen SNI rollout
```

The EpisodeCardV3 branch is a separate research lane. It may become the next
support selector only after it improves fixed-start narrative conditionality
over the start-only baseline while staying competitive on historical scenario
quality.

## Why This Branch Exists

The projected-memory selector compresses a rich narrative into a text embedding
and then into a single SNI final-hidden memory vector. That can lose information
about mechanism, sequencing, ambiguity, and partial evidence. EpisodeCardV3
tests a more human-style support search:

```text
user narrative <-> historical episode narratives
```

The branch keeps numerical guardrails after retrieval:

```text
grounded claim agreement
+ fixed-start compatibility
+ temporal diversity
+ SNI support compatibility
+ top3/90 component-preserving rollout
```

## Approved Narrative Standard

Use the corrected `980a` Codex-authored multi-format packet as the narrative
quality standard for full regeneration:

- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/review_ready_single_period.md`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_review_ready.md`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_testflight_980a/multiformat_codex_report.json`

Earlier `974a`-`979a` packets are superseded and must not seed full
regeneration.

Hard guardrail: local deterministic/template/rule-based prose generation is
banned for all narrative artifacts, including smoke tests and mechanism tests.
This explicitly bans local EpisodeCardV3-style narrative generators. Searchable
narrative views must be authored through Codex's agentic framework, direct
GPT/Codex authoring calls, or trusted human/source text. Local code may compute
market facts, support metadata, supported-angle labels, leakage checks, and
mechanical evidence tables. It must not synthesize searchable narrative prose.

Required generated views:

- `old_factor_baseline`: Codex/GPT-authored mechanical factor baseline for
  audit only, or a non-searchable structured evidence table if generated
  locally.
- `sparse_user_prompt`: declarative, one-to-two-channel user-like market
  description.
- `weekly_risk_monitor`: concise institutional market-monitor tone.
- `institutional_risk_committee_note`: committee-level risk note that works for
  stress and non-stress regimes.
- `mechanism_first_memo`: mechanism/transmission-first memo.
- `full_risk_manager_memo`: complete professional memo with ambiguity and a
  no-forecast caveat.

## Windowing Policy

Use mixed windowing instead of a single cadence:

- Rich institutional narratives: every 15 trading days / half-overlap 30-day
  prefixes.
- Sparse user-like prompts and any technical-evidence narrative view: every
  eligible daily 30-day prefix, directly Codex/GPT-authored if searchable.
- Final support bank: keep every eligible daily support prefix for SNI rollout,
  start-fit checks, and nearby refinement.

This avoids near-duplicate expensive rich memos while keeping dense daily
support coverage where the numerical generator needs it.

## Required Method Comparison

The full run should compare these selectors under matched starts, narratives,
seeds, top3/90 rollout, and sample counts:

1. Incumbent projected-memory selector.
2. Start-only terminal-state selector.
3. Local-control narrative-to-narrative selector using lexical / local
   hashed-vector / mechanism features over Codex/GPT-authored text.
4. True semantic narrative-to-narrative selector using text embeddings.
5. Hybrid selector combining narrative match and start-fit.
6. Optional Codex/agentic reranker over a shortlist only if deterministic
   retrieval is semantically weak.
7. Optional online-enriched episode cards if the Codex/GPT-authored corpus
   still fails sparse-query or mechanism-match checks.

The current deterministic hybrid candidate from the prior 66-window evidence is:

```text
text_weight = 0.25
start_weight = 0.75
```

Treat that as the first hybrid setting to confirm, not as a permanent default.

## Product Gate

The primary product question is conditionality, not raw CRPS alone:

```text
same accepted start
+ different professional or sparse user narratives
-> different selected support episodes
-> different 30-day scenario distributions
```

Required evidence:

- support overlap versus start-only baseline;
- fixed-start factor KS and path-energy distance versus start-only;
- narrative-relevant fan charts and baseline-relative summary tables;
- grounded-claim pass rate on selected supports;
- start-only and shuffled-narrative null controls;
- CRPS, Energy Score, and coverage as historical-quality guardrails;
- qualitative spot checks for gold, rates, commodity, dollar, risk-on, and
  risk-off narratives.

Do not promote the branch if narrative lift appears only because of a quality
regression or an unreproducible LLM rerank.

## Exact Prep Commands Confirmed

The previous local V3 card-generation command is intentionally removed. The
script now fails closed because deterministic/template EpisodeCardV3 prose is
not allowed for any narrative artifact.

Codex-authored multi-format generation skeleton:

```bash
uv run python experiments/backfill/block_ar/nl_episode_card_v3_codex_testflight.py run-multiformat \
  --source-cards-jsonl experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_support_bank_cards_all_972b_tight_taxonomy/episode_narrative_support_cards.jsonl \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_multiformat \
  --max-cases 4010 \
  --min-window-gap 0 \
  --model gpt-5.5 \
  --reasoning-effort xhigh \
  --batch-size 2 \
  --continue-on-error
```

Narrative-to-narrative retrieval:

```bash
uv run python experiments/backfill/block_ar/nl_episode_narrative_retrieval.py \
  --cards-jsonl <multiformat_episode_cards.jsonl> \
  --query-json <query_set.json> \
  --output-dir <retrieval_output_dir> \
  --top-k 8 \
  --temporal-gap 30 \
  --methods lexical dense hybrid
```

Bridge-style reports consumed by the frozen-SNI evaluator:

```bash
uv run python experiments/backfill/block_ar/nl_episode_narrative_bridge_report.py \
  --selector-mode start_only \
  --cards-jsonl <multiformat_episode_cards.jsonl> \
  --support-report experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a/support_bank_report.json \
  --support-arrays experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz \
  --output-dir <start_only_bridge_dir> \
  --top-k 8 \
  --temporal-gap 30 \
  --max-query-windows 66

uv run python experiments/backfill/block_ar/nl_episode_narrative_bridge_report.py \
  --selector-mode hybrid_start_text \
  --cards-jsonl <multiformat_episode_cards.jsonl> \
  --support-report experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a/support_bank_report.json \
  --support-arrays experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz \
  --output-dir <hybrid_bridge_dir> \
  --method hybrid \
  --top-k 8 \
  --temporal-gap 30 \
  --text-candidate-k 128 \
  --text-weight 0.25 \
  --start-weight 0.75 \
  --max-query-windows 66
```

Scenario-level frozen-SNI evaluation:

```bash
uv run python experiments/backfill/block_ar/nl_scenario_level_evaluation.py \
  --bridge-report <bridge_dir>/episode_retrieval_bridge_report.json \
  --bridge-arrays <bridge_dir>/episode_retrieval_bridge_arrays.npz \
  --output-dir <scenario_eval_dir> \
  --top-k 3 \
  --samples 16 \
  --support-sampling-mode field_weight \
  --seed 4 \
  --common-random-numbers-by-query
```

Conditionality lift audit:

```bash
uv run python experiments/backfill/block_ar/nl_episode_narrative_conditionality_lift.py \
  --narrative-report <narrative_eval_dir>/scenario_level_eval_report.json \
  --narrative-arrays <narrative_eval_dir>/scenario_level_eval_arrays.npz \
  --start-only-report <start_only_eval_dir>/scenario_level_eval_report.json \
  --start-only-arrays <start_only_eval_dir>/scenario_level_eval_arrays.npz \
  --output-dir <lift_output_dir>
```

## Prep Fix Applied

The isolated local V3 prose generator is disabled and must not be used. The
Codex multi-format path now writes `multiformat_episode_cards.jsonl` whose
retrieval views are direct Codex/GPT-authored fields, not locally reworded
templates.

## Next Goal Prompt

Use this as the next persistent autoresearch goal:

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
