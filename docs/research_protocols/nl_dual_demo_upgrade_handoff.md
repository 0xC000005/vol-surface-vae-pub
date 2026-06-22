# NL Dual Demo Upgrade Handoff

Date: 2026-06-11

## Scope

This note records the current handoff state for the two natural-language demos:

- Narrative-to-scenario generator: user enters a market narrative plus starting
  state, and the app produces a 30-day conditional scenario distribution.
- Scenario-to-narrative workbench: user selects or uploads a numerical
  30-day scenario, reviews the factor paths, and generates a validated
  14-positive / 14-hard-negative narrative packet.

The purpose is to make the product/demo choices explicit before handing work to
another agent.

## Narrative-To-Scenario Generator

The Gradio app is:

```text
experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py
```

The demo now defaults to the projected-memory retrieval path for the
prefix-latent generator. The narrative-to-narrative retrieval method is not
used in the demo by default.

Current projected-memory artifacts:

```text
experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_support_audit_990f/projected_memory_14x14_top3_90_bridge_report.json
experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_retrieval_training_openai_holdout_991a_seed1/projected_memory/projected_memory_training_arrays.npz
experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_retrieval_training_openai_holdout_991a_seed1/projected_memory/projected_memory_bridge_best.pt
```

The app uses `text-embedding-3-large` so the OpenAI query embedding dimension
matches the 14+14 projected-memory adapter. The bridge loader reads
`checkpoint["config"]["hidden_dim"]` when present, so the 991a adapter with a
256-wide hidden layer loads without falling back to the older local default.

Recommended examples were replaced with six scenario families, each with short,
medium, and full versions:

- dollar squeeze / commodity liquidation
- safe-haven risk-off
- weak-dollar commodity bid
- rates tightening pressure
- post-stress reflation relief
- split credit-quality stress

The intended robustness demo is to select the short and full versions of the
same family while holding the start level constant, then compare whether the
generated factor table and fan chart retain the same core conditionality. The
public claim should be support-grounded robustness to narrative length, not a
new downstream CRPS or SNI promotion claim.

Important caveat: the projected-memory bridge is wired for product evaluation
and demo iteration, but this document does not claim it supersedes the current
paper/demo quantitative baseline. Earlier 14+14 retrieval experiments improved
the corpus and bridge contract, but downstream scenario-generator promotion must
still be judged by the frozen-SNI support and scenario metrics.

## Scenario-To-Narrative Workbench

The Gradio app is:

```text
experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py
```

The historical path now uses one starting-date selector instead of asking users
for an internal `joint39_train_*` id or a separate end date. The selector is
backed by support-bank metadata and maps the selected start date to the next
30 observed market days.

Example mapping:

```text
selected start date: 2008-09-22
displayed historical period: 2008-09-22 to 2008-10-31
```

The main UI intentionally does not expose "Turbulent period preset",
"Historical ending date", or `joint39_train_*` labels. The internal window id is
still present inside machine-readable sidecar artifacts for reproducibility, but
human selection is start-date based. The UI shows the derived 30-day period
before generation and keeps the numerical scenario visualization visible for
review.

### Factor-Index Correction

On 2026-06-11, historical raw-path extraction was corrected for two Joint39
factor-tail indices:

```text
USDJPY: column 27, not 29
AAA_OAS: column 34, not 36
```

The source `data/multi_factor_levels.parquet` and the 939a support-bank raw
frame are numerically valid. The contaminated layer was the local factor-name
mapping used by the scenario-to-narrative display and support-card narrative
builder. Existing support-card-derived rich narrative banks and downstream
14+14 corpora should be regenerated before they are treated as clean training
truth for `USDJPY` and `AAA_OAS` semantics.

For uploaded scenarios, the MVP still requires numeric start and end levels per
factor. This matches the current pipeline contract: facts and validation are
computed from numeric evidence, while narrative prose must be authored by the
agentic generation path.

## Positive And Negative Interpretation

For scenario-to-narrative packets, "positive narrative" means a correct
description of the selected scenario. It does not mean a bullish or benign
market. A severe 2008 window should therefore produce severe positive
narratives across the 14 views.

The 14 hard negatives are same-style contradictory near-misses. Their role is
contrastive: they should be plausible narratives for different historical
windows whose important factor channels conflict with the selected window.
Depending on the selected target, a hard negative may also sound severe, but it
should be severe in the wrong way.

## Validation Status

Focused checks run after this upgrade:

```bash
uv run --no-sync pytest test_code/test_785a_nl_risk_manager_story_gradio_app.py -q
uv run --no-sync pytest test_code/test_nl_scenario_to_narrative_workbench_app.py -q
python -m py_compile \
  experiments/backfill/block_ar/nl_risk_manager_story_smoke.py \
  experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py \
  experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py
```

Observed results:

```text
test_785a_nl_risk_manager_story_gradio_app.py: 48 passed
test_nl_scenario_to_narrative_workbench_app.py: 23 passed
py_compile: exit code 0
```

These checks verify the projected-memory adapter wiring, example replacement,
start-date-to-window mapping, hidden internal ids, and app callback wiring. They
do not replace a live OpenAI smoke or a downstream scenario-quality evaluation.

## Handoff Checklist

- Keep the narrative-to-scenario demo on projected-memory retrieval unless the
  user explicitly asks to compare narrative-to-narrative retrieval.
- Keep short and full examples paired by scenario family when demonstrating
  length robustness.
- Do not describe the 14+14 bridge as a promoted paper result until frozen-SNI
  downstream metrics justify that claim.
- For scenario-to-narrative, use one start-date selector for human demos and
  reserve `joint39_train_*` ids for machine-readable audit artifacts.
- When reviewing 2008 windows, judge whether every positive style describes the
  bad scenario correctly; do not expect positive examples to sound optimistic.
