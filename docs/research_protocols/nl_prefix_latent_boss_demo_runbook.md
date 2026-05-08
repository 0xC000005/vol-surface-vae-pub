# Narrative-Conditioned Scenario Generator Boss Demo Runbook

This runbook is for the current local Gradio demo of the narrative-conditioned
prefix-latent scenario generator. It is written for a boss or risk manager who
cares about the product workflow, provenance, and generated scenario relevance,
not about ML implementation details.

Deployment and artifact packaging boundaries are tracked separately in
`docs/research_protocols/nl_prefix_latent_deployment_readiness.md`.

## Current Demo Claim

A risk manager can provide a market narrative and a starting market state. The
system converts the narrative into condition-only market implications, excludes
forward-looking desired outcomes from conditioning, chooses a supported recent
prefix using a narrative-and-start-compatible analogue mixture, then runs the
frozen joint39 scenario generator to produce a 30-day scenario distribution.

The safe demo path uses cached, previously validated OpenAI grounding. It makes
no OpenAI calls during the presentation.

## Launch

From the repository root:

```bash
uv run python experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py --server-port 7860
```

If port 7860 is busy:

```bash
uv run python experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py --server-port 7861
```

Then open the printed local URL, usually `http://127.0.0.1:7860`.

## Recommended Boss-Demo Path

Use section `6. Prefix-latent live smoke`. This is the current product path.
The earlier top section is useful historical context for grounding and analogue
retrieval, but the prefix-latent section is the one that uses the fixed-start
mixture workflow.

1. In `Cached validated casebook`, select:

   ```text
   Safe-haven gold bid / start 18
   ```

2. Confirm that the story box fills with the safe-haven narrative:

   ```text
   This looks like a safe-haven bid with softer risk appetite: gold is
   rallying, Treasury yields are lower, equities are choppy, and volatility
   remains elevated while the dollar is not providing a clear offset. The
   forward risk is that safe-haven demand becomes a broader risk-off move.
   ```

3. Confirm the casebook status says:

   - cached condition report is available;
   - OpenAI calls are none for this cached run;
   - historical start index is 18.

4. Leave these controls as-is for the first demo run:

   - `Start mode`: `Balanced memory/start support`
   - `Use typed story (OpenAI TestFlight)`: unchecked
   - `Use historical start`: checked
   - `Historical start window index`: `18`
   - `Prefix-latent samples per variant`: `16`

5. Click `Run Prefix-Latent Smoke`.

6. Watch `Prefix-Latent Run Status`. It should show that the run has started,
   then report selected-start status, diagnostic baseline status, and overall
   status.

7. After the run completes, switch `Prefix-latent fan chart factor` from `SPX`
   to an IV cell such as `IV_ATM_3M`. The chart should redraw without rerunning
   the generator.

8. Switch `Prefix-latent start variant` between `All retrieved analogues` and a
   single start variant. This is how to show pooled distribution versus
   individual support behavior.

## What To Say While Showing The Panels

### Story

The story is a current/recent market-condition narrative, not a requested future
path. The model is not being told that the future must be risk-off. It is being
given the present condition: gold up, yields lower, equities choppy, volatility
elevated, and dollar unclear.

### Condition-Only Implications

This table is the conditioning contract. In the safe-haven demo it should show
roughly:

- `GOLD`: up
- `US10Y`: down
- `SPX`: mixed
- `VIX`: up
- `DXY`: flat or unclear

These are used to find supported analogue regimes and condition the prefix
workflow.

### Warning-Only Language

Forward-risk language, such as "safe-haven demand becomes a broader risk-off
move", is shown as a warning and excluded from conditioning. This prevents the
user from secretly prescribing the future. The future distribution is the model
output.

### Proposed Selected Start

The starting state is fixed before the recent-prefix mixture is formed. In this
casebook run, historical start index 18 is used as a stand-in for a
user-specified current market state. In a stricter production workflow, the risk
manager would provide today's joint39 state, or export/edit one of the displayed
historical candidates.

### Historical Start And Support Candidates

These rows are provenance and support diagnostics. The system is not claiming
that one historical window is the answer. The analogue pool is the support prior
used to construct a plausible recent prefix around the fixed starting state.

### Scenario Fan Chart

The fan chart is the product output: a 30-day conditional distribution from the
frozen joint39 generator. It should be discussed as a distribution, not a point
forecast. The important questions are:

- does the scenario distribution react plausibly to the story and starting
  state;
- do the selected IV cells look coherent;
- are the support and warning gates acceptable.

### JSON And Markdown Reports

The markdown report is the readable audit trail. The JSON report is the exact
artifact for reproducibility, including condition source, grounding metadata,
validation gates, analogue labels, and generated path summaries.

## Local Smoke Test For This Demo Path

Before presenting, run:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_cached_smoke.py \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_cached_casebook_smoke_824a_safe_haven \
  --cached-casebook-choice safe_haven_gold_bid:18 \
  --start-mode balanced_memory_start \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

Expected result:

- `status`: `ok`
- `selected_start_status`: `pass`
- `condition_source`: `external_condition_report`
- nonzero fan-chart trace count
- nonzero redraw fan-chart trace count

The latest verified smoke artifact is:

```text
experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_cached_casebook_smoke_824a_safe_haven/gradio_cached_casebook_smoke_summary.json
```

## What Not To Claim

Do not claim that the LLM generates financial scenario paths. It does not.

Do not claim this is a point-forecasting system. Mean-path MAE is secondary;
the stronger evidence is distributional scenario quality and coverage.

Do not claim that forward-looking narrative phrases are conditioning targets.
They are warnings and risk concerns.

Do not claim production readiness yet. The current demo is a convincing local
prototype with cached grounding, fixed-start support, fan-chart redraw, and
auditable artifacts. Production readiness still requires broader live-story QA,
more start-state input validation, persistent case management, deployment
security, and larger validation coverage.

## Current Production Bottlenecks

The main remaining gaps are:

- live OpenAI TestFlight path needs the same boss-demo polish as the cached path;
- the user-supplied joint39 start-state workflow needs stronger validation and
  clearer editing UX;
- mixture weights and residual/refinement diagnostics should be shown more
  explicitly;
- manual browser QA or screenshot capture should be added before an external
  demo;
- the validation set should keep expanding across narratives, starts, and hard
  cases while preserving the same condition-only contract.
