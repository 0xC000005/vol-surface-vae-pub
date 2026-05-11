# Narrative-Conditioned Scenario Generator Boss Demo Runbook

This runbook is for the current local Gradio demo of the narrative-conditioned
prefix-latent scenario generator. It is written for a boss or risk manager who
cares about the product workflow, provenance, and generated scenario relevance,
not about ML implementation details.

Deployment and artifact packaging boundaries are tracked separately in
`docs/research_protocols/nl_prefix_latent_deployment_readiness.md`.

## Current Demo Claim

A risk manager can provide a market narrative and a starting market state. The
system refreshes the OpenAI grounding and text-memory condition, excludes
forward-looking desired outcomes from conditioning, builds a supported recent
prefix using a narrative-and-start-compatible analogue mixture around the fixed
start, then runs the frozen joint39 scenario generator to produce a 30-day
scenario distribution.

The production demo path is live: it should call OpenAI for the typed narrative
and then generate from the explicitly selected historical start. Cached smokes
remain useful for developer regression checks, but they are not the boss-demo
story.

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

Use the main screen. It has three required inputs: narrative, historical start
window index, and scenario factor. The risk manager must select or provide the
starting level before the support mixture is built.

1. Paste this safe-haven narrative into `Risk-manager narrative`:

   ```text
   This looks like a safe-haven bid with softer risk appetite: gold is
   rallying, Treasury yields are lower, equities are choppy, and volatility
   remains elevated while the dollar is not providing a clear offset. The
   forward risk is that safe-haven demand becomes a broader risk-off move.
   ```

2. Set `Historical start window index` to:

   ```text
   18
   ```

   This is a reliability-checked demo start. Other currently supported demo
   starts are `0`, `22`, `40`, and `77`; start `178` is intentionally retained
   only as a high-instability hard case.

3. Keep `Scenario factor` at `SPX` for the first run.

4. Click `Generate 30-Day Scenarios`.

5. Watch `Scenario Workflow Status`. It should show the run start, then report:

   - `Story support`;
   - `Result note`;
   - `Start reliability`;
   - next step guidance.

6. After the run completes, switch `Scenario factor` from `SPX` to an IV cell
   such as `IV ATM 3M, K=1.00`. The chart should redraw without rerunning the
   generator.

7. Open `Audit details` only if the audience asks why the run was accepted or
   warned. The default presentation should stay on the story, selected starting
   level, fan chart, and terminal summary.

## What To Say While Showing The Panels

### Story

The story is a current/recent market-condition narrative, not a requested future
path. The model is not being told that the future must be risk-off. It is being
given the present condition: gold up, yields lower, equities choppy, volatility
elevated, and dollar unclear.

### Grounded Implications

This table is an audit sidecar, not the entire condition. In the safe-haven demo
it should show roughly:

- `GOLD`: up
- `US10Y`: down
- `SPX`: mixed
- `VIX`: up
- `DXY`: flat or unclear

These claims check directionality and help support ranking, while the full
narrative remains the main story channel.

### Warning-Only Language

Forward-risk language, such as "safe-haven demand becomes a broader risk-off
move", is shown as a warning and excluded from conditioning. This prevents the
user from secretly prescribing the future. The future distribution is the model
output.

### Selected Starting Level

The starting state is fixed before the recent-prefix mixture is formed. In this
demo, historical start index 18 is used as a stand-in for today's joint39 market
state. In a stricter production workflow, the risk manager would provide today's
joint39 state directly.

### Support Candidates

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

### Reports

The markdown report is the readable audit trail. The JSON report is the exact
artifact for reproducibility, including condition source, grounding metadata,
validation gates, analogue labels, and generated path summaries.

## Local Smoke Tests

For a no-OpenAI developer regression check, run:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_cached_smoke.py \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_cached_smoke_868a_reliability_fixed \
  --cached-casebook-choice safe_haven_gold_bid:18 \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

Expected result:

- `status`: `ok`
- `selected_start_status`: `pass`
- `start_reliability_status`: `pass`
- `condition_source`: `external_condition_report`
- nonzero fan-chart trace count
- nonzero redraw fan-chart trace count

The latest verified smoke artifact is:

```text
experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_cached_smoke_868a_reliability_fixed/gradio_cached_casebook_smoke_summary.json
```

For the production API path, launch the app and run a live smoke:

```bash
uv run python experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py --server-port 7860
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py \
  --url http://127.0.0.1:7860 \
  --mode live_condition_only \
  --expected-start-index 18 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

This live smoke calls OpenAI for the typed narrative. Use it before an external
demo, not for every local code edit.

## What Not To Claim

Do not claim that the LLM generates financial scenario paths. It does not.

Do not claim this is a point-forecasting system. Mean-path MAE is secondary;
the stronger evidence is distributional scenario quality and coverage.

Do not claim that forward-looking narrative phrases are conditioning targets.
They are warnings and risk concerns.

Do not claim production readiness yet. The current demo is a convincing local
prototype with live narrative conditioning, explicit fixed-start support,
fan-chart redraw, and auditable artifacts. Production readiness still requires
broader live-story QA, more start-state input validation, persistent case
management, deployment security, and larger validation coverage.

## Current Production Bottlenecks

The main remaining gaps are:

- live OpenAI API smoke should be run against the simplified fixed-start app
  before external presentation;
- the user-supplied joint39 start-state workflow still needs stronger validation
  and clearer editing UX;
- mixture weights and residual/refinement diagnostics should be shown more
  explicitly;
- manual browser QA or screenshot capture should be added before an external
  demo;
- the validation set should keep expanding across narratives, starts, and hard
  cases while preserving the same condition-only contract.
