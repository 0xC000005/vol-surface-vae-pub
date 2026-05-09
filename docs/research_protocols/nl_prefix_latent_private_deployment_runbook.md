# Narrative Prefix-Latent Private Deployment Runbook

This runbook describes the current internal-demo deployment boundary for the
narrative-conditioned scenario generator. It is intentionally private-first:
the under-review paper, backups, `.env`, raw data sprawl, and generated caches
must not be uploaded to GitHub or a public artifact store.

## Deployment Goal

The private prototype should let a reviewer open a Gradio app, choose a cached
or live condition-only narrative path, inspect warnings/support/scenario fans,
and leave behind an auditable run record.

The production contract is:

```text
narrative or cached condition report
-> accepted current/start state
-> narrative-and-start-compatible support pool
-> frozen joint39 rollout
-> scenario fans and IV-cell views
-> per-run record
-> registry
-> persistent run store
```

Risk-manager text describes current or recent market conditions. Forward-looking
phrases are warning-only and are not used as desired future targets.

## Required Inputs

Use the 25-file artifact bundle documented in
`docs/research_protocols/nl_prefix_latent_deployment_readiness.md`.

Minimum source/artifact inputs:

- tracked source at a known commit;
- the explicit 25-file data/model/demo bundle;
- platform secret `OPENAI_API_KEY` for live condition-only TestFlight;
- optional platform secrets `NARRATIVE_DEMO_AUTH_USER` and
  `NARRATIVE_DEMO_AUTH_PASSWORD`;
- persistent write location for generated demo outputs and the SQLite run store.

Do not copy these into the deployment image or source repo:

- `.env`;
- `.agents/`, `.claude/`, `.venv/`;
- `paper/`, paper backups, arXiv/manuscript PDFs;
- broad `data/`, `models/`, or `results/` trees beyond the explicit bundle;
- raw user narratives outside approved run records and hashed audit manifests.

## Option A: Private VM Or Workstation

This is the lowest-risk internal demo path.

1. Clone the repo at the selected commit.
2. Install dependencies:

```bash
uv sync
```

3. Copy the explicit artifact bundle into the expected relative paths.
4. Export secrets in the shell or service manager, not in Git:

```bash
export OPENAI_API_KEY='...'
export NARRATIVE_DEMO_AUTH_USER='risk'
export NARRATIVE_DEMO_AUTH_PASSWORD='manager'
```

5. Run cached acceptance before any live OpenAI demo:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_private_prototype_acceptance.py \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/private_acceptance_vm_cached \
  --port 7867 \
  --samples 2
```

6. Launch the app for the reviewer:

```bash
uv run python experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py \
  --server-name 0.0.0.0 \
  --server-port 7860 \
  --require-auth
```

7. After the demo, run registry/store ingestion if the app produced new run
   records:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_demo_run_registry.py \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/private_demo_registry

uv run python experiments/backfill/block_ar/nl_prefix_latent_run_store.py \
  --sqlite experiments/backfill/block_ar/nl_scenario_demo_outputs/private_demo_store/demo_run_store.sqlite \
  --summary-json experiments/backfill/block_ar/nl_scenario_demo_outputs/private_demo_store/demo_run_store_summary.json
```

## Option B: Private Gradio/Hugging Face Space

Use only a private Space or private equivalent.

Recommended boundary:

- app source from Git;
- bundle artifacts supplied through a private artifact mechanism;
- `OPENAI_API_KEY` and auth secrets configured in platform secrets;
- generated outputs written to a mounted persistent volume or private dataset;
- cached casebook mode as the default presentation path;
- live OpenAI mode used only after cached acceptance passes.

The Space entrypoint can run:

```bash
python experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py \
  --server-name 0.0.0.0 \
  --server-port 7860 \
  --require-auth
```

Before exposing the private Space to a reviewer, run the acceptance harness in a
staging instance or equivalent shell. If the platform cannot run the full
acceptance harness because it manages the server process itself, run the pieces
individually against the private URL:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py \
  --url https://PRIVATE_APP_URL \
  --require-auth \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M

uv run python experiments/backfill/block_ar/nl_prefix_latent_browser_qa.py \
  --url https://PRIVATE_APP_URL \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/private_space_browser_qa \
  --viewport desktop:1440x1200 \
  --viewport mobile:390x900
```

## Acceptance Criteria

A private demo is acceptable only if all of the following pass:

- preflight confirms required artifacts exist;
- cached Gradio API smoke returns `ok`;
- browser QA captures nonblank desktop and mobile screenshots;
- run registry passes `qa_packet_pass`, `audit_manifest_pass`,
  `browser_render_pass`, and `prefix_run_record_pass`;
- run store ingests at least one run record;
- server is stopped or intentionally managed by the host after the acceptance
  run;
- no `.env`, paper files, manuscript PDFs, or broad generated trees are staged
  or bundled.

## Current Verified Command

Latest local cached acceptance:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_private_prototype_acceptance.py \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_private_acceptance_844a_cached \
  --port 7867 \
  --samples 2
```

Result: status `pass`; `app_http_ready`, `cached_smoke`, `browser_qa`,
`run_registry`, and `run_store` all passed.

## Remaining Production Gaps

This is ready for an internal private demo, not full production.

Remaining gaps:

- managed authentication and authorization beyond Gradio basic auth;
- centralized private artifact store for the 25-file bundle;
- durable managed database replacing local SQLite;
- production budget/rate controls for live OpenAI calls;
- browser QA across the actual hosted platform and reviewer devices;
- retention policy for run records, generated scenarios, and user decisions;
- monitoring/alerting for grounding failures, OOD narratives, slow rollouts,
  and degenerate scenario samples.
