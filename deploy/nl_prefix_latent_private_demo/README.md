# Private Narrative Demo Container Wrapper

This wrapper is for a private internal prototype of the narrative-conditioned
scenario generator. It intentionally excludes local secrets, paper files, broad
data/model trees, and generated outputs from the image build context.

## Build

From the repository root:

```bash
docker build \
  -f deploy/nl_prefix_latent_private_demo/Dockerfile \
  -t nl-prefix-latent-private-demo:local \
  .
```

The Dockerfile-specific ignore file excludes:

- `.env`;
- `.agents/`, `.claude/`, `.venv/`;
- `autoresearch-session/`;
- `paper/` and `*.pdf`;
- broad `data/`, `models/`, `results/`;
- generated `nl_scenario_demo_outputs/`.

## Run

Mount the explicit 25-file artifact bundle into the same relative paths expected
by the app, and mount a persistent output volume for run records, registry
outputs, screenshots, and the SQLite run store.

Example shape:

```bash
docker run --rm \
  -p 7860:7860 \
  -e OPENAI_API_KEY \
  -e NARRATIVE_DEMO_AUTH_USER \
  -e NARRATIVE_DEMO_AUTH_PASSWORD \
  -v /private/artifacts/data:/app/data:ro \
  -v /private/artifacts/models:/app/models:ro \
  -v /private/artifacts/nl_scenario_demo_outputs:/app/experiments/backfill/block_ar/nl_scenario_demo_outputs \
  nl-prefix-latent-private-demo:local
```

For an internal cached-only review, `OPENAI_API_KEY` can be omitted if live
condition-only mode will not be used. Keep auth enabled for any shared host.

## Acceptance

Run the cached acceptance harness against the container URL before a demo:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py \
  --url http://127.0.0.1:7860 \
  --require-auth \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M

uv run python experiments/backfill/block_ar/nl_prefix_latent_browser_qa.py \
  --url http://127.0.0.1:7860 \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/private_container_browser_qa \
  --viewport desktop:1440x1200 \
  --viewport mobile:390x900
```

Then build the registry and run store using the mounted output directory.

## Production Note

This is not a full production deployment. It is a reproducible private-demo
wrapper. Full production still needs managed auth, a managed artifact store, a
managed database, retention policy, and monitoring.
