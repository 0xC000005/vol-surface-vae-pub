# Narrative Prefix-Latent Deployment Readiness

This note defines the current deployment boundary for the risk-manager-facing
narrative-conditioned scenario generator. It is not a claim that the system is
production-ready. It is the packaging contract for turning the verified local
Gradio demo into a reproducible internal demo or a hosted prototype without
committing private or generated artifacts to Git.

## Current Product Contract

The current demo path is:

```text
risk-manager narrative
-> condition-only market implications and warnings
-> accepted historical or user-supplied joint39 starting level
-> narrative-and-start-compatible analogue pool
-> soft top-k prefix support mixture
-> frozen joint39 SNI autoregressive rollout
-> 30-day scenario distribution, fan charts, IV-cell views, and audit report
```

The narrative describes current and recent market conditions. Forward-looking
phrases are routed to warnings and are not used as future targets. The starting
level is fixed before the prefix mixture is formed. The analogue pool is exposed
as provenance and support, not hidden as a one-neighbor generator.

## What Is Tracked In Git

The following are appropriate to commit:

- application code, reusable experiment scripts, and tests;
- runbooks and research protocol documentation;
- `RESEARCH_LOG.md` entries appended with the tail-append helper;
- small human-readable metadata needed to reproduce a run.

The following should stay out of Git:

- `.env`, API keys, and local credential material;
- generated experiment outputs under `experiments/**/nl_scenario_demo_outputs/`;
- model checkpoints under `models/`;
- broad raw or processed data directories under `data/`; specific demo data
  files must be copied only through the external artifact bundle;
- under-review paper drafts, paper backups, and private manuscript artifacts.

The under-review paper and backups must not be uploaded to GitHub or a public
demo artifact store. A demo can cite internal artifact paths without shipping
the manuscript.

## Minimal Artifact Bundle

A deployable demo needs code from this repo plus a small external artifact
bundle. The current local evidence points to these files:

```text
data/
  multi_factor_levels.parquet
  multi_factor_returns.parquet
  spx_vol_surface_history_full_data_fixed.parquet
  vol_surface_with_ret.npz

models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/
  args.json
  best_model.pt
  train_summary.json
  training_history.json

experiments/backfill/block_ar/nl_scenario_demo_outputs/
  manifest_openai_schema_v2_representative_220/
    narrative_adapter.pt
    narrative_label_cache.jsonl
    narrative_pipeline_arrays.npz
    narrative_pipeline_report.json
  manifest_bridge_eval_openai_schema_v2_representative_220/
    bridge_adapter.pt
    bridge_eval_arrays.npz
    bridge_eval_report.json
  prefix_latent_condition_only_report_822a_commodity/
    condition_only_report.json
    condition_only_report_arrays.npz
  prefix_latent_condition_only_report_823a_dollar/
    condition_only_report.json
    condition_only_report_arrays.npz
  prefix_latent_condition_only_report_823b_safe_haven/
    condition_only_report.json
    condition_only_report_arrays.npz
  prefix_latent_boss_demo_pack_829a_live_casebook/
    boss_demo_pack.json
    boss_demo_pack.md
  prefix_latent_gradio_live_api_casebook_828a_three_story/
    gradio_live_api_casebook_summary.json
    gradio_live_api_casebook_summary.md
```

For the current local demo, the full explicit bundle is roughly 52.8 MB across
25 files. These are small enough for a private artifact bundle, but they should
remain ignored by Git because the project also contains much larger generated
trees and the bundle may evolve.

Recommended bundle policy:

- keep the artifact bundle private by default;
- version it with a manifest containing file paths, sizes, hashes, and source
  commit SHA;
- copy only the minimal required files into the hosted environment;
- never bundle `.env`, local OpenAI caches containing secrets, or paper files;
- treat live OpenAI responses and normalized casebook summaries as auditable
  artifacts, not source code.

## Local Demo Command

From the repository root:

```bash
uv run python experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py --server-port 7862
```

The current verified local instance was served on:

```text
http://127.0.0.1:7862
```

If a port is busy, choose another local port. The app supports `--share`, but
for internal review the safer default is a local URL or a private hosted demo.

## Smoke Gates

Cached path, no OpenAI call:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py \
  --url http://127.0.0.1:7862 \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_api_smoke_830a_restarted_cached \
  --casebook-choice safe_haven_gold_bid:18 \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

Live condition-only TestFlight:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py \
  --url http://127.0.0.1:7862 \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_api_live_testflight_830a_restarted \
  --mode live_condition_only \
  --expected-start-index 18 \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

Three-story live casebook:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_live_api_casebook.py \
  --url http://127.0.0.1:7862 \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_api_casebook_828a_three_story \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

Known verified outputs:

- cached API smoke: status `ok`, selected-start `pass`, 8 fan traces, 8 redraw
  traces;
- live one-story API smoke: status `ok`, selected-start `pass`,
  condition-only validation `pass`, one forward-warning item;
- three-story live casebook: 3/3 pass, 5609 OpenAI tokens, condition-only and
  selected-start gates pass for all cases.

## Artifact Preflight

The deployment boundary is now machine-checkable with:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_demo_preflight.py \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_demo_preflight_834a_complete_bundle \
  --run-cached-smoke \
  --gradio-url http://127.0.0.1:7862 \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

The checker validates:

- every required artifact in the minimal bundle exists;
- required artifacts have byte counts and SHA-256 hashes;
- `.env` exists locally only as an ignored file;
- the report records whether `OPENAI_API_KEY` is present without writing the
  key value;
- generated outputs, checkpoints, data, paper files, and `.env` are not staged;
- optional cached Gradio API smoke passes against the supplied URL.

Latest verified preflight artifact:

```text
experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_demo_preflight_834a_complete_bundle/demo_preflight_report.json
```

Summary: status `pass`, 25/25 artifacts present, 52,835,108 required bytes,
zero unsafe staged paths, `.env` ignored, cached smoke `pass`.

## Hosted Prototype Boundary

A hosted prototype should be treated as a private app until the artifact and
data-governance story is settled.

Minimum hosting requirements:

- install dependencies with `uv sync` or an equivalent lockfile-based setup;
- provide `OPENAI_API_KEY` through the platform's secret manager, not `.env`;
- copy the minimal artifact bundle into the same relative paths expected by
  the app, or add explicit path flags before deployment;
- run the cached smoke first, then the live TestFlight only after the cached
  path passes;
- preserve JSON and Markdown reports for every live run;
- limit live call volume and sample count for demos;
- disable public upload of generated reports if they contain sensitive user
  narratives.

For a Hugging Face Space-style deployment, the Gradio app can be wrapped as the
Space entrypoint, but the model/data artifacts should be supplied through a
private artifact repository or private dataset rather than committed into the
GitHub source repo. Space secrets should hold `OPENAI_API_KEY`. The cached
casebook path is the safest default demo mode because it can run without live
API calls during a presentation.

## Clean Staging Dry Run

Use the staging harness to assemble a clean source-plus-artifact tree before
any private hosted prototype:

```bash
uv run python experiments/backfill/block_ar/nl_prefix_latent_demo_staging.py \
  --stage-root /tmp/nl_prefix_latent_demo_stage_832a \
  --run-preflight
```

The staging harness copies tracked source files and then copies only the
explicit 25-file artifact bundle. It excludes local/runtime/private material
from the source copy:

- `.env`;
- `.venv/`;
- `.agents/`;
- `.claude/`;
- `data/`, except files copied through the explicit artifact list;
- `models/`, except files copied through the explicit artifact list;
- `paper/`;
- `results/`;
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/`, except files
  copied through the explicit artifact list;
- root or nested `*.pdf` files.

Latest verified staging manifest:

```text
/tmp/nl_prefix_latent_demo_stage_832a/demo_staging_manifest.json
```

Latest staged cached API smoke:

```text
/tmp/nl_prefix_latent_demo_stage_832a/experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_staged_gradio_api_smoke_833f_complete_bundle/gradio_api_smoke_summary.json
```

Summary: status `pass`, staged preflight `pass`, staged Gradio page `200`,
1,786 source files copied, 25 required artifacts copied, 52,835,108 artifact
bytes, cached API smoke `ok`, selected-start `pass`, 8 support candidates, 8
fan traces, 8 redraw traces, and no `.env`, PDF, `.venv`, `.agents`, `.claude`,
or `paper/` path present in the staged tree. A `data/` directory is present
only because the four explicit base data artifacts are part of the required
bundle.

## Staged Live TestFlight

The staged app can also run a live condition-only OpenAI TestFlight when
`OPENAI_API_KEY` is supplied as a process secret rather than copied as `.env`.
The local simulation used:

```bash
set -a; . /home/max/Documents/vol-surface-vae-pub/.env; set +a
UV_PROJECT_ENVIRONMENT=/tmp/nl_prefix_latent_demo_stage_832a_app_uv_env \
  uv run python experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py \
  --server-port 7863
```

Then:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/nl_prefix_latent_demo_stage_832a_app_uv_env \
  uv run python experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py \
  --url http://127.0.0.1:7863 \
  --output-dir experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_staged_gradio_api_live_testflight_835a_secret_env \
  --mode live_condition_only \
  --expected-start-index 18 \
  --samples 2 \
  --fan-market SPX \
  --redraw-market IV_ATM_3M
```

Latest staged live artifact:

```text
/tmp/nl_prefix_latent_demo_stage_832a/experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_staged_gradio_api_live_testflight_835a_secret_env/gradio_api_smoke_summary.json
```

Summary: status `ok`, condition source `external_condition_report`,
condition-only validation `pass`, one forward-warning item, selected-start
`pass`, overall `pass`, 8 support candidates, 8 fan traces, 8 redraw traces,
and 1,943 OpenAI tokens. The staged tree still had no `.env`, PDF, `.venv`,
`.agents`, `.claude`, or `paper/` path, and the staged server was stopped after
the run.

## Manual Visual QA Checklist

Before showing the demo externally:

- launch the current HEAD Gradio app from a clean shell;
- confirm the readiness evidence panel loads;
- run cached casebook `safe_haven_gold_bid:18`;
- verify the factor fan chart can switch from `SPX` to `IV_ATM_3M` after
  generation without rerunning the generator;
- verify the start-variant control changes the displayed distribution;
- confirm explicit implications, warnings, support candidates, and report paths
  are visible and readable;
- run one live TestFlight with a small sample count;
- archive the smoke summaries and report paths;
- stop the local server after the demo unless it is intentionally left running.

## Production Gaps

The current system is a strong internal prototype, not a production service.
The main gaps are:

- no authentication, authorization, or persistent case management;
- no centralized artifact registry or hash manifest for deployable bundles;
- limited validation of arbitrary user-supplied joint39 start-state JSON;
- no formal rate limiting or budget controls around live OpenAI calls;
- limited visual QA across browsers, screen sizes, and deployment hardware;
- no durable audit database for narrative, warnings, support, generated paths,
  and user decisions;
- no production monitoring for failed grounding, OOD narratives, slow rollout,
  or degenerate scenario samples.

## Next Production Step

The next most useful production-readiness iteration is a manual/browser QA pass
or private hosted prototype using the same 25-file bundle and platform-secret
contract. Capture screenshots for the readiness panel, cached casebook,
condition-only implications, warnings, support candidates, fan chart redraw,
and final audit report.
