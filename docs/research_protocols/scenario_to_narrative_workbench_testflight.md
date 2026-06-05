# Scenario-To-Narrative Workbench TestFlight

Date: 2026-06-05

## Scope

This TestFlight validates the reverse analyst workbench before demo use. The
workbench normalizes numerical scenario evidence, visualizes factor moves, and
builds strict sidecar and packet artifacts for 14-positive / 14-hard-negative
narrative generation.

## Input Modes

- Historical Joint39 case: selected by `joint39_*` window id plus a cards JSONL path.
- Generated deck artifact: selected by a per-case `prefix_report_snapshot.json` or equivalent report JSON with `generation.terminal_delta_summary`.
- Uploaded factor table: CSV with required `factor,start,end` columns and optional `confidence`.

## Guardrails

- Numeric start/end levels are required for uploaded factor tables.
- Direction-only uploaded tables are rejected for MVP generation.
- Local code computes facts, visualizations, and validation only; narrative prose must be Codex/GPT-authored.
- Partial factor tables are labeled `factor_table_partial`.
- Dry-run packets are diagnostic artifacts and are not valid narrative banks.
- The workbench has no launch auth by default; use `--server-name 0.0.0.0` for remote SSH visibility.

## Verification Commands

```bash
uv run --no-sync pytest \
  test_code/test_nl_scenario_to_narrative_workbench.py \
  test_code/test_nl_scenario_to_narrative_workbench_app.py \
  test_code/test_nl_14_view_variant_pilot.py \
  test_code/test_nl_reverse_caption_scenario_deck.py \
  -q
```

Observed on 2026-06-05: `82 passed in 1.74s`.

```bash
uv run --no-sync python - <<'PY'
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import build_demo

demo = build_demo()
print(type(demo).__name__)
PY
```

Observed on 2026-06-05: `Blocks`.

```bash
uv run --no-sync python -m py_compile \
  experiments/backfill/block_ar/nl_scenario_to_narrative_workbench.py \
  experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py \
  experiments/backfill/block_ar/nl_14_view_variant_pilot.py
```

Observed on 2026-06-05: exit code `0`.

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    normalize_factor_table_csv_text,
    run_workbench_packet,
)

sidecar = normalize_factor_table_csv_text(
    "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\nGOLD,500,540,medium\n",
    scenario_id="dry_run_factor_table",
)
candidates = [
    {
        "window_id": f"joint39_train_{idx:04d}",
        "scenario_title": f"dry candidate {idx}",
        "archetype": "mixed_ambiguous",
        "mechanical_summary": "Mechanical baseline: SPX lower; DXY higher; gold lower.",
        "evidence_used": ["SPX lower", "DXY higher", "gold lower"],
        "contradiction_channels": ["SPX", "DXY", "GOLD"],
        "contradiction_count": 3,
        "agreement_count": 1,
    }
    for idx in range(100, 140)
]
packet = run_workbench_packet(
    sidecar=sidecar,
    negative_candidates=candidates,
    output_dir=Path("experiments/backfill/block_ar/nl_scenario_demo_outputs/scenario_to_narrative_workbench_dry_run"),
    dry_run=True,
)
print(packet.validation["status"])
print(packet.artifact_paths["packet"])
PY
```

Observed on 2026-06-05:

```text
fail
/home/max/Documents/vol-surface-vae-pub/experiments/backfill/block_ar/nl_scenario_demo_outputs/scenario_to_narrative_workbench_dry_run/scenario_narrative_packet.json
```

The `fail` status is expected for the dry-run packet because Codex is not
invoked.

## Launch

Remote-visible launch command:

```bash
uv run --no-sync python experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py \
  --server-name 0.0.0.0 \
  --server-port 7861
```

Local forwarded URL:

```text
http://127.0.0.1:7861/
```
