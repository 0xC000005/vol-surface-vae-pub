#!/usr/bin/env python
"""Create a boss-ready evidence pack for the narrative scenario demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_VALIDATION_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_conditioned_bakeoff_820b_expanded_temp050_s16/"
    "start_conditioned_bakeoff.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_boss_demo_pack_821a"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):+.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def validation_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    variants = report.get("variant_summary", [])
    best = variants[0] if isinstance(variants, list) and variants else {}
    if not isinstance(best, dict):
        best = {}
    rows = [row for row in report.get("rows", []) if isinstance(row, dict)]
    improved_crps = 0
    warning_rows = 0
    pass_rows = 0
    for row in rows:
        status = str(row.get("validation_operational", ""))
        if status == "pass":
            pass_rows += 1
        if status == "warning":
            warning_rows += 1
        metrics = row.get("scenario_metrics", {})
        if isinstance(metrics, dict):
            value = metrics.get("ensemble_crps_z_improvement_vs_persistence")
            try:
                improved_crps += int(float(value) > 0.0)
            except (TypeError, ValueError):
                pass
    return {
        "case_count": int(report.get("case_count", len(rows)) or len(rows)),
        "run_count": int(report.get("run_count", len(rows)) or len(rows)),
        "case_set": str(report.get("case_set", "")),
        "variant_set": str(report.get("variant_set", "")),
        "best_variant": str(best.get("variant_name", "")),
        "status_counts": best.get("operational_status_counts", {}),
        "pass_rows": int(pass_rows),
        "warning_rows": int(warning_rows),
        "improved_crps_rows": int(improved_crps),
        "mean_energy_score_z": best.get("mean_energy_score_z"),
        "mean_ensemble_crps_z": best.get("mean_ensemble_crps_z"),
        "mean_energy_improvement_vs_persistence": best.get(
            "mean_energy_improvement_vs_persistence"
        ),
        "mean_crps_improvement_vs_persistence": best.get(
            "mean_crps_improvement_vs_persistence"
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    snapshot = summary["validation_snapshot"]
    lines = [
        "# Narrative-Conditioned Scenario Generator Evidence Pack",
        "",
        "## Product Claim",
        "",
        (
            "A risk manager can enter a market narrative and either accept a "
            "recommended starting level or provide a joint39 starting level. "
            "The system converts the narrative into grounded current/recent "
            "market implications, forms a narrative-and-start-compatible "
            "analogue-mixture prefix, and rolls the frozen joint39 generator "
            "forward with calibrated sampling."
        ),
        "",
        "## Workflow",
        "",
        "1. Parse the narrative into conditioning implications and non-conditioning forward-risk warnings.",
        "2. Fix the initial joint39 level before building the 30-day prefix mixture.",
        "3. Build a soft top-k analogue mixture conditioned on both narrative memory and fixed start.",
        "4. Decode the recent-prefix object and run the frozen SNI generator autoregressively.",
        "5. Show scenario fans, analogue support, IV-cell views, pass/warning/fail gates, and distributional metrics.",
        "",
        "## Current Evidence",
        "",
        f"- Validation report: `{summary['validation_report']}`",
        f"- Case set: `{snapshot['case_set']}`",
        f"- Runs: `{snapshot['run_count']}`",
        f"- Best variant: `{snapshot['best_variant']}`",
        f"- Status counts: `{json.dumps(snapshot['status_counts'], sort_keys=True)}`",
        f"- Rows improving CRPS vs persistence: `{snapshot['improved_crps_rows']}/{snapshot['run_count']}`",
        f"- Mean energy z: `{_fmt_float(snapshot['mean_energy_score_z'])}`",
        f"- Mean CRPS z: `{_fmt_float(snapshot['mean_ensemble_crps_z'])}`",
        f"- Mean energy improvement vs persistence: `{_fmt_pct(snapshot['mean_energy_improvement_vs_persistence'])}`",
        f"- Mean CRPS improvement vs persistence: `{_fmt_pct(snapshot['mean_crps_improvement_vs_persistence'])}`",
        "",
        "## Warning Semantics",
        "",
        (
            "`pass` means the run is supported and calibrated under current "
            "gates. `warning` means the run can still be useful, but the UI "
            "must show support/shift caveats. In the current expanded "
            "validation, warning rows still improved CRPS versus persistence, "
            "so warning is a trust caveat rather than an automatic scenario "
            "metric failure."
        ),
        "",
        "## Demo Talking Points",
        "",
        "- This is not an LLM inventing future paths; the LLM only grounds the narrative into conditioning language.",
        "- The numerical scenario paths come from the frozen joint39 generator and calibrated rollout.",
        "- Historical analogues are support/provenance for the recent prefix, not a single nearest-neighbor replay.",
        "- The risk manager can override the starting level; the prefix mixture is rebuilt after the start is fixed.",
        "- The demo should show the narrative, extracted implications, warnings, analogue weights, fan charts, selected IV cells, and JSON report.",
        "",
        "## Next Validation Step",
        "",
        (
            "Scale beyond the cached three narratives by adding more "
            "condition-only narratives, then rerun the calibrated fixed-start "
            "validation grid with the same pass/warning semantics."
        ),
    ]
    return "\n".join(lines)


def build_demo_pack(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _load_json(args.validation_report)
    summary = {
        "status": "ok",
        "scope_note": (
            "Boss-ready evidence pack built from existing calibrated validation "
            "artifacts. No OpenAI calls or model rollouts are made."
        ),
        "validation_report": str(args.validation_report),
        "validation_snapshot": validation_snapshot(validation),
        "artifact_paths": {
            "summary_json": str(output_dir / "boss_demo_pack.json"),
            "summary_markdown": str(output_dir / "boss_demo_pack.md"),
        },
    }
    _write_json(summary["artifact_paths"]["summary_json"], summary)
    Path(summary["artifact_paths"]["summary_markdown"]).write_text(
        render_markdown(summary).rstrip() + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_demo_pack(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary_json": summary["artifact_paths"]["summary_json"],
                "summary_markdown": summary["artifact_paths"]["summary_markdown"],
                "mean_crps_improvement_vs_persistence": summary["validation_snapshot"][
                    "mean_crps_improvement_vs_persistence"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
