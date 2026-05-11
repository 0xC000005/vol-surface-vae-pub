#!/usr/bin/env python
"""Summarize scenario-policy stability across repeated generator seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_policy_stability_874e_summary"
)

METRIC_KEYS = {
    "energy": "energy_score_z_improvement_vs_persistence",
    "crps": "ensemble_crps_z_improvement_vs_persistence",
    "coverage_80": "coverage_80_mean",
    "mean_path_mae": "mean_path_mae_z_improvement_vs_persistence",
}


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


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    raw = float(value)
    return round(raw, 12) if np.isfinite(raw) else None


def parse_report_arg(raw: str) -> tuple[str, int, str]:
    """Parse POLICY:SEED:PATH report args."""

    parts = str(raw).split(":", 2)
    if len(parts) != 3:
        raise ValueError("report args must use POLICY:SEED:PATH")
    policy, seed_text, path = parts
    if not policy:
        raise ValueError("policy name must be non-empty")
    return policy, int(seed_text), path


def extract_method_metrics(
    report: dict[str, Any],
    *,
    method: str = "narrative_generator_topk",
) -> dict[str, float | None]:
    row = report.get("summary", {}).get(method, {})
    if not isinstance(row, dict):
        raise ValueError(f"report summary missing method {method!r}")
    return {
        key: (None if row.get(report_key) is None else float(row[report_key]))
        for key, report_key in METRIC_KEYS.items()
    }


def _mean(values: list[float]) -> float | None:
    return _round(float(np.mean(values))) if values else None


def _std(values: list[float]) -> float | None:
    return _round(float(np.std(values, ddof=0))) if values else None


def summarize_policy_stability(
    report_specs: list[tuple[str, int, str]],
    *,
    baseline_policy: str,
    candidate_policy: str,
    min_energy_gain: float = 0.001,
    min_crps_gain: float = 0.001,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy, seed, path in report_specs:
        metrics = extract_method_metrics(_load_json(path))
        rows.append(
            {
                "policy": policy,
                "seed": int(seed),
                "path": path,
                **metrics,
            }
        )
    policies = sorted({str(row["policy"]) for row in rows})
    by_policy: dict[str, dict[str, Any]] = {}
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        by_policy[policy] = {
            "count": len(policy_rows),
            "seeds": [
                int(row["seed"])
                for row in sorted(policy_rows, key=lambda x: int(x["seed"]))
            ],
        }
        for metric in METRIC_KEYS:
            values = [
                float(row[metric]) for row in policy_rows if row.get(metric) is not None
            ]
            by_policy[policy][f"{metric}_mean"] = _mean(values)
            by_policy[policy][f"{metric}_std"] = _std(values)
    baseline_rows = {
        int(row["seed"]): row for row in rows if str(row["policy"]) == baseline_policy
    }
    candidate_rows = {
        int(row["seed"]): row for row in rows if str(row["policy"]) == candidate_policy
    }
    common_seeds = sorted(set(baseline_rows) & set(candidate_rows))
    paired: list[dict[str, Any]] = []
    for seed in common_seeds:
        base = baseline_rows[seed]
        candidate = candidate_rows[seed]
        paired.append(
            {
                "seed": seed,
                **{
                    f"{metric}_delta": (
                        None
                        if base.get(metric) is None or candidate.get(metric) is None
                        else _round(float(candidate[metric]) - float(base[metric]))
                    )
                    for metric in METRIC_KEYS
                },
            }
        )
    paired_summary: dict[str, Any] = {"common_seed_count": len(common_seeds)}
    for metric in METRIC_KEYS:
        values = [
            float(row[f"{metric}_delta"])
            for row in paired
            if row.get(f"{metric}_delta") is not None
        ]
        paired_summary[f"{metric}_delta_mean"] = _mean(values)
        paired_summary[f"{metric}_delta_std"] = _std(values)
        paired_summary[f"{metric}_delta_positive_count"] = sum(
            1 for value in values if value > 0.0
        )
    energy_delta = paired_summary.get("energy_delta_mean")
    crps_delta = paired_summary.get("crps_delta_mean")
    passes = (
        energy_delta is not None
        and crps_delta is not None
        and float(energy_delta) >= float(min_energy_gain)
        and float(crps_delta) >= float(min_crps_gain)
    )
    return {
        "status": "stable_candidate" if passes else "diagnostic_only",
        "scope_note": (
            "Scenario-policy stability over repeated generator seeds. No OpenAI "
            "calls and no model training."
        ),
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "min_energy_gain": float(min_energy_gain),
        "min_crps_gain": float(min_crps_gain),
        "rows": sorted(rows, key=lambda row: (str(row["policy"]), int(row["seed"]))),
        "by_policy": by_policy,
        "paired": paired,
        "paired_summary": paired_summary,
        "decision": {
            "promote_candidate": bool(passes),
            "reason": (
                "candidate clears paired seed energy/CRPS gains"
                if passes
                else "candidate gain is not stable across repeated generator seeds"
            ),
        },
    }


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):+.2f}%"


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Scenario Policy Stability",
        "",
        f"Status: `{report['status']}`",
        f"Decision: `{report['decision']['reason']}`",
        "",
        "## Paired Summary",
        "",
    ]
    paired = report["paired_summary"]
    for metric in METRIC_KEYS:
        lines.append(
            f"- {metric} delta mean: `{_format_pct(paired.get(f'{metric}_delta_mean'))}`"
        )
    lines.extend(["", "## Policies", ""])
    for policy, row in report["by_policy"].items():
        lines.append(f"### {policy}")
        for metric in METRIC_KEYS:
            lines.append(
                f"- {metric}: mean `{_format_pct(row.get(f'{metric}_mean'))}`, "
                f"std `{_format_pct(row.get(f'{metric}_std'))}`"
            )
        lines.append("")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_stability(args: argparse.Namespace) -> dict[str, Any]:
    specs = [parse_report_arg(value) for value in args.report]
    report = summarize_policy_stability(
        specs,
        baseline_policy=str(args.baseline_policy),
        candidate_policy=str(args.candidate_policy),
        min_energy_gain=float(args.min_energy_gain),
        min_crps_gain=float(args.min_crps_gain),
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "policy_stability_summary.json"
    markdown_path = output_dir / "policy_stability_summary.md"
    report["artifact_paths"] = {
        "report": str(json_path),
        "markdown": str(markdown_path),
    }
    _write_json(json_path, report)
    write_markdown_report(markdown_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--baseline-policy", default="baseline")
    parser.add_argument("--candidate-policy", default="calibrated")
    parser.add_argument("--min-energy-gain", type=float, default=0.001)
    parser.add_argument("--min-crps-gain", type=float, default=0.001)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run_stability(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "paired_summary": report["paired_summary"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
