from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PRESENT_STATE = Path("results/world/present_state_probe_head124.json")


def _load_json(root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def _metric(
    probe: dict[str, Any],
    feature: str,
    target: str,
    metric: str,
) -> float:
    return float(probe["probe_metrics"][feature]["targets"][target][metric])


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def assess_state_content_gate(root: Path) -> dict[str, Any]:
    probe = _load_json(root, DEFAULT_PRESENT_STATE)
    feature = "head070_default_barlow_last"
    hard_feature = "head123_hard_barlow_last"
    raw_surface = "raw_surface_last"
    raw_upper = "raw_geometry_last_upper_bound"

    iv_mse = _metric(probe, feature, "iv_surface", "mse")
    raw_iv_mse = _metric(probe, raw_surface, "iv_surface", "mse")
    factor_return_r2 = _metric(probe, feature, "factor_return", "r2")
    factor_level_mse = _metric(probe, feature, "factor_level", "mse")
    raw_factor_level_mse = _metric(probe, raw_surface, "factor_level", "mse")
    upper_factor_level_mse = _metric(probe, raw_upper, "factor_level", "mse")
    side_mse = _metric(probe, feature, "vol_side_channel", "mse")
    raw_side_mse = _metric(probe, raw_surface, "vol_side_channel", "mse")
    hard_all_mse = _metric(probe, hard_feature, "all_geometry", "mse")
    default_all_mse = _metric(probe, feature, "all_geometry", "mse")

    layers = [
        {
            "layer": "non_surface_signal",
            "status": (
                "PASS"
                if (
                    factor_return_r2 >= 0.5
                    and factor_level_mse < raw_factor_level_mse
                    and side_mse < raw_side_mse
                )
                else "FAIL"
            ),
            "evidence": (
                f"factor_return_r2={factor_return_r2:.6f}; "
                f"factor_level_mse={factor_level_mse:.6f} vs raw_surface={raw_factor_level_mse:.6f}; "
                f"side_mse={side_mse:.6f} vs raw_surface={raw_side_mse:.6f}"
            ),
        },
        {
            "layer": "exact_iv_state_retention",
            "status": "FAIL" if iv_mse > raw_iv_mse * 1.25 else "PASS",
            "evidence": (
                f"iv_surface_mse={iv_mse:.6f} vs raw_surface={raw_iv_mse:.6f}"
            ),
        },
        {
            "layer": "factor_level_gap",
            "status": (
                "FAIL" if factor_level_mse > upper_factor_level_mse * 3.0 else "PASS"
            ),
            "evidence": (
                f"factor_level_mse={factor_level_mse:.6f} vs raw_full_geometry_upper_bound={upper_factor_level_mse:.6f}"
            ),
        },
        {
            "layer": "mask_aggression_regression",
            "status": "FAIL" if hard_all_mse > default_all_mse else "PASS",
            "evidence": (
                f"hard_all_geometry_mse={hard_all_mse:.6f} vs default_all_geometry_mse={default_all_mse:.6f}"
            ),
        },
    ]
    passed = all(row["status"] == "PASS" for row in layers)
    return {
        "assessment": "world_model_part1_state_content_gate",
        "date": "2026-05-10",
        "source_probe": str(DEFAULT_PRESENT_STATE),
        "objective_family": "downstream_probe_present_state_information",
        "quality_gate_passed": passed,
        "promotion_decision": "PASS" if passed else "FAIL",
        "layer_results": layers,
        "interpretation": (
            "The default embedding contains useful non-surface signal, especially "
            "factor returns, but the state-content gate fails because exact IV "
            "state retention and factor-level fidelity are not strong enough."
        ),
        "next_required_evidence": [
            "state-content probes must remain separate from future prediction losses",
            "improve exact IV/current-state retention without collapsing factor-return signal",
            "evaluate whether geometry-aware pooling or larger scale fixes the state-content gap before changing objective family",
        ],
    }


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`post_experiment_analysis`",
        "",
        "## Objective Family",
        "",
        "`downstream_probe_present_state_information` gate.",
        "",
        "## Verdict",
        "",
        f"- Quality gate passed: `{result['quality_gate_passed']}`.",
        f"- Promotion decision: `{result['promotion_decision']}`.",
        "",
        "## Layer Results",
        "",
        "| layer | status | evidence |",
        "| --- | --- | --- |",
    ]
    for row in result["layer_results"]:
        lines.append(f"| {row['layer']} | {row['status']} | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
            "## Next Required Evidence",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["next_required_evidence"])
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess Part 1 frozen state-content quality"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/state_content_gate_head125.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head125_state_content_gate.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD125: State-Content Gate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = assess_state_content_gate(args.root)
    output_json = (
        args.output_json
        if args.output_json.is_absolute()
        else args.root / args.output_json
    )
    output_md = (
        args.output_md if args.output_md.is_absolute() else args.root / args.output_md
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(
        render_markdown(result, title=args.report_title), encoding="utf-8"
    )
    print(json.dumps({"quality_gate_passed": result["quality_gate_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
