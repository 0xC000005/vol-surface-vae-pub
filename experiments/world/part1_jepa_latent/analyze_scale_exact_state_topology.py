from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from experiments.world.part1_jepa_latent.analyze_scale_exact_state_gap import (  # noqa: E402
    analyze_exact_state_gap,
)


def _mean_row(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    raw = np.asarray([row["raw_surface_mse"] for row in rows], dtype=np.float64)
    scale = np.asarray([row["scale_barlow_mse"] for row in rows], dtype=np.float64)
    delta = scale - raw
    return {
        "n_cells": int(len(rows)),
        "raw_surface_mse": float(raw.mean()),
        "scale_barlow_mse": float(scale.mean()),
        "scale_minus_raw_mse": float(delta.mean()),
        "scale_to_raw_ratio": float(scale.mean() / raw.mean()),
        "scale_worse_cells": int(np.sum(scale > raw)),
    }


def _group_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_moneyness = {}
    by_maturity = {}
    for idx in range(5):
        by_moneyness[str(idx)] = _mean_row(
            [row for row in cells if int(row["moneyness_index"]) == idx]
        )
        by_maturity[str(idx)] = _mean_row(
            [row for row in cells if int(row["maturity_index"]) == idx]
        )
    wing_rows = [
        row for row in cells if int(row["moneyness_index"]) in {0, 4}
    ]
    core_rows = [
        row for row in cells if int(row["moneyness_index"]) in {1, 2, 3}
    ]
    edge_maturity_rows = [
        row for row in cells if int(row["maturity_index"]) in {0, 4}
    ]
    middle_maturity_rows = [
        row for row in cells if int(row["maturity_index"]) in {1, 2, 3}
    ]
    return {
        "by_moneyness": by_moneyness,
        "by_maturity": by_maturity,
        "wing_vs_core": {
            "wing_moneyness": _mean_row(wing_rows),
            "core_moneyness": _mean_row(core_rows),
        },
        "edge_vs_middle_maturity": {
            "edge_maturity": _mean_row(edge_maturity_rows),
            "middle_maturity": _mean_row(middle_maturity_rows),
        },
    }


def analyze_scale_exact_state_topology(args: argparse.Namespace) -> dict[str, Any]:
    gap_args = argparse.Namespace(
        history_len=args.history_len,
        future_len=args.future_len,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        batch_size=args.batch_size,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        top_cells=25,
        device=args.device,
    )
    gap = analyze_exact_state_gap(gap_args)
    cells = gap["iv_surface_gap"]["largest_scale_minus_raw_cells"]
    topology = _group_cells(cells)
    wing = topology["wing_vs_core"]["wing_moneyness"]
    core = topology["wing_vs_core"]["core_moneyness"]
    edge = topology["edge_vs_middle_maturity"]["edge_maturity"]
    middle = topology["edge_vs_middle_maturity"]["middle_maturity"]
    return {
        "analysis": "world_model_scale_exact_state_topology",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_present_state_information",
        "source_analysis": "analyze_scale_exact_state_gap(top_cells=25)",
        "overall": gap["iv_surface_gap"],
        "topology": topology,
        "decision": {
            "gap_broad_across_surface": gap["iv_surface_gap"]["scale_worse_cells"] >= 20,
            "wing_gap_larger_than_core": wing["scale_minus_raw_mse"]
            > core["scale_minus_raw_mse"],
            "edge_maturity_gap_larger_than_middle": edge["scale_minus_raw_mse"]
            > middle["scale_minus_raw_mse"],
            "largest_gap_cell": cells[0]["factor_id"],
            "promotion_decision": "DO_NOT_PROMOTE",
            "interpretation": (
                "The exact-state gap is broad enough to block promotion, but it is "
                "especially concentrated in wing moneyness and edge maturities. A "
                "future design should target surface-local geometry rather than "
                "global row-level objectives."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _section(lines: list[str], title: str, rows: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| group | cells | raw MSE | scale MSE | scale-raw | ratio | worse cells |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in rows.items():
        lines.append(
            "| {name} | {n} | {raw} | {scale} | {delta} | {ratio} | {worse} |".format(
                name=name,
                n=row["n_cells"],
                raw=_fmt(row["raw_surface_mse"]),
                scale=_fmt(row["scale_barlow_mse"]),
                delta=_fmt(row["scale_minus_raw_mse"]),
                ratio=_fmt(row["scale_to_raw_ratio"]),
                worse=row["scale_worse_cells"],
            )
        )


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    gap = result["overall"]
    decision = result["decision"]
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
        "`downstream_probe_present_state_information`; no model change.",
        "",
        "## Hypothesis",
        "",
        "If the exact-state blocker is a surface-geometry issue, the IV gap should",
        "show structure across moneyness, maturity, wings, or edge maturities rather",
        "than being a single-cell artifact.",
        "",
        "## Overall IV Gap",
        "",
        f"- Raw last-surface IV MSE: `{_fmt(gap['raw_surface_mse'])}`.",
        f"- Scaled Barlow IV MSE: `{_fmt(gap['scale_barlow_mse'])}`.",
        f"- Scaled/raw ratio: `{_fmt(gap['scale_to_raw_ratio'])}`.",
        f"- Cells where scaled Barlow is worse: `{gap['scale_worse_cells']}/{gap['n_surface_cells']}`.",
    ]
    topology = result["topology"]
    _section(lines, "By Moneyness", topology["by_moneyness"])
    _section(lines, "By Maturity", topology["by_maturity"])
    _section(lines, "Wing/Core", topology["wing_vs_core"])
    _section(lines, "Edge/Middle Maturity", topology["edge_vs_middle_maturity"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Gap broad across surface: `{decision['gap_broad_across_surface']}`.",
            f"- Wing gap larger than core: `{decision['wing_gap_larger_than_core']}`.",
            f"- Edge maturity gap larger than middle: `{decision['edge_maturity_gap_larger_than_middle']}`.",
            f"- Largest gap cell: `{decision['largest_gap_cell']}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit topology of the scaled exact-state IV-surface gap"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_exact_state_topology_head149.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head149_scale_exact_state_topology.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD149: Scale Exact-State Topology",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_scale_exact_state_topology(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(
        render_markdown(result, title=args.report_title),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
