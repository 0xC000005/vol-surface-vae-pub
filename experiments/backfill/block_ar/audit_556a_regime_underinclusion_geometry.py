#!/usr/bin/env python
"""556a: compare localized regime/cell under-inclusion across frontier candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def extract_regime_cell_rows(
    name: str,
    result: dict[str, Any],
    min_cell: float = 0.70,
    max_cell: float = 0.95,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer2 = result.get("regime_coverage", {}).get("layer2_regime_cell", {})
    for regime, horizons in layer2.items():
        for horizon, metrics in horizons.items():
            if not isinstance(metrics, dict) or "worst" not in metrics:
                continue
            worst = float(metrics["worst"])
            best = float(metrics.get("best", 0.0))
            rows.append(
                {
                    "candidate": name,
                    "regime": str(regime),
                    "horizon": str(horizon),
                    "worst": worst,
                    "best": best,
                    "worst_cell": list(metrics.get("worst_cell", [])),
                    "best_cell": list(metrics.get("best_cell", [])),
                    "undercovered": bool(worst < float(min_cell)),
                    "overcovered": bool(best > float(max_cell)),
                }
            )
    return sorted(rows, key=lambda row: (float(row["worst"]), row["candidate"], row["regime"], int(row["horizon"])))


def summarize_candidate_regime_geometry(
    name: str,
    result: dict[str, Any],
    min_cell: float = 0.70,
    max_cell: float = 0.95,
) -> dict[str, Any]:
    rows = extract_regime_cell_rows(name, result, min_cell=min_cell, max_cell=max_cell)
    worst = rows[0] if rows else None
    return {
        "name": name,
        "original_n_pass": int(result.get("summary", {}).get("n_pass", 0)),
        "failed_suites": list(result.get("summary", {}).get("failed_suites", [])),
        "layer1_pass": _as_bool(result.get("regime_coverage", {}).get("layer1_pass", False)),
        "layer3_pass": _as_bool(result.get("regime_coverage", {}).get("layer3_pass", False)),
        "layer2_n_passing": int(result.get("regime_coverage", {}).get("layer2_n_passing", 0)),
        "layer2_n_total": int(result.get("regime_coverage", {}).get("layer2_n_total", 0)),
        "n_undercovered_layer2": int(sum(bool(row["undercovered"]) for row in rows)),
        "n_overcovered_layer2": int(sum(bool(row["overcovered"]) for row in rows)),
        "worst_layer2": worst,
        "rows": rows,
    }


def stable_undercovered_cells(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str, tuple[int, ...]]] = Counter()
    values: dict[tuple[str, str, tuple[int, ...]], list[float]] = {}
    for summary in summaries:
        for row in summary["rows"]:
            if not row["undercovered"]:
                continue
            key = (row["regime"], row["horizon"], tuple(row["worst_cell"]))
            counter[key] += 1
            values.setdefault(key, []).append(float(row["worst"]))
    out = []
    for (regime, horizon, cell), count in counter.items():
        vals = values[(regime, horizon, cell)]
        out.append(
            {
                "regime": regime,
                "horizon": horizon,
                "cell": list(cell),
                "candidate_count": int(count),
                "min_worst": float(min(vals)),
                "mean_worst": float(sum(vals) / len(vals)),
            }
        )
    return sorted(out, key=lambda row: (-int(row["candidate_count"]), float(row["min_worst"])))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 556a Regime Under-Inclusion Geometry",
        "",
        "## Question",
        "",
        "Is the remaining risk-readiness blocker global width, persistent collapse, or localized regime/horizon/cell occupancy?",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Suite | Layer2 | Under | Over | Layer3 | Worst |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for summary in report["candidate_summaries"]:
        worst = summary["worst_layer2"]
        worst_txt = "n/a" if worst is None else f"{worst['regime']} h{worst['horizon']} {worst['worst_cell']} = {worst['worst']:.3f}"
        lines.append(
            f"| `{summary['name']}` | `{summary['original_n_pass']}/11` | "
            f"`{summary['layer2_n_passing']}/{summary['layer2_n_total']}` | "
            f"`{summary['n_undercovered_layer2']}` | `{summary['n_overcovered_layer2']}` | "
            f"`{summary['layer3_pass']}` | {worst_txt} |"
        )
    lines.extend(["", "## Stable Undercovered Cells", ""])
    if report["stable_undercovered_cells"]:
        for row in report["stable_undercovered_cells"]:
            lines.append(
                f"- `{row['regime']}` h`{row['horizon']}` cell `{row['cell']}`: "
                f"seen in `{row['candidate_count']}` candidates, min worst `{row['min_worst']:.3f}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Worst Rows", ""])
    for summary in report["candidate_summaries"]:
        lines.append(f"### {summary['name']}")
        for row in summary["rows"][:8]:
            flags = []
            if row["undercovered"]:
                flags.append("under")
            if row["overcovered"]:
                flags.append("over")
            lines.append(
                f"- `{row['regime']}` h`{row['horizon']}` worst `{row['worst']:.3f}` "
                f"cell `{row['worst_cell']}`, best `{row['best']:.3f}` "
                f"cell `{row['best_cell']}` ({', '.join(flags) if flags else 'pass'})"
            )
        lines.append("")
    lines.extend(["## Decision", "", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "FULL11_JSON"),
        required=True,
    )
    parser.add_argument("--min_cell", type=float, default=0.70)
    parser.add_argument("--max_cell", type=float, default=0.95)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    summaries = [
        summarize_candidate_regime_geometry(
            name,
            json.loads(Path(path).read_text(encoding="utf-8")),
            min_cell=args.min_cell,
            max_cell=args.max_cell,
        )
        for name, path in args.candidate
    ]
    stable = stable_undercovered_cells(summaries)
    any_layer3_fail = any(not summary["layer3_pass"] for summary in summaries)
    worst_under = min(
        (
            float(row["worst"])
            for summary in summaries
            for row in summary["rows"]
            if row["undercovered"]
        ),
        default=1.0,
    )
    if not any_layer3_fail and stable:
        decision = (
            "The blocker is localized regime/horizon/cell occupancy, not global width or "
            "persistent scenario collapse. Broadening every path would be a blunt fix and "
            "would likely damage authenticity. The next clean move should target hard "
            "conditional stress states through a learned objective or sampling law that "
            "allocates mass to sparse regime cells without evaluator-time cell tables."
        )
    elif any_layer3_fail:
        decision = (
            "Persistent severe undercoverage remains, so the next move must repair global "
            "scenario support before local regime cells."
        )
    else:
        decision = (
            "The regime failures are not stable across candidates; run a larger split or "
            "sampling audit before adding another objective."
        )
    report = {
        "min_cell": float(args.min_cell),
        "max_cell": float(args.max_cell),
        "worst_undercovered_value": float(worst_under),
        "candidate_summaries": summaries,
        "stable_undercovered_cells": stable,
        "decision": decision,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(json.dumps(make_serializable({"worst_under": worst_under, "stable": stable[:5], "decision": decision}), indent=2))


if __name__ == "__main__":
    main()
