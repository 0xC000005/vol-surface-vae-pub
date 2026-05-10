from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCALE_QUALITY = Path("results/world/scale_downstream_quality_head128.json")


TARGET_FAMILIES = {
    "future_mean_delta": "persistence_or_exact_state_dominated",
    "future_terminal_delta": "persistence_or_exact_state_dominated",
    "future_max_abs_step": "mixed_path_shape",
    "future_range": "path_shape_or_risk_width",
    "future_drawdown": "path_shape_or_risk_width",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in data["scale_rows"]:
        target = row["target"]
        barlow = float(row["barlow_mse"])
        best_raw = float(row["best_raw_mse"])
        raw_last = float(row["raw_last_mse"])
        combo = float(row["raw_last_plus_barlow_mse"])
        rows.append(
            {
                "target": target,
                "target_family": TARGET_FAMILIES[target],
                "barlow_mse": barlow,
                "best_raw_feature": row["best_raw_feature"],
                "best_raw_mse": best_raw,
                "raw_last_mse": raw_last,
                "raw_last_plus_barlow_mse": combo,
                "barlow_beats_best_raw": bool(row["barlow_beats_best_raw"]),
                "barlow_adds_to_raw_last": bool(row["barlow_adds_to_raw_last"]),
                "barlow_to_best_raw_ratio": barlow / best_raw,
                "combo_to_raw_last_ratio": combo / raw_last,
                "combo_improvement_pct": 100.0 * (raw_last - combo) / raw_last,
            }
        )
    return rows


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for family in sorted({row["target_family"] for row in rows}):
        family_rows = [row for row in rows if row["target_family"] == family]
        out[family] = {
            "n_targets": len(family_rows),
            "barlow_best_raw_wins": int(
                sum(row["barlow_beats_best_raw"] for row in family_rows)
            ),
            "barlow_adds_to_raw_last": int(
                sum(row["barlow_adds_to_raw_last"] for row in family_rows)
            ),
            "mean_barlow_to_best_raw_ratio": sum(
                row["barlow_to_best_raw_ratio"] for row in family_rows
            )
            / len(family_rows),
            "mean_combo_improvement_pct": sum(
                row["combo_improvement_pct"] for row in family_rows
            )
            / len(family_rows),
        }
    return out


def analyze_baseline_target_taxonomy() -> dict[str, Any]:
    quality = _load_json(SCALE_QUALITY)
    rows = _target_rows(quality)
    summaries = _summaries(rows)
    persistence = summaries["persistence_or_exact_state_dominated"]
    path_shape = summaries["path_shape_or_risk_width"]
    return {
        "analysis": "world_model_scaled_baseline_target_taxonomy",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_baseline_diagnostic",
        "source": str(SCALE_QUALITY),
        "rows": rows,
        "family_summaries": summaries,
        "decision": {
            "barlow_wins_path_shape_family": path_shape["barlow_best_raw_wins"]
            == path_shape["n_targets"],
            "barlow_loses_persistence_family": persistence["barlow_best_raw_wins"]
            == 0,
            "promotion_decision": "DO_NOT_PROMOTE",
            "interpretation": (
                "The scaled embedding is useful for path-shape/risk-width targets "
                "but loses persistence/exact-state dominated targets. Baseline "
                "superiority should therefore be reported by target family; the "
                "current Part 1 blocker is exact-state retention and target-family "
                "coverage, not a uniform failure of the representation."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


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
        "`downstream_probe_baseline_diagnostic` for frozen scaled Part 1 features.",
        "",
        "## Hypothesis",
        "",
        "If the baseline-superiority failure is structured, scaled Barlow should",
        "win or add value on path-shape/risk-width targets while losing",
        "persistence or exact-state dominated targets.",
        "",
        "## Target Rows",
        "",
        "| target | family | best raw | Barlow MSE | best raw MSE | Barlow/best raw | raw last MSE | raw+Barlow MSE | combo/raw | combo improvement % | Barlow win | adds to raw last |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {target} | {family} | {best_raw} | {barlow} | {best} | {ratio} | {raw_last} | {combo} | {combo_ratio} | {combo_pct} | {win} | {adds} |".format(
                target=row["target"],
                family=row["target_family"],
                best_raw=row["best_raw_feature"],
                barlow=_fmt(row["barlow_mse"]),
                best=_fmt(row["best_raw_mse"]),
                ratio=_fmt(row["barlow_to_best_raw_ratio"]),
                raw_last=_fmt(row["raw_last_mse"]),
                combo=_fmt(row["raw_last_plus_barlow_mse"]),
                combo_ratio=_fmt(row["combo_to_raw_last_ratio"]),
                combo_pct=_fmt(row["combo_improvement_pct"]),
                win=str(row["barlow_beats_best_raw"]),
                adds=str(row["barlow_adds_to_raw_last"]),
            )
        )
    lines.extend(
        [
            "",
            "## Family Summary",
            "",
            "| family | targets | Barlow raw wins | adds to raw last | mean Barlow/best raw | mean combo improvement % |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, row in result["family_summaries"].items():
        lines.append(
            "| {family} | {n} | {wins} | {adds} | {ratio} | {improve} |".format(
                family=family,
                n=row["n_targets"],
                wins=row["barlow_best_raw_wins"],
                adds=row["barlow_adds_to_raw_last"],
                ratio=_fmt(row["mean_barlow_to_best_raw_ratio"]),
                improve=_fmt(row["mean_combo_improvement_pct"]),
            )
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Barlow wins path-shape/risk-width family: `{decision['barlow_wins_path_shape_family']}`.",
            f"- Barlow loses persistence/exact-state family: `{decision['barlow_loses_persistence_family']}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify scaled downstream baseline-superiority failures by target family"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_baseline_target_taxonomy_head134.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head134_scale_baseline_target_taxonomy.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD134: Scale Baseline Target Taxonomy",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_baseline_target_taxonomy()
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
