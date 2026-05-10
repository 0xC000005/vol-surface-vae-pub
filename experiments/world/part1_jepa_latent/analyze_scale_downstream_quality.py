from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DOWNSTREAM = Path(
    "results/world/masked_multiview_downstream_probe_head085.json"
)
SCALE_DOWNSTREAM = Path(
    "results/world/masked_multiview_downstream_probe_scale_head128.json"
)
TARGETS = (
    "future_mean_delta",
    "future_range",
    "future_terminal_delta",
    "future_max_abs_step",
    "future_drawdown",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_raw(regression: dict[str, Any], target: str) -> tuple[str, float]:
    names = ("raw_surface_last", "raw_surface_flat")
    best = min(names, key=lambda name: regression[name]["targets"][target]["mse"])
    return best, float(regression[best]["targets"][target]["mse"])


def _rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    regression = data["regression_probe_metrics"]
    rows = []
    for target in TARGETS:
        best_raw_name, best_raw_mse = _best_raw(regression, target)
        barlow_mse = float(regression["barlow_clean_last"]["targets"][target]["mse"])
        raw_last_mse = float(regression["raw_surface_last"]["targets"][target]["mse"])
        combo_mse = float(
            regression["raw_surface_last_plus_barlow_clean_last"]["targets"][target][
                "mse"
            ]
        )
        rows.append(
            {
                "target": target,
                "barlow_mse": barlow_mse,
                "best_raw_feature": best_raw_name,
                "best_raw_mse": best_raw_mse,
                "barlow_beats_best_raw": barlow_mse < best_raw_mse,
                "raw_last_mse": raw_last_mse,
                "raw_last_plus_barlow_mse": combo_mse,
                "barlow_adds_to_raw_last": combo_mse < raw_last_mse,
            }
        )
    return rows


def _regime(data: dict[str, Any]) -> dict[str, float]:
    cls = data["classification_probe_metrics"]
    return {
        "barlow_accuracy": float(cls["barlow_clean_last"]["regime_label"]["accuracy"]),
        "raw_surface_last_accuracy": float(
            cls["raw_surface_last"]["regime_label"]["accuracy"]
        ),
        "raw_plus_barlow_accuracy": float(
            cls["raw_surface_last_plus_barlow_clean_last"]["regime_label"]["accuracy"]
        ),
        "majority_accuracy": float(
            cls["barlow_clean_last"]["regime_label"]["majority_accuracy"]
        ),
    }


def _health(data: dict[str, Any]) -> dict[str, float]:
    health = data["regression_probe_metrics"]["barlow_clean_last"]["health"]
    return {
        "effective_rank": float(health["effective_rank"]),
        "variance_min": float(health["variance_min"]),
        "offdiag_abs_mean": float(health["offdiag_abs_mean"]),
    }


def analyze_scale_downstream_quality() -> dict[str, Any]:
    default = _load_json(DEFAULT_DOWNSTREAM)
    scale = _load_json(SCALE_DOWNSTREAM)
    default_rows = _rows(default)
    scale_rows = _rows(scale)
    return {
        "analysis": "world_model_part1_scale_downstream_quality",
        "date": "2026-05-10",
        "objective_family": "downstream_probe",
        "default_source": str(DEFAULT_DOWNSTREAM),
        "scale_source": str(SCALE_DOWNSTREAM),
        "default_rows": default_rows,
        "scale_rows": scale_rows,
        "default_barlow_best_raw_wins": int(
            sum(row["barlow_beats_best_raw"] for row in default_rows)
        ),
        "scale_barlow_best_raw_wins": int(
            sum(row["barlow_beats_best_raw"] for row in scale_rows)
        ),
        "default_raw_last_plus_barlow_wins": int(
            sum(row["barlow_adds_to_raw_last"] for row in default_rows)
        ),
        "scale_raw_last_plus_barlow_wins": int(
            sum(row["barlow_adds_to_raw_last"] for row in scale_rows)
        ),
        "regime": {
            "default": _regime(default),
            "scale": _regime(scale),
        },
        "health": {
            "default": _health(default),
            "scale": _health(scale),
        },
        "decision": {
            "scale_improves_downstream_quality": True,
            "scale_clears_baseline_superiority": False,
            "scale_ready_for_part_b": False,
            "reason": "Scaled Barlow improves most downstream diagnostics but still wins only 2/5 standalone IV future targets and remains below majority on regime accuracy.",
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _row_by_target(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    return next(row for row in rows if row["target"] == target)


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
        "`downstream_probe` audit for frozen Part 1 candidates.",
        "",
        "## Hypothesis",
        "",
        "If HEAD127 is a better Part 1 candidate, it should improve frozen",
        "downstream probes and incremental value over raw baselines without using",
        "future targets during pretraining.",
        "",
        "## Falsifier",
        "",
        "The scaled checkpoint is not Part-B-ready if it still fails broad baseline",
        "superiority or market-state regime probes.",
        "",
        "## Baseline Superiority",
        "",
        "| target | default Barlow MSE | default best raw MSE | default win | scale Barlow MSE | scale best raw MSE | scale win | scale raw-last+Barlow improves raw-last |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in TARGETS:
        drow = _row_by_target(result["default_rows"], target)
        srow = _row_by_target(result["scale_rows"], target)
        lines.append(
            "| {target} | {db} | {dr} | {dw} | {sb} | {sr} | {sw} | {adds} |".format(
                target=target,
                db=_fmt(drow["barlow_mse"]),
                dr=_fmt(drow["best_raw_mse"]),
                dw=str(drow["barlow_beats_best_raw"]),
                sb=_fmt(srow["barlow_mse"]),
                sr=_fmt(srow["best_raw_mse"]),
                sw=str(srow["barlow_beats_best_raw"]),
                adds=str(srow["barlow_adds_to_raw_last"]),
            )
        )
    lines.extend(
        [
            "",
            "## Summary Counts",
            "",
            f"- Default Barlow standalone wins: `{result['default_barlow_best_raw_wins']}/5`.",
            f"- Scale Barlow standalone wins: `{result['scale_barlow_best_raw_wins']}/5`.",
            f"- Default raw-last+Barlow improvements: `{result['default_raw_last_plus_barlow_wins']}/5`.",
            f"- Scale raw-last+Barlow improvements: `{result['scale_raw_last_plus_barlow_wins']}/5`.",
            "",
            "## Regime Probe",
            "",
            "| run | Barlow accuracy | raw last accuracy | raw+Barlow accuracy | majority |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["regime"].items():
        lines.append(
            "| {name} | {barlow} | {raw} | {combo} | {majority} |".format(
                name=name,
                barlow=_fmt(row["barlow_accuracy"]),
                raw=_fmt(row["raw_surface_last_accuracy"]),
                combo=_fmt(row["raw_plus_barlow_accuracy"]),
                majority=_fmt(row["majority_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Representation Health In Probe Space",
            "",
            "| run | effective rank | variance min | offdiag abs mean |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["health"].items():
        lines.append(
            "| {name} | {rank} | {var_min} | {offdiag} |".format(
                name=name,
                rank=_fmt(row["effective_rank"]),
                var_min=_fmt(row["variance_min"]),
                offdiag=_fmt(row["offdiag_abs_mean"]),
            )
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Scale improves downstream quality: `{decision['scale_improves_downstream_quality']}`.",
            f"- Scale clears baseline superiority: `{decision['scale_clears_baseline_superiority']}`.",
            f"- Scale ready for Part B: `{decision['scale_ready_for_part_b']}`.",
            f"- Reason: {decision['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze scaled Part 1 downstream quality"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_downstream_quality_head128.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head128_scale_downstream_quality.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD128: Scale Downstream Quality",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_scale_downstream_quality()
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
