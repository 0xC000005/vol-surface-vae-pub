from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCALE_DOWNSTREAM = Path(
    "results/world/masked_multiview_downstream_probe_scale_head128.json"
)
DEFAULT_DOWNSTREAM = Path("results/world/masked_multiview_downstream_probe_head085.json")


FEATURES = (
    "barlow_clean_last",
    "barlow_clean_mean",
    "raw_surface_last",
    "raw_surface_flat",
    "raw_surface_last_plus_barlow_clean_last",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _regime_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = data["classification_probe_metrics"]
    rows = []
    for feature in FEATURES:
        regime = metrics[feature]["regime_label"]
        rows.append(
            {
                "feature": feature,
                "accuracy": float(regime["accuracy"]),
                "majority_accuracy": float(regime["majority_accuracy"]),
                "accuracy_lift": float(regime["accuracy_lift"]),
                "macro_recall": float(regime["macro_recall"]),
                "class_counts": {
                    str(label): int(count)
                    for label, count in regime["class_counts"].items()
                },
                "per_class_recall": {
                    str(label): float(recall)
                    for label, recall in regime["per_class_recall"].items()
                },
            }
        )
    return rows


def _row_by_feature(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    return next(row for row in rows if row["feature"] == feature)


def analyze_regime_probe_gap() -> dict[str, Any]:
    default = _load_json(DEFAULT_DOWNSTREAM)
    scale = _load_json(SCALE_DOWNSTREAM)
    default_rows = _regime_rows(default)
    scale_rows = _regime_rows(scale)
    scale_best_accuracy = max(scale_rows, key=lambda row: row["accuracy"])
    scale_best_macro = max(scale_rows, key=lambda row: row["macro_recall"])
    scaled_barlow = _row_by_feature(scale_rows, "barlow_clean_last")
    raw_last = _row_by_feature(scale_rows, "raw_surface_last")
    class_counts = scaled_barlow["class_counts"]
    minority_labels = [
        label
        for label, count in sorted(class_counts.items(), key=lambda item: item[1])
        if count < max(class_counts.values())
    ]
    return {
        "analysis": "world_model_scaled_regime_probe_gap",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_regime_diagnostic",
        "default_source": str(DEFAULT_DOWNSTREAM),
        "scale_source": str(SCALE_DOWNSTREAM),
        "default_rows": default_rows,
        "scale_rows": scale_rows,
        "class_counts": class_counts,
        "minority_labels": minority_labels,
        "scale_best_accuracy_feature": scale_best_accuracy["feature"],
        "scale_best_macro_recall_feature": scale_best_macro["feature"],
        "scaled_barlow_vs_raw_last": {
            "accuracy_delta": scaled_barlow["accuracy"] - raw_last["accuracy"],
            "macro_recall_delta": scaled_barlow["macro_recall"]
            - raw_last["macro_recall"],
            "per_class_recall_delta": {
                label: scaled_barlow["per_class_recall"][label]
                - raw_last["per_class_recall"][label]
                for label in scaled_barlow["per_class_recall"]
            },
        },
        "decision": {
            "accuracy_gate_passed": scaled_barlow["accuracy"]
            > scaled_barlow["majority_accuracy"],
            "balanced_signal_present": scaled_barlow["macro_recall"]
            > raw_last["macro_recall"],
            "promotion_decision": "DO_NOT_PROMOTE",
            "interpretation": (
                "The scaled embedding does not pass the accuracy gate because the "
                "regime labels are majority-class dominated. It does contain some "
                "minority-regime signal, shown by higher macro recall and class-4 "
                "recall than raw surface features. Treat regime accuracy as a "
                "failed promotion layer, but use balanced accuracy/class recall for "
                "diagnosis before changing the representation objective."
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
        "`downstream_probe_regime_diagnostic` for frozen Part 1 features.",
        "",
        "## Hypothesis",
        "",
        "If the regime failure is purely no-signal, scaled Barlow should be worse",
        "than raw features on both accuracy and balanced class recall. If it is",
        "partly an imbalanced-label probe issue, scaled Barlow may lose majority",
        "accuracy while improving macro recall or minority-class recall.",
        "",
        "## Class Balance",
        "",
        "| class | validation count | share |",
        "| --- | ---: | ---: |",
    ]
    total = sum(result["class_counts"].values())
    for label, count in result["class_counts"].items():
        lines.append(f"| {label} | {count} | {_fmt(count / total)} |")
    lines.extend(
        [
            "",
            "## Scaled Regime Metrics",
            "",
            "| feature | accuracy | majority | lift | macro recall | class 0 recall | class 3 recall | class 4 recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["scale_rows"]:
        recall = row["per_class_recall"]
        lines.append(
            "| {feature} | {acc} | {maj} | {lift} | {macro} | {c0} | {c3} | {c4} |".format(
                feature=row["feature"],
                acc=_fmt(row["accuracy"]),
                maj=_fmt(row["majority_accuracy"]),
                lift=_fmt(row["accuracy_lift"]),
                macro=_fmt(row["macro_recall"]),
                c0=_fmt(recall.get("0")),
                c3=_fmt(recall.get("3")),
                c4=_fmt(recall.get("4")),
            )
        )
    delta = result["scaled_barlow_vs_raw_last"]
    lines.extend(
        [
            "",
            "## Scaled Barlow Versus Raw Last Surface",
            "",
            f"- Accuracy delta: `{_fmt(delta['accuracy_delta'])}`.",
            f"- Macro-recall delta: `{_fmt(delta['macro_recall_delta'])}`.",
            "",
            "| class | recall delta |",
            "| --- | ---: |",
        ]
    )
    for label, value in delta["per_class_recall_delta"].items():
        lines.append(f"| {label} | {_fmt(value)} |")
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Accuracy gate passed: `{decision['accuracy_gate_passed']}`.",
            f"- Balanced signal present: `{decision['balanced_signal_present']}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze scaled Part 1 regime-probe failure"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_regime_probe_gap_head133.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head133_scale_regime_probe_gap.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD133: Scale Regime Probe Gap",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_regime_probe_gap()
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
