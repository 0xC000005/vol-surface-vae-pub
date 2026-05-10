from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DOWNSTREAM = Path(
    "results/world/masked_multiview_downstream_probe_head085.json"
)
DEFAULT_ASSESSMENT = Path("results/world/part1_quality_gate_assessment_head119.json")

REGRESSION_FEATURES = (
    "mean_target_baseline",
    "barlow_clean_last",
    "raw_surface_last",
    "raw_surface_flat",
    "raw_surface_last_plus_barlow_clean_last",
)
TARGETS = (
    "future_mean_delta",
    "future_range",
    "future_terminal_delta",
    "future_max_abs_step",
    "future_drawdown",
)
CLASSIFICATION_FEATURES = (
    "barlow_clean_last",
    "barlow_clean_mean",
    "raw_surface_last",
    "raw_surface_flat",
    "raw_surface_last_plus_barlow_clean_last",
)


def _load_json(root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _target_metrics(
    downstream: dict[str, Any], feature: str, target: str
) -> dict[str, float]:
    if feature == "mean_target_baseline":
        return downstream["mean_target_baseline"][target]
    return downstream["regression_probe_metrics"][feature]["targets"][target]


def _relative_mse_delta(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return (candidate - baseline) / baseline


def _best_feature_for_target(
    downstream: dict[str, Any], target: str
) -> tuple[str, float]:
    best_feature = ""
    best_mse = float("inf")
    for feature in REGRESSION_FEATURES:
        mse = float(_target_metrics(downstream, feature, target)["mse"])
        if mse < best_mse:
            best_feature = feature
            best_mse = mse
    return best_feature, best_mse


def _regression_rows(downstream: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        mean_mse = float(
            _target_metrics(downstream, "mean_target_baseline", target)["mse"]
        )
        barlow_mse = float(
            _target_metrics(downstream, "barlow_clean_last", target)["mse"]
        )
        raw_last_mse = float(
            _target_metrics(downstream, "raw_surface_last", target)["mse"]
        )
        raw_flat_mse = float(
            _target_metrics(downstream, "raw_surface_flat", target)["mse"]
        )
        raw_plus_mse = float(
            _target_metrics(
                downstream,
                "raw_surface_last_plus_barlow_clean_last",
                target,
            )["mse"]
        )
        best_feature, best_mse = _best_feature_for_target(downstream, target)
        rows.append(
            {
                "target": target,
                "mean_mse": mean_mse,
                "barlow_mse": barlow_mse,
                "raw_last_mse": raw_last_mse,
                "raw_flat_mse": raw_flat_mse,
                "raw_plus_mse": raw_plus_mse,
                "best_feature": best_feature,
                "best_mse": best_mse,
                "barlow_vs_mean_delta": _relative_mse_delta(barlow_mse, mean_mse),
                "barlow_vs_best_delta": _relative_mse_delta(barlow_mse, best_mse),
                "raw_plus_vs_raw_last_delta": _relative_mse_delta(
                    raw_plus_mse, raw_last_mse
                ),
            }
        )
    return rows


def _classification_rows(downstream: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metrics = downstream["classification_probe_metrics"]
    for feature in CLASSIFICATION_FEATURES:
        regime = metrics[feature]["regime_label"]
        rows.append(
            {
                "feature": feature,
                "accuracy": float(regime["accuracy"]),
                "majority_accuracy": float(regime["majority_accuracy"]),
                "accuracy_lift": float(regime["accuracy_lift"]),
                "macro_recall": float(regime["macro_recall"]),
                "class_counts": regime["class_counts"],
            }
        )
    return rows


def analyze_failure(root: Path) -> dict[str, Any]:
    downstream = _load_json(root, DEFAULT_DOWNSTREAM)
    assessment = _load_json(root, DEFAULT_ASSESSMENT)
    regression_rows = _regression_rows(downstream)
    classification_rows = _classification_rows(downstream)

    barlow_beats_mean = sum(
        row["barlow_mse"] < row["mean_mse"] for row in regression_rows
    )
    barlow_best = sum(
        row["best_feature"] == "barlow_clean_last" for row in regression_rows
    )
    raw_plus_improves_raw_last = sum(
        row["raw_plus_vs_raw_last_delta"] is not None
        and row["raw_plus_vs_raw_last_delta"] < 0.0
        for row in regression_rows
    )

    failed_layers = [
        row
        for row in assessment["layer_results"]
        if row["status"] in {"FAIL", "PARTIAL"}
    ]
    empirical_failures = [
        "baseline_superiority",
        "regime_probe_accuracy",
    ]
    missing_evidence_failures = [
        "richer_mask_families_and_mask_seed_stability",
        "factor_panel_state_and_future_probes",
        "iv_shape_state_probes",
        "full_data_multi_seed_scale_stability",
        "pca_persistence_rolling_window_baselines",
    ]

    return {
        "analysis": "world_model_part1_quality_gate_failure_analysis",
        "date": "2026-05-10",
        "source_assessment": str(DEFAULT_ASSESSMENT),
        "source_downstream_probe": str(DEFAULT_DOWNSTREAM),
        "promotion_decision": assessment["promotion_decision"],
        "quality_gate_passed": assessment["quality_gate_passed"],
        "failed_or_partial_layers": failed_layers,
        "regression_rows": regression_rows,
        "classification_rows": classification_rows,
        "summary_counts": {
            "barlow_beats_mean_baseline_targets": int(barlow_beats_mean),
            "barlow_is_best_feature_targets": int(barlow_best),
            "raw_plus_barlow_improves_raw_last_targets": int(
                raw_plus_improves_raw_last
            ),
            "total_regression_targets": len(regression_rows),
        },
        "failure_classes": {
            "empirical_failures": empirical_failures,
            "missing_evidence_failures": missing_evidence_failures,
        },
        "diagnosis": [
            "The gate did not fail because of representation collapse; representation health passed.",
            "The strongest empirical failure is baseline superiority: Barlow is useful versus a mean-target baseline but is not better than raw/simple features on enough downstream targets.",
            "Raw last-surface features dominate persistence-like mean and terminal targets; the Barlow representation appears more useful for path-shape and dispersion targets.",
            "Adding Barlow to raw last-surface features improves several path-shape targets, so the representation carries complementary signal, but not enough to certify a standalone market-state representation.",
            "The regime probe is not mature evidence: every feature set is below the majority baseline, so this layer currently indicates probe/label/baseline insufficiency as well as weak Barlow accuracy.",
            "The remaining failures are mostly missing evidence: factor-panel probes, IV-shape state probes, richer mask policies, seed sensitivity, and larger-scale training have not been run.",
        ],
        "next_diagnostics": [
            "Add frozen probes for present-state IV shape summaries and factor-panel summaries before changing the pretraining objective.",
            "Add PCA, persistence, and rolling-window statistics baselines for the same targets.",
            "Evaluate whether Barlow adds incremental value to raw features with consistent split-safe probes.",
            "Rerun mask robustness across seeds and richer held-out mask families.",
            "Only after those diagnostics, decide whether scale, architecture capacity, or objective changes are justified.",
        ],
    }


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Objective Family",
        "",
        "`post_experiment_analysis` for the HEAD119 Part 1 quality-gate failure.",
        "",
        "## Hypothesis",
        "",
        "The quality gate failed because the current representation is not yet",
        "proven as a market-state representation, not because the masked-view",
        "embedding objective simply collapsed.",
        "",
        "## Falsifier",
        "",
        "The diagnosis would be wrong if the saved artifacts showed collapsed",
        "representation health, no utility versus a mean baseline, or no",
        "incremental value when combined with raw surface features.",
        "",
        "## Summary",
        "",
        f"- Promotion decision: `{result['promotion_decision']}`.",
        f"- Quality gate passed: `{result['quality_gate_passed']}`.",
        "- Representation health did not fail; the main blockers are baseline",
        "  superiority, market-state probes, and scale/stability.",
        "",
        "## Regression Failure Anatomy",
        "",
        "| target | mean MSE | Barlow MSE | raw last MSE | raw flat MSE | raw+Barlow MSE | best feature | Barlow vs mean | Barlow vs best | raw+Barlow vs raw last |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in result["regression_rows"]:
        lines.append(
            "| {target} | {mean} | {barlow} | {raw_last} | {raw_flat} | {raw_plus} | `{best}` | {barlow_mean} | {barlow_best} | {raw_plus_delta} |".format(
                target=row["target"],
                mean=_fmt(row["mean_mse"]),
                barlow=_fmt(row["barlow_mse"]),
                raw_last=_fmt(row["raw_last_mse"]),
                raw_flat=_fmt(row["raw_flat_mse"]),
                raw_plus=_fmt(row["raw_plus_mse"]),
                best=row["best_feature"],
                barlow_mean=_pct(row["barlow_vs_mean_delta"]),
                barlow_best=_pct(row["barlow_vs_best_delta"]),
                raw_plus_delta=_pct(row["raw_plus_vs_raw_last_delta"]),
            )
        )

    counts = result["summary_counts"]
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            f"- Barlow beats the mean baseline on `{counts['barlow_beats_mean_baseline_targets']}/5` targets.",
            f"- Barlow is the best standalone feature on `{counts['barlow_is_best_feature_targets']}/5` targets.",
            f"- Adding Barlow to raw last-surface features improves `{counts['raw_plus_barlow_improves_raw_last_targets']}/5` targets.",
            "- The representation is not useless, but it is not yet superior to",
            "  raw/simple baselines across the claimed target family.",
            "",
            "## Regime Probe Anatomy",
            "",
            "| feature | accuracy | majority | lift | macro recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["classification_rows"]:
        lines.append(
            "| {feature} | {accuracy} | {majority} | {lift} | {macro} |".format(
                feature=row["feature"],
                accuracy=_fmt(row["accuracy"]),
                majority=_fmt(row["majority_accuracy"]),
                lift=_fmt(row["accuracy_lift"]),
                macro=_fmt(row["macro_recall"]),
            )
        )

    lines.extend(
        [
            "",
            "The regime layer is not solved by any current feature set. This is a",
            "real gate failure, but it is also a probe-design warning: the current",
            "regime label/probe cannot certify market-state quality.",
            "",
            "## Diagnosis",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["diagnosis"])
    lines.extend(
        [
            "",
            "## Next Diagnostics",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["next_diagnostics"])
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze why the world-model Part 1 quality gate failed"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/part1_quality_gate_failure_analysis_head120.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head120_part1_failure_analysis.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD120: Part 1 Quality Gate Failure Analysis",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_failure(args.root)
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
    print(
        json.dumps(
            {
                "promotion_decision": result["promotion_decision"],
                "barlow_best_targets": result["summary_counts"][
                    "barlow_is_best_feature_targets"
                ],
                "barlow_beats_mean_targets": result["summary_counts"][
                    "barlow_beats_mean_baseline_targets"
                ],
                "raw_plus_improves_raw_last_targets": result["summary_counts"][
                    "raw_plus_barlow_improves_raw_last_targets"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
