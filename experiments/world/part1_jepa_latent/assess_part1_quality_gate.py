from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from experiments.world.part1_jepa_latent.reference_package_check import (  # noqa: E402
    check_reference_package,
)


DEFAULT_ARTIFACTS = {
    "reference": Path("results/world/masked_multiview_barlow_head070.json"),
    "scorecard": Path("results/world/masked_multiview_part1_scorecard_head082.json"),
    "mask_artifact": Path("results/world/masked_multiview_mask_artifact_head083.json"),
    "stratified_mask": Path("results/world/masked_multiview_stratified_head084.json"),
    "downstream_probe": Path(
        "results/world/masked_multiview_downstream_probe_head085.json"
    ),
}


def _load_json(root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def _get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _min_top10(stratified: dict[str, Any]) -> float | None:
    rows = list(stratified.get("by_mask_family_a", {}).values()) + list(
        stratified.get("by_mask_family_b", {}).values()
    )
    if not rows:
        return None
    return min(float(row["retrieval_top10"]) for row in rows)


def _best_raw_mse(
    regression: dict[str, Any],
    target: str,
    *,
    raw_features: tuple[str, ...] = ("raw_surface_last", "raw_surface_flat"),
) -> tuple[str, float]:
    best_name = ""
    best_mse = float("inf")
    for feature_name in raw_features:
        value = float(regression[feature_name]["targets"][target]["mse"])
        if value < best_mse:
            best_name = feature_name
            best_mse = value
    return best_name, best_mse


def _downstream_baseline_summary(downstream: dict[str, Any]) -> dict[str, Any]:
    regression = downstream["regression_probe_metrics"]
    targets = (
        "future_mean_delta",
        "future_range",
        "future_terminal_delta",
        "future_max_abs_step",
        "future_drawdown",
    )
    rows = []
    barlow_all_raw_wins = 0
    raw_best_wins = 0
    for target in targets:
        barlow_mse = float(regression["barlow_clean_last"]["targets"][target]["mse"])
        best_raw_name, best_raw_mse = _best_raw_mse(regression, target)
        barlow_beats_best_raw = barlow_mse < best_raw_mse
        if barlow_beats_best_raw:
            barlow_all_raw_wins += 1
        else:
            raw_best_wins += 1
        rows.append(
            {
                "target": target,
                "barlow_mse": barlow_mse,
                "best_raw_feature": best_raw_name,
                "best_raw_mse": best_raw_mse,
                "barlow_beats_best_raw": barlow_beats_best_raw,
            }
        )
    regime = _get(
        downstream,
        "classification_probe_metrics",
        "barlow_clean_last",
        "regime_label",
        default={},
    )
    return {
        "targets": rows,
        "barlow_all_raw_wins": barlow_all_raw_wins,
        "raw_best_wins": raw_best_wins,
        "regime_accuracy": regime.get("accuracy"),
        "regime_majority_accuracy": regime.get("majority_accuracy"),
        "regime_accuracy_lift": regime.get("accuracy_lift"),
    }


def assess_part1_quality_gate(root: Path) -> dict[str, Any]:
    artifacts = {
        name: _load_json(root, path) for name, path in DEFAULT_ARTIFACTS.items()
    }
    package = check_reference_package(root=root)
    reference = artifacts["reference"]
    downstream = artifacts["downstream_probe"]
    mask_artifact = artifacts["mask_artifact"]
    stratified = artifacts["stratified_mask"]

    val_alignment = _get(reference, "val_metrics", "view_alignment", default={})
    retrieval = val_alignment.get("retrieval", {})
    health_a = val_alignment.get("view_a_health", {})
    health_b = val_alignment.get("view_b_health", {})
    raw_retrieval = _get(reference, "raw_val_baseline", "retrieval", default={})

    top10 = float(retrieval["top10"])
    raw_top10 = float(raw_retrieval["top10"])
    mrr = float(retrieval["mrr"])
    raw_mrr = float(raw_retrieval["mrr"])
    rank_a = float(health_a["effective_rank"])
    rank_b = float(health_b["effective_rank"])
    var_min_a = float(health_a["variance_min"])
    var_min_b = float(health_b["variance_min"])
    top1_share_a = _singular_share(health_a.get("singular_values", []), top_k=1)
    top1_share_b = _singular_share(health_b.get("singular_values", []), top_k=1)

    mask_probe_a = mask_artifact["probes"]["view_a_to_mask_family_a"]
    mask_probe_b = mask_artifact["probes"]["view_b_to_mask_family_b"]
    min_mask_top10 = _min_top10(stratified)
    downstream_summary = _downstream_baseline_summary(downstream)

    layer_results = [
        {
            "layer": "1. Package Integrity",
            "status": "PASS" if package["ok"] else "FAIL",
            "evidence": (
                f"package_check ok={package['ok']}; "
                f"reports={package['checked_reports']}; "
                f"guardrail_docs={package['checked_guardrail_docs']}; "
                f"artifacts={package['checked_artifacts']}"
            ),
            "decision": (
                "Reference package is internally consistent."
                if package["ok"]
                else "Reference package has missing or mismatched files."
            ),
        },
        {
            "layer": "2. Representation Health",
            "status": (
                "PASS"
                if (
                    top10 > raw_top10
                    and mrr > raw_mrr
                    and min(rank_a, rank_b) >= 8.0
                    and min(var_min_a, var_min_b) > 1e-4
                    and max(top1_share_a, top1_share_b) < 0.20
                )
                else "FAIL"
            ),
            "evidence": (
                f"top10={top10:.6f} vs raw={raw_top10:.6f}; "
                f"mrr={mrr:.6f} vs raw={raw_mrr:.6f}; "
                f"effective_rank={rank_a:.3f}/{rank_b:.3f}; "
                f"variance_min={var_min_a:.6f}/{var_min_b:.6f}; "
                f"singular_top1_share={top1_share_a:.3f}/{top1_share_b:.3f}"
            ),
            "decision": "Smoke-scale same-state masked-view embedding learning is real.",
        },
        {
            "layer": "3. Corruption Robustness",
            "status": (
                "PARTIAL"
                if (
                    mask_probe_a["accuracy"] <= mask_probe_a["majority_accuracy"]
                    and mask_probe_b["accuracy"] <= mask_probe_b["majority_accuracy"]
                    and min_mask_top10 is not None
                    and min_mask_top10 >= 0.70
                )
                else "FAIL"
            ),
            "evidence": (
                "mask-family probes below majority; "
                f"view_a={mask_probe_a['accuracy']:.6f}/{mask_probe_a['majority_accuracy']:.6f}; "
                f"view_b={mask_probe_b['accuracy']:.6f}/{mask_probe_b['majority_accuracy']:.6f}; "
                f"default-family min_top10={min_mask_top10:.6f}"
            ),
            "decision": (
                "Default structured masks look robust, but richer held-out mask "
                "families and seed stability are not yet validated."
            ),
        },
        {
            "layer": "4. Baseline Superiority",
            "status": (
                "FAIL" if downstream_summary["barlow_all_raw_wins"] < 5 else "PASS"
            ),
            "evidence": (
                "barlow_clean_last beats best raw baseline on "
                f"{downstream_summary['barlow_all_raw_wins']}/5 IV future targets; "
                f"best raw baseline wins {downstream_summary['raw_best_wins']}/5."
            ),
            "decision": (
                "Broad baseline superiority is not established; the representation "
                "wins some path-shape targets but loses mean/terminal and max-step "
                "claims against raw features."
            ),
        },
        {
            "layer": "5. Market-State Linear Probes",
            "status": "FAIL",
            "evidence": (
                "regime probe accuracy="
                f"{downstream_summary['regime_accuracy']:.6f} vs majority="
                f"{downstream_summary['regime_majority_accuracy']:.6f}; "
                "factor-panel state probes and IV-shape state probes are not complete."
            ),
            "decision": "Current frozen probes do not certify market-state information.",
        },
        {
            "layer": "6. Temporal Utility Probes",
            "status": "PARTIAL",
            "evidence": (
                "IV-surface future probes exist, but performance is mixed and "
                "factor-panel future targets, horizon sensitivity, and held-out "
                "time-split robustness are not complete."
            ),
            "decision": "Temporal utility is diagnostic-only, not promotion evidence yet.",
        },
        {
            "layer": "7. Scale And Stability",
            "status": "FAIL",
            "evidence": (
                "reference checkpoint uses 384 train windows, 128 validation "
                "windows, 8 epochs, one seed, CPU smoke scale."
            ),
            "decision": "No full-data, multi-seed, or horizon-stability claim is supported.",
        },
    ]

    promoted = all(row["status"] == "PASS" for row in layer_results)
    return {
        "assessment": "world_model_part1_quality_gate",
        "date": "2026-05-10",
        "reference_run": "HEAD070",
        "quality_gate_passed": promoted,
        "promotion_decision": "PROMOTE" if promoted else "DO_NOT_PROMOTE",
        "part1_ready_for_part_b": promoted,
        "embedding_learning_signal": "PASS",
        "package_check": package,
        "artifacts": {name: str(path) for name, path in DEFAULT_ARTIFACTS.items()},
        "layer_results": layer_results,
        "downstream_baseline_summary": downstream_summary,
        "overall_interpretation": (
            "The corruption-based masked-multiview Barlow representation works "
            "as a smoke-scale embedding-learning signal, but the current Part 1 "
            "package is not good enough to promote to a certified joint "
            "market-state representation for Part B."
        ),
        "next_required_evidence": [
            "frozen market-state probes that beat raw/simple baselines",
            "factor-panel future target probes before joint-factor claims",
            "simple PCA, persistence, and rolling-window baseline comparisons",
            "held-out mask-family and mask-seed robustness",
            "larger-scale and multi-seed stability without adding objective knobs",
        ],
    }


def _singular_share(values: list[float], *, top_k: int) -> float:
    if not values:
        return 1.0
    total = float(sum(float(value) for value in values))
    if total <= 0.0:
        return 1.0
    return float(sum(float(value) for value in values[:top_k]) / total)


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Objective Family",
        "",
        "`post_experiment_analysis` for the frozen `masked_multiview_invariance` Part 1 reference.",
        "",
        "## Execution",
        "",
        "- Ran the package-integrity checker.",
        "- Read the saved HEAD070, HEAD082, HEAD083, HEAD084, and HEAD085 artifacts.",
        "- Scored the literature-aligned Part 1 quality-gate layers without changing the pretraining objective.",
        "",
        "## Verdict",
        "",
        f"- Quality gate passed: `{result['quality_gate_passed']}`.",
        f"- Promotion decision: `{result['promotion_decision']}`.",
        f"- Part 1 ready for Part B: `{result['part1_ready_for_part_b']}`.",
        "- Corruption-based embedding learning signal: `PASS` at smoke scale.",
        "",
        "## Layer Results",
        "",
        "| layer | status | evidence | decision |",
        "| --- | --- | --- | --- |",
    ]
    for row in result["layer_results"]:
        lines.append(
            f"| {row['layer']} | `{row['status']}` | {row['evidence']} | {row['decision']} |"
        )

    lines.extend(
        [
            "",
            "## Downstream Raw-Baseline Check",
            "",
            "| target | Barlow MSE | best raw feature | best raw MSE | Barlow beats best raw |",
            "| --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in result["downstream_baseline_summary"]["targets"]:
        lines.append(
            "| {target} | {barlow} | `{raw_feature}` | {raw_mse} | `{win}` |".format(
                target=row["target"],
                barlow=_fmt(row["barlow_mse"]),
                raw_feature=row["best_raw_feature"],
                raw_mse=_fmt(row["best_raw_mse"]),
                win=row["barlow_beats_best_raw"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["overall_interpretation"],
            "",
            "The current result is useful for diagnostics and for designing the next",
            "frozen probes. It should not be used to start Part B as if Part 1 were",
            "already certified.",
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
        description="Assess the world-model Part 1 quality gate"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/part1_quality_gate_assessment_head119.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head119_part1_quality_gate_assessment.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD119: Part 1 Quality Gate Assessment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = assess_part1_quality_gate(args.root)
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
                "quality_gate_passed": result["quality_gate_passed"],
                "promotion_decision": result["promotion_decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
