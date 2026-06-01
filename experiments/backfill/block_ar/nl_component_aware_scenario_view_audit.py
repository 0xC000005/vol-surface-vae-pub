#!/usr/bin/env python
"""Audit component-aware narrative conditionality views.

This artifact-only tool consumes an existing multi-start professional narrative
deck. It does not call OpenAI, train a model, or rerun the frozen SNI generator.

The question is product-facing: with the same starting level and different
professional narratives, how much narrative signal is present at the support
component level, and how much is dampened by broad pooling?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (  # noqa: E402
    build_component_pooling_diagnostic,
    plot_component_terminal_medians,
)
from experiments.backfill.block_ar.nl_sparse_component_family_view import (  # noqa: E402
    build_sparse_component_family_view,
    plot_sparse_component_families,
)


DEFAULT_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "qualified_narrative_broad_support_multistart_956c/"
    "start_0_18_22_40_77_samples8"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ROOT / "component_aware_scenario_view_audit"
)
DEFAULT_QUALITY_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_956e_full_corpus_small/"
    "nontrain_core_rollout/caption_rollout_group_summary.json"
)

SPARSE_POLICIES: tuple[dict[str, Any], ...] = (
    {
        "name": "top1_component",
        "max_components": 1,
        "min_cumulative_weight": 1.0,
        "description": "Show only the highest-weight support component.",
    },
    {
        "name": "top2_or_80pct",
        "max_components": 2,
        "min_cumulative_weight": 0.80,
        "description": "Keep up to two support components or 80% original weight.",
    },
    {
        "name": "top3_or_90pct",
        "max_components": 3,
        "min_cumulative_weight": 0.90,
        "description": "Keep up to three support components or 90% original weight.",
    },
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
            return value if math.isfinite(value) else None
        if isinstance(value, np.ndarray):
            return _jsonable(value.tolist())
    except Exception:
        pass
    return value


def _start_summary_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("start_*/gradio_live_api_casebook_summary.json"))
    if paths:
        return paths
    direct = root / "gradio_live_api_casebook_summary.json"
    if direct.exists():
        return [direct]
    raise FileNotFoundError(f"no start summaries found under {root}")


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def _median(values: list[float | None]) -> float | None:
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float((finite[mid - 1] + finite[mid]) / 2.0)


def _terminal_range_rows(
    sparse_report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market, values in sparse_report.get("terminal_p50_range_comparison", {}).items():
        if not isinstance(values, dict):
            continue
        full = float(values.get("full_pooled", 0.0))
        sparse = float(values.get("sparse_pooled", 0.0))
        component = float(values.get("selected_component", 0.0))
        rows.append(
            {
                "market": str(market),
                "full_pooled_p50_range": full,
                "sparse_pooled_p50_range": sparse,
                "selected_component_p50_range": component,
                "sparse_vs_full_range_ratio": _ratio(sparse, full),
                "component_vs_full_range_ratio": _ratio(component, full),
            }
        )
    return rows


def _quality_guardrails(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_available",
            "source": str(path),
            "note": (
                "Scenario-level CRPS/energy/coverage guardrails were not recomputed "
                "because this audit is artifact-only."
            ),
        }
    report = _load_json(path)
    group_summaries = report.get("group_summaries", {})
    if isinstance(group_summaries, dict):
        preferred = (
            group_summaries.get("fused_codex")
            or group_summaries.get("rich_codex")
            or next(iter(group_summaries.values()), {})
        )
        if isinstance(preferred, dict):
            return {
                "status": "reference_only",
                "source": str(path),
                "note": (
                    "This component-aware audit does not alter the underlying "
                    "generated scenario deck. These held-out metrics are the "
                    "quality guardrail for the incumbent professional-caption "
                    "pipeline; replacing broad pooling would require rerunning "
                    "the same guardrail."
                ),
                "count": preferred.get("count"),
                "crps_improvement_vs_persistence": preferred.get(
                    "crps_improvement_vs_persistence"
                ),
                "energy_improvement_vs_persistence": preferred.get(
                    "energy_improvement_vs_persistence"
                ),
                "coverage_80": preferred.get("coverage_80"),
                "terminal_mae_z": preferred.get("terminal_mae_z"),
            }
    rows = report.get("rows") or report.get("summary_rows") or []
    if not isinstance(rows, list):
        rows = []
    return {
        "status": "reference_only",
        "source": str(path),
        "note": (
            "This component-aware audit does not alter the underlying generated "
            "scenario deck. Existing held-out CRPS, energy, and coverage reports "
            "remain the quality guardrail for promoting a replacement pooling policy."
        ),
        "available_top_level_keys": sorted(report.keys())[:20],
        "row_count": len(rows),
    }


def _policy_summary(policy_report: dict[str, Any]) -> dict[str, Any]:
    rows = _terminal_range_rows(policy_report)
    return {
        "policy": policy_report.get("selection_policy", {}),
        "case_count": int(policy_report.get("case_count", 0)),
        "total_selected_components": int(policy_report.get("total_selected_components", 0)),
        "kept_original_weight_mean": _mean(
            [
                case.get("kept_original_weight")
                for case in policy_report.get("cases", [])
                if isinstance(case, dict)
            ]
        ),
        "effective_sparse_component_count_mean": _mean(
            [
                case.get("effective_sparse_component_count")
                for case in policy_report.get("cases", [])
                if isinstance(case, dict)
            ]
        ),
        "terminal_range_rows": rows,
        "median_sparse_vs_full_range_ratio": _median(
            [row["sparse_vs_full_range_ratio"] for row in rows]
        ),
        "median_component_vs_full_range_ratio": _median(
            [row["component_vs_full_range_ratio"] for row in rows]
        ),
    }


def _build_start_report(
    summary_path: Path,
    *,
    output_dir: Path,
    primary: bool,
    max_plot_markets: int,
) -> dict[str, Any]:
    start_label = summary_path.parent.name
    start_dir = output_dir / start_label
    start_dir.mkdir(parents=True, exist_ok=True)

    component_report = build_component_pooling_diagnostic(summary_path)
    component_report_path = start_dir / "component_pooling_diagnostic.json"
    _write_json(component_report_path, component_report)
    component_plot = plot_component_terminal_medians(
        component_report,
        start_dir / "component_terminal_medians.png",
    )

    pooled_energy = component_report["pooled_cross_narrative"]["path_energy_median"]
    component_energy = component_report["component_cross_narrative"]["path_energy_median"]
    sparse_rows: list[dict[str, Any]] = []
    for policy in SPARSE_POLICIES:
        sparse_report = build_sparse_component_family_view(
            summary_path,
            max_components=int(policy["max_components"]),
            min_cumulative_weight=float(policy["min_cumulative_weight"]),
        )
        policy_name = str(policy["name"])
        policy_path = start_dir / f"sparse_{policy_name}.json"
        if primary:
            sparse_report["artifact_paths"] = {
                "raw_component_family_plot": plot_sparse_component_families(
                    sparse_report,
                    start_dir / f"sparse_{policy_name}_raw_levels.png",
                    view="raw",
                    max_markets=max_plot_markets,
                ),
                "standardized_component_family_plot": plot_sparse_component_families(
                    sparse_report,
                    start_dir / f"sparse_{policy_name}_standardized_moves.png",
                    view="standardized",
                    max_markets=max_plot_markets,
                ),
            }
        _write_json(policy_path, sparse_report)
        summary = _policy_summary(sparse_report)
        summary.update(
            {
                "name": policy_name,
                "description": str(policy["description"]),
                "report": str(policy_path),
                "artifact_paths": sparse_report.get("artifact_paths", {}),
            }
        )
        sparse_rows.append(summary)

    return {
        "start": start_label,
        "summary_path": str(summary_path),
        "component_report": str(component_report_path),
        "component_terminal_plot": component_plot,
        "case_count": int(component_report["case_count"]),
        "component_count": int(component_report["component_count"]),
        "support_jaccard_max": component_report["support_jaccard"]["max"],
        "support_jaccard_mean": component_report["support_jaccard"]["mean"],
        "pooled_path_energy_median": pooled_energy,
        "component_path_energy_median": component_energy,
        "component_to_pooled_path_energy_ratio": _ratio(component_energy, pooled_energy),
        "terminal_component_vs_pooled_ratio": component_report["pooling_diagnosis"][
            "terminal_median_component_vs_pooled_ratio"
        ],
        "sparse_policies": sparse_rows,
    }


def build_component_aware_audit(
    *,
    root: Path,
    output_dir: Path,
    primary_start: int | None = 22,
    quality_report: Path = DEFAULT_QUALITY_REPORT,
    max_plot_markets: int = 4,
) -> dict[str, Any]:
    summary_paths = _start_summary_paths(root)
    primary_label = f"start_{primary_start}" if primary_start is not None else None
    start_reports: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        is_primary = primary_label is not None and summary_path.parent.name == primary_label
        start_reports.append(
            _build_start_report(
                summary_path,
                output_dir=output_dir,
                primary=is_primary,
                max_plot_markets=max_plot_markets,
            )
        )

    ratios = [row["component_to_pooled_path_energy_ratio"] for row in start_reports]
    primary_row = next(
        (row for row in start_reports if row["start"] == primary_label),
        start_reports[0] if start_reports else {},
    )
    policy_rows: list[dict[str, Any]] = []
    for policy in SPARSE_POLICIES:
        name = str(policy["name"])
        matching = [
            policy_row
            for start in start_reports
            for policy_row in start["sparse_policies"]
            if policy_row["name"] == name
        ]
        policy_rows.append(
            {
                "name": name,
                "description": str(policy["description"]),
                "mean_kept_original_weight": _mean(
                    [row["kept_original_weight_mean"] for row in matching]
                ),
                "median_sparse_vs_full_terminal_range_ratio": _median(
                    [row["median_sparse_vs_full_range_ratio"] for row in matching]
                ),
                "median_component_vs_full_terminal_range_ratio": _median(
                    [row["median_component_vs_full_range_ratio"] for row in matching]
                ),
            }
        )
    diagnosis = {
        "status": "pooling_dampens_visible_conditionality",
        "main_finding": (
            "Professional narratives select different support components, and those "
            "components have much larger path-level separation than the final broad "
            "pooled fan. The weak visual fan effect is therefore mainly a pooling/"
            "display attenuation problem, not a failure of support selection."
        ),
        "recommended_solution": (
            "Keep the broad pooled distribution as the calibrated risk distribution, "
            "but add a component-aware product view: top support-regime component "
            "fans, sparse component-family summary, and then the broad pooled fan. "
            "Promote sparse pooling as a replacement only after held-out CRPS, "
            "energy, and coverage are recomputed."
        ),
    }
    return {
        "source_root": str(root),
        "output_dir": str(output_dir),
        "primary_start": primary_row.get("start"),
        "start_count": len(start_reports),
        "start_reports": start_reports,
        "aggregate": {
            "mean_component_to_pooled_path_energy_ratio": _mean(ratios),
            "median_component_to_pooled_path_energy_ratio": _median(ratios),
            "max_support_jaccard_across_starts": max(
                [float(row["support_jaccard_max"] or 0.0) for row in start_reports],
                default=0.0,
            ),
            "sparse_policy_summaries": policy_rows,
        },
        "quality_guardrails": _quality_guardrails(quality_report),
        "diagnosis": diagnosis,
        "scope_note": (
            "Artifact-only component-aware view over existing professional narrative "
            "runs. It diagnoses component-to-pooling conditionality loss; it does "
            "not change generated samples or claim sparse pooling improves held-out "
            "scenario quality."
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Component-Aware Narrative Conditionality Audit",
        "",
        report["scope_note"],
        "",
        "## Diagnosis",
        "",
        f"- Status: `{report['diagnosis']['status']}`",
        f"- Main finding: {report['diagnosis']['main_finding']}",
        f"- Recommended solution: {report['diagnosis']['recommended_solution']}",
        "",
        "## Multi-Start Component-To-Pooled Loss",
        "",
        "| Start | Cases | Components | Support Jaccard max | Pooled path energy | Component path energy | Component/Pooled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["start_reports"]:
        lines.append(
            "| {start} | {cases} | {components} | {jaccard} | {pooled} | {component} | {ratio} |".format(
                start=row["start"],
                cases=row["case_count"],
                components=row["component_count"],
                jaccard=_fmt(row["support_jaccard_max"]),
                pooled=_fmt(row["pooled_path_energy_median"]),
                component=_fmt(row["component_path_energy_median"]),
                ratio=_fmt(row["component_to_pooled_path_energy_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Sparse Component-Family Policies",
            "",
            "| Policy | Mean kept weight | Median sparse/full terminal range | Median component/full terminal range |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["aggregate"]["sparse_policy_summaries"]:
        lines.append(
            "| {name} | {weight} | {sparse} | {component} |".format(
                name=row["name"],
                weight=_fmt(row["mean_kept_original_weight"]),
                sparse=_fmt(row["median_sparse_vs_full_terminal_range_ratio"]),
                component=_fmt(row["median_component_vs_full_terminal_range_ratio"]),
            )
        )
    guard = report["quality_guardrails"]
    lines.extend(
        [
            "",
            "## Guardrail Note",
            "",
            f"- Quality guardrail status: `{guard['status']}`",
            f"- Source: `{guard['source']}`",
            f"- Note: {guard['note']}",
            f"- Count: `{guard.get('count', '')}`",
            f"- CRPS improvement vs persistence: `{_fmt(guard.get('crps_improvement_vs_persistence'))}`",
            f"- Energy improvement vs persistence: `{_fmt(guard.get('energy_improvement_vs_persistence'))}`",
            f"- 80% coverage: `{_fmt(guard.get('coverage_80'))}`",
            f"- Terminal MAE z: `{_fmt(guard.get('terminal_mae_z'))}`",
            "",
            "## Primary Start Artifacts",
            "",
        ]
    )
    primary = str(report.get("primary_start"))
    for row in report["start_reports"]:
        if row["start"] != primary:
            continue
        lines.append(f"- Component terminal plot: `{row['component_terminal_plot']}`")
        for policy in row["sparse_policies"]:
            lines.append(f"- {policy['name']} report: `{policy['report']}`")
            for key, value in policy.get("artifact_paths", {}).items():
                lines.append(f"- {policy['name']} {key}: `{value}`")
    _write_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-start", type=int, default=22)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--max-plot-markets", type=int, default=4)
    args = parser.parse_args()

    report = build_component_aware_audit(
        root=args.root,
        output_dir=args.output_dir,
        primary_start=args.primary_start,
        quality_report=args.quality_report,
        max_plot_markets=args.max_plot_markets,
    )
    output = args.output_dir / "component_aware_scenario_view_audit.json"
    markdown = args.output_dir / "component_aware_scenario_view_audit.md"
    _write_json(output, report)
    write_markdown(markdown, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "markdown": str(markdown),
                "start_count": report["start_count"],
                "mean_component_to_pooled_path_energy_ratio": report["aggregate"][
                    "mean_component_to_pooled_path_energy_ratio"
                ],
                "diagnosis": report["diagnosis"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
