#!/usr/bin/env python
"""Render product-facing scenario reports from risk-manager casebooks.

This layer is intentionally offline. It packages saved casebook evidence into a
report contract that says exactly what the product is: a distributional scenario
and historical-analogue tool, not a point forecast.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRODUCT_POSITIONING = "scenario_distribution_not_point_forecast"
REQUIRED_SECTIONS = [
    "positioning",
    "input_narrative",
    "market_implications",
    "grounding",
    "historical_analogues",
    "distribution_metrics",
]


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


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw != raw or raw in {float("inf"), float("-inf")}:
        return None
    return raw


def _fmt_float(value: Any, digits: int = 3) -> str:
    raw = _float(value)
    return "n/a" if raw is None else f"{raw:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    raw = _float(value)
    return "n/a" if raw is None else f"{raw * 100.0:.{digits}f}%"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "case"


def _acceptance_by_window(
    acceptance_audit: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not acceptance_audit:
        return {}
    rows = {}
    for row in _as_list(acceptance_audit.get("case_results")):
        if isinstance(row, dict) and row.get("window_id"):
            rows[str(row["window_id"])] = row
    return rows


def _compact_implications(rows: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "market": str(row.get("market", "")),
                "direction": str(row.get("direction", "")),
                "magnitude": str(row.get("magnitude", "")),
                "evidence": (
                    list(row.get("evidence", []))
                    if isinstance(row.get("evidence"), list)
                    else []
                ),
            }
        )
    return compact


def _distribution_metrics(case: dict[str, Any]) -> dict[str, Any]:
    narrative = _as_dict(_as_dict(case.get("scores")).get("narrative_generator_topk"))
    return {
        "energy_score_improvement_vs_persistence": _float(
            narrative.get("energy_score_improvement_vs_persistence")
        ),
        "ensemble_crps_improvement_vs_persistence": _float(
            narrative.get("ensemble_crps_improvement_vs_persistence")
        ),
        "coverage_80": _float(narrative.get("coverage_80")),
        "mean_path_mae_improvement_vs_persistence": _float(
            narrative.get("mean_path_mae_improvement_vs_persistence")
        ),
        "product_interpretation": (
            "Use energy score, CRPS, coverage, analogues, and path dispersion as "
            "scenario evidence. Mean-path errors are diagnostic only and should "
            "not be presented as a point forecast."
        ),
    }


def _grounding_section(case: dict[str, Any]) -> dict[str, Any]:
    grounding = _as_dict(case.get("grounding"))
    return {
        "external_news_used": bool(grounding.get("external_news_used", False)),
        "validation_errors": _as_list(grounding.get("validation_errors")),
        "validation_warnings": _as_list(grounding.get("validation_warnings")),
        "non_observed_catalysts": _as_list(grounding.get("non_observed_catalysts")),
        "policy": (
            "Named events and macro stories are allowed only as analogies or "
            "hypotheses unless explicitly grounded by evidence."
        ),
    }


def _framing_checks(report: dict[str, Any]) -> dict[str, bool]:
    sections = _as_dict(report.get("sections"))
    metrics = _as_dict(sections.get("distribution_metrics"))
    return {
        "distribution_metrics_present": (
            metrics.get("energy_score_improvement_vs_persistence") is not None
            and metrics.get("ensemble_crps_improvement_vs_persistence") is not None
        ),
        "grounding_section_present": bool(_as_dict(sections.get("grounding"))),
        "historical_analogues_present": bool(
            _as_list(sections.get("historical_analogues"))
        ),
        "not_point_forecast_statement": "not a point forecast"
        in str(sections.get("positioning", "")).lower(),
    }


def _framing_ready(checks: dict[str, bool]) -> bool:
    return all(bool(value) for value in checks.values())


def _build_case_report(
    case: dict[str, Any],
    *,
    case_no: int,
    acceptance_row: dict[str, Any] | None,
) -> dict[str, Any]:
    window_id = str(case.get("window_id", ""))
    sections = {
        "positioning": (
            "This report is a scenario distribution and historical-analogue view, "
            "not a point forecast. The generated paths should be read as a "
            "conditional range of plausible 30-day outcomes under the stated "
            "market narrative."
        ),
        "input_narrative": str(case.get("input_narrative", "")),
        "market_implications": _compact_implications(
            _as_list(case.get("market_implications"))
        ),
        "grounding": _grounding_section(case),
        "historical_analogues": _as_list(case.get("historical_analogues")),
        "distribution_metrics": _distribution_metrics(case),
    }
    report = {
        "case_no": int(case_no),
        "window_id": window_id,
        "regime_tags": (
            list(case.get("regime_tags", []))
            if isinstance(case.get("regime_tags"), list)
            else []
        ),
        "acceptance_status": (acceptance_row or {}).get("status"),
        "acceptance_bottlenecks": (
            list((acceptance_row or {}).get("bottleneck_tags", []))
            if isinstance((acceptance_row or {}).get("bottleneck_tags"), list)
            else []
        ),
        "report_contract": PRODUCT_POSITIONING,
        "sections": sections,
    }
    report["framing_checks"] = _framing_checks(report)
    report["framing_ready"] = _framing_ready(report["framing_checks"])
    return report


def build_product_report_package(
    casebook: dict[str, Any],
    acceptance_audit: dict[str, Any] | None = None,
    *,
    title: str = "Product Scenario Reports",
) -> dict[str, Any]:
    """Build product-facing report objects from a casebook and optional audit."""

    acceptance_rows = _acceptance_by_window(acceptance_audit)
    cases = [case for case in _as_list(casebook.get("cases")) if isinstance(case, dict)]
    reports = [
        _build_case_report(
            case,
            case_no=index,
            acceptance_row=acceptance_rows.get(str(case.get("window_id", ""))),
        )
        for index, case in enumerate(cases, start=1)
    ]
    acceptance_status_counts = Counter(
        str(report.get("acceptance_status"))
        for report in reports
        if report.get("acceptance_status") is not None
    )
    return {
        "title": title,
        "status": "product_scenario_report_package",
        "source_casebook_title": casebook.get("title"),
        "product_contract": {
            "report_type": "risk_manager_scenario_distribution_report",
            "positioning": PRODUCT_POSITIONING,
            "required_sections": list(REQUIRED_SECTIONS),
        },
        "summary": {
            "case_count": len(reports),
            "distributional_framing_ready_count": sum(
                1 for report in reports if bool(report.get("framing_ready"))
            ),
            "acceptance_status_counts": dict(sorted(acceptance_status_counts.items())),
        },
        "case_framing": {
            report["window_id"]: {
                "framing_ready": bool(report["framing_ready"]),
                "framing_checks": report["framing_checks"],
                "report_contract": report["report_contract"],
            }
            for report in reports
            if report.get("window_id")
        },
        "case_reports": reports,
    }


def _implication_line(rows: list[Any]) -> str:
    pieces = []
    for row in rows:
        if isinstance(row, dict):
            pieces.append(
                f"{row.get('market')} {row.get('direction')} {row.get('magnitude')}".strip()
            )
    return "; ".join(pieces) if pieces else "No market implications listed."


def _render_analogues(rows: list[Any]) -> list[str]:
    lines = [
        "| Rank | Window | Similarity | Forecast dates | Narrative |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        dates = f"{row.get('forecast_start_date') or '?'} to {row.get('forecast_end_date') or '?'}"
        narrative = str(row.get("primary_narrative", "")).replace("\n", " ")
        if len(narrative) > 180:
            narrative = narrative[:177].rstrip() + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank", "")),
                    str(row.get("window_id", "")),
                    _fmt_float(row.get("similarity")),
                    dates,
                    narrative or "n/a",
                ]
            )
            + " |"
        )
    return lines


def _render_distribution_metrics(metrics: dict[str, Any]) -> list[str]:
    return [
        f"- Energy score improvement vs persistence: {_fmt_pct(metrics.get('energy_score_improvement_vs_persistence'))}",
        f"- Ensemble CRPS improvement vs persistence: {_fmt_pct(metrics.get('ensemble_crps_improvement_vs_persistence'))}",
        f"- 80% coverage: {_fmt_float(metrics.get('coverage_80'))}",
        f"- Mean-path MAE improvement vs persistence: {_fmt_pct(metrics.get('mean_path_mae_improvement_vs_persistence'))}",
        f"- Product interpretation: {metrics.get('product_interpretation')}",
    ]


def render_case_report_markdown(report: dict[str, Any]) -> str:
    """Render a single product case report."""

    sections = _as_dict(report.get("sections"))
    grounding = _as_dict(sections.get("grounding"))
    lines = [
        f"# Case {report.get('case_no')}: {report.get('window_id')}",
        "",
        "## Scenario Distribution, Not Point Forecast",
        "",
        str(sections.get("positioning", "")),
        "",
        f"- Acceptance status: {report.get('acceptance_status') or 'not audited'}",
        f"- Regime tags: {', '.join(report.get('regime_tags', [])) or 'unclassified'}",
        f"- Framing ready: {report.get('framing_ready')}",
        "",
        "## Input Narrative",
        "",
        str(sections.get("input_narrative", "")),
        "",
        "## Extracted Market Implications",
        "",
        _implication_line(_as_list(sections.get("market_implications"))),
        "",
        "## Grounding and Hallucination Controls",
        "",
        f"- External news used: {grounding.get('external_news_used')}",
        f"- Validation errors: {len(_as_list(grounding.get('validation_errors')))}",
        f"- Validation warnings: {len(_as_list(grounding.get('validation_warnings')))}",
        f"- Non-observed catalysts/analogies: {len(_as_list(grounding.get('non_observed_catalysts')))}",
        f"- Policy: {grounding.get('policy')}",
        "",
        "## Historical Analogues",
        "",
    ]
    lines.extend(_render_analogues(_as_list(sections.get("historical_analogues"))))
    lines.extend(["", "## Distribution Metrics", ""])
    lines.extend(
        _render_distribution_metrics(_as_dict(sections.get("distribution_metrics")))
    )
    return "\n".join(lines)


def render_product_report_markdown(package: dict[str, Any]) -> str:
    """Render an aggregate product report package."""

    summary = _as_dict(package.get("summary"))
    contract = _as_dict(package.get("product_contract"))
    lines = [
        f"# {package.get('title') or 'Product Scenario Reports'}",
        "",
        "## Product Contract",
        "",
        f"- Report type: {contract.get('report_type')}",
        f"- Positioning: {contract.get('positioning')}",
        f"- Required sections: {', '.join(_as_list(contract.get('required_sections')))}",
        "",
        "## Summary",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Distributional framing ready: {summary.get('distributional_framing_ready_count', 0)}",
        f"- Acceptance status counts: {json.dumps(summary.get('acceptance_status_counts', {}), sort_keys=True)}",
        "",
    ]
    for report in _as_list(package.get("case_reports")):
        if isinstance(report, dict):
            lines.append(
                render_case_report_markdown(report).replace("# Case", "## Case", 1)
            )
            lines.append("")
    return "\n".join(lines)


def write_product_report_package(
    package: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    """Write aggregate and per-case product report artifacts."""

    root = Path(output_dir)
    paths = {
        "summary_json": str(root / "product_report_summary.json"),
        "summary_markdown": str(root / "product_report.md"),
    }
    _write_json(paths["summary_json"], package)
    _write_text(paths["summary_markdown"], render_product_report_markdown(package))
    cases_dir = root / "cases"
    for report in _as_list(package.get("case_reports")):
        if not isinstance(report, dict):
            continue
        slug = _slug(str(report.get("window_id", "")))
        _write_json(cases_dir / f"{slug}.json", report)
        _write_text(cases_dir / f"{slug}.md", render_case_report_markdown(report))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", required=True)
    parser.add_argument("--acceptance-audit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Product Scenario Reports")
    args = parser.parse_args()

    acceptance = _load_json(args.acceptance_audit) if args.acceptance_audit else None
    package = build_product_report_package(
        _load_json(args.casebook),
        acceptance,
        title=args.title,
    )
    paths = write_product_report_package(package, args.output_dir)
    print(
        json.dumps(
            {
                **paths,
                "case_count": package["summary"]["case_count"],
                "distributional_framing_ready_count": package["summary"][
                    "distributional_framing_ready_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
