#!/usr/bin/env python
"""Build a risk-manager-facing casebook from narrative scenario artifacts.

This is an offline product-evaluation layer. It does not call OpenAI and does
not resample the generator; it reads the saved narrative pipeline, bridge, and
scenario-level evaluation reports and turns them into inspectable cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LOWER_IS_BETTER_METRICS = {
    "energy_score_z": "energy_score",
    "ensemble_crps_z": "ensemble_crps",
    "mean_path_mae_z": "mean_path_mae",
    "mean_path_rmse_z": "mean_path_rmse",
    "terminal_mae_z": "terminal_mae",
}
OBSERVED_GROUNDING_STATUSES = {
    "market_fact_supported",
    "observed_market_facts_only",
    "observed_market_pattern",
}


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


def _round_float(value: Any, digits: int = 12) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw != raw or raw in {float("inf"), float("-inf")}:
        return None
    return round(raw, int(digits))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def metric_improvement_vs_baseline(score: Any, baseline: Any) -> float | None:
    """Return fractional improvement for lower-is-better metrics."""

    score_f = _round_float(score)
    baseline_f = _round_float(baseline)
    if score_f is None or baseline_f is None or baseline_f == 0.0:
        return None
    return _round_float((baseline_f - score_f) / baseline_f)


def classify_regime_tags(market_implications: list[dict[str, Any]]) -> list[str]:
    """Classify a case using only observed market implication directions."""

    by_market = {
        str(item.get("market", "")).upper(): {
            "direction": str(item.get("direction", "")).lower(),
            "magnitude": str(item.get("magnitude", "")).lower(),
        }
        for item in market_implications
        if isinstance(item, dict)
    }

    def is_dir(market: str, direction: str) -> bool:
        return by_market.get(market, {}).get("direction") == direction

    def is_material(market: str) -> bool:
        return by_market.get(market, {}).get("magnitude") in {"medium", "large"}

    credit_wider = is_dir("BBB_OAS", "wider") or is_dir("AAA_OAS", "wider")
    credit_tighter = is_dir("BBB_OAS", "tighter") or is_dir("AAA_OAS", "tighter")
    tags: list[str] = []
    if is_dir("SPX", "down") and (is_dir("VIX", "up") or credit_wider):
        tags.append("risk_off_stress")
    if is_dir("SPX", "up") and (is_dir("VIX", "down") or credit_tighter):
        tags.append("risk_on_recovery")
    if ((is_dir("US2Y", "up") or is_dir("US2Y", "down")) and is_material("US2Y")) or (
        (is_dir("US10Y", "up") or is_dir("US10Y", "down")) and is_material("US10Y")
    ):
        tags.append("rates_shock")
    if ((is_dir("VIX", "up") or is_dir("VIX", "down")) and is_material("VIX")) or (
        (is_dir("IV_SURFACE", "up") or is_dir("IV_SURFACE", "down"))
        and is_material("IV_SURFACE")
    ):
        tags.append("volatility_shock")
    if not tags:
        tags.append("mixed_or_ambiguous")
    return tags


def _index_by_window_id(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        window_id = str(row.get("window_id", ""))
        if not window_id:
            continue
        if window_id in indexed:
            raise ValueError(f"duplicate {label} window_id: {window_id}")
        indexed[window_id] = row
    return indexed


def _first_narrative_text(bundle: dict[str, Any]) -> str:
    narratives = _as_list(bundle.get("narratives"))
    if not narratives or not isinstance(narratives[0], dict):
        return ""
    return str(narratives[0].get("text", "")).strip()


def _compact_implications(
    market_implications: list[dict[str, Any]],
    *,
    include_flat: bool = False,
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in market_implications:
        if not isinstance(item, dict):
            continue
        direction = str(item.get("direction", "")).lower()
        if not include_flat and direction == "flat":
            continue
        rows.append(
            {
                "market": str(item.get("market", "")),
                "direction": direction,
                "magnitude": str(item.get("magnitude", "")),
                "evidence": (
                    list(item.get("evidence", []))
                    if isinstance(item.get("evidence"), list)
                    else []
                ),
            }
        )
    return rows[: int(limit)]


def _unique_catalysts(catalysts: list[Any]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for raw in catalysts:
        if not isinstance(raw, dict):
            continue
        key = (str(raw.get("label", "")), str(raw.get("description", "")))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "label": str(raw.get("label", "")),
                "grounding_status": str(raw.get("grounding_status", "")),
                "description": str(raw.get("description", "")),
            }
        )
    return rows


def _grounding_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    audit = _as_dict(bundle.get("hallucination_audit"))
    validation_issues = _as_list(audit.get("validation_issues"))
    catalysts = list(_as_list(bundle.get("narrative_catalysts")))
    for narrative in _as_list(bundle.get("narratives")):
        if isinstance(narrative, dict):
            catalysts.extend(_as_list(narrative.get("narrative_catalysts")))
    unique = _unique_catalysts(catalysts)
    non_observed = [
        row
        for row in unique
        if str(row.get("grounding_status", "")).lower()
        not in OBSERVED_GROUNDING_STATUSES
    ]
    return {
        "external_news_used": bool(audit.get("external_news_used", False)),
        "validation_issues": validation_issues,
        "validation_errors": [
            issue
            for issue in validation_issues
            if isinstance(issue, dict)
            and str(issue.get("severity", "")).lower() == "error"
        ],
        "validation_warnings": [
            issue
            for issue in validation_issues
            if isinstance(issue, dict)
            and str(issue.get("severity", "")).lower() != "error"
        ],
        "audit_critique": [str(item) for item in _as_list(audit.get("critique"))],
        "non_observed_catalysts": non_observed,
    }


def _bridge_rows_by_window(
    bridge_report: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    heldout: dict[str, dict[str, Any]] = {}
    for row in _as_list(
        _as_dict(bridge_report.get("evaluation")).get("heldout_examples")
    ):
        if not isinstance(row, dict) or str(row.get("role", "")) != "anchor":
            continue
        window_id = str(row.get("window_id", ""))
        if window_id and window_id not in heldout:
            heldout[window_id] = row
    hard_negatives = {
        str(row.get("window_id", "")): row
        for row in _as_list(
            _as_dict(
                _as_dict(bridge_report.get("evaluation")).get(
                    "hard_negative_separation"
                )
            ).get("windows")
        )
        if isinstance(row, dict) and row.get("window_id")
    }
    return heldout, hard_negatives


def _score_methods(methods: dict[str, Any]) -> dict[str, dict[str, Any]]:
    persistence = _as_dict(methods.get("persistence"))
    scored: dict[str, dict[str, Any]] = {}
    for method, raw_metrics in methods.items():
        if not isinstance(raw_metrics, dict):
            continue
        row = {
            key: _round_float(value)
            for key, value in raw_metrics.items()
            if isinstance(value, (int, float)) or value is None
        }
        if method != "persistence":
            for metric_key, label in LOWER_IS_BETTER_METRICS.items():
                row[f"{label}_improvement_vs_persistence"] = (
                    metric_improvement_vs_baseline(
                        raw_metrics.get(metric_key),
                        persistence.get(metric_key),
                    )
                )
        scored[str(method)] = row
    return scored


def _analogue_rows(
    scenario_row: dict[str, Any],
    bridge_row: dict[str, Any],
    bundles_by_id: dict[str, dict[str, Any]],
    bundles: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    ids = [str(value) for value in _as_list(scenario_row.get("top_train_window_ids"))]
    cosines = [
        _round_float(value) for value in _as_list(scenario_row.get("top_train_cosines"))
    ]
    if not ids:
        for item in _as_list(bridge_row.get("top_train_pool")):
            if isinstance(item, dict):
                ids.append(str(item.get("window_id", "")))
                cosines.append(_round_float(item.get("cosine")))
    rows: list[dict[str, Any]] = []
    for rank, window_id in enumerate(ids[: int(top_k)], start=1):
        bundle = bundles_by_id.get(window_id)
        if bundle is None:
            window_index = None
            for item in _as_list(bridge_row.get("top_train_pool")):
                if (
                    isinstance(item, dict)
                    and str(item.get("window_id", "")) == window_id
                ):
                    window_index = item.get("window_index")
                    break
            if window_index is not None and 0 <= int(window_index) < len(bundles):
                bundle = bundles[int(window_index)]
        bundle = bundle or {}
        metadata = _as_dict(bundle.get("window_metadata"))
        rows.append(
            {
                "rank": rank,
                "window_id": window_id,
                "similarity": cosines[rank - 1] if rank - 1 < len(cosines) else None,
                "window_index": bundle.get("window_index"),
                "source_index": bundle.get("source_index"),
                "forecast_start_date": metadata.get("forecast_start_date"),
                "forecast_end_date": metadata.get("forecast_end_date"),
                "primary_narrative": _first_narrative_text(bundle),
                "market_implications": _compact_implications(
                    _as_list(bundle.get("market_implications")),
                    limit=6,
                ),
            }
        )
    return rows


def _failure_flags(
    *,
    bridge_row: dict[str, Any],
    grounding: dict[str, Any],
    scores: dict[str, dict[str, Any]],
) -> list[str]:
    flags: list[str] = []
    target_cosine = _round_float(bridge_row.get("target_cosine"))
    if target_cosine is not None and target_cosine < 0.65:
        flags.append("weak_bridge_alignment")
    if grounding.get("validation_errors"):
        flags.append("label_validation_issue")
    narrative = scores.get("narrative_generator_topk", {})
    energy = narrative.get("energy_score_improvement_vs_persistence")
    crps = narrative.get("ensemble_crps_improvement_vs_persistence")
    point = narrative.get("mean_path_mae_improvement_vs_persistence")
    if (
        energy is not None
        and crps is not None
        and point is not None
        and energy > 0.0
        and crps > 0.0
        and point < 0.0
    ):
        flags.append("distribution_good_point_path_weak")
    if not flags:
        flags.append("no_major_failure_flag")
    return flags


def build_casebook(
    pipeline_report: dict[str, Any],
    bridge_report: dict[str, Any],
    scenario_report: dict[str, Any],
    *,
    title: str = "Risk Manager Narrative Scenario Casebook",
    max_cases: int = 0,
    analogue_top_k: int = 3,
) -> dict[str, Any]:
    """Combine saved reports into case-level risk-manager evidence."""

    bundles = [
        item
        for item in _as_list(pipeline_report.get("narrative_bundles"))
        if isinstance(item, dict)
    ]
    bundles_by_id = _index_by_window_id(bundles, label="narrative bundle")
    bridge_by_id, hard_negative_by_id = _bridge_rows_by_window(bridge_report)
    scenario_rows = [
        row
        for row in _as_list(scenario_report.get("window_scores"))
        if isinstance(row, dict)
    ]
    if int(max_cases) > 0:
        scenario_rows = scenario_rows[: int(max_cases)]

    cases: list[dict[str, Any]] = []
    for row_no, scenario_row in enumerate(scenario_rows, start=1):
        window_id = str(scenario_row.get("window_id", ""))
        bundle = bundles_by_id.get(window_id, {})
        metadata = _as_dict(bundle.get("window_metadata"))
        bridge_row = bridge_by_id.get(window_id, {})
        hard_negative = hard_negative_by_id.get(window_id, {})
        implications = _compact_implications(
            _as_list(bundle.get("market_implications")),
            include_flat=False,
            limit=12,
        )
        grounding = _grounding_summary(bundle)
        scores = _score_methods(_as_dict(scenario_row.get("methods")))
        cases.append(
            {
                "case_no": row_no,
                "window_id": window_id,
                "bridge_local_window_index": scenario_row.get("window_index"),
                "block_window_index": scenario_row.get("block_window_index"),
                "manifest_window_index": bundle.get("window_index"),
                "source_index": bundle.get("source_index"),
                "manifest_split": metadata.get("manifest_split"),
                "calendar": {
                    "history_start": metadata.get("calendar_start_date"),
                    "history_end": metadata.get("calendar_end_date"),
                    "forecast_start": metadata.get("forecast_start_date"),
                    "forecast_end": metadata.get("forecast_end_date"),
                },
                "selection_reasons": (
                    list(metadata.get("selection_reasons", []))
                    if isinstance(metadata.get("selection_reasons"), list)
                    else (
                        list(bundle.get("selection_reasons", []))
                        if isinstance(bundle.get("selection_reasons"), list)
                        else []
                    )
                ),
                "regime_tags": classify_regime_tags(implications),
                "input_narrative": _first_narrative_text(bundle),
                "market_implications": implications,
                "grounding": grounding,
                "bridge": {
                    "query_kind": bridge_row.get("kind")
                    or scenario_row.get("query_kind"),
                    "target_cosine": _round_float(bridge_row.get("target_cosine")),
                    "true_rank_test_pool": bridge_row.get("true_rank_test_pool"),
                    "true_rank_full_pool": bridge_row.get("true_rank_full_pool"),
                    "hard_negative_positive_mean_cosine": _round_float(
                        hard_negative.get("positive_mean_cosine")
                    ),
                    "hard_negative_negative_mean_cosine": _round_float(
                        hard_negative.get("negative_mean_cosine")
                    ),
                    "hard_negative_gap": _round_float(
                        hard_negative.get("negative_gap")
                    ),
                },
                "historical_analogues": _analogue_rows(
                    scenario_row,
                    bridge_row,
                    bundles_by_id,
                    bundles,
                    top_k=int(analogue_top_k),
                ),
                "scores": scores,
                "failure_flags": _failure_flags(
                    bridge_row=bridge_row,
                    grounding=grounding,
                    scores=scores,
                ),
            }
        )

    regime_counts = Counter(tag for case in cases for tag in case["regime_tags"])
    flag_counts = Counter(flag for case in cases for flag in case["failure_flags"])
    return {
        "title": title,
        "status": "risk_manager_casebook",
        "summary": {
            "case_count": len(cases),
            "regime_counts": dict(sorted(regime_counts.items())),
            "failure_flag_counts": dict(sorted(flag_counts.items())),
            "label_backend": pipeline_report.get("label_backend"),
            "embedding_backend": pipeline_report.get("embedding_backend"),
            "selection_manifest": pipeline_report.get("selection_manifest"),
            "rejected_label_windows": list(
                pipeline_report.get("rejected_label_windows", [])
            ),
            "bridge_summary": _as_dict(bridge_report.get("summary")),
            "scenario_summary": _as_dict(scenario_report.get("summary")),
        },
        "cases": cases,
    }


def _format_float(value: Any, digits: int = 3) -> str:
    rounded = _round_float(value, digits)
    return "n/a" if rounded is None else f"{rounded:.{digits}f}"


def _format_pct(value: Any, digits: int = 1) -> str:
    rounded = _round_float(value)
    return "n/a" if rounded is None else f"{rounded * 100.0:.{digits}f}%"


def _implication_text(rows: list[dict[str, Any]]) -> str:
    parts = [
        f"{row['market']} {row['direction']} {row['magnitude']}".strip() for row in rows
    ]
    return "; ".join(parts) if parts else "No material non-flat implication listed."


def _render_method_table(scores: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | Energy | CRPS | Coverage 80 | Mean-path MAE | Energy vs persistence | CRPS vs persistence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in [
        "narrative_generator_topk",
        "historical_replay_topk",
        "oracle_generator_true_history",
        "persistence",
        "train_median_delta",
    ]:
        row = scores.get(method)
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _format_float(row.get("energy_score_z")),
                    _format_float(row.get("ensemble_crps_z")),
                    _format_float(row.get("coverage_80")),
                    _format_float(row.get("mean_path_mae_z")),
                    _format_pct(row.get("energy_score_improvement_vs_persistence")),
                    _format_pct(row.get("ensemble_crps_improvement_vs_persistence")),
                ]
            )
            + " |"
        )
    return lines


def _render_analogues(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rank | Window | Similarity | Forecast dates | Narrative |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
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
                    _format_float(row.get("similarity")),
                    dates,
                    narrative or "n/a",
                ]
            )
            + " |"
        )
    return lines


def render_casebook_markdown(casebook: dict[str, Any]) -> str:
    """Render the casebook as a compact Markdown report."""

    title = str(casebook.get("title") or "Risk Manager Narrative Scenario Casebook")
    summary = _as_dict(casebook.get("summary"))
    lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Label backend: {summary.get('label_backend')}",
        f"- Embedding backend: {summary.get('embedding_backend')}",
        f"- Rejected label windows: {len(_as_list(summary.get('rejected_label_windows')))}",
        f"- Regime counts: {json.dumps(summary.get('regime_counts', {}), sort_keys=True)}",
        f"- Failure flags: {json.dumps(summary.get('failure_flag_counts', {}), sort_keys=True)}",
        "",
        "## Aggregate Scenario Metrics",
        "",
    ]
    narrative_summary = _as_dict(
        _as_dict(summary.get("scenario_summary")).get("narrative_generator_topk")
    )
    bridge_summary = _as_dict(summary.get("bridge_summary"))
    lines.extend(
        [
            f"- Narrative generator energy improvement vs persistence: {_format_pct(narrative_summary.get('energy_score_z_improvement_vs_persistence'))}",
            f"- Narrative generator CRPS improvement vs persistence: {_format_pct(narrative_summary.get('ensemble_crps_z_improvement_vs_persistence'))}",
            f"- Narrative generator coverage 80 mean: {_format_float(narrative_summary.get('coverage_80_mean'))}",
            f"- Bridge mean target cosine: {_format_float(bridge_summary.get('heldout_mean_target_cosine'))}",
            f"- Bridge hard-negative gap: {_format_float(bridge_summary.get('heldout_hard_negative_mean_gap'))}",
            "",
        ]
    )
    for case in _as_list(casebook.get("cases")):
        if not isinstance(case, dict):
            continue
        lines.extend(
            [
                f"## Case {case.get('case_no')}: {case.get('window_id')}",
                "",
                f"- Regime tags: {', '.join(case.get('regime_tags', []))}",
                f"- Source index: {case.get('source_index')}; validation window: {case.get('manifest_window_index')}; bridge-local row: {case.get('bridge_local_window_index')}",
                f"- Forecast window: {_as_dict(case.get('calendar')).get('forecast_start')} to {_as_dict(case.get('calendar')).get('forecast_end')}",
                f"- Selection reasons: {', '.join(case.get('selection_reasons', [])) or 'n/a'}",
                "",
                "### Input Narrative",
                "",
                str(case.get("input_narrative", "") or "n/a"),
                "",
                "### Extracted Market Implications",
                "",
                _implication_text(_as_list(case.get("market_implications"))),
                "",
                "### Grounding and Hallucination Checks",
                "",
                f"- External news used: {_as_dict(case.get('grounding')).get('external_news_used')}",
                f"- Validation errors: {len(_as_list(_as_dict(case.get('grounding')).get('validation_errors')))}",
                f"- Validation warnings: {len(_as_list(_as_dict(case.get('grounding')).get('validation_warnings')))}",
                f"- Non-observed catalysts/analogies: {len(_as_list(_as_dict(case.get('grounding')).get('non_observed_catalysts')))}",
                "",
                "### Historical Analogues",
                "",
            ]
        )
        lines.extend(_render_analogues(_as_list(case.get("historical_analogues"))))
        lines.extend(
            [
                "",
                "### Bridge and Scenario Scores",
                "",
                f"- Bridge target cosine: {_format_float(_as_dict(case.get('bridge')).get('target_cosine'))}",
                f"- Test-pool true rank: {_as_dict(case.get('bridge')).get('true_rank_test_pool')}",
                f"- Hard-negative gap: {_format_float(_as_dict(case.get('bridge')).get('hard_negative_gap'))}",
                "",
            ]
        )
        lines.extend(_render_method_table(_as_dict(case.get("scores"))))
        lines.extend(
            [
                "",
                "### Failure Flags",
                "",
                ", ".join(case.get("failure_flags", [])),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", required=True)
    parser.add_argument("--bridge-report", required=True)
    parser.add_argument("--scenario-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Risk Manager Narrative Scenario Casebook")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--analogue-top-k", type=int, default=3)
    args = parser.parse_args()

    casebook = build_casebook(
        _load_json(args.pipeline_report),
        _load_json(args.bridge_report),
        _load_json(args.scenario_report),
        title=args.title,
        max_cases=int(args.max_cases),
        analogue_top_k=int(args.analogue_top_k),
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "casebook.json"
    markdown_path = output_dir / "casebook.md"
    _write_json(json_path, casebook)
    _write_text(markdown_path, render_casebook_markdown(casebook))
    print(
        json.dumps(
            {
                "case_count": casebook["summary"]["case_count"],
                "casebook_json": str(json_path),
                "casebook_markdown": str(markdown_path),
                "regime_counts": casebook["summary"]["regime_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
