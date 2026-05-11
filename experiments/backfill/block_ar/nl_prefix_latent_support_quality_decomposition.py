#!/usr/bin/env python
"""Decompose support-quality changes between two narrative bridge reports.

This is a post-experiment diagnostic, not a new model family. It compares a
baseline bridge/scenario report against a candidate bridge/scenario report and
asks whether support retrieval, replay quality, and generator rollout quality
move together or against each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BASE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_BASE_SCENARIO_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_scenario_level_eval_openai_schema_v2_representative_220/"
    "scenario_level_eval_report.json"
)
DEFAULT_CANDIDATE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_memory_blend_pareto_871b_alpha025_artifacts/"
    "bridge_eval_report_blend_alpha250.json"
)
DEFAULT_CANDIDATE_SCENARIO_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_blend_alpha025_scenario_eval_872b_full/"
    "scenario_level_eval_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_quality_decomposition_872c_alpha025"
)

METRICS = (
    "energy_score_z",
    "ensemble_crps_z",
    "coverage_80",
    "mean_path_mae_z",
    "terminal_mae_z",
)


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


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    raw = float(value)
    return round(raw, 12) if np.isfinite(raw) else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(float(np.mean(np.asarray(values, dtype=np.float64))))


def _corr(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return _round(float(np.corrcoef(x, y)[0, 1]))


def _heldout_anchor_rows(bridge_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = bridge_report.get("evaluation", {}).get("heldout_examples", [])
    if not isinstance(rows, list):
        raise ValueError("bridge report missing evaluation.heldout_examples")
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("role", "")) != "anchor":
            continue
        window_index = int(row["window_index"])
        selected.setdefault(window_index, row)
    return selected


def _scenario_rows(scenario_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = scenario_report.get("window_scores", [])
    if not isinstance(rows, list):
        raise ValueError("scenario report missing window_scores")
    return {int(row["window_index"]): row for row in rows}


def _top_windows(row: dict[str, Any], *, top_k: int) -> list[str]:
    values = row.get("top_train_window_ids", [])
    if isinstance(values, list) and values:
        return [str(value) for value in values[: int(top_k)]]
    pool = row.get("top_train_pool", [])
    if isinstance(pool, list) and pool:
        return [
            str(item.get("window_id", item.get("window_index", "")))
            for item in pool[: int(top_k)]
        ]
    return [str(value) for value in values[: int(top_k)]]


def _top_cosines(row: dict[str, Any], *, top_k: int) -> list[float]:
    values = row.get("top_train_cosines", [])
    if isinstance(values, list) and values:
        return [float(value) for value in values[: int(top_k)]]
    pool = row.get("top_train_pool", [])
    if isinstance(pool, list) and pool:
        return [
            float(item["cosine"])
            for item in pool[: int(top_k)]
            if isinstance(item, dict) and item.get("cosine") is not None
        ]
    return [float(value) for value in values[: int(top_k)]]


def _method_metric(
    scenario_row: dict[str, Any],
    method: str,
    metric: str,
) -> float | None:
    methods = scenario_row.get("methods", {})
    if not isinstance(methods, dict):
        return None
    method_row = methods.get(method, {})
    if not isinstance(method_row, dict) or method_row.get(metric) is None:
        return None
    return float(method_row[metric])


def _method_summary_delta(
    base_report: dict[str, Any],
    candidate_report: dict[str, Any],
    method: str,
    metric: str,
) -> float | None:
    suffix = f"{metric}_improvement_vs_persistence"
    base_row = base_report.get("summary", {}).get(method, {})
    candidate_row = candidate_report.get("summary", {}).get(method, {})
    if not isinstance(base_row, dict) or not isinstance(candidate_row, dict):
        return None
    if base_row.get(suffix) is None or candidate_row.get(suffix) is None:
        return None
    return _round(float(candidate_row[suffix]) - float(base_row[suffix]))


def support_overlap_fraction(base_top: list[str], candidate_top: list[str]) -> float:
    """Return set overlap fraction relative to the smaller nonempty support set."""

    if not base_top or not candidate_top:
        return 0.0
    base_set = set(base_top)
    candidate_set = set(candidate_top)
    denom = max(1, min(len(base_set), len(candidate_set)))
    return float(len(base_set & candidate_set) / denom)


def decompose_support_quality(
    *,
    base_bridge_report: dict[str, Any],
    base_scenario_report: dict[str, Any],
    candidate_bridge_report: dict[str, Any],
    candidate_scenario_report: dict[str, Any],
    top_k: int = 3,
) -> dict[str, Any]:
    """Compare support changes and their scenario-score consequences."""

    base_bridge = _heldout_anchor_rows(base_bridge_report)
    candidate_bridge = _heldout_anchor_rows(candidate_bridge_report)
    base_scenario = _scenario_rows(base_scenario_report)
    candidate_scenario = _scenario_rows(candidate_scenario_report)
    common = sorted(
        set(base_bridge)
        & set(candidate_bridge)
        & set(base_scenario)
        & set(candidate_scenario)
    )
    if not common:
        raise ValueError("reports have no common held-out scenario windows")

    rows: list[dict[str, Any]] = []
    replay_energy_deltas: list[float] = []
    generator_energy_deltas: list[float] = []
    overlap_values: list[float] = []
    replay_better_generator_worse = 0
    for window_index in common:
        base_top = _top_windows(base_bridge[window_index], top_k=top_k)
        candidate_top = _top_windows(candidate_bridge[window_index], top_k=top_k)
        overlap = support_overlap_fraction(base_top, candidate_top)
        overlap_values.append(overlap)
        base_scenario_row = base_scenario[window_index]
        candidate_scenario_row = candidate_scenario[window_index]
        method_deltas: dict[str, dict[str, float | None]] = {}
        for method in ("historical_replay_topk", "narrative_generator_topk"):
            method_deltas[method] = {}
            for metric in METRICS:
                base_value = _method_metric(base_scenario_row, method, metric)
                candidate_value = _method_metric(candidate_scenario_row, method, metric)
                method_deltas[method][metric] = (
                    None
                    if base_value is None or candidate_value is None
                    else _round(candidate_value - base_value)
                )
        replay_energy = method_deltas["historical_replay_topk"]["energy_score_z"]
        generator_energy = method_deltas["narrative_generator_topk"]["energy_score_z"]
        if replay_energy is not None:
            replay_energy_deltas.append(float(replay_energy))
        if generator_energy is not None:
            generator_energy_deltas.append(float(generator_energy))
        if (
            replay_energy is not None
            and generator_energy is not None
            and replay_energy < 0.0
            and generator_energy > 0.0
        ):
            replay_better_generator_worse += 1
        rows.append(
            {
                "window_index": window_index,
                "window_id": str(base_bridge[window_index].get("window_id", "")),
                "base_top_windows": base_top,
                "candidate_top_windows": candidate_top,
                "topk_overlap_fraction": _round(overlap),
                "base_top1": base_top[0] if base_top else "",
                "candidate_top1": candidate_top[0] if candidate_top else "",
                "top1_changed": bool(
                    base_top and candidate_top and base_top[0] != candidate_top[0]
                ),
                "base_mean_top_cosine": _mean(
                    _top_cosines(base_bridge[window_index], top_k=top_k)
                ),
                "candidate_mean_top_cosine": _mean(
                    _top_cosines(candidate_bridge[window_index], top_k=top_k)
                ),
                "method_raw_score_deltas": method_deltas,
            }
        )

    top1_changed = [1.0 if row["top1_changed"] else 0.0 for row in rows]
    summary = {
        "window_count": len(rows),
        "mean_topk_overlap_fraction": _mean(overlap_values),
        "top1_changed_fraction": _mean(top1_changed),
        "replay_energy_raw_delta_mean": _mean(replay_energy_deltas),
        "generator_energy_raw_delta_mean": _mean(generator_energy_deltas),
        "replay_vs_generator_energy_delta_corr": _corr(
            replay_energy_deltas,
            generator_energy_deltas,
        ),
        "overlap_vs_generator_energy_delta_corr": _corr(
            overlap_values[: len(generator_energy_deltas)],
            generator_energy_deltas,
        ),
        "replay_better_generator_worse_count": replay_better_generator_worse,
        "summary_improvement_deltas": {
            "historical_replay_topk_energy": _method_summary_delta(
                base_scenario_report,
                candidate_scenario_report,
                "historical_replay_topk",
                "energy_score_z",
            ),
            "historical_replay_topk_crps": _method_summary_delta(
                base_scenario_report,
                candidate_scenario_report,
                "historical_replay_topk",
                "ensemble_crps_z",
            ),
            "narrative_generator_topk_energy": _method_summary_delta(
                base_scenario_report,
                candidate_scenario_report,
                "narrative_generator_topk",
                "energy_score_z",
            ),
            "narrative_generator_topk_crps": _method_summary_delta(
                base_scenario_report,
                candidate_scenario_report,
                "narrative_generator_topk",
                "ensemble_crps_z",
            ),
        },
    }
    decision = {
        "promote_candidate": False,
        "mechanism": (
            "support_replay_generator_mismatch"
            if (
                (
                    summary["summary_improvement_deltas"][
                        "historical_replay_topk_energy"
                    ]
                    or 0.0
                )
                > 0.0
                and (
                    summary["summary_improvement_deltas"][
                        "narrative_generator_topk_energy"
                    ]
                    or 0.0
                )
                < 0.0
            )
            else "support_change_needs_review"
        ),
        "next_step": (
            "Do not optimize memory target cosine alone. Inspect whether the "
            "candidate support pool improves actual future replay while moving "
            "the frozen generator into less calibrated analogue histories."
        ),
    }
    return {
        "status": "ok",
        "scope_note": (
            "Post-experiment support-quality decomposition. No model training, "
            "no OpenAI calls, and no generator rollout."
        ),
        "summary": summary,
        "decision": decision,
        "rows": rows,
    }


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):+.2f}%"


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Support Quality Decomposition",
        "",
        "This diagnostic compares the incumbent representative bridge with the candidate bridge.",
        "",
        "## Summary",
        "",
        f"- Windows: `{summary['window_count']}`",
        f"- Mean top-k support overlap: `{summary['mean_topk_overlap_fraction']}`",
        f"- Top-1 support changed fraction: `{summary['top1_changed_fraction']}`",
        f"- Replay raw energy delta: `{summary['replay_energy_raw_delta_mean']}`",
        f"- Generator raw energy delta: `{summary['generator_energy_raw_delta_mean']}`",
        f"- Replay/generator energy-delta correlation: `{summary['replay_vs_generator_energy_delta_corr']}`",
        f"- Replay-better but generator-worse windows: `{summary['replay_better_generator_worse_count']}`",
        "",
        "## Improvement Deltas",
        "",
    ]
    for key, value in summary["summary_improvement_deltas"].items():
        lines.append(f"- {key}: `{_format_pct(value)}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Promote candidate: `{report['decision']['promote_candidate']}`",
            f"- Mechanism: `{report['decision']['mechanism']}`",
            f"- Next step: {report['decision']['next_step']}",
            "",
        ]
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_decomposition(args: argparse.Namespace) -> dict[str, Any]:
    report = decompose_support_quality(
        base_bridge_report=_load_json(args.base_bridge_report),
        base_scenario_report=_load_json(args.base_scenario_report),
        candidate_bridge_report=_load_json(args.candidate_bridge_report),
        candidate_scenario_report=_load_json(args.candidate_scenario_report),
        top_k=int(args.top_k),
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "support_quality_decomposition.json"
    markdown_path = output_dir / "support_quality_decomposition.md"
    report["artifact_paths"] = {
        "report": str(json_path),
        "markdown": str(markdown_path),
    }
    _write_json(json_path, report)
    write_markdown_report(markdown_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bridge-report", default=DEFAULT_BASE_BRIDGE_REPORT)
    parser.add_argument("--base-scenario-report", default=DEFAULT_BASE_SCENARIO_REPORT)
    parser.add_argument(
        "--candidate-bridge-report",
        default=DEFAULT_CANDIDATE_BRIDGE_REPORT,
    )
    parser.add_argument(
        "--candidate-scenario-report",
        default=DEFAULT_CANDIDATE_SCENARIO_REPORT,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    report = run_decomposition(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "summary": report["summary"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
