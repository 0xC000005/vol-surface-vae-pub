#!/usr/bin/env python
"""Compare episode-narrative retrieval against a start-only support baseline.

This is an isolated Phase 3 audit for the episode-level narrative retrieval
branch. CRPS and Energy remain guardrails, but the primary question here is
conditionality: does the narrative condition add support and distributional
movement beyond what the accepted starting level alone would produce?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_NARRATIVE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_scenario_eval_970i_top3_full66_s4_oracle/"
    "scenario_level_eval_report.json"
)
DEFAULT_NARRATIVE_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_scenario_eval_970i_top3_full66_s4_oracle/"
    "scenario_level_eval_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_conditionality_lift"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _finite_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _finite_median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 12)


def _window_rows(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = report.get("window_scores", [])
    if not isinstance(rows, list):
        raise ValueError("scenario report missing window_scores")
    return {int(row["window_index"]): row for row in rows if "window_index" in row}


def _support_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("top_train_window_ids")
    if isinstance(ids, list) and ids:
        return [str(item) for item in ids]
    indices = row.get("top_train_indices")
    if isinstance(indices, list):
        return [str(int(item)) for item in indices]
    return []


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 0.0


def _array_key(window_index: int) -> str:
    return f"narrative_{int(window_index)}"


def _ks_1d(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    right_arr = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    if left_arr.size == 0 or right_arr.size == 0:
        return 0.0
    values = np.concatenate([left_arr, right_arr])
    left_cdf = np.searchsorted(left_arr, values, side="right") / float(left_arr.size)
    right_cdf = np.searchsorted(right_arr, values, side="right") / float(right_arr.size)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(left.shape[0], -1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(right.shape[0], -1)
    if left_flat.size == 0 or right_flat.size == 0:
        return 0.0
    cross = np.linalg.norm(
        left_flat[:, None, :] - right_flat[None, :, :], axis=-1
    ).mean()
    left_self = (
        np.linalg.norm(left_flat[:, None, :] - left_flat[None, :, :], axis=-1).mean()
        if left_flat.shape[0] > 1
        else 0.0
    )
    right_self = (
        np.linalg.norm(right_flat[:, None, :] - right_flat[None, :, :], axis=-1).mean()
        if right_flat.shape[0] > 1
        else 0.0
    )
    return float(
        max(0.0, 2.0 * cross - left_self - right_self) / math.sqrt(left_flat.shape[1])
    )


def _quality_metric(report: dict[str, Any], metric: str) -> float | None:
    block = report.get("summary", {}).get("narrative_generator_topk", {})
    value = block.get(f"{metric}_mean")
    return None if value is None else float(value)


def build_conditionality_lift_report(
    *,
    narrative_report: dict[str, Any],
    start_only_report: dict[str, Any],
    narrative_arrays: dict[str, np.ndarray],
    start_only_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Return support and distribution lift of narrative over start-only retrieval."""

    narrative_rows = _window_rows(narrative_report)
    start_rows = _window_rows(start_only_report)
    common_windows = sorted(set(narrative_rows).intersection(start_rows))
    if not common_windows:
        raise ValueError("reports have no common evaluated window_index")
    scale = np.asarray(narrative_arrays.get("delta_scale"), dtype=np.float32)
    if scale.ndim != 2:
        raise ValueError("narrative arrays missing delta_scale [T,C]")
    rows: list[dict[str, Any]] = []
    for window_index in common_windows:
        key = _array_key(window_index)
        if key not in narrative_arrays or key not in start_only_arrays:
            continue
        narrative_samples = np.asarray(narrative_arrays[key], dtype=np.float32)
        start_samples = np.asarray(start_only_arrays[key], dtype=np.float32)
        if narrative_samples.ndim != 3 or start_samples.ndim != 3:
            continue
        safe_scale = np.maximum(scale, 1e-8)
        narrative_z = narrative_samples / safe_scale[None, :, :]
        start_z = start_samples / safe_scale[None, :, :]
        channel_count = narrative_z.shape[-1]
        terminal_ks = [
            _ks_1d(narrative_z[:, -1, channel], start_z[:, -1, channel])
            for channel in range(channel_count)
        ]
        terminal_mean_shift_z = np.abs(
            narrative_z[:, -1, :].mean(axis=0) - start_z[:, -1, :].mean(axis=0)
        )
        narrative_ids = _support_ids(narrative_rows[window_index])
        start_ids = _support_ids(start_rows[window_index])
        rows.append(
            {
                "window_index": int(window_index),
                "support_jaccard": _jaccard(narrative_ids, start_ids),
                "support_jaccard_distance": 1.0 - _jaccard(narrative_ids, start_ids),
                "mean_terminal_factor_ks": float(np.mean(terminal_ks)),
                "max_terminal_factor_ks": float(np.max(terminal_ks)),
                "mean_abs_terminal_mean_shift_z": float(np.mean(terminal_mean_shift_z)),
                "max_abs_terminal_mean_shift_z": float(np.max(terminal_mean_shift_z)),
                "path_energy_distance_z": _energy_distance(narrative_z, start_z),
                "narrative_support_ids": narrative_ids,
                "start_only_support_ids": start_ids,
            }
        )
    if not rows:
        raise ValueError("no comparable narrative/start-only sample arrays found")

    aggregate = {
        "mean_support_jaccard": _round(
            _finite_mean([r["support_jaccard"] for r in rows])
        ),
        "median_support_jaccard": _round(
            _finite_median([r["support_jaccard"] for r in rows])
        ),
        "mean_support_jaccard_distance": _round(
            _finite_mean([r["support_jaccard_distance"] for r in rows])
        ),
        "mean_terminal_factor_ks": _round(
            _finite_mean([r["mean_terminal_factor_ks"] for r in rows])
        ),
        "median_terminal_factor_ks": _round(
            _finite_median([r["mean_terminal_factor_ks"] for r in rows])
        ),
        "mean_path_energy_distance_z": _round(
            _finite_mean([r["path_energy_distance_z"] for r in rows])
        ),
        "median_path_energy_distance_z": _round(
            _finite_median([r["path_energy_distance_z"] for r in rows])
        ),
        "mean_abs_terminal_mean_shift_z": _round(
            _finite_mean([r["mean_abs_terminal_mean_shift_z"] for r in rows])
        ),
    }
    narrative_crps = _quality_metric(narrative_report, "ensemble_crps_z")
    start_crps = _quality_metric(start_only_report, "ensemble_crps_z")
    narrative_energy = _quality_metric(narrative_report, "energy_score_z")
    start_energy = _quality_metric(start_only_report, "energy_score_z")
    quality_guardrail = {
        "narrative_crps": _round(narrative_crps),
        "start_only_crps": _round(start_crps),
        "crps_delta_vs_start_only": _round(
            None
            if narrative_crps is None or start_crps is None
            else narrative_crps - start_crps
        ),
        "narrative_energy": _round(narrative_energy),
        "start_only_energy": _round(start_energy),
        "energy_delta_vs_start_only": _round(
            None
            if narrative_energy is None or start_energy is None
            else narrative_energy - start_energy
        ),
    }

    support_lift = float(aggregate["mean_support_jaccard_distance"] or 0.0)
    factor_lift = float(aggregate["mean_terminal_factor_ks"] or 0.0)
    path_lift = float(aggregate["mean_path_energy_distance_z"] or 0.0)
    crps_delta = quality_guardrail["crps_delta_vs_start_only"]
    quality_ok = crps_delta is None or float(crps_delta) <= 0.05
    has_distribution_lift = support_lift >= 0.50 and (
        factor_lift >= 0.10 or path_lift >= 0.10
    )
    if has_distribution_lift and quality_ok:
        verdict = "conditionality_lift_detected"
    elif has_distribution_lift:
        verdict = "conditionality_lift_detected_quality_warning"
    elif support_lift >= 0.50 and factor_lift < 0.10 and path_lift < 0.10:
        verdict = "support_changes_but_distribution_lift_weak"
    elif not quality_ok:
        verdict = "conditionality_lift_quality_regression"
    else:
        verdict = "conditionality_lift_inconclusive"

    return {
        "schema_version": "nl_episode_narrative_conditionality_lift_v1",
        "status": "ok",
        "primary_question": (
            "For the same accepted starts, does episode-level narrative retrieval "
            "change support and generated scenario distributions beyond a "
            "start-only terminal-state support baseline?"
        ),
        "window_count": len(rows),
        "verdict": verdict,
        "aggregate": aggregate,
        "quality_guardrail": quality_guardrail,
        "rows": rows,
        "interpretation": {
            "primary_metric": (
                "distributional lift against start-only baseline; CRPS and Energy "
                "are guardrails, not the main differentiator for this branch"
            ),
            "support_jaccard": (
                "Lower overlap means the narrative condition changed which "
                "historical supports were used beyond the accepted start."
            ),
            "terminal_factor_ks": (
                "Higher values mean the generated terminal factor distributions "
                "differ between narrative-conditioned and start-only supports."
            ),
            "path_energy_distance_z": (
                "Higher values mean the whole generated path distribution differs "
                "in standardized multivariate path space."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    quality = report["quality_guardrail"]
    return "\n".join(
        [
            "# Episode Narrative Conditionality Lift",
            "",
            f"Verdict: `{report['verdict']}`.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Windows | {report['window_count']} |",
            f"| Mean support Jaccard | {aggregate['mean_support_jaccard']:.3f} |",
            f"| Mean support Jaccard distance | {aggregate['mean_support_jaccard_distance']:.3f} |",
            f"| Mean terminal factor KS | {aggregate['mean_terminal_factor_ks']:.3f} |",
            f"| Mean path energy distance | {aggregate['mean_path_energy_distance_z']:.3f} |",
            f"| Mean abs terminal mean shift | {aggregate['mean_abs_terminal_mean_shift_z']:.3f} |",
            f"| CRPS delta vs start-only | {quality['crps_delta_vs_start_only']:.3f} |",
            f"| Energy delta vs start-only | {quality['energy_delta_vs_start_only']:.3f} |",
            "",
            "This audit treats fixed-start narrative lift as the main branch target. "
            "Historical CRPS/Energy are guardrails.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--narrative-report", type=Path, default=DEFAULT_NARRATIVE_REPORT
    )
    parser.add_argument(
        "--narrative-arrays", type=Path, default=DEFAULT_NARRATIVE_ARRAYS
    )
    parser.add_argument("--start-only-report", type=Path, required=True)
    parser.add_argument("--start-only-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = build_conditionality_lift_report(
        narrative_report=_load_json(args.narrative_report),
        start_only_report=_load_json(args.start_only_report),
        narrative_arrays=_load_npz(args.narrative_arrays),
        start_only_arrays=_load_npz(args.start_only_arrays),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "conditionality_lift_report.json"
    md_path = args.output_dir / "conditionality_lift_report.md"
    report["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    _write_json(json_path, report)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "verdict": report["verdict"],
                "window_count": report["window_count"],
                "json": str(json_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
