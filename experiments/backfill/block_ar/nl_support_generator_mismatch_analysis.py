#!/usr/bin/env python
"""Analyze replay-vs-generator mismatch for narrative support policies.

This is a post-experiment diagnostic. It does not train a model and makes no
OpenAI calls. The purpose is to separate two questions that looked similar in
earlier support-reranker tests:

1. Does a candidate support policy choose histories that replay the realized
   future more closely?
2. Does the frozen SNI generator behave better when conditioned on those
   selected support histories?

The second question is the production one. A support policy that improves
historical replay but worsens generator self-calibration is not a promotable
conditioning bridge.
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


DEFAULT_BASELINE_BRIDGE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_CANDIDATE_BRIDGE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_learned_support_reranker_879a_testflight/"
    "learned_support_reranked_bridge_report.json"
)
DEFAULT_ORACLE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_oracle_fullheldout_786c/prefix_latent_oracle_arrays.npz"
)
DEFAULT_GENERATOR_CALIBRATION_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_generator_self_calibration_873b_train128/"
    "scenario_level_eval_report.json"
)
DEFAULT_ACTUAL_DECOMPOSITION_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_learned_support_reranker_879d_support_quality_decomposition/"
    "support_quality_decomposition.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_support_generator_mismatch_analysis_880a"
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


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _finite_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _as_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def replay_loss_z(
    *,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    query_window_index: int,
    support_window_index: int,
) -> float:
    scale = np.maximum(np.asarray(delta_scale, dtype=np.float32), 1e-8)
    future = np.asarray(future_delta, dtype=np.float32)
    query_future = future[int(query_window_index)] / scale
    support_future = future[int(support_window_index)] / scale
    return float(np.mean(np.abs(query_future - support_future)))


def _standardized_start_states(
    history_level: np.ndarray,
    fit_indices: np.ndarray | None = None,
) -> np.ndarray:
    history = np.asarray(history_level, dtype=np.float32)
    if history.ndim != 3:
        raise ValueError("history_level must have shape [N,T,C]")
    starts = history[:, -1, :]
    if fit_indices is None:
        fit = np.arange(starts.shape[0], dtype=np.int64)
    else:
        fit = np.asarray(fit_indices, dtype=np.int64)
    mean = starts[fit].mean(axis=0, keepdims=True)
    std = np.maximum(starts[fit].std(axis=0, keepdims=True), 1e-6)
    return ((starts - mean) / std).astype(np.float32)


def extract_generator_self_quality(
    calibration_scenario_report: dict[str, Any],
    *,
    method: str = "narrative_generator_topk",
) -> dict[int, dict[str, float | None]]:
    """Return frozen-generator self-calibration metrics by support index."""

    quality: dict[int, dict[str, float | None]] = {}
    for row in calibration_scenario_report.get("window_scores", []):
        if not isinstance(row, dict) or row.get("window_index") is None:
            continue
        method_row = row.get("methods", {}).get(method, {})
        if not isinstance(method_row, dict):
            continue
        quality[int(row["window_index"])] = {
            "energy_score_z": _as_number(method_row.get("energy_score_z")),
            "ensemble_crps_z": _as_number(method_row.get("ensemble_crps_z")),
            "coverage_80": _as_number(method_row.get("coverage_80")),
        }
    if not quality:
        raise ValueError("no generator self-calibration metrics found")
    return quality


def _row_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(row.get("embedding_index", row.get("window_index", -1))),
        int(row.get("window_index", -1)),
        str(row.get("kind", "")),
        str(row.get("role", "")),
    )


def _rows_by_key(
    report: dict[str, Any],
) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    rows = report.get("evaluation", {}).get("heldout_examples", [])
    output: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("window_index") is not None:
            output[_row_key(row)] = row
    return output


def _support_indices(row: dict[str, Any], *, top_k: int) -> list[int]:
    pool = row.get("top_train_pool", [])
    if not isinstance(pool, list):
        return []
    indices: list[int] = []
    for item in pool[: int(top_k)]:
        if isinstance(item, dict) and item.get("window_index") is not None:
            indices.append(int(item["window_index"]))
    return indices


def _mean_support_cosine(row: dict[str, Any], *, top_k: int) -> float | None:
    values: list[float] = []
    pool = row.get("top_train_pool", [])
    if isinstance(pool, list):
        for item in pool[: int(top_k)]:
            if isinstance(item, dict):
                number = _as_number(item.get("cosine"))
                if number is not None:
                    values.append(number)
    return _finite_mean(values)


def _support_stats(
    *,
    row: dict[str, Any],
    history_level: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    start_z: np.ndarray,
    generator_quality: dict[int, dict[str, float | None]],
    top_k: int,
) -> dict[str, Any]:
    query_idx = int(row["window_index"])
    support = _support_indices(row, top_k=top_k)
    replay_losses: list[float] = []
    start_distances: list[float] = []
    energy: list[float] = []
    crps: list[float] = []
    coverage: list[float] = []
    for support_idx in support:
        replay_losses.append(
            replay_loss_z(
                future_delta=future_delta,
                delta_scale=delta_scale,
                query_window_index=query_idx,
                support_window_index=support_idx,
            )
        )
        start_distances.append(
            float(np.linalg.norm(start_z[int(support_idx)] - start_z[query_idx]))
        )
        quality = generator_quality.get(int(support_idx), {})
        for target, dest in (
            ("energy_score_z", energy),
            ("ensemble_crps_z", crps),
            ("coverage_80", coverage),
        ):
            number = _as_number(quality.get(target))
            if number is not None:
                dest.append(number)
    return {
        "support_indices": support,
        "mean_replay_loss_z": _finite_mean(replay_losses),
        "mean_start_distance_z": _finite_mean(start_distances),
        "mean_memory_cosine": _mean_support_cosine(row, top_k=top_k),
        "mean_generator_energy_z": _finite_mean(energy),
        "mean_generator_crps_z": _finite_mean(crps),
        "mean_generator_coverage_80": _finite_mean(coverage),
    }


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def compare_support_policies(
    *,
    baseline_bridge: dict[str, Any],
    candidate_bridge: dict[str, Any],
    history_level: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    generator_quality: dict[int, dict[str, float | None]],
    top_k: int = 3,
    fit_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare support policies on replay, start, and generator calibration."""

    baseline_rows = _rows_by_key(baseline_bridge)
    candidate_rows = _rows_by_key(candidate_bridge)
    common_keys = sorted(set(baseline_rows) & set(candidate_rows))
    if not common_keys:
        raise ValueError("no common heldout rows between baseline and candidate")
    start_z = _standardized_start_states(history_level, fit_indices)

    rows: list[dict[str, Any]] = []
    for key in common_keys:
        base_row = baseline_rows[key]
        cand_row = candidate_rows[key]
        base_stats = _support_stats(
            row=base_row,
            history_level=history_level,
            future_delta=future_delta,
            delta_scale=delta_scale,
            start_z=start_z,
            generator_quality=generator_quality,
            top_k=top_k,
        )
        cand_stats = _support_stats(
            row=cand_row,
            history_level=history_level,
            future_delta=future_delta,
            delta_scale=delta_scale,
            start_z=start_z,
            generator_quality=generator_quality,
            top_k=top_k,
        )
        base_support = set(base_stats["support_indices"])
        cand_support = set(cand_stats["support_indices"])
        denom = max(1, min(len(base_support), len(cand_support), int(top_k)))
        overlap = len(base_support & cand_support) / float(denom)
        rows.append(
            {
                "embedding_index": key[0],
                "window_index": key[1],
                "kind": key[2],
                "role": key[3],
                "window_id": str(base_row.get("window_id", "")),
                "baseline": base_stats,
                "candidate": cand_stats,
                "deltas": {
                    "replay_loss": _delta(
                        cand_stats["mean_replay_loss_z"],
                        base_stats["mean_replay_loss_z"],
                    ),
                    "start_distance": _delta(
                        cand_stats["mean_start_distance_z"],
                        base_stats["mean_start_distance_z"],
                    ),
                    "memory_cosine": _delta(
                        cand_stats["mean_memory_cosine"],
                        base_stats["mean_memory_cosine"],
                    ),
                    "generator_energy": _delta(
                        cand_stats["mean_generator_energy_z"],
                        base_stats["mean_generator_energy_z"],
                    ),
                    "generator_crps": _delta(
                        cand_stats["mean_generator_crps_z"],
                        base_stats["mean_generator_crps_z"],
                    ),
                    "generator_coverage": _delta(
                        cand_stats["mean_generator_coverage_80"],
                        base_stats["mean_generator_coverage_80"],
                    ),
                    "topk_overlap_fraction": float(overlap),
                },
            }
        )

    def mean_delta(name: str) -> float | None:
        return _finite_mean(
            [row["deltas"][name] for row in rows if row["deltas"].get(name) is not None]
        )

    replay_delta = mean_delta("replay_loss")
    energy_delta = mean_delta("generator_energy")
    crps_delta = mean_delta("generator_crps")
    coverage_delta = mean_delta("generator_coverage")
    summary = {
        "window_count": len(rows),
        "top_k": int(top_k),
        "mean_topk_overlap_fraction": mean_delta("topk_overlap_fraction"),
        "replay_loss_delta_mean": replay_delta,
        "start_distance_delta_mean": mean_delta("start_distance"),
        "memory_cosine_delta_mean": mean_delta("memory_cosine"),
        "generator_energy_delta_mean": energy_delta,
        "generator_crps_delta_mean": crps_delta,
        "generator_coverage_delta_mean": coverage_delta,
        "replay_better_count": sum(
            1
            for row in rows
            if (row["deltas"].get("replay_loss") is not None)
            and row["deltas"]["replay_loss"] < 0.0
        ),
        "generator_energy_better_count": sum(
            1
            for row in rows
            if (row["deltas"].get("generator_energy") is not None)
            and row["deltas"]["generator_energy"] < 0.0
        ),
        "replay_better_generator_worse_count": sum(
            1
            for row in rows
            if (row["deltas"].get("replay_loss") is not None)
            and row["deltas"]["replay_loss"] < 0.0
            and (row["deltas"].get("generator_energy") is not None)
            and row["deltas"]["generator_energy"] > 0.0
        ),
    }
    mechanism = "mixed_or_inconclusive"
    if (
        replay_delta is not None
        and replay_delta < 0.0
        and (
            (energy_delta is not None and energy_delta > 0.0)
            or (crps_delta is not None and crps_delta > 0.0)
            or (coverage_delta is not None and coverage_delta < 0.0)
        )
    ):
        mechanism = "replay_better_generator_worse"
    elif (
        energy_delta is not None
        and energy_delta < 0.0
        and crps_delta is not None
        and crps_delta < 0.0
    ):
        mechanism = "generator_aligned_candidate"

    return {
        "status": "ok",
        "scope_note": (
            "Post-experiment support-policy analysis. Historical replay "
            "closeness is compared with frozen-generator self-calibration; "
            "lower replay/energy/CRPS deltas are better, higher coverage "
            "delta is better."
        ),
        "summary": summary,
        "decision": {
            "mechanism": mechanism,
            "promote_candidate": False,
            "next_step": (
                "Use replay-only support reranking as a diagnostic, not as a "
                "promotion path. The next candidate should train/rank against "
                "generator-calibrated response labels or a stability-aware "
                "rollout proxy while preserving the support mixture."
            ),
        },
        "rows": rows,
    }


def attach_actual_rollout_decomposition(
    report: dict[str, Any],
    decomposition_report: dict[str, Any],
) -> dict[str, Any]:
    """Attach actual rollout deltas and update the mechanism label."""

    output = json.loads(json.dumps(report))
    deltas = decomposition_report.get("summary", {}).get(
        "summary_improvement_deltas", {}
    )
    actual = {
        "historical_replay_topk_crps": _as_number(
            deltas.get("historical_replay_topk_crps")
        ),
        "historical_replay_topk_energy": _as_number(
            deltas.get("historical_replay_topk_energy")
        ),
        "narrative_generator_topk_crps": _as_number(
            deltas.get("narrative_generator_topk_crps")
        ),
        "narrative_generator_topk_energy": _as_number(
            deltas.get("narrative_generator_topk_energy")
        ),
    }
    generator_regressed = (
        actual["narrative_generator_topk_crps"] is not None
        and actual["narrative_generator_topk_crps"] < 0.0
    ) or (
        actual["narrative_generator_topk_energy"] is not None
        and actual["narrative_generator_topk_energy"] < 0.0
    )
    replay_improved = (
        actual["historical_replay_topk_crps"] is not None
        and actual["historical_replay_topk_crps"] > 0.0
    ) or (
        actual["historical_replay_topk_energy"] is not None
        and actual["historical_replay_topk_energy"] > 0.0
    )
    actual["candidate_generator_regressed"] = bool(generator_regressed)
    actual["candidate_replay_improved"] = bool(replay_improved)
    output["actual_rollout_comparison"] = actual

    proxy_mechanism = str(output.get("decision", {}).get("mechanism", ""))
    if generator_regressed and proxy_mechanism == "generator_aligned_candidate":
        output["decision"]["mechanism"] = "generator_proxy_false_positive"
        output["decision"]["next_step"] = (
            "The learned reranker looked better under replay loss and train "
            "self-calibration proxy, but the same-seed frozen rollout "
            "regressed. Do not add another replay or self-calibration ranking "
            "knob; the next candidate needs a rollout-response label, paired "
            "seed stability gate, or learned weighting objective trained "
            "against generator-level response."
        )
    elif generator_regressed and replay_improved:
        output["decision"]["mechanism"] = "replay_better_generator_worse"
    return output


def _format_delta(value: float | None, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.{precision}f}"


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# NL Support Replay vs Generator Mismatch",
        "",
        "This diagnostic compares the current support pool with the learned "
        "replay-reranked pool. Lower replay loss, generator energy, and "
        "generator CRPS deltas are better; higher coverage delta is better.",
        "",
        "## Summary",
        "",
        f"- Rows compared: `{summary['window_count']}`",
        f"- Top-k: `{summary['top_k']}`",
        f"- Mean top-k overlap: `{summary['mean_topk_overlap_fraction']:.4f}`",
        f"- Replay loss delta: `{_format_delta(summary['replay_loss_delta_mean'])}`",
        f"- Generator energy delta: `{_format_delta(summary['generator_energy_delta_mean'])}`",
        f"- Generator CRPS delta: `{_format_delta(summary['generator_crps_delta_mean'])}`",
        f"- Generator coverage delta: `{_format_delta(summary['generator_coverage_delta_mean'])}`",
        f"- Start-distance delta: `{_format_delta(summary['start_distance_delta_mean'])}`",
        f"- Memory-cosine delta: `{_format_delta(summary['memory_cosine_delta_mean'])}`",
        f"- Replay better but generator worse count: "
        f"`{summary['replay_better_generator_worse_count']}`",
        "",
    ]
    actual = report.get("actual_rollout_comparison")
    if isinstance(actual, dict):
        lines.extend(
            [
                "## Actual Rollout Smoke",
                "",
                "These deltas come from the same-seed frozen-generator smoke. "
                "They are positive-is-better, unlike the raw proxy deltas above.",
                "",
                f"- Replay CRPS improvement delta: "
                f"`{_format_delta(actual.get('historical_replay_topk_crps'))}`",
                f"- Replay energy improvement delta: "
                f"`{_format_delta(actual.get('historical_replay_topk_energy'))}`",
                f"- Generator CRPS improvement delta: "
                f"`{_format_delta(actual.get('narrative_generator_topk_crps'))}`",
                f"- Generator energy improvement delta: "
                f"`{_format_delta(actual.get('narrative_generator_topk_energy'))}`",
                f"- Candidate generator regressed: "
                f"`{actual.get('candidate_generator_regressed')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- Mechanism: `{report['decision']['mechanism']}`",
            f"- Promote candidate: `{report['decision']['promote_candidate']}`",
            f"- Next step: {report['decision']['next_step']}",
            "",
        ]
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _load_json(args.baseline_bridge_report)
    candidate = _load_json(args.candidate_bridge_report)
    calibration = _load_json(args.generator_calibration_report)
    arrays = _load_npz(args.oracle_arrays)
    generator_quality = extract_generator_self_quality(calibration)
    report = compare_support_policies(
        baseline_bridge=baseline,
        candidate_bridge=candidate,
        history_level=arrays["history_level"],
        future_delta=arrays["future_delta"],
        delta_scale=arrays["delta_scale"],
        generator_quality=generator_quality,
        top_k=int(args.top_k),
        fit_indices=arrays.get("train_indices"),
    )
    if args.actual_decomposition_report:
        actual_path = Path(args.actual_decomposition_report)
        if actual_path.exists():
            report = attach_actual_rollout_decomposition(
                report, _load_json(actual_path)
            )
    report["artifact_paths"] = {
        "baseline_bridge_report": str(args.baseline_bridge_report),
        "candidate_bridge_report": str(args.candidate_bridge_report),
        "oracle_arrays": str(args.oracle_arrays),
        "generator_calibration_report": str(args.generator_calibration_report),
        "actual_decomposition_report": str(args.actual_decomposition_report),
        "report": str(
            Path(args.output_dir) / "support_generator_mismatch_analysis.json"
        ),
        "markdown": str(
            Path(args.output_dir) / "support_generator_mismatch_analysis.md"
        ),
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "support_generator_mismatch_analysis.json", report)
    write_markdown_report(
        output_dir / "support_generator_mismatch_analysis.md",
        report,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bridge-report", default=DEFAULT_BASELINE_BRIDGE)
    parser.add_argument("--candidate-bridge-report", default=DEFAULT_CANDIDATE_BRIDGE)
    parser.add_argument("--oracle-arrays", default=DEFAULT_ORACLE_ARRAYS)
    parser.add_argument(
        "--generator-calibration-report",
        default=DEFAULT_GENERATOR_CALIBRATION_REPORT,
    )
    parser.add_argument(
        "--actual-decomposition-report",
        default=DEFAULT_ACTUAL_DECOMPOSITION_REPORT,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    report = run_analysis(parse_args(argv))
    summary = report["summary"]
    print(
        "wrote "
        f"{report['artifact_paths']['report']} "
        f"mechanism={report['decision']['mechanism']} "
        f"replay_delta={_format_delta(summary['replay_loss_delta_mean'])} "
        f"generator_energy_delta={_format_delta(summary['generator_energy_delta_mean'])}"
    )


if __name__ == "__main__":
    main()
