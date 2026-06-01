#!/usr/bin/env python
"""Audit where narrative conditionality is preserved or attenuated.

This is an artifact-only diagnostic. It requires a prior scenario-level gain
gate to pass, then compares the same fixed-start narrative runs at three
successive layers:

1. support distribution over historical prefixes;
2. decoded text/start prefix memory;
3. frozen SNI rollout path distribution.

The goal is not to promote a new model. The goal is to locate the bottleneck
when support pools differ but final scenario fans are still too similar or too
noisy for a clean risk-manager-visible conditionality claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_shape_audit import (  # noqa: E402
    _load_repeat_cases,
    _load_start_only_cases,
    load_observed_cases,
    pair_distribution_metrics,
    path_energy_distance,
)
from experiments.backfill.block_ar.nl_prefix_latent_conditionality_strength_benchmark import (  # noqa: E402
    DEFAULT_COMPONENT_ROOT,
    DEFAULT_CONTROL_ROOT,
    DEFAULT_VARIANT_DIR,
)


DEFAULT_GAIN_SUMMARY = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "caption_conditionality_refresh_917f_balanced80/"
    "caption_conditionality_refresh_summary.json"
)
DEFAULT_BENCHMARK_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_strength_benchmark_917f_balanced80_caption_quality/"
    "conditionality_strength_benchmark.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_transmission_audit_918a_balanced80"
)
DEFAULT_PAPER_FIGURE_DIR = Path("paper/narrative_grounded_scenarios/figures")


LAYER_METRICS = [
    ("support_tv_distance", "Support TV"),
    ("decoded_prefix_norm_rmse", "Decoded prefix"),
    ("rollout_path_energy_distance", "Rollout energy"),
    ("rollout_path_wasserstein_z", "Rollout Wasserstein"),
]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _scenario_gain_gate(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    gate = report.get("scenario_gain_gate", {})
    if not isinstance(gate, dict):
        raise ValueError(f"{path}: missing scenario_gain_gate")
    return gate


def scenario_gain_persists(gate: dict[str, Any]) -> bool:
    checks = gate.get("checks", {})
    if not isinstance(checks, dict) or not checks:
        return False
    return bool(gate.get("passes_all_models")) and all(
        bool(item.get("passes")) for item in checks.values() if isinstance(item, dict)
    )


def _array_path_for_case(case: dict[str, Any]) -> Path:
    return Path(str(case["run_report"])).with_name("prefix_latent_story_smoke_arrays.npz")


def _case_base_name(case_name: str) -> str:
    return str(case_name).split("#seed_")[0].split("#")[0]


def _case_label(case: dict[str, Any]) -> str:
    return str(case.get("label") or _case_base_name(str(case.get("case_name", ""))))


def _feature_key(case: dict[str, Any]) -> str:
    return str(Path(str(case["run_report"])).resolve())


def _short_label(label: str) -> str:
    return re.sub(r"\s+", "\n", str(label).replace("start18", "").strip())


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def _support_distribution(indices: np.ndarray, weights: np.ndarray) -> dict[int, float]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    count = min(idx.size, w.size)
    out: dict[int, float] = {}
    for i, weight in zip(idx[:count], w[:count], strict=False):
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        out[int(i)] = out.get(int(i), 0.0) + float(weight)
    total = sum(out.values())
    if total <= 1e-12:
        return {}
    return {key: float(value / total) for key, value in out.items()}


def support_metrics(left: dict[int, float], right: dict[int, float]) -> dict[str, float | int]:
    left_keys = set(left)
    right_keys = set(right)
    union = left_keys | right_keys
    shared = left_keys & right_keys
    overlap = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in union)
    tv = 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in union)
    return {
        "support_shared_count": int(len(shared)),
        "support_union_count": int(len(union)),
        "support_jaccard": float(len(shared) / len(union)) if union else 0.0,
        "support_jaccard_distance": 1.0 - (float(len(shared) / len(union)) if union else 0.0),
        "support_weighted_overlap": float(overlap),
        "support_tv_distance": float(tv),
    }


def load_case_features(case: dict[str, Any]) -> dict[str, Any]:
    array_path = _array_path_for_case(case)
    with np.load(array_path) as arrays:
        support = _support_distribution(
            arrays["rollout_component_window_index"],
            arrays["rollout_component_weight"],
        )
        return {
            "case_name": str(case["case_name"]),
            "label": _case_label(case),
            "array_path": str(array_path),
            "support_distribution": support,
            "text_memory": np.asarray(arrays["text_memory"][0], dtype=np.float64),
            "decoded_memory": np.asarray(arrays["decoded_memory"][0], dtype=np.float64),
            "decoded_history_norm": np.asarray(
                arrays["decoded_history_norm"][0],
                dtype=np.float64,
            ),
        }


def decoded_prefix_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_prefix = np.asarray(left["decoded_history_norm"], dtype=np.float64)
    right_prefix = np.asarray(right["decoded_history_norm"], dtype=np.float64)
    diff = left_prefix - right_prefix
    terminal = left_prefix[-1] - right_prefix[-1]
    memory_diff = np.asarray(left["decoded_memory"]) - np.asarray(right["decoded_memory"])
    return {
        "text_memory_cosine_distance": _cosine_distance(
            np.asarray(left["text_memory"]),
            np.asarray(right["text_memory"]),
        ),
        "decoded_memory_cosine_distance": _cosine_distance(
            np.asarray(left["decoded_memory"]),
            np.asarray(right["decoded_memory"]),
        ),
        "decoded_memory_l2_per_dim": float(
            np.linalg.norm(memory_diff) / math.sqrt(max(memory_diff.size, 1))
        ),
        "decoded_prefix_norm_rmse": float(np.sqrt(np.mean(diff**2))),
        "decoded_prefix_terminal_norm_l2_per_dim": float(
            np.linalg.norm(terminal) / math.sqrt(max(terminal.size, 1))
        ),
    }


def rollout_metrics(left_case: dict[str, Any], right_case: dict[str, Any]) -> dict[str, float]:
    pair = pair_distribution_metrics(left_case, right_case)
    aggregate = pair["aggregate"]
    return {
        "rollout_path_energy_distance": float(path_energy_distance(left_case, right_case)),
        "rollout_path_wasserstein_z": float(
            aggregate.get("path_wasserstein_z_mean_median", 0.0)
        ),
        "rollout_path_variance_log_gap": float(
            aggregate.get("path_std_log_ratio_mean_median", 0.0)
        ),
        "rollout_path_shape_ks": float(aggregate.get("path_raw_ks_mean_median", 0.0)),
        "rollout_drawdown_prob_gap": float(
            aggregate.get("path_drawdown_prob_gap_1sigma_mean_median", 0.0)
        ),
        "rollout_rally_prob_gap": float(
            aggregate.get("path_rally_prob_gap_1sigma_mean_median", 0.0)
        ),
    }


def build_pair_row(
    left_case: dict[str, Any],
    right_case: dict[str, Any],
    *,
    left_features: dict[str, Any] | None,
    right_features: dict[str, Any] | None,
    control: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "control": control,
        "left_case": str(left_case["case_name"]),
        "right_case": str(right_case["case_name"]),
        "left_label": _case_label(left_case),
        "right_label": _case_label(right_case),
    }
    if left_features is not None and right_features is not None:
        row.update(
            support_metrics(
                left_features["support_distribution"],
                right_features["support_distribution"],
            )
        )
        row.update(decoded_prefix_metrics(left_features, right_features))
    else:
        for key in [
            "support_shared_count",
            "support_union_count",
            "support_jaccard",
            "support_jaccard_distance",
            "support_weighted_overlap",
            "support_tv_distance",
            "text_memory_cosine_distance",
            "decoded_memory_cosine_distance",
            "decoded_memory_l2_per_dim",
            "decoded_prefix_norm_rmse",
            "decoded_prefix_terminal_norm_l2_per_dim",
        ]:
            row[key] = float("nan")
    row.update(rollout_metrics(left_case, right_case))
    return row


def _pairwise_rows(
    cases: list[dict[str, Any]],
    features_by_case: dict[str, dict[str, Any]],
    *,
    control: str,
) -> list[dict[str, Any]]:
    return [
        build_pair_row(
            left,
            right,
            left_features=features_by_case[_feature_key(left)],
            right_features=features_by_case[_feature_key(right)],
            control=control,
        )
        for left, right in combinations(cases, 2)
    ]


def _repeat_rows(
    repeat_cases: list[dict[str, Any]],
    features_by_case: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in repeat_cases:
        grouped.setdefault(_case_base_name(str(case["case_name"])), []).append(case)
    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        rows.extend(_pairwise_rows(group, features_by_case, control="same_narrative_repeat"))
    return rows


def _bootstrap_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        states = np.asarray(case["states"], dtype=np.float32)
        split = states.shape[0] // 2
        if split < 2:
            continue
        left = {
            **case,
            "case_name": f"{case['case_name']}#bootstrap_a",
            "states": states[:split],
        }
        right = {
            **case,
            "case_name": f"{case['case_name']}#bootstrap_b",
            "states": states[split : split * 2],
        }
        rows.append(
            build_pair_row(
                left,
                right,
                left_features=None,
                right_features=None,
                control="within_run_bootstrap",
            )
        )
    return rows


def _finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_rows(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"pair_count": int(len(rows))}
    for metric in metrics:
        values = _finite_values(rows, metric)
        summary[f"{metric}_count"] = int(len(values))
        if not values:
            summary[f"{metric}_median"] = None
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_p90"] = None
            continue
        summary[f"{metric}_median"] = float(median(values))
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_p90"] = float(np.quantile(values, 0.90))
    return summary


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(float(den)) <= 1e-12:
        return None
    return float(num) / float(den)


def build_transmission_decision(
    summaries: dict[str, dict[str, Any]],
    benchmark_report: dict[str, Any] | None,
) -> dict[str, Any]:
    obs = summaries["observed_cross_narrative"]
    repeat = summaries["same_narrative_repeat"]
    start_only = summaries["start_only_null"]
    bootstrap = summaries["within_run_bootstrap"]

    ratios = {
        "repeat_support_tv_to_observed": _ratio(
            repeat.get("support_tv_distance_median"),
            obs.get("support_tv_distance_median"),
        ),
        "start_only_support_tv_to_observed": _ratio(
            start_only.get("support_tv_distance_median"),
            obs.get("support_tv_distance_median"),
        ),
        "repeat_decoded_prefix_to_observed": _ratio(
            repeat.get("decoded_prefix_norm_rmse_median"),
            obs.get("decoded_prefix_norm_rmse_median"),
        ),
        "start_only_decoded_prefix_to_observed": _ratio(
            start_only.get("decoded_prefix_norm_rmse_median"),
            obs.get("decoded_prefix_norm_rmse_median"),
        ),
        "repeat_rollout_energy_to_observed": _ratio(
            repeat.get("rollout_path_energy_distance_median"),
            obs.get("rollout_path_energy_distance_median"),
        ),
        "bootstrap_rollout_energy_to_observed": _ratio(
            bootstrap.get("rollout_path_energy_distance_median"),
            obs.get("rollout_path_energy_distance_median"),
        ),
        "start_only_rollout_energy_to_observed": _ratio(
            start_only.get("rollout_path_energy_distance_median"),
            obs.get("rollout_path_energy_distance_median"),
        ),
        "bootstrap_rollout_wasserstein_to_observed": _ratio(
            bootstrap.get("rollout_path_wasserstein_z_median"),
            obs.get("rollout_path_wasserstein_z_median"),
        ),
    }
    key_ratios = {}
    if benchmark_report:
        decision = benchmark_report.get("decision", {})
        key_ratios = decision.get("key_ratios", {}) if isinstance(decision, dict) else {}

    bottlenecks: list[str] = []
    warnings: list[str] = []

    observed_support = float(obs.get("support_tv_distance_median") or 0.0)
    observed_prefix = float(obs.get("decoded_prefix_norm_rmse_median") or 0.0)
    repeat_prefix_ratio = ratios["repeat_decoded_prefix_to_observed"]
    repeat_rollout_ratio = ratios["repeat_rollout_energy_to_observed"]
    bootstrap_rollout_ratio = ratios["bootstrap_rollout_energy_to_observed"]

    if observed_support < 0.50:
        bottlenecks.append("support_selection_too_similar")
    if observed_prefix <= 1e-6:
        bottlenecks.append("decoded_prefix_not_changing")
    elif repeat_prefix_ratio is not None and repeat_prefix_ratio > 0.25:
        bottlenecks.append("decoded_prefix_repeat_noise")
    if repeat_rollout_ratio is not None and repeat_rollout_ratio > 0.75:
        warnings.append("same_narrative_rollout_noise_close_to_observed")
    if bootstrap_rollout_ratio is not None and bootstrap_rollout_ratio > 0.75:
        bottlenecks.append("within_run_rollout_bootstrap_noise_close_to_observed")
    portfolio_var_ratio = key_ratios.get("portfolio_var95_vs_repeat")
    if portfolio_var_ratio is not None and float(portfolio_var_ratio) < 1.25:
        bottlenecks.append("portfolio_tail_separation_below_repeat_control")

    if not bottlenecks:
        verdict = "transmission_supported"
    elif (
        "support_selection_too_similar" not in bottlenecks
        and "decoded_prefix_not_changing" not in bottlenecks
        and "decoded_prefix_repeat_noise" not in bottlenecks
    ):
        verdict = "support_and_prefix_preserved_rollout_tail_bottleneck"
    else:
        verdict = "conditionality_transmission_bottleneck_before_rollout"

    return {
        "verdict": verdict,
        "bottlenecks": bottlenecks,
        "warnings": warnings,
        "ratios": ratios,
        "benchmark_key_ratios": key_ratios,
        "interpretation": _decision_interpretation(verdict, bottlenecks),
    }


def _decision_interpretation(verdict: str, bottlenecks: list[str]) -> str:
    if verdict == "transmission_supported":
        return (
            "Narrative signal is visible in support selection, decoded prefix, "
            "and rollout distributions above controls for this audit."
        )
    if verdict == "support_and_prefix_preserved_rollout_tail_bottleneck":
        return (
            "Narrative signal enters the system through different support pools "
            "and decoded prefixes. The remaining weakness is downstream: frozen "
            "rollout sampling/readout noise and portfolio-tail separation are too "
            "close to controls for a clean conditionality claim."
        )
    if "support_selection_too_similar" in bottlenecks:
        return (
            "The support layer is not changing enough across narratives. Improve "
            "support scoring or direction-aware retrieval before changing the "
            "generator/readout."
        )
    return (
        "The support layer changes, but the decoded prefix/memory does not "
        "preserve enough of that difference. Improve prefix construction or "
        "component-preserving readout before tuning rollout calibration."
    )


def _observed_pair_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"{row['left_label']} vs {row['right_label']}"
        for row in rows
        if row.get("control") == "observed_cross_narrative"
    ]


def _plot_transmission_ladder(
    summaries: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> None:
    controls = [
        ("same_narrative_repeat", "Repeat", "#7B1FA2"),
        ("within_run_bootstrap", "Bootstrap", "#EF6C00"),
        ("start_only_null", "Start-only", "#78909C"),
    ]
    x = np.arange(len(LAYER_METRICS), dtype=np.float64)
    width = 0.23
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    observed = summaries["observed_cross_narrative"]
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.0, label="Observed narrative effect")
    for offset, (control, label, color) in enumerate(controls):
        values = []
        for metric, _metric_label in LAYER_METRICS:
            den = observed.get(f"{metric}_median")
            num = summaries.get(control, {}).get(f"{metric}_median")
            ratio = _ratio(num, den)
            values.append(np.nan if ratio is None else ratio)
        ax.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=label,
            color=color,
            alpha=0.85,
        )
    ax.axhspan(0.0, 0.75, color="#E8F5E9", alpha=0.45, zorder=-5)
    ax.set_xticks(x, [label for _metric, label in LAYER_METRICS])
    ax.set_ylabel("Control / observed narrative effect")
    ax.set_title("Conditionality transmission: where controls catch up")
    ax.set_ylim(0.0, 1.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pairwise_scatter(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    observed = [row for row in rows if row["control"] == "observed_cross_narrative"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    axes[0].scatter(
        [float(row["support_tv_distance"]) for row in observed],
        [float(row["decoded_prefix_norm_rmse"]) for row in observed],
        color="#1565C0",
        s=45,
    )
    axes[0].set_xlabel("Support TV distance")
    axes[0].set_ylabel("Decoded prefix RMSE")
    axes[0].set_title("Support -> prefix")
    axes[1].scatter(
        [float(row["decoded_prefix_norm_rmse"]) for row in observed],
        [float(row["rollout_path_energy_distance"]) for row in observed],
        color="#2E7D32",
        s=45,
    )
    axes[1].set_xlabel("Decoded prefix RMSE")
    axes[1].set_ylabel("Rollout path energy")
    axes[1].set_title("Prefix -> rollout")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    for idx, row in enumerate(observed):
        if idx >= 8:
            continue
        axes[1].annotate(
            str(idx + 1),
            (
                float(row["decoded_prefix_norm_rmse"]),
                float(row["rollout_path_energy_distance"]),
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    fig.suptitle("Observed cross-narrative pair transmission")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_observed_heatmaps(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    labels = [_short_label(_case_label(case)) for case in cases]
    names = [str(case["case_name"]) for case in cases]
    metrics = [
        ("support_tv_distance", "Support TV"),
        ("decoded_prefix_norm_rmse", "Decoded prefix"),
        ("rollout_path_energy_distance", "Rollout energy"),
    ]
    lookup = {}
    for row in rows:
        if row["control"] != "observed_cross_narrative":
            continue
        left = str(row["left_case"])
        right = str(row["right_case"])
        lookup[(left, right)] = row
        lookup[(right, left)] = row
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        matrix = np.zeros((len(names), len(names)), dtype=np.float64)
        for i, left in enumerate(names):
            for j, right in enumerate(names):
                if i == j:
                    matrix[i, j] = 0.0
                    continue
                matrix[i, j] = float(lookup[(left, right)][metric])
        image = ax.imshow(matrix, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle("Pairwise narrative differences by transmission layer")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _correlation(rows: list[dict[str, Any]], left_metric: str, right_metric: str) -> float | None:
    x = _finite_values(rows, left_metric)
    y = _finite_values(rows, right_metric)
    count = min(len(x), len(y))
    if count < 3:
        return None
    x_arr = np.asarray(x[:count], dtype=np.float64)
    y_arr = np.asarray(y[:count], dtype=np.float64)
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    summaries = report["summaries"]
    obs = summaries["observed_cross_narrative"]
    repeat = summaries["same_narrative_repeat"]
    bootstrap = summaries["within_run_bootstrap"]
    lines = [
        "# NL Conditionality Transmission Audit",
        "",
        f"Status: **{decision['verdict']}**",
        "",
        "## Scenario-Level Gate",
        "",
        f"- Gain gate passed: `{report['scenario_gain_gate']['passes_all_models']}`.",
    ]
    for name, check in report["scenario_gain_gate"].get("checks", {}).items():
        lines.append(
            f"- {name}: CRPS gain `{float(check['crps_gain']):+.3f}`, "
            f"energy gain `{float(check['energy_gain']):+.3f}`."
        )
    lines += [
        "",
        "## Layer Summary",
        "",
        (
            "- Observed support TV distance: "
            f"`{float(obs['support_tv_distance_median']):.3f}` "
            f"(repeat ratio `{_fmt(decision['ratios']['repeat_support_tv_to_observed'])}`)."
        ),
        (
            "- Observed decoded-prefix RMSE: "
            f"`{float(obs['decoded_prefix_norm_rmse_median']):.3f}` "
            f"(repeat ratio `{_fmt(decision['ratios']['repeat_decoded_prefix_to_observed'])}`)."
        ),
        (
            "- Observed rollout path energy: "
            f"`{float(obs['rollout_path_energy_distance_median']):.3f}`; "
            f"repeat ratio `{_fmt(decision['ratios']['repeat_rollout_energy_to_observed'])}`, "
            f"bootstrap ratio `{_fmt(decision['ratios']['bootstrap_rollout_energy_to_observed'])}`."
        ),
        (
            "- Repeat rollout median path energy: "
            f"`{float(repeat['rollout_path_energy_distance_median']):.3f}`; "
            f"bootstrap median path energy: "
            f"`{float(bootstrap['rollout_path_energy_distance_median']):.3f}`."
        ),
        "",
        "## Diagnosis",
        "",
        decision["interpretation"],
        "",
    ]
    if decision["bottlenecks"]:
        lines.append("Bottlenecks:")
        lines.extend(f"- `{item}`" for item in decision["bottlenecks"])
        lines.append("")
    if decision["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- `{item}`" for item in decision["warnings"])
        lines.append("")
    lines += [
        "## Artifacts",
        "",
    ]
    for key, value in report["artifact_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def build_transmission_audit(args: argparse.Namespace) -> dict[str, Any]:
    gain_gate = _scenario_gain_gate(Path(args.scenario_gain_summary))
    if bool(args.require_scenario_gain) and not scenario_gain_persists(gain_gate):
        raise RuntimeError(
            "scenario-level gain gate did not pass; refusing to refresh transmission audit"
        )
    benchmark_report = (
        _load_json(args.benchmark_report)
        if args.benchmark_report and Path(args.benchmark_report).exists()
        else None
    )
    observed_cases = load_observed_cases(
        Path(args.component_root),
        variant_dir=str(args.variant_dir),
        fan_scale=float(args.fan_scale),
    )
    repeat_cases = _load_repeat_cases(Path(args.control_root), fan_scale=float(args.fan_scale))
    start_only_cases = _load_start_only_cases(
        Path(args.control_root),
        fan_scale=float(args.fan_scale),
    )
    all_feature_cases = observed_cases + repeat_cases + start_only_cases
    features_by_case = {_feature_key(case): load_case_features(case) for case in all_feature_cases}
    observed_rows = _pairwise_rows(
        observed_cases,
        features_by_case,
        control="observed_cross_narrative",
    )
    repeat_rows = _repeat_rows(repeat_cases, features_by_case)
    start_only_rows = _pairwise_rows(
        start_only_cases,
        features_by_case,
        control="start_only_null",
    )
    bootstrap_rows = _bootstrap_rows(observed_cases)
    pair_rows = observed_rows + repeat_rows + start_only_rows + bootstrap_rows

    metrics = [
        "support_tv_distance",
        "support_jaccard_distance",
        "support_weighted_overlap",
        "text_memory_cosine_distance",
        "decoded_memory_cosine_distance",
        "decoded_memory_l2_per_dim",
        "decoded_prefix_norm_rmse",
        "decoded_prefix_terminal_norm_l2_per_dim",
        "rollout_path_energy_distance",
        "rollout_path_wasserstein_z",
        "rollout_path_variance_log_gap",
        "rollout_path_shape_ks",
        "rollout_drawdown_prob_gap",
        "rollout_rally_prob_gap",
    ]
    summaries = {
        control: summarize_rows(
            [row for row in pair_rows if row["control"] == control],
            metrics,
        )
        for control in [
            "observed_cross_narrative",
            "same_narrative_repeat",
            "within_run_bootstrap",
            "start_only_null",
        ]
    }
    decision = build_transmission_decision(summaries, benchmark_report)
    observed_only = [row for row in pair_rows if row["control"] == "observed_cross_narrative"]
    correlations = {
        "support_tv_to_decoded_prefix": _correlation(
            observed_only,
            "support_tv_distance",
            "decoded_prefix_norm_rmse",
        ),
        "decoded_prefix_to_rollout_energy": _correlation(
            observed_only,
            "decoded_prefix_norm_rmse",
            "rollout_path_energy_distance",
        ),
        "support_tv_to_rollout_energy": _correlation(
            observed_only,
            "support_tv_distance",
            "rollout_path_energy_distance",
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ladder_path = output_dir / "conditionality_transmission_ladder.png"
    scatter_path = output_dir / "conditionality_transmission_pair_scatter.png"
    heatmap_path = output_dir / "conditionality_transmission_pair_heatmaps.png"
    _plot_transmission_ladder(summaries, ladder_path)
    _plot_pairwise_scatter(pair_rows, scatter_path)
    _plot_observed_heatmaps(observed_cases, pair_rows, heatmap_path)

    paper_paths: dict[str, str] = {}
    if args.paper_figure_dir:
        paper_dir = Path(args.paper_figure_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)
        for key, src in [
            ("paper_transmission_ladder", ladder_path),
            ("paper_transmission_pair_scatter", scatter_path),
            ("paper_transmission_pair_heatmaps", heatmap_path),
        ]:
            dst = paper_dir / f"{src.stem}_914c_balanced80.png"
            shutil.copyfile(src, dst)
            paper_paths[key] = str(dst)

    report = {
        "scope_note": (
            "Artifact-only conditionality transmission audit. It runs only after "
            "the balanced-80 scenario-level caption gain gate passes, then checks "
            "where narrative signal is preserved or attenuated: support mixture, "
            "decoded prefix, and frozen SNI rollout distribution."
        ),
        "component_root": str(args.component_root),
        "control_root": str(args.control_root),
        "variant_dir": str(args.variant_dir),
        "fan_scale": float(args.fan_scale),
        "scenario_gain_gate": gain_gate,
        "decision": decision,
        "summaries": summaries,
        "correlations": correlations,
        "observed_pair_labels": _observed_pair_labels(observed_rows),
        "pair_rows": pair_rows,
        "artifact_paths": {
            "report_json": str(output_dir / "conditionality_transmission_audit.json"),
            "report_markdown": str(output_dir / "conditionality_transmission_audit.md"),
            "transmission_ladder": str(ladder_path),
            "pair_scatter": str(scatter_path),
            "pair_heatmaps": str(heatmap_path),
            **paper_paths,
        },
    }
    _write_json(report["artifact_paths"]["report_json"], report)
    _write_text(report["artifact_paths"]["report_markdown"], _markdown_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--scenario-gain-summary", type=Path, default=DEFAULT_GAIN_SUMMARY)
    parser.add_argument("--benchmark-report", type=Path, default=DEFAULT_BENCHMARK_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-figure-dir", type=Path, default=DEFAULT_PAPER_FIGURE_DIR)
    parser.add_argument("--fan-scale", type=float, default=1.0)
    parser.add_argument("--require-scenario-gain", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = build_transmission_audit(args)
    print(
        json.dumps(
            _jsonable(
                {
                    "verdict": report["decision"]["verdict"],
                    "bottlenecks": report["decision"]["bottlenecks"],
                    "warnings": report["decision"]["warnings"],
                    "ratios": report["decision"]["ratios"],
                    "correlations": report["correlations"],
                    "report": report["artifact_paths"]["report_json"],
                    "markdown": report["artifact_paths"]["report_markdown"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
