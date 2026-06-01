#!/usr/bin/env python
"""Refresh caption-side conditionality plots from reverse A/B rollout artifacts.

This script is deliberately artifact-only: it does not call OpenAI, regenerate
captions, or retrain the bridge. It checks whether the balanced caption
representation gain persists at scenario level, then produces quantitative and
qualitative plots showing how the richer text representation changes support
selection and generated raw-level scenario paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_caption_rollout_group_summary import (  # noqa: E402
    summarize_groups,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)


DEFAULT_REVERSE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_917e_balanced80/caption_reverse_ab_report.json"
)
DEFAULT_SMALL_ROLLOUT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_917e_balanced80/small_rollout/"
    "scenario_level_eval_report.json"
)
DEFAULT_SMALL_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_917e_balanced80/small_rollout/"
    "scenario_level_eval_arrays.npz"
)
DEFAULT_LARGE_ROLLOUT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_917e_balanced80/large_rollout/"
    "scenario_level_eval_report.json"
)
DEFAULT_LARGE_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_917e_balanced80/large_rollout/"
    "scenario_level_eval_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "caption_conditionality_refresh_917f_balanced80"
)
DEFAULT_PAPER_FIGURE_DIR = Path("paper/narrative_grounded_scenarios/figures")


MARKETS = [
    ("SPX", 25),
    ("VIX", 38),
    ("Gold", 37),
    ("1Y ATM IV", 17),
]
PLOT_VARIANTS = [
    ("simple_fact_tokens", "Simple fact tokens", "#607D8B"),
    ("codex_v2_training_caption", "Codex professional caption", "#1565C0"),
    ("codex_v2_fused_fact_training_caption", "Fused Codex + facts", "#2E7D32"),
]
GROUP_ORDER = [
    "simple",
    "generic_demo",
    "rich_codex",
    "fused_codex",
    "rich_api",
    "fused_api",
]
GROUP_LABELS = {
    "simple": "Simple facts",
    "generic_demo": "Old demo",
    "rich_codex": "Codex caption",
    "fused_codex": "Codex+facts",
    "rich_api": "API caption",
    "fused_api": "API+facts",
}


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
        return float(value)
    return value


def parse_query_id(query_id: str) -> tuple[str, str, str]:
    parts = str(query_id).rsplit("::", 2)
    if len(parts) != 3:
        raise ValueError(f"cannot parse query_id: {query_id}")
    return parts[0], parts[1], parts[2]


def _safe_query_key(row_no: int, query_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(query_id))
    return f"{row_no:04d}_{safe}"


def _variant_rows_by_model(reverse_report: dict[str, Any], embedding_model: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for model_report in reverse_report.get("model_reports", []):
        if str(model_report.get("embedding_model")) != str(embedding_model):
            continue
        for row in model_report.get("rows", []):
            out[(str(row["window_id"]), str(row["variant_id"]))] = row
    return out


def _score_rows_by_query(rollout_report: dict[str, Any], embedding_model: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rollout_report.get("window_scores", []):
        window_id, variant_id, model = parse_query_id(str(row["query_id"]))
        if model != str(embedding_model):
            continue
        out[(window_id, variant_id)] = row
    return out


def _scenario_gain(row: dict[str, Any]) -> dict[str, float]:
    methods = row.get("methods", {})
    narrative = methods.get("narrative_generator_topk", {})
    persistence = methods.get("persistence", {})
    crps = float(narrative.get("ensemble_crps_z", np.nan))
    energy = float(narrative.get("energy_score_z", np.nan))
    base_crps = float(persistence.get("ensemble_crps_z", np.nan))
    base_energy = float(persistence.get("energy_score_z", np.nan))
    return {
        "crps": crps,
        "energy": energy,
        "crps_improvement_vs_persistence": 1.0 - crps / base_crps,
        "energy_improvement_vs_persistence": 1.0 - energy / base_energy,
    }


def _group_metric(group_report: dict[str, Any], group: str, metric: str) -> float:
    row = group_report["group_summaries"][group]
    value = row[metric]
    if value is None:
        return float("nan")
    return float(value)


def gain_persists(
    small_group_report: dict[str, Any],
    large_group_report: dict[str, Any],
    *,
    candidate_group: str = "fused_codex",
    baseline_group: str = "simple",
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, report in [("small", small_group_report), ("large", large_group_report)]:
        cand_crps = _group_metric(report, candidate_group, "crps_improvement_vs_persistence")
        base_crps = _group_metric(report, baseline_group, "crps_improvement_vs_persistence")
        cand_energy = _group_metric(report, candidate_group, "energy_improvement_vs_persistence")
        base_energy = _group_metric(report, baseline_group, "energy_improvement_vs_persistence")
        checks[name] = {
            "candidate_crps_improvement": cand_crps,
            "baseline_crps_improvement": base_crps,
            "candidate_energy_improvement": cand_energy,
            "baseline_energy_improvement": base_energy,
            "crps_gain": cand_crps - base_crps,
            "energy_gain": cand_energy - base_energy,
            "passes": bool(cand_crps > base_crps and cand_energy > base_energy),
        }
    return {
        "candidate_group": candidate_group,
        "baseline_group": baseline_group,
        "passes_all_models": bool(all(item["passes"] for item in checks.values())),
        "checks": checks,
    }


def _history_args(max_windows: int = 441) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=DEFAULT_CHECKPOINT,
        state_scope="joint38",
        eval_split="val",
        test_start=4511,
        val_size=441,
        max_windows=int(max_windows),
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )


def load_history_future(checkpoint: str, *, device_name: str) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(device_name if str(device_name) == "cpu" or torch.cuda.is_available() else "cpu")
    _model, payload = load_model(checkpoint, device)
    *_unused, history_raw, _specs, block = build_val_block(_history_args(), payload)
    future_raw = np.asarray(block.future_state[: history_raw.shape[0], :, : history_raw.shape[-1]], dtype=np.float32)
    return np.asarray(history_raw, dtype=np.float32), future_raw


def select_qualitative_window(
    reverse_report: dict[str, Any],
    rollout_report: dict[str, Any],
    *,
    embedding_model: str,
    prefer_non_train: bool = True,
) -> dict[str, Any]:
    variants = _variant_rows_by_model(reverse_report, embedding_model)
    scores = _score_rows_by_query(rollout_report, embedding_model)
    by_window: dict[str, dict[str, Any]] = defaultdict(dict)
    for (window_id, variant_id), score in scores.items():
        variant = variants.get((window_id, variant_id), {})
        by_window[window_id][variant_id] = {
            "score": score,
            "variant": variant,
            "gain": _scenario_gain(score),
        }

    candidates: list[dict[str, Any]] = []
    for window_id, rows in by_window.items():
        if "simple_fact_tokens" not in rows or "codex_v2_fused_fact_training_caption" not in rows:
            continue
        fused = rows["codex_v2_fused_fact_training_caption"]
        simple = rows["simple_fact_tokens"]
        split = str(fused["variant"].get("split", ""))
        if prefer_non_train and split == "train":
            continue
        fused_support = {
            int(item["window_index"])
            for item in fused["variant"].get("support_rows", [])
            if "window_index" in item
        }
        simple_support = {
            int(item["window_index"])
            for item in simple["variant"].get("support_rows", [])
            if "window_index" in item
        }
        union = len(fused_support | simple_support)
        jaccard = len(fused_support & simple_support) / union if union else 0.0
        row = {
            "window_id": window_id,
            "window_index": int(fused["score"]["window_index"]),
            "block_window_index": int(fused["score"]["block_window_index"]),
            "split": split,
            "simple_crps": simple["gain"]["crps"],
            "fused_crps": fused["gain"]["crps"],
            "simple_energy": simple["gain"]["energy"],
            "fused_energy": fused["gain"]["energy"],
            "crps_delta_simple_minus_fused": simple["gain"]["crps"] - fused["gain"]["crps"],
            "energy_delta_simple_minus_fused": simple["gain"]["energy"] - fused["gain"]["energy"],
            "support_jaccard_simple_fused": float(jaccard),
        }
        candidates.append(row)
    if not candidates and prefer_non_train:
        return select_qualitative_window(
            reverse_report,
            rollout_report,
            embedding_model=embedding_model,
            prefer_non_train=False,
        )
    if not candidates:
        raise ValueError("no qualitative window candidates found")
    candidates.sort(
        key=lambda row: (
            float(row["crps_delta_simple_minus_fused"]),
            -float(row["support_jaccard_simple_fused"]),
        ),
        reverse=True,
    )
    return candidates[0] | {"candidate_count": len(candidates), "top_candidates": candidates[:8]}


def _plot_group_metrics(
    small_report: dict[str, Any],
    large_report: dict[str, Any],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    models = [("Small embedding", small_report), ("Large embedding", large_report)]
    colors = {
        "simple": "#607D8B",
        "generic_demo": "#9E9E9E",
        "rich_codex": "#1565C0",
        "fused_codex": "#2E7D32",
        "rich_api": "#8E24AA",
        "fused_api": "#EF6C00",
    }
    for col, (model_label, report) in enumerate(models):
        groups = [g for g in GROUP_ORDER if g in report["group_summaries"]]
        labels = [GROUP_LABELS.get(g, g) for g in groups]
        x = np.arange(len(groups))
        width = 0.38

        ax = axes[0, col]
        target = [_group_metric(report, g, "mean_target_cosine") for g in groups]
        support = [_group_metric(report, g, "mean_support_cosine") for g in groups]
        ax.bar(x - width / 2, target, width=width, color="#1565C0", label="Target")
        ax.bar(x + width / 2, support, width=width, color="#2E7D32", label="Support")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(f"{model_label}: bridge/support alignment")
        ax.set_ylim(0, max(1.0, max(target + support) * 1.10))
        ax.grid(axis="y", alpha=0.18)
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[1, col]
        crps = [_group_metric(report, g, "crps_improvement_vs_persistence") for g in groups]
        energy = [_group_metric(report, g, "energy_improvement_vs_persistence") for g in groups]
        ax.bar(x - width / 2, crps, width=width, color="#1565C0", label="CRPS")
        ax.bar(x + width / 2, energy, width=width, color="#2E7D32", label="Energy")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(f"{model_label}: rollout improvement vs persistence")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.18)
        if col == 0:
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Cosine")
    axes[1, 0].set_ylabel("Relative improvement")
    fig.suptitle("Balanced-80 caption representation gate", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_support_scatter(
    reverse_report: dict[str, Any],
    rollout_report: dict[str, Any],
    *,
    embedding_model: str,
    output: Path,
) -> None:
    variants = _variant_rows_by_model(reverse_report, embedding_model)
    scores = _score_rows_by_query(rollout_report, embedding_model)
    rows: list[dict[str, Any]] = []
    for key, score in scores.items():
        variant = variants.get(key)
        if not variant:
            continue
        gain = _scenario_gain(score)
        rows.append(
            {
                "group": str(variant["variant_group"]),
                "target_cosine": float(variant["target_cosine"]),
                "support_cosine": float(variant["mean_support_cosine"]),
                "crps_improvement": gain["crps_improvement_vs_persistence"],
                "energy_improvement": gain["energy_improvement_vs_persistence"],
            }
        )
    colors = {
        "simple": "#607D8B",
        "generic_demo": "#9E9E9E",
        "rich_codex": "#1565C0",
        "fused_codex": "#2E7D32",
        "rich_api": "#8E24AA",
        "fused_api": "#EF6C00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for group in GROUP_ORDER:
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        axes[0].scatter(
            [row["support_cosine"] for row in group_rows],
            [row["crps_improvement"] for row in group_rows],
            s=22,
            alpha=0.58,
            color=colors.get(group, "#777777"),
            label=GROUP_LABELS.get(group, group),
        )
        axes[1].scatter(
            [row["target_cosine"] for row in group_rows],
            [row["energy_improvement"] for row in group_rows],
            s=22,
            alpha=0.58,
            color=colors.get(group, "#777777"),
            label=GROUP_LABELS.get(group, group),
        )
    axes[0].set_xlabel("Mean support cosine")
    axes[0].set_ylabel("CRPS improvement vs persistence")
    axes[1].set_xlabel("Target memory cosine")
    axes[1].set_ylabel("Energy improvement vs persistence")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Per-variant support alignment and rollout quality ({embedding_model})")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_fan(
    ax: plt.Axes,
    delta_paths: np.ndarray,
    *,
    factor_index: int,
    start: float,
    actual_future: np.ndarray,
    color: str,
    title: str,
) -> None:
    days = np.arange(31)
    paths = np.asarray(delta_paths, dtype=np.float32) + float(start)
    full = np.concatenate(
        [
            np.full((paths.shape[0], 1), float(start), dtype=np.float32),
            paths[:, :, factor_index],
        ],
        axis=1,
    )
    q10 = np.quantile(full, 0.10, axis=0)
    q50 = np.quantile(full, 0.50, axis=0)
    q90 = np.quantile(full, 0.90, axis=0)
    mean = np.mean(full, axis=0)
    ax.fill_between(days, q10, q90, color=color, alpha=0.16)
    ax.plot(days, q50, color=color, linewidth=2.0, label="median")
    ax.plot(days, mean, color="#455A64", linewidth=1.2, linestyle="--", label="mean")
    picks = np.linspace(0, paths.shape[0] - 1, min(4, paths.shape[0]), dtype=int)
    for pick in picks:
        ax.plot(days, full[pick], color=color, alpha=0.35, linewidth=0.85)
    ax.plot(days, np.r_[start, actual_future[:, factor_index]], color="black", linewidth=1.5, linestyle="--", label="realized")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 30)
    ax.grid(alpha=0.15)


def _array_key(arrays: np.lib.npyio.NpzFile, row_no: int, query_id: str) -> str:
    suffix = _safe_query_key(row_no, query_id)
    key = f"narrative_{suffix}"
    if key in arrays.files:
        return key
    prefix = f"narrative_{row_no:04d}_"
    matches = [name for name in arrays.files if name.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"could not find generated path key for row {row_no}: {query_id}")


def _plot_qualitative_casebook(
    rollout_report: dict[str, Any],
    arrays_path: Path,
    selected: dict[str, Any],
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    *,
    output: Path,
) -> dict[str, Any]:
    arrays = np.load(arrays_path)
    score_lookup = {
        parse_query_id(str(row["query_id"]))[1]: row
        for row in rollout_report.get("window_scores", [])
        if parse_query_id(str(row["query_id"]))[0] == str(selected["window_id"])
    }
    block_idx = int(selected["block_window_index"])
    start_vec = np.asarray(history_raw[block_idx, -1, :], dtype=np.float32)
    actual_future = np.asarray(future_raw[block_idx], dtype=np.float32)
    fig, axes = plt.subplots(len(MARKETS), len(PLOT_VARIANTS), figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"Same historical query, different text conditions ({selected['window_id']}, {selected['split']})\n"
        "Raw levels; black dashed line is realized future",
        fontsize=13,
        fontweight="bold",
    )
    variant_summaries: list[dict[str, Any]] = []
    for col, (variant_id, label, color) in enumerate(PLOT_VARIANTS):
        score = score_lookup[variant_id]
        key = _array_key(arrays, int(score["row_no"]), str(score["query_id"]))
        paths = np.asarray(arrays[key], dtype=np.float32)
        gain = _scenario_gain(score)
        variant_summaries.append(
            {
                "variant_id": variant_id,
                "label": label,
                "array_key": key,
                "crps": gain["crps"],
                "energy": gain["energy"],
                "crps_improvement_vs_persistence": gain["crps_improvement_vs_persistence"],
                "energy_improvement_vs_persistence": gain["energy_improvement_vs_persistence"],
                "top_train_window_ids": score.get("top_train_window_ids", []),
                "top_train_indices": score.get("top_train_indices", []),
                "top_train_cosines": score.get("top_train_cosines", []),
            }
        )
        for row_no, (market, idx) in enumerate(MARKETS):
            ax = axes[row_no, col]
            _plot_fan(
                ax,
                paths,
                factor_index=idx,
                start=float(start_vec[idx]),
                actual_future=actual_future,
                color=color,
                title=label if row_no == 0 else market,
            )
            if col == 0:
                ax.set_ylabel(f"{market}\nraw level")
            if row_no == len(MARKETS) - 1:
                ax.set_xlabel("Forward day")
            if row_no == 0 and col == 0:
                ax.legend(fontsize=7)
    for row_no in range(len(MARKETS)):
        row_axes = [axes[row_no, col] for col in range(len(PLOT_VARIANTS))]
        lows, highs = zip(*(ax.get_ylim() for ax in row_axes), strict=True)
        low = float(min(lows))
        high = float(max(highs))
        pad = 0.03 * max(high - low, 1e-8)
        for ax in row_axes:
            ax.set_ylim(low - pad, high + pad)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "selected_window": selected,
        "start_levels": {name: float(start_vec[idx]) for name, idx in MARKETS},
        "variant_summaries": variant_summaries,
    }


def _markdown(report: dict[str, Any]) -> str:
    gain = report["scenario_gain_gate"]
    selected = report["qualitative_casebook"]["selected_window"]
    lines = [
        "# Caption Conditionality Refresh",
        "",
        "## Scenario-Level Gate",
        "",
        f"Candidate: `{gain['candidate_group']}`; baseline: `{gain['baseline_group']}`.",
        "",
    ]
    for model, row in gain["checks"].items():
        lines.append(
            f"- {model}: CRPS gain `{row['crps_gain']:+.3f}`, "
            f"energy gain `{row['energy_gain']:+.3f}`, pass `{row['passes']}`."
        )
    lines += [
        "",
        "## Qualitative Case",
        "",
        (
            f"Selected `{selected['window_id']}` ({selected['split']}) because fused Codex "
            f"reduced CRPS by `{selected['crps_delta_simple_minus_fused']:.3f}` "
            "versus simple fact tokens while changing the support set."
        ),
        "",
        "## Artifacts",
        "",
    ]
    for key, path in report["artifact_paths"].items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _copy_to_paper(path: Path, paper_dir: Path | None) -> str | None:
    if paper_dir is None:
        return None
    paper_dir.mkdir(parents=True, exist_ok=True)
    target = paper_dir / path.name
    shutil.copy2(path, target)
    return str(target)


def build_refresh(args: argparse.Namespace) -> dict[str, Any]:
    reverse_report = _load_json(args.reverse_report)
    small_rollout = _load_json(args.small_rollout_report)
    large_rollout = _load_json(args.large_rollout_report)
    small_group = summarize_groups(reverse_report, small_rollout, embedding_model="text-embedding-3-small")
    large_group = summarize_groups(reverse_report, large_rollout, embedding_model="text-embedding-3-large")
    gate = gain_persists(small_group, large_group)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_plot = output_dir / "caption_group_metric_gate_balanced80.png"
    scatter_plot = output_dir / "caption_support_quality_scatter_small_balanced80.png"
    _plot_group_metrics(small_group, large_group, metric_plot)
    _plot_support_scatter(
        reverse_report,
        small_rollout,
        embedding_model="text-embedding-3-small",
        output=scatter_plot,
    )

    selected = select_qualitative_window(
        reverse_report,
        small_rollout,
        embedding_model="text-embedding-3-small",
    )
    history_raw, future_raw = load_history_future(str(args.checkpoint), device_name=str(args.device))
    casebook_plot = output_dir / "caption_same_window_raw_level_casebook_balanced80.png"
    qualitative = _plot_qualitative_casebook(
        small_rollout,
        Path(args.small_arrays),
        selected,
        history_raw,
        future_raw,
        output=casebook_plot,
    )

    paper_dir = Path(args.paper_figure_dir) if args.paper_figure_dir else None
    artifact_paths = {
        "summary_json": str(output_dir / "caption_conditionality_refresh_summary.json"),
        "summary_markdown": str(output_dir / "caption_conditionality_refresh_summary.md"),
        "group_metric_plot": str(metric_plot),
        "support_quality_scatter": str(scatter_plot),
        "same_window_casebook": str(casebook_plot),
    }
    for key in ["group_metric_plot", "support_quality_scatter", "same_window_casebook"]:
        copied = _copy_to_paper(Path(artifact_paths[key]), paper_dir)
        if copied:
            artifact_paths[f"paper_{key}"] = copied

    report = {
        "scope_note": (
            "Balanced-80 caption conditionality refresh. The gate is scenario-level "
            "CRPS/energy improvement of fused Codex captions versus the simple text "
            "floor. Qualitative plots use raw levels for the same historical query "
            "under simple, rich Codex, and fused Codex text conditions."
        ),
        "scenario_gain_gate": gate,
        "small_group_summary": small_group["group_summaries"],
        "large_group_summary": large_group["group_summaries"],
        "qualitative_casebook": qualitative,
        "artifact_paths": artifact_paths,
    }
    _write_json(artifact_paths["summary_json"], report)
    _write_text(artifact_paths["summary_markdown"], _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverse-report", type=Path, default=DEFAULT_REVERSE_REPORT)
    parser.add_argument("--small-rollout-report", type=Path, default=DEFAULT_SMALL_ROLLOUT)
    parser.add_argument("--small-arrays", type=Path, default=DEFAULT_SMALL_ARRAYS)
    parser.add_argument("--large-rollout-report", type=Path, default=DEFAULT_LARGE_ROLLOUT)
    parser.add_argument("--large-arrays", type=Path, default=DEFAULT_LARGE_ARRAYS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-figure-dir", type=Path, default=DEFAULT_PAPER_FIGURE_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = build_refresh(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "scenario_gain_gate": report["scenario_gain_gate"],
                "selected_window": report["qualitative_casebook"]["selected_window"],
                "artifact_paths": report["artifact_paths"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
