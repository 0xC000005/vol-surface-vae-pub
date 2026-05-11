#!/usr/bin/env python
"""Diagnose whether fixed start state helps narrative-to-memory alignment.

This script makes no OpenAI calls. It reuses the representative OpenAI text
embeddings, the existing train/test split, and the incumbent
MLP+hard-negative memory target. The only tested mechanism is whether appending
the fixed starting level to each caption improves the generator-memory target.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
from experiments.backfill.block_ar.nl_bridge_architecture_bakeoff import (  # noqa: E402
    run_single_method,
    summarize_bakeoff,
)
from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
    load_pipeline_artifacts,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    selected_bridge_window_indices,
)
from experiments.backfill.block_ar.nl_prefix_latent_text_bridge import (  # noqa: E402
    DEFAULT_PIPELINE_ARRAYS,
    DEFAULT_PIPELINE_REPORT,
)
from experiments.backfill.block_ar.nl_text_conditioning import (
    normalize_rows,
)  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_text_start_memory_diagnostic_875a"
)


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
    return round(float(value), 12)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True)
    return np.maximum(std, 1e-6).astype(np.float32)


def standardize_start_features(
    start_state: np.ndarray,
    example_window_indices: np.ndarray,
    *,
    fit_window_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return standardized start features aligned to example rows.

    Negative captions have no memory-regression target, but they still belong
    to a source window. Use that source window's start state so the contrastive
    comparison tests text direction, not a missing or synthetic start feature.
    """

    starts = np.asarray(start_state, dtype=np.float32)
    example_windows = np.asarray(example_window_indices, dtype=np.int64)
    fit = np.asarray(fit_window_indices, dtype=np.int64)
    if starts.ndim != 2:
        raise ValueError("start_state must have shape [N, C]")
    if example_windows.ndim != 1:
        raise ValueError("example_window_indices must be 1-D")
    if np.any(example_windows < 0) or np.any(example_windows >= starts.shape[0]):
        raise ValueError("example_window_indices point outside start_state")
    if fit.size == 0:
        raise ValueError("fit_window_indices must be non-empty")
    if np.any(fit < 0) or np.any(fit >= starts.shape[0]):
        raise ValueError("fit_window_indices point outside start_state")
    mean = starts[fit].mean(axis=0, keepdims=True).astype(np.float32)
    std = _safe_std(starts[fit])
    features = ((starts[example_windows] - mean) / std).astype(np.float32)
    return features, {"start_mean": mean, "start_std": std}


def build_input_features(
    text_embeddings: np.ndarray,
    start_state: np.ndarray,
    examples: list[dict[str, Any]],
    *,
    fit_window_indices: np.ndarray,
    input_mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build text-only, start-only, or text-plus-start bridge inputs."""

    text = normalize_rows(np.asarray(text_embeddings, dtype=np.float32))
    if text.ndim != 2:
        raise ValueError("text_embeddings must have shape [M, D]")
    if len(examples) != int(text.shape[0]):
        raise ValueError("examples length must match text_embeddings rows")
    example_windows = np.asarray(
        [int(example["window_index"]) for example in examples],
        dtype=np.int64,
    )
    start_features, stats = standardize_start_features(
        start_state,
        example_windows,
        fit_window_indices=np.asarray(fit_window_indices, dtype=np.int64),
    )
    mode = str(input_mode)
    if mode == "text_only":
        return text.astype(np.float32), stats
    if mode == "start_only":
        return start_features.astype(np.float32), stats
    if mode == "text_start":
        return np.concatenate([text, start_features], axis=1).astype(np.float32), stats
    raise ValueError("input_mode must be text_only, start_only, or text_start")


def caption_coverage_audit(
    examples: list[dict[str, Any]],
    split: dict[str, Any],
) -> dict[str, Any]:
    """Summarize caption role/kind coverage by split."""

    split_by_window: dict[int, str] = {}
    for key, label in (
        ("train_indices", "train"),
        ("test_indices", "test"),
        ("excluded_indices", "excluded"),
    ):
        for idx in split.get(key, []):
            split_by_window[int(idx)] = label
    role_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    split_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    positives_by_window: dict[int, int] = defaultdict(int)
    negatives_by_window: dict[int, int] = defaultdict(int)
    for example in examples:
        window = int(example["window_index"])
        role = str(example.get("role", ""))
        kind = str(example.get("kind", ""))
        split_name = split_by_window.get(window, "unknown")
        role_counts[role] += 1
        kind_counts[kind] += 1
        split_role_counts[split_name][role] += 1
        if role == "positive":
            positives_by_window[window] += 1
        if role == "negative":
            negatives_by_window[window] += 1
    windows = sorted({int(example["window_index"]) for example in examples})
    positive_counts = np.asarray(
        [positives_by_window[idx] for idx in windows], dtype=np.float64
    )
    negative_counts = np.asarray(
        [negatives_by_window[idx] for idx in windows], dtype=np.float64
    )
    return {
        "window_count": len(windows),
        "example_count": len(examples),
        "role_counts": dict(sorted(role_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "split_role_counts": {
            split_name: dict(sorted(counter.items()))
            for split_name, counter in sorted(split_role_counts.items())
        },
        "positive_captions_per_window": {
            "min": (
                _round(float(positive_counts.min())) if positive_counts.size else None
            ),
            "mean": (
                _round(float(positive_counts.mean())) if positive_counts.size else None
            ),
            "max": (
                _round(float(positive_counts.max())) if positive_counts.size else None
            ),
        },
        "hard_negatives_per_window": {
            "min": (
                _round(float(negative_counts.min())) if negative_counts.size else None
            ),
            "mean": (
                _round(float(negative_counts.mean())) if negative_counts.size else None
            ),
            "max": (
                _round(float(negative_counts.max())) if negative_counts.size else None
            ),
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_delta(
    results: dict[str, dict[str, Any]],
    left: str,
    right: str,
    metric: str,
) -> float | None:
    lval = results.get(left, {}).get("summary", {}).get(metric)
    rval = results.get(right, {}).get("summary", {}).get(metric)
    if lval is None or rval is None:
        return None
    return _round(float(lval) - float(rval))


def _decision_block(
    results: dict[str, dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    target_cosine_floor_delta: float,
    hard_negative_floor_delta: float,
) -> dict[str, Any]:
    target_delta = _metric_delta(
        results, candidate, baseline, "heldout_mean_target_cosine"
    )
    gap_delta = _metric_delta(
        results, candidate, baseline, "heldout_hard_negative_mean_gap"
    )
    margin_delta = _metric_delta(
        results, candidate, baseline, "heldout_hard_negative_mean_margin"
    )
    passes = (
        target_delta is not None
        and gap_delta is not None
        and margin_delta is not None
        and target_delta >= float(target_cosine_floor_delta)
        and gap_delta >= float(hard_negative_floor_delta)
        and margin_delta >= float(hard_negative_floor_delta)
    )
    return {
        "status": "pass" if passes else "diagnostic_only",
        "candidate": candidate,
        "baseline": baseline,
        "target_cosine_delta": target_delta,
        "hard_negative_gap_delta": gap_delta,
        "hard_negative_margin_delta": margin_delta,
        "target_cosine_floor_delta": float(target_cosine_floor_delta),
        "hard_negative_floor_delta": float(hard_negative_floor_delta),
        "interpretation": (
            "text_plus_start clears the local target diagnostic"
            if passes
            else "text_plus_start does not clear the local target diagnostic"
        ),
    }


def _load_selected_start_states(
    args: argparse.Namespace,
    selected_windows: np.ndarray,
) -> np.ndarray:
    """Rebuild selected-window start states from the frozen SNI validation block."""

    model, payload = load_model(args.checkpoint, torch.device("cpu"))
    del model
    (
        all_history_level,
        _all_history_norm,
        _all_center,
        _all_scale,
        _all_drift_feature,
        _all_history_raw,
        _specs,
        _block,
    ) = build_val_block(args, payload)
    selected = np.asarray(selected_windows, dtype=np.int64)
    if all_history_level.shape[0] <= int(np.max(selected)):
        raise ValueError(
            f"rebuilt block has {all_history_level.shape[0]} windows, "
            f"need {int(np.max(selected)) + 1}"
        )
    return np.asarray(all_history_level[selected, -1, :], dtype=np.float32)


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Text Plus Start Memory Diagnostic",
        "",
        report["scope_note"],
        "",
        "## Decision",
        "",
        f"- Status: `{report['decision']['status']}`.",
        f"- Candidate: `{report['decision']['candidate']}`.",
        f"- Baseline: `{report['decision']['baseline']}`.",
        f"- Target cosine delta: `{report['decision']['target_cosine_delta']}`.",
        f"- Hard-negative gap delta: `{report['decision']['hard_negative_gap_delta']}`.",
        f"- Hard-negative margin delta: `{report['decision']['hard_negative_margin_delta']}`.",
        "",
        "## Caption Coverage",
        "",
        f"- Windows: `{report['caption_coverage']['window_count']}`.",
        f"- Text examples: `{report['caption_coverage']['example_count']}`.",
        f"- Role counts: `{report['caption_coverage']['role_counts']}`.",
        f"- Positive captions/window: `{report['caption_coverage']['positive_captions_per_window']}`.",
        f"- Hard negatives/window: `{report['caption_coverage']['hard_negatives_per_window']}`.",
        "",
        "## Method Summary",
        "",
        "| Variant | Target Cosine | Hard Gap | Hard Margin | Recall@1 Test | Recall@3 Test |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in report["results"].items():
        summary = payload.get("summary", {})
        lines.append(
            f"| `{name}` | "
            f"{summary.get('heldout_mean_target_cosine')} | "
            f"{summary.get('heldout_hard_negative_mean_gap')} | "
            f"{summary.get('heldout_hard_negative_mean_margin')} | "
            f"{summary.get('heldout_recall_at_1_test_pool')} | "
            f"{summary.get('heldout_recall_at_3_test_pool')} |"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_text_start_memory_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_report, arrays = load_pipeline_artifacts(
        args.pipeline_report, args.pipeline_arrays
    )
    bridge_report = _load_json(args.bridge_report)
    examples = build_bridge_examples(pipeline_report)
    text_embeddings = np.asarray(arrays["text_embeddings"], dtype=np.float32)
    memory_targets = np.asarray(arrays["memory_targets"], dtype=np.float32)
    if text_embeddings.shape[0] != len(examples):
        raise ValueError("text embedding rows must match rebuilt examples")
    selected_windows = selected_bridge_window_indices(bridge_report)
    if selected_windows.shape[0] != memory_targets.shape[0]:
        raise ValueError(
            "selected windows and memory targets must have the same row count"
        )
    split = bridge_report.get("split")
    if (
        not isinstance(split, dict)
        or not split.get("train_indices")
        or not split.get("test_indices")
    ):
        raise ValueError("bridge report must contain a train/test split")
    start_state = _load_selected_start_states(args, selected_windows)
    modes = [item.strip() for item in str(args.input_modes).split(",") if item.strip()]
    results: dict[str, dict[str, Any]] = {}
    start_stats: dict[str, dict[str, Any]] = {}
    for mode in modes:
        features, stats = build_input_features(
            text_embeddings,
            start_state,
            examples,
            fit_window_indices=np.asarray(split["train_indices"], dtype=np.int64),
            input_mode=mode,
        )
        start_stats[mode] = {
            "input_dim": int(features.shape[1]),
            "start_mean_shape": list(stats["start_mean"].shape),
            "start_std_min": _round(float(np.min(stats["start_std"]))),
            "start_std_max": _round(float(np.max(stats["start_std"]))),
        }
        name = f"mlp_mse_contrastive__{args.training_policy}__{mode}"
        results[name] = run_single_method(
            "mlp_mse_contrastive",
            str(args.training_policy),
            examples,
            features,
            memory_targets,
            split,
            args,
        )
    baseline = f"mlp_mse_contrastive__{args.training_policy}__text_only"
    candidate = f"mlp_mse_contrastive__{args.training_policy}__text_start"
    report = {
        "status": "ok",
        "scope_note": (
            "Cached text/start generator-memory diagnostic. This compares "
            "text-only, start-only, and text-plus-start inputs under the same "
            "incumbent MLP memory-regression plus hard-negative objective. "
            "No OpenAI API calls are made."
        ),
        "pipeline_report": str(args.pipeline_report),
        "pipeline_arrays": str(args.pipeline_arrays),
        "bridge_report": str(args.bridge_report),
        "selected_window_count": int(selected_windows.shape[0]),
        "embedding_backend": pipeline_report.get("embedding_backend"),
        "embedding_model": pipeline_report.get("embedding_model"),
        "input_modes": modes,
        "training_policy": str(args.training_policy),
        "split": split,
        "caption_coverage": caption_coverage_audit(examples, split),
        "start_feature_stats": start_stats,
        "summary": summarize_bakeoff(results),
        "decision": _decision_block(
            results,
            candidate=candidate,
            baseline=baseline,
            target_cosine_floor_delta=float(args.target_cosine_floor_delta),
            hard_negative_floor_delta=float(args.hard_negative_floor_delta),
        ),
        "results": results,
        "artifact_paths": {
            "report": str(Path(args.output_dir) / "text_start_memory_diagnostic.json"),
            "markdown": str(Path(args.output_dir) / "text_start_memory_diagnostic.md"),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "text_start_memory_diagnostic.json", report)
    write_markdown_report(output_dir / "text_start_memory_diagnostic.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--pipeline-arrays", default=DEFAULT_PIPELINE_ARRAYS)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--input-modes",
        default="text_only,text_start,start_only",
        help="Comma-separated modes: text_only,text_start,start_only.",
    )
    parser.add_argument("--training-policy", default="multi_caption_with_negatives")
    parser.add_argument("--adapter-steps", type=int, default=700)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mlp-contrastive-weight", type=float, default=0.25)
    parser.add_argument("--hard-negative-margin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=775)
    parser.add_argument("--target-cosine-floor-delta", type=float, default=-0.01)
    parser.add_argument("--hard-negative-floor-delta", type=float, default=-0.05)
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument(
        "--eval_split", choices=["val", "train", "train_tail"], default="val"
    )
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument(
        "--iv_transform", choices=["log_level", "bounded_logit"], default="log_level"
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument(
        "--drift_feature_mode", choices=["none", "ewma_mean"], default="none"
    )
    args = parser.parse_args()
    report = run_text_start_memory_diagnostic(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "decision": report["decision"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
