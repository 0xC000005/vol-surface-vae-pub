#!/usr/bin/env python
"""Fixed-start conditionality audit for the full professional caption corpus.

This script asks one narrow question: after the full professional caption corpus
improves text-to-support backtests, does that signal survive into final
fixed-start scenario distributions?

It reuses cached caption condition vectors from the reverse A/B bridge report,
forces all selected narratives to share one explicit starting level, and runs
three conditions:

1. professional Codex+fact caption;
2. simple fact-token text;
3. start-only support selection.

No caption generation or embedding calls are made here.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)


DEFAULT_BRIDGE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_956e_full_corpus_small/"
    "reverse_ab_bridge_report_text_embedding_3_small.json"
)
DEFAULT_BRIDGE_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_956e_full_corpus_small/"
    "reverse_ab_bridge_arrays_text_embedding_3_small.npz"
)
DEFAULT_SUPPORT_BANK_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_SUPPORT_BANK_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "full_corpus_fixed_start_conditionality_audit_957a"
)
STORY_SMOKE_SCRIPT = "experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py"

PROFESSIONAL_KIND = "codex_v2_fused_fact_training_caption"
SIMPLE_KIND = "simple_fact_tokens"
FACTOR_INDEX = {
    "SPX": 25,
    "VIX": 38,
    "DXY": 28,
    "CRUDE_OIL": 31,
    "US10Y": 33,
    "BBB_OAS": 35,
    "GOLD": 37,
    "IV_ATM_1Y": 17,
}
SUMMARY_FACTORS = ("SPX", "VIX", "CRUDE_OIL", "US10Y", "BBB_OAS", "GOLD")
PORTFOLIO_EXPOSURES = (
    ("SPX", 1.00),
    ("VIX", -0.55),
    ("BBB_OAS", -0.45),
    ("US10Y", -0.35),
    ("DXY", -0.25),
    ("CRUDE_OIL", 0.20),
    ("GOLD", 0.15),
)


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
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    denom = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    return arr / denom


def _rows_by_window_kind(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = report.get("evaluation", {}).get("heldout_examples", [])
    if not isinstance(rows, list):
        raise ValueError("bridge report must contain evaluation.heldout_examples")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("role", "")) != "anchor":
            continue
        out[(str(row.get("window_id")), str(row.get("kind")))] = row
    return out


def select_diverse_caption_cases(
    report: dict[str, Any],
    *,
    condition_vectors: np.ndarray,
    case_count: int = 6,
    professional_kind: str = PROFESSIONAL_KIND,
    simple_kind: str = SIMPLE_KIND,
    non_train_only: bool = True,
) -> list[dict[str, Any]]:
    """Select paired simple/professional caption cases with diverse text memory."""

    by_key = _rows_by_window_kind(report)
    train = {int(idx) for idx in report.get("split", {}).get("train_indices", [])}
    candidates: list[dict[str, Any]] = []
    for (window_id, kind), professional in sorted(by_key.items()):
        if kind != professional_kind:
            continue
        simple = by_key.get((window_id, simple_kind))
        if simple is None:
            continue
        window_index = int(professional["window_index"])
        if non_train_only and window_index in train:
            continue
        candidates.append(
            {
                "window_id": window_id,
                "window_index": window_index,
                "professional_kind": professional_kind,
                "simple_kind": simple_kind,
                "professional_embedding_index": int(professional["embedding_index"]),
                "simple_embedding_index": int(simple["embedding_index"]),
            }
        )
    if not candidates:
        raise ValueError("no paired simple/professional caption candidates found")
    if len(candidates) <= int(case_count):
        return candidates

    vectors = _normalize_rows(
        np.asarray(
            [
                condition_vectors[int(case["professional_embedding_index"])]
                for case in candidates
            ],
            dtype=np.float32,
        )
    )
    distance = 1.0 - vectors @ vectors.T
    first = int(np.argmax(np.mean(distance, axis=1)))
    selected = [first]
    while len(selected) < int(case_count):
        selected_set = set(selected)
        best_idx = None
        best_key = (-np.inf, "")
        for idx, case in enumerate(candidates):
            if idx in selected_set:
                continue
            min_distance = float(np.min(distance[idx, selected]))
            key = (min_distance, str(case["window_id"]))
            if key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
    return [candidates[idx] for idx in selected]


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    a = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, grid, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(left.shape[0], -1)
    b = np.asarray(right, dtype=np.float64).reshape(right.shape[0], -1)
    if a.size == 0 or b.size == 0:
        return 0.0
    ab = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).mean()
    aa = 0.0 if a.shape[0] <= 1 else np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1).mean()
    bb = 0.0 if b.shape[0] <= 1 else np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1).mean()
    scale = math.sqrt(max(a.shape[1], 1))
    return float(max(2.0 * ab - aa - bb, 0.0) / scale)


def _support_jaccard(left: list[int], right: list[int]) -> float:
    a = {int(v) for v in left}
    b = {int(v) for v in right}
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _portfolio_terminal(states: np.ndarray, start: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    terms = []
    for market, sensitivity in PORTFOLIO_EXPOSURES:
        idx = FACTOR_INDEX[market]
        if idx >= arr.shape[-1]:
            continue
        scale = max(abs(float(start_arr[idx])), 1.0)
        terms.append((arr[:, -1, idx] - float(start_arr[idx])) / scale * sensitivity * 100.0)
    if not terms:
        return np.mean(arr[:, -1, :], axis=-1)
    return np.sum(np.stack(terms, axis=-1), axis=-1)


def summarize_condition_group(
    case_results: list[dict[str, Any]],
    *,
    factor_indices: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Summarize cross-narrative separation for one condition group."""

    factors = factor_indices or {
        name: idx
        for name, idx in FACTOR_INDEX.items()
        if name in SUMMARY_FACTORS
    }
    pair_rows: list[dict[str, Any]] = []
    for left, right in combinations(case_results, 2):
        factor_ks = {
            name: _ks_statistic(
                left["states"][:, -1, idx],
                right["states"][:, -1, idx],
            )
            for name, idx in factors.items()
            if idx < left["states"].shape[-1] and idx < right["states"].shape[-1]
        }
        pair_rows.append(
            {
                "left": str(left["case"]),
                "right": str(right["case"]),
                "support_jaccard": _support_jaccard(
                    left.get("support_indices", []),
                    right.get("support_indices", []),
                ),
                "mean_factor_terminal_ks": (
                    float(np.mean(list(factor_ks.values()))) if factor_ks else 0.0
                ),
                "factor_terminal_ks": factor_ks,
                "portfolio_terminal_ks": _ks_statistic(
                    _portfolio_terminal(left["states"], left["start"]),
                    _portfolio_terminal(right["states"], right["start"]),
                ),
                "path_energy": _energy_distance(left["states"], right["states"]),
            }
        )
    terminal_ranges: dict[str, float] = {}
    for name, idx in factors.items():
        if idx >= case_results[0]["states"].shape[-1]:
            continue
        means = [
            float(np.mean(case["states"][:, -1, idx]))
            for case in case_results
        ]
        terminal_ranges[name] = float(max(means) - min(means)) if means else 0.0
    return {
        "case_count": int(len(case_results)),
        "pair_count": int(len(pair_rows)),
        "mean_support_jaccard": (
            float(np.mean([row["support_jaccard"] for row in pair_rows]))
            if pair_rows
            else 0.0
        ),
        "max_support_jaccard": (
            float(np.max([row["support_jaccard"] for row in pair_rows]))
            if pair_rows
            else 0.0
        ),
        "mean_factor_terminal_ks": (
            float(np.mean([row["mean_factor_terminal_ks"] for row in pair_rows]))
            if pair_rows
            else 0.0
        ),
        "mean_portfolio_terminal_ks": (
            float(np.mean([row["portfolio_terminal_ks"] for row in pair_rows]))
            if pair_rows
            else 0.0
        ),
        "mean_path_energy": (
            float(np.mean([row["path_energy"] for row in pair_rows]))
            if pair_rows
            else 0.0
        ),
        "terminal_mean_level_ranges": terminal_ranges,
        "pairs": pair_rows,
    }


def fixed_start_decision(
    *,
    professional_summary: dict[str, Any],
    simple_summary: dict[str, Any],
    start_only_summary: dict[str, Any],
) -> dict[str, Any]:
    """Classify whether narrative effects survive the final pooled distribution."""

    prof_factor = float(professional_summary["mean_factor_terminal_ks"])
    prof_portfolio = float(professional_summary["mean_portfolio_terminal_ks"])
    prof_energy = float(professional_summary["mean_path_energy"])
    start_factor = float(start_only_summary["mean_factor_terminal_ks"])
    start_portfolio = float(start_only_summary["mean_portfolio_terminal_ks"])
    simple_factor = float(simple_summary["mean_factor_terminal_ks"])
    simple_portfolio = float(simple_summary["mean_portfolio_terminal_ks"])
    checks = {
        "above_start_only_factor": prof_factor >= max(0.05, start_factor + 0.05),
        "above_start_only_portfolio": prof_portfolio >= max(0.05, start_portfolio + 0.05),
        "above_start_only_path_energy": prof_energy >= float(start_only_summary["mean_path_energy"]) + 0.10,
        "support_not_collapsed": float(professional_summary["max_support_jaccard"]) <= 0.50,
    }
    status = "pass" if all(checks.values()) else "warning"
    if prof_factor <= start_factor + 1e-8 and prof_portfolio <= start_portfolio + 1e-8:
        status = "fail"
    return {
        "status": status,
        "checks": checks,
        "professional_minus_start_only": {
            "factor_terminal_ks": prof_factor - start_factor,
            "portfolio_terminal_ks": prof_portfolio - start_portfolio,
            "path_energy": prof_energy - float(start_only_summary["mean_path_energy"]),
        },
        "professional_minus_simple": {
            "factor_terminal_ks": prof_factor - simple_factor,
            "portfolio_terminal_ks": prof_portfolio - simple_portfolio,
            "path_energy": prof_energy - float(simple_summary["mean_path_energy"]),
        },
        "interpretation": (
            "pass means professional captions create nonzero final pooled "
            "distribution separation above a start-only null; warning means "
            "the signal is present but not strong across all gates; fail means "
            "fixed-start final distributions are effectively start-only."
        ),
    }


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build_story_smoke_command(
    *,
    bridge_report: str | Path,
    bridge_arrays: str | Path,
    support_bank_report: str | Path,
    support_bank_arrays: str | Path,
    output_dir: str | Path,
    query_window_id: str,
    query_kind: str,
    start_index: int,
    memory_prior_mode: str,
    samples: int,
    steps: int,
    seed: int,
    device: str,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
) -> list[str]:
    return [
        sys.executable,
        STORY_SMOKE_SCRIPT,
        "--bridge-report",
        str(bridge_report),
        "--bridge-arrays",
        str(bridge_arrays),
        "--support-bank-report",
        str(support_bank_report),
        "--support-bank-arrays",
        str(support_bank_arrays),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--query-role",
        "anchor",
        "--query-kind",
        str(query_kind),
        "--query-window-id",
        str(query_window_id),
        "--start-mode",
        "explicit_start_window",
        "--explicit-start-window-index",
        str(int(start_index)),
        "--memory-prior-mode",
        str(memory_prior_mode),
        "--memory-prior-top-k",
        "8",
        "--memory-prior-temperature",
        "0.2",
        "--memory-prior-diverse-min-index-gap",
        "30",
        "--prefix-prior-mode",
        "decoder",
        "--rollout-mixture-mode",
        "component_prefix_mixture",
        "--samples",
        str(int(samples)),
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
        "--rollout-seed",
        str(int(seed)),
        "--device",
        str(device),
    ]


def _operational_variant_index(report: dict[str, Any], generated_states: np.ndarray) -> int:
    cached = report.get("cached_query", {})
    if isinstance(cached, dict):
        raw = cached.get("operational_memory_prior_variant_index")
        if raw is not None:
            idx = int(raw)
            if 0 <= idx < generated_states.shape[0]:
                return idx
    return 0


def load_run_result(path: str | Path, *, case: str, condition: str) -> dict[str, Any]:
    output_dir = Path(path)
    report = _load_json(output_dir / "prefix_latent_story_smoke_report.json")
    arrays = np.load(output_dir / "prefix_latent_story_smoke_arrays.npz")
    generated = np.asarray(arrays["generated_states"], dtype=np.float32)
    op_idx = _operational_variant_index(report, generated)
    support_indices = [
        int(idx)
        for idx in np.asarray(arrays["rollout_component_window_index"]).reshape(-1)
    ]
    return {
        "case": str(case),
        "condition": str(condition),
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "prefix_latent_story_smoke_report.json"),
        "arrays_path": str(output_dir / "prefix_latent_story_smoke_arrays.npz"),
        "states": generated[op_idx],
        "start": np.asarray(arrays["requested_raw"], dtype=np.float32)[op_idx],
        "support_indices": support_indices,
        "support_count": len(set(support_indices)),
    }


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_")


def run_case_condition(
    *,
    case: dict[str, Any],
    condition: str,
    query_kind: str,
    memory_prior_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = (
        Path(args.output_dir)
        / f"start_{int(args.start_index)}"
        / _slug(str(case["window_id"]))
        / condition
    )
    report_path = output_dir / "prefix_latent_story_smoke_report.json"
    arrays_path = output_dir / "prefix_latent_story_smoke_arrays.npz"
    if not (bool(args.reuse_existing) and report_path.exists() and arrays_path.exists()):
        command = build_story_smoke_command(
            bridge_report=args.bridge_report,
            bridge_arrays=args.bridge_arrays,
            support_bank_report=args.support_bank_report,
            support_bank_arrays=args.support_bank_arrays,
            checkpoint=args.checkpoint,
            output_dir=output_dir,
            query_window_id=str(case["window_id"]),
            query_kind=query_kind,
            start_index=int(args.start_index),
            memory_prior_mode=memory_prior_mode,
            samples=int(args.samples),
            steps=int(args.steps),
            seed=int(args.seed),
            device=str(args.device),
        )
        if bool(args.dry_run):
            return {
                "case": str(case["window_id"]),
                "condition": condition,
                "output_dir": str(output_dir),
                "command": command,
            }
        _run_command(command)
    return load_run_result(output_dir, case=str(case["window_id"]), condition=condition)


def _plot_terminal_ranges(report: dict[str, Any], output: Path) -> str:
    groups = report["group_summaries"]
    factors = [name for name in SUMMARY_FACTORS if name in groups["professional"]["terminal_mean_level_ranges"]]
    x = np.arange(len(factors))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = {"professional": "#1565C0", "simple": "#607D8B", "start_only": "#9E9E9E"}
    labels = {"professional": "Professional captions", "simple": "Simple facts", "start_only": "Start-only"}
    for offset, group in [(-width, "professional"), (0.0, "simple"), (width, "start_only")]:
        values = [groups[group]["terminal_mean_level_ranges"].get(factor, 0.0) for factor in factors]
        ax.bar(x + offset, values, width=width, color=colors[group], label=labels[group])
    ax.set_xticks(x, factors, rotation=20, ha="right")
    ax.set_ylabel("Range of terminal mean raw levels across narratives")
    ax.set_title("Fixed-start final-distribution narrative separation")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def _plot_group_fan_panel(
    results: dict[str, list[dict[str, Any]]],
    *,
    factors: tuple[str, ...] = ("SPX", "VIX", "CRUDE_OIL", "GOLD"),
    output: Path,
) -> str:
    """Plot raw-level fan paths for professional/simple/start-only conditions."""

    groups = ("professional", "simple", "start_only")
    labels = {
        "professional": "Professional captions",
        "simple": "Simple facts",
        "start_only": "Start-only null",
    }
    colors = ["#1565C0", "#C62828", "#EF6C00", "#2E7D32", "#6A1B9A", "#00838F"]
    fig, axes = plt.subplots(
        len(factors),
        len(groups),
        figsize=(14, 2.4 * len(factors)),
        sharex=True,
        squeeze=False,
    )
    days = np.arange(31)
    y_limits: dict[str, list[float]] = {factor: [] for factor in factors}

    for row_no, factor in enumerate(factors):
        idx = FACTOR_INDEX[factor]
        for col_no, group in enumerate(groups):
            ax = axes[row_no, col_no]
            for case_no, case in enumerate(results[group]):
                states = np.asarray(case["states"], dtype=np.float32)
                start = np.asarray(case["start"], dtype=np.float32)
                full = np.concatenate(
                    [
                        np.full((states.shape[0], 1), float(start[idx]), dtype=np.float32),
                        states[:, :, idx],
                    ],
                    axis=1,
                )
                q10 = np.quantile(full, 0.10, axis=0)
                q50 = np.quantile(full, 0.50, axis=0)
                q90 = np.quantile(full, 0.90, axis=0)
                color = colors[case_no % len(colors)]
                ax.fill_between(days, q10, q90, color=color, alpha=0.06)
                ax.plot(days, q50, color=color, linewidth=1.25, alpha=0.9)
                y_limits[factor].extend([float(np.min(q10)), float(np.max(q90))])
            ax.grid(alpha=0.14)
            if row_no == 0:
                ax.set_title(labels[group], fontsize=10, fontweight="bold")
            if col_no == 0:
                ax.set_ylabel(f"{factor}\nraw level")
            if row_no == len(factors) - 1:
                ax.set_xlabel("Forward day")

    for row_no, factor in enumerate(factors):
        values = y_limits[factor]
        if not values:
            continue
        low = min(values)
        high = max(values)
        pad = 0.03 * max(high - low, 1e-8)
        for col_no in range(len(groups)):
            axes[row_no, col_no].set_ylim(low - pad, high + pad)
    fig.suptitle(
        "Fixed-start raw-level fan paths across caption conditions",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def _markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Full-Corpus Fixed-Start Conditionality Audit",
        "",
        f"- Status: `{decision['status']}`",
        f"- Start index: `{report['start_index']}`",
        f"- Case count: `{report['case_count']}`",
        f"- Samples per condition: `{report['samples']}`",
        "",
        "## Decision Metrics",
        "",
        "| Metric | Professional | Simple | Start-only |",
        "|---|---:|---:|---:|",
    ]
    for metric in ["mean_factor_terminal_ks", "mean_portfolio_terminal_ks", "mean_path_energy", "mean_support_jaccard"]:
        lines.append(
            "| "
            + metric
            + " | "
            + " | ".join(
                f"{float(report['group_summaries'][group][metric]):.4f}"
                for group in ["professional", "simple", "start_only"]
            )
            + " |"
        )
    lines.extend(["", "## Selected Cases", ""])
    for case in report["selected_cases"]:
        lines.append(
            f"- `{case['window_id']}`: professional `{case['professional_kind']}`, simple `{case['simple_kind']}`"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, value in report["artifact_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    bridge_report = _load_json(args.bridge_report)
    with np.load(args.bridge_arrays) as arrays:
        condition_vectors = np.asarray(arrays["condition_vectors"], dtype=np.float32)
    cases = select_diverse_caption_cases(
        bridge_report,
        condition_vectors=condition_vectors,
        case_count=int(args.case_count),
        professional_kind=str(args.professional_kind),
        simple_kind=str(args.simple_kind),
    )
    if bool(args.dry_run):
        dry_runs = []
    results = {"professional": [], "simple": [], "start_only": []}
    start_only_reference: dict[str, Any] | None = None
    for case_no, case in enumerate(cases):
        run_seed = (
            int(args.seed) + case_no * 101
            if bool(getattr(args, "vary_seed_by_case", False))
            else int(args.seed)
        )
        local_args = argparse.Namespace(**vars(args))
        local_args.seed = run_seed
        if bool(args.dry_run):
            dry_runs.extend(
                [
                    run_case_condition(
                        case=case,
                        condition="professional",
                        query_kind=str(args.professional_kind),
                        memory_prior_mode="diverse_topk_narrative_start_checked",
                        args=local_args,
                    ),
                    run_case_condition(
                        case=case,
                        condition="simple",
                        query_kind=str(args.simple_kind),
                        memory_prior_mode="diverse_topk_narrative_start_checked",
                        args=local_args,
                    ),
                ]
            )
            if start_only_reference is None:
                dry_runs.append(
                    run_case_condition(
                        case=case,
                        condition="start_only_reference",
                        query_kind=str(args.simple_kind),
                        memory_prior_mode="soft_topk_start_only",
                        args=local_args,
                    )
                )
                start_only_reference = {"case": str(case["window_id"])}
            continue
        results["professional"].append(
            run_case_condition(
                case=case,
                condition="professional",
                query_kind=str(args.professional_kind),
                memory_prior_mode="diverse_topk_narrative_start_checked",
                args=local_args,
            )
        )
        results["simple"].append(
            run_case_condition(
                case=case,
                condition="simple",
                query_kind=str(args.simple_kind),
                memory_prior_mode="diverse_topk_narrative_start_checked",
                args=local_args,
            )
        )
        if start_only_reference is None:
            start_only_reference = run_case_condition(
                case=case,
                condition="start_only_reference",
                query_kind=str(args.simple_kind),
                memory_prior_mode="soft_topk_start_only",
                args=local_args,
            )
        repeated_start_only = {
            **start_only_reference,
            "case": str(case["window_id"]),
            "condition": "start_only",
        }
        results["start_only"].append(repeated_start_only)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.dry_run):
        report = {
            "status": "dry_run",
            "selected_cases": cases,
            "commands": dry_runs,
        }
        _write_json(output_dir / "full_corpus_fixed_start_conditionality_audit_dry_run.json", report)
        return report

    group_summaries = {
        group: summarize_condition_group(rows)
        for group, rows in results.items()
    }
    decision = fixed_start_decision(
        professional_summary=group_summaries["professional"],
        simple_summary=group_summaries["simple"],
        start_only_summary=group_summaries["start_only"],
    )
    report = {
        "status": "ok",
        "scope_note": (
            "Fixed-start audit using cached full-corpus professional caption condition "
            "vectors. All cases share the same explicit starting level; no caption or "
            "embedding calls are made."
        ),
        "bridge_report": str(args.bridge_report),
        "bridge_arrays": str(args.bridge_arrays),
        "support_bank_report": str(args.support_bank_report),
        "support_bank_arrays": str(args.support_bank_arrays),
        "start_index": int(args.start_index),
        "case_count": len(cases),
        "samples": int(args.samples),
        "selected_cases": cases,
        "group_summaries": group_summaries,
        "decision": decision,
        "run_artifacts": {
            group: [
                {
                    "case": row["case"],
                    "report_path": row["report_path"],
                    "arrays_path": row["arrays_path"],
                    "support_count": row["support_count"],
                }
                for row in rows
            ]
            for group, rows in results.items()
        },
        "artifact_paths": {
            "summary_json": str(output_dir / "full_corpus_fixed_start_conditionality_audit.json"),
            "summary_markdown": str(output_dir / "full_corpus_fixed_start_conditionality_audit.md"),
            "terminal_range_plot": str(output_dir / "full_corpus_fixed_start_terminal_range.png"),
            "raw_level_fan_panel": str(output_dir / "full_corpus_fixed_start_raw_level_fan_panel.png"),
        },
    }
    _plot_terminal_ranges(report, Path(report["artifact_paths"]["terminal_range_plot"]))
    _plot_group_fan_panel(
        results,
        output=Path(report["artifact_paths"]["raw_level_fan_panel"]),
    )
    _write_json(report["artifact_paths"]["summary_json"], report)
    _write_text(report["artifact_paths"]["summary_markdown"], _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", type=Path, default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", type=Path, default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--support-bank-report", type=Path, default=DEFAULT_SUPPORT_BANK_REPORT)
    parser.add_argument("--support-bank-arrays", type=Path, default=DEFAULT_SUPPORT_BANK_ARRAYS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--professional-kind", default=PROFESSIONAL_KIND)
    parser.add_argument("--simple-kind", default=SIMPLE_KIND)
    parser.add_argument("--case-count", type=int, default=6)
    parser.add_argument("--start-index", type=int, default=22)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=957)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vary-seed-by-case",
        action="store_true",
        help=(
            "Diagnostic mode only. The default uses a common seed across fixed-start "
            "caption cases so the start-only null is actually flat."
        ),
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_audit(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report.get("decision"),
                "case_count": report.get("case_count", len(report.get("selected_cases", []))),
                "artifact_paths": report.get("artifact_paths"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
