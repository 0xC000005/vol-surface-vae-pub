#!/usr/bin/env python
"""573a: generate an IV stress deck with coherent anchor-factor overlays."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.panel_daily_cholesky_transition_model import load_model as load_panel_model  # noqa: E402
from experiments.backfill.block_ar._panel_law_535_utils import load_aligned_iv_factor_panel  # noqa: E402
from experiments.backfill.block_ar._rollout_220_utils import load_one_day_kernel  # noqa: E402
from experiments.backfill.block_ar.audit_572a_joint_panel_quality import (  # noqa: E402
    reconstruct_factor_levels_from_returns,
)
from experiments.backfill.block_ar.evaluate_564a_stress_selected_510a import (  # noqa: E402
    severity_stratified_indices,
)
from experiments.backfill.block_ar.generate_568a_risk_scenario_deck import (  # noqa: E402
    BlockwiseLongHorizonModel,
    _load_history,
    scenario_diagnostics,
    severity_bucket_labels,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


@dataclass(frozen=True)
class SelectedFactorPaths:
    factor_scenarios: np.ndarray
    factor_columns: list[str]
    selected_indices: np.ndarray
    internal_panel_path_mean_iv: np.ndarray
    internal_panel_severity_quantiles: np.ndarray
    pairing_policy: str


def anchor_factor_columns(columns: list[str], iv_count: int = 25) -> list[str]:
    return [str(col) for col in columns[iv_count:]]


def _rounded_float(value: float) -> float:
    return round(float(value), 6)


def _nearest_unused_rank(target_rank: int, n_candidates: int, used: set[int]) -> int:
    target_rank = int(np.clip(target_rank, 0, n_candidates - 1))
    for radius in range(n_candidates):
        lo = target_rank - radius
        hi = target_rank + radius
        if lo >= 0 and lo not in used:
            return lo
        if hi < n_candidates and hi not in used:
            return hi
    raise RuntimeError("no unused candidate rank available")


def rank_matched_indices(candidate_severity: np.ndarray, target_severity: np.ndarray) -> np.ndarray:
    """Match selected IV scenario severity ranks to factor-candidate severity ranks."""
    candidate = np.asarray(candidate_severity, dtype=np.float64).reshape(-1)
    target = np.asarray(target_severity, dtype=np.float64).reshape(-1)
    if candidate.size == 0 or target.size == 0:
        raise ValueError("candidate and target severities must be non-empty")
    if target.size > candidate.size:
        raise ValueError("target size cannot exceed candidate size")
    candidate_order = np.argsort(candidate, kind="mergesort")
    target_order = np.argsort(target, kind="mergesort")
    target_position = np.empty(target.size, dtype=np.int64)
    target_position[target_order] = np.arange(target.size, dtype=np.int64)
    denom = max(target.size - 1, 1)
    used_ranks: set[int] = set()
    selected = []
    for pos in target_position:
        raw_rank = int(np.rint((int(pos) / denom) * (candidate.size - 1)))
        candidate_rank = _nearest_unused_rank(raw_rank, candidate.size, used_ranks)
        used_ranks.add(candidate_rank)
        selected.append(int(candidate_order[candidate_rank]))
    return np.asarray(selected, dtype=np.int64)


def quantile_matched_indices(candidate_severity: np.ndarray, target_quantiles: np.ndarray) -> np.ndarray:
    """Select candidate indices at the same severity quantiles as selected IV paths."""
    candidate = np.asarray(candidate_severity, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(target_quantiles, dtype=np.float64).reshape(-1)
    if candidate.size == 0 or quantiles.size == 0:
        raise ValueError("candidate severity and target quantiles must be non-empty")
    if quantiles.size > candidate.size:
        raise ValueError("target quantile count cannot exceed candidate count")
    candidate_order = np.argsort(candidate, kind="mergesort")
    used_ranks: set[int] = set()
    selected = []
    for q in quantiles:
        raw_rank = int(np.rint(float(np.clip(q, 0.0, 1.0)) * (candidate.size - 1)))
        candidate_rank = _nearest_unused_rank(raw_rank, candidate.size, used_ranks)
        used_ranks.add(candidate_rank)
        selected.append(int(candidate_order[candidate_rank]))
    return np.asarray(selected, dtype=np.int64)


def select_panel_factor_paths(
    *,
    panel_history: np.ndarray,
    panel_candidates: np.ndarray,
    columns: list[str],
    n_select: int,
    iv_count: int = 25,
    target_iv_path_mean: np.ndarray | None = None,
    target_iv_severity_quantiles: np.ndarray | None = None,
) -> SelectedFactorPaths:
    """Select factor overlays from panel candidates using internal IV severity.

    The panel model produces both IV and factor channels. For the joint stress deck
    we keep the stronger 510a/568a IV scenarios, but select factor paths from the
    panel model with the same calm/central/stress severity policy measured on the
    panel model's internal IV channels.
    """
    candidates = np.asarray(panel_candidates, dtype=np.float32)
    history = np.asarray(panel_history, dtype=np.float32)
    if candidates.ndim != 4:
        raise ValueError("panel_candidates must have shape (windows, candidates, horizon, vars)")
    if history.ndim != 3:
        raise ValueError("panel_history must have shape (windows, history, vars)")
    if candidates.shape[0] != history.shape[0] or candidates.shape[-1] != history.shape[-1]:
        raise ValueError(f"shape mismatch: history={history.shape}, candidates={candidates.shape}")
    if n_select > candidates.shape[1]:
        raise ValueError("n_select cannot exceed candidate count")

    coherent = reconstruct_factor_levels_from_returns(
        history,
        candidates,
        columns=columns,
        iv_count=iv_count,
    )
    severity = coherent[..., :iv_count].mean(axis=(2, 3))
    if target_iv_path_mean is not None and target_iv_severity_quantiles is not None:
        raise ValueError("Provide only one of target_iv_path_mean or target_iv_severity_quantiles")
    if target_iv_severity_quantiles is not None:
        target_q = np.asarray(target_iv_severity_quantiles, dtype=np.float64)
        if target_q.ndim == 1 and severity.shape[0] == 1:
            target_q = target_q[None, :]
        if target_q.shape != (severity.shape[0], n_select):
            raise ValueError(
                f"target_iv_severity_quantiles must have shape {(severity.shape[0], n_select)}"
            )
        selected_indices = np.stack(
            [quantile_matched_indices(severity[i], target_q[i]) for i in range(severity.shape[0])],
            axis=0,
        )
        pairing_policy = "candidate_quantile_matched_to_selected_iv_severity"
    elif target_iv_path_mean is None:
        selected_indices = np.stack(
            [severity_stratified_indices(row, n_select=n_select) for row in severity],
            axis=0,
        )
        pairing_policy = "severity_stratified_by_internal_panel_iv"
    else:
        target = np.asarray(target_iv_path_mean, dtype=np.float64)
        if target.ndim == 1 and severity.shape[0] == 1:
            target = target[None, :]
        if target.shape != (severity.shape[0], n_select):
            raise ValueError(f"target_iv_path_mean must have shape {(severity.shape[0], n_select)}")
        selected_indices = np.stack(
            [rank_matched_indices(severity[i], target[i]) for i in range(severity.shape[0])],
            axis=0,
        )
        pairing_policy = "rank_matched_to_selected_iv_path_severity"
    selected = []
    internal_mean = []
    internal_quantiles = []
    for window_idx in range(coherent.shape[0]):
        idx = selected_indices[window_idx]
        order = np.argsort(severity[window_idx], kind="mergesort")
        rank_by_index = np.empty(severity.shape[1], dtype=np.float32)
        rank_by_index[order] = np.arange(severity.shape[1], dtype=np.float32)
        rank_by_index = rank_by_index / float(max(severity.shape[1] - 1, 1))
        selected.append(coherent[window_idx, idx, :, iv_count:])
        internal_mean.append(severity[window_idx, idx])
        internal_quantiles.append(rank_by_index[idx])
    return SelectedFactorPaths(
        factor_scenarios=np.stack(selected, axis=0).astype(np.float32),
        factor_columns=anchor_factor_columns(columns, iv_count=iv_count),
        selected_indices=selected_indices.astype(np.int64),
        internal_panel_path_mean_iv=np.stack(internal_mean, axis=0).astype(np.float32),
        internal_panel_severity_quantiles=np.stack(internal_quantiles, axis=0).astype(np.float32),
        pairing_policy=pairing_policy,
    )


def build_joint_manifest(
    *,
    history_start_index: int,
    history_end_index: int,
    history_len: int,
    future_len: int,
    samples: int,
    iv_candidate_count: int,
    factor_candidate_count: int,
    seed: int,
    iv_scenario_shape: tuple[int, ...],
    factor_scenario_shape: tuple[int, ...],
    factor_columns: list[str],
    output_npz: str,
    iv_evidence_path: str,
    factor_evidence_path: str,
    scenario_diag: dict[str, float] | None = None,
    factor_diag: dict[str, float] | None = None,
    pairing_diag: dict[str, float] | None = None,
    factor_pairing_policy: str = "rank_matched_to_selected_iv_path_severity",
) -> dict[str, Any]:
    labels = severity_bucket_labels(samples)
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "system": "573a_joint_anchor_factor_stress_deck",
        "iv_base_system": "568a_564a_risk_scenario_deck",
        "iv_base_law": "510a",
        "factor_base_law": "537a_panel_daily_cholesky_transition",
        "policy": "severity_stratified_selection",
        "severity_metric": "average_future_iv_level",
        "probability_interpretation": "stress_scenario_set_not_calibrated_law",
        "history_start_index": int(history_start_index),
        "history_end_index": int(history_end_index),
        "history_len": int(history_len),
        "future_len": int(future_len),
        "samples": int(samples),
        "iv_candidate_count": int(iv_candidate_count),
        "factor_candidate_count": int(factor_candidate_count),
        "seed": int(seed),
        "bucket_counts": {str(k): int(v) for k, v in zip(unique, counts, strict=True)},
        "iv_scenario_shape": [int(x) for x in iv_scenario_shape],
        "factor_scenario_shape": [int(x) for x in factor_scenario_shape],
        "factor_columns": list(factor_columns),
        "factor_count": int(len(factor_columns)),
        "factor_level_policy": "deterministically_reconstruct_levels_from_generated_returns_or_diffs",
        "factor_pairing_policy": factor_pairing_policy,
        "iv_risk_contract": {
            "risk_manager_acceptable": True,
            "evidence": iv_evidence_path,
            "scope": "IV-surface conditional stress review, not calibrated probabilities",
            "required_caveats": [
                "selected scenario frequencies are policy-balanced, not probabilities",
                "level-frequency KS remains a warning, not a stress blocker",
                "regime layer2 and cointegration near-miss caveats remain disclosed",
            ],
        },
        "joint_anchor_factor_contract": {
            "risk_manager_acceptable": True,
            "evidence": factor_evidence_path,
            "scope": "anchor-factor stress overlays coherent with generated increments",
            "required_caveats": [
                "factor overlays are generated by the panel law and severity-aligned, not a fully calibrated joint probability law with the IV deck",
                "use for scenario review, not capital-model probability weights",
            ],
        },
        "scenario_diagnostics": scenario_diag,
        "factor_diagnostics": factor_diag,
        "pairing_diagnostics": pairing_diag,
        "output_npz": output_npz,
    }


def factor_diagnostics(factors: np.ndarray) -> dict[str, float]:
    arr = np.asarray(factors, dtype=np.float32)
    finite = np.isfinite(arr)
    return {
        "finite_rate": _rounded_float(finite.mean()),
        "min_factor_value": _rounded_float(np.nanmin(arr)),
        "max_factor_value": _rounded_float(np.nanmax(arr)),
        "terminal_mean_factor_value": _rounded_float(np.nanmean(arr[:, -1])),
    }


def pairing_diagnostics(iv_quantiles: np.ndarray, factor_quantiles: np.ndarray) -> dict[str, float]:
    iv_q = np.asarray(iv_quantiles, dtype=np.float64).reshape(-1)
    factor_q = np.asarray(factor_quantiles, dtype=np.float64).reshape(-1)
    if iv_q.shape != factor_q.shape:
        raise ValueError(f"quantile shape mismatch: {iv_q.shape} vs {factor_q.shape}")
    err = np.abs(iv_q - factor_q)
    return {
        "mean_abs_quantile_error": _rounded_float(np.mean(err)),
        "max_abs_quantile_error": _rounded_float(np.max(err)),
        "iv_quantile_min": _rounded_float(np.min(iv_q)),
        "iv_quantile_max": _rounded_float(np.max(iv_q)),
        "factor_quantile_min": _rounded_float(np.min(factor_q)),
        "factor_quantile_max": _rounded_float(np.max(factor_q)),
    }


@torch.no_grad()
def _sample_iv_scenarios_and_quantiles(
    args: argparse.Namespace,
    history_norm: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    base_model, _payload = load_one_day_kernel(args.iv_model_type, args.iv_checkpoint, device)
    max_native_steps = int(getattr(getattr(base_model, "cfg", None), "future_len", args.future_len))
    if args.future_len > max_native_steps:
        base_model = BlockwiseLongHorizonModel(base_model, max_block_steps=max_native_steps).eval()
    candidate_count = max(int(args.iv_candidate_count), int(args.samples))
    candidates = base_model.sample_batched(
        history_norm,
        n_samples=candidate_count,
        n_steps=args.future_len,
        chunk_size=args.chunk_size,
        history_is_normalized=True,
    )
    candidates_np = candidates.detach().cpu().numpy().astype(np.float32)
    severity = candidates_np.mean(axis=(2, 3, 4))
    selected = []
    selected_quantiles = []
    for window_idx in range(candidates_np.shape[0]):
        indices = severity_stratified_indices(severity[window_idx], n_select=int(args.samples))
        order = np.argsort(severity[window_idx], kind="mergesort")
        rank_by_index = np.empty(candidate_count, dtype=np.float64)
        rank_by_index[order] = np.arange(candidate_count, dtype=np.float64)
        denom = max(candidate_count - 1, 1)
        selected.append(candidates_np[window_idx, indices])
        selected_quantiles.append(rank_by_index[indices] / float(denom))
    return (
        np.stack(selected, axis=0).squeeze(0).astype(np.float32),
        np.stack(selected_quantiles, axis=0).squeeze(0).astype(np.float32),
    )


@torch.no_grad()
def _sample_iv_scenarios(args: argparse.Namespace, history_norm: torch.Tensor, device: torch.device) -> np.ndarray:
    scenarios, _quantiles = _sample_iv_scenarios_and_quantiles(args, history_norm, device)
    return scenarios


@torch.no_grad()
def _sample_factor_candidates(
    args: argparse.Namespace,
    panel_history: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    panel_model, payload = load_panel_model(args.factor_checkpoint, device)
    if int(payload["config"]["future_len"]) != int(args.future_len):
        raise ValueError(
            f"factor checkpoint future_len={payload['config']['future_len']} does not match requested {args.future_len}"
        )
    history_tensor = torch.from_numpy(panel_history).to(device)
    candidates = panel_model.sample_batched(
        history_tensor,
        n_samples=args.factor_candidate_count,
        n_steps=args.future_len,
        chunk_size=args.chunk_size,
    )
    return candidates.detach().cpu().numpy().astype(np.float32), list(payload["panel_columns"])


@torch.no_grad()
def generate_joint_deck(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    history_01, history_start, history_end = _load_history(
        args.data_path,
        args.history_len,
        args.history_end_index,
    )
    history_tensor = torch.from_numpy(history_01).unsqueeze(0).to(device)
    history_norm = normalize_iv(history_tensor)
    iv_scenarios, iv_severity_quantiles = _sample_iv_scenarios_and_quantiles(args, history_norm, device)
    path_mean_iv = iv_scenarios.mean(axis=(1, 2, 3)).astype(np.float32)

    panel, columns, _dates = load_aligned_iv_factor_panel()
    panel_history = panel[history_start:history_end][None].astype(np.float32)
    if panel_history.shape[1] != args.history_len:
        raise RuntimeError(f"panel history shape mismatch: {panel_history.shape}")
    panel_candidates, panel_columns = _sample_factor_candidates(args, panel_history, device)
    if panel_columns != columns:
        raise RuntimeError("factor checkpoint columns do not match loaded panel columns")
    selected_factors = select_panel_factor_paths(
        panel_history=panel_history,
        panel_candidates=panel_candidates,
        columns=columns,
        n_select=args.samples,
        iv_count=25,
        target_iv_severity_quantiles=iv_severity_quantiles[None],
    )
    factor_history = panel_history[0, :, 25:]
    labels = severity_bucket_labels(args.samples)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        iv_history=history_01.astype(np.float32),
        iv_scenarios=iv_scenarios.astype(np.float32),
        factor_history=factor_history.astype(np.float32),
        factor_scenarios=selected_factors.factor_scenarios[0].astype(np.float32),
        factor_columns=np.asarray(selected_factors.factor_columns, dtype="U64"),
        bucket_labels=labels,
        path_mean_iv=path_mean_iv,
        selected_factor_candidate_indices=selected_factors.selected_indices[0].astype(np.int64),
        factor_internal_panel_path_mean_iv=selected_factors.internal_panel_path_mean_iv[0].astype(np.float32),
        iv_selected_severity_quantiles=iv_severity_quantiles.astype(np.float32),
        factor_selected_severity_quantiles=selected_factors.internal_panel_severity_quantiles[0].astype(
            np.float32
        ),
    )

    manifest = build_joint_manifest(
        history_start_index=history_start,
        history_end_index=history_end,
        history_len=args.history_len,
        future_len=args.future_len,
        samples=args.samples,
        iv_candidate_count=args.iv_candidate_count,
        factor_candidate_count=args.factor_candidate_count,
        seed=args.seed,
        iv_scenario_shape=tuple(iv_scenarios.shape),
        factor_scenario_shape=tuple(selected_factors.factor_scenarios[0].shape),
        factor_columns=selected_factors.factor_columns,
        output_npz=str(output_npz),
        iv_evidence_path=args.iv_evidence_path,
        factor_evidence_path=args.factor_evidence_path,
        scenario_diag=scenario_diagnostics(iv_scenarios),
        factor_diag=factor_diagnostics(selected_factors.factor_scenarios[0]),
        pairing_diag=pairing_diagnostics(
            iv_severity_quantiles,
            selected_factors.internal_panel_severity_quantiles[0],
        ),
        factor_pairing_policy=selected_factors.pairing_policy,
    )
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iv_model_type", default="340c")
    parser.add_argument(
        "--iv_checkpoint",
        default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt",
    )
    parser.add_argument(
        "--factor_checkpoint",
        default="models/backfill/537a_panel_daily_cholesky_transition_s537/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--history_end_index", type=int, default=None)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--iv_candidate_count", type=int, default=192)
    parser.add_argument("--factor_candidate_count", type=int, default=192)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=573)
    parser.add_argument(
        "--iv_evidence_path",
        default="experiments/backfill/block_ar/REPORT_567a_564a_risk_manager_deployability_package.md",
    )
    parser.add_argument(
        "--factor_evidence_path",
        default="results/autoresearch/572e_537a_joint_panel_reconstructed_quality/quality.json",
    )
    parser.add_argument(
        "--output_npz",
        default="results/autoresearch/573a_joint_anchor_factor_deck/joint_anchor_factor_deck.npz",
    )
    parser.add_argument(
        "--output_manifest",
        default="results/autoresearch/573a_joint_anchor_factor_deck/manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    manifest = generate_joint_deck(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
