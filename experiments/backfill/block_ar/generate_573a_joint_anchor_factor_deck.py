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
    StressSelectedScenarioModel,
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


def anchor_factor_columns(columns: list[str], iv_count: int = 25) -> list[str]:
    return [str(col) for col in columns[iv_count:]]


def _rounded_float(value: float) -> float:
    return round(float(value), 6)


def select_panel_factor_paths(
    *,
    panel_history: np.ndarray,
    panel_candidates: np.ndarray,
    columns: list[str],
    n_select: int,
    iv_count: int = 25,
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
    selected_indices = np.stack(
        [severity_stratified_indices(row, n_select=n_select) for row in severity],
        axis=0,
    )
    selected = []
    internal_mean = []
    for window_idx in range(coherent.shape[0]):
        idx = selected_indices[window_idx]
        selected.append(coherent[window_idx, idx, :, iv_count:])
        internal_mean.append(severity[window_idx, idx])
    return SelectedFactorPaths(
        factor_scenarios=np.stack(selected, axis=0).astype(np.float32),
        factor_columns=anchor_factor_columns(columns, iv_count=iv_count),
        selected_indices=selected_indices.astype(np.int64),
        internal_panel_path_mean_iv=np.stack(internal_mean, axis=0).astype(np.float32),
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


@torch.no_grad()
def _sample_iv_scenarios(args: argparse.Namespace, history_norm: torch.Tensor, device: torch.device) -> np.ndarray:
    base_model, _payload = load_one_day_kernel(args.iv_model_type, args.iv_checkpoint, device)
    max_native_steps = int(getattr(getattr(base_model, "cfg", None), "future_len", args.future_len))
    if args.future_len > max_native_steps:
        base_model = BlockwiseLongHorizonModel(base_model, max_block_steps=max_native_steps).eval()
    policy_model = StressSelectedScenarioModel(
        base_model,
        candidate_count=args.iv_candidate_count,
        chunk_size=args.chunk_size,
    ).eval()
    scenarios = policy_model.sample_batched(
        history_norm,
        n_samples=args.samples,
        n_steps=args.future_len,
        chunk_size=args.chunk_size,
        history_is_normalized=True,
    )
    return scenarios.squeeze(0).detach().cpu().numpy().astype(np.float32)


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
    iv_scenarios = _sample_iv_scenarios(args, history_norm, device)

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
    )
    factor_history = panel_history[0, :, 25:]
    labels = severity_bucket_labels(args.samples)
    path_mean_iv = iv_scenarios.mean(axis=(1, 2, 3)).astype(np.float32)

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
