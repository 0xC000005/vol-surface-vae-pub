#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    HistoryFutureDictDataset,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (
    suite_summary,
)


def severity_stratified_indices(severity: np.ndarray, n_select: int) -> np.ndarray:
    severity = np.asarray(severity, dtype=np.float64).reshape(-1)
    if severity.size == 0:
        raise ValueError("severity must be non-empty")
    if n_select <= 0:
        raise ValueError("n_select must be positive")
    if n_select > severity.size:
        raise ValueError("n_select cannot exceed number of candidates")

    order = np.argsort(severity, kind="mergesort")
    n_low = n_select // 3
    n_mid = n_select // 3
    n_high = n_select - n_low - n_mid
    quantiles = np.concatenate(
        [
            np.linspace(0.00, 0.15, max(1, n_low), endpoint=True),
            np.linspace(0.45, 0.55, max(1, n_mid), endpoint=True),
            np.linspace(0.85, 1.00, max(1, n_high), endpoint=True),
        ]
    )
    ranks = np.rint(quantiles * (severity.size - 1)).astype(np.int64)

    selected: list[int] = []
    used: set[int] = set()
    for rank in ranks:
        idx = int(order[int(np.clip(rank, 0, severity.size - 1))])
        if idx not in used:
            selected.append(idx)
            used.add(idx)
    if len(selected) < n_select:
        for idx in order:
            idx_int = int(idx)
            if idx_int not in used:
                selected.append(idx_int)
                used.add(idx_int)
            if len(selected) == n_select:
                break
    return np.asarray(selected[:n_select], dtype=np.int64)


def select_severity_stratified_paths(
    candidates: np.ndarray,
    n_select: int,
) -> np.ndarray:
    if candidates.ndim != 5:
        raise ValueError(
            "candidates must have shape (windows, candidates, horizon, height, width)"
        )
    selected = []
    severity = candidates.mean(axis=(2, 3, 4))
    for window_idx in range(candidates.shape[0]):
        indices = severity_stratified_indices(severity[window_idx], n_select)
        selected.append(candidates[window_idx, indices])
    return np.stack(selected, axis=0)


class StressSelectedScenarioModel(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        candidate_count: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.candidate_count = int(candidate_count)
        self.chunk_size = int(chunk_size)

    def eval(self) -> "StressSelectedScenarioModel":
        self.base_model.eval()
        return self

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int | None = None,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        candidate_count = max(int(self.candidate_count), int(n_samples))
        raw = self.base_model.sample_batched(
            history,
            n_samples=candidate_count,
            n_steps=n_steps,
            chunk_size=max(1, int(chunk_size or self.chunk_size)),
            history_is_normalized=history_is_normalized,
            **kwargs,
        )
        selected = select_severity_stratified_paths(
            raw.detach().cpu().numpy(),
            n_select=int(n_samples),
        )
        return torch.as_tensor(selected, device=history.device, dtype=raw.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="564a conservative stress-selected scenario policy around a base 510a law"
    )
    parser.add_argument("--model_type", type=str, default="340c")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--candidate_count", type=int, default=192)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=564)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    policy_model = StressSelectedScenarioModel(
        base_model,
        candidate_count=args.candidate_count,
        chunk_size=args.chunk_size,
    ).eval()

    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )

    from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv

    outputs = []
    for start in range(0, batch.history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, batch.history_01.shape[0])
        hist_batch = normalize_iv(batch.history_01[start:end])
        with torch.no_grad():
            samp = policy_model.sample_batched(
                hist_batch,
                n_samples=args.samples,
                n_steps=batch.future_01.shape[1],
                chunk_size=args.chunk_size,
            )
        outputs.append(samp.cpu().numpy())
    cond_samples = np.concatenate(outputs, axis=0)
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)

    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    conditionality = run_conditionality_tests(
        policy_model,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    cointegration = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    regime_coverage = run_regime_coverage_tests(cond_samples, ground_truth, history_01)
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results: dict[str, Any] = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "candidate_count": args.candidate_count,
            "conditionality_samples": args.conditionality_samples,
            "policy": "severity_stratified_selection",
            "probability_interpretation": "stress_scenario_set_not_calibrated_law",
            "rollout_start": int(rollout_start),
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "regime_coverage": regime_coverage,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {
        "n_pass": n_pass,
        "n_total": 11,
        "failed_suites": failed,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- policy: severity-stratified scenario selection",
        f"- candidates per window: `{args.candidate_count}`",
        f"- selected samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Risk Interpretation**",
        "- selected paths are a conservative stress scenario set, not calibrated probabilities",
        "",
        "**Key Metrics**",
        f"- overall cov90: `{coverage['overall'].get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- regime coverage overall: `{regime_coverage['overall_pass']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- cointegration gen/GT ratio: `{cointegration.get('gen_gt_ratio', float('nan')):.3f}`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- mean-reversion ratio h1: `{mean_reversion.get('mr_gt_ratio', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(
        args.output_md,
        "564a Stress-Selected 510a Scenario Policy",
        lines,
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
