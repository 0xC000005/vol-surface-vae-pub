#!/usr/bin/env python
"""
220n evaluation: hybrid rollout using a learned slow-path predictor and frozen 212ai.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_220m_oracle_slow_state import (
    _coerce_next_iv,
    run_custom_conditionality,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)
from experiments.backfill.block_ar.train_220n_slow_path_predictor import (
    SlowPathSeq2Seq,
    ema_surfaces,
    logit_clip,
)


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    ordered = [
        ("surface", results["surface"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("conditionality", results["conditionality"]["overall_pass"]),
        ("time_series", results["time_series"]["overall_pass"]),
        ("block_ar", results["block_ar"]["overall_pass"]),
        ("cointegration", results["cointegration"]["overall_pass"]),
        ("regime_coverage", results["regime_coverage"]["overall_pass"]),
        ("distributional_fidelity", results["distributional_fidelity"]["overall_pass"]),
        ("cross_cell_correlation", results["cross_cell_correlation"]["overall_pass"]),
        ("mean_reversion", results["mean_reversion"]["overall_pass"]),
        ("pathwise_jump_realism", results["pathwise_jump_realism"]["overall_pass"]),
    ]
    failed = [name for name, passed in ordered if not passed]
    return sum(int(passed) for _name, passed in ordered), failed


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_slow_predictor(checkpoint_path: str, device: torch.device) -> tuple[SlowPathSeq2Seq, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    model = SlowPathSeq2Seq(config["factor_dim"], config["hidden_dim"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def build_slow_histories(history_01: np.ndarray, slow_params: dict[str, np.ndarray]) -> torch.Tensor:
    alpha = float(np.asarray(slow_params["slow_alpha"]).item())
    eps = float(np.asarray(slow_params["eps"]).item())
    mu = np.asarray(slow_params["mu"], dtype=np.float64)
    basis = np.asarray(slow_params["basis"], dtype=np.float64)

    slow_hist = []
    for window in history_01:
        slow = ema_surfaces(window.astype(np.float32), alpha)
        y = logit_clip(slow.reshape(slow.shape[0], -1), eps)
        factors = (y - mu) @ basis.T
        slow_hist.append(factors)
    return torch.from_numpy(np.stack(slow_hist).astype(np.float32))


@torch.no_grad()
def predict_future_slow_surfaces(
    predictor: SlowPathSeq2Seq,
    history_01: np.ndarray,
    slow_params: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.asarray(slow_params["slow_alpha"]).item())
    mu = np.asarray(slow_params["mu"], dtype=np.float64)
    basis = np.asarray(slow_params["basis"], dtype=np.float64)

    slow_hist_factors = build_slow_histories(history_01, slow_params)
    pred_factors = []
    for start in range(0, slow_hist_factors.shape[0], batch_size):
        hist = slow_hist_factors[start : start + batch_size].to(device)
        pred = predictor(hist, future=None, teacher_forcing_ratio=0.0)
        pred_factors.append(pred.cpu().numpy())
    future_factors = np.concatenate(pred_factors, axis=0)
    future_y = mu.reshape(1, 1, -1) + future_factors @ basis
    future_slow = sigmoid(future_y).reshape(future_y.shape[0], future_y.shape[1], 5, 5).astype(np.float32)

    slow_prev = []
    for window in history_01:
        slow = ema_surfaces(window.astype(np.float32), alpha)
        slow_prev.append(slow[-1])
    return np.stack(slow_prev).astype(np.float32), future_slow


@torch.no_grad()
def hybrid_rollout_samples(
    model: torch.nn.Module,
    history_norm: torch.Tensor,
    predicted_prev_slow: np.ndarray,
    predicted_future_slow: np.ndarray,
    slow_alpha: float,
    slow_mix_start: float,
    slow_mix_end: float,
    n_samples: int,
    chunk_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    history_01 = history_norm.detach().cpu().numpy()
    batch_size, hist_len = history_01.shape[:2]
    future_len = predicted_future_slow.shape[1]
    chunk_size = max(1, min(int(chunk_size), int(n_samples)))

    all_chunks: list[np.ndarray] = []
    for start in range(0, n_samples, chunk_size):
        k = min(chunk_size, n_samples - start)
        hist_k = torch.from_numpy(history_01).to(device=device, dtype=torch.float32)
        hist_k = hist_k.unsqueeze(1).expand(batch_size, k, hist_len, 5, 5)
        hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()
        slow_prev = (
            torch.from_numpy(predicted_prev_slow)
            .to(device=device, dtype=hist_k.dtype)
            .unsqueeze(1)
            .expand(batch_size, k, 5, 5)
            .reshape(batch_size * k, 5, 5)
            .clone()
        )
        future_slow = (
            torch.from_numpy(predicted_future_slow)
            .to(device=device, dtype=hist_k.dtype)
            .unsqueeze(1)
            .expand(batch_size, k, future_len, 5, 5)
            .reshape(batch_size * k, future_len, 5, 5)
        )

        frames = []
        for step in range(future_len):
            next_iv = _coerce_next_iv(model.sample_next_iv(hist_k, n_samples=1)).squeeze(1)
            model_slow_next = (1.0 - slow_alpha) * slow_prev + slow_alpha * next_iv
            residual_next = next_iv - model_slow_next
            if future_len <= 1:
                slow_mix = slow_mix_end
            else:
                frac = step / float(future_len - 1)
                slow_mix = slow_mix_start + (slow_mix_end - slow_mix_start) * frac
            blended_slow = (1.0 - slow_mix) * model_slow_next + slow_mix * future_slow[:, step]
            hybrid_next = (blended_slow + residual_next).clamp(0.0, 1.0)
            frames.append(hybrid_next.view(batch_size, k, 5, 5))
            hist_k = torch.cat([hist_k[:, 1:], hybrid_next.unsqueeze(1)], dim=1)
            slow_prev = blended_slow
        all_chunks.append(torch.stack(frames, dim=2).cpu().numpy())
    return np.concatenate(all_chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 220n slow-path hybrid rollout")
    parser.add_argument("--base_model_type", type=str, default="212ai")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--slow_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--slow_mix_start", type=float, default=1.0)
    parser.add_argument("--slow_mix_end", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, base_payload = load_one_day_kernel(args.base_model_type, args.base_checkpoint, device)
    base_model.eval()
    slow_model, slow_payload = load_slow_predictor(args.slow_checkpoint, device)
    slow_params = slow_payload["slow_params"]
    slow_alpha = float(np.asarray(slow_params["slow_alpha"]).item())

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
    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    history_01 = batch.history_01.detach().cpu().numpy()
    slow_prev, slow_future = predict_future_slow_surfaces(
        predictor=slow_model,
        history_01=history_01,
        slow_params=slow_params,
        device=device,
        batch_size=args.batch_size,
    )
    cond_samples = hybrid_rollout_samples(
        model=base_model,
        history_norm=batch.history_norm,
        predicted_prev_slow=slow_prev,
        predicted_future_slow=slow_future,
        slow_alpha=slow_alpha,
        slow_mix_start=args.slow_mix_start,
        slow_mix_end=args.slow_mix_end,
        n_samples=args.samples,
        chunk_size=args.chunk_size,
    )

    zero_hist = np.zeros_like(history_01)
    zero_prev, zero_future = predict_future_slow_surfaces(
        predictor=slow_model,
        history_01=zero_hist,
        slow_params=slow_params,
        device=device,
        batch_size=args.batch_size,
    )
    uncond_samples = hybrid_rollout_samples(
        model=base_model,
        history_norm=torch.zeros_like(batch.history_norm),
        predicted_prev_slow=zero_prev,
        predicted_future_slow=zero_future,
        slow_alpha=slow_alpha,
        slow_mix_start=args.slow_mix_start,
        slow_mix_end=args.slow_mix_end,
        n_samples=min(args.samples, args.conditionality_samples),
        chunk_size=args.chunk_size,
    )

    ground_truth = batch.future_01.detach().cpu().numpy()

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    conditionality = run_custom_conditionality(cond_samples, uncond_samples, ground_truth, history_01)
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

    results = {
        "config": {
            "type": "220n_slow_path_hybrid",
            "base_model_type": args.base_model_type,
            "base_checkpoint": args.base_checkpoint,
            "slow_checkpoint": args.slow_checkpoint,
            "base_checkpoint_epoch": int(base_payload.get("epoch", -1)),
            "slow_checkpoint_epoch": int(slow_payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "slow_mix_start": args.slow_mix_start,
            "slow_mix_end": args.slow_mix_end,
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
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- base model: `{args.base_model_type}`",
        f"- base checkpoint: `{args.base_checkpoint}`",
        f"- slow checkpoint: `{args.slow_checkpoint}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- slow mix start/end: `{args.slow_mix_start}` / `{args.slow_mix_end}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- MR ratio h1: `{mean_reversion.get('mr_gt_ratio', float('nan')):.3f}`",
        f"- MR ratio h30: `{mean_reversion.get('full_horizon', {}).get('per_horizon', {}).get(30, {}).get('ratio', float('nan')):.3f}`",
        "",
        "**Fidelity / Structure**",
        f"- time-series ACF corr: `{time_series['acf']['acf_correlation']:.3f}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "220n Slow-Path Hybrid Evaluation", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
