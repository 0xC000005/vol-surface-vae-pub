#!/usr/bin/env python
"""583a: richer sample-quality audit for unified increment-flow checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.train_577a_unified_increment_flow import (  # noqa: E402
    UnifiedIncrementFlow,
    UnifiedIncrementFlowConfig,
    _make_train_val_blocks,
    _reconstruct_samples,
    _standardize,
    normal_score_transform,
)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def effective_rank(paths: np.ndarray) -> float:
    """Entropy effective rank for flattened generated paths."""
    flat = np.asarray(paths, dtype=np.float64).reshape(-1, paths.shape[-2] * paths.shape[-1])
    if flat.shape[0] < 3:
        return float("nan")
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = (flat @ flat.T) / max(flat.shape[1] - 1, 1)
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 1e-12:
        return 0.0
    p = eig / total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    return float(np.exp(entropy))


def summarize_samples(
    samples_state: np.ndarray,
    samples_increment: np.ndarray,
    *,
    gt_state: np.ndarray,
    history_state: np.ndarray,
    train_state: np.ndarray,
    label: str,
) -> dict[str, Any]:
    samples = np.asarray(samples_state, dtype=np.float64)
    increments = np.asarray(samples_increment, dtype=np.float64)
    gt = np.asarray(gt_state, dtype=np.float64)
    hist = np.asarray(history_state, dtype=np.float64)
    train = np.asarray(train_state, dtype=np.float64)

    iv = samples[..., :25]
    factor = samples[..., 25:]
    gt_iv = gt[..., :25]
    gt_factor = gt[..., 25:]
    train_iv = train[..., :25]
    train_factor = train[..., 25:]

    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    coverage = (gt >= q05) & (gt <= q95)
    mean_state = np.mean(samples, axis=1)
    gt_iv_move = gt[:, -1, :25].mean(axis=1) - hist[:, -1, :25].mean(axis=1)
    gen_iv_move = mean_state[:, -1, :25].mean(axis=1) - hist[:, -1, :25].mean(axis=1)
    gt_factor_move = gt[:, -1, 25:].mean(axis=1) - hist[:, -1, 25:].mean(axis=1)
    gen_factor_move = mean_state[:, -1, 25:].mean(axis=1) - hist[:, -1, 25:].mean(axis=1)

    return {
        "label": label,
        "shape": list(samples.shape),
        "finite_state_rate": float(np.isfinite(samples).mean()),
        "finite_increment_rate": float(np.isfinite(increments).mean()),
        "iv_min": float(np.nanmin(iv)),
        "iv_q001": float(np.nanquantile(iv, 0.001)),
        "iv_q999": float(np.nanquantile(iv, 0.999)),
        "iv_max": float(np.nanmax(iv)),
        "factor_min": float(np.nanmin(factor)),
        "factor_q001": float(np.nanquantile(factor, 0.001)),
        "factor_q999": float(np.nanquantile(factor, 0.999)),
        "factor_max": float(np.nanmax(factor)),
        "train_iv_min": float(np.nanmin(train_iv)),
        "train_iv_max": float(np.nanmax(train_iv)),
        "train_factor_min": float(np.nanmin(train_factor)),
        "train_factor_max": float(np.nanmax(train_factor)),
        "iv_above_train_max_rate": float((iv > np.nanmax(train_iv)).mean()),
        "iv_below_train_min_rate": float((iv < np.nanmin(train_iv)).mean()),
        "factor_above_train_max_rate": float((factor > np.nanmax(train_factor)).mean()),
        "factor_below_train_min_rate": float((factor < np.nanmin(train_factor)).mean()),
        "iv_90_interval_coverage": float(coverage[..., :25].mean()),
        "factor_90_interval_coverage": float(coverage[..., 25:].mean()),
        "state_90_interval_coverage": float(coverage.mean()),
        "iv_mean_abs_error": float(np.mean(np.abs(mean_state[..., :25] - gt_iv))),
        "factor_mean_abs_error": float(np.mean(np.abs(mean_state[..., 25:] - gt_factor))),
        "iv_sample_std_to_gt_std": float(np.std(iv) / max(float(np.std(gt_iv)), 1e-12)),
        "factor_sample_std_to_gt_std": float(np.std(factor) / max(float(np.std(gt_factor)), 1e-12)),
        "increment_std": float(np.std(increments)),
        "increment_effective_rank": effective_rank(increments),
        "iv_endpoint_move_corr": _safe_corr(gen_iv_move, gt_iv_move),
        "factor_endpoint_move_corr": _safe_corr(gen_factor_move, gt_factor_move),
        "iv_endpoint_move_mae": float(np.mean(np.abs(gen_iv_move - gt_iv_move))),
        "factor_endpoint_move_mae": float(np.mean(np.abs(gen_factor_move - gt_factor_move))),
    }


def _prepare_blocks_and_tensors(args: argparse.Namespace, payload: dict[str, Any], device: torch.device):
    cfg = payload["config"]
    block_args = argparse.Namespace(
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        iv_count=int(args.iv_count),
        max_train_windows=int(args.max_train_windows),
        clean_nonpositive_log_levels=bool(args.clean_nonpositive_log_levels),
    )
    _panel, _columns, train_block, val_block = _make_train_val_blocks(block_args)
    hist_mean = payload["history_mean"]
    hist_std = payload["history_std"]
    inc_mean = payload["increment_mean"]
    inc_std = payload["increment_std"]
    increment_transform = payload.get("increment_transform", "standard")
    inc_quantiles = payload.get("increment_quantiles")
    normal_levels = payload.get("normal_score_levels")

    train_hist = torch.from_numpy(_standardize(train_block.history_state, hist_mean, hist_std)).to(device)
    val_hist = torch.from_numpy(_standardize(val_block.history_state, hist_mean, hist_std)).to(device)
    if increment_transform == "standard":
        train_inc_np = _standardize(train_block.future_increment, inc_mean, inc_std)
    elif increment_transform == "normal_score":
        train_inc_np = normal_score_transform(train_block.future_increment, inc_quantiles, normal_levels)
    else:
        raise ValueError(f"unknown increment_transform {increment_transform}")
    train_inc = torch.from_numpy(train_inc_np).to(device)
    return train_block, val_block, train_hist, val_hist, train_inc


def load_model_for_audit(
    checkpoint: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[UnifiedIncrementFlow, dict[str, Any], Any, Any, torch.Tensor]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    train_block, val_block, train_hist, val_hist, train_inc = _prepare_blocks_and_tensors(args, payload, device)
    cfg = UnifiedIncrementFlowConfig(**payload["config"])
    model = UnifiedIncrementFlow(cfg).to(device)
    if cfg.source_mode in {"empirical_path", "empirical_conditional"}:
        bank = train_inc.reshape(train_inc.shape[0], -1)
        keys = train_hist[:, -1, :] if cfg.source_mode == "empirical_conditional" else None
        model.set_source_bank(bank, keys)
    elif cfg.source_mode == "path_gaussian":
        model.set_source_gaussian(
            payload["model_state_dict"]["source_mean"],
            payload["model_state_dict"]["source_cholesky"],
        )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload, train_block, val_block, val_hist


@torch.no_grad()
def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload, train_block, val_block, val_hist = load_model_for_audit(
        args.checkpoint,
        args=args,
        device=device,
    )
    n = min(int(args.sample_windows), int(val_hist.shape[0]))
    hist = val_hist[:n]
    repeated = hist.repeat_interleave(int(args.n_samples), dim=0)
    source_inc = model.draw_source(
        n * int(args.n_samples),
        device=device,
        dtype=hist.dtype,
        history=repeated,
    ).reshape(n, int(args.n_samples), model.cfg.future_len, model.cfg.n_vars)
    post_inc = model.sample(hist, n_samples=int(args.n_samples), n_steps=int(args.sample_steps))

    common = {
        "increment_transform": payload.get("increment_transform", "standard"),
        "inc_mean": payload["increment_mean"],
        "inc_std": payload["increment_std"],
        "inc_quantiles": payload.get("increment_quantiles"),
        "normal_levels": payload.get("normal_score_levels"),
        "specs": val_block.specs,
    }
    source_state = _reconstruct_samples(
        val_block.history_state[:n],
        source_inc.detach().cpu().numpy(),
        **common,
    )
    post_state = _reconstruct_samples(
        val_block.history_state[:n],
        post_inc.detach().cpu().numpy(),
        **common,
    )
    train_state = np.concatenate([train_block.history_state, train_block.future_state], axis=1)
    source_summary = summarize_samples(
        source_state,
        source_inc.detach().cpu().numpy(),
        gt_state=val_block.future_state[:n],
        history_state=val_block.history_state[:n],
        train_state=train_state,
        label="source_only",
    )
    post_summary = summarize_samples(
        post_state,
        post_inc.detach().cpu().numpy(),
        gt_state=val_block.future_state[:n],
        history_state=val_block.history_state[:n],
        train_state=train_state,
        label="post_flow",
    )
    return {
        "scope": "583a_unified_flow_sample_quality_audit",
        "checkpoint": args.checkpoint,
        "sample_windows": int(n),
        "n_samples": int(args.n_samples),
        "sample_steps": int(args.sample_steps),
        "seed": int(args.seed),
        "config": payload["config"],
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_best_val": float(payload["best_val"]),
        "source_only": source_summary,
        "post_flow": post_summary,
        "flow_delta": {
            "iv_max_delta": float(post_summary["iv_max"] - source_summary["iv_max"]),
            "factor_max_delta": float(post_summary["factor_max"] - source_summary["factor_max"]),
            "iv_coverage_delta": float(
                post_summary["iv_90_interval_coverage"] - source_summary["iv_90_interval_coverage"]
            ),
            "factor_coverage_delta": float(
                post_summary["factor_90_interval_coverage"] - source_summary["factor_90_interval_coverage"]
            ),
            "increment_rank_delta": float(
                post_summary["increment_effective_rank"] - source_summary["increment_effective_rank"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/582c_clean_conditional_empirical_source_top4_flow_s586/best_model.pt",
    )
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--sample_windows", type=int, default=441)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=583)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = run_audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(report["post_flow"]), indent=2))
    print(json.dumps(make_serializable(report["flow_delta"]), indent=2))


if __name__ == "__main__":
    main()
