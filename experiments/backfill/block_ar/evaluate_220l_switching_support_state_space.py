#!/usr/bin/env python
"""
220l: switching support-aware latent state-space rollout built around frozen 212ai.

This extends 220k by making the slow state multi-modal:
- fit a bounded slow state-space in logit space
- infer a 3-state regime process from slow move intensity
- fit regime-conditioned factor dynamics and residual scale
- preserve the frozen 212ai fast residual around the switched slow path
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


def logit_clip(x: np.ndarray, eps: float) -> np.ndarray:
    x_clip = np.clip(x, eps, 1.0 - eps)
    return np.log(x_clip / (1.0 - x_clip))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ema_surfaces(surfaces: np.ndarray, alpha: float) -> np.ndarray:
    slow = np.empty_like(surfaces)
    slow[0] = surfaces[0]
    for t in range(1, surfaces.shape[0]):
        slow[t] = (1.0 - alpha) * slow[t - 1] + alpha * surfaces[t]
    return slow


def label_regimes(score: np.ndarray, q20: float, q80: float) -> np.ndarray:
    labels = np.ones_like(score, dtype=np.int64)
    labels[score <= q20] = 0
    labels[score >= q80] = 2
    return labels


def _solve_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def fit_switching_support_state_space(
    surfaces: np.ndarray,
    fit_end_idx: int,
    slow_alpha: float,
    factor_dim: int,
    eps: float,
    ridge: float,
) -> dict[str, np.ndarray]:
    slow = ema_surfaces(surfaces[:fit_end_idx], slow_alpha)
    y_slow = logit_clip(slow.reshape(slow.shape[0], -1), eps)
    mu = y_slow.mean(axis=0, keepdims=True)
    centered = y_slow - mu

    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:factor_dim].copy()
    factors = centered @ basis.T

    slow_move_score = np.abs(np.diff(slow, axis=0)).mean(axis=(1, 2))
    q20 = float(np.quantile(slow_move_score, 0.20))
    q80 = float(np.quantile(slow_move_score, 0.80))
    regimes = label_regimes(slow_move_score, q20, q80)

    transition_counts = np.ones((3, 3), dtype=np.float64) * 0.5
    for a, b in zip(regimes[:-1], regimes[1:]):
        transition_counts[int(a), int(b)] += 1.0
    transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    y_full = logit_clip(surfaces[1:fit_end_idx].reshape(slow.shape[0] - 1, -1), eps)
    fast_norm = np.mean(np.abs(y_full - y_slow[1:]), axis=1)

    regime_intercepts = []
    regime_mats = []
    regime_covs = []
    regime_scale_refs = []
    regime_scale_betas = []
    regime_fast_q95 = []

    default_x = np.concatenate([np.ones((factors.shape[0] - 1, 1)), factors[:-1]], axis=1)
    default_beta = _solve_ridge(default_x, factors[1:], ridge)
    default_intercept = default_beta[0]
    default_mat = default_beta[1:].T
    default_resid = factors[1:] - (default_x @ default_beta)
    default_cov = np.cov(default_resid, rowvar=False)
    if factor_dim == 1:
        default_cov = np.array([[float(default_cov)]], dtype=np.float64)
    default_cov = default_cov + 1e-6 * np.eye(factor_dim, dtype=np.float64)
    default_scale_x = np.concatenate([np.ones((factors.shape[0] - 1, 1)), np.abs(factors[:-1])], axis=1)
    default_scale_beta = _solve_ridge(
        default_scale_x,
        np.log(np.maximum(fast_norm, 1e-6)),
        ridge,
    )
    default_scale_ref = float(np.exp(np.mean(default_scale_x @ default_scale_beta)))
    default_fast_q95 = float(np.quantile(fast_norm, 0.95))

    for regime in range(3):
        mask = regimes == regime
        if int(mask.sum()) < max(32, factor_dim * 6):
            regime_intercepts.append(default_intercept.copy())
            regime_mats.append(default_mat.copy())
            regime_covs.append(default_cov.copy())
            regime_scale_refs.append(default_scale_ref)
            regime_scale_betas.append(default_scale_beta.copy())
            regime_fast_q95.append(default_fast_q95)
            continue

        x = np.concatenate([np.ones((int(mask.sum()), 1)), factors[:-1][mask]], axis=1)
        y = factors[1:][mask]
        beta = _solve_ridge(x, y, ridge)
        intercept = beta[0]
        mat = beta[1:].T
        resid = y - (x @ beta)
        cov = np.cov(resid, rowvar=False)
        if factor_dim == 1:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov = cov + 1e-6 * np.eye(factor_dim, dtype=np.float64)

        scale_x = np.concatenate([np.ones((int(mask.sum()), 1)), np.abs(factors[:-1][mask])], axis=1)
        scale_beta = _solve_ridge(
            scale_x,
            np.log(np.maximum(fast_norm[mask], 1e-6)),
            ridge,
        )
        scale_ref = float(np.exp(np.mean(scale_x @ scale_beta)))
        fast_q95 = float(np.quantile(fast_norm[mask], 0.95))

        regime_intercepts.append(intercept.astype(np.float64))
        regime_mats.append(mat.astype(np.float64))
        regime_covs.append(cov.astype(np.float64))
        regime_scale_refs.append(scale_ref)
        regime_scale_betas.append(scale_beta.astype(np.float64))
        regime_fast_q95.append(fast_q95)

    return {
        "slow_alpha": np.array(slow_alpha, dtype=np.float64),
        "eps": np.array(eps, dtype=np.float64),
        "mu": mu.astype(np.float64),
        "basis": basis.astype(np.float64),
        "q20": np.array(q20, dtype=np.float64),
        "q80": np.array(q80, dtype=np.float64),
        "transition_probs": transition_probs.astype(np.float64),
        "regime_intercepts": np.stack(regime_intercepts).astype(np.float64),
        "regime_mats": np.stack(regime_mats).astype(np.float64),
        "regime_covs": np.stack(regime_covs).astype(np.float64),
        "regime_scale_refs": np.asarray(regime_scale_refs, dtype=np.float64),
        "regime_scale_betas": np.stack(regime_scale_betas).astype(np.float64),
        "regime_fast_q95": np.asarray(regime_fast_q95, dtype=np.float64),
    }


def history_factor_state(history_01: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slow_alpha = float(params["slow_alpha"])
    eps = float(params["eps"])
    hist = history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
    slow = hist[:, 0].copy()
    for t in range(1, hist.shape[1]):
        slow = (1.0 - slow_alpha) * slow + slow_alpha * hist[:, t]
    y_slow = logit_clip(slow, eps)
    factors = (y_slow - params["mu"]) @ params["basis"].T

    mean_path = hist.mean(axis=2)
    score = np.abs(np.diff(mean_path[:, -5:], axis=1)).mean(axis=1)
    regimes = label_regimes(score, float(params["q20"]), float(params["q80"]))
    return y_slow, factors, regimes


def sample_regime_step(
    regimes: np.ndarray,
    transition_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    next_regimes = np.empty_like(regimes)
    for i, r in enumerate(regimes):
        next_regimes[i] = rng.choice(3, p=transition_probs[int(r)])
    return next_regimes


def predict_factor_step(
    factors: np.ndarray,
    regimes: np.ndarray,
    params: dict[str, np.ndarray],
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    next_factors = np.empty_like(factors)
    for regime in range(3):
        mask = regimes == regime
        if not np.any(mask):
            continue
        mean = params["regime_intercepts"][regime][None, :] + factors[mask] @ params["regime_mats"][regime].T
        eps = rng.multivariate_normal(
            mean=np.zeros(params["regime_covs"][regime].shape[0], dtype=np.float64),
            cov=params["regime_covs"][regime] * (noise_scale ** 2),
            size=int(mask.sum()),
        )
        next_factors[mask] = mean + eps
    return next_factors


def predict_scale_multiplier(
    factors: np.ndarray,
    regimes: np.ndarray,
    params: dict[str, np.ndarray],
    max_scale_mult: float,
) -> np.ndarray:
    scale_mult = np.ones(factors.shape[0], dtype=np.float64)
    for regime in range(3):
        mask = regimes == regime
        if not np.any(mask):
            continue
        x = np.concatenate([np.ones((int(mask.sum()), 1)), np.abs(factors[mask])], axis=1)
        pred = np.exp(x @ params["regime_scale_betas"][regime])
        mult = pred / max(float(params["regime_scale_refs"][regime]), 1e-8)
        scale_mult[mask] = np.clip(mult, 1.0 / max_scale_mult, max_scale_mult)
    return scale_mult


@torch.no_grad()
def rollout_switching_support_state_space(
    model: torch.nn.Module,
    history_norm: torch.Tensor,
    params: dict[str, np.ndarray],
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    noise_scale: float,
    max_scale_mult: float,
) -> np.ndarray:
    device = next(model.parameters()).device
    history_01 = history_norm.detach().cpu().numpy()
    batch_size, hist_len = history_01.shape[:2]
    chunk_size = max(1, min(int(chunk_size), int(n_samples)))
    y_slow_base, factors_base, regimes_base = history_factor_state(history_01, params)
    slow_alpha = float(params["slow_alpha"])
    eps = float(params["eps"])

    all_chunks: list[np.ndarray] = []
    for start in range(0, n_samples, chunk_size):
        k = min(chunk_size, n_samples - start)
        rng = np.random.default_rng(1234 + start)

        hist_k = torch.from_numpy(history_01).to(device=device, dtype=torch.float32)
        hist_k = hist_k.unsqueeze(1).expand(batch_size, k, hist_len, 5, 5).reshape(batch_size * k, hist_len, 5, 5).clone()

        y_slow = np.repeat(y_slow_base, k, axis=0).copy()
        factors = np.repeat(factors_base, k, axis=0).copy()
        regimes = np.repeat(regimes_base, k, axis=0).copy()

        frames: list[np.ndarray] = []
        for _step in range(n_steps):
            base_next = _coerce_next_iv(model.sample_next_iv(hist_k, n_samples=1)).squeeze(1).detach().cpu().numpy()
            y_model = logit_clip(base_next.reshape(base_next.shape[0], -1), eps)
            model_slow_next = (1.0 - slow_alpha) * y_slow + slow_alpha * y_model
            fast_resid = y_model - model_slow_next

            regimes = sample_regime_step(regimes, params["transition_probs"], rng)
            factors = predict_factor_step(factors, regimes, params=params, rng=rng, noise_scale=noise_scale)
            y_slow_next = params["mu"] + factors @ params["basis"]
            scale_mult = predict_scale_multiplier(factors, regimes, params, max_scale_mult=max_scale_mult)[:, None]
            y_next = y_slow_next + scale_mult * fast_resid
            next_flat = sigmoid(y_next).reshape(base_next.shape[0], 5, 5).astype(np.float32)

            frames.append(next_flat.reshape(batch_size, k, 5, 5))
            next_torch = torch.from_numpy(next_flat).to(device=device, dtype=torch.float32)
            hist_k = torch.cat([hist_k[:, 1:], next_torch.unsqueeze(1)], dim=1)
            y_slow = y_slow_next
        all_chunks.append(np.stack(frames, axis=2))
    return np.concatenate(all_chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="220l switching support-aware state-space evaluation")
    parser.add_argument("--base_model_type", type=str, default="212ai")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--slow_alpha", type=float, default=0.08)
    parser.add_argument("--factor_dim", type=int, default=4)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--max_scale_mult", type=float, default=1.5)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.base_model_type, args.checkpoint, device)
    model.eval()

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
    surfaces = raw["surface"].astype(np.float32)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    fit_end_idx = max_train_idx - args.val_size + args.history_len
    params = fit_switching_support_state_space(
        surfaces=surfaces,
        fit_end_idx=fit_end_idx,
        slow_alpha=args.slow_alpha,
        factor_dim=args.factor_dim,
        eps=args.eps,
        ridge=args.ridge,
    )

    cond_samples = rollout_switching_support_state_space(
        model=model,
        history_norm=batch.history_norm,
        params=params,
        n_samples=args.samples,
        n_steps=batch.future_01.shape[1],
        chunk_size=args.chunk_size,
        noise_scale=args.noise_scale,
        max_scale_mult=args.max_scale_mult,
    )
    zero_hist = torch.zeros_like(batch.history_norm)
    uncond_samples = rollout_switching_support_state_space(
        model=model,
        history_norm=zero_hist,
        params=params,
        n_samples=min(args.samples, 32),
        n_steps=batch.future_01.shape[1],
        chunk_size=args.chunk_size,
        noise_scale=args.noise_scale,
        max_scale_mult=args.max_scale_mult,
    )

    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()
    rollout_start = max_train_idx - args.val_size

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
            "type": "220l_switching_support_state_space",
            "base_model_type": args.base_model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "slow_alpha": args.slow_alpha,
            "factor_dim": args.factor_dim,
            "noise_scale": args.noise_scale,
            "max_scale_mult": args.max_scale_mult,
            "fit_end_idx": int(fit_end_idx),
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
        f"- switching support-aware base model: `{args.base_model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- slow alpha: `{args.slow_alpha}`",
        f"- factor dim: `{args.factor_dim}`",
        f"- noise scale: `{args.noise_scale}`",
        f"- max residual scale mult: `{args.max_scale_mult}`",
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
    write_markdown_summary(args.output_md, "220l Switching Support-Aware State-Space Evaluation", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
