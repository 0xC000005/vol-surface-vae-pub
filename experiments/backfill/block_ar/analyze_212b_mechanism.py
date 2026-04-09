#!/usr/bin/env python
"""
Mechanistic audit for 212b minimal direct stochastic delta model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
    load_model,
)


def decode_bounded_delta(model, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    raw = model.decoder(torch.cat([state, noise], dim=-1))
    return torch.tanh(raw) * model.delta_scale.view(1, -1)


def fit_auc(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> float:
    if train_y.min() == train_y.max() or val_y.min() == val_y.max():
        return float("nan")
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(train_x, train_y)
    prob = clf.predict_proba(val_x)[:, 1]
    return float(roc_auc_score(val_y, prob))


def main() -> None:
    parser = argparse.ArgumentParser(description="212b mechanism audit")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--sample_count", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)[: args.max_train_windows]
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_val_windows]

    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    model, payload = load_model(args.checkpoint, device)
    model.eval()

    train_prev = train_hist[:, -1].reshape(train_hist.shape[0], -1)
    val_prev = val_hist[:, -1].reshape(val_hist.shape[0], -1)
    train_delta = train_target - train_prev
    val_delta = val_target - val_prev
    train_abs = train_delta.abs()
    val_abs = val_delta.abs()

    flat_train_abs = train_abs.detach().cpu().numpy().reshape(-1)
    q95 = float(np.quantile(flat_train_abs, 0.95))
    q99 = float(np.quantile(flat_train_abs, 0.99))

    # Baseline: zero delta / copy last surface.
    zero_samples = torch.zeros(val_hist.shape[0], args.sample_count, train_prev.shape[1], device=device)
    zero_energy = float(energy_score(zero_samples, val_delta).item())
    zero_mae = float(val_delta.abs().mean().item())
    zero_small_share = float((val_delta.abs() <= 0.005).float().mean().item())

    with torch.no_grad():
        train_state = model.encode(train_hist).detach().cpu().numpy()
        val_state = model.encode(val_hist).detach().cpu().numpy()

    train_q95_any = (train_abs.detach().cpu().numpy() >= q95).any(axis=1).astype(np.int64)
    train_q99_any = (train_abs.detach().cpu().numpy() >= q99).any(axis=1).astype(np.int64)
    val_q95_any = (val_abs.detach().cpu().numpy() >= q95).any(axis=1).astype(np.int64)
    val_q99_any = (val_abs.detach().cpu().numpy() >= q99).any(axis=1).astype(np.int64)

    encoder_probe = {
        "state_mean_norm": float(np.mean(np.linalg.norm(val_state, axis=1))),
        "state_std_mean": float(np.mean(np.std(val_state, axis=0))),
        "q95_any_auc": fit_auc(train_state, train_q95_any, val_state, val_q95_any),
        "q99_any_auc": fit_auc(train_state, train_q99_any, val_state, val_q99_any),
    }

    # State/noise sensitivity on the same fixed noise draw.
    batch_hist = val_hist
    with torch.no_grad():
        state = model.encode(batch_hist)
        batch = state.shape[0]
        noise = torch.randn(batch, model.noise_dim, device=device)
        perm = torch.randperm(batch, device=device)
        base_delta = decode_bounded_delta(model, state, noise)
        shuf_state_delta = decode_bounded_delta(model, state[perm], noise)
        zero_state_delta = decode_bounded_delta(model, torch.zeros_like(state), noise)
        shuf_noise_delta = decode_bounded_delta(model, state, noise[perm])

        W = model.decoder[0].weight
        b = model.decoder[0].bias
        state_part = state @ W[:, : model.hidden_dim].T
        noise_part = noise @ W[:, model.hidden_dim :].T
        preact = state_part + noise_part + b

    state_noise_mech = {
        "base_abs_delta_mean": float(base_delta.abs().mean().item()),
        "shuffle_state_delta_mae": float((base_delta - shuf_state_delta).abs().mean().item()),
        "zero_state_delta_mae": float((base_delta - zero_state_delta).abs().mean().item()),
        "shuffle_noise_delta_mae": float((base_delta - shuf_noise_delta).abs().mean().item()),
        "state_vs_noise_shuffle_ratio": float(
            ((base_delta - shuf_state_delta).abs().mean() / (base_delta - shuf_noise_delta).abs().mean().clamp_min(1e-8)).item()
        ),
        "first_layer_state_part_abs_mean": float(state_part.abs().mean().item()),
        "first_layer_noise_part_abs_mean": float(noise_part.abs().mean().item()),
        "first_layer_state_weight_fro": float(torch.linalg.norm(W[:, : model.hidden_dim]).item()),
        "first_layer_noise_weight_fro": float(torch.linalg.norm(W[:, model.hidden_dim :]).item()),
        "preact_abs_mean": float(preact.abs().mean().item()),
    }

    # Conditional dispersion alignment.
    with torch.no_grad():
        sample_delta = model.sample_delta(val_hist, n_samples=args.sample_count)
    pred_std = sample_delta.std(dim=1).mean(dim=1).detach().cpu().numpy()
    pred_mean_abs = sample_delta.mean(dim=1).abs().mean(dim=1).detach().cpu().numpy()
    realized_mean_abs = val_delta.abs().mean(dim=1).detach().cpu().numpy()
    realized_max_abs = val_delta.abs().max(dim=1).values.detach().cpu().numpy()

    cond_alignment = {
        "pred_std_vs_realized_mean_abs_corr": float(np.corrcoef(pred_std, realized_mean_abs)[0, 1]),
        "pred_std_vs_realized_max_abs_corr": float(np.corrcoef(pred_std, realized_max_abs)[0, 1]),
        "pred_meanabs_vs_realized_mean_abs_corr": float(np.corrcoef(pred_mean_abs, realized_mean_abs)[0, 1]),
        "pred_std_vs_realized_max_abs_spearman": float(spearmanr(pred_std, realized_max_abs).statistic),
    }

    # Tail reachability under fixed per-cell bounds.
    delta_scale = model.delta_scale.detach().cpu().numpy()
    val_abs_np = val_abs.detach().cpu().numpy()
    tail_reachability = {
        "delta_scale_min": float(delta_scale.min()),
        "delta_scale_median": float(np.median(delta_scale)),
        "delta_scale_max": float(delta_scale.max()),
        "gt_exceed_percell_scale_frac": float((val_abs_np > delta_scale).mean()),
        "windows_any_gt_exceed_percell_scale_frac": float((val_abs_np > delta_scale).any(axis=1).mean()),
        "q99_cells_exceed_percell_scale_frac": float(((val_abs_np >= q99) & (val_abs_np > delta_scale)).sum() / max((val_abs_np >= q99).sum(), 1)),
        "val_max_abs_delta": float(val_abs_np.max()),
    }

    # Compare against model sample law on val.
    with torch.no_grad():
        sample_iv = model.sample_next_iv(val_hist, n_samples=args.sample_count)
    q05 = sample_iv.quantile(0.05, dim=1)
    q95_iv = sample_iv.quantile(0.95, dim=1)
    cover90 = float(((val_target >= q05) & (val_target <= q95_iv)).float().mean().item())
    q99_mask = val_abs >= q99
    q99_cov = float((((val_target[q99_mask] >= q05[q99_mask]) & (val_target[q99_mask] <= q95_iv[q99_mask])).float().mean().item()) if q99_mask.any() else float("nan"))

    model_vs_zero = {
        "zero_delta_energy": zero_energy,
        "zero_delta_mae": zero_mae,
        "zero_delta_exact_coverage": 0.0,
        "zero_delta_very_small_share": zero_small_share,
        "model_sample_energy": float(energy_score(sample_delta, val_delta).item()),
        "model_coverage_90": cover90,
        "model_realized_q99_coverage_90": q99_cov,
    }

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "model_vs_zero_baseline": model_vs_zero,
        "encoder_probe": encoder_probe,
        "state_noise_mechanism": state_noise_mech,
        "conditional_alignment": cond_alignment,
        "tail_reachability": tail_reachability,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
