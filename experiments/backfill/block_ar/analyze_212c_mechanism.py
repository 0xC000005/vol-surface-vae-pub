#!/usr/bin/env python
"""
Mechanistic audit for 212c mean-plus-residual direct delta model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import energy_score
from experiments.backfill.block_ar.train_212c_h1_mean_residual_direct_delta import load_model


def fit_auc(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> float:
    if train_y.min() == train_y.max() or val_y.min() == val_y.max():
        return float("nan")
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(train_x, train_y)
    prob = clf.predict_proba(val_x)[:, 1]
    return float(roc_auc_score(val_y, prob))


def main() -> None:
    parser = argparse.ArgumentParser(description="212c mechanism audit")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--n_clusters", type=int, default=4)
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

    with torch.no_grad():
        train_state = model.encode(train_hist)
        val_state = model.encode(val_hist)
        train_mean_delta = model.mean_delta(train_hist)
        val_mean_delta = model.mean_delta(val_hist)
        sample_delta, _ = model.sample_delta(val_hist, n_samples=args.sample_count)

    train_state_np = train_state.detach().cpu().numpy()
    val_state_np = val_state.detach().cpu().numpy()
    train_q95_any = (train_abs.detach().cpu().numpy() >= q95).any(axis=1).astype(np.int64)
    train_q99_any = (train_abs.detach().cpu().numpy() >= q99).any(axis=1).astype(np.int64)
    val_q95_any = (val_abs.detach().cpu().numpy() >= q95).any(axis=1).astype(np.int64)
    val_q99_any = (val_abs.detach().cpu().numpy() >= q99).any(axis=1).astype(np.int64)

    encoder_probe = {
        "state_mean_norm": float(train_state.norm(dim=1).mean().item()),
        "state_std_mean": float(train_state.std(dim=0).mean().item()),
        "q95_any_auc": fit_auc(train_state_np, train_q95_any, val_state_np, val_q95_any),
        "q99_any_auc": fit_auc(train_state_np, train_q99_any, val_state_np, val_q99_any),
    }

    # Mean-head conditionality.
    mean_abs = val_mean_delta.abs().mean(dim=1).detach().cpu().numpy()
    realized_mean_abs = val_delta.abs().mean(dim=1).detach().cpu().numpy()
    realized_max_abs = val_delta.abs().max(dim=1).values.detach().cpu().numpy()
    sample_std = sample_delta.std(dim=1).mean(dim=1).detach().cpu().numpy()

    mean_residual_alignment = {
        "mean_abs_vs_realized_mean_abs_corr": float(np.corrcoef(mean_abs, realized_mean_abs)[0, 1]),
        "mean_abs_vs_realized_max_abs_corr": float(np.corrcoef(mean_abs, realized_max_abs)[0, 1]),
        "mean_abs_vs_realized_max_abs_spearman": float(spearmanr(mean_abs, realized_max_abs).statistic),
        "residual_std_vs_realized_mean_abs_corr": float(np.corrcoef(sample_std, realized_mean_abs)[0, 1]),
        "residual_std_vs_realized_max_abs_corr": float(np.corrcoef(sample_std, realized_max_abs)[0, 1]),
        "residual_std_vs_realized_max_abs_spearman": float(spearmanr(sample_std, realized_max_abs).statistic),
    }

    # Zero / mean-only / full sampled comparison.
    zero_samples = torch.zeros(val_hist.shape[0], args.sample_count, train_prev.shape[1], device=device)
    mean_only_samples = val_mean_delta.unsqueeze(1).expand(-1, args.sample_count, -1)
    model_vs_baselines = {
        "zero_delta_energy": float(energy_score(zero_samples, val_delta).item()),
        "mean_only_energy": float(energy_score(mean_only_samples, val_delta).item()),
        "full_model_energy": float(energy_score(sample_delta, val_delta).item()),
        "mean_only_mae": float((val_mean_delta - val_delta).abs().mean().item()),
    }

    # State/noise usage in residual branch.
    with torch.no_grad():
        batch = val_state.shape[0]
        noise = torch.randn(batch, model.noise_dim, device=device)
        perm = torch.randperm(batch, device=device)
        mean_base = model.mean_delta(val_hist)
        state_rep = val_state
        resid_base = torch.tanh(model.residual_head(torch.cat([state_rep, noise], dim=-1))) * (
            model.residual_scale_factor * model.delta_scale.view(1, model.n_cells)
        )
        resid_shuf_state = torch.tanh(model.residual_head(torch.cat([state_rep[perm], noise], dim=-1))) * (
            model.residual_scale_factor * model.delta_scale.view(1, model.n_cells)
        )
        resid_shuf_noise = torch.tanh(model.residual_head(torch.cat([state_rep, noise[perm]], dim=-1))) * (
            model.residual_scale_factor * model.delta_scale.view(1, model.n_cells)
        )
        resid_zero_state = torch.tanh(model.residual_head(torch.cat([torch.zeros_like(state_rep), noise], dim=-1))) * (
            model.residual_scale_factor * model.delta_scale.view(1, model.n_cells)
        )

    residual_mech = {
        "mean_abs_delta_mean": float(mean_base.abs().mean().item()),
        "residual_abs_delta_mean": float(resid_base.abs().mean().item()),
        "shuffle_state_residual_mae": float((resid_base - resid_shuf_state).abs().mean().item()),
        "shuffle_noise_residual_mae": float((resid_base - resid_shuf_noise).abs().mean().item()),
        "zero_state_residual_mae": float((resid_base - resid_zero_state).abs().mean().item()),
        "state_vs_noise_shuffle_ratio": float(
            ((resid_base - resid_shuf_state).abs().mean() / (resid_base - resid_shuf_noise).abs().mean().clamp_min(1e-8)).item()
        ),
    }

    # Cluster flow: cluster by predicted mean-delta surface.
    km = KMeans(n_clusters=args.n_clusters, random_state=0, n_init=20)
    mean_np = val_mean_delta.detach().cpu().numpy()
    labels = km.fit_predict(mean_np)
    clusters = []
    val_delta_np = val_delta.detach().cpu().numpy()
    sample_std_np = sample_std
    for k in range(args.n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        clusters.append({
            "cluster": int(k),
            "size": int(mask.sum()),
            "pred_mean_abs_mean": float(np.abs(mean_np[mask]).mean()),
            "pred_mean_abs_median": float(np.median(np.abs(mean_np[mask]))),
            "realized_mean_abs_mean": float(np.abs(val_delta_np[mask]).mean()),
            "realized_mean_abs_median": float(np.median(np.abs(val_delta_np[mask]))),
            "realized_max_abs_mean": float(np.abs(val_delta_np[mask]).max(axis=1).mean()),
            "q95_any_rate": float((np.abs(val_delta_np[mask]) >= q95).any(axis=1).mean()),
            "q99_any_rate": float((np.abs(val_delta_np[mask]) >= q99).any(axis=1).mean()),
            "residual_std_mean": float(sample_std_np[mask].mean()),
        })

    # Severity-bin flow: does predicted mean rise with realized severity?
    order = np.argsort(realized_max_abs)
    bins = np.array_split(order, 4)
    severity_flow = []
    for i, idx in enumerate(bins):
        severity_flow.append({
            "severity_bin": int(i),
            "size": int(len(idx)),
            "realized_max_abs_median": float(np.median(realized_max_abs[idx])),
            "pred_mean_abs_median": float(np.median(mean_abs[idx])),
            "residual_std_median": float(np.median(sample_std[idx])),
        })

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "encoder_probe": encoder_probe,
        "mean_residual_alignment": mean_residual_alignment,
        "model_vs_baselines": model_vs_baselines,
        "residual_mechanism": residual_mech,
        "mean_clusters": clusters,
        "severity_flow": severity_flow,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
