#!/usr/bin/env python
"""
Mechanistic audit for 212e asymmetric modulated direct delta model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import energy_score
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import load_model


def fit_auc(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> float:
    if train_y.min() == train_y.max() or val_y.min() == val_y.max():
        return float("nan")
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(train_x, train_y)
    prob = clf.predict_proba(val_x)[:, 1]
    return float(roc_auc_score(val_y, prob))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def signed_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "abs_mean": float("nan"),
            "up_rate": float("nan"),
            "down_rate": float("nan"),
            "positive_abs_mean": float("nan"),
            "negative_abs_mean": float("nan"),
            "sign_balance": float("nan"),
        }
    pos = flat[flat > 0.0]
    neg = flat[flat < 0.0]
    up_rate = float((flat > 0.0).mean())
    down_rate = float((flat < 0.0).mean())
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "abs_mean": float(np.abs(flat).mean()),
        "up_rate": up_rate,
        "down_rate": down_rate,
        "positive_abs_mean": float(np.abs(pos).mean()) if pos.size else 0.0,
        "negative_abs_mean": float(np.abs(neg).mean()) if neg.size else 0.0,
        "sign_balance": float(up_rate - down_rate),
    }


def split_level_masks(train_prev: np.ndarray, val_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q25 = np.quantile(train_prev, 0.25, axis=0)
    q75 = np.quantile(train_prev, 0.75, axis=0)
    low_mask = val_prev <= q25[None, :]
    high_mask = val_prev >= q75[None, :]
    return low_mask, high_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="212e mechanism audit")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--sample_count", type=int, default=256)
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
        samples, aux = model.sample_delta(val_hist, n_samples=args.sample_count)
        center, pos_scale, neg_scale, gamma, beta = model.decode_params(val_hist)

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

    center_np = center.detach().cpu().numpy()
    pos_scale_np = pos_scale.detach().cpu().numpy()
    neg_scale_np = neg_scale.detach().cpu().numpy()
    gamma_np = gamma.detach().cpu().numpy()
    beta_np = beta.detach().cpu().numpy()
    sample_np = samples.detach().cpu().numpy()
    eps_np = aux["eps"].detach().cpu().numpy()
    realized_max_abs = val_delta.abs().max(dim=1).values.detach().cpu().numpy()
    realized_mean_abs = val_delta.abs().mean(dim=1).detach().cpu().numpy()
    sample_std = samples.std(dim=1).mean(dim=1).detach().cpu().numpy()

    law_alignment = {
        "center_abs_vs_realized_max_abs_corr": safe_corr(np.abs(center_np).mean(axis=1), realized_max_abs),
        "pos_scale_vs_realized_max_abs_corr": safe_corr(pos_scale_np.mean(axis=1), realized_max_abs),
        "neg_scale_vs_realized_max_abs_corr": safe_corr(neg_scale_np.mean(axis=1), realized_max_abs),
        "sample_std_vs_realized_max_abs_corr": safe_corr(sample_std, realized_max_abs),
        "sample_std_vs_realized_mean_abs_corr": safe_corr(sample_std, realized_mean_abs),
        "gamma_norm_vs_realized_max_abs_corr": safe_corr(np.linalg.norm(gamma_np, axis=1), realized_max_abs),
        "beta_norm_vs_realized_max_abs_corr": safe_corr(np.linalg.norm(beta_np, axis=1), realized_max_abs),
    }

    model_vs_baselines = {
        "zero_delta_energy": float(energy_score(torch.zeros_like(samples), val_delta).item()),
        "center_only_energy": float(
            energy_score(center.unsqueeze(1).expand(-1, args.sample_count, -1), val_delta).item()
        ),
        "full_model_energy": float(energy_score(samples, val_delta).item()),
    }

    def sample_from_state_prev(state: torch.Tensor, prev: torch.Tensor, noise: torch.Tensor):
        center_s, pos_scale_s, neg_scale_s, gamma_s, beta_s = model.decode_params_from_state(state)
        eps_s, _u_mod = model.noise_pattern(state, noise)
        delta_s = (
            center_s.unsqueeze(1)
            + pos_scale_s.unsqueeze(1) * torch.relu(eps_s)
            - neg_scale_s.unsqueeze(1) * torch.relu(-eps_s)
        )
        lower = -prev.unsqueeze(1)
        upper = 1.0 - prev.unsqueeze(1)
        delta_s = torch.maximum(torch.minimum(delta_s, upper), lower)
        aux_s = {
            "center": center_s,
            "pos_scale": pos_scale_s,
            "neg_scale": neg_scale_s,
            "eps": eps_s,
            "gamma": gamma_s,
            "beta": beta_s,
        }
        return delta_s, aux_s

    with torch.no_grad():
        bsz = val_state.shape[0]
        noise = torch.randn(bsz, args.sample_count, model.noise_dim, device=device)
        perm = torch.randperm(bsz, device=device)
        sample_base, aux_base = sample_from_state_prev(val_state, val_prev, noise)
        sample_shuf_state, aux_shuf_state = sample_from_state_prev(val_state[perm], val_prev, noise)
        sample_shuf_noise, aux_shuf_noise = sample_from_state_prev(val_state, val_prev, noise[perm])

    decoder_usage = {
        "shuffle_state_sample_mae": float((sample_base - sample_shuf_state).abs().mean().item()),
        "shuffle_noise_sample_mae": float((sample_base - sample_shuf_noise).abs().mean().item()),
        "state_noise_sample_ratio": float(
            (sample_base - sample_shuf_state).abs().mean().item()
            / max((sample_base - sample_shuf_noise).abs().mean().item(), 1e-8)
        ),
        "shuffle_state_center_mae": float((aux_base["center"] - aux_shuf_state["center"]).abs().mean().item()),
        "shuffle_state_pos_scale_mae": float((aux_base["pos_scale"] - aux_shuf_state["pos_scale"]).abs().mean().item()),
        "shuffle_state_neg_scale_mae": float((aux_base["neg_scale"] - aux_shuf_state["neg_scale"]).abs().mean().item()),
        "shuffle_state_eps_mae": float((aux_base["eps"] - aux_shuf_state["eps"]).abs().mean().item()),
        "shuffle_noise_eps_mae": float((aux_base["eps"] - aux_shuf_noise["eps"]).abs().mean().item()),
    }

    train_prev_np = train_prev.detach().cpu().numpy()
    val_prev_np = val_prev.detach().cpu().numpy()
    val_delta_np = val_delta.detach().cpu().numpy()
    low_mask, high_mask = split_level_masks(train_prev_np, val_prev_np)
    low_sample_mask = np.repeat(low_mask[:, None, :], args.sample_count, axis=1)
    high_sample_mask = np.repeat(high_mask[:, None, :], args.sample_count, axis=1)

    asymmetry = {
        "low_state": {
            "realized_delta": signed_summary(val_delta_np[low_mask]),
            "center": signed_summary(center_np[low_mask]),
            "sample_delta": signed_summary(sample_np[low_sample_mask]),
            "eps": signed_summary(eps_np[low_sample_mask]),
        },
        "high_state": {
            "realized_delta": signed_summary(val_delta_np[high_mask]),
            "center": signed_summary(center_np[high_mask]),
            "sample_delta": signed_summary(sample_np[high_sample_mask]),
            "eps": signed_summary(eps_np[high_sample_mask]),
        },
    }

    train_prev_mean = train_prev.mean(dim=0)
    train_prev_std = train_prev.std(dim=0).clamp_min(1e-6)
    level_score = ((val_prev - train_prev_mean.view(1, -1)) / train_prev_std.view(1, -1)).mean(dim=1)
    level_score_np = level_score.detach().cpu().numpy()
    center_sign_balance = np.sign(center_np).mean(axis=1)
    sample_sign_balance = np.sign(sample_np).mean(axis=(1, 2))
    eps_sign_balance = np.sign(eps_np).mean(axis=(1, 2))
    realized_sign_balance = np.sign(val_delta_np).mean(axis=1)
    window_level_response = {
        "level_score_vs_realized_sign_balance_corr": safe_corr(level_score_np, realized_sign_balance),
        "level_score_vs_center_sign_balance_corr": safe_corr(level_score_np, center_sign_balance),
        "level_score_vs_sample_sign_balance_corr": safe_corr(level_score_np, sample_sign_balance),
        "level_score_vs_eps_sign_balance_corr": safe_corr(level_score_np, eps_sign_balance),
    }

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "encoder_probe": encoder_probe,
        "law_alignment": law_alignment,
        "model_vs_baselines": model_vs_baselines,
        "decoder_usage": decoder_usage,
        "state_level_asymmetry": asymmetry,
        "window_level_response": window_level_response,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
