#!/usr/bin/env python
"""
Focused residual-path audit for 212c.

Questions:
1. Where does state information get washed out inside the residual head?
2. Is the residual law approximately symmetric around the mean, even when
   realized next-day moves are state-asymmetric?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_212c_h1_mean_residual_direct_delta import load_model


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


def forward_parts(model, state_flat: torch.Tensor, noise_flat: torch.Tensor) -> dict[str, torch.Tensor]:
    lin1 = model.residual_head[0]
    lin2 = model.residual_head[2]
    lin3 = model.residual_head[4]
    x = torch.cat([state_flat, noise_flat], dim=-1)
    pre1 = lin1(x)
    act1 = F.silu(pre1)
    pre2 = lin2(act1)
    act2 = F.silu(pre2)
    pre3 = lin3(act2)
    resid = torch.tanh(pre3) * (model.residual_scale_factor * model.delta_scale.view(1, model.n_cells))
    return {
        "pre1": pre1,
        "act1": act1,
        "pre2": pre2,
        "act2": act2,
        "pre3": pre3,
        "resid": resid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="212c residual-path audit")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--layer_sample_count", type=int, default=64)
    parser.add_argument("--grad_batch_size", type=int, default=64)
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

    with torch.no_grad():
        val_state = model.encode(val_hist)
        mean_delta = model.mean_delta(val_hist)
        sample_delta, _ = model.sample_delta(val_hist, n_samples=args.sample_count)
    residual = sample_delta - mean_delta.unsqueeze(1)

    realized_max_abs = val_delta.abs().max(dim=1).values.detach().cpu().numpy()
    realized_mean_abs = val_delta.abs().mean(dim=1).detach().cpu().numpy()

    # Low/high state masks based on per-cell training quartiles.
    train_prev_np = train_prev.detach().cpu().numpy()
    val_prev_np = val_prev.detach().cpu().numpy()
    val_delta_np = val_delta.detach().cpu().numpy()
    mean_delta_np = mean_delta.detach().cpu().numpy()
    sample_delta_np = sample_delta.detach().cpu().numpy()
    residual_np = residual.detach().cpu().numpy()
    low_mask, high_mask = split_level_masks(train_prev_np, val_prev_np)

    # Pooled low/high summaries on realized, mean, sampled delta, and residual.
    low_sample_mask = np.repeat(low_mask[:, None, :], args.sample_count, axis=1)
    high_sample_mask = np.repeat(high_mask[:, None, :], args.sample_count, axis=1)
    asymmetry = {
        "low_state": {
            "realized_delta": signed_summary(val_delta_np[low_mask]),
            "mean_delta": signed_summary(mean_delta_np[low_mask]),
            "sample_delta": signed_summary(sample_delta_np[low_sample_mask]),
            "sample_residual": signed_summary(residual_np[low_sample_mask]),
        },
        "high_state": {
            "realized_delta": signed_summary(val_delta_np[high_mask]),
            "mean_delta": signed_summary(mean_delta_np[high_mask]),
            "sample_delta": signed_summary(sample_delta_np[high_sample_mask]),
            "sample_residual": signed_summary(residual_np[high_sample_mask]),
        },
    }

    # Window-level sign-balance by average level score to show asymmetry response.
    train_prev_mean = train_prev.mean(dim=0)
    train_prev_std = train_prev.std(dim=0).clamp_min(1e-6)
    level_score = ((val_prev - train_prev_mean.view(1, -1)) / train_prev_std.view(1, -1)).mean(dim=1)
    level_score_np = level_score.detach().cpu().numpy()
    mean_sign_balance = np.sign(mean_delta_np).mean(axis=1)
    sample_sign_balance = np.sign(sample_delta_np).mean(axis=(1, 2))
    realized_sign_balance = np.sign(val_delta_np).mean(axis=1)
    residual_sign_balance = np.sign(residual_np).mean(axis=(1, 2))
    window_level_response = {
        "level_score_vs_realized_sign_balance_corr": safe_corr(level_score_np, realized_sign_balance),
        "level_score_vs_mean_sign_balance_corr": safe_corr(level_score_np, mean_sign_balance),
        "level_score_vs_sample_sign_balance_corr": safe_corr(level_score_np, sample_sign_balance),
        "level_score_vs_residual_sign_balance_corr": safe_corr(level_score_np, residual_sign_balance),
    }

    # Layerwise state-vs-noise influence.
    with torch.no_grad():
        bsz = val_state.shape[0]
        k = args.layer_sample_count
        noise = torch.randn(bsz, k, model.noise_dim, device=device)
        state_rep = val_state.unsqueeze(1).expand(bsz, k, model.hidden_dim)

        perm_state = torch.randperm(bsz, device=device)
        perm_noise = torch.randperm(bsz, device=device)
        state_shuf = val_state[perm_state].unsqueeze(1).expand(bsz, k, model.hidden_dim)
        noise_shuf = noise[perm_noise]

        base = forward_parts(model, state_rep.reshape(bsz * k, -1), noise.reshape(bsz * k, -1))
        shuf_state = forward_parts(model, state_shuf.reshape(bsz * k, -1), noise.reshape(bsz * k, -1))
        shuf_noise = forward_parts(model, state_rep.reshape(bsz * k, -1), noise_shuf.reshape(bsz * k, -1))

        layerwise = {}
        for key in ("pre1", "act1", "pre2", "act2", "pre3", "resid"):
            state_mae = float((base[key] - shuf_state[key]).abs().mean().item())
            noise_mae = float((base[key] - shuf_noise[key]).abs().mean().item())
            layerwise[key] = {
                "shuffle_state_mae": state_mae,
                "shuffle_noise_mae": noise_mae,
                "state_noise_ratio": float(state_mae / max(noise_mae, 1e-8)),
            }

        lin1 = model.residual_head[0]
        w_state = lin1.weight[:, : model.hidden_dim]
        w_noise = lin1.weight[:, model.hidden_dim :]
        state_flat = state_rep.reshape(bsz * k, -1)
        noise_flat = noise.reshape(bsz * k, -1)
        pre1_state = state_flat @ w_state.t()
        pre1_noise = noise_flat @ w_noise.t()
        repeated_severity = np.repeat(realized_max_abs, k)
        repeated_mean_abs = np.repeat(realized_mean_abs, k)
        pre1_contrib = {
            "state_abs_mean": float(pre1_state.abs().mean().item()),
            "noise_abs_mean": float(pre1_noise.abs().mean().item()),
            "state_noise_abs_ratio": float(pre1_state.abs().mean().item() / max(pre1_noise.abs().mean().item(), 1e-8)),
            "state_dominates_fraction": float((pre1_state.abs() > pre1_noise.abs()).float().mean().item()),
            "row_state_abs_vs_realized_max_abs_corr": safe_corr(
                pre1_state.abs().mean(dim=1).detach().cpu().numpy(), repeated_severity
            ),
            "row_noise_abs_vs_realized_max_abs_corr": safe_corr(
                pre1_noise.abs().mean(dim=1).detach().cpu().numpy(), repeated_severity
            ),
            "row_state_abs_vs_realized_mean_abs_corr": safe_corr(
                pre1_state.abs().mean(dim=1).detach().cpu().numpy(), repeated_mean_abs
            ),
            "row_noise_abs_vs_realized_mean_abs_corr": safe_corr(
                pre1_noise.abs().mean(dim=1).detach().cpu().numpy(), repeated_mean_abs
            ),
        }

    # Local gradient sensitivity of the residual branch.
    grad_b = min(args.grad_batch_size, val_state.shape[0])
    state_g = val_state[:grad_b].detach().clone().requires_grad_(True)
    noise_g = torch.randn(grad_b, model.noise_dim, device=device, requires_grad=True)
    resid_g = torch.tanh(model.residual_head(torch.cat([state_g, noise_g], dim=-1))) * (
        model.residual_scale_factor * model.delta_scale.view(1, model.n_cells)
    )
    scalar = (resid_g ** 2).mean()
    grad_state, grad_noise = torch.autograd.grad(scalar, [state_g, noise_g], retain_graph=False)
    gradient_sensitivity = {
        "state_grad_rms": float(grad_state.pow(2).mean(dim=1).sqrt().mean().item()),
        "noise_grad_rms": float(grad_noise.pow(2).mean(dim=1).sqrt().mean().item()),
        "state_noise_grad_ratio": float(
            grad_state.pow(2).mean(dim=1).sqrt().mean().item()
            / max(grad_noise.pow(2).mean(dim=1).sqrt().mean().item(), 1e-8)
        ),
    }

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "residual_path_layerwise": layerwise,
        "residual_path_pre1_contrib": pre1_contrib,
        "residual_gradient_sensitivity": gradient_sensitivity,
        "state_level_asymmetry": asymmetry,
        "window_level_response": window_level_response,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
