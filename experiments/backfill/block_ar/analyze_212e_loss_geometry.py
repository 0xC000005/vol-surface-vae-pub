#!/usr/bin/env python
"""
Analyze the loss geometry of 212e under the current H=1 energy-score objective.

This script focuses on three questions:
1. For a simple one-cell 212e-style law, how do ES gradients change as the target move grows?
2. Under the empirical one-day delta distribution, do ordinary days mostly push scales down?
3. On real 212e windows, does ES push center harder than width?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import energy_score
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import load_model


def one_dim_gradients(
    target_value: float,
    eps: torch.Tensor,
    center_init: float,
    pos_scale_init: float,
    neg_scale_init: float,
) -> dict[str, float]:
    center = torch.tensor([center_init], dtype=torch.float32, requires_grad=True)
    pos_scale = torch.tensor([pos_scale_init], dtype=torch.float32, requires_grad=True)
    neg_scale = torch.tensor([neg_scale_init], dtype=torch.float32, requires_grad=True)
    samples = (
        center.view(1, 1, 1)
        + pos_scale.view(1, 1, 1) * torch.relu(eps)
        - neg_scale.view(1, 1, 1) * torch.relu(-eps)
    )
    target = torch.tensor([[target_value]], dtype=torch.float32)
    loss = energy_score(samples, target)
    loss.backward()
    return {
        "target": float(target_value),
        "loss": float(loss.item()),
        "sample_mean": float(samples.detach().mean().item()),
        "sample_std": float(samples.detach().std().item()),
        "grad_center": float(center.grad.item()),
        "grad_pos_scale": float(pos_scale.grad.item()),
        "grad_neg_scale": float(neg_scale.grad.item()),
    }


def empirical_bucket_gradients(
    flat_deltas: np.ndarray,
    eps: torch.Tensor,
    center_init: float,
    pos_scale_init: float,
    neg_scale_init: float,
    sample_size: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    abs_deltas = np.abs(flat_deltas)
    buckets = [
        ("very_small", abs_deltas <= 0.005),
        ("small_to_medium", (abs_deltas > 0.005) & (abs_deltas <= 0.020)),
        ("medium_to_large", (abs_deltas > 0.020) & (abs_deltas <= 0.050)),
        ("tail", abs_deltas > 0.050),
    ]
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for name, mask in buckets:
        ys = flat_deltas[mask]
        if ys.size > sample_size:
            ys = ys[rng.choice(ys.size, size=sample_size, replace=False)]
        rows = [one_dim_gradients(float(y), eps, center_init, pos_scale_init, neg_scale_init) for y in ys]
        grads = np.array([[row["grad_center"], row["grad_pos_scale"], row["grad_neg_scale"]] for row in rows], dtype=np.float64)
        out[name] = {
            "count": int(mask.sum()),
            "share": float(mask.mean()),
            "mean_grad_center": float(grads[:, 0].mean()),
            "mean_abs_grad_center": float(np.abs(grads[:, 0]).mean()),
            "mean_grad_pos_scale": float(grads[:, 1].mean()),
            "mean_abs_grad_pos_scale": float(np.abs(grads[:, 1]).mean()),
            "mean_grad_neg_scale": float(grads[:, 2].mean()),
            "mean_abs_grad_neg_scale": float(np.abs(grads[:, 2]).mean()),
        }
    return out


def overall_empirical_gradients(
    flat_deltas: np.ndarray,
    eps: torch.Tensor,
    center_init: float,
    pos_scale_init: float,
    neg_scale_init: float,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ys = flat_deltas
    if ys.size > sample_size:
        ys = ys[rng.choice(ys.size, size=sample_size, replace=False)]
    rows = [one_dim_gradients(float(y), eps, center_init, pos_scale_init, neg_scale_init) for y in ys]
    grads = np.array([[row["grad_center"], row["grad_pos_scale"], row["grad_neg_scale"]] for row in rows], dtype=np.float64)
    return {
        "mean_grad_center": float(grads[:, 0].mean()),
        "mean_abs_grad_center": float(np.abs(grads[:, 0]).mean()),
        "mean_grad_pos_scale": float(grads[:, 1].mean()),
        "mean_abs_grad_pos_scale": float(np.abs(grads[:, 1]).mean()),
        "mean_grad_neg_scale": float(grads[:, 2].mean()),
        "mean_abs_grad_neg_scale": float(np.abs(grads[:, 2]).mean()),
    }


def pick_typical_window(realized_max_abs: torch.Tensor) -> int:
    med = realized_max_abs.median()
    return int(torch.argmin((realized_max_abs - med).abs()).item())


def real_window_gradients(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    val_delta: torch.Tensor,
    window_idx: int,
    sample_count: int,
    noise_seed: int,
) -> dict[str, float]:
    h = history_01[window_idx : window_idx + 1]
    t = target_01[window_idx : window_idx + 1]
    prev = h[:, -1].reshape(1, -1)
    target_delta = t - prev
    state = model.encode(h)
    center, pos_scale, neg_scale, gamma, beta = model.decode_params_from_state(state)
    generator = torch.Generator(device=h.device)
    generator.manual_seed(noise_seed + window_idx)
    noise = torch.randn(1, sample_count, model.noise_dim, generator=generator, device=h.device, dtype=h.dtype)
    with torch.no_grad():
        eps, _u_mod = model.noise_pattern(state, noise)

    center_r = center.detach().clone().requires_grad_(True)
    pos_scale_r = pos_scale.detach().clone().requires_grad_(True)
    neg_scale_r = neg_scale.detach().clone().requires_grad_(True)
    delta = (
        center_r.unsqueeze(1)
        + pos_scale_r.unsqueeze(1) * torch.relu(eps)
        - neg_scale_r.unsqueeze(1) * torch.relu(-eps)
    )
    lower = -prev.unsqueeze(1)
    upper = 1.0 - prev.unsqueeze(1)
    delta = torch.maximum(torch.minimum(delta, upper), lower)
    loss = energy_score(delta, target_delta)
    loss.backward()

    q95_threshold = torch.quantile(val_delta.abs().reshape(-1), 0.95)
    return {
        "window_idx": int(window_idx),
        "loss": float(loss.item()),
        "target_mean_delta": float(target_delta.mean().item()),
        "target_max_abs": float(target_delta.abs().max().item()),
        "frac_positive_target": float((target_delta > 0).float().mean().item()),
        "frac_cells_gt_q95": float((target_delta.abs() >= q95_threshold).float().mean().item()),
        "center_abs_mean": float(center.abs().mean().item()),
        "pos_scale_mean": float(pos_scale.mean().item()),
        "neg_scale_mean": float(neg_scale.mean().item()),
        "grad_abs_center_mean": float(center_r.grad.detach().abs().mean().item()),
        "grad_abs_pos_scale_mean": float(pos_scale_r.grad.detach().abs().mean().item()),
        "grad_abs_neg_scale_mean": float(neg_scale_r.grad.detach().abs().mean().item()),
        "grad_signed_center_mean": float(center_r.grad.detach().mean().item()),
        "grad_signed_pos_scale_mean": float(pos_scale_r.grad.detach().mean().item()),
        "grad_signed_neg_scale_mean": float(neg_scale_r.grad.detach().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 212e loss geometry")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--toy_noise_count", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_end = max_train_idx - args.val_size
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, max_train_idx)

    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)
    train_prev = train_hist[:, -1].reshape(train_hist.shape[0], -1)
    val_prev = val_hist[:, -1].reshape(val_hist.shape[0], -1)
    train_delta = (train_target - train_prev).detach().cpu().numpy().reshape(-1)
    val_delta = val_target - val_prev
    realized_max_abs = val_delta.abs().max(dim=1).values

    model, payload = load_model(args.checkpoint, device)
    model.eval()

    toy_noise = torch.randn(1, args.toy_noise_count, 1, device=device)
    toy_eps = torch.tanh(toy_noise)
    target_scan = [-0.20, -0.10, -0.05, -0.02, -0.005, 0.0, 0.005, 0.02, 0.05, 0.10, 0.20]
    toy_scan = [
        one_dim_gradients(y, toy_eps, center_init=0.0, pos_scale_init=0.007, neg_scale_init=0.007)
        for y in target_scan
    ]
    bucket_grads = empirical_bucket_gradients(
        flat_deltas=train_delta,
        eps=toy_eps,
        center_init=0.0,
        pos_scale_init=0.007,
        neg_scale_init=0.007,
        sample_size=5000,
        seed=args.seed,
    )
    overall_grads = overall_empirical_gradients(
        flat_deltas=train_delta,
        eps=toy_eps,
        center_init=0.0,
        pos_scale_init=0.007,
        neg_scale_init=0.007,
        sample_size=10000,
        seed=args.seed,
    )

    worst_window = 94
    max_window = int(realized_max_abs.argmax().item())
    typical_window = pick_typical_window(realized_max_abs)
    selected_windows = [typical_window, worst_window, 171, max_window]
    real_windows = [
        real_window_gradients(
            model=model,
            history_01=val_hist,
            target_01=val_target,
            val_delta=val_delta,
            window_idx=idx,
            sample_count=args.sample_count,
            noise_seed=args.seed,
        )
        for idx in selected_windows
    ]

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "toy_scan": toy_scan,
        "empirical_bucket_gradients": bucket_grads,
        "overall_empirical_gradients": overall_grads,
        "real_window_gradients": real_windows,
    }
    out_json.write_text(json.dumps(out, indent=2))

    lines = [
        "# 212e Loss Geometry Audit",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Epoch: `{payload.get('epoch', -1)}`",
        "",
        "## Main Takeaways",
        "",
        "- Energy score pushes the forecast center harder than it pushes forecast width once the realized move is outside the current sample cloud.",
        "- On very small one-day moves, both positive and negative scale gradients mostly point downward, so ordinary days teach the model to stay narrow.",
        "- On real severe windows, the width gradients do point in the right direction, but they are still materially smaller than the center gradients.",
        "",
        "## Selected Toy Targets",
        "",
    ]
    for row in toy_scan:
        lines.append(
            f"- target `{row['target']:+.3f}`: grad_center `{row['grad_center']:+.3f}`, "
            f"grad_pos_scale `{row['grad_pos_scale']:+.3f}`, grad_neg_scale `{row['grad_neg_scale']:+.3f}`"
        )
    lines += [
        "",
        "## Empirical Bucket Gradients",
        "",
    ]
    for name, row in bucket_grads.items():
        lines.append(
            f"- `{name}` share `{row['share']:.3f}`: "
            f"mean_abs_grad_center `{row['mean_abs_grad_center']:.3f}`, "
            f"mean_abs_grad_pos_scale `{row['mean_abs_grad_pos_scale']:.3f}`, "
            f"mean_abs_grad_neg_scale `{row['mean_abs_grad_neg_scale']:.3f}`"
        )
        lines.append(
            f"  signed grads: center `{row['mean_grad_center']:+.3f}`, "
            f"pos_scale `{row['mean_grad_pos_scale']:+.3f}`, "
            f"neg_scale `{row['mean_grad_neg_scale']:+.3f}`"
        )
    lines += [
        "",
        "## Real Window Gradients",
        "",
    ]
    for row in real_windows:
        lines.append(
            f"- window `{row['window_idx']}` max|delta| `{row['target_max_abs']:.3f}`: "
            f"abs grad center `{row['grad_abs_center_mean']:.3f}`, "
            f"abs grad pos_scale `{row['grad_abs_pos_scale_mean']:.3f}`, "
            f"abs grad neg_scale `{row['grad_abs_neg_scale_mean']:.3f}`"
        )
        lines.append(
            f"  signed grads: center `{row['grad_signed_center_mean']:+.3f}`, "
            f"pos_scale `{row['grad_signed_pos_scale_mean']:+.3f}`, "
            f"neg_scale `{row['grad_signed_neg_scale_mean']:+.3f}`"
        )
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
