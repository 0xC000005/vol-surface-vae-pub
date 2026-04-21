#!/usr/bin/env python
"""
264b-v0: 263b prior/decoder with a learned future-factor epsilon teacher.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.joint_state_space_latent_factor_fm_epsilon_teacher import (
    JointStateSpaceLatentFactorFMEpsilonTeacher,
    JointStateSpaceLatentFactorFMEpsilonTeacherConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows


def build_raw_target_changes(history_norm: torch.Tensor, future_norm: torch.Tensor) -> torch.Tensor:
    first = future_norm[:, :1] - history_norm[:, -1:]
    rest = future_norm[:, 1:] - future_norm[:, :-1]
    return torch.cat([first, rest], dim=1)


def fm_step(
    model: JointStateSpaceLatentFactorFMEpsilonTeacher,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    *,
    lambda_change: float,
    lambda_level: float,
    lambda_terminal: float,
    lambda_nll: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz = future_norm.shape[0]
    raw_target_change = build_raw_target_changes(history_norm, future_norm)
    cond = model.condition(history_norm)
    residual_raw_target = model.residualize_raw_change(raw_target_change, history_norm, future_norm, cond=cond)
    target_coord = model.transform_change(residual_raw_target, history_norm)
    z_target, eps_target = model.posterior_target(target_coord, cond)

    x0 = torch.randn_like(eps_target)
    t = torch.rand(bsz, device=eps_target.device, dtype=eps_target.dtype)
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * x0 + t_view * eps_target
    target_v = eps_target - x0
    pred_v = model.velocity(x_t, t, cond)
    fm_loss = (pred_v - target_v).pow(2).mean()

    zero_eps = torch.zeros_like(eps_target)
    posterior_change = model.decode_common(cond["loadings"], z_target)
    change_loss = F.smooth_l1_loss(posterior_change, target_coord)
    _, center_levels = model.deterministic_center_path(history_norm, cond=cond)
    level_loss = F.smooth_l1_loss(center_levels, future_norm)
    terminal_loss = F.smooth_l1_loss(center_levels[:, -1], future_norm[:, -1])

    sigma = cond["sigma"].clamp_min(model.cfg.sigma_floor)
    latent_nll = 0.5 * (((z_target.detach() - cond["mu"]) / sigma) ** 2 + 2.0 * torch.log(sigma)).mean()
    ortho = model.ortho_penalty(cond["loadings"])

    loss = (
        fm_loss
        + lambda_change * change_loss
        + lambda_level * level_loss
        + lambda_terminal * terminal_loss
        + lambda_nll * latent_nll
        + model.cfg.ortho_reg_weight * ortho
    )
    metrics = {
        "total": loss.detach(),
        "fm_loss": fm_loss.detach(),
        "change_loss": change_loss.detach(),
        "level_loss": level_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "latent_nll": latent_nll.detach(),
        "ortho": ortho.detach(),
        "sigma_mean": sigma.mean().detach(),
        "ec_gain_mean": cond["ec_gain"].mean().detach(),
    }
    return loss, metrics


def make_dataset(data_path: str, history_len: int, future_len: int, test_start: int, val_size: int, device: str):
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    _, h, w = surfaces.shape
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    return (train_hist, train_future, val_hist, val_future), h, w, h * w


def main() -> None:
    parser = argparse.ArgumentParser(description="264b-v0 epsilon-teacher state-space latent-factor FM")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--encoder_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--state_hidden", type=int, default=128)
    parser.add_argument("--flow_hidden", type=int, default=128)
    parser.add_argument("--flow_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--future_pos_embed", type=int, default=16)
    parser.add_argument("--flow_time_embed", type=int, default=16)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--sigma_floor", type=float, default=0.05)
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--ortho_reg_weight", type=float, default=0.01)
    parser.add_argument("--pinv_ridge", type=float, default=1e-4)
    parser.add_argument("--change_coord", type=str, default="asinh_local_scale", choices=["raw", "asinh_local_scale"])
    parser.add_argument("--change_scale_eps", type=float, default=1e-3)
    parser.add_argument("--ec_gain_max", type=float, default=0.20)
    parser.add_argument("--short_ec_boost_max", type=float, default=1.0)
    parser.add_argument("--short_ec_horizons", type=int, default=3)

    parser.add_argument("--lambda_change", type=float, default=0.75)
    parser.add_argument("--lambda_level", type=float, default=0.50)
    parser.add_argument("--lambda_terminal", type=float, default=0.50)
    parser.add_argument("--lambda_nll", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    tensors, h, w, dims = make_dataset(args.data_path, args.history_len, args.future_len, args.test_start, args.val_size, args.device)
    train_hist, train_future, val_hist, val_future = tensors
    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False, num_workers=0)

    cfg = JointStateSpaceLatentFactorFMEpsilonTeacherConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=dims,
        latent_dim=args.L,
        encoder_hidden=args.encoder_hidden,
        bottleneck_dim=args.bottleneck_dim,
        encoder_dropout=args.encoder_dropout,
        state_hidden=args.state_hidden,
        flow_hidden=args.flow_hidden,
        flow_layers=args.flow_layers,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
        model_dropout=args.model_dropout,
        future_pos_embed=args.future_pos_embed,
        flow_time_embed=args.flow_time_embed,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        sigma_floor=args.sigma_floor,
        ode_steps=args.ode_steps,
        ortho_reg_weight=args.ortho_reg_weight,
        pinv_ridge=args.pinv_ridge,
        change_coord=args.change_coord,
        change_scale_eps=args.change_scale_eps,
        ec_gain_max=args.ec_gain_max,
        short_ec_boost_max=args.short_ec_boost_max,
        short_ec_horizons=args.short_ec_horizons,
    )
    model = JointStateSpaceLatentFactorFMEpsilonTeacher(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history_records = []
    best_val = float("inf")
    best_epoch = -1
    best_path = out_dir / "best_model.pt"
    final_path = out_dir / "final_model.pt"

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_01, fut_01 in loader:
            hist_01 = hist_01.to(device, non_blocking=True)
            fut_01 = fut_01.to(device, non_blocking=True)
            hist_norm = normalize_iv(hist_01).view(hist_01.shape[0], hist_01.shape[1], -1)
            fut_norm = normalize_iv(fut_01).view(fut_01.shape[0], fut_01.shape[1], -1)
            with torch.set_grad_enabled(train_mode):
                loss, metrics = fm_step(
                    model,
                    hist_norm,
                    fut_norm,
                    lambda_change=args.lambda_change,
                    lambda_level=args.lambda_level,
                    lambda_terminal=args.lambda_terminal,
                    lambda_nll=args.lambda_nll,
                )
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v.item())
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in sums.items()}

    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={h} W={w} D={dims}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_fm": train_avg["fm_loss"],
            "train_change": train_avg["change_loss"],
            "train_level": train_avg["level_loss"],
            "train_terminal": train_avg["terminal_loss"],
            "train_nll": train_avg["latent_nll"],
            "train_sigma": train_avg["sigma_mean"],
            "train_ec_gain": train_avg["ec_gain_mean"],
            "val_total": val_avg["total"],
            "val_fm": val_avg["fm_loss"],
            "val_change": val_avg["change_loss"],
            "val_level": val_avg["level_loss"],
            "val_terminal": val_avg["terminal_loss"],
            "val_nll": val_avg["latent_nll"],
            "val_sigma": val_avg["sigma_mean"],
            "val_ec_gain": val_avg["ec_gain_mean"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history_records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} val={rec['val_total']:.5f} "
            f"chg={rec['val_change']:.4f} lvl={rec['val_level']:.4f} sigma={rec['val_sigma']:.3f} "
            f"ec={rec['val_ec_gain']:.3f} lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            best_epoch = epoch
            save_checkpoint(str(best_path), model, cfg, epoch, best_val)

    save_checkpoint(str(final_path), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(history_records, indent=2))
    summary = {"best_epoch": best_epoch, "best_val_total": best_val, "config": asdict(cfg)}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
