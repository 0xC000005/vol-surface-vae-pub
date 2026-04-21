#!/usr/bin/env python
"""
260a-v0: minimal conditional factor flow-matching baseline.

The restart baseline is intentionally plain:
- future change path target
- vanilla conditional flow matching objective
- single temporal backbone
- explicit low-rank readout
- bounded idio path
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.minimal_factor_fm import (
    MinimalFactorFM,
    MinimalFactorFMConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


def build_raw_target_changes(history_norm: torch.Tensor, future_norm: torch.Tensor) -> torch.Tensor:
    first = future_norm[:, :1] - history_norm[:, -1:]
    rest = future_norm[:, 1:] - future_norm[:, :-1]
    return torch.cat([first, rest], dim=1)


def fm_step(
    model: MinimalFactorFM,
    history_norm: torch.Tensor,
    target_change_coord: torch.Tensor,
    future_norm: torch.Tensor,
    level_path_weight: float,
    terminal_level_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    B = target_change_coord.shape[0]
    cond = model.condition(history_norm)
    x0 = torch.randn_like(target_change_coord)
    t = torch.rand(B, device=target_change_coord.device, dtype=target_change_coord.dtype)
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * x0 + t_view * target_change_coord
    target_v = target_change_coord - x0
    pred_v, aux = model.velocity(x_t, t, cond)

    fm_loss = (pred_v - target_v).pow(2).mean()
    common_rms = aux["common"].pow(2).mean().sqrt()
    idio_rms = aux["idio"].pow(2).mean().sqrt()
    idio_ratio = idio_rms / (common_rms + 1e-8)
    ortho = model.ortho_penalty(aux["loadings"])
    level_path_loss = torch.zeros((), device=history_norm.device, dtype=history_norm.dtype)
    terminal_level_loss = torch.zeros((), device=history_norm.device, dtype=history_norm.dtype)
    if level_path_weight > 0.0 or terminal_level_weight > 0.0:
        _, center_levels = model.deterministic_center_path(history_norm)
        if level_path_weight > 0.0:
            level_path_loss = torch.nn.functional.smooth_l1_loss(center_levels, future_norm)
        if terminal_level_weight > 0.0:
            terminal_level_loss = torch.nn.functional.smooth_l1_loss(center_levels[:, -1], future_norm[:, -1])
    loss = (
        fm_loss
        + model.cfg.ortho_reg_weight * ortho
        + level_path_weight * level_path_loss
        + terminal_level_weight * terminal_level_loss
    )
    metrics = {
        "total": loss.detach(),
        "fm_loss": fm_loss.detach(),
        "idio_ratio": idio_ratio.detach(),
        "common_rms": common_rms.detach(),
        "idio_rms": idio_rms.detach(),
        "ortho": ortho.detach(),
        "level_path_loss": level_path_loss.detach(),
        "terminal_level_loss": terminal_level_loss.detach(),
    }
    return loss, metrics


def make_dataset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: str,
) -> tuple[tuple[torch.Tensor, ...], int, int, int]:
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    _, H, W = surfaces.shape
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, history_len, future_len
    )
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, history_len, future_len
    )
    D = H * W
    return (train_hist, train_future, val_hist, val_future), H, W, D


def main() -> None:
    parser = argparse.ArgumentParser(description="260a-v0 minimal conditional factor FM")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--encoder_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--model_hidden", type=int, default=128)
    parser.add_argument("--model_layers", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--flow_time_embed", type=int, default=16)
    parser.add_argument("--future_pos_embed", type=int, default=16)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--max_idio_ratio", type=float, default=0.25)
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--ortho_reg_weight", type=float, default=0.01)
    parser.add_argument("--change_coord", type=str, default="raw", choices=["raw", "asinh_local_scale"])
    parser.add_argument("--change_scale_eps", type=float, default=1e-3)
    parser.add_argument("--ec_anchor_mode", type=str, default="none", choices=["none", "history_mean", "learned_history_residual"])
    parser.add_argument("--ec_gain_max", type=float, default=0.0)
    parser.add_argument("--anchor_delta_mult", type=float, default=0.0)
    parser.add_argument("--short_ec_boost_max", type=float, default=0.0)
    parser.add_argument("--short_ec_horizons", type=int, default=0)
    parser.add_argument("--level_path_weight", type=float, default=0.0)
    parser.add_argument("--terminal_level_weight", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=40)
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)

    tensors, H, W, D = make_dataset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=args.device,
    )
    train_hist, train_future, val_hist, val_future = tensors
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    cfg = MinimalFactorFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=D,
        latent_dim=args.L,
        encoder_hidden=args.encoder_hidden,
        bottleneck_dim=args.bottleneck_dim,
        model_hidden=args.model_hidden,
        model_layers=args.model_layers,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
        model_dropout=args.model_dropout,
        flow_time_embed=args.flow_time_embed,
        future_pos_embed=args.future_pos_embed,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        max_idio_ratio=args.max_idio_ratio,
        ode_steps=args.ode_steps,
        ortho_reg_weight=args.ortho_reg_weight,
        change_coord=args.change_coord,
        change_scale_eps=args.change_scale_eps,
        ec_anchor_mode=args.ec_anchor_mode,
        ec_gain_max=args.ec_gain_max,
        anchor_delta_mult=args.anchor_delta_mult,
        short_ec_boost_max=args.short_ec_boost_max,
        short_ec_horizons=args.short_ec_horizons,
    )
    model = MinimalFactorFM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history_records: list[dict[str, float]] = []
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
            raw_target_change = build_raw_target_changes(hist_norm, fut_norm)
            raw_target_change = model.residualize_raw_change(raw_target_change, hist_norm, fut_norm)
            target_change = model.transform_change(raw_target_change, hist_norm)
            with torch.set_grad_enabled(train_mode):
                loss, metrics = fm_step(
                    model,
                    hist_norm,
                    target_change,
                    fut_norm,
                    level_path_weight=args.level_path_weight,
                    terminal_level_weight=args.terminal_level_weight,
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
    print(f"Grid: H={H} W={W} D={D}")
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
            "train_idio_ratio": train_avg["idio_ratio"],
            "train_level_path": train_avg["level_path_loss"],
            "train_terminal_level": train_avg["terminal_level_loss"],
            "val_total": val_avg["total"],
            "val_fm": val_avg["fm_loss"],
            "val_idio_ratio": val_avg["idio_ratio"],
            "val_level_path": val_avg["level_path_loss"],
            "val_terminal_level": val_avg["terminal_level_loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history_records.append(rec)
        print(
            f"[ep {epoch:03d}] "
            f"train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"lvl={rec['val_level_path']:.4f} "
            f"idio={rec['val_idio_ratio']:.3f} "
            f"lr={rec['lr']:.2e} "
            f"time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            best_epoch = epoch
            save_checkpoint(str(best_path), model, cfg, epoch, best_val)

    save_checkpoint(str(final_path), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(history_records, indent=2))
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "config": asdict(cfg),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
