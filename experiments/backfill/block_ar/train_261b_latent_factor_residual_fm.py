#!/usr/bin/env python
"""
261b-v0: latent-factor residual FM on top of frozen 260e.
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

from diffusion.block_ar.latent_factor_residual_fm import (
    LatentFactorResidualFM,
    LatentFactorResidualFMConfig,
    save_checkpoint,
)
from diffusion.block_ar.minimal_factor_fm import load_model as load_260_model
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


def build_raw_target_changes(history_norm: torch.Tensor, future_norm: torch.Tensor) -> torch.Tensor:
    first = future_norm[:, :1] - history_norm[:, -1:]
    rest = future_norm[:, 1:] - future_norm[:, :-1]
    return torch.cat([first, rest], dim=1)


def make_dataset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: str,
) -> tuple[tuple[torch.Tensor, ...], int]:
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    _, h, w = surfaces.shape
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    return (train_hist, train_future, val_hist, val_future), h * w


def fm_step(
    model: LatentFactorResidualFM,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    *,
    zero_mean_weight: float,
    zero_mean_samples: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz = history_norm.shape[0]
    with torch.no_grad():
        cond = model.condition(history_norm)
        raw_target_change = build_raw_target_changes(history_norm, future_norm)
        target_coord = model.base_model.transform_change(raw_target_change, history_norm)
        panel_resid_target = target_coord - cond["center_coord"]
        factor_target = torch.einsum("bld,btd->btl", cond["pinv"], panel_resid_target)

    x0 = torch.randn_like(factor_target)
    t = torch.rand(bsz, device=factor_target.device, dtype=factor_target.dtype)
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * x0 + t_view * factor_target
    target_v = factor_target - x0
    pred_v = model.velocity(x_t, t, cond)
    fm_loss = (pred_v - target_v).pow(2).mean()

    sampled_factor = model.sample_factor_residuals(
        history_norm,
        n_samples=zero_mean_samples,
        cond=cond,
    )
    zero_mean_loss = sampled_factor.mean(dim=1).pow(2).mean()
    panel_resid_coord = model.decode_panel_residual(sampled_factor, cond["loadings"])
    panel_resid_std = panel_resid_coord.std(dim=1).mean()

    loss = fm_loss + zero_mean_weight * zero_mean_loss
    metrics = {
        "total": loss.detach(),
        "fm_loss": fm_loss.detach(),
        "zero_mean_loss": zero_mean_loss.detach(),
        "panel_resid_std": panel_resid_std.detach(),
    }
    return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="261b-v0 latent-factor residual FM")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--model_hidden", type=int, default=96)
    parser.add_argument("--model_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--flow_time_embed", type=int, default=16)
    parser.add_argument("--future_pos_embed", type=int, default=16)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--pinv_ridge", type=float, default=1e-4)
    parser.add_argument("--zero_mean_weight", type=float, default=0.5)
    parser.add_argument("--zero_mean_samples", type=int, default=4)

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
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    tensors, dims = make_dataset(
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
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    base_model, _ = load_260_model(args.base_checkpoint, device)
    cfg = LatentFactorResidualFMConfig(
        base_checkpoint=args.base_checkpoint,
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=dims,
        latent_dim=base_model.cfg.latent_dim,
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
        ode_steps=args.ode_steps,
        pinv_ridge=args.pinv_ridge,
    )
    model = LatentFactorResidualFM(cfg, base_model=base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            with torch.set_grad_enabled(train_mode):
                loss, metrics = fm_step(
                    model,
                    hist_norm,
                    fut_norm,
                    zero_mean_weight=args.zero_mean_weight,
                    zero_mean_samples=args.zero_mean_samples,
                )
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in sums.items()}

    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Dims: {dims}  latent={cfg.latent_dim}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_fm": train_avg["fm_loss"],
            "train_zero_mean": train_avg["zero_mean_loss"],
            "train_panel_resid_std": train_avg["panel_resid_std"],
            "val_total": val_avg["total"],
            "val_fm": val_avg["fm_loss"],
            "val_zero_mean": val_avg["zero_mean_loss"],
            "val_panel_resid_std": val_avg["panel_resid_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history_records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} val={rec['val_total']:.5f} "
            f"zero={rec['val_zero_mean']:.5f} pstd={rec['val_panel_resid_std']:.5f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
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
