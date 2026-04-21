#!/usr/bin/env python
"""
255a-v0: deterministic sparse future-motif model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.single_pass_ar import normalize_iv
from diffusion.block_ar.sparse_future_motif_model import (
    SparseFutureMotifModel,
    SparseFutureMotifConfig,
    config_to_dict,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss.mean()


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
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    D = H * W
    return (train_hist, train_future, val_hist, val_future), H, W, D


def build_losses(
    history_norm: torch.Tensor,
    pred_level: torch.Tensor,
    gt_level: torch.Tensor,
    aux: dict[str, torch.Tensor],
    lambda_level: float,
    lambda_change: float,
    lambda_jump: float,
    lambda_terminal: float,
    lambda_resid: float,
    lambda_entropy: float,
    route_entropy_floor: float,
    terminal_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _, T, _ = pred_level.shape
    h_weights = torch.linspace(
        1.0, terminal_weight, T, device=pred_level.device, dtype=pred_level.dtype
    ).view(1, T, 1)

    pred_change = aux["mean_change"]
    last_hist = history_norm[:, -1:, :]
    gt_prev = torch.cat([last_hist, gt_level[:, :-1]], dim=1)
    gt_change = gt_level - gt_prev

    level_loss = weighted_smooth_l1(pred_level, gt_level, h_weights)
    change_loss = weighted_smooth_l1(pred_change, gt_change, h_weights)
    pred_pmax = pred_change.abs().amax(dim=1)
    gt_pmax = gt_change.abs().amax(dim=1)
    jump_loss = F.smooth_l1_loss(pred_pmax, gt_pmax)
    terminal_loss = F.smooth_l1_loss(pred_level[:, -1, :], gt_level[:, -1, :])

    resid_rms = aux["mean_resid"].pow(2).mean().sqrt()
    p = aux["motif_weights"].clamp_min(1e-8)
    entropy = -(p * torch.log(p)).sum(dim=-1).mean()
    entropy_floor_penalty = torch.relu(route_entropy_floor - entropy).square()

    total = (
        lambda_level * level_loss
        + lambda_change * change_loss
        + lambda_jump * jump_loss
        + lambda_terminal * terminal_loss
        + lambda_resid * resid_rms
        + lambda_entropy * entropy_floor_penalty
    )
    metrics = {
        "level_loss": level_loss.detach(),
        "change_loss": change_loss.detach(),
        "jump_loss": jump_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "resid_rms": resid_rms.detach(),
        "route_entropy": entropy.detach(),
        "entropy_floor_penalty": entropy_floor_penalty.detach(),
        "motif_top1": aux["motif_weights"].max(dim=-1).values.mean().detach(),
        "total": total.detach(),
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="255a-v0 sparse future motif model")
    parser.add_argument("--n_motifs", type=int, default=16)
    parser.add_argument("--route_topk", type=int, default=4)
    parser.add_argument("--route_temperature", type=float, default=1.0)
    parser.add_argument("--encoder_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--route_hidden", type=int, default=256)
    parser.add_argument("--route_layers", type=int, default=2)
    parser.add_argument("--route_dropout", type=float, default=0.1)
    parser.add_argument("--route_entropy_floor", type=float, default=0.50)
    parser.add_argument("--adapter_hidden", type=int, default=128)
    parser.add_argument("--adapter_layers", type=int, default=2)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--obs_hidden", type=int, default=64)
    parser.add_argument("--time_embed_dim", type=int, default=16)
    parser.add_argument("--max_resid_ratio", type=float, default=0.25)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)

    parser.add_argument("--lambda_level", type=float, default=0.75)
    parser.add_argument("--lambda_change", type=float, default=1.25)
    parser.add_argument("--lambda_jump", type=float, default=0.50)
    parser.add_argument("--lambda_terminal", type=float, default=0.75)
    parser.add_argument("--lambda_resid", type=float, default=0.05)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--terminal_weight", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "none"])

    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))

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
    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={H}  W={W}  D={D}")

    cfg = SparseFutureMotifConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=D,
        encoder_hidden=args.encoder_hidden,
        bottleneck_dim=args.bottleneck_dim,
        encoder_dropout=args.encoder_dropout,
        n_motifs=args.n_motifs,
        route_hidden=args.route_hidden,
        route_layers=args.route_layers,
        route_dropout=args.route_dropout,
        route_temperature=args.route_temperature,
        route_topk=args.route_topk,
        route_entropy_floor=args.route_entropy_floor,
        adapter_hidden=args.adapter_hidden,
        adapter_layers=args.adapter_layers,
        adapter_dropout=args.adapter_dropout,
        obs_hidden=args.obs_hidden,
        time_embed_dim=args.time_embed_dim,
        max_resid_ratio=args.max_resid_ratio,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
    )
    model = SparseFutureMotifModel(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    device_is_cuda = device.type == "cuda"
    use_bf16 = bool(args.bf16) and device_is_cuda

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return (
            torch.autocast(device_type="cuda", enabled=False)
            if device_is_cuda
            else torch.autocast(device_type="cpu", enabled=False)
        )

    history_path = Path(args.output_dir) / "training_history.json"
    history: list[dict] = []
    best_val = float("inf")
    best_path = Path(args.output_dir) / "best_model.pt"
    final_path = Path(args.output_dir) / "final_model.pt"
    global_step = 0

    def run_epoch(loader: DataLoader, train_mode: bool, epoch: int) -> dict[str, float]:
        nonlocal global_step
        model.train() if train_mode else model.eval()
        sums: dict[str, float] = {}
        nb = 0
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            for batch_idx, (hist_01, fut_flat) in enumerate(loader):
                hist_01 = hist_01.to(device, non_blocking=True)
                fut_flat = fut_flat.to(device, non_blocking=True)
                B = hist_01.shape[0]
                hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
                with autocast_ctx():
                    pred, aux = model(hist_norm)
                    loss, metrics = build_losses(
                        history_norm=hist_norm,
                        pred_level=pred,
                        gt_level=fut_flat,
                        aux=aux,
                        lambda_level=args.lambda_level,
                        lambda_change=args.lambda_change,
                        lambda_jump=args.lambda_jump,
                        lambda_terminal=args.lambda_terminal,
                        lambda_resid=args.lambda_resid,
                        lambda_entropy=args.lambda_entropy,
                        route_entropy_floor=args.route_entropy_floor,
                        terminal_weight=args.terminal_weight,
                    )
                if train_mode:
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_idx}")
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                for k, v in metrics.items():
                    sums[k] = sums.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
                nb += 1
                global_step += 1
                if train_mode and global_step % args.log_every_n_steps == 0:
                    print(
                        f"  [ep {epoch} step {global_step}] "
                        f"level={metrics['level_loss'].item():.4f} "
                        f"chg={metrics['change_loss'].item():.4f} "
                        f"jump={metrics['jump_loss'].item():.4f} "
                        f"top1={metrics['motif_top1'].item():.4f} "
                        f"H={metrics['route_entropy'].item():.4f} "
                        f"loss={metrics['total'].item():.4f}"
                    )
        return {k: v / max(nb, 1) for k, v in sums.items()}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True, epoch=epoch)
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            val_avg = run_epoch(val_loader, train_mode=False, epoch=epoch)
        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_level={train_avg.get('level_loss', 0):.4f}  "
            f"train_chg={train_avg.get('change_loss', 0):.4f}  "
            f"val_level={val_avg.get('level_loss', 0):.4f}  "
            f"val_chg={val_avg.get('change_loss', 0):.4f}  "
            f"val_jump={val_avg.get('jump_loss', 0):.4f}  "
            f"val_top1={val_avg.get('motif_top1', 0):.4f}  "
            f"val_H={val_avg.get('route_entropy', 0):.4f}  "
            f"val_total={val_avg.get('total', 0):.4f}  "
            f"time={dt:.1f}s"
        )
        rec = {"epoch": epoch, "train": train_avg, "val": val_avg, "lr": float(optimizer.param_groups[0]["lr"]), "time_sec": dt}
        history.append(rec)
        history_path.write_text(json.dumps(history, indent=2))
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_total": best_val, "config": config_to_dict(cfg)}, best_path)
    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_total": best_val, "config": config_to_dict(cfg)}, final_path)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
