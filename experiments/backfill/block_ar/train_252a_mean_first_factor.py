#!/usr/bin/env python
"""
252a: mean-first explicit factor prototype.

Goal:
    Attack the oracle-reachable suites by improving deterministic center-path quality
    first, before reintroducing stochastic uncertainty.

Design:
    - same broad non-AR low-rank family as 250/251
    - deterministic center path only
    - no SDE / no latent FM / no marginal head / no width losses

Loss stack:
    total =
        lambda_level   * weighted SmoothL1(level)
      + lambda_change  * weighted SmoothL1(change)
      + lambda_jump    * SmoothL1(pathwise max |change|)
      + lambda_terminal* SmoothL1(final horizon level)
      + lambda_ortho   * Lambda orthogonality penalty
      + lambda_idio    * mean_idio L2 penalty

This is intentionally narrow: one decisive prototype for the 4 deterministic-gap suites.
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

from diffusion.block_ar.mean_first_factor import MeanFirstFactorConfig, MeanFirstFactorModel
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def build_point_losses(
    pred_level: torch.Tensor,
    gt_level: torch.Tensor,
    aux: dict[str, torch.Tensor],
    lambda_level: float,
    lambda_change: float,
    lambda_jump: float,
    lambda_terminal: float,
    lambda_ortho: float,
    lambda_idio: float,
    terminal_weight: float,
    model: MeanFirstFactorModel,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    B, T, D = pred_level.shape
    h_weights = torch.linspace(
        1.0, terminal_weight, T, device=pred_level.device, dtype=pred_level.dtype
    ).view(1, T, 1)

    pred_change = pred_level[:, 1:] - pred_level[:, :-1]
    gt_change = gt_level[:, 1:] - gt_level[:, :-1]
    chg_weights = h_weights[:, 1:, :]

    level_loss = weighted_smooth_l1(pred_level, gt_level, h_weights)
    change_loss = weighted_smooth_l1(pred_change, gt_change, chg_weights)

    pred_pmax = pred_change.abs().amax(dim=1)
    gt_pmax = gt_change.abs().amax(dim=1)
    jump_loss = F.smooth_l1_loss(pred_pmax, gt_pmax)

    terminal_loss = F.smooth_l1_loss(pred_level[:, -1, :], gt_level[:, -1, :])
    ortho = model.orthogonality_penalty(aux["Lambda"])
    idio_l2 = aux["mean_idio"].pow(2).mean()

    total = (
        lambda_level * level_loss
        + lambda_change * change_loss
        + lambda_jump * jump_loss
        + lambda_terminal * terminal_loss
        + lambda_ortho * ortho
        + lambda_idio * idio_l2
    )
    metrics = {
        "level_loss": level_loss.detach(),
        "change_loss": change_loss.detach(),
        "jump_loss": jump_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "ortho": ortho.detach(),
        "idio_l2": idio_l2.detach(),
        "total": total.detach(),
    }
    return total, metrics


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
    parser = argparse.ArgumentParser(description="252a: mean-first explicit factor prototype")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--encoder_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)

    parser.add_argument("--lambda_level", type=float, default=1.0)
    parser.add_argument("--lambda_change", type=float, default=1.0)
    parser.add_argument("--lambda_jump", type=float, default=0.5)
    parser.add_argument("--lambda_terminal", type=float, default=1.0)
    parser.add_argument("--lambda_ortho", type=float, default=1e-3)
    parser.add_argument("--lambda_idio", type=float, default=1e-4)
    parser.add_argument("--terminal_weight", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--clip_grad", type=float, default=1.0)

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
    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={H}  W={W}  D={D}")

    cfg = MeanFirstFactorConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=D,
        latent_dim=args.L,
        encoder_hidden=args.encoder_hidden,
        bottleneck_dim=args.bottleneck_dim,
        encoder_dropout=args.encoder_dropout,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
    )
    model = MeanFirstFactorModel(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    device_is_cuda = device.type == "cuda"
    use_bf16 = bool(args.bf16) and device_is_cuda

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", enabled=False) if device_is_cuda else torch.autocast(device_type="cpu", enabled=False)

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
                    loss, metrics = build_point_losses(
                        pred_level=pred,
                        gt_level=fut_flat,
                        aux=aux,
                        lambda_level=args.lambda_level,
                        lambda_change=args.lambda_change,
                        lambda_jump=args.lambda_jump,
                        lambda_terminal=args.lambda_terminal,
                        lambda_ortho=args.lambda_ortho,
                        lambda_idio=args.lambda_idio,
                        terminal_weight=args.terminal_weight,
                        model=model,
                    )
                if train_mode:
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_idx}")
                    (loss / args.grad_accum).backward()
                    if (batch_idx + 1) % args.grad_accum == 0:
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
                        f"term={metrics['terminal_loss'].item():.4f} "
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
            f"val_total={val_avg.get('total', 0):.4f}  "
            f"({dt:.1f}s)"
        )

        payload = {
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_avg,
            "val_metrics": val_avg,
        }
        if val_avg.get("total", float("inf")) < best_val:
            best_val = val_avg["total"]
            torch.save(payload, best_path)
            print(f"    best (val_total={best_val:.4f}) saved to {best_path}")

        history.append({
            "epoch": epoch,
            "train": train_avg,
            "val": val_avg,
            "time_sec": dt,
        })
        history_path.write_text(json.dumps(history, indent=2))

    final_payload = {
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
    }
    torch.save(final_payload, final_path)
    print(f"\nFinal checkpoint saved to {final_path}")
    print(f"Best val total: {best_val:.4f}")


if __name__ == "__main__":
    main()
