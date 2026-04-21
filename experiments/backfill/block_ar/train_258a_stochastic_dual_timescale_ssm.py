#!/usr/bin/env python
"""
258a-v0: stochastic dual-timescale latent state-space generator.
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
from diffusion.block_ar.stochastic_dual_timescale_ssm import (
    StochasticDualTimescaleSSM,
    StochasticDualTimescaleSSMConfig,
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


def sample_crps(samples: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    target_term = torch.abs(samples - target.unsqueeze(1))
    if weights is not None:
        target_term = target_term * weights.view(1, 1, weights.shape[1], 1)
    term1 = target_term.mean()
    pairwise = torch.abs(samples.unsqueeze(2) - samples.unsqueeze(1))
    if weights is not None:
        pairwise = pairwise * weights.view(1, 1, 1, weights.shape[1], 1)
    term2 = 0.5 * pairwise.mean()
    return term1 - term2


def build_losses(
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    sample_levels: torch.Tensor,
    aux: dict[str, torch.Tensor],
    model: StochasticDualTimescaleSSM,
    lambda_level: float,
    lambda_change: float,
    lambda_jump: float,
    lambda_terminal: float,
    lambda_ms_level: float,
    lambda_ms_change: float,
    lambda_ortho: float,
    lambda_scale: float,
    terminal_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _, T, _ = future_norm.shape
    h_weights = torch.linspace(
        1.0, terminal_weight, T, device=future_norm.device, dtype=future_norm.dtype
    ).view(1, T, 1)

    ens_mean_level = sample_levels.mean(dim=1)
    last_hist = history_norm[:, -1:, :]
    gt_prev = torch.cat([last_hist, future_norm[:, :-1]], dim=1)
    gt_change = future_norm - gt_prev

    sample_changes = sample_levels[:, :, 1:] - sample_levels[:, :, :-1]
    first_change = sample_levels[:, :, :1] - last_hist.unsqueeze(1)
    sample_changes = torch.cat([first_change, sample_changes], dim=2)
    ens_mean_change = sample_changes.mean(dim=1)

    level_loss = weighted_smooth_l1(ens_mean_level, future_norm, h_weights)
    change_loss = weighted_smooth_l1(ens_mean_change, gt_change, h_weights)
    pred_pmax = ens_mean_change.abs().amax(dim=1)
    gt_pmax = gt_change.abs().amax(dim=1)
    jump_loss = F.smooth_l1_loss(pred_pmax, gt_pmax)
    terminal_loss = F.smooth_l1_loss(ens_mean_level[:, -1, :], future_norm[:, -1, :])

    ms_level_loss = sample_crps(sample_levels, future_norm, h_weights)
    ms_change_loss = sample_crps(sample_changes, gt_change, h_weights)

    ortho = model.orthogonality_penalty(aux["loadings"])
    scale_reg = aux["slow_scale"].mean() + aux["fast_scale"].mean()

    total = (
        lambda_level * level_loss
        + lambda_change * change_loss
        + lambda_jump * jump_loss
        + lambda_terminal * terminal_loss
        + lambda_ms_level * ms_level_loss
        + lambda_ms_change * ms_change_loss
        + lambda_ortho * ortho
        + lambda_scale * scale_reg
    )
    metrics = {
        "level_loss": level_loss.detach(),
        "change_loss": change_loss.detach(),
        "jump_loss": jump_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "ms_level_loss": ms_level_loss.detach(),
        "ms_change_loss": ms_change_loss.detach(),
        "ortho": ortho.detach(),
        "scale_reg": scale_reg.detach(),
        "slow_scale_mean": aux["slow_scale"].mean().detach(),
        "fast_scale_mean": aux["fast_scale"].mean().detach(),
        "fast_gate_mean": aux["fast_gate"].mean().detach(),
        "common_rms": aux["mean_common"].pow(2).mean().sqrt().detach(),
        "idio_rms": aux["mean_idio"].pow(2).mean().sqrt().detach(),
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
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    D = H * W
    return (train_hist, train_future, val_hist, val_future), H, W, D


def main() -> None:
    parser = argparse.ArgumentParser(description="258a stochastic dual-timescale latent SSM")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--encoder_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--token_hidden", type=int, default=96)
    parser.add_argument("--slow_layers", type=int, default=3)
    parser.add_argument("--fast_layers", type=int, default=2)
    parser.add_argument("--slow_kernel", type=int, default=7)
    parser.add_argument("--fast_kernel", type=int, default=3)
    parser.add_argument("--fast_dilation", type=int, default=2)
    parser.add_argument("--branch_dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--obs_hidden", type=int, default=64)
    parser.add_argument("--time_embed_dim", type=int, default=16)
    parser.add_argument("--max_idio_ratio", type=float, default=0.30)
    parser.add_argument("--max_slow_noise", type=float, default=0.08)
    parser.add_argument("--max_fast_noise", type=float, default=0.16)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)

    parser.add_argument("--lambda_level", type=float, default=0.75)
    parser.add_argument("--lambda_change", type=float, default=1.25)
    parser.add_argument("--lambda_jump", type=float, default=0.50)
    parser.add_argument("--lambda_terminal", type=float, default=0.75)
    parser.add_argument("--lambda_ms_level", type=float, default=0.10)
    parser.add_argument("--lambda_ms_change", type=float, default=0.20)
    parser.add_argument("--lambda_ortho", type=float, default=1e-3)
    parser.add_argument("--lambda_scale", type=float, default=1e-3)
    parser.add_argument("--terminal_weight", type=float, default=2.0)
    parser.add_argument("--ms_samples", type=int, default=4)

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

    cfg = StochasticDualTimescaleSSMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=D,
        latent_dim=args.L,
        encoder_hidden=args.encoder_hidden,
        bottleneck_dim=args.bottleneck_dim,
        encoder_dropout=args.encoder_dropout,
        token_hidden=args.token_hidden,
        slow_layers=args.slow_layers,
        fast_layers=args.fast_layers,
        slow_kernel=args.slow_kernel,
        fast_kernel=args.fast_kernel,
        fast_dilation=args.fast_dilation,
        branch_dropout=args.branch_dropout,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        obs_hidden=args.obs_hidden,
        time_embed_dim=args.time_embed_dim,
        max_idio_ratio=args.max_idio_ratio,
        max_slow_noise=args.max_slow_noise,
        max_fast_noise=args.max_fast_noise,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
    )
    model = StochasticDualTimescaleSSM(cfg).to(device)
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
    history = []
    best_val = float("inf")
    best_path = Path(args.output_dir) / "best_model.pt"
    final_path = Path(args.output_dir) / "final_model.pt"
    global_step = 0

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        nonlocal global_step
        model.train() if train_mode else model.eval()
        sums: dict[str, float] = {}
        nb = 0
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            for hist_01, fut_flat in loader:
                hist_01 = hist_01.to(device, non_blocking=True)
                fut_flat = fut_flat.to(device, non_blocking=True)
                B = hist_01.shape[0]
                hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
                future_norm = fut_flat
                with autocast_ctx():
                    sample_levels, aux = model.draw_samples(hist_norm, n_samples=args.ms_samples)
                    loss, metrics = build_losses(
                        history_norm=hist_norm,
                        future_norm=future_norm,
                        sample_levels=sample_levels,
                        aux=aux,
                        model=model,
                        lambda_level=args.lambda_level,
                        lambda_change=args.lambda_change,
                        lambda_jump=args.lambda_jump,
                        lambda_terminal=args.lambda_terminal,
                        lambda_ms_level=args.lambda_ms_level,
                        lambda_ms_change=args.lambda_ms_change,
                        lambda_ortho=args.lambda_ortho,
                        lambda_scale=args.lambda_scale,
                        terminal_weight=args.terminal_weight,
                    )
                if train_mode:
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss")
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                for k, v in metrics.items():
                    sums[k] = sums.get(k, 0.0) + float(v.item())
                nb += 1
                global_step += 1
                if train_mode and global_step % args.log_every_n_steps == 0:
                    print(
                        f"  [step {global_step}] "
                        f"level={metrics['level_loss'].item():.4f} "
                        f"chg={metrics['change_loss'].item():.4f} "
                        f"msC={metrics['ms_change_loss'].item():.4f} "
                        f"slowS={metrics['slow_scale_mean'].item():.4f} "
                        f"fastS={metrics['fast_scale_mean'].item():.4f} "
                        f"loss={metrics['total'].item():.4f}"
                    )
        return {k: v / max(nb, 1) for k, v in sums.items()}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            val_avg = run_epoch(val_loader, train_mode=False)
        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_level={train_avg.get('level_loss', 0):.4f}  "
            f"train_chg={train_avg.get('change_loss', 0):.4f}  "
            f"train_msC={train_avg.get('ms_change_loss', 0):.4f}  "
            f"val_level={val_avg.get('level_loss', 0):.4f}  "
            f"val_chg={val_avg.get('change_loss', 0):.4f}  "
            f"val_msC={val_avg.get('ms_change_loss', 0):.4f}  "
            f"val_jump={val_avg.get('jump_loss', 0):.4f}  "
            f"val_fastS={val_avg.get('fast_scale_mean', 0):.4f}  "
            f"val_total={val_avg.get('total', 0):.4f}  "
            f"time={dt:.1f}s"
        )
        rec = {"epoch": epoch, "train": train_avg, "val": val_avg, "lr": float(optimizer.param_groups[0]["lr"]), "time_sec": dt}
        history.append(rec)
        history_path.write_text(json.dumps(history, indent=2))
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_total": best_val,
                    "config": config_to_dict(cfg),
                },
                best_path,
            )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_total": best_val,
            "config": config_to_dict(cfg),
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
