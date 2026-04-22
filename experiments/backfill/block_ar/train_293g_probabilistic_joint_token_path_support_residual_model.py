#!/usr/bin/env python
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

from diffusion.block_ar.probabilistic_joint_token_path_support_residual_model import (
    ProbabilisticJointTokenPathSupportResidualModel,
    ProbabilisticJointTokenPathSupportResidualModelConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_293d_probabilistic_joint_token_path_support_model import (
    build_coarse_codebook,
)


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
    _, h, w = surfaces.shape
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
    return (train_hist, train_future, val_hist, val_future), h, w, h * w


def compute_change_scale(history_norm: torch.Tensor, eps: float) -> torch.Tensor:
    hist_change = history_norm[:, 1:] - history_norm[:, :-1]
    scale = hist_change.pow(2).mean(dim=1, keepdim=True).sqrt()
    return scale.clamp_min(eps)


def build_knot_interpolation(future_len: int, n_knots: int) -> tuple[torch.Tensor, torch.Tensor]:
    knot_steps = torch.round(torch.linspace(0, future_len - 1, steps=n_knots)).long()
    weights = torch.zeros(future_len, n_knots, dtype=torch.float32)
    for step in range(future_len):
        if step <= int(knot_steps[0].item()):
            weights[step, 0] = 1.0
            continue
        if step >= int(knot_steps[-1].item()):
            weights[step, -1] = 1.0
            continue
        for knot_idx in range(n_knots - 1):
            left = int(knot_steps[knot_idx].item())
            right = int(knot_steps[knot_idx + 1].item())
            if left <= step <= right:
                alpha = float(step - left) / max(right - left, 1)
                weights[step, knot_idx] = 1.0 - alpha
                weights[step, knot_idx + 1] = alpha
                break
    return knot_steps, weights


def target_coords(history_norm: torch.Tensor, future_norm: torch.Tensor, eps: float) -> torch.Tensor:
    hist_window = history_norm.clone()
    curr = hist_window[:, -1]
    coords = []
    for step in range(future_norm.shape[1]):
        next_level = future_norm[:, step]
        raw_change = next_level - curr
        scale = compute_change_scale(hist_window, eps).squeeze(1)
        coord = torch.asinh(raw_change / scale)
        coords.append(coord)
        hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
        curr = next_level
    return torch.stack(coords, dim=1)


def scaffold_coords_for_training(
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    coarse_scaffold: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hist_window = history_norm.clone()
    future = future_norm
    curr = hist_window[:, -1]
    coords = []
    for step in range(future.shape[1]):
        scaffold_level = coarse_scaffold[:, step]
        raw_change = scaffold_level - curr
        scale = compute_change_scale(hist_window, eps).squeeze(1)
        coord = torch.asinh(raw_change / scale)
        coords.append(coord)
        next_level = future[:, step]
        hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
        curr = next_level
    return torch.stack(coords, dim=1)


def build_residual_codebook(
    train_hist: torch.Tensor,
    train_future: torch.Tensor,
    coarse_codebook: torch.Tensor,
    future_len: int,
    n_knots: int,
    codebook_size: int,
    change_scale_eps: float,
    max_points: int = 50000,
    iters: int = 20,
) -> torch.Tensor:
    hist_norm = normalize_iv(train_hist).view(train_hist.shape[0], train_hist.shape[1], -1)
    fut_norm = normalize_iv(train_future).view(train_future.shape[0], train_future.shape[1], -1)
    knot_steps, step_weights = build_knot_interpolation(future_len, n_knots)
    coarse_targets = fut_norm[:, knot_steps]
    flat = coarse_targets.reshape(coarse_targets.shape[0], -1)
    code_flat = coarse_codebook.reshape(coarse_codebook.shape[0], -1)
    dists = torch.cdist(flat.cpu(), code_flat.cpu())
    coarse_ids = dists.argmin(dim=-1)
    coarse_vecs = coarse_codebook[coarse_ids]
    coarse_scaffold = torch.einsum("tk,bkc->btc", step_weights, coarse_vecs)
    residual_coords = (
        target_coords(hist_norm, fut_norm, change_scale_eps)
        - scaffold_coords_for_training(hist_norm, fut_norm, coarse_scaffold, change_scale_eps)
    ).reshape(-1, hist_norm.shape[-1]).cpu()

    if residual_coords.shape[0] > max_points:
        idx = torch.randperm(residual_coords.shape[0])[:max_points]
        residual_coords = residual_coords[idx]
    n = residual_coords.shape[0]
    k = min(coarse_codebook.shape[0] * 2, n)
    k = min(k, codebook_size)
    perm = torch.randperm(n)[:k]
    centers = residual_coords[perm].clone()
    if k < codebook_size:
        extra = residual_coords[torch.randint(0, n, (codebook_size - k,))]
        centers = torch.cat([centers, extra], dim=0)
    for _ in range(iters):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(centers.shape[0], dtype=torch.float32)
        for start in range(0, n, 4096):
            batch = residual_coords[start : start + 4096]
            dists = torch.cdist(batch, centers)
            assign = dists.argmin(dim=-1)
            sums.index_add_(0, assign, batch)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float32))
        empty = counts == 0
        non_empty = ~empty
        centers[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(1)
        if empty.any():
            repl = residual_coords[torch.randint(0, n, (int(empty.sum().item()),))]
            centers[empty] = repl
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="293g-v0 probabilistic fixed-horizon joint-token path model with explicit coarse support and residual daily law"
    )
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--change_coord",
        type=str,
        default="asinh_local_scale",
        choices=["raw", "asinh_local_scale"],
    )
    parser.add_argument("--change_scale_eps", type=float, default=1e-3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--coarse_codebook_size", type=int, default=128)
    parser.add_argument("--n_knots", type=int, default=5)
    parser.add_argument("--coarse_loss_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--sample_temperature", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
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

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tensors, h, w, d = make_dataset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=str(device),
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

    print("Building coarse path codebook...")
    coarse_codebook = build_coarse_codebook(
        train_future=train_future.cpu(),
        future_len=args.future_len,
        n_knots=args.n_knots,
        coarse_codebook_size=args.coarse_codebook_size,
    )
    print("Building residual daily-change codebook...")
    residual_codebook = build_residual_codebook(
        train_hist=train_hist.cpu(),
        train_future=train_future.cpu(),
        coarse_codebook=coarse_codebook.cpu(),
        future_len=args.future_len,
        n_knots=args.n_knots,
        codebook_size=args.codebook_size,
        change_scale_eps=args.change_scale_eps,
    )
    torch.save(residual_codebook, out_dir / "residual_codebook.pt")
    torch.save(coarse_codebook, out_dir / "coarse_path_codebook.pt")

    cfg = ProbabilisticJointTokenPathSupportResidualModelConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        change_coord=args.change_coord,
        change_scale_eps=args.change_scale_eps,
        codebook_size=args.codebook_size,
        coarse_codebook_size=args.coarse_codebook_size,
        n_knots=args.n_knots,
        coarse_loss_weight=args.coarse_loss_weight,
        label_smoothing=args.label_smoothing,
        sample_temperature=args.sample_temperature,
    )
    model = ProbabilisticJointTokenPathSupportResidualModel(
        cfg,
        residual_codebook.to(device),
        coarse_codebook.to(device),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    history_records: list[dict[str, float]] = []
    best_path = out_dir / "best_model.pt"
    final_path = out_dir / "final_model.pt"

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_01, fut_01 in loader:
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                hist_01.shape[0], hist_01.shape[1], -1
            )
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                fut_01.shape[0], fut_01.shape[1], -1
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(hist_norm, fut_norm)
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
    print(f"Grid: H={h} W={w} D={d}")
    print(f"Residual codebook size: {args.codebook_size}")
    print(f"Coarse codebook size: {args.coarse_codebook_size}")
    print(f"Knots: {args.n_knots}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_token_nll": train_avg["token_nll"],
            "train_coarse_ce": train_avg["coarse_ce"],
            "train_token_acc": train_avg["token_acc"],
            "train_coarse_acc": train_avg["coarse_acc"],
            "train_token_entropy": train_avg["token_entropy"],
            "val_total": val_avg["total"],
            "val_token_nll": val_avg["token_nll"],
            "val_coarse_ce": val_avg["coarse_ce"],
            "val_token_acc": val_avg["token_acc"],
            "val_coarse_acc": val_avg["coarse_acc"],
            "val_token_entropy": val_avg["token_entropy"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history_records.append(rec)
        print(
            f"[ep {epoch:03d}] "
            f"train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"tok_acc={rec['val_token_acc']:.4f} "
            f"coarse_acc={rec['val_coarse_acc']:.4f} "
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
