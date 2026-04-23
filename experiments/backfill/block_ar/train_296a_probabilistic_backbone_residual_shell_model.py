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

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    load_model as load_277d_model,
)
from diffusion.block_ar.probabilistic_backbone_residual_shell_model import (
    ProbabilisticBackboneResidualShellConfig,
    ResidualShellCore,
    compute_residual_coords,
    flatten_panels,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
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


@torch.no_grad()
def build_backbone_center_paths(
    backbone: torch.nn.Module,
    history_01: torch.Tensor,
    future_len: int,
    batch_size: int,
) -> torch.Tensor:
    centers: list[torch.Tensor] = []
    for start in range(0, history_01.shape[0], batch_size):
        hist_batch = history_01[start : start + batch_size]
        hist_norm = normalize_iv(hist_batch)
        center = backbone.sample_batched(
            hist_norm,
            n_samples=1,
            n_steps=future_len,
            chunk_size=max(1, min(8, batch_size)),
            history_is_normalized=True,
        ).squeeze(1)
        centers.append(center)
    return torch.cat(centers, dim=0)


def build_residual_codebook(
    train_hist: torch.Tensor,
    train_center: torch.Tensor,
    train_future: torch.Tensor,
    codebook_size: int,
    change_coord: str,
    change_scale_eps: float,
    max_points: int = 50000,
    iters: int = 20,
) -> torch.Tensor:
    hist_norm = flatten_panels(normalize_iv(train_hist))
    center_norm = flatten_panels(normalize_iv(train_center))
    future_norm = flatten_panels(normalize_iv(train_future))
    residual_coords = compute_residual_coords(
        hist_norm,
        center_norm,
        future_norm,
        eps=change_scale_eps,
        change_coord=change_coord,
    ).reshape(-1, hist_norm.shape[-1]).cpu()
    if residual_coords.shape[0] > max_points:
        idx = torch.randperm(residual_coords.shape[0])[:max_points]
        residual_coords = residual_coords[idx]
    n = residual_coords.shape[0]
    k = min(codebook_size, n)
    perm = torch.randperm(n)[:k]
    centers = residual_coords[perm].clone()
    if k < codebook_size:
        extra = residual_coords[torch.randint(0, n, (codebook_size - k,))]
        centers = torch.cat([centers, extra], dim=0)
    for _ in range(iters):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(codebook_size, dtype=torch.float32)
        for start in range(0, n, 4096):
            batch = residual_coords[start : start + 4096]
            dists = torch.cdist(batch, centers)
            assign = dists.argmin(dim=-1)
            sums.index_add_(0, assign, batch)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float32))
        mask = counts > 0
        centers[mask] = sums[mask] / counts[mask].unsqueeze(-1)
    return centers.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="296a-v0 probabilistic residual shell around frozen 277d backbone"
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
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--sample_temperature", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument(
        "--backbone_checkpoint",
        type=str,
        default="models/backfill/277d_v0_s42/best_model.pt",
    )
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

    backbone, _ = load_277d_model(args.backbone_checkpoint, device)
    backbone.eval()
    print("Building frozen 277d center paths...")
    train_center = build_backbone_center_paths(
        backbone, train_hist, future_len=args.future_len, batch_size=args.batch_size
    )
    val_center = build_backbone_center_paths(
        backbone, val_hist, future_len=args.future_len, batch_size=args.batch_size
    )
    torch.save(train_center.cpu(), out_dir / "train_center_future_01.pt")
    torch.save(val_center.cpu(), out_dir / "val_center_future_01.pt")

    train_loader = DataLoader(
        TensorDataset(train_hist, train_center, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_center, val_future),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Building residual codebook...")
    codebook = build_residual_codebook(
        train_hist=train_hist.cpu(),
        train_center=train_center.cpu(),
        train_future=train_future.cpu(),
        codebook_size=args.codebook_size,
        change_coord=args.change_coord,
        change_scale_eps=args.change_scale_eps,
    )
    torch.save(codebook, out_dir / "residual_codebook.pt")

    cfg = ProbabilisticBackboneResidualShellConfig(
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
        label_smoothing=args.label_smoothing,
        sample_temperature=args.sample_temperature,
    )
    model = ResidualShellCore(cfg, codebook.to(device)).to(device)
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
        for hist_01, center_01, fut_01 in loader:
            hist_norm = flatten_panels(normalize_iv(hist_01.to(device, non_blocking=True)))
            center_norm = flatten_panels(normalize_iv(center_01.to(device, non_blocking=True)))
            fut_norm = flatten_panels(normalize_iv(fut_01.to(device, non_blocking=True)))
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(hist_norm, center_norm, fut_norm)
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
            "train_token_acc": train_avg["token_acc"],
            "train_token_entropy": train_avg["token_entropy"],
            "train_residual_raw_mae": train_avg["residual_raw_mae"],
            "train_center_raw_abs_mean": train_avg["center_raw_abs_mean"],
            "train_residual_coord_abs_mean": train_avg["residual_coord_abs_mean"],
            "val_total": val_avg["total"],
            "val_token_nll": val_avg["token_nll"],
            "val_token_acc": val_avg["token_acc"],
            "val_token_entropy": val_avg["token_entropy"],
            "val_residual_raw_mae": val_avg["residual_raw_mae"],
            "val_center_raw_abs_mean": val_avg["center_raw_abs_mean"],
            "val_residual_coord_abs_mean": val_avg["residual_coord_abs_mean"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history_records.append(rec)
        print(
            f"[ep {epoch:03d}] "
            f"train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"acc={rec['val_token_acc']:.4f} "
            f"raw_mae={rec['val_residual_raw_mae']:.4f} "
            f"entropy={rec['val_token_entropy']:.3f} "
            f"lr={rec['lr']:.2e} "
            f"time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            best_epoch = epoch
            save_checkpoint(
                str(best_path),
                model,
                cfg,
                epoch,
                best_val,
                backbone_checkpoint_path=args.backbone_checkpoint,
            )

    save_checkpoint(
        str(final_path),
        model,
        cfg,
        args.epochs,
        best_val,
        backbone_checkpoint_path=args.backbone_checkpoint,
    )
    (out_dir / "training_history.json").write_text(json.dumps(history_records, indent=2))
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "config": asdict(cfg),
        "backbone_checkpoint": args.backbone_checkpoint,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
