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

from diffusion.block_ar.deterministic_obs_encoded_latent_world_model import (
    load_model as load_289c_model,
)
from diffusion.block_ar.probabilistic_obs_world_profiled_zero_mean_coarse_shell_model import (
    ObsWorldProfiledZeroMeanCoarseShellCore,
    ProbabilisticObsWorldProfiledZeroMeanCoarseShellConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_296b_probabilistic_backbone_zero_mean_coarse_shell_model import (
    build_backbone_center_paths,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="308a-v0 profiled zero-mean coarse shell around learned 289c-style world center"
    )
    parser.add_argument("--d_model", type=int, default=160)
    parser.add_argument("--nhead", type=int, default=5)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--change_coord", type=str, default="asinh_local_scale")
    parser.add_argument("--change_scale_eps", type=float, default=1e-3)
    parser.add_argument("--knot_positions", type=str, default="1,5,10,15,22,30")
    parser.add_argument("--scale_floor", type=float, default=0.02)
    parser.add_argument("--profile_log_amp", type=float, default=0.8)
    parser.add_argument("--sample_scale_mult", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument(
        "--backbone_checkpoint",
        type=str,
        default="models/backfill/308a_stage1_289c_v0_s42/best_model.pt",
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

    knot_positions = tuple(int(x.strip()) for x in args.knot_positions.split(",") if x.strip())
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

    backbone, _ = load_289c_model(args.backbone_checkpoint, device)
    backbone.eval()
    print("Building learned world-model center paths...")
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

    cfg = ProbabilisticObsWorldProfiledZeroMeanCoarseShellConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        change_coord=args.change_coord,
        change_scale_eps=args.change_scale_eps,
        knot_positions=knot_positions,
        scale_floor=args.scale_floor,
        profile_log_amp=args.profile_log_amp,
        sample_scale_mult=args.sample_scale_mult,
    )
    model = ObsWorldProfiledZeroMeanCoarseShellCore(cfg).to(device)
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
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                hist_01.shape[0], hist_01.shape[1], -1
            )
            center_norm = normalize_iv(center_01.to(device, non_blocking=True)).view(
                center_01.shape[0], center_01.shape[1], -1
            )
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                fut_01.shape[0], fut_01.shape[1], -1
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(hist_norm, center_norm, fut_norm)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
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
    print(f"Knot positions: {knot_positions}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_control_abs_mean": train_avg["control_abs_mean"],
            "train_pred_scale_mean": train_avg["pred_scale_mean"],
            "train_pred_scale_daily_mean": train_avg["pred_scale_daily_mean"],
            "train_profile_scale_mean": train_avg["profile_scale_mean"],
            "train_profile_scale_h1": train_avg["profile_scale_h1"],
            "train_z_abs_mean": train_avg["z_abs_mean"],
            "val_total": val_avg["total"],
            "val_control_abs_mean": val_avg["control_abs_mean"],
            "val_pred_scale_mean": val_avg["pred_scale_mean"],
            "val_pred_scale_daily_mean": val_avg["pred_scale_daily_mean"],
            "val_profile_scale_mean": val_avg["profile_scale_mean"],
            "val_profile_scale_h1": val_avg["profile_scale_h1"],
            "val_z_abs_mean": val_avg["z_abs_mean"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history_records.append(rec)
        print(
            f"[ep {epoch:03d}] "
            f"train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"scale={rec['val_pred_scale_mean']:.4f} "
            f"profile={rec['val_profile_scale_mean']:.4f} "
            f"h1prof={rec['val_profile_scale_h1']:.4f} "
            f"z={rec['val_z_abs_mean']:.4f} "
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
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
