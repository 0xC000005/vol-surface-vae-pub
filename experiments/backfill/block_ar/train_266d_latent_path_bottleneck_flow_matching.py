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

from diffusion.block_ar.latent_path_bottleneck_flow_matching import (
    LatentPathBottleneckFM,
    LatentPathBottleneckFMConfig,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="266d-v0 temporal bottleneck with latent flow matching")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_tokens", type=int, default=5)
    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--history_hidden", type=int, default=64)
    parser.add_argument("--future_hidden", type=int, default=96)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", type=int, default=128)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--decoder_dropout", type=float, default=0.1)
    parser.add_argument("--pos_dim", type=int, default=32)
    parser.add_argument("--velocity_hidden", type=int, default=128)
    parser.add_argument("--velocity_layers", type=int, default=3)
    parser.add_argument("--velocity_dropout", type=float, default=0.1)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--token_kernel", type=int, default=3)
    parser.add_argument("--token_dilation", type=int, default=1)
    parser.add_argument("--flow_steps", type=int, default=16)

    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--fm_weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
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
    tensors, h, w, d = make_dataset(
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

    cfg = LatentPathBottleneckFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        context_dim=args.context_dim,
        latent_dim=args.latent_dim,
        latent_tokens=args.latent_tokens,
        history_hidden=args.history_hidden,
        future_hidden=args.future_hidden,
        encoder_dropout=args.encoder_dropout,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        decoder_dropout=args.decoder_dropout,
        pos_dim=args.pos_dim,
        velocity_hidden=args.velocity_hidden,
        velocity_layers=args.velocity_layers,
        velocity_dropout=args.velocity_dropout,
        time_dim=args.time_dim,
        token_kernel=args.token_kernel,
        token_dilation=args.token_dilation,
        flow_steps=args.flow_steps,
    )
    model = LatentPathBottleneckFM(cfg).to(device)
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
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(hist_01.shape[0], hist_01.shape[1], -1)
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(fut_01.shape[0], fut_01.shape[1], -1)
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(
                    hist_norm,
                    fut_norm,
                    recon_weight=args.recon_weight,
                    fm_weight=args.fm_weight,
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
    print(f"Grid: H={h} W={w} D={d}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_recon": train_avg["recon_loss"],
            "train_fm": train_avg["fm_loss"],
            "train_latent_std": train_avg["latent_std"],
            "val_total": val_avg["total"],
            "val_recon": val_avg["recon_loss"],
            "val_fm": val_avg["fm_loss"],
            "val_latent_std": val_avg["latent_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history_records.append(rec)
        print(
            f"[ep {epoch:03d}] "
            f"train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} "
            f"recon={rec['val_recon']:.5f} "
            f"fm={rec['val_fm']:.5f} "
            f"zstd={rec['val_latent_std']:.3f} "
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
