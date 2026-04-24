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

from diffusion.block_ar.shared_latent_implicit_future_logit_path_generator import (
    SharedLatentImplicitFutureLogitPathGenerator,
    SharedLatentImplicitPathGeneratorConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_320a_future_scalar_ar_mixture_density_model import (
    compute_level_logit_stats,
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
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    return (train_hist, train_future, val_hist, val_future), h, w, h * w


def main() -> None:
    parser = argparse.ArgumentParser(description="326a-v0 shared-latent implicit future-logit path generator")
    parser.add_argument("--context_dim", type=int, default=160)
    parser.add_argument("--history_hidden", type=int, default=96)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--latent_tokens", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--use_local_noise", action="store_true")
    parser.add_argument("--demean_local_noise", action="store_true")
    parser.add_argument("--token_dim", type=int, default=96)
    parser.add_argument("--token_layers", type=int, default=4)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=192)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--flow_time_dim", type=int, default=32)
    parser.add_argument("--logit_eps", type=float, default=1e-4)
    parser.add_argument("--flow_steps", type=int, default=1)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--max_sample_chunk", type=int, default=32)
    parser.add_argument("--standardize_logits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit_std_floor", type=float, default=1e-3)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--score_eps", type=float, default=1e-6)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    cfg = SharedLatentImplicitPathGeneratorConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        context_dim=args.context_dim,
        history_hidden=args.history_hidden,
        encoder_dropout=args.encoder_dropout,
        latent_tokens=args.latent_tokens,
        latent_dim=args.latent_dim,
        use_local_noise=args.use_local_noise,
        demean_local_noise=args.demean_local_noise,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_heads=args.token_heads,
        token_ff=args.token_ff,
        model_dropout=args.model_dropout,
        flow_time_dim=args.flow_time_dim,
        logit_eps=args.logit_eps,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
        max_sample_chunk=args.max_sample_chunk,
        standardize_logits=args.standardize_logits,
        logit_std_floor=args.logit_std_floor,
        train_sample_count=args.train_sample_count,
        score_eps=args.score_eps,
    )
    model = SharedLatentImplicitFutureLogitPathGenerator(cfg).to(device)
    if args.standardize_logits:
        logit_mean, logit_std = compute_level_logit_stats(train_hist, train_future, logit_eps=args.logit_eps, std_floor=args.logit_std_floor)
        model.set_logit_stats(logit_mean.to(device), logit_std.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_01, fut_01 in loader:
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(hist_01.shape[0], hist_01.shape[1], -1)
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(fut_01.shape[0], fut_01.shape[1], -1)
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(hist_norm, fut_norm)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

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
            "val_total": val_avg["total"],
            "val_target_dist": val_avg["target_dist"],
            "val_pair_dist": val_avg["pair_dist"],
            "val_target_std": val_avg["target_std"],
            "val_sample_std": val_avg["sample_std"],
            "val_sample_abs": val_avg["sample_abs"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} val={rec['val_total']:.5f} "
            f"target={rec['val_target_dist']:.3f} pair={rec['val_pair_dist']:.3f} "
            f"target_std={rec['val_target_std']:.3f} sample_std={rec['val_sample_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "train_summary.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_total": best_val, "config": asdict(cfg)}, indent=2))


if __name__ == "__main__":
    main()
