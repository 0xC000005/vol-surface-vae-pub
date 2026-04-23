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

from diffusion.block_ar.conditional_marginal_copula_model import (
    ConditionalMarginalCopulaConfig,
    ConditionalMarginalCopulaModel,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_320a_future_scalar_ar_mixture_density_model import (
    compute_level_logit_stats,
    make_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="323a conditional marginal + neural copula factorization"
    )
    parser.add_argument("--context_dim", type=int, default=192)
    parser.add_argument("--history_hidden", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--marginal_hidden", type=int, default=256)
    parser.add_argument("--logit_eps", type=float, default=1e-4)
    parser.add_argument("--logit_std_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-3)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--max_sample_chunk", type=int, default=8)
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/321c_v0_s42/best_model.pt")
    parser.add_argument(
        "--marginal_family",
        type=str,
        default="gaussian",
        choices=["gaussian", "quantile"],
    )

    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=64)
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

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
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
        drop_last=False,
        num_workers=0,
    )

    cfg = ConditionalMarginalCopulaConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        context_dim=args.context_dim,
        history_hidden=args.history_hidden,
        encoder_dropout=args.encoder_dropout,
        marginal_hidden=args.marginal_hidden,
        logit_eps=args.logit_eps,
        logit_std_floor=args.logit_std_floor,
        scale_floor=args.scale_floor,
        sample_temperature=args.sample_temperature,
        max_sample_chunk=args.max_sample_chunk,
        base_checkpoint=args.base_checkpoint,
        marginal_family=args.marginal_family,
    )
    model = ConditionalMarginalCopulaModel(cfg).to(device)
    logit_mean, logit_std = compute_level_logit_stats(
        train_hist,
        train_future,
        logit_eps=args.logit_eps,
        std_floor=args.logit_std_floor,
    )
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
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                hist_01.shape[0], hist_01.shape[1], -1
            )
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                fut_01.shape[0], fut_01.shape[1], -1
            )
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
    print(f"Base copula checkpoint: {args.base_checkpoint}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_nll": train_avg["nll"],
            "val_nll": val_avg["nll"],
            "val_target_std": val_avg["target_std"],
            "val_scale_mean": val_avg["scale_mean"],
            "val_mean_abs_err": val_avg["mean_abs_err"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_nll']:.5f} val={rec['val_nll']:.5f} "
            f"target_std={rec['val_target_std']:.3f} scale={rec['val_scale_mean']:.3f} "
            f"mean_abs={rec['val_mean_abs_err']:.3f} lr={rec['lr']:.2e} "
            f"time={rec['sec']:.1f}s"
        )
        if val_avg["nll"] < best_val:
            best_val = val_avg["nll"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {"best_epoch": best_epoch, "best_val_nll": best_val, "config": asdict(cfg)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
