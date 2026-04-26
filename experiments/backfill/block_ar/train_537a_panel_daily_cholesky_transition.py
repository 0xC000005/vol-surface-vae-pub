#!/usr/bin/env python
"""537a: train an autoregressive daily Cholesky transition law over IV+factor panels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.panel_daily_cholesky_transition_model import (  # noqa: E402
    PanelDailyCholeskyTransitionConfig,
    PanelDailyCholeskyTransitionModel,
    save_checkpoint,
)
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    build_panel_block,
    load_aligned_iv_factor_panel,
    panel_summary,
)


def fit_value_quantiles(
    panel_block: torch.Tensor,
    n_quantiles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = panel_block.reshape(-1, panel_block.shape[-1]).detach().cpu()
    levels = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / float(n_quantiles)
    quantiles = torch.quantile(flat, levels, dim=0).T.contiguous()
    return quantiles, levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--history_hidden", type=int, default=192)
    parser.add_argument("--context_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=537)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    panel, columns, dates = load_aligned_iv_factor_panel()
    train_indices, val_indices = official_train_val_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    if args.max_train_windows > 0:
        train_indices = train_indices[-int(args.max_train_windows) :]
    train_block = build_panel_block(panel, columns, train_indices, args.history_len, args.future_len, device)
    val_block = build_panel_block(panel, columns, val_indices, args.history_len, args.future_len, device)

    cfg = PanelDailyCholeskyTransitionConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_vars=panel.shape[1],
        n_quantiles=args.n_quantiles,
        history_hidden=args.history_hidden,
        context_hidden=args.context_hidden,
        dropout=args.dropout,
    )
    model = PanelDailyCholeskyTransitionModel(cfg).to(device)
    train_panel_for_quantiles = torch.cat([train_block.history_panel, train_block.future_panel], dim=1)
    quantiles, levels = fit_value_quantiles(train_panel_for_quantiles, args.n_quantiles)
    model.set_empirical_quantiles(quantiles.to(device), levels.to(device))

    train_loader = DataLoader(
        TensorDataset(train_block.history_panel, train_block.future_panel),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_block.history_panel, val_block.future_panel),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist, fut in loader:
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(hist.to(device), fut.to(device))
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

    print(f"Date range: {dates[0]} .. {dates[-1]}")
    print(json.dumps({"train": panel_summary(train_block), "val": panel_summary(val_block)}, indent=2))
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Daily Cholesky dim: {model.n_tril} lower-triangular entries")

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_nll": train_avg["nll"],
            "train_innovation_mae": train_avg["innovation_mae"],
            "train_diag_mean": train_avg["diag_mean"],
            "train_offdiag_abs": train_avg["offdiag_abs"],
            "val_nll": val_avg["nll"],
            "val_innovation_mae": val_avg["innovation_mae"],
            "val_diag_mean": val_avg["diag_mean"],
            "val_offdiag_abs": val_avg["offdiag_abs"],
            "val_innovation_std": val_avg["innovation_std"],
            "val_whitened_std": val_avg["whitened_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train_nll={rec['train_nll']:.5f} "
            f"val_nll={rec['val_nll']:.5f} mae={rec['val_innovation_mae']:.3f} "
            f"diag={rec['val_diag_mean']:.3f} offdiag={rec['val_offdiag_abs']:.3f} "
            f"white={rec['val_whitened_std']:.3f} lr={rec['lr']:.2e} "
            f"time={rec['sec']:.1f}s",
            flush=True,
        )
        if rec["val_nll"] < best_val:
            best_val = rec["val_nll"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, epoch, best_val, columns)

    save_checkpoint(str(out_dir / "final_model.pt"), model, args.epochs, best_val, columns)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "best_epoch": best_epoch,
        "best_val_nll": best_val,
        "train": panel_summary(train_block),
        "val": panel_summary(val_block),
        "n_tril": int(model.n_tril),
        "params": int(sum(p.numel() for p in model.parameters())),
        "config": vars(args),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

