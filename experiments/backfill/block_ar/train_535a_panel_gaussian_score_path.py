#!/usr/bin/env python
"""535a: train a coherent Gaussian score-path law over aligned IV+factor panels."""

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

from diffusion.block_ar.coherent_panel_score_path_model import (  # noqa: E402
    CoherentPanelScorePathConfig,
    CoherentPanelScorePathModel,
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
    future_panel: torch.Tensor,
    n_quantiles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = future_panel.reshape(-1, future_panel.shape[-1]).detach().cpu()
    levels = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / float(n_quantiles)
    quantiles = torch.quantile(flat, levels, dim=0).T.contiguous()
    return quantiles, levels


@torch.no_grad()
def fit_global_score_cholesky(
    model: CoherentPanelScorePathModel,
    future_panel: torch.Tensor,
    batch_size: int,
    shrinkage: float,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, future_panel.shape[0], batch_size):
        end = min(start + batch_size, future_panel.shape[0])
        scores = model.values_to_scores(future_panel[start:end]).reshape(end - start, model.path_dim)
        chunks.append(scores.detach().cpu())
    x = torch.cat(chunks, dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    corr = (x.T @ x) / max(int(x.shape[0]) - 1, 1)
    corr = 0.5 * (corr + corr.T)
    eye = torch.eye(corr.shape[0], dtype=corr.dtype)
    shrink = float(max(0.0, min(1.0, shrinkage)))
    corr = (1.0 - shrink) * corr + shrink * eye
    for jitter in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        try:
            return torch.linalg.cholesky(corr + jitter * eye)
        except RuntimeError:
            continue
    return torch.linalg.cholesky(corr + 5e-2 * eye)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--history_hidden", type=int, default=192)
    parser.add_argument("--token_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--cov_shrinkage", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=535)
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

    cfg = CoherentPanelScorePathConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_vars=panel.shape[1],
        n_quantiles=args.n_quantiles,
        history_hidden=args.history_hidden,
        token_hidden=args.token_hidden,
        dropout=args.dropout,
    )
    model = CoherentPanelScorePathModel(cfg).to(device)
    quantiles, levels = fit_value_quantiles(train_block.future_panel, args.n_quantiles)
    model.set_empirical_quantiles(quantiles.to(device), levels.to(device))
    chol = fit_global_score_cholesky(
        model=model,
        future_panel=train_block.future_panel,
        batch_size=args.batch_size,
        shrinkage=args.cov_shrinkage,
    )
    model.set_residual_cholesky(chol.to(device))

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
    print(f"Path dim: {model.path_dim}  cholesky logdet: {float(model.residual_cholesky_logdet):.4f}")

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
            "train_mean_abs_err": train_avg["mean_abs_err"],
            "train_scale_mean": train_avg["scale_mean"],
            "val_nll": val_avg["nll"],
            "val_mean_abs_err": val_avg["mean_abs_err"],
            "val_scale_mean": val_avg["scale_mean"],
            "val_target_std": val_avg["target_std"],
            "val_residual_std": val_avg["residual_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train_nll={rec['train_nll']:.5f} "
            f"val_nll={rec['val_nll']:.5f} mae={rec['val_mean_abs_err']:.3f} "
            f"scale={rec['val_scale_mean']:.3f} resid={rec['val_residual_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s",
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
        "path_dim": int(model.path_dim),
        "params": int(sum(p.numel() for p in model.parameters())),
        "cholesky_logdet": float(model.residual_cholesky_logdet.detach().cpu().item()),
        "config": vars(args),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
