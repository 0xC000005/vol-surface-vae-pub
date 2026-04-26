#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar.synthetic_iv_prior import (
    SyntheticIVPriorConfig,
    generate_synthetic_iv_windows,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


def _flatten_cells(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.reshape(-1, x.shape[-2] * x.shape[-1])
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"Expected 3D or 4D window tensor, got shape {tuple(x.shape)}")


def compute_shared_level_quantiles(
    train_hist_01: torch.Tensor,
    train_future_01: torch.Tensor,
    n_quantiles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    levels = (torch.arange(n_quantiles, device=train_hist_01.device, dtype=torch.float32) + 0.5)
    levels = levels / float(n_quantiles)
    all_levels = torch.cat([_flatten_cells(train_hist_01), _flatten_cells(train_future_01)], dim=0)
    quantiles = torch.quantile(all_levels.float(), levels, dim=0).transpose(0, 1)
    return quantiles.contiguous(), levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="560a TimePFN-style synthetic-prior pretraining for empirical-score AR flow"
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--model_hidden", type=int, default=256)
    parser.add_argument("--model_layers", type=int, default=4)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=3)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_ff", type=int, default=256)
    parser.add_argument(
        "--conditioning_mode",
        choices=["additive", "prefix"],
        default="additive",
    )
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument(
        "--prefix_feature_mode",
        choices=["basic", "scale"],
        default="basic",
    )
    parser.add_argument("--flow_steps", type=int, default=32)
    parser.add_argument("--sample_temperature", type=float, default=1.0)

    parser.add_argument("--n_windows", type=int, default=4096)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--stress_prob", type=float, default=0.35)
    parser.add_argument("--jump_prob", type=float, default=0.06)
    parser.add_argument("--idio_noise", type=float, default=0.006)
    parser.add_argument("--min_iv", type=float, default=0.03)
    parser.add_argument("--max_iv", type=float, default=0.95)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=560)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("val_frac must be in (0,1)")
    if args.n_windows < 4:
        raise ValueError("n_windows must be at least 4")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    synth_cfg = SyntheticIVPriorConfig(
        n_windows=args.n_windows,
        history_len=args.history_len,
        future_len=args.future_len,
        seed=args.seed,
        min_iv=args.min_iv,
        max_iv=args.max_iv,
        stress_prob=args.stress_prob,
        jump_prob=args.jump_prob,
        idio_noise=args.idio_noise,
    )
    synthetic = generate_synthetic_iv_windows(synth_cfg)

    split = max(1, min(args.n_windows - 1, int(round(args.n_windows * (1.0 - args.val_frac)))))
    train_hist = synthetic.history[:split].to(device)
    train_future = synthetic.future[:split].to(device)
    val_hist = synthetic.history[split:].to(device)
    val_future = synthetic.future[split:].to(device)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    cfg = EmpiricalNormalScoreCausalMemoryTransitionFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=25,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        model_hidden=args.model_hidden,
        model_layers=args.model_layers,
        model_dropout=args.model_dropout,
        time_dim=args.time_dim,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_heads=args.token_heads,
        token_ff=args.token_ff,
        memory_dim=args.memory_dim,
        memory_layers=args.memory_layers,
        memory_heads=args.memory_heads,
        memory_ff=args.memory_ff,
        conditioning_mode=args.conditioning_mode,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
        n_quantiles=args.n_quantiles,
        cdf_eps=args.cdf_eps,
        prefix_feature_mode=args.prefix_feature_mode,
    )
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(cfg).to(device)
    quantiles, quantile_levels = compute_shared_level_quantiles(
        train_hist,
        train_future,
        n_quantiles=args.n_quantiles,
    )
    model.set_empirical_quantiles(quantiles, quantile_levels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        context = torch.enable_grad() if train_mode else torch.no_grad()
        with context:
            for hist_01, fut_01 in loader:
                hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                    hist_01.shape[0], args.history_len, 5, 5
                )
                fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                    fut_01.shape[0], args.future_len, 5, 5
                )
                loss, metrics = model.training_loss(hist_norm, fut_norm)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu())
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
                n_batches += 1
        return {key: value / max(1, n_batches) for key, value in sums.items()}

    print(f"Synthetic windows: train={len(train_hist)} val={len(val_hist)}")
    print(f"Stress labels: {int(synthetic.regime_labels.sum())}/{len(synthetic.regime_labels)}")
    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        val_loss = float(val_avg["loss"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_avg["loss"]),
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        records.append(row)
        print(
            f"Epoch {epoch:03d} train={row['train_loss']:.6f} "
            f"val={row['val_loss']:.6f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2))
    summary = {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "elapsed_sec": time.time() - start_time,
        "n_train": int(train_hist.shape[0]),
        "n_val": int(val_hist.shape[0]),
        "stress_windows": int(synthetic.regime_labels.sum()),
        "total_windows": int(synthetic.regime_labels.numel()),
    }
    (out_dir / "synthetic_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
