#!/usr/bin/env python
"""421a: joint empirical-normal-score transition-path flow."""

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

from diffusion.block_ar.empirical_normal_score_transition_path_flow_matching import (  # noqa: E402
    EmpiricalNormalScoreTransitionPathFMConfig,
    EmpiricalNormalScoreTransitionPathFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import build_multistep_windows  # noqa: E402
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_340a_empirical_normal_score_causal_memory_transition_flow import (  # noqa: E402
    compute_shared_level_quantiles,
)


def build_recent_block(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    adaptation_windows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    val_start = max_train_idx - val_size
    n = min(int(adaptation_windows), int(val_start))
    indices = np.arange(val_start - n, val_start)
    history_01, future_01 = build_multistep_windows(indices, surf_tensor, history_len, future_len)
    future_01 = future_01.view(history_01.shape[0], future_len, 5, 5)
    return history_01, future_01, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--global_mixer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow_time_dim", type=int, default=32)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--flow_steps", type=int, default=32)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--max_sample_chunk", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--holdout_frac", type=float, default=0.2)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    hist_01, fut_01, indices = build_recent_block(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        adaptation_windows=args.adaptation_windows,
        device=device,
    )
    n_total = hist_01.shape[0]
    n_val = max(1, int(round(n_total * float(args.holdout_frac))))
    n_train = n_total - n_val
    train_hist, val_hist = hist_01[:n_train], hist_01[n_train:]
    train_fut, val_fut = fut_01[:n_train], fut_01[n_train:]
    _, h, w = hist_01.shape[-3:]
    d = h * w

    cfg = EmpiricalNormalScoreTransitionPathFMConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_ff=args.token_ff,
        model_dropout=args.model_dropout,
        global_mixer=args.global_mixer,
        flow_time_dim=args.flow_time_dim,
        n_quantiles=args.n_quantiles,
        cdf_eps=args.cdf_eps,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
        max_sample_chunk=args.max_sample_chunk,
    )
    model = EmpiricalNormalScoreTransitionPathFlowMatching(cfg).to(device)
    level_q, quantile_levels = compute_shared_level_quantiles(
        hist_01,
        fut_01,
        n_quantiles=args.n_quantiles,
    )
    model.set_empirical_quantiles(level_q, quantile_levels)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_fut),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_fut),
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
        for hist_batch, fut_batch in loader:
            hist_norm = normalize_iv(hist_batch.to(device, non_blocking=True)).view(
                hist_batch.shape[0], hist_batch.shape[1], -1
            )
            fut_norm = normalize_iv(fut_batch.to(device, non_blocking=True)).view(
                fut_batch.shape[0], fut_batch.shape[1], -1
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

    print(f"Recent windows: {n_total} index range: {indices[0]}..{indices[-1]}")
    print(f"Train/holdout: {n_train}/{n_val}")
    print(f"Grid: H={h} W={w} D={d}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

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
            "train_total": train_avg["total"],
            "val_total": val_avg["total"],
            "val_future_score_std": val_avg["future_score_std"],
            "val_transition_std": val_avg["transition_std"],
            "val_transition_abs": val_avg["transition_abs"],
            "val_path_t_std": val_avg["path_t_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} score_std={rec['val_future_score_std']:.3f} "
            f"trans_std={rec['val_transition_std']:.3f} "
            f"trans_abs={rec['val_transition_abs']:.3f} "
            f"path_std={rec['val_path_t_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "config": asdict(cfg),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
