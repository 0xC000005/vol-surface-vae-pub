#!/usr/bin/env python
"""
224c: Noise-augmented history training — ablation for Self-Forcing compass.

Lower bound experiment: add Gaussian noise to last N history days before GRU
encoding, train with standard 212ai loss on both clean and noisy batches.

If noise injection improves multi-day metrics → perturbation robustness matters.
If no change → directed gradient (Self-Forcing H1) is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    evaluate_h1,
)
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import (
    ConditionalFlowLocalScaleAsinhNLLModel,
)


def add_history_noise(
    history_01: torch.Tensor,
    model: ConditionalFlowLocalScaleAsinhNLLModel,
    noise_mult: float = 0.5,
    n_noisy_days: int = 1,
) -> torch.Tensor:
    """Add Gaussian noise scaled by local_scale to last n days of history."""
    B = history_01.shape[0]
    with torch.no_grad():
        _, local_scale = model.encode_with_scale(history_01)
        local_scale = local_scale.clamp_min(model.scale_floor)  # (B, 25)
    noise = (
        torch.randn(B, n_noisy_days, 25, device=history_01.device)
        * local_scale.unsqueeze(1)
        * noise_mult
    )
    noisy_hist = history_01.clone()
    tail = noisy_hist[:, -n_noisy_days:].reshape(B, n_noisy_days, 25)
    noisy_hist[:, -n_noisy_days:] = (tail + noise).clamp(0.0, 1.0).reshape(
        B, n_noisy_days, 5, 5
    )
    return noisy_hist


def main() -> None:
    parser = argparse.ArgumentParser(description="224c noise-augmented history ablation")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--nll_weight", type=float, default=0.05)
    parser.add_argument("--noise_mult", type=float, default=0.5)
    parser.add_argument("--n_noisy_days", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_payload = torch.load(
        args.init_checkpoint, map_location=device, weights_only=False
    )
    cfg = init_payload["config"]
    model = ConditionalFlowLocalScaleAsinhNLLModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        flow_hidden=cfg["flow_hidden"],
        n_coupling_layers=cfg["n_coupling_layers"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
    ).to(device)
    model.load_state_dict(init_payload["model_state_dict"])

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 1
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
    val_indices = val_indices[: args.max_val_windows]

    train_hist, train_target = build_one_step_windows(
        train_indices, surf_tensor, args.history_len
    )
    val_hist, val_target = build_one_step_windows(
        val_indices, surf_tensor, args.history_len
    )

    train_delta_np = np.diff(
        surfaces[: args.test_start].reshape(-1, 25), axis=0
    )
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(
        TensorDataset(train_hist, train_target),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_target),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print("224c noise-augmented history ablation")
    print(f"  init={args.init_checkpoint}")
    print(f"  noise_mult={args.noise_mult} n_noisy_days={args.n_noisy_days}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "clean_loss": 0.0, "noisy_loss": 0.0}
        count = 0

        for history_01, target_01 in train_loader:
            # Clean branch
            clean_loss, _ = model.training_loss(
                history_01,
                target_01,
                n_samples=args.train_samples,
                nll_weight=args.nll_weight,
            )

            # Noisy branch: same target, perturbed history
            noisy_hist = add_history_noise(
                history_01, model, args.noise_mult, args.n_noisy_days
            )
            noisy_loss, _ = model.training_loss(
                noisy_hist,
                target_01,
                n_samples=args.train_samples,
                nll_weight=args.nll_weight,
            )

            loss = 0.5 * clean_loss + 0.5 * noisy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["clean_loss"] += float(clean_loss.item()) * batch
            running["noisy_loss"] += float(noisy_loss.item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}

        model.eval()
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
        )

        score = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
        )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_conditional_flow_223_generated_history",
                "n_cells": cfg["n_cells"],
                "history_feat_dim": cfg["history_feat_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "gru_layers": cfg["gru_layers"],
                "gru_dropout": cfg["gru_dropout"],
                "flow_hidden": cfg["flow_hidden"],
                "n_coupling_layers": cfg["n_coupling_layers"],
                "history_len": cfg["history_len"],
                "ewma_alpha": cfg["ewma_alpha"],
                "scale_floor": cfg["scale_floor"],
                "include_scale_feature": cfg["include_scale_feature"],
                "nll_weight": args.nll_weight,
                "noise_mult": args.noise_mult,
                "n_noisy_days": args.n_noisy_days,
                "init_checkpoint": args.init_checkpoint,
            },
            "metrics": history[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"loss={train_metrics['train_loss']:.4f} "
            f"clean={train_metrics['train_clean_loss']:.4f} "
            f"noisy={train_metrics['train_noisy_loss']:.4f} "
            f"cov90={val_metrics['val_coverage_90']:.3f} "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f} "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f} "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")
    print(f"Done. Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
