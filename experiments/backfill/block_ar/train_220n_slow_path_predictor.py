#!/usr/bin/env python
"""
220n: train a slow-path predictor for hybrid multi-day rollout.

The model predicts the full 30-day slow path in factor/logit space from the
last 30 observed days. A frozen 212ai one-day kernel will later generate the
fast residual around this predicted slow backbone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import write_markdown_summary


def logit_clip(x: np.ndarray, eps: float) -> np.ndarray:
    x_clip = np.clip(x, eps, 1.0 - eps)
    return np.log(x_clip / (1.0 - x_clip))


def ema_surfaces(surfaces: np.ndarray, alpha: float) -> np.ndarray:
    slow = np.empty_like(surfaces)
    slow[0] = surfaces[0]
    for t in range(1, surfaces.shape[0]):
        slow[t] = (1.0 - alpha) * slow[t - 1] + alpha * surfaces[t]
    return slow


class FactorWindowDataset(Dataset):
    def __init__(self, hist: torch.Tensor, fut: torch.Tensor):
        self.hist = hist
        self.fut = fut

    def __len__(self) -> int:
        return int(self.hist.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hist[idx], self.fut[idx]


class SlowPathSeq2Seq(nn.Module):
    def __init__(self, factor_dim: int, hidden_dim: int):
        super().__init__()
        self.factor_dim = factor_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(input_size=factor_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.GRUCell(input_size=factor_dim, hidden_size=hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, factor_dim),
        )

    def forward(
        self,
        hist: torch.Tensor,
        future: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        batch, _, factor_dim = hist.shape
        if factor_dim != self.factor_dim:
            raise ValueError(f"Expected factor_dim={self.factor_dim}, got {factor_dim}")
        _, hidden = self.encoder(hist)
        hidden_t = hidden[-1]
        prev = hist[:, -1]
        steps = future.shape[1] if future is not None else 30
        outs = []
        for step in range(steps):
            hidden_t = self.decoder(prev, hidden_t)
            pred = self.out(hidden_t)
            outs.append(pred)
            if future is not None and teacher_forcing_ratio > 0.0:
                use_teacher = torch.rand(batch, device=hist.device) < teacher_forcing_ratio
                prev = torch.where(use_teacher[:, None], future[:, step], pred)
            else:
                prev = pred
        return torch.stack(outs, dim=1)


def build_factor_windows(
    surfaces: np.ndarray,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    slow_alpha: float,
    factor_dim: int,
    eps: float,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    slow = ema_surfaces(surfaces, slow_alpha)
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    fit_end_idx = max_train_idx - val_size + history_len
    y_slow = logit_clip(slow[:fit_end_idx].reshape(fit_end_idx, -1), eps)
    mu = y_slow.mean(axis=0, keepdims=True)
    centered = y_slow - mu
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:factor_dim].copy()

    full_y = logit_clip(slow.reshape(slow.shape[0], -1), eps)
    factors = (full_y - mu) @ basis.T

    def make_windows(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        hist = []
        fut = []
        for idx in indices:
            hist.append(factors[idx : idx + history_len])
            fut.append(factors[idx + history_len : idx + history_len + future_len])
        return (
            torch.from_numpy(np.stack(hist).astype(np.float32)),
            torch.from_numpy(np.stack(fut).astype(np.float32)),
        )

    train_hist, train_fut = make_windows(train_indices)
    val_hist, val_fut = make_windows(val_indices)
    payload = {
        "train_hist": train_hist,
        "train_fut": train_fut,
        "val_hist": val_hist,
        "val_fut": val_fut,
    }
    params = {
        "slow_alpha": np.array(slow_alpha, dtype=np.float64),
        "eps": np.array(eps, dtype=np.float64),
        "mu": mu.astype(np.float64),
        "basis": basis.astype(np.float64),
    }
    return payload, params


@torch.no_grad()
def evaluate_factor_mse(
    model: SlowPathSeq2Seq,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for hist, fut in loader:
        hist = hist.to(device)
        fut = fut.to(device)
        pred = model(hist, future=None, teacher_forcing_ratio=0.0)
        losses.append(torch.mean((pred - fut) ** 2).item())
    return float(np.mean(losses)) if losses else math.inf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 220n slow path predictor")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--slow_alpha", type=float, default=0.08)
    parser.add_argument("--factor_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher_forcing_start", type=float, default=0.8)
    parser.add_argument("--teacher_forcing_end", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    payload, params = build_factor_windows(
        surfaces=surfaces,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        slow_alpha=args.slow_alpha,
        factor_dim=args.factor_dim,
        eps=args.eps,
    )

    train_loader = DataLoader(
        FactorWindowDataset(payload["train_hist"], payload["train_fut"]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        FactorWindowDataset(payload["val_hist"], payload["val_fut"]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SlowPathSeq2Seq(args.factor_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = math.inf
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tf_ratio = args.teacher_forcing_start + (args.teacher_forcing_end - args.teacher_forcing_start) * ((epoch - 1) / max(args.epochs - 1, 1))
        train_losses = []
        for hist, fut in train_loader:
            hist = hist.to(device)
            fut = fut.to(device)
            pred = model(hist, future=fut, teacher_forcing_ratio=tf_ratio)
            loss = torch.mean((pred - fut) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_mse = evaluate_factor_mse(model, val_loader, device)
        train_mse = float(np.mean(train_losses)) if train_losses else math.inf
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "teacher_forcing_ratio": tf_ratio})
        print(json.dumps(history[-1]))
        if val_mse < best_val:
            best_val = val_mse
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_mse": val_mse,
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        raise RuntimeError("No checkpoint captured")

    ckpt = {
        "config": vars(args),
        "model_state": best_state["model"],
        "epoch": int(best_state["epoch"]),
        "val_mse": float(best_state["val_mse"]),
        "slow_params": {k: v for k, v in params.items()},
    }
    torch.save(ckpt, out_dir / "best_model.pt")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    lines = [
        f"- output dir: `{out_dir}`",
        f"- factor dim: `{args.factor_dim}`",
        f"- hidden dim: `{args.hidden_dim}`",
        f"- slow alpha: `{args.slow_alpha}`",
        f"- best epoch: `{best_state['epoch']}`",
        f"- best val factor MSE: `{best_state['val_mse']:.6f}`",
    ]
    write_markdown_summary(
        out_dir / "training_summary.md",
        "220n Slow Path Predictor Training",
        lines,
    )


if __name__ == "__main__":
    main()
