#!/usr/bin/env python
"""
220o0: teacher-forced slow-conditioned fast residual generator.

Goal:
  Test the slow-fast coupling hypothesis without post-hoc replacement.

Mechanism:
  - use an oracle/teacher-forced slow surface process (EMA slow state)
  - train a fast residual conditional flow directly around that slow path
  - condition the fast generator on both the slow history and the next slow
    frame, so the coupling is learned during training rather than bolted on
    after the fact

This is still an oracle diagnostic, not a deployable model.
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import make_serializable
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import energy_score
from experiments.backfill.block_ar.train_212ae_h1_conditional_flow_local_scale_asinh import AffineCoupling
from experiments.backfill.block_ar._rollout_220_utils import write_markdown_summary


def compute_slow_surface_series(surfaces: np.ndarray, alpha: float) -> np.ndarray:
    slow = np.empty_like(surfaces)
    slow[0] = surfaces[0]
    for t in range(1, surfaces.shape[0]):
        slow[t] = (1.0 - alpha) * slow[t - 1] + alpha * surfaces[t]
    return slow


def build_residual_scale_history_features(
    history_01: torch.Tensor,
    slow_history_01: torch.Tensor,
    ewma_alpha: float,
    scale_floor: float,
    include_scale_feature: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, hist_len, _, _ = history_01.shape
    residual = (history_01 - slow_history_01).reshape(batch, hist_len, -1)
    slow_flat = slow_history_01.reshape(batch, hist_len, -1)
    residual_delta_obs = residual[:, 1:] - residual[:, :-1]
    abs_obs = residual_delta_obs.abs().clamp_min(scale_floor)
    scale_obs = torch.empty_like(abs_obs)
    scale_obs[:, 0] = abs_obs[:, 0]
    for t in range(1, residual_delta_obs.shape[1]):
        scale_obs[:, t] = ewma_alpha * abs_obs[:, t] + (1.0 - ewma_alpha) * scale_obs[:, t - 1]

    scale_seq = torch.cat([scale_obs[:, :1], scale_obs], dim=1).clamp_min(scale_floor)
    residual_delta = torch.zeros_like(residual)
    residual_delta[:, 1:] = residual_delta_obs
    residual_std = residual_delta / scale_seq

    slow_norm = slow_flat * 2.0 - 1.0
    slow_delta = torch.zeros_like(slow_flat)
    slow_delta[:, 1:] = slow_flat[:, 1:] - slow_flat[:, :-1]

    feat_parts = [residual, residual_std, slow_norm, slow_delta]
    if include_scale_feature:
        feat_parts.append(torch.log(scale_seq))
    feat = torch.cat(feat_parts, dim=-1)
    return feat, scale_seq[:, -1]


class SlowConditionedOneDayDataset(Dataset):
    def __init__(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        target_01: torch.Tensor,
        slow_target_01: torch.Tensor,
    ) -> None:
        self.history_01 = history_01
        self.slow_history_01 = slow_history_01
        self.target_01 = target_01
        self.slow_target_01 = slow_target_01

    def __len__(self) -> int:
        return int(self.history_01.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.history_01[idx],
            self.slow_history_01[idx],
            self.target_01[idx],
            self.slow_target_01[idx],
        )


class TeacherForcedSlowConditionedFastFlow(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 125,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        flow_hidden: int = 256,
        n_coupling_layers: int = 6,
        ewma_alpha: float = 0.20,
        scale_floor: float = 1e-4,
        include_scale_feature: bool = True,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = n_cells
        self.hidden_dim = hidden_dim
        self.ewma_alpha = float(ewma_alpha)
        self.scale_floor = float(scale_floor)
        self.include_scale_feature = bool(include_scale_feature)

        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        cond_dim = hidden_dim + 3 * n_cells
        masks = []
        base = torch.tensor([(i % 2) for i in range(n_cells)], dtype=torch.float32)
        for i in range(n_coupling_layers):
            masks.append(base if i % 2 == 0 else 1.0 - base)
        self.layers = nn.ModuleList(
            [AffineCoupling(n_cells, cond_dim, flow_hidden, mask) for mask in masks]
        )

    def _expand_condition(self, cond_state: torch.Tensor, n_samples: int) -> torch.Tensor:
        return cond_state.unsqueeze(1).expand(cond_state.shape[0], n_samples, cond_state.shape[-1])

    def _flow_forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        logdet = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            x, layer_logdet = layer.forward_with_logdet(x, cond)
            logdet = logdet + layer_logdet
        return x, logdet

    def _flow_inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, layer_logdet = layer.inverse_with_logdet(z, cond)
            logdet = logdet + layer_logdet
        return z, logdet

    def encode_with_scale(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_next_01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, local_scale = build_residual_scale_history_features(
            history_01=history_01,
            slow_history_01=slow_history_01,
            ewma_alpha=self.ewma_alpha,
            scale_floor=self.scale_floor,
            include_scale_feature=self.include_scale_feature,
        )
        _out, h_n = self.gru(feat)
        state = h_n[-1]
        slow_prev = slow_history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        slow_next = slow_next_01.reshape(history_01.shape[0], self.n_cells)
        slow_delta = slow_next - slow_prev
        cond_state = torch.cat([state, slow_prev, slow_next, slow_delta], dim=-1)
        prev_residual = (history_01[:, -1] - slow_history_01[:, -1]).reshape(history_01.shape[0], self.n_cells)
        return cond_state, local_scale, prev_residual

    def sample_transformed_innovation(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_next_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_state, local_scale, prev_residual = self.encode_with_scale(history_01, slow_history_01, slow_next_01)
        batch = history_01.shape[0]
        if noise is None:
            z = torch.randn(batch, n_samples, self.n_cells, device=history_01.device, dtype=history_01.dtype)
        else:
            z = noise.to(device=history_01.device, dtype=history_01.dtype)
        cond = self._expand_condition(cond_state, n_samples)
        v, _ = self._flow_forward(z, cond)
        return v, local_scale, prev_residual

    def log_prob_transformed_innovation(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_next_01: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        cond_state, _local_scale, _prev_residual = self.encode_with_scale(history_01, slow_history_01, slow_next_01)
        cond = self._expand_condition(cond_state, 1)
        y = target_v.unsqueeze(1)
        z, logdet_inv = self._flow_inverse(y, cond)
        log_base = -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)
        return (log_base + logdet_inv).squeeze(1)

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_next_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v, local_scale, prev_residual = self.sample_transformed_innovation(
            history_01=history_01,
            slow_history_01=slow_history_01,
            slow_next_01=slow_next_01,
            n_samples=n_samples,
            noise=noise,
        )
        residual_delta = torch.sinh(v) * local_scale.unsqueeze(1)
        next_residual = prev_residual.unsqueeze(1) + residual_delta
        slow_next = slow_next_01.reshape(history_01.shape[0], self.n_cells).unsqueeze(1)
        next_iv = (slow_next + next_residual).clamp(0.0, 1.0)
        return next_iv.view(history_01.shape[0], n_samples, 5, 5)

    def training_loss(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        target_01: torch.Tensor,
        slow_next_01: torch.Tensor,
        n_samples: int,
        nll_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_residual = target_01 - slow_next_01.reshape(history_01.shape[0], self.n_cells)
        _cond_state, local_scale, prev_residual = self.encode_with_scale(history_01, slow_history_01, slow_next_01)
        target_residual_delta = target_residual - prev_residual
        target_v = torch.asinh(target_residual_delta / local_scale.clamp_min(self.scale_floor))
        v_samples, _local_scale2, _prev_residual2 = self.sample_transformed_innovation(
            history_01, slow_history_01, slow_next_01, n_samples=n_samples
        )
        es = energy_score(v_samples, target_v)
        log_prob = self.log_prob_transformed_innovation(history_01, slow_history_01, slow_next_01, target_v)
        nll = -log_prob.mean() / self.n_cells
        loss = es + nll_weight * nll
        residual_delta = torch.sinh(v_samples) * local_scale.unsqueeze(1)
        next_residual = prev_residual.unsqueeze(1) + residual_delta
        next_iv = slow_next_01.reshape(history_01.shape[0], 1, self.n_cells) + next_residual
        metrics = {
            "energy": es.detach(),
            "nll": nll.detach(),
            "sample_residual_delta_std": residual_delta.std(dim=1).mean().detach(),
            "sample_next_iv_std": next_iv.std(dim=1).mean().detach(),
        }
        return loss, metrics


def build_teacher_forced_windows(
    indices: np.ndarray,
    surface_tensor: torch.Tensor,
    slow_tensor: torch.Tensor,
    history_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = []
    slow_history = []
    target = []
    slow_target = []
    for idx in indices:
        history.append(surface_tensor[idx : idx + history_len])
        slow_history.append(slow_tensor[idx : idx + history_len])
        target.append(surface_tensor[idx + history_len].reshape(-1))
        slow_target.append(slow_tensor[idx + history_len].reshape(-1))
    return (
        torch.stack(history, dim=0),
        torch.stack(slow_history, dim=0),
        torch.stack(target, dim=0),
        torch.stack(slow_target, dim=0),
    )


@torch.no_grad()
def evaluate_h1_teacher_forced(
    model: TeacherForcedSlowConditionedFastFlow,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    running = {
        "energy": 0.0,
        "coverage_90": 0.0,
        "realized_q95_coverage_90": 0.0,
        "realized_q99_coverage_90": 0.0,
        "h1_quiet_ratio": 0.0,
        "h1_shoulder_ratio": 0.0,
        "h1_extreme_ratio": 0.0,
        "h1_kurtosis_ratio": 0.0,
    }
    count = 0

    train_ref_stats = getattr(model, "_train_ref_stats")
    for history_01, slow_history_01, target_01, slow_target_01 in loader:
        samples = model.sample_next_iv(
            history_01,
            slow_history_01,
            slow_target_01,
            n_samples=eval_samples,
        )
        samples_flat = samples.view(samples.shape[0], samples.shape[1], 25)
        target_flat = target_01.view(target_01.shape[0], 25)
        prev_flat = history_01[:, -1].view(history_01.shape[0], 25)
        target_delta = target_flat - prev_flat
        sample_delta = samples_flat - prev_flat.unsqueeze(1)
        lo = torch.quantile(samples_flat, 0.05, dim=1)
        hi = torch.quantile(samples_flat, 0.95, dim=1)
        coverage = ((target_flat >= lo) & (target_flat <= hi)).float().mean()

        realized_abs = target_delta.abs()
        q95_mask = realized_abs >= q95_threshold
        q99_mask = realized_abs >= q99_threshold
        if q95_mask.any():
            q95_cov = ((target_flat[q95_mask] >= lo[q95_mask]) & (target_flat[q95_mask] <= hi[q95_mask])).float().mean()
        else:
            q95_cov = torch.tensor(1.0, device=target_01.device)
        if q99_mask.any():
            q99_cov = ((target_flat[q99_mask] >= lo[q99_mask]) & (target_flat[q99_mask] <= hi[q99_mask])).float().mean()
        else:
            q99_cov = torch.tensor(1.0, device=target_01.device)

        batch_abs = sample_delta.abs().reshape(-1)
        quiet = (batch_abs <= 0.005).float().mean().item() / max(train_ref_stats["quiet"], 1e-8)
        shoulder = (
            ((batch_abs > 0.01) & (batch_abs <= 0.05)).float().mean().item()
            / max(train_ref_stats["shoulder"], 1e-8)
        )
        extreme = (batch_abs > q99_threshold).float().mean().item() / max(train_ref_stats["extreme"], 1e-8)
        batch_kurt = _raw_kurtosis(batch_abs.cpu().numpy())
        kurt_ratio = batch_kurt / max(train_ref_stats["kurtosis"], 1e-8)
        es = energy_score(sample_delta, target_delta)

        batch = history_01.shape[0]
        running["energy"] += float(es.item()) * batch
        running["coverage_90"] += float(coverage.item()) * batch
        running["realized_q95_coverage_90"] += float(q95_cov.item()) * batch
        running["realized_q99_coverage_90"] += float(q99_cov.item()) * batch
        running["h1_quiet_ratio"] += quiet * batch
        running["h1_shoulder_ratio"] += shoulder * batch
        running["h1_extreme_ratio"] += extreme * batch
        running["h1_kurtosis_ratio"] += kurt_ratio * batch
        count += batch

    return {f"val_{k}": v / max(count, 1) for k, v in running.items()}


def _raw_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 3.0
    mu = x.mean()
    centered = x - mu
    var = np.mean(centered**2)
    if var <= 1e-12:
        return 3.0
    return float(np.mean(centered**4) / max(var**2, 1e-12))


def load_model(checkpoint_path: str, device: torch.device) -> tuple[TeacherForcedSlowConditionedFastFlow, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "220o0_teacher_forced_slow_conditioned_fast":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = TeacherForcedSlowConditionedFastFlow(
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
    )
    model.load_state_dict(payload["model_state_dict"])
    model._train_ref_stats = payload["train_ref_stats"]
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="220o0 teacher-forced slow-conditioned fast residual generator")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--flow_hidden", type=int, default=256)
    parser.add_argument("--n_coupling_layers", type=int, default=6)
    parser.add_argument("--train_samples", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--slow_alpha", type=float, default=0.08)
    parser.add_argument("--include_scale_feature", action="store_true", default=True)
    parser.add_argument("--no_scale_feature", action="store_false", dest="include_scale_feature")
    parser.add_argument("--lambda_nll_max", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    slow_surfaces = compute_slow_surface_series(surfaces, alpha=args.slow_alpha)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    slow_tensor = torch.from_numpy(slow_surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)[: args.max_train_windows]
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_val_windows]

    train_hist, train_slow_hist, train_target, train_slow_target = build_teacher_forced_windows(
        train_indices, surf_tensor, slow_tensor, args.history_len
    )
    val_hist, val_slow_hist, val_target, val_slow_target = build_teacher_forced_windows(
        val_indices, surf_tensor, slow_tensor, args.history_len
    )

    train_delta_np = np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))
    train_abs = np.abs(train_delta_np.reshape(-1))
    train_ref_stats = {
        "quiet": float(np.mean(train_abs <= 0.005)),
        "shoulder": float(np.mean((train_abs > 0.01) & (train_abs <= 0.05))),
        "extreme": float(np.mean(train_abs > q99_threshold)),
        "kurtosis": _raw_kurtosis(train_abs),
    }

    train_loader = DataLoader(
        SlowConditionedOneDayDataset(train_hist, train_slow_hist, train_target, train_slow_target),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SlowConditionedOneDayDataset(val_hist, val_slow_hist, val_target, val_slow_target),
        batch_size=args.batch_size,
        shuffle=False,
    )

    history_feat_dim = 25 * 4 + (25 if args.include_scale_feature else 0)
    model = TeacherForcedSlowConditionedFastFlow(
        n_cells=25,
        history_feat_dim=history_feat_dim,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        flow_hidden=args.flow_hidden,
        n_coupling_layers=args.n_coupling_layers,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=args.include_scale_feature,
    ).to(device)
    model._train_ref_stats = train_ref_stats
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("220o0 teacher-forced slow-conditioned fast residual generator")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  train_samples={args.train_samples} lambda_nll_max={args.lambda_nll_max} slow_alpha={args.slow_alpha}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_payload: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lambda_nll = args.lambda_nll_max * (epoch / args.epochs)
        model.train()
        running = {"loss": 0.0, "energy": 0.0, "nll": 0.0, "sample_residual_delta_std": 0.0, "sample_next_iv_std": 0.0}
        count = 0

        for history_01, slow_history_01, target_01, slow_target_01 in train_loader:
            loss, metrics = model.training_loss(
                history_01=history_01,
                slow_history_01=slow_history_01,
                target_01=target_01,
                slow_next_01=slow_target_01,
                n_samples=args.train_samples,
                nll_weight=lambda_nll,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["energy"] += float(metrics["energy"].item()) * batch
            running["nll"] += float(metrics["nll"].item()) * batch
            running["sample_residual_delta_std"] += float(metrics["sample_residual_delta_std"].item()) * batch
            running["sample_next_iv_std"] += float(metrics["sample_next_iv_std"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        model._train_ref_stats = train_ref_stats
        val_metrics = evaluate_h1_teacher_forced(
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
            "lambda_nll": lambda_nll,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "220o0_teacher_forced_slow_conditioned_fast",
                "n_cells": 25,
                "history_feat_dim": history_feat_dim,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "flow_hidden": args.flow_hidden,
                "n_coupling_layers": args.n_coupling_layers,
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
                "ewma_alpha": args.ewma_alpha,
                "scale_floor": args.scale_floor,
                "include_scale_feature": args.include_scale_feature,
                "lambda_nll_max": args.lambda_nll_max,
                "slow_alpha": args.slow_alpha,
            },
            "metrics": history[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
            "train_ref_stats": train_ref_stats,
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            best_payload = payload
            torch.save(payload, out_dir / "best_model.pt")

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"lambda={lambda_nll:.4f} "
            f"loss={train_metrics['train_loss']:.4f} "
            f"trainNLL={train_metrics['train_nll']:.4f} "
            f"cov90={val_metrics['val_coverage_90']:.3f} "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f} "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f} "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f} "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f} "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    if best_payload is None:
        raise RuntimeError("No best checkpoint saved")
    torch.save(best_payload, out_dir / "final_model.pt")
    write_markdown_summary(
        out_dir / "training_summary.md",
        "220o0 Teacher-Forced Slow-Conditioned Fast Training",
        [
            f"- output dir: `{out_dir}`",
            f"- slow alpha: `{args.slow_alpha}`",
            f"- best epoch: `{best_payload['epoch']}`",
            f"- best selection score: `{best_score:.3f}`",
            f"- best val coverage90: `{best_payload['metrics']['val_coverage_90']:.3f}`",
            f"- best val q99 coverage90: `{best_payload['metrics']['val_realized_q99_coverage_90']:.3f}`",
        ],
    )


if __name__ == "__main__":
    main()
