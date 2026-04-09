#!/usr/bin/env python
"""
220o1: jointly trained slow-fast support-aware one-step state-space model.

Design:
  - slow state: learned one-step predictor in low-rank factor/logit space
  - fast state: conditional flow on residual dynamics in logit IV space
  - training: joint optimization of slow prediction and fast residual law
  - rollout: recursively predict slow_next, then sample fast residual around it

This is the first learned version of the slow-fast coupling that 220m / 220o0
motivated.
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

from experiments.backfill.block_ar._rollout_220_utils import write_markdown_summary
from experiments.backfill.block_ar.train_169a_transformed_student_t import make_serializable
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import energy_score
from experiments.backfill.block_ar.train_212ae_h1_conditional_flow_local_scale_asinh import AffineCoupling


def compute_slow_surface_series(surfaces: np.ndarray, alpha: float) -> np.ndarray:
    slow = np.empty_like(surfaces)
    slow[0] = surfaces[0]
    for t in range(1, surfaces.shape[0]):
        slow[t] = (1.0 - alpha) * slow[t - 1] + alpha * surfaces[t]
    return slow


def logit_clip_np(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def logit_clip_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.logit(x)


def build_slow_factor_params(
    slow_surfaces: np.ndarray,
    fit_end_idx: int,
    factor_dim: int,
    eps: float,
) -> dict[str, np.ndarray]:
    y_slow = logit_clip_np(slow_surfaces[:fit_end_idx].reshape(fit_end_idx, -1), eps)
    mu = y_slow.mean(axis=0, keepdims=True)
    centered = y_slow - mu
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:factor_dim].copy()
    return {
        "eps": np.array(eps, dtype=np.float64),
        "mu": mu.astype(np.float64),
        "basis": basis.astype(np.float64),
    }


def slow_surfaces_to_factors(slow_surfaces: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    y = logit_clip_np(slow_surfaces.reshape(slow_surfaces.shape[0], -1), float(np.asarray(params["eps"]).item()))
    mu = np.asarray(params["mu"], dtype=np.float64)
    basis = np.asarray(params["basis"], dtype=np.float64)
    return ((y - mu) @ basis.T).astype(np.float32)


def factors_to_slow_iv_torch(factors: torch.Tensor, params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mu = params["mu"]
    basis = params["basis"]
    y = mu.unsqueeze(0) + factors @ basis
    iv = torch.sigmoid(y)
    return iv.view(factors.shape[0], 5, 5), y


def build_support_residual_history_features(
    history_01: torch.Tensor,
    slow_history_01: torch.Tensor,
    eps: float,
    ewma_alpha: float,
    scale_floor: float,
    include_scale_feature: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, hist_len, _, _ = history_01.shape
    history_y = logit_clip_torch(history_01.reshape(batch, hist_len, -1), eps)
    slow_y = logit_clip_torch(slow_history_01.reshape(batch, hist_len, -1), eps)
    residual_y = history_y - slow_y

    residual_delta_obs = residual_y[:, 1:] - residual_y[:, :-1]
    abs_obs = residual_delta_obs.abs().clamp_min(scale_floor)
    scale_obs = torch.empty_like(abs_obs)
    scale_obs[:, 0] = abs_obs[:, 0]
    for t in range(1, residual_delta_obs.shape[1]):
        scale_obs[:, t] = ewma_alpha * abs_obs[:, t] + (1.0 - ewma_alpha) * scale_obs[:, t - 1]
    scale_seq = torch.cat([scale_obs[:, :1], scale_obs], dim=1).clamp_min(scale_floor)

    residual_delta = torch.zeros_like(residual_y)
    residual_delta[:, 1:] = residual_delta_obs
    residual_std = residual_delta / scale_seq
    slow_delta_y = torch.zeros_like(slow_y)
    slow_delta_y[:, 1:] = slow_y[:, 1:] - slow_y[:, :-1]

    feat_parts = [residual_y, residual_std, slow_y, slow_delta_y]
    if include_scale_feature:
        feat_parts.append(torch.log(scale_seq))
    feat = torch.cat(feat_parts, dim=-1)
    return feat, scale_seq[:, -1]


class JointSlowFastDataset(Dataset):
    def __init__(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_hist_factor: torch.Tensor,
        target_01: torch.Tensor,
        slow_target_01: torch.Tensor,
        slow_target_factor: torch.Tensor,
    ) -> None:
        self.history_01 = history_01
        self.slow_history_01 = slow_history_01
        self.slow_hist_factor = slow_hist_factor
        self.target_01 = target_01
        self.slow_target_01 = slow_target_01
        self.slow_target_factor = slow_target_factor

    def __len__(self) -> int:
        return int(self.history_01.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.history_01[idx],
            self.slow_history_01[idx],
            self.slow_hist_factor[idx],
            self.target_01[idx],
            self.slow_target_01[idx],
            self.slow_target_factor[idx],
        )


class SlowOneStepFactorPredictor(nn.Module):
    def __init__(self, factor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.factor_dim = factor_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=factor_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, factor_dim),
        )

    def forward(self, slow_hist_factor: torch.Tensor) -> torch.Tensor:
        _out, h_n = self.gru(slow_hist_factor)
        return self.out(h_n[-1])


class SupportAwareFastResidualFlow(nn.Module):
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
        support_eps: float = 1e-4,
        include_scale_feature: bool = True,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.noise_dim = n_cells
        self.hidden_dim = hidden_dim
        self.ewma_alpha = float(ewma_alpha)
        self.scale_floor = float(scale_floor)
        self.support_eps = float(support_eps)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, local_scale = build_support_residual_history_features(
            history_01=history_01,
            slow_history_01=slow_history_01,
            eps=self.support_eps,
            ewma_alpha=self.ewma_alpha,
            scale_floor=self.scale_floor,
            include_scale_feature=self.include_scale_feature,
        )
        _out, h_n = self.gru(feat)
        state = h_n[-1]
        slow_prev_y = logit_clip_torch(slow_history_01[:, -1].reshape(history_01.shape[0], self.n_cells), self.support_eps)
        slow_next_y = logit_clip_torch(slow_next_01.reshape(history_01.shape[0], self.n_cells), self.support_eps)
        slow_delta_y = slow_next_y - slow_prev_y
        cond_state = torch.cat([state, slow_prev_y, slow_next_y, slow_delta_y], dim=-1)
        prev_y = logit_clip_torch(history_01[:, -1].reshape(history_01.shape[0], self.n_cells), self.support_eps)
        prev_residual_y = prev_y - slow_prev_y
        return cond_state, local_scale, prev_residual_y, slow_next_y

    def sample_transformed_innovation(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_next_01: torch.Tensor,
        n_samples: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_state, local_scale, prev_residual_y, slow_next_y = self.encode_with_scale(
            history_01, slow_history_01, slow_next_01
        )
        batch = history_01.shape[0]
        if noise is None:
            z = torch.randn(batch, n_samples, self.n_cells, device=history_01.device, dtype=history_01.dtype)
        else:
            z = noise.to(device=history_01.device, dtype=history_01.dtype)
        cond = self._expand_condition(cond_state, n_samples)
        v, _ = self._flow_forward(z, cond)
        return v, local_scale, prev_residual_y, slow_next_y

    def log_prob_transformed_innovation(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_next_01: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        cond_state, _local_scale, _prev_residual_y, _slow_next_y = self.encode_with_scale(
            history_01, slow_history_01, slow_next_01
        )
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
        v, local_scale, prev_residual_y, slow_next_y = self.sample_transformed_innovation(
            history_01=history_01,
            slow_history_01=slow_history_01,
            slow_next_01=slow_next_01,
            n_samples=n_samples,
            noise=noise,
        )
        residual_delta_y = torch.sinh(v) * local_scale.unsqueeze(1)
        next_residual_y = prev_residual_y.unsqueeze(1) + residual_delta_y
        next_y = slow_next_y.unsqueeze(1) + next_residual_y
        next_iv = torch.sigmoid(next_y)
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
        target_y = logit_clip_torch(target_01, self.support_eps)
        cond_state, local_scale, prev_residual_y, slow_next_y = self.encode_with_scale(
            history_01, slow_history_01, slow_next_01
        )
        target_residual_y = target_y - slow_next_y
        target_residual_delta_y = target_residual_y - prev_residual_y
        target_v = torch.asinh(target_residual_delta_y / local_scale.clamp_min(self.scale_floor))
        cond = self._expand_condition(cond_state, n_samples)
        z = torch.randn(history_01.shape[0], n_samples, self.n_cells, device=history_01.device, dtype=history_01.dtype)
        v_samples, _ = self._flow_forward(z, cond)
        es = energy_score(v_samples, target_v)
        log_prob = self.log_prob_transformed_innovation(history_01, slow_history_01, slow_next_01, target_v)
        nll = -log_prob.mean() / self.n_cells
        loss = es + nll_weight * nll
        residual_delta_y = torch.sinh(v_samples) * local_scale.unsqueeze(1)
        metrics = {
            "energy": es.detach(),
            "nll": nll.detach(),
            "sample_residual_delta_y_std": residual_delta_y.std(dim=1).mean().detach(),
        }
        return loss, metrics


class JointSlowFastSupportStateSpaceModel(nn.Module):
    def __init__(
        self,
        factor_dim: int,
        slow_hidden_dim: int,
        fast_history_feat_dim: int,
        fast_hidden_dim: int,
        fast_gru_layers: int,
        fast_gru_dropout: float,
        flow_hidden: int,
        n_coupling_layers: int,
        ewma_alpha: float,
        scale_floor: float,
        support_eps: float,
        include_scale_feature: bool,
        factor_params: dict[str, np.ndarray | torch.Tensor],
    ) -> None:
        super().__init__()
        self.slow_predictor = SlowOneStepFactorPredictor(factor_dim=factor_dim, hidden_dim=slow_hidden_dim)
        self.fast_model = SupportAwareFastResidualFlow(
            history_feat_dim=fast_history_feat_dim,
            hidden_dim=fast_hidden_dim,
            gru_layers=fast_gru_layers,
            gru_dropout=fast_gru_dropout,
            flow_hidden=flow_hidden,
            n_coupling_layers=n_coupling_layers,
            ewma_alpha=ewma_alpha,
            scale_floor=scale_floor,
            support_eps=support_eps,
            include_scale_feature=include_scale_feature,
        )
        mu = factor_params["mu"]
        basis = factor_params["basis"]
        if not torch.is_tensor(mu):
            mu = torch.from_numpy(np.asarray(mu, dtype=np.float32))
        if not torch.is_tensor(basis):
            basis = torch.from_numpy(np.asarray(basis, dtype=np.float32))
        self.register_buffer("factor_mu", mu.reshape(-1))
        self.register_buffer("factor_basis", basis)

    def factor_params_torch(self) -> dict[str, torch.Tensor]:
        return {"mu": self.factor_mu, "basis": self.factor_basis}

    def predict_slow_next(self, slow_hist_factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_factor = self.slow_predictor(slow_hist_factor)
        pred_iv, pred_y = factors_to_slow_iv_torch(pred_factor, self.factor_params_torch())
        return pred_factor, pred_iv, pred_y

    def training_loss(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_hist_factor: torch.Tensor,
        target_01: torch.Tensor,
        slow_target_01: torch.Tensor,
        slow_target_factor: torch.Tensor,
        n_samples: int,
        nll_weight: float,
        slow_mix: float,
        slow_loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_factor, pred_slow_iv, _pred_y = self.predict_slow_next(slow_hist_factor)
        slow_cond_iv = (1.0 - slow_mix) * slow_target_01.view_as(pred_slow_iv) + slow_mix * pred_slow_iv
        fast_loss, fast_metrics = self.fast_model.training_loss(
            history_01=history_01,
            slow_history_01=slow_history_01,
            target_01=target_01,
            slow_next_01=slow_cond_iv.reshape(history_01.shape[0], 25),
            n_samples=n_samples,
            nll_weight=nll_weight,
        )
        slow_mse = torch.mean((pred_factor - slow_target_factor) ** 2)
        loss = fast_loss + slow_loss_weight * slow_mse
        metrics = {
            **fast_metrics,
            "slow_mse": slow_mse.detach(),
        }
        return loss, metrics

    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        slow_history_01: torch.Tensor,
        slow_hist_factor: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_factor, pred_slow_iv, _pred_y = self.predict_slow_next(slow_hist_factor)
        next_iv = self.fast_model.sample_next_iv(
            history_01=history_01,
            slow_history_01=slow_history_01,
            slow_next_01=pred_slow_iv.reshape(history_01.shape[0], 25),
            n_samples=n_samples,
        )
        return next_iv, pred_slow_iv, pred_factor


def build_joint_windows(
    indices: np.ndarray,
    surface_tensor: torch.Tensor,
    slow_tensor: torch.Tensor,
    slow_factor_tensor: torch.Tensor,
    history_len: int,
) -> tuple[torch.Tensor, ...]:
    history = []
    slow_history = []
    slow_hist_factor = []
    target = []
    slow_target = []
    slow_target_factor = []
    for idx in indices:
        history.append(surface_tensor[idx : idx + history_len])
        slow_history.append(slow_tensor[idx : idx + history_len])
        slow_hist_factor.append(slow_factor_tensor[idx : idx + history_len])
        target.append(surface_tensor[idx + history_len].reshape(-1))
        slow_target.append(slow_tensor[idx + history_len].reshape(-1))
        slow_target_factor.append(slow_factor_tensor[idx + history_len])
    return (
        torch.stack(history, dim=0),
        torch.stack(slow_history, dim=0),
        torch.stack(slow_hist_factor, dim=0),
        torch.stack(target, dim=0),
        torch.stack(slow_target, dim=0),
        torch.stack(slow_target_factor, dim=0),
    )


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


@torch.no_grad()
def evaluate_h1_joint(
    model: JointSlowFastSupportStateSpaceModel,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    eval_samples: int,
    train_ref_stats: dict[str, float],
) -> dict[str, float]:
    model.eval()
    running = {
        "coverage_90": 0.0,
        "realized_q99_coverage_90": 0.0,
        "h1_quiet_ratio": 0.0,
        "h1_shoulder_ratio": 0.0,
        "h1_kurtosis_ratio": 0.0,
        "slow_mse": 0.0,
    }
    count = 0
    for history_01, slow_history_01, slow_hist_factor, target_01, _slow_target_01, slow_target_factor in loader:
        pred_factor, pred_slow_iv, _ = model.predict_slow_next(slow_hist_factor)
        samples = model.fast_model.sample_next_iv(
            history_01=history_01,
            slow_history_01=slow_history_01,
            slow_next_01=pred_slow_iv.reshape(history_01.shape[0], 25),
            n_samples=eval_samples,
        )
        samples_flat = samples.view(samples.shape[0], samples.shape[1], 25)
        target_flat = target_01.view(target_01.shape[0], 25)
        prev_flat = history_01[:, -1].view(history_01.shape[0], 25)
        sample_delta = samples_flat - prev_flat.unsqueeze(1)
        target_delta = target_flat - prev_flat

        lo = torch.quantile(samples_flat, 0.05, dim=1)
        hi = torch.quantile(samples_flat, 0.95, dim=1)
        coverage = ((target_flat >= lo) & (target_flat <= hi)).float().mean()
        q99_mask = target_delta.abs() >= q99_threshold
        if q99_mask.any():
            q99_cov = ((target_flat[q99_mask] >= lo[q99_mask]) & (target_flat[q99_mask] <= hi[q99_mask])).float().mean()
        else:
            q99_cov = torch.tensor(1.0, device=target_01.device)

        batch_abs = sample_delta.abs().reshape(-1).cpu().numpy()
        quiet = float(np.mean(batch_abs <= 0.005) / max(train_ref_stats["quiet"], 1e-8))
        shoulder = float(np.mean((batch_abs > 0.01) & (batch_abs <= 0.05)) / max(train_ref_stats["shoulder"], 1e-8))
        kurt_ratio = float(_raw_kurtosis(batch_abs) / max(train_ref_stats["kurtosis"], 1e-8))
        slow_mse = torch.mean((pred_factor - slow_target_factor) ** 2)

        batch = history_01.shape[0]
        running["coverage_90"] += float(coverage.item()) * batch
        running["realized_q99_coverage_90"] += float(q99_cov.item()) * batch
        running["h1_quiet_ratio"] += quiet * batch
        running["h1_shoulder_ratio"] += shoulder * batch
        running["h1_kurtosis_ratio"] += kurt_ratio * batch
        running["slow_mse"] += float(slow_mse.item()) * batch
        count += batch

    return {f"val_{k}": v / max(count, 1) for k, v in running.items()}


def load_model(checkpoint_path: str, device: torch.device) -> tuple[JointSlowFastSupportStateSpaceModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "220o1_joint_slow_fast_support_state_space":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    factor_params = {
        "mu": torch.from_numpy(np.asarray(payload["factor_params"]["mu"], dtype=np.float32)),
        "basis": torch.from_numpy(np.asarray(payload["factor_params"]["basis"], dtype=np.float32)),
    }
    model = JointSlowFastSupportStateSpaceModel(
        factor_dim=cfg["factor_dim"],
        slow_hidden_dim=cfg["slow_hidden_dim"],
        fast_history_feat_dim=cfg["fast_history_feat_dim"],
        fast_hidden_dim=cfg["fast_hidden_dim"],
        fast_gru_layers=cfg["fast_gru_layers"],
        fast_gru_dropout=cfg["fast_gru_dropout"],
        flow_hidden=cfg["flow_hidden"],
        n_coupling_layers=cfg["n_coupling_layers"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        support_eps=cfg["support_eps"],
        include_scale_feature=cfg["include_scale_feature"],
        factor_params=factor_params,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="220o1 jointly trained slow-fast support-aware state-space")
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
    parser.add_argument("--factor_dim", type=int, default=4)
    parser.add_argument("--slow_hidden_dim", type=int, default=64)
    parser.add_argument("--fast_hidden_dim", type=int, default=128)
    parser.add_argument("--fast_gru_layers", type=int, default=2)
    parser.add_argument("--fast_gru_dropout", type=float, default=0.1)
    parser.add_argument("--flow_hidden", type=int, default=256)
    parser.add_argument("--n_coupling_layers", type=int, default=6)
    parser.add_argument("--train_samples", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--support_eps", type=float, default=1e-4)
    parser.add_argument("--slow_alpha", type=float, default=0.08)
    parser.add_argument("--lambda_nll_max", type=float, default=0.05)
    parser.add_argument("--slow_loss_weight", type=float, default=10.0)
    parser.add_argument("--include_scale_feature", action="store_true", default=True)
    parser.add_argument("--no_scale_feature", action="store_false", dest="include_scale_feature")
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
    max_train_idx = args.test_start - args.history_len - 30
    fit_end_idx = max_train_idx - args.val_size + args.history_len
    factor_params_np = build_slow_factor_params(
        slow_surfaces=slow_surfaces,
        fit_end_idx=fit_end_idx,
        factor_dim=args.factor_dim,
        eps=args.support_eps,
    )
    slow_factors = slow_surfaces_to_factors(slow_surfaces, factor_params_np)

    surf_tensor = torch.from_numpy(surfaces).to(device)
    slow_tensor = torch.from_numpy(slow_surfaces).to(device)
    slow_factor_tensor = torch.from_numpy(slow_factors).to(device)

    train_indices = np.arange(0, max_train_idx - args.val_size)[: args.max_train_windows]
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_val_windows]
    train_pack = build_joint_windows(train_indices, surf_tensor, slow_tensor, slow_factor_tensor, args.history_len)
    val_pack = build_joint_windows(val_indices, surf_tensor, slow_tensor, slow_factor_tensor, args.history_len)

    train_delta_np = np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))
    train_abs = np.abs(train_delta_np.reshape(-1))
    train_ref_stats = {
        "quiet": float(np.mean(train_abs <= 0.005)),
        "shoulder": float(np.mean((train_abs > 0.01) & (train_abs <= 0.05))),
        "kurtosis": _raw_kurtosis(train_abs),
    }

    train_loader = DataLoader(JointSlowFastDataset(*train_pack), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(JointSlowFastDataset(*val_pack), batch_size=args.batch_size, shuffle=False)

    fast_history_feat_dim = 25 * 4 + (25 if args.include_scale_feature else 0)
    model = JointSlowFastSupportStateSpaceModel(
        factor_dim=args.factor_dim,
        slow_hidden_dim=args.slow_hidden_dim,
        fast_history_feat_dim=fast_history_feat_dim,
        fast_hidden_dim=args.fast_hidden_dim,
        fast_gru_layers=args.fast_gru_layers,
        fast_gru_dropout=args.fast_gru_dropout,
        flow_hidden=args.flow_hidden,
        n_coupling_layers=args.n_coupling_layers,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        support_eps=args.support_eps,
        include_scale_feature=args.include_scale_feature,
        factor_params=factor_params_np,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("220o1 jointly trained slow-fast support-aware state-space")
    print(f"  train_windows={train_pack[0].shape[0]} val_windows={val_pack[0].shape[0]}")
    print(f"  factor_dim={args.factor_dim} slow_alpha={args.slow_alpha} slow_loss_weight={args.slow_loss_weight}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_payload: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lambda_nll = args.lambda_nll_max * (epoch / args.epochs)
        slow_mix = epoch / args.epochs
        model.train()
        running = {"loss": 0.0, "energy": 0.0, "nll": 0.0, "sample_residual_delta_y_std": 0.0, "slow_mse": 0.0}
        count = 0

        for history_01, slow_history_01, slow_hist_factor, target_01, slow_target_01, slow_target_factor in train_loader:
            loss, metrics = model.training_loss(
                history_01=history_01,
                slow_history_01=slow_history_01,
                slow_hist_factor=slow_hist_factor,
                target_01=target_01,
                slow_target_01=slow_target_01,
                slow_target_factor=slow_target_factor,
                n_samples=args.train_samples,
                nll_weight=lambda_nll,
                slow_mix=slow_mix,
                slow_loss_weight=args.slow_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["energy"] += float(metrics["energy"].item()) * batch
            running["nll"] += float(metrics["nll"].item()) * batch
            running["sample_residual_delta_y_std"] += float(metrics["sample_residual_delta_y_std"].item()) * batch
            running["slow_mse"] += float(metrics["slow_mse"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        val_metrics = evaluate_h1_joint(
            model=model,
            loader=val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
            train_ref_stats=train_ref_stats,
        )
        score = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
            + 0.10 * val_metrics["val_slow_mse"]
        )
        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            "lambda_nll": lambda_nll,
            "slow_mix": slow_mix,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "220o1_joint_slow_fast_support_state_space",
                "factor_dim": args.factor_dim,
                "slow_hidden_dim": args.slow_hidden_dim,
                "fast_history_feat_dim": fast_history_feat_dim,
                "fast_hidden_dim": args.fast_hidden_dim,
                "fast_gru_layers": args.fast_gru_layers,
                "fast_gru_dropout": args.fast_gru_dropout,
                "flow_hidden": args.flow_hidden,
                "n_coupling_layers": args.n_coupling_layers,
                "history_len": args.history_len,
                "train_samples": args.train_samples,
                "eval_samples": args.eval_samples,
                "ewma_alpha": args.ewma_alpha,
                "scale_floor": args.scale_floor,
                "support_eps": args.support_eps,
                "include_scale_feature": args.include_scale_feature,
                "lambda_nll_max": args.lambda_nll_max,
                "slow_loss_weight": args.slow_loss_weight,
                "slow_alpha": args.slow_alpha,
            },
            "factor_params": factor_params_np,
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
            f"lambda={lambda_nll:.4f} slowMix={slow_mix:.2f} "
            f"loss={train_metrics['train_loss']:.4f} "
            f"slowMSE={train_metrics['train_slow_mse']:.4f} "
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
        raise RuntimeError("No checkpoint saved")
    torch.save(best_payload, out_dir / "final_model.pt")
    write_markdown_summary(
        out_dir / "training_summary.md",
        "220o1 Joint Slow-Fast Support State-Space Training",
        [
            f"- output dir: `{out_dir}`",
            f"- best epoch: `{best_payload['epoch']}`",
            f"- best selection score: `{best_score:.3f}`",
            f"- slow alpha: `{args.slow_alpha}`",
            f"- factor dim: `{args.factor_dim}`",
            f"- best val coverage90: `{best_payload['metrics']['val_coverage_90']:.3f}`",
            f"- best val q99 coverage90: `{best_payload['metrics']['val_realized_q99_coverage_90']:.3f}`",
        ],
    )


if __name__ == "__main__":
    main()
