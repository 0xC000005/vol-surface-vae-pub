#!/usr/bin/env python
"""
210e: H=1 categorical-latent transformer.

Direct one-step multimodal branch with harder scenario commitment:
  - trusted transformer history encoder from 201a/201b
  - discrete latent scenario code z
  - prior p(z | history)
  - posterior q(z | history, next_day)
  - one Student-t decoder conditioned on history + sampled code
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    iv_to_unconstrained,
    make_serializable,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    SpatialShapeScaleStudentTDecoder,
)
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    TemporalTransformerHistoryEncoder,
)
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import (
    compute_h1_shape_stats,
)


class TargetPosteriorEncoder(nn.Module):
    def __init__(self, n_cells: int = 25, hidden_dim: int = 128, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        input_dim = n_cells * 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, prev_u: torch.Tensor, target_u: torch.Tensor) -> torch.Tensor:
        delta = target_u - prev_u
        feat = torch.cat([prev_u, target_u, delta, delta.abs()], dim=-1)
        return self.net(feat)


class H1CategoricalLatentTransformer(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        n_codes: int = 16,
        code_dim: int = 128,
        posterior_hidden_dim: int = 128,
        posterior_dropout: float = 0.1,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = TemporalTransformerHistoryEncoder(**encoder_config)
        self.decoder = SpatialShapeScaleStudentTDecoder(**decoder_config)
        self.posterior_encoder = TargetPosteriorEncoder(
            n_cells=decoder_config.get("n_cells", 25),
            hidden_dim=posterior_hidden_dim,
            out_dim=code_dim,
            dropout=posterior_dropout,
        )
        self.prior_head = nn.Linear(encoder_config["bottleneck_dim"], n_codes)
        self.posterior_head = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] + code_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, n_codes),
        )
        self.codebook = nn.Embedding(n_codes, code_dim)

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
        history_norm = history_flat * 2.0 - 1.0
        cond, _attn = self.encoder(history_norm)
        return cond

    def _prev_u(self, history_01: torch.Tensor) -> torch.Tensor:
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        return iv_to_unconstrained(prev_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)

    def prior_logits(self, cond: torch.Tensor) -> torch.Tensor:
        return self.prior_head(cond)

    def posterior_logits(self, cond: torch.Tensor, prev_u: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        post_feat = self.posterior_encoder(prev_u, target_u)
        return self.posterior_head(torch.cat([cond, post_feat], dim=-1))

    def decode_from_code(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
        z_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_embed = z_onehot @ self.codebook.weight
        cond_z = cond + z_embed
        return self.decoder(cond_z, prev_u)

    def normalized_components(
        self, factor: torch.Tensor, diag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_diag = factor.pow(2).sum(dim=-1) + diag.pow(2) + self.cov_jitter
        avg_var = raw_diag.mean(dim=-1).clamp_min(self.cov_jitter)
        norm = avg_var.sqrt().unsqueeze(-1)
        factor_norm = factor / norm.unsqueeze(-1)
        diag_norm = diag / norm
        return factor_norm, diag_norm, avg_var

    def covariance(self, factor: torch.Tensor, diag: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        cov = factor_norm @ factor_norm.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag_norm.pow(2) + self.cov_jitter)
        cov = cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
        return cov

    def student_t_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)
        diff = (target_u - mu).unsqueeze(-1)
        solved = torch.cholesky_solve(diff, chol).squeeze(-1)
        mahal = (diff.squeeze(-1) * solved).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        d = target_u.shape[-1]
        nu = nu.clamp_min(self.decoder.nu_floor + 1e-6)
        pi = target_u.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (d * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + d) * torch.log1p(mahal / nu)
        return log_norm + log_kernel

    def nll_from_params(
        self,
        target_01: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        return -self.student_t_log_prob(target_u, mu, factor, diag, scale, nu)

    def train_objective(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        temperature: float,
        kl_weight: float,
        usage_balance_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        prior_logits = self.prior_logits(cond)
        post_logits = self.posterior_logits(cond, prev_u, target_01)

        post_probs = torch.softmax(post_logits, dim=-1)
        prior_log_probs = torch.log_softmax(prior_logits, dim=-1)
        post_log_probs = torch.log(post_probs.clamp_min(1e-8))
        z_st = F.gumbel_softmax(post_logits, tau=temperature, hard=True, dim=-1)

        mu, factor, diag, scale, nu = self.decode_from_code(cond, prev_u, z_st)
        nll = self.nll_from_params(target_01, mu, factor, diag, scale, nu)
        kl = (post_probs * (post_log_probs - prior_log_probs)).sum(dim=-1)

        marginal = post_probs.mean(dim=0)
        target = torch.full_like(marginal, 1.0 / marginal.numel())
        usage_balance = F.kl_div(marginal.clamp_min(1e-8).log(), target, reduction="batchmean")

        total = nll.mean() + kl_weight * kl.mean() + usage_balance_weight * usage_balance
        metrics = {
            "nll": nll.mean().detach(),
            "kl": kl.mean().detach(),
            "usage_balance": usage_balance.detach(),
            "prior_top1": torch.softmax(prior_logits, dim=-1).max(dim=-1).values.mean().detach(),
            "prior_entropy": (-(torch.softmax(prior_logits, dim=-1) * prior_log_probs).sum(dim=-1)).mean().detach(),
            "post_top1": post_probs.max(dim=-1).values.mean().detach(),
            "post_entropy": (-(post_probs * post_log_probs).sum(dim=-1)).mean().detach(),
            "active_codes": torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum()).detach(),
        }
        return total, metrics

    @torch.no_grad()
    def exact_marginal_nll(self, history_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        prior_logits = self.prior_logits(cond)
        prior_log_probs = torch.log_softmax(prior_logits, dim=-1)
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)

        batch = history_01.shape[0]
        eye = torch.eye(self.n_codes, device=history_01.device, dtype=history_01.dtype)
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_codes, self.code_dim).reshape(batch * self.n_codes, self.code_dim)
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_codes, prev_u.shape[-1]).reshape(batch * self.n_codes, prev_u.shape[-1])
        z_rep = eye.unsqueeze(0).expand(batch, self.n_codes, self.n_codes).reshape(batch * self.n_codes, self.n_codes)

        mu, factor, diag, scale, nu = self.decode_from_code(cond_rep, prev_rep, z_rep)
        log_prob = self.student_t_log_prob(
            target_u.unsqueeze(1).expand(batch, self.n_codes, target_u.shape[-1]).reshape(batch * self.n_codes, target_u.shape[-1]),
            mu,
            factor,
            diag,
            scale,
            nu,
        ).view(batch, self.n_codes)
        return -torch.logsumexp(prior_log_probs + log_prob, dim=-1)

    @torch.no_grad()
    def prior_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        probs = torch.softmax(self.prior_logits(cond), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        marginal = probs.mean(dim=0)
        active = torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum())
        return {
            "prior_top1_mean": probs.max(dim=-1).values.mean(),
            "prior_entropy_mean": entropy.mean(),
            "prior_active_codes": active,
            "prior_marginal_probs": marginal,
        }

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        probs = torch.softmax(self.prior_logits(cond), dim=-1)
        z_idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)

        batch = history_01.shape[0]
        cond_rep = cond.unsqueeze(1).expand(batch, n_samples, self.code_dim).reshape(batch * n_samples, self.code_dim)
        prev_rep = prev_u.unsqueeze(1).expand(batch, n_samples, prev_u.shape[-1]).reshape(batch * n_samples, prev_u.shape[-1])
        z_onehot = F.one_hot(z_idx, num_classes=self.n_codes).to(cond.dtype).reshape(batch * n_samples, self.n_codes)

        mu, factor, diag, scale, nu = self.decode_from_code(cond_rep, prev_rep, z_onehot)
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        eps_lowrank = torch.randn(batch * n_samples, factor.shape[-1], device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(batch * n_samples, mu.shape[-1], device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bcr,br->bc", factor_norm, eps_lowrank)
        diag_noise = diag_norm * eps_diag
        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.sample().clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)
        samples = mu + (lowrank_noise + diag_noise) * scale.unsqueeze(-1) * t_scale
        return samples.view(batch, n_samples, -1)

    @torch.no_grad()
    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        return unconstrained_to_iv(self.sample_next_u(history_01, n_samples=n_samples), lo=self.support_lo, hi=self.support_hi)


def warm_start_from_201b(model: H1CategoricalLatentTransformer, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        print("No warm start")
        return
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Warm start missing: {checkpoint_path}")
        return
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    model_state = model.state_dict()
    loaded = 0
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape and (
            key.startswith("encoder.") or key.startswith("decoder.")
        ):
            model_state[key] = value
            loaded += 1
    model.load_state_dict(model_state, strict=False)
    print(f"Loaded warm start from {checkpoint_path} ({loaded} tensors)")


@torch.no_grad()
def evaluate_h1(
    model: H1CategoricalLatentTransformer,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "val_nll": 0.0,
        "val_mae": 0.0,
        "val_coverage_90": 0.0,
        "val_width_90": 0.0,
        "val_prior_top1_mean": 0.0,
        "val_prior_entropy_mean": 0.0,
    }
    q95_cover_sum = 0.0
    q99_cover_sum = 0.0
    q95_count = 0
    q99_count = 0
    total_count = 0
    gt_delta_all = []
    sample_delta_all = []
    marginals = []

    for history_01, target_01 in loader:
        nll = model.exact_marginal_nll(history_01, target_01)
        samples = model.sample_next_iv(history_01, n_samples=eval_samples)
        stats = model.prior_statistics(history_01)

        q05 = samples.quantile(0.05, dim=1)
        q95 = samples.quantile(0.95, dim=1)
        mean_pred = samples.mean(dim=1)

        prev = history_01[:, -1].reshape(history_01.shape[0], -1)
        target_abs = (target_01 - prev).abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold

        coverage = ((target_01 >= q05) & (target_01 <= q95)).float().mean()
        mae = (mean_pred - target_01).abs().mean()
        width = (q95 - q05).mean()
        q95_cov = ((target_01[q95_mask] >= q05[q95_mask]) & (target_01[q95_mask] <= q95[q95_mask])).float().mean() if q95_mask.any() else target_01.new_tensor(0.0)
        q99_cov = ((target_01[q99_mask] >= q05[q99_mask]) & (target_01[q99_mask] <= q95[q99_mask])).float().mean() if q99_mask.any() else target_01.new_tensor(0.0)

        batch_size = history_01.shape[0]
        totals["val_nll"] += float(nll.mean().item()) * batch_size
        totals["val_mae"] += float(mae.item()) * batch_size
        totals["val_coverage_90"] += float(coverage.item()) * batch_size
        totals["val_width_90"] += float(width.item()) * batch_size
        totals["val_prior_top1_mean"] += float(stats["prior_top1_mean"].item()) * batch_size
        totals["val_prior_entropy_mean"] += float(stats["prior_entropy_mean"].item()) * batch_size

        if q95_mask.any():
            q95_cover_sum += float(q95_cov.item()) * int(q95_mask.sum().item())
            q95_count += int(q95_mask.sum().item())
        if q99_mask.any():
            q99_cover_sum += float(q99_cov.item()) * int(q99_mask.sum().item())
            q99_count += int(q99_mask.sum().item())

        gt_delta_all.append((target_01 - prev).detach().cpu().numpy())
        sample_delta_all.append((samples - prev.unsqueeze(1)).detach().cpu().numpy())
        marginals.append(stats["prior_marginal_probs"].detach().cpu().numpy())
        total_count += batch_size

    gt_delta = np.concatenate(gt_delta_all, axis=0)
    sample_delta = np.concatenate(sample_delta_all, axis=0)
    shape = compute_h1_shape_stats(gt_delta, sample_delta)
    marginal = np.mean(np.stack(marginals, axis=0), axis=0)
    active = float(np.exp(-(marginal * np.log(np.clip(marginal, 1e-8, None))).sum()))

    metrics = {k: v / max(total_count, 1) for k, v in totals.items()}
    metrics.update(
        {
            "val_realized_q95_coverage_90": q95_cover_sum / max(q95_count, 1),
            "val_realized_q99_coverage_90": q99_cover_sum / max(q99_count, 1),
            "val_q95_cell_count": q95_count,
            "val_q99_cell_count": q99_count,
            "val_h1_quiet_ratio": shape["quiet_ratio"],
            "val_h1_shoulder_ratio": shape["shoulder_ratio"],
            "val_h1_extreme_ratio": shape["extreme_ratio"],
            "val_h1_kurtosis_ratio": shape["kurtosis_ratio"],
            "val_prior_active_codes": active,
        }
    )
    return metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[H1CategoricalLatentTransformer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_h1_categorical_latent_student_t_210e":
        raise ValueError(f"Expected 210e checkpoint, got {raw_config['type']}")
    model = H1CategoricalLatentTransformer(
        encoder_config=raw_config["encoder"],
        decoder_config=raw_config["decoder"],
        n_codes=raw_config["n_codes"],
        code_dim=raw_config["code_dim"],
        posterior_hidden_dim=raw_config["posterior_hidden_dim"],
        posterior_dropout=raw_config["posterior_dropout"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-4),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="210e H=1 categorical latent transformer")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--val_samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_codes", type=int, default=16)
    parser.add_argument("--code_dim", type=int, default=128)
    parser.add_argument("--posterior_hidden_dim", type=int, default=128)
    parser.add_argument("--posterior_dropout", type=float, default=0.10)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--usage_balance_weight", type=float, default=0.01)
    parser.add_argument("--gumbel_temp", type=float, default=0.50)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    encoder_config = dict(
        input_dim=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        bottleneck_dim=args.d_model,
        max_len=max(args.history_len, 64),
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=args.code_dim,
        rank=args.rank,
        fixed_nu=args.fixed_nu,
    )

    model = H1CategoricalLatentTransformer(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_codes=args.n_codes,
        code_dim=args.code_dim,
        posterior_hidden_dim=args.posterior_hidden_dim,
        posterior_dropout=args.posterior_dropout,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    warm_start_from_201b(model, args.base_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print("210e H=1 categorical latent transformer")
    print(f"  Train windows: {train_hist.shape[0]}")
    print(f"  Val windows:   {val_hist.shape[0]}")
    print(f"  Params:        {n_params:,}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(f"  Codes={args.n_codes} code_dim={args.code_dim}")

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "train_loss": 0.0,
            "train_nll": 0.0,
            "train_kl": 0.0,
            "train_usage_balance": 0.0,
            "train_prior_top1": 0.0,
            "train_prior_entropy": 0.0,
            "train_post_top1": 0.0,
            "train_post_entropy": 0.0,
            "train_active_codes": 0.0,
        }
        nb = 0

        for history_01, target_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = model.train_objective(
                history_01,
                target_01,
                temperature=args.gumbel_temp,
                kl_weight=args.kl_weight,
                usage_balance_weight=args.usage_balance_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            ep["train_loss"] += float(loss.item())
            ep["train_nll"] += float(metrics["nll"].item())
            ep["train_kl"] += float(metrics["kl"].item())
            ep["train_usage_balance"] += float(metrics["usage_balance"].item())
            ep["train_prior_top1"] += float(metrics["prior_top1"].item())
            ep["train_prior_entropy"] += float(metrics["prior_entropy"].item())
            ep["train_post_top1"] += float(metrics["post_top1"].item())
            ep["train_post_entropy"] += float(metrics["post_entropy"].item())
            ep["train_active_codes"] += float(metrics["active_codes"].item())
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in ep.items()}
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.val_samples,
        )

        gap = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, val_metrics["val_coverage_90"] - 0.93)
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
            + max(0.0, 1.50 - val_metrics["val_prior_active_codes"])
        )
        selection_score = gap + 0.01 * val_metrics["val_nll"]
        val_metrics["selection_score"] = selection_score
        val_metrics["selection_gap"] = gap

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "elapsed_sec": time.time() - t0,
        }
        history.append(make_serializable(row))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "type": "transformer_h1_categorical_latent_student_t_210e",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "n_codes": args.n_codes,
                "code_dim": args.code_dim,
                "posterior_hidden_dim": args.posterior_hidden_dim,
                "posterior_dropout": args.posterior_dropout,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
            },
            "metrics": make_serializable(row),
        }
        torch.save(ckpt, output_dir / "final_model.pt")
        if selection_score < best_score:
            best_score = selection_score
            best_metrics = dict(row)
            torch.save(ckpt, output_dir / "best_model.pt")

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"loss={train_metrics['train_loss']:.4f}  "
            f"nll={train_metrics['train_nll']:.4f}  "
            f"kl={train_metrics['train_kl']:.4f}  "
            f"postTop={train_metrics['train_post_top1']:.3f}  "
            f"act={train_metrics['train_active_codes']:.2f}  "
            f"valNLL={val_metrics['val_nll']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"codes={val_metrics['val_prior_active_codes']:.2f}  "
            f"score={selection_score:.4f}"
        )

    result = {
        "best_score": best_score,
        "best_metrics": make_serializable(best_metrics),
        "n_params": n_params,
        "q95_threshold": q95_threshold,
        "q99_threshold": q99_threshold,
    }
    with open(output_dir / "result_summary.json", "w") as f:
        json.dump(make_serializable(result), f, indent=2)
    print("\nDone.")
    print(json.dumps(make_serializable(result), indent=2))


if __name__ == "__main__":
    main()
