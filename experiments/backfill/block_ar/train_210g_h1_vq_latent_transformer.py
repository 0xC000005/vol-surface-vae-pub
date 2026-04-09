#!/usr/bin/env python
"""
210g: H=1 VQ/codebook latent transformer.

Parallel branch to 210f:
  - same trusted transformer history encoder from 201a/201b
  - same one-step Student-t decoder family as 210e
  - stronger discrete commitment via VQ / nearest-code quantization
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
from experiments.backfill.block_ar.train_210e_h1_categorical_latent_transformer import (
    TargetPosteriorEncoder,
    evaluate_h1,
    warm_start_from_201b,
)


class H1VQLatentTransformer(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        n_codes: int = 16,
        code_dim: int = 128,
        posterior_hidden_dim: int = 128,
        posterior_dropout: float = 0.1,
        posterior_temp: float = 0.25,
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
        self.posterior_proj = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] + code_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
        )
        self.prior_head = nn.Linear(encoder_config["bottleneck_dim"], n_codes)
        self.codebook = nn.Embedding(n_codes, code_dim)

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.posterior_temp = posterior_temp
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.05)

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

    def posterior_embedding(self, cond: torch.Tensor, prev_u: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        post_feat = self.posterior_encoder(prev_u, target_u)
        return self.posterior_proj(torch.cat([cond, post_feat], dim=-1))

    def code_distances(self, post_embed: torch.Tensor) -> torch.Tensor:
        codes = self.codebook.weight
        post_sq = post_embed.pow(2).sum(dim=-1, keepdim=True)
        code_sq = codes.pow(2).sum(dim=-1).unsqueeze(0)
        cross = post_embed @ codes.t()
        return post_sq + code_sq - 2.0 * cross

    def quantize(self, post_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = self.code_distances(post_embed)
        code_idx = distances.argmin(dim=-1)
        quantized = self.codebook(code_idx)
        return quantized, code_idx, distances

    def decode_from_embedding(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
        z_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        prior_ce_weight: float,
        codebook_weight: float,
        commitment_weight: float,
        usage_balance_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        prior_logits = self.prior_logits(cond)
        post_embed = self.posterior_embedding(cond, prev_u, target_01)
        quantized, code_idx, distances = self.quantize(post_embed)

        quantized_st = post_embed + (quantized - post_embed).detach()
        mu, factor, diag, scale, nu = self.decode_from_embedding(cond, prev_u, quantized_st)
        nll = self.nll_from_params(target_01, mu, factor, diag, scale, nu)

        prior_ce = F.cross_entropy(prior_logits, code_idx)
        codebook_loss = F.mse_loss(quantized, post_embed.detach())
        commitment_loss = F.mse_loss(post_embed, quantized.detach())

        assignment = F.one_hot(code_idx, num_classes=self.n_codes).to(post_embed.dtype)
        marginal = assignment.mean(dim=0)
        target = torch.full_like(marginal, 1.0 / marginal.numel())
        usage_balance = F.kl_div(marginal.clamp_min(1e-8).log(), target, reduction="batchmean")

        prior_probs = torch.softmax(prior_logits, dim=-1)
        prior_log_probs = torch.log_softmax(prior_logits, dim=-1)
        post_probs = torch.softmax(-distances / self.posterior_temp, dim=-1)
        post_log_probs = torch.log(post_probs.clamp_min(1e-8))

        total = (
            nll.mean()
            + prior_ce_weight * prior_ce
            + codebook_weight * codebook_loss
            + commitment_weight * commitment_loss
            + usage_balance_weight * usage_balance
        )
        metrics = {
            "nll": nll.mean().detach(),
            "prior_ce": prior_ce.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "usage_balance": usage_balance.detach(),
            "prior_top1": prior_probs.max(dim=-1).values.mean().detach(),
            "prior_entropy": (-(prior_probs * prior_log_probs).sum(dim=-1)).mean().detach(),
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
        codes = self.codebook.weight
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_codes, self.code_dim).reshape(batch * self.n_codes, self.code_dim)
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_codes, prev_u.shape[-1]).reshape(batch * self.n_codes, prev_u.shape[-1])
        code_rep = codes.unsqueeze(0).expand(batch, self.n_codes, self.code_dim).reshape(batch * self.n_codes, self.code_dim)

        mu, factor, diag, scale, nu = self.decode_from_embedding(cond_rep, prev_rep, code_rep)
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
        z_embed = self.codebook(z_idx.reshape(-1))

        mu, factor, diag, scale, nu = self.decode_from_embedding(cond_rep, prev_rep, z_embed)
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple[H1VQLatentTransformer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_h1_vq_latent_student_t_210g":
        raise ValueError(f"Expected 210g checkpoint, got {raw_config['type']}")
    model = H1VQLatentTransformer(
        encoder_config=raw_config["encoder"],
        decoder_config=raw_config["decoder"],
        n_codes=raw_config["n_codes"],
        code_dim=raw_config["code_dim"],
        posterior_hidden_dim=raw_config["posterior_hidden_dim"],
        posterior_dropout=raw_config["posterior_dropout"],
        posterior_temp=raw_config.get("posterior_temp", 0.25),
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-4),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="210g H=1 VQ latent transformer")
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
    parser.add_argument("--posterior_temp", type=float, default=0.25)
    parser.add_argument("--prior_ce_weight", type=float, default=0.50)
    parser.add_argument("--codebook_weight", type=float, default=1.00)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--usage_balance_weight", type=float, default=0.01)
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

    model = H1VQLatentTransformer(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_codes=args.n_codes,
        code_dim=args.code_dim,
        posterior_hidden_dim=args.posterior_hidden_dim,
        posterior_dropout=args.posterior_dropout,
        posterior_temp=args.posterior_temp,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    warm_start_from_201b(model, args.base_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print("210g H=1 VQ latent transformer")
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
            "train_prior_ce": 0.0,
            "train_codebook_loss": 0.0,
            "train_commitment_loss": 0.0,
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
                prior_ce_weight=args.prior_ce_weight,
                codebook_weight=args.codebook_weight,
                commitment_weight=args.commitment_weight,
                usage_balance_weight=args.usage_balance_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            ep["train_loss"] += float(loss.item())
            ep["train_nll"] += float(metrics["nll"].item())
            ep["train_prior_ce"] += float(metrics["prior_ce"].item())
            ep["train_codebook_loss"] += float(metrics["codebook_loss"].item())
            ep["train_commitment_loss"] += float(metrics["commitment_loss"].item())
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
                "type": "transformer_h1_vq_latent_student_t_210g",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "n_codes": args.n_codes,
                "code_dim": args.code_dim,
                "posterior_hidden_dim": args.posterior_hidden_dim,
                "posterior_dropout": args.posterior_dropout,
                "posterior_temp": args.posterior_temp,
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
            f"pce={train_metrics['train_prior_ce']:.4f}  "
            f"vq={train_metrics['train_codebook_loss']:.4f}  "
            f"com={train_metrics['train_commitment_loss']:.4f}  "
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
