#!/usr/bin/env python
"""
210h: H=1 teacher-guided scenario-token transformer.

Direct supervised-token follow-up to 210e/210f/210g:
  - keep the trusted transformer history encoder
  - replace unsupervised latent discovery with teacher-guided discrete tokens
  - use one calm token plus a small event-token vocabulary
  - build event tokens from the existing local-family GT analysis
  - decode next-day Student-t law conditional on the predicted token
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_window_metadata,
    load_model as load_teacher_model,
)
from experiments.backfill.block_ar.analyze_205a_conditional_shape_family_audit import (
    collect_teacher_forced_records,
)
from experiments.backfill.block_ar.analyze_206b_local_conditional_scenario_family_pretest import (
    farthest_first_subset,
    fit_feature_space,
    transform_feat,
)
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


def build_event_prototypes_from_local_families(
    train_records: dict[str, np.ndarray],
    n_event_tokens: int,
    family_size: int,
    knn: int,
) -> np.ndarray:
    severe_idx = np.flatnonzero(train_records["q99_any"] == 1)
    if severe_idx.size == 0:
        raise RuntimeError("No q99 severe train windows available for token construction.")

    feat_bundle, train_z = fit_feature_space(train_records["cond_feat"][severe_idx], pca_dim=32)
    nn = NearestNeighbors(n_neighbors=min(knn + 1, severe_idx.size), metric="euclidean")
    nn.fit(train_z)
    _, nbr_pos = nn.kneighbors(train_z, return_distance=True)
    nbr_idx = severe_idx[nbr_pos]

    local_teacher_patterns: list[np.ndarray] = []
    for pool_pos, q_abs in enumerate(severe_idx):
        neighbors = nbr_idx[pool_pos]
        neighbors = neighbors[neighbors != q_abs]
        if neighbors.size == 0:
            neighbors = nbr_idx[pool_pos][:1]
        fam = train_records["signed_delta_norm"][neighbors]
        fam_sel = farthest_first_subset(fam, family_size)
        fam = fam[fam_sel]
        query = train_records["signed_delta_norm"][q_abs]
        mae = np.abs(fam - query[None, :]).mean(axis=1)
        local_teacher_patterns.append(fam[int(mae.argmin())].astype(np.float32))

    local_teacher_patterns_np = np.stack(local_teacher_patterns, axis=0)
    proto_sel = farthest_first_subset(local_teacher_patterns_np, n_event_tokens)
    prototypes = local_teacher_patterns_np[proto_sel].astype(np.float32)

    # Order prototypes by frequency after assignment so token ids have stable semantics.
    assign = np.abs(local_teacher_patterns_np[:, None, :] - prototypes[None, :, :]).mean(axis=-1).argmin(axis=1)
    counts = np.bincount(assign, minlength=n_event_tokens)
    order = np.argsort(counts)[::-1]
    return prototypes[order]


def assign_tokens_from_prototypes(
    records: dict[str, np.ndarray],
    event_prototypes: np.ndarray,
) -> np.ndarray:
    n_windows = int(records["window_idx"].max()) + 1
    tokens = np.zeros((n_windows,), dtype=np.int64)
    event_steps = np.flatnonzero(records["q95_any"] == 1)
    if event_steps.size == 0:
        return tokens

    patt = records["signed_delta_norm"][event_steps]
    cost = np.abs(patt[:, None, :] - event_prototypes[None, :, :]).mean(axis=-1)
    event_tokens = 1 + cost.argmin(axis=1).astype(np.int64)
    tokens[records["window_idx"][event_steps]] = event_tokens
    return tokens


def build_teacher_token_targets(
    teacher_checkpoint: str,
    train_history: torch.Tensor,
    train_target: torch.Tensor,
    val_history: torch.Tensor,
    val_target: torch.Tensor,
    q95_threshold: float,
    q99_threshold: float,
    batch_size: int,
    device: torch.device,
    n_event_tokens: int,
    family_size: int,
    knn: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    teacher_model, _payload = load_teacher_model(teacher_checkpoint, device)

    train_future = train_target.unsqueeze(1)
    val_future = val_target.unsqueeze(1)
    train_future_np = train_future.detach().cpu().numpy()
    val_future_np = val_future.detach().cpu().numpy()
    train_history_np = train_history.detach().cpu().numpy()
    val_history_np = val_history.detach().cpu().numpy()

    train_meta = build_window_metadata(train_history_np, train_future_np)
    val_meta = build_window_metadata(
        val_history_np,
        val_future_np,
        q80_vov_train=train_meta["q80_vov"],
        q80_h30_turb_train=train_meta["q80_h30_turb"],
    )

    train_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=train_history,
        future_flat=train_future,
        window_meta=train_meta,
        split_name="train",
        q95=q95_threshold,
        q99=q99_threshold,
        batch_size=batch_size,
        device=device,
    )
    val_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=val_history,
        future_flat=val_future,
        window_meta=val_meta,
        split_name="val",
        q95=q95_threshold,
        q99=q99_threshold,
        batch_size=batch_size,
        device=device,
    )

    event_prototypes = build_event_prototypes_from_local_families(
        train_records=train_records,
        n_event_tokens=n_event_tokens,
        family_size=family_size,
        knn=knn,
    )
    train_tokens = assign_tokens_from_prototypes(train_records, event_prototypes)
    val_tokens = assign_tokens_from_prototypes(val_records, event_prototypes)
    return torch.from_numpy(train_tokens), torch.from_numpy(val_tokens), event_prototypes


class H1TeacherGuidedTokenTransformer(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        n_tokens: int,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = TemporalTransformerHistoryEncoder(**encoder_config)
        self.decoder = SpatialShapeScaleStudentTDecoder(**decoder_config)
        self.severity_proj = nn.Sequential(
            nn.Linear(3, encoder_config["bottleneck_dim"]),
            nn.SiLU(),
            nn.Linear(encoder_config["bottleneck_dim"], encoder_config["bottleneck_dim"]),
        )
        self.token_embed = nn.Embedding(n_tokens, encoder_config["bottleneck_dim"])
        self.token_head = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] * 2, encoder_config["bottleneck_dim"]),
            nn.SiLU(),
            nn.Linear(encoder_config["bottleneck_dim"], n_tokens),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] * 3, encoder_config["bottleneck_dim"]),
            nn.SiLU(),
            nn.Linear(encoder_config["bottleneck_dim"], encoder_config["bottleneck_dim"]),
        )
        self.film_scale = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] * 2, encoder_config["bottleneck_dim"]),
            nn.Tanh(),
        )
        self.film_shift = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] * 2, encoder_config["bottleneck_dim"]),
            nn.SiLU(),
            nn.Linear(encoder_config["bottleneck_dim"], encoder_config["bottleneck_dim"]),
        )

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_tokens = n_tokens
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
        history_norm = history_flat * 2.0 - 1.0
        cond, _attn = self.encoder(history_norm)
        return cond

    def _prev_u(self, history_01: torch.Tensor) -> torch.Tensor:
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        return iv_to_unconstrained(prev_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)

    def severity_features(self, history_01: torch.Tensor) -> torch.Tensor:
        hist_mean = history_01.mean(dim=(-1, -2))
        vov = (hist_mean[:, 1:] - hist_mean[:, :-1]).std(dim=1)
        last_mean = hist_mean[:, -1]
        trend = hist_mean[:, -1] - hist_mean[:, 0]
        return self.severity_proj(torch.stack([vov, last_mean, trend], dim=-1))

    def token_logits(self, cond: torch.Tensor, sev_feat: torch.Tensor) -> torch.Tensor:
        return self.token_head(torch.cat([cond, sev_feat], dim=-1))

    def decode_from_token(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
        sev_feat: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tok_embed = self.token_embed(token_ids)
        fused = self.cond_fuse(torch.cat([cond, sev_feat, tok_embed], dim=-1))
        film_in = torch.cat([sev_feat, tok_embed], dim=-1)
        cond_tok = fused * (1.0 + self.film_scale(film_in)) + self.film_shift(film_in)
        return self.decoder(cond_tok, prev_u)

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
        pi = target_u.new_tensor(np.pi)
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
        token_targets: torch.Tensor,
        class_weights: torch.Tensor,
        token_loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        sev_feat = self.severity_features(history_01)
        logits = self.token_logits(cond, sev_feat)
        token_ce = F.cross_entropy(logits, token_targets, weight=class_weights)

        mu, factor, diag, scale, nu = self.decode_from_token(cond, prev_u, sev_feat, token_targets)
        nll = self.nll_from_params(target_01, mu, factor, diag, scale, nu)

        probs = torch.softmax(logits, dim=-1)
        token_top1 = (logits.argmax(dim=-1) == token_targets).float().mean()
        event_mask = token_targets > 0
        if event_mask.any():
            event_top1 = (logits[event_mask].argmax(dim=-1) == token_targets[event_mask]).float().mean()
            event_top3 = (
                logits[event_mask].topk(k=min(3, self.n_tokens), dim=-1).indices == token_targets[event_mask].unsqueeze(-1)
            ).any(dim=-1).float().mean()
        else:
            event_top1 = token_targets.new_tensor(0.0, dtype=torch.float32)
            event_top3 = token_targets.new_tensor(0.0, dtype=torch.float32)
        marginal = probs.mean(dim=0)
        total = nll.mean() + token_loss_weight * token_ce
        metrics = {
            "nll": nll.mean().detach(),
            "token_ce": token_ce.detach(),
            "token_top1": token_top1.detach(),
            "event_token_top1": event_top1.detach(),
            "event_token_top3": event_top3.detach(),
            "prior_top1": probs.max(dim=-1).values.mean().detach(),
            "prior_entropy": (-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)).mean().detach(),
            "active_tokens": torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum()).detach(),
        }
        return total, metrics

    @torch.no_grad()
    def exact_marginal_nll(self, history_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        sev_feat = self.severity_features(history_01)
        logits = self.token_logits(cond, sev_feat)
        log_probs = torch.log_softmax(logits, dim=-1)
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)

        batch = history_01.shape[0]
        token_ids = torch.arange(self.n_tokens, device=history_01.device)
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_tokens, cond.shape[-1]).reshape(batch * self.n_tokens, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_tokens, prev_u.shape[-1]).reshape(batch * self.n_tokens, prev_u.shape[-1])
        sev_rep = sev_feat.unsqueeze(1).expand(batch, self.n_tokens, sev_feat.shape[-1]).reshape(batch * self.n_tokens, sev_feat.shape[-1])
        tok_rep = token_ids.unsqueeze(0).expand(batch, self.n_tokens).reshape(-1)

        mu, factor, diag, scale, nu = self.decode_from_token(cond_rep, prev_rep, sev_rep, tok_rep)
        log_prob = self.student_t_log_prob(
            target_u.unsqueeze(1).expand(batch, self.n_tokens, target_u.shape[-1]).reshape(batch * self.n_tokens, target_u.shape[-1]),
            mu,
            factor,
            diag,
            scale,
            nu,
        ).view(batch, self.n_tokens)
        return -torch.logsumexp(log_probs + log_prob, dim=-1)

    @torch.no_grad()
    def prior_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        sev_feat = self.severity_features(history_01)
        probs = torch.softmax(self.token_logits(cond, sev_feat), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        marginal = probs.mean(dim=0)
        active = torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum())
        return {
            "prior_top1_mean": probs.max(dim=-1).values.mean(),
            "prior_entropy_mean": entropy.mean(),
            "prior_active_tokens": active,
            "prior_marginal_probs": marginal,
        }

    @torch.no_grad()
    def token_effect_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        sev_feat = self.severity_features(history_01)
        batch = history_01.shape[0]
        token_ids = torch.arange(self.n_tokens, device=history_01.device)
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_tokens, cond.shape[-1]).reshape(batch * self.n_tokens, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_tokens, prev_u.shape[-1]).reshape(batch * self.n_tokens, prev_u.shape[-1])
        sev_rep = sev_feat.unsqueeze(1).expand(batch, self.n_tokens, sev_feat.shape[-1]).reshape(batch * self.n_tokens, sev_feat.shape[-1])
        tok_rep = token_ids.unsqueeze(0).expand(batch, self.n_tokens).reshape(-1)
        mu, _factor, _diag, scale, _nu = self.decode_from_token(cond_rep, prev_rep, sev_rep, tok_rep)
        mu = mu.view(batch, self.n_tokens, -1)
        scale = scale.view(batch, self.n_tokens)
        mu_disp = (mu - mu.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2)).sqrt().mean()
        scale_disp = (scale - scale.mean(dim=1, keepdim=True)).pow(2).mean(dim=1).sqrt().mean()
        return {"mu_dispersion": mu_disp, "scale_dispersion": scale_disp}

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        sev_feat = self.severity_features(history_01)
        probs = torch.softmax(self.token_logits(cond, sev_feat), dim=-1)
        token_idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)

        batch = history_01.shape[0]
        cond_rep = cond.unsqueeze(1).expand(batch, n_samples, cond.shape[-1]).reshape(batch * n_samples, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, n_samples, prev_u.shape[-1]).reshape(batch * n_samples, prev_u.shape[-1])
        sev_rep = sev_feat.unsqueeze(1).expand(batch, n_samples, sev_feat.shape[-1]).reshape(batch * n_samples, sev_feat.shape[-1])
        tok_rep = token_idx.reshape(-1)

        mu, factor, diag, scale, nu = self.decode_from_token(cond_rep, prev_rep, sev_rep, tok_rep)
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


def warm_start_encoder_decoder(model: H1TeacherGuidedTokenTransformer, checkpoint_path: str | None) -> None:
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
    model: H1TeacherGuidedTokenTransformer,
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
        "val_mu_dispersion": 0.0,
        "val_scale_dispersion": 0.0,
        "val_token_top1": 0.0,
        "val_event_token_top1": 0.0,
        "val_event_token_top3": 0.0,
    }
    q95_cover_sum = 0.0
    q99_cover_sum = 0.0
    q95_count = 0
    q99_count = 0
    event_batch_count = 0
    total_count = 0
    gt_delta_all = []
    sample_delta_all = []
    marginals = []

    for history_01, target_01, token_targets in loader:
        nll = model.exact_marginal_nll(history_01, target_01)
        samples = model.sample_next_iv(history_01, n_samples=eval_samples)
        stats = model.prior_statistics(history_01)
        disp = model.token_effect_statistics(history_01)

        cond = model.encode(history_01)
        sev_feat = model.severity_features(history_01)
        logits = model.token_logits(cond, sev_feat)
        token_top1 = (logits.argmax(dim=-1) == token_targets).float().mean()
        event_mask = token_targets > 0
        if event_mask.any():
            event_top1 = (logits[event_mask].argmax(dim=-1) == token_targets[event_mask]).float().mean()
            event_top3 = (
                logits[event_mask].topk(k=min(3, model.n_tokens), dim=-1).indices == token_targets[event_mask].unsqueeze(-1)
            ).any(dim=-1).float().mean()
        else:
            event_top1 = token_targets.new_tensor(0.0, dtype=torch.float32)
            event_top3 = token_targets.new_tensor(0.0, dtype=torch.float32)

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
        totals["val_mu_dispersion"] += float(disp["mu_dispersion"].item()) * batch_size
        totals["val_scale_dispersion"] += float(disp["scale_dispersion"].item()) * batch_size
        totals["val_token_top1"] += float(token_top1.item()) * batch_size

        if event_mask.any():
            n_event = int(event_mask.sum().item())
            totals["val_event_token_top1"] += float(event_top1.item()) * n_event
            totals["val_event_token_top3"] += float(event_top3.item()) * n_event
            event_batch_count += n_event

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
            "val_prior_active_tokens": active,
            "val_event_token_top1": totals["val_event_token_top1"] / max(event_batch_count, 1),
            "val_event_token_top3": totals["val_event_token_top3"] / max(event_batch_count, 1),
            "val_event_window_count": event_batch_count,
        }
    )
    return metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[H1TeacherGuidedTokenTransformer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_h1_teacher_guided_token_student_t_210h":
        raise ValueError(f"Expected 210h checkpoint, got {raw_config['type']}")
    model = H1TeacherGuidedTokenTransformer(
        encoder_config=raw_config["encoder"],
        decoder_config=raw_config["decoder"],
        n_tokens=raw_config["n_tokens"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-4),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="210h H=1 teacher-guided token transformer")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default="models/backfill/transformer_h1_categorical_latent_student_t_210e_smoke512/best_model.pt",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--val_samples", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_event_tokens", type=int, default=8)
    parser.add_argument("--token_loss_weight", type=float, default=0.5)
    parser.add_argument("--family_size", type=int, default=3)
    parser.add_argument("--knn", type=int, default=32)
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

    train_tokens, val_tokens, event_prototypes = build_teacher_token_targets(
        teacher_checkpoint=args.teacher_checkpoint,
        train_history=train_hist,
        train_target=train_target,
        val_history=val_hist,
        val_target=val_target,
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        batch_size=args.batch_size,
        device=device,
        n_event_tokens=args.n_event_tokens,
        family_size=args.family_size,
        knn=args.knn,
    )

    n_tokens = args.n_event_tokens + 1
    train_counts = torch.bincount(train_tokens, minlength=n_tokens).float()
    class_weights = 1.0 / torch.sqrt(train_counts.clamp_min(1.0))
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    train_loader = DataLoader(
        TensorDataset(train_hist, train_target, train_tokens.to(device)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_target, val_tokens.to(device)),
        batch_size=args.batch_size,
        shuffle=False,
    )

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
        cond_dim=args.d_model,
        rank=args.rank,
        fixed_nu=args.fixed_nu,
    )

    model = H1TeacherGuidedTokenTransformer(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_tokens=n_tokens,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    warm_start_encoder_decoder(model, args.base_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print("210h H=1 teacher-guided token transformer")
    print(f"  Train windows: {train_hist.shape[0]}")
    print(f"  Val windows:   {val_hist.shape[0]}")
    print(f"  Params:        {n_params:,}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(f"  Tokens={n_tokens} (1 calm + {args.n_event_tokens} event)")
    print(f"  Train token counts: {train_counts.tolist()}")

    best_score = float("inf")
    best_metrics: dict[str, Any] | None = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "train_loss": 0.0,
            "train_nll": 0.0,
            "train_token_ce": 0.0,
            "train_token_top1": 0.0,
            "train_event_token_top1": 0.0,
            "train_event_token_top3": 0.0,
            "train_prior_top1": 0.0,
            "train_prior_entropy": 0.0,
            "train_active_tokens": 0.0,
        }
        nb = 0

        for history_01, target_01, token_targets in train_loader:
            optimizer.zero_grad()
            loss, metrics = model.train_objective(
                history_01,
                target_01,
                token_targets=token_targets,
                class_weights=class_weights,
                token_loss_weight=args.token_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            ep["train_loss"] += float(loss.item())
            ep["train_nll"] += float(metrics["nll"].item())
            ep["train_token_ce"] += float(metrics["token_ce"].item())
            ep["train_token_top1"] += float(metrics["token_top1"].item())
            ep["train_event_token_top1"] += float(metrics["event_token_top1"].item())
            ep["train_event_token_top3"] += float(metrics["event_token_top3"].item())
            ep["train_prior_top1"] += float(metrics["prior_top1"].item())
            ep["train_prior_entropy"] += float(metrics["prior_entropy"].item())
            ep["train_active_tokens"] += float(metrics["active_tokens"].item())
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in ep.items()}
        val_metrics = evaluate_h1(model, val_loader, q95_threshold, q99_threshold, args.val_samples)

        gap = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, val_metrics["val_coverage_90"] - 0.93)
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
            + max(0.0, 0.10 - val_metrics["val_prior_top1_mean"])
            + max(0.0, 0.01 - val_metrics["val_mu_dispersion"])
        )
        selection_score = gap + 0.01 * val_metrics["val_nll"]
        val_metrics["selection_score"] = selection_score
        val_metrics["selection_gap"] = gap

        row = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_sec": time.time() - t0}
        history.append(make_serializable(row))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "type": "transformer_h1_teacher_guided_token_student_t_210h",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "n_tokens": n_tokens,
                "n_event_tokens": args.n_event_tokens,
                "family_size": args.family_size,
                "knn": args.knn,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
                "teacher_checkpoint": args.teacher_checkpoint,
                "event_prototypes": event_prototypes.tolist(),
                "train_token_counts": train_counts.tolist(),
                "class_weights": class_weights.detach().cpu().tolist(),
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
            f"tokCE={train_metrics['train_token_ce']:.4f}  "
            f"tokTop1={train_metrics['train_token_top1']:.3f}  "
            f"evtTop1={train_metrics['train_event_token_top1']:.3f}  "
            f"valNLL={val_metrics['val_nll']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"ptop={val_metrics['val_prior_top1_mean']:.3f}  "
            f"disp={val_metrics['val_mu_dispersion']:.4f}  "
            f"score={selection_score:.4f}"
        )

    summary = {
        "best_score": best_score,
        "best_metrics": make_serializable(best_metrics),
        "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        "train_token_counts": train_counts.tolist(),
        "event_prototypes": event_prototypes.tolist(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(make_serializable(summary), f, indent=2)

    print("Done.")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
