#!/usr/bin/env python
"""
208b: H=1 hard-separated local-family Student-t.

Strict calm/event separation:
  - warm-start base path from 201b
  - freeze calm/base path
  - train only event gate + local family heads
  - use hard severe-family assignment
  - use localized event noise instead of shared broad covariance
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    load_model as load_201b_teacher_model,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    iv_to_unconstrained,
    make_serializable,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_207a_local_conditional_family_student_t import (
    TransformerLocalConditionalFamilyARModel,
    evaluate_family_statistics,
)
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import (
    build_teacher_guidance_targets,
    compute_h1_shape_stats,
)


class HardSeparatedH1LocalFamilyModel(TransformerLocalConditionalFamilyARModel):
    def __init__(
        self,
        *args,
        event_noise_floor: float = 0.002,
        event_noise_mult: float = 0.35,
        event_shape_floor: float = 0.05,
        event_nu: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_noise_floor = float(event_noise_floor)
        self.event_noise_mult = float(event_noise_mult)
        self.event_shape_floor = float(event_shape_floor)
        self.event_nu = float(event_nu)

    def base_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        cov = self.covariance(factor, diag, scale)
        return self._student_t_log_prob(target_u, mu, cov, nu)

    def event_diag(
        self,
        family_scale: torch.Tensor,
        chosen_shape: torch.Tensor,
    ) -> torch.Tensor:
        return self.event_noise_floor + self.event_noise_mult * (
            family_scale.unsqueeze(-1) * chosen_shape.abs().clamp_min(self.event_shape_floor)
        )

    def diag_student_t_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        diag: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        var = diag.pow(2) + self.cov_jitter
        diff = target_u - mu
        mahal = (diff.pow(2) / var).sum(dim=-1)
        logdet = torch.log(var).sum(dim=-1)
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

    def event_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        family_scale: torch.Tensor,
        chosen_shape: torch.Tensor,
    ) -> torch.Tensor:
        event_mu = mu + family_scale.unsqueeze(-1) * chosen_shape
        event_diag = self.event_diag(family_scale, chosen_shape)
        nu = target_u.new_full((target_u.shape[0],), self.event_nu)
        return self.diag_student_t_log_prob(target_u, event_mu, event_diag, nu)

    @torch.no_grad()
    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mu, factor, diag, scale, nu = self.forward_from_history(history_01)
        family = self.last_family_params()

        base_samples_u = self._sample_base_u_from_params(mu, factor, diag, scale, nu, n_samples=n_samples)
        gate_prob = torch.sigmoid(family["gate_logit"]).clamp(self.gate_prob_eps, 1.0 - self.gate_prob_eps)
        family_probs = torch.softmax(family["family_logits"], dim=-1)

        batch_size = mu.shape[0]
        gate = (torch.rand(batch_size, n_samples, device=mu.device) < gate_prob.unsqueeze(1)).to(mu.dtype)
        family_idx = torch.multinomial(family_probs, num_samples=n_samples, replacement=True)
        batch_idx = torch.arange(batch_size, device=mu.device)[:, None]
        chosen_shape = family["family_shapes"][batch_idx, family_idx]

        event_mu = mu.unsqueeze(1) + family["family_scale"].unsqueeze(1).unsqueeze(-1) * chosen_shape
        event_diag = self.event_diag(
            family["family_scale"].unsqueeze(1).expand(batch_size, n_samples).reshape(-1),
            chosen_shape.reshape(batch_size * n_samples, -1),
        ).reshape(batch_size, n_samples, -1)
        eps = torch.randn_like(event_mu)
        gamma = torch.distributions.Gamma(
            event_mu.new_full((batch_size, n_samples), self.event_nu / 2.0),
            event_mu.new_full((batch_size, n_samples), self.event_nu / 2.0),
        )
        mix = gamma.sample().clamp_min(1e-6)
        event_samples_u = event_mu + eps * event_diag * torch.rsqrt(mix).unsqueeze(-1)

        samples_u = (1.0 - gate.unsqueeze(-1)) * base_samples_u + gate.unsqueeze(-1) * event_samples_u
        return unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)


def warm_start_from_201b(model: HardSeparatedH1LocalFamilyModel, checkpoint: str) -> None:
    teacher_model, payload = load_201b_teacher_model(checkpoint, next(model.parameters()).device)
    teacher_state = teacher_model.state_dict()
    model.load_state_dict(teacher_state, strict=False)


def freeze_base_path(model: HardSeparatedH1LocalFamilyModel) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for name, param in model.decoder.named_parameters():
        if any(key in name for key in [
            "gate_logit_head",
            "family_logit_head",
            "family_scale_head",
            "family_shape_head",
        ]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def hard_separated_objective(
    model: HardSeparatedH1LocalFamilyModel,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    gate_targets: torch.Tensor,
    family_mask: torch.Tensor,
    family_shapes_target: torch.Tensor,
    family_probs_target: torch.Tensor,
    gate_pos_weight: float,
    gate_rate_target: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mu, factor, diag, scale, nu = model.forward_from_history(history_01)
    family = model.last_family_params()
    target_u = iv_to_unconstrained(
        target_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )

    gate_t = gate_targets.to(history_01.device)
    family_mask_t = family_mask.to(history_01.device) > 0.5
    base_mask = gate_t < 0.5
    hard_idx = family_probs_target.argmax(dim=-1).to(history_01.device)

    pos_weight = history_01.new_tensor(gate_pos_weight)
    gate_prob = torch.sigmoid(family["gate_logit"]).clamp(model.gate_prob_eps, 1.0 - model.gate_prob_eps)
    gate_bce = F.binary_cross_entropy_with_logits(
        family["gate_logit"],
        gate_t,
        pos_weight=pos_weight,
        reduction="mean",
    )
    gate_rate_loss = (gate_prob.mean() - history_01.new_tensor(gate_rate_target)).pow(2)

    zero = history_01.new_tensor(0.0)
    base_nll = zero
    if base_mask.any():
        base_logp = model.base_log_prob(
            target_u[base_mask],
            mu[base_mask],
            factor[base_mask],
            diag[base_mask],
            scale[base_mask],
            nu[base_mask],
        )
        base_nll = (-base_logp).mean()

    event_nll = zero
    family_ce = zero
    shape_loss = zero
    hard_top1 = zero
    if family_mask_t.any():
        pred_logits = family["family_logits"][family_mask_t]
        labels = hard_idx[family_mask_t]
        pred_shapes = family["family_shapes"][family_mask_t]
        chosen_pred = pred_shapes[torch.arange(pred_shapes.shape[0], device=pred_shapes.device), labels]
        tgt_shapes = family_shapes_target.to(history_01.device)[family_mask_t]
        chosen_tgt = tgt_shapes[torch.arange(tgt_shapes.shape[0], device=tgt_shapes.device), labels]

        event_logp = model.event_log_prob(
            target_u[family_mask_t],
            mu[family_mask_t],
            family["family_scale"][family_mask_t],
            chosen_pred,
        )
        event_nll = (-event_logp).mean()
        family_ce = F.cross_entropy(pred_logits, labels)
        shape_loss = (chosen_pred - chosen_tgt).abs().mean()
        hard_top1 = (pred_logits.argmax(dim=-1) == labels).float().mean()

    total = base_nll + gate_bce + gate_rate_loss + event_nll + family_ce + shape_loss
    metrics = {
        "base_nll": base_nll.detach(),
        "gate_bce": gate_bce.detach(),
        "gate_rate_loss": gate_rate_loss.detach(),
        "event_nll": event_nll.detach(),
        "family_ce": family_ce.detach(),
        "shape_loss": shape_loss.detach(),
        "gate_rate": gate_prob.mean().detach(),
        "hard_family_top1": hard_top1.detach(),
        "family_active_rate": family_mask.float().mean().detach(),
        "total_loss": total.detach(),
    }
    return total, metrics


@torch.no_grad()
def evaluate_h1(
    model: HardSeparatedH1LocalFamilyModel,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    gate_pos_weight: float,
    gate_rate_target: float,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "val_total_loss": 0.0,
        "val_base_nll": 0.0,
        "val_gate_bce": 0.0,
        "val_gate_rate_loss": 0.0,
        "val_event_nll": 0.0,
        "val_family_ce": 0.0,
        "val_shape_loss": 0.0,
        "val_hard_family_top1": 0.0,
        "val_gate_rate": 0.0,
        "val_coverage_90": 0.0,
        "val_mae": 0.0,
        "val_width_90": 0.0,
    }
    q95_cover_sum = 0.0
    q99_cover_sum = 0.0
    q95_count = 0
    q99_count = 0
    gt_delta_all = []
    sample_delta_all = []
    total_count = 0

    for history_01, target_01, gate_targets, family_mask, family_shapes, family_probs in loader:
        loss, metrics = hard_separated_objective(
            model,
            history_01,
            target_01,
            gate_targets,
            family_mask,
            family_shapes,
            family_probs,
            gate_pos_weight=gate_pos_weight,
            gate_rate_target=gate_rate_target,
        )
        samples = model.sample_next_iv(history_01, n_samples=eval_samples)
        q05 = samples.quantile(0.05, dim=1)
        q95 = samples.quantile(0.95, dim=1)
        mean_pred = samples.mean(dim=1)

        prev = history_01[:, -1].reshape(history_01.shape[0], -1)
        target_abs = (target_01 - prev).abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold
        q95_cov = ((target_01[q95_mask] >= q05[q95_mask]) & (target_01[q95_mask] <= q95[q95_mask])).float().mean() if q95_mask.any() else target_01.new_tensor(0.0)
        q99_cov = ((target_01[q99_mask] >= q05[q99_mask]) & (target_01[q99_mask] <= q95[q99_mask])).float().mean() if q99_mask.any() else target_01.new_tensor(0.0)

        batch_size = history_01.shape[0]
        totals["val_total_loss"] += float(metrics["total_loss"].item()) * batch_size
        totals["val_base_nll"] += float(metrics["base_nll"].item()) * batch_size
        totals["val_gate_bce"] += float(metrics["gate_bce"].item()) * batch_size
        totals["val_gate_rate_loss"] += float(metrics["gate_rate_loss"].item()) * batch_size
        totals["val_event_nll"] += float(metrics["event_nll"].item()) * batch_size
        totals["val_family_ce"] += float(metrics["family_ce"].item()) * batch_size
        totals["val_shape_loss"] += float(metrics["shape_loss"].item()) * batch_size
        totals["val_hard_family_top1"] += float(metrics["hard_family_top1"].item()) * batch_size
        totals["val_gate_rate"] += float(metrics["gate_rate"].item()) * batch_size
        totals["val_coverage_90"] += float(((target_01 >= q05) & (target_01 <= q95)).float().mean().item()) * batch_size
        totals["val_mae"] += float((mean_pred - target_01).abs().mean().item()) * batch_size
        totals["val_width_90"] += float((q95 - q05).mean().item()) * batch_size
        q95_cover_sum += float(q95_cov.item()) * int(q95_mask.sum().item())
        q99_cover_sum += float(q99_cov.item()) * int(q99_mask.sum().item())
        q95_count += int(q95_mask.sum().item())
        q99_count += int(q99_mask.sum().item())
        total_count += batch_size

        gt_delta_all.append((target_01 - prev).detach().cpu().numpy())
        sample_delta_all.append((samples - prev[:, None, :]).detach().cpu().numpy())

    out = {k: v / max(total_count, 1) for k, v in totals.items()}
    out["val_realized_q95_coverage_90"] = q95_cover_sum / max(q95_count, 1)
    out["val_realized_q99_coverage_90"] = q99_cover_sum / max(q99_count, 1)

    gt_delta = np.concatenate(gt_delta_all, axis=0)
    sample_delta = np.concatenate(sample_delta_all, axis=0)
    shape = compute_h1_shape_stats(gt_delta, sample_delta)
    out.update({f"val_h1_{k}": v for k, v in shape.items()})
    return out


def load_model(checkpoint_path: str, device: torch.device) -> tuple[HardSeparatedH1LocalFamilyModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_h1_hard_separated_local_family_student_t_208b":
        raise ValueError(f"Unexpected model type: {raw_config['type']}")
    model = HardSeparatedH1LocalFamilyModel(
        encoder_config=raw_config["encoder"],
        decoder_config=raw_config["decoder"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-4),
        event_noise_floor=raw_config.get("event_noise_floor", 0.002),
        event_noise_mult=raw_config.get("event_noise_mult", 0.35),
        event_shape_floor=raw_config.get("event_shape_floor", 0.05),
        event_nu=raw_config.get("event_nu", 4.0),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="208b H=1 hard-separated local-family Student-t")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_event", type=float, default=3e-3)
    parser.add_argument("--weight_decay_event", type=float, default=0.01)
    parser.add_argument("--enc_d_model", type=int, default=128)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_d_model", type=int, default=128)
    parser.add_argument("--dec_heads", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--teacher_checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--teacher_knn", type=int, default=32)
    parser.add_argument("--teacher_target_temp", type=float, default=0.15)
    parser.add_argument("--gate_pos_weight", type=float, default=3.0)
    parser.add_argument("--gate_rate_penalty", type=float, default=1.0)
    parser.add_argument("--n_family", type=int, default=3)
    parser.add_argument("--family_scale_floor", type=float, default=5e-4)
    parser.add_argument("--init_gate_prob", type=float, default=0.10)
    parser.add_argument("--init_family_scale", type=float, default=0.05)
    parser.add_argument("--family_shape_scale", type=float, default=0.05)
    parser.add_argument("--family_nu", type=float, default=4.0)
    parser.add_argument("--gate_temperature", type=float, default=0.5)
    parser.add_argument("--family_temperature", type=float, default=0.6)
    parser.add_argument("--gate_prob_eps", type=float, default=1e-4)
    parser.add_argument("--event_noise_floor", type=float, default=0.002)
    parser.add_argument("--event_noise_mult", type=float, default=0.35)
    parser.add_argument("--event_shape_floor", type=float, default=0.05)
    parser.add_argument("--event_nu", type=float, default=4.0)
    parser.add_argument("--val_samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[:4511], axis=0).reshape(-1))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

    hist_len = args.history_len
    test_start = 4511
    max_train_idx = test_start - hist_len - 30
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, hist_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, hist_len)

    train_targets, val_targets = build_teacher_guidance_targets(
        teacher_checkpoint=args.teacher_checkpoint,
        train_history=train_hist,
        train_target=train_target,
        val_history=val_hist,
        val_target=val_target,
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        batch_size=args.batch_size,
        device=device,
        family_size=args.n_family,
        knn=args.teacher_knn,
        target_temp=args.teacher_target_temp,
    )
    gate_rate_target = float(train_targets["gate_targets"].mean().item())

    train_loader = DataLoader(
        TensorDataset(
            train_hist,
            train_target,
            train_targets["gate_targets"].to(device),
            train_targets["family_mask"].to(device),
            train_targets["family_shapes"].to(device),
            train_targets["family_probs"].to(device),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            val_hist,
            val_target,
            val_targets["gate_targets"].to(device),
            val_targets["family_mask"].to(device),
            val_targets["family_shapes"].to(device),
            val_targets["family_probs"].to(device),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    family_val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    encoder_config = dict(
        input_dim=25,
        d_model=args.enc_d_model,
        n_heads=args.enc_heads,
        n_layers=args.enc_layers,
        dropout=args.enc_dropout,
        bottleneck_dim=128,
        max_len=max(hist_len, 64),
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.dec_d_model,
        n_heads=args.dec_heads,
        n_layers=args.dec_layers,
        cond_dim=128,
        rank=args.rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
        n_family=args.n_family,
        family_scale_floor=args.family_scale_floor,
        init_gate_prob=args.init_gate_prob,
        init_family_scale=args.init_family_scale,
        family_shape_scale=args.family_shape_scale,
        family_nu=args.family_nu,
        gate_temperature=args.gate_temperature,
        family_temperature=args.family_temperature,
        gate_prob_eps=args.gate_prob_eps,
    )

    model = HardSeparatedH1LocalFamilyModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        event_noise_floor=args.event_noise_floor,
        event_noise_mult=args.event_noise_mult,
        event_shape_floor=args.event_shape_floor,
        event_nu=args.event_nu,
    ).to(device)
    warm_start_from_201b(model, args.base_checkpoint)
    freeze_base_path(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr_event, weight_decay=args.weight_decay_event)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n{'=' * 76}")
    print("208b: H=1 hard-separated local-family Student-t")
    print(f"{'=' * 76}")
    print(f"  Encoder params:   {n_enc:,}")
    print(f"  Decoder params:   {n_dec:,}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Thresholds: q95={q95_threshold:.5f}, q99={q99_threshold:.5f}")
    print(f"  Base warm start: {args.base_checkpoint}")
    print(f"  Gate-rate target: {gate_rate_target:.4f}")

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "train_total_loss": 0.0,
            "train_base_nll": 0.0,
            "train_gate_bce": 0.0,
            "train_gate_rate_loss": 0.0,
            "train_event_nll": 0.0,
            "train_family_ce": 0.0,
            "train_shape_loss": 0.0,
            "train_hard_family_top1": 0.0,
            "train_gate_rate": 0.0,
        }
        nb = 0

        for history_01, target_01, gate_targets, family_mask, family_shapes, family_probs in train_loader:
            optimizer.zero_grad()
            loss, metrics = hard_separated_objective(
                model,
                history_01,
                target_01,
                gate_targets,
                family_mask,
                family_shapes,
                family_probs,
                gate_pos_weight=args.gate_pos_weight,
                gate_rate_target=gate_rate_target,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            for k in ep:
                ep[k] += float(metrics[k.replace("train_", "")].item())
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in ep.items()}
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            gate_pos_weight=args.gate_pos_weight,
            gate_rate_target=gate_rate_target,
            eval_samples=args.val_samples,
        )
        family_metrics = evaluate_family_statistics(model, family_val_loader)
        val_metrics.update(family_metrics)
        selection_score = val_metrics["val_total_loss"]
        val_metrics["selection_score"] = selection_score

        elapsed = time.time() - t0
        row = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_sec": elapsed}
        history.append(make_serializable(row))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "type": "transformer_h1_hard_separated_local_family_student_t_208b",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
                "event_noise_floor": args.event_noise_floor,
                "event_noise_mult": args.event_noise_mult,
                "event_shape_floor": args.event_shape_floor,
                "event_nu": args.event_nu,
            },
            "metrics": make_serializable(row),
        }
        torch.save(ckpt, Path(args.output_dir) / "final_model.pt")

        if selection_score < best_score:
            best_score = selection_score
            best_metrics = row
            torch.save(ckpt, Path(args.output_dir) / "best_model.pt")
            best_flag = "  *best"
        else:
            best_flag = ""

        print(
            f"Ep {epoch:>3d}  "
            f"train_total={train_metrics['train_total_loss']:.4f}  "
            f"val_total={val_metrics['val_total_loss']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.4f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.4f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"gate={val_metrics['val_gate_rate']:.3f}  "
            f"hard_top1={val_metrics['val_hard_family_top1']:.3f}  "
            f"fam_top1={val_metrics['val_family_top1_mean']:.3f}  "
            f"({elapsed:.1f}s){best_flag}"
        )

        with open(Path(args.output_dir) / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    if best_metrics is not None:
        print("\nBest metrics:")
        for k, v in best_metrics.items():
            if k == "epoch":
                print(f"  {k}: {v}")
            elif isinstance(v, (float, int)):
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
