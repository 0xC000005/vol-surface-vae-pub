#!/usr/bin/env python
"""
241f: 241b multi-CRPS with encoder/decoder/flow/prior UNFROZEN (advisor ablation).

Tests advisor's alternative hypothesis: "path_context is detached → the scalable DOF
is frozen out of scope. A 2% CRPS drop may reflect gradient reaches path_transport
but encoder/prior parameters can't move to minimise CRPS."

If CRPS terms drop MORE than 241b's ≤6% → the 241b plateau was NOT interior capacity
limit; it was DOF access. Continue multi-CRPS with unfreeze.
If CRPS terms still plateau at ≤6% → interior capacity genuinely limited, go Stage 4.

Only difference from 241b: encoder/decoder/flow/prior are TRAINABLE (with small LR).
Same loss, same K, same ODE steps.

Also: multi_crps_loss drops `@torch.no_grad` around forward_from_history so grad flows
to encoder/decoder params during CRPS backward. This is the KEY change.

Original 241b multi-CRPS description:
Attacks three failing suites in ONE training run via a single sampled ensemble:
  - mean_reversion           → afCRPS @ h=30 (terminal vector)
  - pathwise_jump_realism    → afCRPS on max_t|Δsamples| (pathwise max functional)
  - time_series / kurtosis   → twCRPS on per-cell changes (tail-weighted)
Guard term:
  - cross-cell corr          → Energy Score on 25-dim terminal vector

L = L_183c_fm + L_183c_controls
  + λ_term  · afCRPS(h30)
  + λ_pmax  · afCRPS(max_t|Δ|)
  + λ_tail  · twCRPS(cell changes, β)
  + λ_es    · ES_25d(h30)

Sampling during training:
  - Teacher pathway (forward_from_history, covariance_parts, Cholesky, block assignments,
    shared_local_delta) stays inside no_grad — encoder/decoder/prior frozen.
  - sample_basis_paths runs grad-enabled with a partial ODE unroll (train_ode_steps)
    and optional gradient checkpointing per step.
  - Only path_transport, path_context_adapter, width_allocator, band_tail, local/band
    state gates, and local/band metric budgets receive CRPS gradient.

Bitter Lesson check: cell_median (0) and cell_iqr (per-cell std of changes) are a
one-time scaling statistic computed from training data, equivalent to standard
normalisation — no lookup tables, no regime-specific constants.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from diffusion.block_ar.single_pass_ar import afcrps_loss, energy_score
from experiments.backfill.block_ar.analyze_170d_mechanisms import make_serializable
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import evaluate_joint_subset
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    checkpoint_key,
    evaluate_frontier_subset,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel,
    evaluate_teacher_forced,
)


def warm_start_with_gates(model: StateMetricTransportModel, ckpt_path: str, device):
    """Warm-start that includes local_state_gate / band_state_gate (183c's own skips them)."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = dict(payload["model_state_dict"])
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  Warm start (with gates) loaded from {ckpt_path}")
    print(f"  Warm start missing: {len(missing)} | unexpected: {len(unexpected)} | shape-skipped: {len(skipped)}")


def sample_basis_paths_with_grad(
    model: StateMetricTransportModel,
    path_context: torch.Tensor,
    n_samples: int,
    n_steps: int,
    use_checkpoint: bool,
) -> torch.Tensor:
    """Grad-enabled ODE unroll that mirrors model.sample_basis_paths (from 183a) body.

    Prior sample is no-grad (non-learnable); gradient flows through transport_velocity
    parameters (path_transport, state gates, metric budgets, width_allocator, band_tail).
    """
    batch = path_context.shape[0]
    dtype = path_context.dtype
    device = path_context.device
    with torch.no_grad():
        z, _prior_stats = model.prior.sample(batch * n_samples, device=device, dtype=dtype)
    ctx = path_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
    dt = 1.0 / float(n_steps)
    for step in range(n_steps):
        t = torch.full((batch * n_samples,), fill_value=step * dt, device=device, dtype=dtype)
        if use_checkpoint and z.requires_grad:
            def _vel(z_in):
                v, _metrics = model.transport_velocity(z_in, t, ctx)
                return v
            v = checkpoint(_vel, z, use_reentrant=False)
        else:
            v, _metrics = model.transport_velocity(z, t, ctx)
        z = z + dt * v
    return z


def sample_future_u_with_grad(
    model: StateMetricTransportModel,
    history_01: torch.Tensor,
    n_samples: int,
    train_ode_steps: int,
    use_checkpoint: bool,
    grad_through_teacher: bool = True,
) -> torch.Tensor:
    """Mirrors sample_future_u() body; optionally grad flows through forward_from_history.

    241f: grad_through_teacher=True lets CRPS gradient reach encoder+decoder+prior
    parameters (advisor's proposed DOF-access fix). Block sampling (torch.multinomial)
    is always no_grad — it's discrete and not trainable.

    Returns IV samples in [support_lo, support_hi] range, shape (B, K, T, H, W).
    """
    # Teacher + covariance + path_context pathway.
    teacher_ctx = torch.enable_grad() if grad_through_teacher else torch.no_grad()
    with teacher_ctx:
        (
            mu, time_factor, time_diag, cell_factor, cell_diag, scale,
            flow_context, base_local_delta, block_logits,
        ) = model.forward_from_history(history_01)
        batch, n_frames, n_cells = mu.shape
        cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        chol_t = torch.linalg.cholesky(cov_t)
        chol_c = torch.linalg.cholesky(cov_c)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        shared_local_delta = model.decoder.build_shared_local_delta(base_local_delta)
        local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)

    # Block-template sampling is discrete; always no_grad.
    with torch.no_grad():
        block_probs = F.softmax(block_logits.detach(), dim=-1)
        sampled_blocks = []
        for b in range(model.decoder.n_blocks):
            sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
            sampled_blocks.append(sampled)
        sampled_assign = torch.stack(sampled_blocks, dim=-1)
        assign_flat = sampled_assign.reshape(batch * n_samples, model.decoder.n_blocks)
    # build_template_factors uses decoder params — keep grad so decoder receives signal
    # through the routing factors (teacher-forced template activation).
    sampled_factors, _logdet, _offdiag = model.decoder.build_template_factors(assign_flat)

    # Grad-enabled ODE unroll on basis paths.
    basis_paths = sample_basis_paths_with_grad(
        model, path_context, n_samples, train_ode_steps, use_checkpoint,
    )
    basis_paths = basis_paths.view(batch, n_samples, n_frames, n_cells)

    base_white_flat = model.path_geometry.from_basis(
        basis_paths.reshape(batch * n_samples, n_frames, n_cells)
    ).reshape(batch * n_samples, n_frames * n_cells)
    base_white = base_white_flat.view(
        batch * n_samples, model.decoder.n_blocks, model.decoder.block_len, n_cells,
    )
    lhs = base_white.permute(0, 1, 3, 2).reshape(
        batch * n_samples * model.decoder.n_blocks, n_cells, model.decoder.block_len,
    )
    factor_flat = sampled_factors.reshape(
        batch * n_samples * model.decoder.n_blocks, n_cells, n_cells,
    )
    routed_white = torch.matmul(factor_flat, lhs)
    routed_white = routed_white.reshape(
        batch * n_samples, model.decoder.n_blocks, n_cells, model.decoder.block_len,
    ).permute(0, 1, 3, 2)
    routed_white = routed_white.reshape(batch, n_samples, n_frames, n_cells)

    temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
    noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
    u_samples = mu.unsqueeze(1) + noise * local_scale  # (B, K, n_frames, n_cells)
    iv_samples = unconstrained_to_iv(u_samples, lo=model.support_lo, hi=model.support_hi)
    iv_samples = iv_samples.view(batch, n_samples, n_frames, 5, 5)
    return iv_samples


def compute_cell_change_stats(train_loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One-time pass over training data to get per-cell change-distribution stats.

    Returns (cell_median, cell_iqr) each of shape (T-1, H, W) for twCRPS weighting.
    Since the twCRPS weight function is w(y) = 1 + β((y-μ)/σ)², and the weight depends
    only on the SHAPE of the tail, flattening over T is acceptable. We report per-cell
    (H×W) stats.
    """
    all_changes = []
    for history_01, future_01 in train_loader:
        future_01 = future_01.to(device)
        ch = future_01[:, 1:] - future_01[:, :-1]  # (B, T-1, H, W)
        all_changes.append(ch.reshape(-1, 5, 5).cpu())
    flat = torch.cat(all_changes, dim=0)  # (N, 5, 5)
    # Median = 0 by construction for diff distributions with approximate symmetry —
    # use actual median for robustness; IQR via quantile.
    q = torch.quantile(flat, torch.tensor([0.25, 0.5, 0.75]), dim=0)  # (3, 5, 5)
    cell_median = q[1].to(device)
    cell_iqr = (q[2] - q[0]).clamp_min(1e-6).to(device)
    return cell_median, cell_iqr


def multi_crps_loss(
    model: StateMetricTransportModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    lambdas: dict,
    K: int,
    train_ode_steps: int,
    afcrps_alpha: float,
    tail_beta: float,
    cell_median: torch.Tensor,
    cell_iqr: torch.Tensor,
    apply_crps: bool,
    use_checkpoint: bool,
    ref_pred_mean_batch: torch.Tensor | None = None,
    lambda_anchor: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full Stage-2 loss = 183c FM+controls + multi-functional proper scoring rules."""
    # Part 1: 183c FM + controls (teacher-forced, same as 183c).
    target_u = iv_to_unconstrained(
        future_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps,
    )
    with torch.no_grad():
        (
            mu, time_factor, time_diag, cell_factor, cell_diag, scale,
            flow_context, base_local_delta, block_logits,
        ) = model.forward_from_history(history_01)
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u, mu=mu, time_factor=time_factor, time_diag=time_diag,
            cell_factor=cell_factor, cell_diag=cell_diag, scale=scale,
            base_local_delta=base_local_delta, block_logits=block_logits,
        ).view(history_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, target_band_log = compute_control_targets(model, target_basis)

    target_basis_flat = target_basis.reshape(target_basis.shape[0], -1)
    z0, prior_stats = model.prior.sample(
        target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype,
    )
    t = torch.rand(target_basis.shape[0], device=target_basis.device, dtype=target_basis.dtype)
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis_flat
    target_v = target_basis_flat - z0

    pred_v, control_metrics = model.transport_velocity(z_t, t, path_context)
    fm_loss = F.mse_loss(pred_v, target_v)

    raw_local, raw_band, metric_local, metric_band, _lg, _bg = model.build_state_metric_controls(
        z_t, t, path_context,
    )
    local_loss = F.smooth_l1_loss(metric_local, target_local_log)
    band_loss = F.smooth_l1_loss(metric_band, target_band_log)
    surf = metric_local.view(metric_local.shape[0], metric_local.shape[1], 5, 5)
    time_smooth = (metric_local[:, 1:] - metric_local[:, :-1]).abs().mean()
    row_smooth = (surf[:, :, 1:] - surf[:, :, :-1]).abs().mean()
    col_smooth = (surf[:, :, :, 1:] - surf[:, :, :, :-1]).abs().mean()
    smooth_reg = time_smooth + row_smooth + col_smooth
    budget_reg = (
        (model.local_metric_budget() - model.metric_config["init_local_metric_budget"]).pow(2)
        + (model.band_metric_budget() - model.metric_config["init_band_metric_budget"]).pow(2)
    )

    total = (
        fm_loss
        + model.integrated_config["local_loss_weight"] * local_loss
        + model.integrated_config["band_loss_weight"] * band_loss
        + model.integrated_config["smooth_reg_weight"] * smooth_reg
        + model.metric_config["budget_reg_weight"] * budget_reg
    )

    metrics: dict[str, torch.Tensor] = {
        "flow_match_loss": fm_loss.detach(),
        "sdr_local_loss": local_loss.detach(),
        "sdr_band_loss": band_loss.detach(),
        "sdr_smooth_reg": smooth_reg.detach(),
        "sdr_budget_reg": budget_reg.detach(),
        "target_basis_std_mean": target_basis_flat.std(dim=-1).mean(),
        "target_local_abs_mean": target_local_log.abs().mean(),
        "target_band_high_mean": target_band_log[:, 2].mean(),
        "prior_std_mean": prior_stats["prior_std_mean"],
        "prior_jump_prob_mean": prior_stats["prior_jump_prob_mean"],
        "prior_jump_scale_mean": prior_stats["prior_jump_scale_mean"],
        "prior_active_rate": prior_stats["prior_active_rate"],
        **control_metrics,
    }

    if apply_crps:
        # Part 2: sample-with-grad ensemble and apply 4 proper-scoring-rule terms.
        iv_samples = sample_future_u_with_grad(
            model, history_01, n_samples=K, train_ode_steps=train_ode_steps,
            use_checkpoint=use_checkpoint,
        )  # (B, K, T, 5, 5)
        # future_01 is (B, T, 25); reshape to grid form for twCRPS weighting.
        iv_gt = future_01.view(future_01.shape[0], future_01.shape[1], 5, 5)  # (B, T, 5, 5)

        # afCRPS on terminal (h=30) — primary driver of mean_reversion.
        terminal_samples = iv_samples[:, :, -1:, :, :]  # (B, K, 1, 5, 5)
        terminal_gt = iv_gt[:, -1:, :, :]  # (B, 1, 5, 5)
        afcrps_h30, _, _ = afcrps_loss(
            terminal_samples, terminal_gt, alpha=afcrps_alpha, reduction="frame_sum",
        )

        # afCRPS on pathwise max|Δ| — primary driver of pathwise_jump_realism.
        iv_s_change = iv_samples[:, :, 1:] - iv_samples[:, :, :-1]  # (B, K, T-1, 5, 5)
        iv_g_change = iv_gt[:, 1:] - iv_gt[:, :-1]  # (B, T-1, 5, 5)
        max_samples = iv_s_change.abs().amax(dim=2, keepdim=True)  # (B, K, 1, 5, 5)
        max_gt = iv_g_change.abs().amax(dim=1, keepdim=True)  # (B, 1, 5, 5)
        pmax_crps, _, _ = afcrps_loss(
            max_samples, max_gt, alpha=afcrps_alpha, reduction="frame_sum",
        )

        # twCRPS tail-weighted on per-cell changes — primary driver of kurtosis.
        # frame_sum produces a T-1-scaled term; normalise by (T-1) so the
        # magnitude is comparable to afcrps_h30 / pmax (which both have T=1).
        tail_crps_raw, _, _ = afcrps_loss(
            iv_s_change, iv_g_change, alpha=afcrps_alpha, reduction="frame_sum",
            cell_median=cell_median, cell_iqr=cell_iqr, twcrps_beta=tail_beta,
        )
        n_change_frames = max(iv_s_change.shape[2], 1)
        tail_crps = tail_crps_raw / float(n_change_frames)

        # ES on 25-dim terminal vector — cross-cell structure guard (240a lesson).
        es_h30 = energy_score(terminal_samples, terminal_gt)

        total = (
            total
            + lambdas["term"] * afcrps_h30
            + lambdas["pmax"] * pmax_crps
            + lambdas["tail"] * tail_crps
            + lambdas["es"] * es_h30
        )

        metrics["afcrps_h30"] = afcrps_h30.detach()
        metrics["twcrps_pmax"] = pmax_crps.detach()
        metrics["twcrps_tail"] = tail_crps.detach()
        metrics["es_h30"] = es_h30.detach()
        metrics["sample_std_h30_mean"] = iv_samples[:, :, -1].std(dim=1).mean().detach()

        # 241f: KS-preservation anchor. Uses a PRECOMPUTED per-window ref_pred_mean
        # (pre-cached at training start using ref_model over many K samples) so the
        # anchor target is DETERMINISTIC and noise-free. Pulled from a cache indexed
        # by window_indices (optionally passed via kwarg).
        if ref_pred_mean_batch is not None and lambda_anchor > 0:
            pred_mean = iv_samples.mean(dim=1)  # (B, T, 5, 5) — grad flows
            anchor_loss = (pred_mean - ref_pred_mean_batch).abs().mean()
            total = total + lambda_anchor * anchor_loss
            metrics["anchor_loss"] = anchor_loss.detach()

    metrics["sdr_total_loss"] = total.detach()
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="241b: 183c + multi-CRPS fine-tune")
    # Core training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr_ctrl", type=float, default=2.5e-4)
    parser.add_argument("--lr_path", type=float, default=1.0e-4)
    parser.add_argument("--lr_backbone", type=float, default=3.0e-5,
                        help="241f: LR for UNFROZEN encoder/decoder/flow/prior. Small "
                             "because those were trained end-to-end before and we don't "
                             "want to destroy their learned structure.")
    parser.add_argument("--lambda_anchor", type=float, default=1.0,
                        help="241f: weight on |pred_mean - ref_183c_pred_mean|.mean(). "
                             "Pulls pred_mean toward 183c's well-calibrated marginals. "
                             "Default 1.0; set to 0 to disable (recovers 241c).")
    parser.add_argument("--K_ref", type=int, default=128,
                        help="241f: K samples used to precompute each window's ref_pred_mean. "
                             "128 gives noise ~1/sqrt(128) ≈ 0.09. Higher is tighter but "
                             "slower at startup (one-time cost).")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    # Multi-CRPS
    parser.add_argument("--lambda_term", type=float, default=0.10,
                        help="afCRPS@h30 weight (mean_reversion).")
    parser.add_argument("--lambda_pmax", type=float, default=0.05,
                        help="afCRPS@max_t|Δ| weight (pathwise_jump_realism).")
    parser.add_argument("--lambda_tail", type=float, default=0.03,
                        help="twCRPS tail-weighted per-cell-change weight (kurtosis).")
    parser.add_argument("--lambda_es", type=float, default=0.05,
                        help="ES@25d terminal (cross-cell structure guard).")
    parser.add_argument("--afcrps_alpha", type=float, default=0.95)
    parser.add_argument("--tail_beta", type=float, default=2.0,
                        help="twCRPS tail weight exponent. w(y)=1+β((y-med)/IQR)².")
    parser.add_argument("--K", type=int, default=32, help="Ensemble members per window.")
    parser.add_argument("--train_ode_steps", type=int, default=4,
                        help="ODE unroll steps at training (inference uses path_ode_steps).")
    parser.add_argument("--ensemble_every_n", type=int, default=1,
                        help="Apply CRPS every N batches (1=every batch, 5=every 5th).")
    parser.add_argument("--grad_ckpt", action="store_true", default=True)
    parser.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    # Model hyperparameters (must match 183c checkpoint config)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_low_scale_clip", type=float, default=1.2)
    parser.add_argument("--flow_mid_scale_clip", type=float, default=0.7)
    parser.add_argument("--flow_high_scale_clip", type=float, default=0.35)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--n_blocks", type=int, default=5)
    parser.add_argument("--n_templates", type=int, default=3)
    parser.add_argument("--mix_chunk_size", type=int, default=27)
    parser.add_argument("--template_diag_clip", type=float, default=0.30)
    parser.add_argument("--template_offdiag_clip", type=float, default=0.18)
    parser.add_argument("--drift_strength_max", type=float, default=0.75)
    parser.add_argument("--equilibrium_offset_clip", type=float, default=0.20)
    parser.add_argument("--init_drift_strength", type=float, default=0.20)
    parser.add_argument("--path_context_dim", type=int, default=256)
    parser.add_argument("--path_context_hidden_dim", type=int, default=256)
    parser.add_argument("--path_d_model", type=int, default=192)
    parser.add_argument("--path_heads", type=int, default=4)
    parser.add_argument("--path_layers", type=int, default=4)
    parser.add_argument("--path_ff_mult", type=int, default=4)
    parser.add_argument("--path_time_embed_dim", type=int, default=64)
    parser.add_argument("--path_ode_steps", type=int, default=8)
    parser.add_argument("--prior_low_std", type=float, default=1.00)
    parser.add_argument("--prior_mid_std", type=float, default=0.75)
    parser.add_argument("--prior_high_std", type=float, default=0.35)
    parser.add_argument("--prior_low_jump_prob", type=float, default=0.005)
    parser.add_argument("--prior_mid_jump_prob", type=float, default=0.025)
    parser.add_argument("--prior_high_jump_prob", type=float, default=0.070)
    parser.add_argument("--prior_low_jump_scale", type=float, default=0.10)
    parser.add_argument("--prior_mid_jump_scale", type=float, default=0.30)
    parser.add_argument("--prior_high_jump_scale", type=float, default=0.75)
    parser.add_argument("--width_rank", type=int, default=4)
    parser.add_argument("--width_hidden_dim", type=int, default=256)
    parser.add_argument("--width_clip", type=float, default=0.80)
    parser.add_argument("--band_hidden_dim", type=int, default=128)
    parser.add_argument("--band_clip", type=float, default=0.45)
    parser.add_argument("--local_loss_weight", type=float, default=0.75)
    parser.add_argument("--band_loss_weight", type=float, default=0.50)
    parser.add_argument("--smooth_reg_weight", type=float, default=0.04)
    parser.add_argument("--state_gate_hidden_dim", type=int, default=128)
    parser.add_argument("--init_local_state_gate", type=float, default=0.30)
    parser.add_argument("--init_band_state_gate", type=float, default=0.25)
    parser.add_argument("--local_metric_min", type=float, default=0.12)
    parser.add_argument("--local_metric_max", type=float, default=0.55)
    parser.add_argument("--band_metric_min", type=float, default=0.08)
    parser.add_argument("--band_metric_max", type=float, default=0.35)
    parser.add_argument("--init_local_metric_budget", type=float, default=0.24)
    parser.add_argument("--init_band_metric_budget", type=float, default=0.14)
    parser.add_argument("--local_metric_clip", type=float, default=0.60)
    parser.add_argument("--band_metric_clip", type=float, default=0.35)
    parser.add_argument("--budget_reg_weight", type=float, default=0.02)
    # Infra
    parser.add_argument("--warm_start_path", type=str, default=None)
    parser.add_argument("--warmstart_include_gates", action="store_true", default=True)
    parser.add_argument("--no_warmstart_include_gates", dest="warmstart_include_gates", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Disabled by default: bf16 breaks Cholesky in teacher_basis_flat.")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    surf_tensor = torch.from_numpy(surfaces).to(args.device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, args.history_len, args.future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, args.history_len, args.future_len)
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    print("Computing per-cell change statistics (one-time) for twCRPS weighting …")
    cell_median, cell_iqr = compute_cell_change_stats(train_loader, args.device)
    print(f"  cell_median range: [{cell_median.min().item():.5f}, {cell_median.max().item():.5f}]")
    print(f"  cell_iqr    range: [{cell_iqr.min().item():.5f}, {cell_iqr.max().item():.5f}]")

    encoder_config = EncoderConfig(input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0)
    decoder_config = dict(
        n_frames=args.future_len, n_cells=25, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        cond_dim=128, time_rank=args.time_rank, cell_rank=args.cell_rank,
        diag_floor=args.diag_floor, scale_floor=args.scale_floor, init_diag=args.init_diag, init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim, local_delta_clip=args.local_delta_clip,
        n_blocks=args.n_blocks, n_templates=args.n_templates,
        template_diag_clip=args.template_diag_clip, template_offdiag_clip=args.template_offdiag_clip,
        drift_strength_max=args.drift_strength_max, equilibrium_offset_clip=args.equilibrium_offset_clip,
        init_drift_strength=args.init_drift_strength,
    )
    flow_config = dict(
        dim=args.future_len * 25, context_dim=args.flow_context_dim, n_frames=args.future_len,
        grid_h=5, grid_w=5, hidden_dim=args.flow_hidden_dim, n_layers=args.flow_layers,
        low_scale_clip=args.flow_low_scale_clip, mid_scale_clip=args.flow_mid_scale_clip,
        high_scale_clip=args.flow_high_scale_clip,
    )
    path_config = dict(
        context_dim=args.path_context_dim, context_hidden_dim=args.path_context_hidden_dim,
        d_model=args.path_d_model, n_heads=args.path_heads, n_layers=args.path_layers,
        ff_mult=args.path_ff_mult, time_embed_dim=args.path_time_embed_dim, n_ode_steps=args.path_ode_steps,
    )
    prior_config = dict(
        low_std=args.prior_low_std, mid_std=args.prior_mid_std, high_std=args.prior_high_std,
        low_jump_prob=args.prior_low_jump_prob, mid_jump_prob=args.prior_mid_jump_prob, high_jump_prob=args.prior_high_jump_prob,
        low_jump_scale=args.prior_low_jump_scale, mid_jump_scale=args.prior_mid_jump_scale, high_jump_scale=args.prior_high_jump_scale,
    )
    integrated_config = dict(
        width_rank=args.width_rank, width_hidden_dim=args.width_hidden_dim, width_clip=args.width_clip,
        band_hidden_dim=args.band_hidden_dim, band_clip=args.band_clip,
        local_loss_weight=args.local_loss_weight, band_loss_weight=args.band_loss_weight,
        smooth_reg_weight=args.smooth_reg_weight,
        init_local_strength=0.05, init_band_strength=0.05,
    )
    state_config = dict(
        gate_hidden_dim=args.state_gate_hidden_dim,
        init_local_state_gate=args.init_local_state_gate,
        init_band_state_gate=args.init_band_state_gate,
    )
    metric_config = dict(
        local_metric_min=args.local_metric_min, local_metric_max=args.local_metric_max,
        band_metric_min=args.band_metric_min, band_metric_max=args.band_metric_max,
        init_local_metric_budget=args.init_local_metric_budget, init_band_metric_budget=args.init_band_metric_budget,
        local_metric_clip=args.local_metric_clip, band_metric_clip=args.band_metric_clip,
        budget_reg_weight=args.budget_reg_weight,
    )

    model = StateMetricTransportModel(
        encoder_config=encoder_config, decoder_config=decoder_config, flow_config=flow_config,
        path_config=path_config, prior_config=prior_config, integrated_config=integrated_config,
        state_config=state_config, metric_config=metric_config,
        support_lo=args.support_lo, support_hi=args.support_hi, support_eps=args.support_eps,
        cov_jitter=args.cov_jitter, base_nu=args.base_nu, mix_chunk_size=args.mix_chunk_size,
    ).to(args.device)

    if args.warm_start_path:
        if args.warmstart_include_gates:
            warm_start_with_gates(model, args.warm_start_path, args.device)
        else:
            model.maybe_load_warm_start(args.warm_start_path, args.device)

    # 241f: Build a FROZEN reference 183c and PRECOMPUTE per-window pred_mean ONCE
    # with large K_ref=128 (noise ~1/sqrt(128) ≈ 0.09, 3x tighter than K=32). The
    # cached tensor is used as a DETERMINISTIC anchor target during training — avoids
    # stochastic moving-target bug that diverged the earlier 241e attempt.
    train_ref_means = None
    if args.lambda_anchor > 0 and args.warm_start_path is not None:
        print(f"Building frozen reference 183c (λ_anchor={args.lambda_anchor}, K_ref={args.K_ref})...")
        ref_model = StateMetricTransportModel(
            encoder_config=encoder_config, decoder_config=decoder_config, flow_config=flow_config,
            path_config=path_config, prior_config=prior_config, integrated_config=integrated_config,
            state_config=state_config, metric_config=metric_config,
            support_lo=args.support_lo, support_hi=args.support_hi, support_eps=args.support_eps,
            cov_jitter=args.cov_jitter, base_nu=args.base_nu, mix_chunk_size=args.mix_chunk_size,
        ).to(args.device)
        warm_start_with_gates(ref_model, args.warm_start_path, args.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print(f"Precomputing per-window 183c pred_mean over {len(train_indices)} train windows…")
        all_means = []
        ref_batch = 16
        with torch.no_grad():
            for i in range(0, len(train_hist), ref_batch):
                hist = train_hist[i : i + ref_batch].to(args.device)
                samp = ref_model.sample_batched(hist, n_samples=args.K_ref)
                if samp.dim() == 4:
                    samp = samp.view(samp.shape[0], samp.shape[1], samp.shape[2], 5, 5)
                mean = samp.mean(dim=1).cpu()  # (b, T, 5, 5)
                all_means.append(mean)
        train_ref_means = torch.cat(all_means, dim=0).contiguous()  # (N_train, T, 5, 5)
        print(f"  Cached train_ref_means: shape={tuple(train_ref_means.shape)}, "
              f"abs_mean={train_ref_means.abs().mean():.4f}")
        # Free ref_model — no longer needed during training.
        del ref_model
        torch.cuda.empty_cache()
        # Rebuild train_loader to include ref_means as third tensor in dataset.
        train_loader = DataLoader(
            TensorDataset(train_hist, train_future, train_ref_means),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
        )

    # 241f: PARTIAL unfreeze — keep encoder+prior FROZEN (they shape the marginal
    # distributions that 241c collapsed on level_KS/change_KS). Unfreeze decoder+flow
    # (routing/covariance heads) plus the 241b-trainable pieces.
    model.encoder.requires_grad_(False)
    model.prior.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.flow.requires_grad_(True)
    model.path_transport.requires_grad_(True)
    model.path_context_adapter.requires_grad_(True)
    model.width_allocator.requires_grad_(True)
    model.band_tail.requires_grad_(True)
    model.local_state_gate.requires_grad_(True)
    model.band_state_gate.requires_grad_(True)
    ctrl_params = (
        list(model.width_allocator.parameters())
        + list(model.band_tail.parameters())
        + list(model.local_state_gate.parameters())
        + list(model.band_state_gate.parameters())
        + [model.local_metric_budget_logit, model.band_metric_budget_logit]
    )
    path_params = list(model.path_transport.parameters()) + list(model.path_context_adapter.parameters())
    # 241f: partial unfreeze — only decoder+flow in the "backbone" group.
    backbone_params = (
        list(model.decoder.parameters())
        + list(model.flow.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": ctrl_params,     "lr": args.lr_ctrl},
            {"params": path_params,     "lr": args.lr_path},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print(f"\n{'=' * 72}\n241b: 183c + Multi-CRPS Fine-Tune\n{'=' * 72}")
    print(f"  Epochs: {args.epochs} | B={args.batch_size} ga={args.grad_accum} (effective {args.batch_size*args.grad_accum})")
    print(f"  λ_term={args.lambda_term}  λ_pmax={args.lambda_pmax}  λ_tail={args.lambda_tail}  λ_es={args.lambda_es}")
    print(f"  K={args.K}  train_ode_steps={args.train_ode_steps}  afcrps_α={args.afcrps_alpha}  tail_β={args.tail_beta}")
    print(f"  ensemble_every_n={args.ensemble_every_n}  grad_ckpt={args.grad_ckpt}  bf16={args.bf16}")
    print(f"  Warm start (with gates): {args.warm_start_path}")

    history_path = Path(args.output_dir) / "training_history.json"
    best_key = None
    best_metrics = None
    history = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )

    lambdas = {
        "term": args.lambda_term,
        "pmax": args.lambda_pmax,
        "tail": args.lambda_tail,
        "es": args.lambda_es,
    }

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_totals: dict[str, float] = {}
        nb = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            if train_ref_means is not None:
                history_01, future_01, ref_mean_batch = batch
                ref_mean_batch = ref_mean_batch.to(args.device, non_blocking=True)
            else:
                history_01, future_01 = batch
                ref_mean_batch = None
            history_01 = history_01.to(args.device, non_blocking=True)
            future_01 = future_01.to(args.device, non_blocking=True)
            apply_crps = (batch_idx % args.ensemble_every_n == 0)
            with autocast_ctx:
                loss, metrics = multi_crps_loss(
                    model, history_01, future_01,
                    lambdas=lambdas, K=args.K, train_ode_steps=args.train_ode_steps,
                    afcrps_alpha=args.afcrps_alpha, tail_beta=args.tail_beta,
                    cell_median=cell_median, cell_iqr=cell_iqr,
                    apply_crps=apply_crps, use_checkpoint=args.grad_ckpt,
                    ref_pred_mean_batch=ref_mean_batch, lambda_anchor=args.lambda_anchor,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {batch_idx}")
            (loss / args.grad_accum).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            for key, value in metrics.items():
                v = value.item() if torch.is_tensor(value) else float(value)
                ep_totals[key] = ep_totals.get(key, 0.0) + v
            nb += 1
            global_step += 1

            if global_step % args.log_every_n_steps == 0:
                fm = metrics["flow_match_loss"].item()
                msg = f"  [ep {epoch} step {global_step}] fm={fm:.4f}"
                if apply_crps:
                    afc = metrics["afcrps_h30"].item()
                    pmax = metrics["twcrps_pmax"].item()
                    tailc = metrics["twcrps_tail"].item()
                    esv = metrics["es_h30"].item()
                    msg += f" afcrps={afc:.3f} pmax={pmax:.3f} tail={tailc:.3f} es={esv:.3f}"
                msg += f" total={loss.item():.4f}"
                print(msg)

        scheduler.step()
        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in ep_totals.items()}
        val_metrics = evaluate_teacher_forced(model, val_loader)
        joint_metrics = evaluate_joint_subset(
            model, val_loader,
            joint_val_samples=args.joint_val_samples, eval_limit=args.joint_eval_limit,
        )
        frontier_metrics = evaluate_frontier_subset(
            model, val_loader,
            joint_val_samples=args.joint_val_samples, eval_limit=args.joint_eval_limit,
        )
        current_key = checkpoint_key(val_metrics, frontier_metrics, joint_metrics)
        is_best = best_key is None or current_key < best_key
        if is_best:
            best_key = current_key
            best_metrics = {**val_metrics, **joint_metrics, **frontier_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": {
                        "type": "241f_partial_unfreeze_multi_crps_state_metric_transport",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "path": path_config,
                        "prior": prior_config,
                        "integrated": integrated_config,
                        "state": state_config,
                        "metric": metric_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "cov_jitter": args.cov_jitter,
                        "base_nu": args.base_nu,
                        "mix_chunk_size": args.mix_chunk_size,
                        "lambda_term": args.lambda_term,
                        "lambda_pmax": args.lambda_pmax,
                        "lambda_tail": args.lambda_tail,
                        "lambda_es": args.lambda_es,
                        "afcrps_alpha": args.afcrps_alpha,
                        "tail_beta": args.tail_beta,
                        "K": args.K,
                        "train_ode_steps": args.train_ode_steps,
                    },
                    "args": vars(args),
                },
                Path(args.output_dir) / "best_model.pt",
            )

        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "time_sec": epoch_time,
            **train_metrics,
            **val_metrics,
            **joint_metrics,
            **frontier_metrics,
            "is_best": is_best,
        }
        history.append(make_serializable(row))
        history_path.write_text(json.dumps(history, indent=2))

        fm = train_metrics.get("train_flow_match_loss", 0.0)
        afc = train_metrics.get("train_afcrps_h30", 0.0)
        print(
            f"  [ep {epoch}/{args.epochs}] fm={fm:.4f} afc={afc:.4f} "
            f"val_fm={val_metrics.get('val_flow_match_loss', 0.0):.4f} "
            f"best={is_best} time={epoch_time:.1f}s"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "config": {
                "type": "241f_partial_unfreeze_multi_crps_state_metric_transport",
                "encoder": vars(encoder_config),
                "decoder": decoder_config, "flow": flow_config, "path": path_config,
                "prior": prior_config, "integrated": integrated_config,
                "state": state_config, "metric": metric_config,
                "support_lo": args.support_lo, "support_hi": args.support_hi,
                "support_eps": args.support_eps, "cov_jitter": args.cov_jitter,
                "base_nu": args.base_nu, "mix_chunk_size": args.mix_chunk_size,
                "lambda_term": args.lambda_term, "lambda_pmax": args.lambda_pmax,
                "lambda_tail": args.lambda_tail, "lambda_es": args.lambda_es,
                "afcrps_alpha": args.afcrps_alpha, "tail_beta": args.tail_beta,
                "K": args.K, "train_ode_steps": args.train_ode_steps,
            },
            "args": vars(args),
        },
        Path(args.output_dir) / "final_model.pt",
    )
    print(f"\n{'=' * 72}\nDone. Best selection key: {best_key}\n{'=' * 72}")


if __name__ == "__main__":
    main()
