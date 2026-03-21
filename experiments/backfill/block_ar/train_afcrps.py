"""
Training script for Exp 89: afCRPS Single-Pass Block-AR.

Replaces DDPM's 100-step diffusion loop with a single forward pass trained
with almost-fair CRPS loss on IV-space output. Keeps encoder, Conv3D backbone,
AdaGN, and exp(z × vol_scale) denormalization.

Usage:
    # Pretrained init
    PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
        --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --no_ema --epochs 30 --batch_size 16 --noise_dim 16 --n_members 4 \
        --lr_noise 1e-3 --lr_decoder 1e-4 --lambda_vs 0.1 \
        --output_dir models/backfill/afcrps_v1_pretrained --device cuda

    # Scratch init (encoder still from pretrained)
    PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
        --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --no_ema --from_scratch --epochs 30 --batch_size 16 --noise_dim 16 \
        --n_members 4 --lr 1e-3 --lambda_vs 0.1 \
        --output_dir models/backfill/afcrps_v1_scratch --device cuda
"""

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
    load_pretrained_weights,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def _hash_file(path: str | None) -> str | None:
    """Return a short SHA256 for provenance tracking."""
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def parse_progressive_schedule(spec: str | None) -> list[int]:
    """Parse comma-separated rollout horizons."""
    if spec is None:
        return []
    schedule = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("progressive schedule values must be positive")
        schedule.append(value)
    if not schedule:
        raise ValueError("progressive schedule must contain at least one horizon")
    return schedule


def build_progressive_epoch_plan(schedule: list[int], total_epochs: int) -> list[dict]:
    """Allocate early stages across 60% of epochs and hold the final horizon longest."""
    if not schedule:
        return []
    if total_epochs < len(schedule):
        raise ValueError(
            f"epochs={total_epochs} is too small for schedule with {len(schedule)} stages"
        )
    if len(schedule) == 1:
        return [{"start_epoch": 1, "end_epoch": total_epochs, "n_frames": schedule[0]}]

    warmup_epochs = max(len(schedule) - 1, int(round(total_epochs * 0.6)))
    warmup_epochs = min(total_epochs - 1, warmup_epochs)
    final_stage_epochs = total_epochs - warmup_epochs

    n_warmup_stages = len(schedule) - 1
    base = warmup_epochs // n_warmup_stages
    remainder = warmup_epochs % n_warmup_stages
    stage_epochs = [base + (1 if idx < remainder else 0) for idx in range(n_warmup_stages)]
    stage_epochs.append(final_stage_epochs)

    plan = []
    start_epoch = 1
    for n_frames, n_stage_epochs in zip(schedule, stage_epochs):
        end_epoch = start_epoch + n_stage_epochs - 1
        plan.append(
            {
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "n_frames": n_frames,
            }
        )
        start_epoch = end_epoch + 1
    return plan


def resolve_progressive_frames(epoch: int, epoch_plan: list[dict]) -> int:
    """Return the active rollout horizon for an epoch."""
    for stage in epoch_plan:
        if stage["start_epoch"] <= epoch <= stage["end_epoch"]:
            return stage["n_frames"]
    return epoch_plan[-1]["n_frames"]


def train_epoch(model, loader, optimizer, device, n_members, lambda_vs, grad_clip, n_train_blocks=1, lambda_is=0.0, lambda_cs_reg=0.0, lambda_kurt=0.0, lambda_es=0.0, lambda_cell_var=0.0, lambda_cum_cal=0.0, lambda_vr=0.0, lambda_ortho=0.0, lambda_acf=0.0, lambda_rank=0.0, n_frames=0, unfreeze_encoder=False, lambda_ortho_enc=0.0):
    model.train()
    # Keep encoder in eval mode (frozen, no dropout) unless unfrozen
    if not unfreeze_encoder:
        model.encoder.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_spread = 0.0
    total_vs = 0.0
    total_es = 0.0
    total_es_acc = 0.0
    total_es_spr = 0.0
    total_is = 0.0
    total_kurt = 0.0
    total_raw_kurt = 0.0
    total_bias = 0.0
    total_cell_var = 0.0
    total_cum_cal = 0.0
    total_ortho = 0.0
    total_ortho_enc = 0.0
    total_acf = 0.0
    total_acf_mean = 0.0
    total_rank_loss = 0.0
    total_eff_rank = 0.0
    n_batches = 0

    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        extra_hist = batch.get("history_returns")
        if extra_hist is not None:
            extra_hist = extra_hist.to(device)

        result = model(history, future, n_members=n_members, lambda_vs=lambda_vs,
                       lambda_is=lambda_is, lambda_cs_reg=lambda_cs_reg,
                       lambda_kurt=lambda_kurt, lambda_es=lambda_es,
                       lambda_cell_var=lambda_cell_var,
                       lambda_cum_cal=lambda_cum_cal,
                       lambda_vr=lambda_vr,
                       lambda_acf=lambda_acf,
                       lambda_rank=lambda_rank,
                       n_train_blocks=n_train_blocks,
                       n_frames=n_frames,
                       extra_hist=extra_hist)
        loss = result["loss"]

        # Orthogonal regularization on noise_skip_proj (Exp 123b)
        # Penalizes cosine similarity between skip weight rows → prevents rank-1 alignment
        ortho_loss = torch.tensor(0.0, device=device)
        if lambda_ortho > 0 and hasattr(model, 'frame_decoder') and model.frame_decoder.noise_skip_proj is not None:
            W = model.frame_decoder.noise_skip_proj.weight  # (n_cells, noise_dim)
            W_norm = F.normalize(W, dim=1)  # normalize each row
            cosim = W_norm @ W_norm.T  # (n_cells, n_cells)
            # Zero out diagonal (self-similarity = 1, not penalized)
            mask = 1.0 - torch.eye(cosim.shape[0], device=device)
            ortho_loss = (cosim * mask).abs().mean()
            loss = loss + lambda_ortho * ortho_loss

        # Orthogonal regularization on encoder weights (Exp 139a — RC6 Step 1)
        # Penalizes ||W^TW - I||^2_F on encoder weight matrices to prevent rank collapse
        ortho_enc_loss = torch.tensor(0.0, device=device)
        if lambda_ortho_enc > 0 and unfreeze_encoder:
            for name, param in model.encoder.named_parameters():
                if 'weight' in name and param.dim() == 2 and min(param.shape) > 1:
                    W = param  # (M, N)
                    # Regularize on the smaller dimension for efficiency
                    if W.shape[0] <= W.shape[1]:
                        gram = W @ W.T  # (M, M)
                        eye = torch.eye(W.shape[0], device=device)
                    else:
                        gram = W.T @ W  # (N, N)
                        eye = torch.eye(W.shape[1], device=device)
                    ortho_enc_loss = ortho_enc_loss + ((gram - eye) ** 2).sum()
            loss = loss + lambda_ortho_enc * ortho_enc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()

        total_loss += loss.item()
        total_mae += result["mae"].item()
        total_spread += result["spread"].item()
        total_vs += result["variogram"].item()
        total_es += result["energy_score"].item()
        total_es_acc += result.get("es_accuracy", torch.tensor(0.0)).item()
        total_es_spr += result.get("es_spread", torch.tensor(0.0)).item()
        total_is += result["interval_score"].item()
        total_kurt += result["kurt_loss"].item()
        total_raw_kurt += result["raw_kurt"].item()
        total_bias += result.get("bias_loss", torch.tensor(0.0)).item()
        total_cell_var += result.get("cell_var_loss", torch.tensor(0.0)).item()
        total_cum_cal += result.get("cum_cal_loss", torch.tensor(0.0)).item()
        total_ortho += ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss
        total_ortho_enc += ortho_enc_loss.item() if isinstance(ortho_enc_loss, torch.Tensor) else ortho_enc_loss
        total_acf += result.get("acf_loss", torch.tensor(0.0)).item()
        total_acf_mean += result.get("acf_mean", torch.tensor(0.0)).item()
        total_rank_loss += result.get("rank_loss", torch.tensor(0.0)).item()
        total_eff_rank += result.get("eff_rank", torch.tensor(0.0)).item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": total_mae / max(n_batches, 1),
        "spread": total_spread / max(n_batches, 1),
        "variogram": total_vs / max(n_batches, 1),
        "energy_score": total_es / max(n_batches, 1),
        "es_accuracy": total_es_acc / max(n_batches, 1),
        "es_spread": total_es_spr / max(n_batches, 1),
        "interval_score": total_is / max(n_batches, 1),
        "kurt_loss": total_kurt / max(n_batches, 1),
        "raw_kurt": total_raw_kurt / max(n_batches, 1),
        "spread_mae_ratio": total_spread / max(total_mae, 1e-8),
        "bias_loss": total_bias / max(n_batches, 1),
        "cell_var_loss": total_cell_var / max(n_batches, 1),
        "cum_cal_loss": total_cum_cal / max(n_batches, 1),
        "ortho_loss": total_ortho / max(n_batches, 1),
        "ortho_enc_loss": total_ortho_enc / max(n_batches, 1),
        "acf_loss": total_acf / max(n_batches, 1),
        "acf_mean": total_acf_mean / max(n_batches, 1),
        "rank_loss": total_rank_loss / max(n_batches, 1),
        "eff_rank": total_eff_rank / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, device, n_members):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_spread = 0.0
    n_batches = 0

    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        extra_hist = batch.get("history_returns")
        if extra_hist is not None:
            extra_hist = extra_hist.to(device)
        result = model(history, future, n_members=n_members, extra_hist=extra_hist)
        total_loss += result["loss"].item()
        total_mae += result["mae"].item()
        total_spread += result["spread"].item()
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_mae": total_mae / max(n_batches, 1),
        "val_spread": total_spread / max(n_batches, 1),
        "val_spread_mae_ratio": total_spread / max(total_mae, 1e-8),
    }


@torch.no_grad()
def quick_eval(model, loader, device, n_samples=50, max_batches=5):
    """Quick evaluation: CI coverage and kurtosis on val set."""
    model.eval()
    all_coverages = []
    all_gt_changes = []
    all_gen_changes = []
    all_member_samples = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        extra_hist = batch.get("history_returns")
        if extra_hist is not None:
            extra_hist = extra_hist.to(device)

        samples = model.sample(history, n_samples=n_samples, extra_hist=extra_hist)  # (B, K, 30, 5, 5)
        gt = denormalize_iv(future)  # (B, 30, 5, 5)

        # 90% CI coverage
        q05 = torch.quantile(samples, 0.05, dim=1)
        q95 = torch.quantile(samples, 0.95, dim=1)
        covered = ((gt >= q05) & (gt <= q95)).float().mean().item()
        all_coverages.append(covered)

        # Kurtosis: daily changes of ensemble mean
        gt_changes = (gt[:, 1:] - gt[:, :-1]).reshape(-1)
        gen_mean = samples.mean(dim=1)
        gen_changes = (gen_mean[:, 1:] - gen_mean[:, :-1]).reshape(-1)
        all_gt_changes.append(gt_changes.cpu())
        all_gen_changes.append(gen_changes.cpu())
        all_member_samples.append(samples.cpu())

    coverage = np.mean(all_coverages) if all_coverages else 0.0

    if all_gt_changes:
        gt_all = torch.cat(all_gt_changes)
        gen_all = torch.cat(all_gen_changes)
        gt_kurt = torch.mean((gt_all - gt_all.mean()) ** 4) / (gt_all.std() ** 4 + 1e-8)
        gen_kurt = torch.mean((gen_all - gen_all.mean()) ** 4) / (gen_all.std() ** 4 + 1e-8)
        kurtosis_ratio_mean = (gen_kurt / gt_kurt).item() if gt_kurt > 0 else 0.0

        # Also compute kurtosis from individual members (more representative)
        all_member_changes = []
        for batch_samples in all_member_samples:
            # batch_samples: (B, K, T, H, W)
            member_changes = batch_samples[:, :, 1:] - batch_samples[:, :, :-1]
            all_member_changes.append(member_changes.reshape(-1))
        if all_member_changes:
            mc = torch.cat(all_member_changes)
            member_kurt = torch.mean((mc - mc.mean()) ** 4) / (mc.std() ** 4 + 1e-8)
            kurtosis_ratio = (member_kurt / gt_kurt).item() if gt_kurt > 0 else 0.0
        else:
            kurtosis_ratio = kurtosis_ratio_mean
    else:
        kurtosis_ratio = 0.0
        kurtosis_ratio_mean = 0.0

    return {
        "coverage_90": coverage,
        "kurtosis_ratio": kurtosis_ratio,
        "kurtosis_ratio_mean": kurtosis_ratio_mean,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train afCRPS single-pass Block-AR")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to pretrained DDPM checkpoint (for encoder + optional weights)")
    parser.add_argument("--no_ema", action="store_true",
                        help="Use model_state_dict instead of ema_params")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Random init for decoder (only encoder from pretrained)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--n_members", type=int, default=4, help="K ensemble members per step")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (used for scratch init)")
    parser.add_argument("--lr_noise", type=float, default=1e-3,
                        help="LR for noise MLP (new params)")
    parser.add_argument("--lr_decoder", type=float, default=1e-4,
                        help="LR for decoder (pretrained params)")
    parser.add_argument("--lambda_vs", type=float, default=0.1,
                        help="Variogram score weight")
    parser.add_argument("--shared_noise_input", action="store_true",
                        help="Inject first noise element as shared spatial input (cross-cell correlation)")
    parser.add_argument("--lambda_es", type=float, default=0.0,
                        help="Energy score weight (multivariate decorrelation)")
    parser.add_argument("--lambda_is", type=float, default=0.0,
                        help="Interval score weight (CI calibration pressure)")
    parser.add_argument("--is_warmup_epoch", type=int, default=0,
                        help="Keep lambda_is=0 for first N epochs, then use configured value")
    parser.add_argument("--lambda_cs_reg", type=float, default=0.0,
                        help="L2 penalty pulling cell_scale toward its spatial mean")
    parser.add_argument("--lambda_kurt", type=float, default=0.0,
                        help="Kurtosis matching loss weight")
    parser.add_argument("--lambda_cell_var", type=float, default=0.0,
                        help="Per-cell variance matching loss weight")
    parser.add_argument("--lambda_vr", type=float, default=0.0,
                        help="Variance ratio loss weight (Exp 113a)")
    parser.add_argument("--lambda_cum_cal", type=float, default=0.0,
                        help="Cumulative calibration loss weight (matches ensemble var to MSE at h=7,14,30)")
    parser.add_argument("--n_train_blocks", type=int, default=1,
                        help="Number of AR blocks to generate during training (1=block1 only, 3=full 30 frames)")
    parser.add_argument("--oneshot_additive", action="store_true",
                        help="One-shot Conv3D with additive dynamics (Exp 111a)")
    parser.add_argument("--noise_bottleneck_dim", type=int, default=0,
                        help="Factor noise constraint: sample in k dims, project to noise_dim (Exp 115a)")
    parser.add_argument("--direct_iv", action="store_true",
                        help="Direct IV prediction (no exp/baseline transform)")
    parser.add_argument("--no_tanh", action="store_true",
                        help="Remove tanh bounding from decoder output")
    parser.add_argument("--learned_vol_scale", action="store_true",
                        help="Per-cell vol_scale from condition MLP (replaces scalar vol_scale)")
    parser.add_argument("--cond_noise_mlp", action="store_true",
                        help="Feed condition into noise MLP for regime-dependent diversity")
    parser.add_argument("--noise_dist", type=str, default="gaussian",
                        choices=["gaussian", "student_t"],
                        help="Noise distribution for ensemble diversity")
    parser.add_argument("--student_t_df", type=float, default=4.0,
                        help="Degrees of freedom for Student-t noise")
    parser.add_argument("--twcrps_beta", type=float, default=0.0,
                        help="twCRPS beta (0=standard, 2.0=3x weight at ±1 IQR)")
    parser.add_argument("--spread_weight", type=float, default=0.5,
                        help="CRPS spread term coefficient (0.5=standard, lower=less over-spread)")
    parser.add_argument("--ar_frame", action="store_true",
                        help="Use per-frame AR decoder instead of Conv3D")
    parser.add_argument("--progressive_rollout", action="store_true",
                        help="Progressively increase AR rollout length across epochs")
    parser.add_argument("--progressive_schedule", type=str, default=None,
                        help="Comma-separated rollout horizons, e.g. 30,60,120,252")
    parser.add_argument("--ar_cell_spread", action="store_true",
                        help="Learned per-cell spread scaling for AR frame decoder")
    parser.add_argument("--ar_static_cell_scale", action="store_true",
                        help="Static per-cell scale (nn.Parameter, no condition dependence)")
    parser.add_argument("--ar_bias_lambda", type=float, default=0.0,
                        help="Delta zero-mean bias loss weight")
    parser.add_argument("--ar_frame_hidden", type=int, default=128,
                        help="FrameDecoder MLP hidden dim")
    parser.add_argument("--ar_percell_vol_scale", action="store_true",
                        help="Per-cell vol_scale from history daily change std")
    parser.add_argument("--ar_dual_pos", action="store_true",
                        help="Use local position plus coarse horizon bucket in AR frame decoder")
    parser.add_argument("--ar_local_pos_period", type=int, default=30,
                        help="Period for local AR-frame position index")
    parser.add_argument("--ar_horizon_bucket_size", type=int, default=30,
                        help="Frames per coarse horizon bucket when --ar_dual_pos is enabled")
    parser.add_argument("--ar_horizon_max_buckets", type=int, default=9,
                        help="Maximum coarse horizon buckets for dual-position AR frame decoder")
    parser.add_argument("--ar_horizon_embed_dim", type=int, default=8,
                        help="Embedding dim for coarse horizon bucket")
    parser.add_argument("--ar_dynamic_vs", action="store_true",
                        help="Enable bounded dynamic vol-scale modulation in AR frame mode")
    parser.add_argument("--ar_dynamic_vs_mode", type=str, default="scalar",
                        choices=["scalar", "tenor"],
                        help="Dynamic vol-scale mode for AR frame decoder")
    parser.add_argument("--ar_dynamic_vs_scale", type=float, default=0.15,
                        help="Multiplier strength for dynamic vol-scale modulation")
    parser.add_argument("--ar_log_space", action="store_true",
                        help="Multiplicative dynamics: iv = prev * exp(vs * delta)")
    parser.add_argument("--ar_logit_space", action="store_true",
                        help="Logit-space dynamics: iv = sigmoid(logit(prev) + vs * delta)")
    parser.add_argument("--ar_logit_jac", action="store_true",
                        help="Logit + Jacobian: iv = sigmoid(logit(prev) + vs*delta/(prev*(1-prev)))")
    parser.add_argument("--ar_frame_rho", type=float, default=0.8,
                        help="AR noise temporal correlation (default: 0.8, 0.0=iid)")
    parser.add_argument("--ar_reflect", action="store_true",
                        help="Reflecting boundaries: bounce off [floor, 1.0] instead of clamping")
    parser.add_argument("--ar_factor_noise", action="store_true",
                        help="Factor model noise: z_factors @ loadings.T for per-cell noise")
    parser.add_argument("--ar_n_factors", type=int, default=5,
                        help="Number of latent noise factors (requires --ar_factor_noise)")
    parser.add_argument("--ar_factor_noise_norm", action="store_true",
                        help="Normalize cell_noise to unit variance (loadings control corr only)")
    parser.add_argument("--ar_factor_init_scale", type=float, default=0.1,
                        help="Factor loadings initialization std (default: 0.1)")
    parser.add_argument("--ar_floor_clamp", type=float, default=0.001,
                        help="Lower IV clamp in AR frame generation (default: 0.001)")
    parser.add_argument("--ar_cell_embed", action="store_true",
                        help="Per-cell FrameDecoder with learned cell embedding")
    parser.add_argument("--ar_cell_embed_dim", type=int, default=8,
                        help="Cell embedding dimension (default: 8)")
    parser.add_argument("--ar_cell_cond_offset", action="store_true",
                        help="Per-cell condition offsets for spatial decorrelation (Exp 93d)")
    parser.add_argument("--ar_input_noise_std", type=float, default=0.0,
                        help="Noise std on GRU input during training (Exp 93f, default: 0.0)")
    parser.add_argument("--ar_percell_bias", action="store_true",
                        help="Per-cell conditional bias loss (Exp 94a)")
    parser.add_argument("--ar_independent_cells", action="store_true",
                        help="25 independent per-cell MLPs (Exp 98a)")
    parser.add_argument("--ar_cell_hidden", type=int, default=32,
                        help="Hidden dim for per-cell MLPs (default: 32)")
    parser.add_argument("--ar_noise_skip", action="store_true",
                        help="Per-cell noise skip connection bypassing shared MLP (Exp 99b)")
    parser.add_argument("--ar_skip_bypass_spread", action="store_true",
                        help="Skip connection bypasses cell_spread (Exp 99j)")
    parser.add_argument("--ar_noise_scale_cond", action="store_true",
                        help="Condition-dependent per-cell noise scale (Exp 102a)")
    parser.add_argument("--ar_noise_scale_min", type=float, default=0.1,
                        help="Lower bound for per-cell noise scale")
    parser.add_argument("--ar_learned_rho", action="store_true",
                        help="Condition-dependent AR noise rho (Exp 103a)")
    parser.add_argument("--ar_learned_rho_init", type=float, default=1.1,
                        help="Init bias for rho head (sigmoid(1.1)≈0.75)")
    parser.add_argument("--ar_learned_rho_min", type=float, default=0.0,
                        help="Lower clamp for learned rho")
    parser.add_argument("--ar_learned_rho_max", type=float, default=1.0,
                        help="Upper clamp for learned rho")
    parser.add_argument("--ar_mean_revert", action="store_true",
                        help="Mean-reversion dynamics (Exp 104a)")
    parser.add_argument("--ar_mean_revert_alpha_init", type=float, default=-3.0,
                        help="Init for mean-reversion alpha (sigmoid(-3)≈0.047)")
    parser.add_argument("--ar_mean_revert_percell", action="store_true",
                        help="Per-cell mean-reversion alpha (Exp 109a)")
    parser.add_argument("--ar_percell_spread_cond", action="store_true",
                        help="Per-cell condition for cell_spread (Exp 110a)")
    parser.add_argument("--ar_adagn_noise", action="store_true",
                        help="AdaGN noise conditioning in FrameDecoder MLP (Exp 120a)")
    parser.add_argument("--ar_noisefree_mlp", action="store_true",
                        help="Noise-free MLP: noise only through skip (Exp 120b)")
    parser.add_argument("--ar_lowrank_spread", type=int, default=0,
                        help="Low-rank cell_spread factors (0=off, 3=Exp 124a)")
    parser.add_argument("--ar_frame_n_layers", type=int, default=2,
                        help="MLP hidden layer count (2=default, 3=Exp 128a)")
    parser.add_argument("--lambda_ortho", type=float, default=0.0,
                        help="Orthogonal reg on noise_skip_proj rows (Exp 123b)")
    parser.add_argument("--lambda_ortho_enc", type=float, default=0.0,
                        help="Orthogonal reg on encoder weights: ||W^TW - I||^2_F (Exp 139a, RC6)")
    parser.add_argument("--lambda_acf", type=float, default=0.0,
                        help="Explicit ACF loss on ensemble deltas (Exp 123a)")
    parser.add_argument("--lambda_rank", type=float, default=0.0,
                        help="Log-det covariance penalty for ensemble diversity (H1 diagnostic)")
    parser.add_argument("--ar_cln_warmup", type=int, default=0,
                        help="CLN warmup frames: scale modulation by min(1, t/N) (Exp 132b)")
    parser.add_argument("--joint_decoder", action="store_true",
                        help="Use joint transformer decoder instead of AR loop (H4)")
    parser.add_argument("--joint_n_layers", type=int, default=4,
                        help="Transformer layers for joint decoder")
    parser.add_argument("--joint_d_model", type=int, default=128,
                        help="Hidden dim for joint transformer decoder")
    parser.add_argument("--joint_noise_factors", type=int, default=0,
                        help="Shared noise factors for joint decoder (0=per-cell, 5=shared k factors)")
    parser.add_argument("--curriculum_noise_epoch", type=int, default=0,
                        help="Switch from gaussian to student_t noise at this epoch (Exp 126a)")
    parser.add_argument("--extra_features", type=int, default=0,
                        help="Number of extra encoder features (e.g. 1 for returns)")
    parser.add_argument("--return_scale", type=float, default=0.05,
                        help="Scale for tanh bounding of returns: tanh(ret / scale)")
    parser.add_argument("--freeze_after_epoch", type=int, default=0,
                        help="Freeze frame_decoder MLP after this epoch, keep only skip/vol_scale/spread trainable (0=disabled)")
    parser.add_argument("--freeze_spread_too", action="store_true",
                        help="Also freeze cell_spread_linear when --freeze_after_epoch triggers")
    parser.add_argument("--freeze_skip_too", action="store_true",
                        help="Also freeze noise_skip_proj when --freeze_after_epoch triggers (Exp 114a)")
    parser.add_argument("--freeze_encoder_too", action="store_true",
                        help="Also freeze encoder when --freeze_after_epoch triggers (Exp 139a_v2)")
    parser.add_argument("--unfreeze_encoder", action="store_true",
                        help="Unfreeze GRU encoder")
    parser.add_argument("--lr_encoder", type=float, default=1e-4,
                        help="LR for encoder when unfrozen (default: 1e-4)")
    parser.add_argument("--no_pretrained_encoder", action="store_true",
                        help="Skip loading pretrained encoder weights (random init)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--disable_early_stop", action="store_true",
                        help="Disable heuristic early-stop guards during training")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a saved checkpoint")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--n_eval_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="models/backfill/afcrps_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.from_scratch and args.no_pretrained_encoder:
        parser.error(
            "--from_scratch cannot be combined with --no_pretrained_encoder; "
            "use --no_pretrained_encoder alone for full random init"
        )
    if not args.no_pretrained_encoder and not args.base_model:
        parser.error(
            "--base_model is required unless --no_pretrained_encoder is set"
        )
    if args.progressive_schedule and not args.progressive_rollout:
        parser.error("--progressive_schedule requires --progressive_rollout")
    if args.ar_dynamic_vs and not args.ar_frame:
        parser.error("--ar_dynamic_vs requires --ar_frame")
    if args.ar_dual_pos and not args.ar_frame:
        parser.error("--ar_dual_pos requires --ar_frame")
    if sum([args.ar_independent_cells, args.ar_cell_embed, args.ar_cell_cond_offset]) > 1:
        parser.error("--ar_independent_cells, --ar_cell_embed, --ar_cell_cond_offset are mutually exclusive")
    if args.ar_horizon_bucket_size <= 0 or args.ar_horizon_max_buckets <= 0:
        parser.error("horizon bucket settings must be positive")
    if args.ar_local_pos_period <= 0:
        parser.error("--ar_local_pos_period must be positive")
    if args.resume_from and not Path(args.resume_from).exists():
        parser.error(f"--resume_from not found: {args.resume_from}")

    try:
        progressive_schedule = (
            parse_progressive_schedule(args.progressive_schedule)
            if args.progressive_rollout and args.progressive_schedule
            else ([5, 15, 30] if args.progressive_rollout else [])
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        progressive_epoch_plan = (
            build_progressive_epoch_plan(progressive_schedule, args.epochs)
            if progressive_schedule else []
        )
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load base checkpoint to get encoder config
    if args.base_model:
        base_ckpt = torch.load(args.base_model, map_location="cpu", weights_only=False)
        base_cfg = base_ckpt.get("config", {})
    else:
        base_ckpt = None
        base_cfg = {}
    if args.no_pretrained_encoder:
        pretrained_init_mode = "random_all"
    elif args.from_scratch:
        pretrained_init_mode = "encoder_only"
    else:
        pretrained_init_mode = "full_pretrained"
    base_model_hash = _hash_file(args.base_model)
    resolved_future_len = (
        max(progressive_schedule)
        if args.ar_frame and progressive_schedule
        else base_cfg.get("future_len", 30)
    )

    # Build SinglePassConfig
    config = SinglePassConfig(
        history_len=base_cfg.get("history_len", 30),
        future_len=resolved_future_len,
        surface_h=base_cfg.get("surface_h", 5),
        surface_w=base_cfg.get("surface_w", 5),
        block_size=base_cfg.get("block_size", 10),
        gru_hidden_dim=base_cfg.get("gru_hidden_dim", 64),
        bottleneck_dim=base_cfg.get("bottleneck_dim", 128),
        encoder_dropout=base_cfg.get("encoder_dropout", 0.1),
        conv3d_base_channels=base_cfg.get("conv3d_base_channels", 32),
        conv3d_n_res_blocks=base_cfg.get("conv3d_n_res_blocks", 6),
        conv3d_groups=base_cfg.get("conv3d_groups", 8),
        pos_embed_dim=base_cfg.get("pos_embed_dim", 16),
        noise_dim=args.noise_dim,
        noise_embed_dim=base_cfg.get("conv3d_noise_embed_dim", 64),
        shared_noise_input=args.shared_noise_input,
        cond_noise_mlp=args.cond_noise_mlp,
        noise_dist=args.noise_dist,
        student_t_df=args.student_t_df,
        global_mean_vol=base_cfg.get("global_mean_vol", 0.0187),
        vol_scale_min=base_cfg.get("vol_scale_min", 0.5),
        vol_scale_max=base_cfg.get("vol_scale_max", 2.0),
        vol_scale_power=base_cfg.get("vol_scale_power", 1.0),
        direct_iv=args.direct_iv,
        oneshot_additive=args.oneshot_additive,
        noise_bottleneck_dim=args.noise_bottleneck_dim,
        no_tanh=args.no_tanh,
        learned_vol_scale=args.learned_vol_scale,
        twcrps_beta=args.twcrps_beta,
        spread_weight=args.spread_weight,
        ar_frame=args.ar_frame,
        ar_frame_cell_spread=args.ar_cell_spread,
        ar_frame_static_cell_scale=args.ar_static_cell_scale,
        ar_frame_hidden=args.ar_frame_hidden,
        ar_frame_bias_lambda=args.ar_bias_lambda,
        ar_frame_percell_vol_scale=args.ar_percell_vol_scale,
        ar_dual_pos=args.ar_dual_pos,
        ar_local_pos_period=args.ar_local_pos_period,
        ar_horizon_bucket_size=args.ar_horizon_bucket_size,
        ar_horizon_max_buckets=args.ar_horizon_max_buckets,
        ar_horizon_embed_dim=args.ar_horizon_embed_dim,
        ar_dynamic_vs=args.ar_dynamic_vs,
        ar_dynamic_vs_mode=args.ar_dynamic_vs_mode,
        ar_dynamic_vs_scale=args.ar_dynamic_vs_scale,
        ar_frame_log_space=args.ar_log_space,
        ar_frame_logit_space=getattr(args, 'ar_logit_space', False),
        ar_frame_logit_jac=getattr(args, 'ar_logit_jac', False),
        ar_frame_rho=args.ar_frame_rho,
        ar_frame_reflect=getattr(args, 'ar_reflect', False),
        ar_factor_noise=args.ar_factor_noise,
        ar_n_factors=args.ar_n_factors,
        ar_factor_noise_norm=args.ar_factor_noise_norm,
        ar_factor_init_scale=args.ar_factor_init_scale,
        ar_frame_floor_clamp=args.ar_floor_clamp,
        ar_cell_embed=args.ar_cell_embed,
        ar_cell_embed_dim=args.ar_cell_embed_dim,
        ar_cell_cond_offset=args.ar_cell_cond_offset,
        ar_input_noise_std=args.ar_input_noise_std,
        ar_percell_bias=args.ar_percell_bias,
        ar_independent_cells=args.ar_independent_cells,
        ar_cell_hidden=args.ar_cell_hidden,
        ar_noise_skip=args.ar_noise_skip,
        ar_skip_bypass_spread=args.ar_skip_bypass_spread,
        ar_noise_scale_cond=args.ar_noise_scale_cond,
        ar_noise_scale_min=args.ar_noise_scale_min,
        ar_learned_rho=args.ar_learned_rho,
        ar_learned_rho_init=args.ar_learned_rho_init,
        ar_learned_rho_min=args.ar_learned_rho_min,
        ar_learned_rho_max=args.ar_learned_rho_max,
        ar_mean_revert=args.ar_mean_revert,
        ar_mean_revert_alpha_init=args.ar_mean_revert_alpha_init,
        ar_mean_revert_percell=args.ar_mean_revert_percell,
        ar_percell_spread_cond=getattr(args, 'ar_percell_spread_cond', False),
        ar_adagn_noise=getattr(args, 'ar_adagn_noise', False),
        ar_cln_warmup=getattr(args, 'ar_cln_warmup', 0),
        joint_decoder=getattr(args, 'joint_decoder', False),
        joint_n_layers=getattr(args, 'joint_n_layers', 4),
        joint_d_model=getattr(args, 'joint_d_model', 128),
        joint_noise_factors=getattr(args, 'joint_noise_factors', 0),
        ar_noisefree_mlp=getattr(args, 'ar_noisefree_mlp', False),
        ar_lowrank_spread=getattr(args, 'ar_lowrank_spread', 0),
        ar_frame_n_layers=getattr(args, 'ar_frame_n_layers', 2),
        extra_features=args.extra_features,
        return_scale=args.return_scale,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Create model
    model = SinglePassBlockAR(config).to(device)

    # Load pretrained weights
    if args.no_pretrained_encoder:
        print("Random encoder init (no pretrained weights loaded)")
    elif args.from_scratch:
        # Only load encoder weights
        src_state = base_ckpt["model_state_dict"] if args.no_ema else base_ckpt.get("ema_params", base_ckpt["model_state_dict"])
        tgt_state = model.state_dict()
        enc_transferred = 0
        for key, val in src_state.items():
            if key.startswith("encoder."):
                if key in tgt_state and tgt_state[key].shape == val.shape:
                    tgt_state[key] = val
                    enc_transferred += 1
                elif (key == "encoder.gru.weight_ih_l0"
                      and key in tgt_state
                      and tgt_state[key].shape[0] == val.shape[0]
                      and tgt_state[key].shape[1] > val.shape[1]):
                    # GRU input expanded by extra_features — zero then partial copy
                    tgt_state[key].zero_()
                    tgt_state[key][:, :val.shape[1]] = val
                    enc_transferred += 1
        model.load_state_dict(tgt_state)
        print(f"Scratch init: transferred {enc_transferred} encoder params")
    else:
        stats = load_pretrained_weights(model, args.base_model, device="cpu", use_ema=not args.no_ema)
        print(f"Pretrained init: {stats}")
    model = model.to(device)

    # Re-init conv_out for direct IV mode (pretrained weights learned z-scores for exp())
    if args.direct_iv and hasattr(model, 'decoder'):
        nn.init.normal_(model.decoder.conv_out.weight, std=0.01)
        nn.init.zeros_(model.decoder.conv_out.bias)
        print("  Re-initialized conv_out for direct IV mode")

    # Freeze encoder (unless --unfreeze_encoder)
    if not args.unfreeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    enc_status = "encoder trainable" if args.unfreeze_encoder else "encoder frozen"
    print(f"Model: {n_total:,} total, {n_trainable:,} trainable ({enc_status})")

    # Optimizer with differential learning rates
    if args.from_scratch:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    elif args.ar_frame:
        # AR frame mode: only frame_decoder params (no NoiseMLP, no Conv3D decoder)
        if args.ar_cell_cond_offset and hasattr(model.frame_decoder, 'cond_offsets') and model.frame_decoder.cond_offsets is not None:
            offset_ids = {id(model.frame_decoder.cond_offsets)}
            decoder_params = [p for p in model.frame_decoder.parameters() if id(p) not in offset_ids]
            param_groups = [
                {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
                {"params": [model.frame_decoder.cond_offsets], "lr": 1e-3, "weight_decay": 0.0},
            ]
        elif args.ar_cell_embed and hasattr(model.frame_decoder, 'cell_emb') and model.frame_decoder.cell_emb is not None:
            cell_emb_ids = {id(p) for p in model.frame_decoder.cell_emb.parameters()}
            decoder_params = [p for p in model.frame_decoder.parameters() if id(p) not in cell_emb_ids]
            param_groups = [
                {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
                {"params": list(model.frame_decoder.cell_emb.parameters()), "lr": 1e-3, "weight_decay": 0.0},
            ]
        else:
            decoder_params = list(model.frame_decoder.parameters())
            param_groups = [
                {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
            ]
        # Joint transformer decoder (H4) — add its params to optimizer
        if hasattr(model, 'joint_transformer'):
            param_groups.append(
                {"params": list(model.joint_transformer.parameters()), "lr": args.lr_decoder, "weight_decay": 1e-4},
            )
        if args.unfreeze_encoder:
            encoder_params = list(model.encoder.parameters())
            param_groups.append(
                {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": 1e-4},
            )
        if hasattr(model, 'cell_spread_linear') and model.cell_spread_linear is not None:
            spread_params = list(model.cell_spread_linear.parameters())
            param_groups.append(
                {"params": spread_params, "lr": args.lr_decoder, "weight_decay": 0.1},
            )
        if hasattr(model, 'cell_spread_factor') and model.cell_spread_factor is not None:
            spread_params = list(model.cell_spread_factor.parameters()) + list(model.cell_spread_expand.parameters())
            param_groups.append(
                {"params": spread_params, "lr": args.lr_decoder, "weight_decay": 0.1},
            )
        if hasattr(model, 'spread_cell_proj'):
            param_groups.append(
                {"params": list(model.spread_cell_proj.parameters()), "lr": args.lr_decoder, "weight_decay": 1e-4},
            )
        if hasattr(model, 'cell_scale'):
            param_groups.append(
                {"params": [model.cell_scale], "lr": 1e-3, "weight_decay": 0.0},
            )
        if hasattr(model, 'dynamic_vs_head'):
            param_groups.append(
                {"params": list(model.dynamic_vs_head.parameters()), "lr": args.lr_decoder, "weight_decay": 1e-4},
            )
        if hasattr(model, 'factor_loadings'):
            param_groups.append(
                {"params": [model.factor_loadings], "lr": 1e-3, "weight_decay": 0.0},
            )
        if hasattr(model, 'noise_scale_head'):
            param_groups.append(
                {"params": list(model.noise_scale_head.parameters()), "lr": args.lr_decoder, "weight_decay": 1e-4},
            )
        if hasattr(model, 'noise_bottleneck'):
            param_groups.append(
                {"params": list(model.noise_bottleneck.parameters()), "lr": args.lr_decoder, "weight_decay": 1e-4},
            )
        if hasattr(model, 'rho_head'):
            param_groups.append(
                {"params": list(model.rho_head.parameters()), "lr": args.lr_decoder, "weight_decay": 1e-4},
            )
        if hasattr(model, 'mr_mu_head'):
            mr_params = list(model.mr_mu_head.parameters()) + list(model.mr_alpha_head.parameters())
            param_groups.append(
                {"params": mr_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
            )
        optimizer = torch.optim.AdamW(param_groups)
    else:
        noise_params = list(model.noise_mlp.parameters())
        spread_params = []
        if hasattr(model, 'cell_spread_mlp'):
            spread_params = list(model.cell_spread_mlp.parameters())
        decoder_params = [p for n, p in model.decoder.named_parameters()
                          if p.requires_grad]
        param_groups = [
            {"params": noise_params, "lr": args.lr_noise, "weight_decay": 1e-4},
            {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
        ]
        if spread_params:
            param_groups.append({"params": spread_params, "lr": args.lr_noise, "weight_decay": 0.1})
        optimizer = torch.optim.AdamW(param_groups)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_coverage = 0.0
    history_log = []
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        model = model.to(device)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        scheduler_state = resume_ckpt.get("lr_scheduler_state_dict")
        if scheduler_state is not None:
            lr_scheduler.load_state_dict(scheduler_state)
        else:
            lr_scheduler = None
            print("Resume checkpoint has no lr_scheduler_state_dict; continuing with fixed current LR")

        best_model_path = Path(args.output_dir) / "best_model.pt"
        if best_model_path.exists():
            best_model_ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
            best_val_loss = best_model_ckpt.get("metrics", {}).get("val_loss", best_val_loss)
        best_cov_path = Path(args.output_dir) / "best_coverage_model.pt"
        if best_cov_path.exists():
            best_cov_ckpt = torch.load(best_cov_path, map_location="cpu", weights_only=False)
            best_coverage = best_cov_ckpt.get("metrics", {}).get("coverage_90", best_coverage)
        history_path = Path(args.output_dir) / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history_log = json.load(f)
        print(f"Resumed from {args.resume_from} at epoch {start_epoch}")

    # Dataset
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    returns = data["ret"] if args.extra_features > 0 else None
    if returns is not None:
        print(f"Returns loaded: {returns.shape}, scale={args.return_scale}")
    train_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=0, end_idx=4040,
        returns=returns, return_scale=args.return_scale,
    )
    val_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=4040, end_idx=4540,
        returns=returns, return_scale=args.return_scale,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Data: {len(train_dataset)} train, {len(val_dataset)} val")

    # Precompute per-cell median/IQR for twCRPS
    if args.twcrps_beta > 0:
        train_surfaces = surfaces[0:4040]  # raw IV [0, 1]
        cell_median = torch.from_numpy(np.median(train_surfaces, axis=0)).float()
        q75 = torch.from_numpy(np.percentile(train_surfaces, 75, axis=0)).float()
        q25 = torch.from_numpy(np.percentile(train_surfaces, 25, axis=0)).float()
        cell_iqr = (q75 - q25).clamp(min=0.01)  # prevent division by tiny IQR
        model.cell_median.copy_(cell_median.to(device))
        model.cell_iqr.copy_(cell_iqr.to(device))
        print(f"  twCRPS beta={args.twcrps_beta}")
        print(f"  cell_median: [{cell_median.min():.3f}, {cell_median.max():.3f}]")
        print(f"  cell_iqr: [{cell_iqr.min():.3f}, {cell_iqr.max():.3f}]")

    # Precompute target kurtosis for kurtosis matching loss
    if args.lambda_kurt > 0:
        train_surfaces = surfaces[0:4040]
        daily_changes = np.diff(train_surfaces, axis=0)  # (N-1, 5, 5)
        m2 = (daily_changes ** 2).mean(axis=0)
        m4 = (daily_changes ** 4).mean(axis=0)
        target_kurt = torch.from_numpy(m4 / (m2 ** 2 + 1e-8)).float()
        model.target_kurt.copy_(target_kurt.to(device))
        print(f"  Kurtosis matching: lambda={args.lambda_kurt}")
        print(f"  Target kurtosis per cell: [{target_kurt.min():.1f}, {target_kurt.max():.1f}]")

    # Load GT cumulative variance targets for bilateral VR loss (Exp 117a)
    if args.lambda_vr > 0:
        import os
        gt_cv_path = os.path.join(os.path.dirname(__file__), '../../../data/gt_cumulative_variance.npz')
        if not os.path.exists(gt_cv_path):
            gt_cv_path = 'data/gt_cumulative_variance.npz'
        if os.path.exists(gt_cv_path):
            gt_cv = np.load(gt_cv_path)
            for h in [4, 9, 19, 29]:
                key = f'h{h}'
                if key in gt_cv:
                    getattr(model, f'gt_cum_var_h{h}').copy_(
                        torch.from_numpy(gt_cv[key]).float().to(device))
            print(f"  Loaded GT cumulative variance targets from {gt_cv_path}")
        else:
            print(f"  WARNING: GT cum var file not found, using defaults")

    print(f"\n{'='*70}")
    print(f"Training afCRPS single-pass model")
    print(f"  noise_dim={config.noise_dim}, n_members={args.n_members}, n_train_blocks={args.n_train_blocks}")
    print(f"  lambda_vs={args.lambda_vs}, from_scratch={args.from_scratch}")
    if args.ar_frame:
        print(f"  AR FRAME MODE: rho={config.ar_frame_rho}, hidden={config.ar_frame_hidden}, progressive={args.progressive_rollout}, encoder={'unfrozen' if args.unfreeze_encoder else 'frozen'}")
        if progressive_epoch_plan:
            schedule_desc = ", ".join(
                f"ep{stage['start_epoch']}-{stage['end_epoch']}=>{stage['n_frames']}"
                for stage in progressive_epoch_plan
            )
            print(f"  rollout schedule: {schedule_desc}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Progressive rollout: gradually increase generated frames
        if args.progressive_rollout:
            n_frames = resolve_progressive_frames(epoch, progressive_epoch_plan)
        else:
            n_frames = config.future_len if args.ar_frame else 0  # 0 = use block-based n_frames

        # Freeze-at-peak: freeze MLP after specified epoch, keep skip/scale trainable
        if args.freeze_after_epoch > 0 and epoch == args.freeze_after_epoch + 1:
            frozen_count = 0
            kept_count = 0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # Keep: noise_skip_proj (unless frozen), log_vol_scale, noise_scale_head, etc.
                keep = (
                    ("noise_skip_proj" in name and not args.freeze_skip_too)
                    or "log_vol_scale" in name
                    or "cell_scale" in name
                    or "noise_scale_head" in name
                    or "rho_head" in name
                    or "mr_mu_head" in name
                    or "mr_alpha_head" in name
                    or "spread_cell_proj" in name
                )
                if not args.freeze_spread_too:
                    keep = keep or "cell_spread_linear" in name or "cell_spread_factor" in name or "cell_spread_expand" in name
                should_freeze = name.startswith("frame_decoder.") or (
                    args.freeze_spread_too and ("cell_spread_linear" in name or "cell_spread_factor" in name or "cell_spread_expand" in name)
                ) or (
                    getattr(args, 'freeze_encoder_too', False) and name.startswith("encoder.")
                )
                if not keep and should_freeze:
                    param.requires_grad_(False)
                    frozen_count += 1
                else:
                    kept_count += 1
            n_still_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  >> FREEZE at epoch {epoch}: froze {frozen_count} params, kept {kept_count} trainable ({n_still_trainable:,} params)")

        # Curriculum noise: switch from gaussian to student_t at specified epoch
        curriculum_ep = getattr(args, 'curriculum_noise_epoch', 0)
        if curriculum_ep > 0 and epoch == curriculum_ep + 1 and model.config.noise_dist == 'gaussian':
            model.config.noise_dist = 'student_t'
            model.config.student_t_df = args.student_t_df
            print(f"  >> CURRICULUM NOISE: switched to student_t(df={args.student_t_df}) at epoch {epoch}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            n_members=args.n_members, lambda_vs=args.lambda_vs,
            grad_clip=args.grad_clip, n_train_blocks=args.n_train_blocks,
            lambda_is=args.lambda_is if epoch > getattr(args, 'is_warmup_epoch', 0) else 0.0,
            lambda_cs_reg=args.lambda_cs_reg,
            lambda_kurt=args.lambda_kurt, lambda_es=args.lambda_es,
            lambda_cell_var=args.lambda_cell_var,
            lambda_cum_cal=args.lambda_cum_cal,
            lambda_vr=args.lambda_vr,
            lambda_ortho=getattr(args, 'lambda_ortho', 0.0),
            lambda_ortho_enc=getattr(args, 'lambda_ortho_enc', 0.0),
            lambda_acf=getattr(args, 'lambda_acf', 0.0),
            lambda_rank=getattr(args, 'lambda_rank', 0.0),
            n_frames=n_frames,
            unfreeze_encoder=args.unfreeze_encoder,
        )
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, device, n_members=args.n_members)

        # Quick eval (CI coverage, kurtosis)
        eval_metrics = {}
        if epoch % args.eval_every == 0:
            eval_metrics = quick_eval(
                model, val_loader, device,
                n_samples=args.n_eval_samples, max_batches=5,
            )

        elapsed = time.time() - t0

        # Log
        log_entry = {
            "epoch": epoch,
            "elapsed": elapsed,
            **train_metrics,
            **val_metrics,
            **eval_metrics,
        }
        history_log.append(log_entry)

        # Print
        spread_ratio = train_metrics["spread_mae_ratio"]
        eval_str = ""
        if eval_metrics:
            eval_str = f"  CI={eval_metrics['coverage_90']:.1%}  Kurt={eval_metrics['kurtosis_ratio']:.3f}(mean:{eval_metrics['kurtosis_ratio_mean']:.3f})"
        es_str = (f"  es={train_metrics['energy_score']:.4f}"
                  f"(acc={train_metrics['es_accuracy']:.3f},spr={train_metrics['es_spread']:.3f})"
                  ) if args.lambda_es > 0 else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"mae={train_metrics['mae']:.4f}  "
            f"spread={train_metrics['spread']:.6f}  "
            f"s/m={spread_ratio:.4f}  "
            f"vs={train_metrics['variogram']:.4f}"
            f"{es_str}  "
            f"val={val_metrics['val_loss']:.4f}"
            f"{eval_str}  "
            f"({elapsed:.1f}s)"
        )

        # Log kurtosis matching if applicable
        if args.lambda_kurt > 0:
            print(f"  kurt_loss={train_metrics['kurt_loss']:.4f}  raw_kurt={train_metrics['raw_kurt']:.2f}")

        # Log frame_decoder stats if applicable
        if hasattr(model, 'frame_decoder'):
            if model.frame_decoder.mlp is not None:
                w = model.frame_decoder.mlp[-1].weight.detach()
                extra = f"  n_frames={n_frames}" if args.progressive_rollout else ""
                if hasattr(model.frame_decoder, 'cond_offsets') and model.frame_decoder.cond_offsets is not None:
                    off = model.frame_decoder.cond_offsets.detach()
                    extra += f"  offset_norm={off.norm():.3f}  offset_per_cell=[{off.norm(dim=1).min():.3f},{off.norm(dim=1).max():.3f}]"
                print(f"  frame_decoder: w_norm={w.norm():.3f}" + extra)
            elif hasattr(model.frame_decoder, 'cell_mlps'):
                w_norms = [mlp[-1].weight.detach().norm().item()
                           for mlp in model.frame_decoder.cell_mlps]
                extra = f"  n_frames={n_frames}" if args.progressive_rollout else ""
                print(f"  frame_decoder (indep): w_norm=[{min(w_norms):.3f},"
                      f"{max(w_norms):.3f}] mean={sum(w_norms)/len(w_norms):.3f}" + extra)

        # Log cell_scale stats if applicable (static per-cell scale)
        if hasattr(model, 'cell_scale'):
            cs = model.cell_scale.detach().clamp(0.3, 3.0)
            print(f"  cell_scale: [{cs.min():.3f}, {cs.max():.3f}] mean={cs.mean():.3f}")

        # Log bias loss if applicable
        if 'bias_loss' in train_metrics and train_metrics['bias_loss'] > 0:
            print(f"  bias_loss: {train_metrics['bias_loss']:.6f}")
        if 'cell_var_loss' in train_metrics and train_metrics['cell_var_loss'] > 0:
            print(f"  cell_var_loss: {train_metrics['cell_var_loss']:.4f}")
        if 'cum_cal_loss' in train_metrics and train_metrics['cum_cal_loss'] > 0:
            print(f"  cum_cal_loss: {train_metrics['cum_cal_loss']:.4f}")
        if 'eff_rank' in train_metrics and train_metrics['eff_rank'] > 0:
            print(f"  rank_loss={train_metrics['rank_loss']:.4f}  eff_rank={train_metrics['eff_rank']:.3f}")

        # Log mean |IV change| for logit-space monitoring (epochs 1,5,10,15,20,25,30,35,40)
        if getattr(config, 'ar_frame_logit_space', False) and epoch in {1, 5, 10, 15, 20, 25, 30, 35, 40}:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                hist = sample_batch["history"].to(device)
                prev = denormalize_iv(hist[:, -1])  # (B, 5, 5)
                s = model.sample(hist, n_samples=4, n_frames=5)  # (B, 4, 5, 5, 5)
                iv_changes = (s[:, :, 0, :, :] - prev.unsqueeze(1)).abs().mean(dim=(0, 1))  # (5, 5)
                row_means = iv_changes.mean(dim=1)
                print(f"  logit_monitor: mean|dIV| per row: [{', '.join(f'{row_means[r]:.5f}' for r in range(5))}]")

        # Cross-cell correlation diagnostic at key epochs (Exp 98a / 99k)
        if (args.ar_independent_cells or args.lambda_es > 0) and epoch in {1, 5, 10, 20, 40}:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                hist = sample_batch["history"].to(device)
                prev = denormalize_iv(hist[:, -1])  # (B, 5, 5)
                n_diag = 20
                all_d = []
                for _ in range(n_diag):
                    s = model.sample(hist, n_samples=1, n_frames=1)  # (B, 1, 1, 5, 5)
                    all_d.append((s[:, 0, 0] - prev).reshape(-1, 25))
                deltas = torch.cat(all_d, dim=0)  # (n_diag*B, 25)
                cell_std = deltas.std(dim=0)
                deltas_c = deltas - deltas.mean(dim=0, keepdim=True)
                cov = (deltas_c.T @ deltas_c) / (deltas_c.shape[0] - 1)
                std_out = cell_std.unsqueeze(0) * cell_std.unsqueeze(1) + 1e-8
                corr = cov / std_out
                mask = ~torch.eye(25, dtype=torch.bool, device=device)
                mean_corr = corr[mask].mean().item()
                # Effective rank via participation ratio
                eigvals = torch.linalg.eigvalsh(cov)
                eigvals = eigvals.clamp(min=0)
                pr = (eigvals.sum()**2) / (eigvals.pow(2).sum() + 1e-12)
                pc1 = eigvals[-1] / (eigvals.sum() + 1e-12)
                print(f"  [DIAG] cell_std: [{cell_std.min():.5f}, {cell_std.max():.5f}]"
                      f" mean={cell_std.mean():.5f}")
                print(f"  [DIAG] cross-cell corr: {mean_corr:.3f} (GT ~0.38)"
                      f"  eff_rank: {pr.item():.2f} (GT ~2.6)"
                      f"  PC1: {pc1.item()*100:.1f}% (GT ~59%)")

        # Log cell_spread_linear stats if applicable (AR frame mode)
        if hasattr(model, 'cell_spread_linear') and model.cell_spread_linear is not None:
            w = model.cell_spread_linear.weight.detach()
            b = model.cell_spread_linear.bias.detach()
            base_out = F.softplus(b)
            print(f"  cell_spread: out=[{base_out.min():.3f}, {base_out.max():.3f}] w_norm={w.norm():.3f}")
        elif hasattr(model, 'cell_spread_expand') and model.cell_spread_expand is not None:
            w = model.cell_spread_expand.weight.detach()
            b = model.cell_spread_expand.bias.detach()
            base_out = F.softplus(b)
            print(f"  cell_spread(LR): out=[{base_out.min():.3f}, {base_out.max():.3f}] w_norm={w.norm():.3f}")

        if hasattr(model, 'noise_scale_head'):
            w = model.noise_scale_head.weight.detach()
            b = model.noise_scale_head.bias.detach()
            base_out = F.softplus(b)
            print(f"  noise_scale: out=[{base_out.min():.3f}, {base_out.max():.3f}] "
                  f"range={base_out.max()/base_out.min():.1f}x w_norm={w.norm():.3f}")

        if hasattr(model, 'rho_head'):
            b = model.rho_head.bias.detach()
            base_rho = torch.sigmoid(b).item()
            w_norm = model.rho_head.weight.detach().norm().item()
            print(f"  learned_rho: base={base_rho:.4f} w_norm={w_norm:.3f}")

        if hasattr(model, 'mr_mu_head'):
            mu_b = torch.sigmoid(model.mr_mu_head.bias.detach())
            alpha_raw = torch.sigmoid(model.mr_alpha_head.bias.detach())
            alpha_cap = 0.05 if model.config.ar_mean_revert_percell else 0.2
            alpha_b = alpha_cap * alpha_raw
            mu_w = model.mr_mu_head.weight.detach().norm().item()
            if alpha_b.numel() == 1:
                print(f"  mean_revert: alpha={alpha_b.item():.4f} mu=[{mu_b.min():.3f}, {mu_b.max():.3f}] "
                      f"w_norm={mu_w:.3f}")
            else:
                print(f"  mean_revert: alpha=[{alpha_b.min():.4f}, {alpha_b.max():.4f}] "
                      f"mu=[{mu_b.min():.3f}, {mu_b.max():.3f}] w_norm={mu_w:.3f}")

        # Log cell_spread MLP stats if applicable
        if hasattr(model, 'cell_spread_mlp'):
            # Find the last Linear layer in the Sequential
            linear_layers = [m for m in model.cell_spread_mlp if isinstance(m, nn.Linear)]
            if linear_layers:
                w = linear_layers[-1].weight.detach()
                b = linear_layers[-1].bias.detach()
                base_out = F.softplus(b)
                print(f"  cell_spread: bias_out=[{base_out.min():.3f}, {base_out.max():.3f}] w_norm={w.norm():.3f}")

        # Save checkpoint dict for potential saving
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
            "config": dataclasses.asdict(config),
            "metrics": log_entry,
            "training_config": {
                **vars(args),
                "parsed_progressive_schedule": progressive_schedule,
                "progressive_epoch_plan": progressive_epoch_plan,
                "resolved_future_len": resolved_future_len,
                "pretrained_init_mode": pretrained_init_mode,
                "resolved_device": str(device),
                "base_model_resolved": (
                    str(Path(args.base_model).resolve()) if args.base_model else None
                ),
                "base_model_hash": base_model_hash,
                "base_checkpoint_epoch": (
                    base_ckpt.get("epoch") if base_ckpt is not None else None
                ),
            },
        }

        # Early stopping checks (only on member kurtosis, and only if very low)
        if (not args.disable_early_stop
                and eval_metrics.get("kurtosis_ratio", 1.0) < 0.1
                and epoch >= 10):
            print(f"EARLY STOP: member kurtosis {eval_metrics['kurtosis_ratio']:.3f} < 0.1 at epoch {epoch}")
            break
        if not args.disable_early_stop and epoch >= 5 and spread_ratio < 0.05:
            print(f"WARNING: spread/MAE ratio {spread_ratio:.4f} < 0.05 — noise injection may not be working")

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(save_dict, f"{args.output_dir}/best_model.pt")
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        if eval_metrics.get("coverage_90", 0) > best_coverage:
            best_coverage = eval_metrics["coverage_90"]
            torch.save(save_dict, f"{args.output_dir}/best_coverage_model.pt")
            print(f"  → Saved best coverage model (coverage={best_coverage:.1%})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(save_dict, f"{args.output_dir}/checkpoint_epoch_{epoch}.pt")

    # Save final
    torch.save(save_dict, f"{args.output_dir}/final_model.pt")

    # Save training history
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}, best coverage={best_coverage:.1%}")
    print(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
