#!/usr/bin/env python
"""
167a Gradient Analysis: Verify claim that CRPS doesn't provide enough gradient
to load_head, causing L_norm to die over training.

Loads checkpoint_epoch_20.pt (L active, L_norm~0.08) and checkpoint_epoch_60.pt
(L dying, L_norm~0.01), runs one forward+backward pass on a batch of training data,
and measures gradient magnitudes to each component.

Also computes: weight_decay_removal vs gradient_addition for load_head to determine
if weight decay alone explains L dying.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig


# ====== Inline model definitions (from train_167a_factorized.py) ======

def normalize_iv(surfaces):
    return surfaces * 2.0 - 1.0

def denormalize_iv(surfaces):
    return (surfaces + 1.0) / 2.0


class ConditionalNorm(nn.Module):
    def __init__(self, d_model, noise_dim, n_cells=25):
        super().__init__()
        self.n_cells = n_cells
        self.d_model = d_model
        self.scale_proj = nn.Linear(noise_dim, n_cells * d_model)
        self.bias_proj = nn.Linear(noise_dim, n_cells * d_model)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, x, z):
        B = z.shape[0]
        scale = self.scale_proj(z).view(B, self.n_cells, self.d_model)
        bias = self.bias_proj(z).view(B, self.n_cells, self.d_model)
        return (scale + 1.0) * x + bias


class SpatialTransformerDecoder(nn.Module):
    def __init__(self, n_cells=25, d_model=128, n_heads=4, n_layers=4,
                 cond_dim=128, noise_dim=32):
        super().__init__()
        self.n_cells = n_cells
        self.d_model = d_model
        self.noise_dim = noise_dim

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.SiLU(), nn.Linear(d_model, noise_dim),
        )

        self.layers = nn.ModuleList()
        self.ls_params = nn.ParameterList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'cln': ConditionalNorm(d_model, noise_dim, n_cells),
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'ff_cln': ConditionalNorm(d_model, noise_dim, n_cells),
            }))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))

        self.output_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._he_init()

    def _he_init(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'output_proj' in name or 'scale_proj' in name or 'bias_proj' in name:
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_trunk(self, cond, prev_frame, noise):
        h = self.input_proj(prev_frame.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos
        z = self.noise_proj(noise)

        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[li * 2]
            ls_f = self.ls_params[li * 2 + 1]
            h_norm = layer['cln'](h, z)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out
            h = h + ls_f * layer['ff'](layer['ff_cln'](h, z))

        return h

    def forward(self, cond, prev_frame, noise):
        h = self.forward_trunk(cond, prev_frame, noise)
        delta = self.output_proj(h).squeeze(-1)
        return delta


class FactorizedSpatialTransformerDecoder(SpatialTransformerDecoder):
    def __init__(self, n_cells=25, d_model=128, n_heads=4, n_layers=4,
                 cond_dim=128, noise_dim=32, n_factors=5):
        super().__init__(n_cells=n_cells, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, cond_dim=cond_dim, noise_dim=noise_dim)
        self.n_factors = n_factors

        self.base_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)

        self.load_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, n_factors),
        )
        nn.init.normal_(self.load_head[-1].weight, std=0.01)
        nn.init.zeros_(self.load_head[-1].bias)

        self.cond_resid_film = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )

        self.register_buffer('cond_ref', torch.zeros(cond_dim))

    def forward(self, cond, prev_frame, noise):
        h = self.forward_trunk(cond, prev_frame, noise)
        delta_base = self.base_head(h).squeeze(-1)

        cond_resid = cond - self.cond_ref
        film_params = self.cond_resid_film(cond_resid)
        gamma, beta = film_params.chunk(2, dim=-1)
        h_modulated = gamma.unsqueeze(1) * h + beta.unsqueeze(1)

        L = self.load_head(h_modulated)
        return delta_base, L


class ARFactorizedTransformerModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, n_factors=5):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = FactorizedSpatialTransformerDecoder(**decoder_config, n_factors=n_factors)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_factors = n_factors

    def encode(self, history):
        return self.encoder(history)


# ====== Loss functions (from train_167a_factorized.py) ======

def afcrps_per_frame(samples, gt, alpha=0.95, spread_weight=0.5):
    K = samples.shape[1]
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)
    mae = (samples - gt.unsqueeze(1)).abs().mean()
    spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()
    fcrps = mae - spread_weight * spread
    loss = alpha * fcrps + (1 - alpha) * mae
    return loss, mae, spread


def variogram_score_per_frame(samples, gt, p=0.5):
    B, K, C = samples.shape
    eps = 1e-8
    s_diff = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    s_mean = s_diff.mean(dim=1)
    g_diff = (gt.unsqueeze(-1) - gt.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    loss = (g_diff - s_mean).pow(2)
    mask = torch.triu(torch.ones(C, C, device=samples.device), diagonal=1).bool()
    return loss[:, mask].mean()


def interval_score(samples, gt, alpha=0.1):
    lo = samples.quantile(alpha / 2, dim=1)
    hi = samples.quantile(1 - alpha / 2, dim=1)
    width = hi - lo
    return (width + (2 / alpha) * (F.relu(lo - gt) + F.relu(gt - hi))).mean()


def reflecting_boundary(x, lo=0.01, hi=1.0):
    below = x < lo
    above = x > hi
    x = torch.where(below, 2 * lo - x, x)
    x = torch.where(above, 2 * hi - x, x)
    return x.clamp(lo, hi)


# ====== Gradient measurement utilities ======

def get_grad_norm(param):
    """Get gradient L2 norm of a parameter, or None if no grad."""
    if param.grad is None:
        return None
    return param.grad.norm().item()


def get_grad_stats(param, name):
    """Get detailed gradient stats for a parameter."""
    if param.grad is None:
        return {"name": name, "grad_norm": None, "grad_mean": None, "grad_std": None,
                "grad_abs_mean": None, "param_norm": param.data.norm().item(),
                "param_numel": param.numel(), "has_grad": False}
    g = param.grad
    return {
        "name": name,
        "grad_norm": g.norm().item(),
        "grad_mean": g.mean().item(),
        "grad_std": g.std().item(),
        "grad_abs_mean": g.abs().mean().item(),
        "grad_max": g.abs().max().item(),
        "param_norm": param.data.norm().item(),
        "param_numel": param.numel(),
        "has_grad": True,
        "grad_to_param_ratio": g.norm().item() / max(param.data.norm().item(), 1e-12),
    }


def run_one_training_step(model, train_hist, train_future, train_last, device,
                          n_members=16, noise_dim=32, n_factors=5,
                          lambda_vs=1.0, lambda_is=0.005, lambda_floor=2.5,
                          floor_tau=0.005, bptt_steps=5, batch_size=16):
    """Run exactly one forward+backward pass on training data, matching train loop."""
    H, T, C = 30, 30, 25
    model.train()

    # Use first batch_size windows
    hist = train_hist[:batch_size]
    gt_frames = train_future[:batch_size]
    last_frame = train_last[:batch_size]

    B = hist.shape[0]
    K = n_members

    # Forward pass (encoder)
    hist_norm = normalize_iv(hist)
    hist_flat = hist_norm.reshape(B, H, C)
    gru_outputs, h_last = model.encoder.gru(hist_flat)

    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    cond = model.encoder.bottleneck(h_pooled)

    cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    prev = last_frame.unsqueeze(1).expand(B, K, C).reshape(B * K, C)

    h_gru = h_last.detach().unsqueeze(2).expand(1, B, K, -1).reshape(1, B * K, -1).contiguous()
    gru_outs = gru_outputs.detach().unsqueeze(1).expand(B, K, H, -1).reshape(B * K, H, -1)

    N = bptt_steps

    # Zero all grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    window_loss = 0

    for t in range(T):
        z_t = torch.randn(B * K, noise_dim, device=device)

        if t > 0:
            attn_logits_t = model.encoder.attn_proj(gru_outs).squeeze(-1)
            attn_weights_t = F.softmax(attn_logits_t, dim=1)
            h_pooled_t = (attn_weights_t.unsqueeze(-1) * gru_outs).sum(dim=1)
            cond_K = model.encoder.bottleneck(h_pooled_t)

        # Factorized decoder output
        delta_base, L = model.decoder(cond_K, prev, z_t)
        eps = torch.randn(B * K, n_factors, device=device)
        delta = delta_base + torch.einsum("bcr,br->bc", L, eps)
        frame_t = prev + torch.tanh(delta)

        # Per-frame losses
        frame_BK = frame_t.reshape(B, K, C)
        gt_t = gt_frames[:, t, :]

        loss_t, mae_t, spread_t = afcrps_per_frame(frame_BK, gt_t, alpha=0.95, spread_weight=0.5)
        is_t = interval_score(frame_BK, gt_t)
        vs_t = variogram_score_per_frame(frame_BK, gt_t)

        floor_barrier = floor_tau * F.softplus(-frame_t / floor_tau).mean()

        step_loss = (loss_t + lambda_is * is_t + lambda_vs * vs_t + lambda_floor * floor_barrier) / T
        window_loss = window_loss + step_loss

        is_window_end = ((t + 1) % N == 0) or (t == T - 1)
        if is_window_end:
            window_loss.backward()
            window_loss = 0
            prev = frame_t.detach()
            h_gru = h_gru.detach()
            gru_outs = gru_outs.detach()
        else:
            prev = frame_t

        frame_norm = normalize_iv(prev.detach()).unsqueeze(1)
        gru_out, h_gru = model.encoder.gru(frame_norm, h_gru)
        gru_outs = torch.cat([gru_outs, gru_out], dim=1)

    # DO NOT clip gradients -- we want raw gradient magnitudes
    return model


def analyze_checkpoint(ckpt_path, device, train_hist, train_future, train_last,
                       n_members, noise_dim, n_factors):
    """Load a checkpoint, run one step, measure all gradients."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {ckpt_path}")
    print(f"{'='*70}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    encoder_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1, cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_cells=25, d_model=config["decoder"]["d_model"],
        n_heads=config["decoder"]["n_heads"], n_layers=config["decoder"]["n_layers"],
        cond_dim=128, noise_dim=config["decoder"]["noise_dim"],
    )

    model = ARFactorizedTransformerModel(encoder_config, decoder_config,
                                          n_factors=config["n_factors"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    epoch = ckpt["epoch"]
    val_loss = ckpt["val_loss"]
    print(f"  Epoch: {epoch}, Val loss: {val_loss:.6f}")

    # Measure L_norm before step
    model.eval()
    with torch.no_grad():
        B_diag = 16
        hist_diag = normalize_iv(train_hist[:B_diag])
        hist_flat = hist_diag.reshape(B_diag, 30, 25)
        gru_outs, _ = model.encoder.gru(hist_flat)
        attn = F.softmax(model.encoder.attn_proj(gru_outs).squeeze(-1), dim=1)
        cond_diag = model.encoder.bottleneck((attn.unsqueeze(-1) * gru_outs).sum(dim=1))
        z_diag = torch.randn(B_diag, noise_dim, device=device)
        _, L_diag = model.decoder(cond_diag, train_last[:B_diag], z_diag)
        L_norm_before = L_diag.norm().item() / B_diag
        L_std_before = L_diag.std().item()
        L_fro_per_sample = (L_diag ** 2).sum(dim=(1, 2)).sqrt().mean().item()
    print(f"  L_norm (per sample): {L_norm_before:.6f}")
    print(f"  L_std: {L_std_before:.6f}")
    print(f"  L_fro (per sample mean): {L_fro_per_sample:.6f}")

    # Run one training step
    model = run_one_training_step(
        model, train_hist, train_future, train_last, device,
        n_members=n_members, noise_dim=noise_dim, n_factors=n_factors,
        lambda_vs=1.0, lambda_is=0.005, lambda_floor=2.5,
        floor_tau=0.005, bptt_steps=5, batch_size=16,
    )

    # Collect gradient stats for all key parameters
    results = {
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "val_loss": val_loss,
        "L_norm_before_step": L_norm_before,
        "L_std_before_step": L_std_before,
        "L_fro_per_sample": L_fro_per_sample,
        "gradients": {},
    }

    # --- Key parameter groups ---
    param_groups = {}

    # 1. base_head
    param_groups["base_head.weight"] = model.decoder.base_head.weight
    param_groups["base_head.bias"] = model.decoder.base_head.bias

    # 2. load_head (final Linear layer)
    param_groups["load_head[-1].weight"] = model.decoder.load_head[-1].weight
    param_groups["load_head[-1].bias"] = model.decoder.load_head[-1].bias

    # 3. load_head (first Linear layer)
    param_groups["load_head[0].weight"] = model.decoder.load_head[0].weight
    param_groups["load_head[0].bias"] = model.decoder.load_head[0].bias

    # 4. cond_resid_film (final Linear)
    param_groups["cond_resid_film[-1].weight"] = model.decoder.cond_resid_film[-1].weight
    param_groups["cond_resid_film[-1].bias"] = model.decoder.cond_resid_film[-1].bias

    # 5. cond_resid_film (first Linear)
    param_groups["cond_resid_film[0].weight"] = model.decoder.cond_resid_film[0].weight
    param_groups["cond_resid_film[0].bias"] = model.decoder.cond_resid_film[0].bias

    # 6. output_proj (original, for comparison)
    param_groups["output_proj.weight"] = model.decoder.output_proj.weight
    param_groups["output_proj.bias"] = model.decoder.output_proj.bias

    # 7. trunk CLN scale_proj (layer 0)
    param_groups["layers[0].cln.scale_proj.weight"] = model.decoder.layers[0]['cln'].scale_proj.weight
    param_groups["layers[0].cln.scale_proj.bias"] = model.decoder.layers[0]['cln'].scale_proj.bias

    # 8. trunk CLN scale_proj (last layer)
    n_layers = len(model.decoder.layers)
    param_groups[f"layers[{n_layers-1}].cln.scale_proj.weight"] = model.decoder.layers[n_layers-1]['cln'].scale_proj.weight

    # 9. input_proj
    param_groups["input_proj.weight"] = model.decoder.input_proj.weight

    # 10. cond_proj (first linear)
    param_groups["cond_proj[0].weight"] = model.decoder.cond_proj[0].weight

    # 11. noise_proj (first linear)
    param_groups["noise_proj[0].weight"] = model.decoder.noise_proj[0].weight

    # Collect all stats
    for name, param in param_groups.items():
        stats = get_grad_stats(param, name)
        results["gradients"][name] = stats
        grad_norm = stats["grad_norm"]
        param_norm = stats["param_norm"]
        if grad_norm is not None:
            print(f"  {name:45s}  grad_norm={grad_norm:.6e}  param_norm={param_norm:.6e}  ratio={stats['grad_to_param_ratio']:.6e}")
        else:
            print(f"  {name:45s}  NO GRADIENT  param_norm={param_norm:.6e}")

    # --- Aggregate group norms ---
    group_norms = {}

    # base_head total
    base_grad_sq = sum(p.grad.norm().item() ** 2
                       for p in model.decoder.base_head.parameters()
                       if p.grad is not None)
    group_norms["base_head_total"] = base_grad_sq ** 0.5

    # load_head total
    load_grad_sq = sum(p.grad.norm().item() ** 2
                       for p in model.decoder.load_head.parameters()
                       if p.grad is not None)
    group_norms["load_head_total"] = load_grad_sq ** 0.5

    # cond_resid_film total
    film_grad_sq = sum(p.grad.norm().item() ** 2
                       for p in model.decoder.cond_resid_film.parameters()
                       if p.grad is not None)
    group_norms["cond_resid_film_total"] = film_grad_sq ** 0.5

    # trunk total (all CLN params)
    trunk_cln_sq = 0
    for layer in model.decoder.layers:
        for cln_name in ['cln', 'ff_cln']:
            for p in layer[cln_name].parameters():
                if p.grad is not None:
                    trunk_cln_sq += p.grad.norm().item() ** 2
    group_norms["trunk_cln_total"] = trunk_cln_sq ** 0.5

    # attention total
    attn_sq = 0
    for layer in model.decoder.layers:
        for p in layer['attn'].parameters():
            if p.grad is not None:
                attn_sq += p.grad.norm().item() ** 2
    group_norms["trunk_attn_total"] = attn_sq ** 0.5

    # FF total
    ff_sq = 0
    for layer in model.decoder.layers:
        for p in layer['ff'].parameters():
            if p.grad is not None:
                ff_sq += p.grad.norm().item() ** 2
    group_norms["trunk_ff_total"] = ff_sq ** 0.5

    results["group_norms"] = group_norms

    print(f"\n  Group norms:")
    for name, norm in group_norms.items():
        print(f"    {name:30s}: {norm:.6e}")

    # --- Weight decay analysis for load_head ---
    wd_rate = 0.01  # AdamW weight_decay
    lr = 1e-3  # base lr from training

    wd_analysis = {}
    for pname in ["load_head[-1].weight", "load_head[-1].bias",
                   "load_head[0].weight", "load_head[0].bias"]:
        param = param_groups[pname]
        param_norm = param.data.norm().item()
        grad_norm = param.grad.norm().item() if param.grad is not None else 0.0

        # AdamW: param -= lr * (adam_update + wd * param)
        # Weight decay removal per step ~ lr * wd * param_norm
        wd_removal = lr * wd_rate * param_norm

        # Gradient addition per step (rough estimate: lr * grad_norm / sqrt(numel))
        # In Adam, the effective step is lr * grad / (sqrt(v) + eps)
        # For a rough bound: lr * grad_abs_mean
        grad_abs_mean = param.grad.abs().mean().item() if param.grad is not None else 0.0
        grad_addition_rough = lr * grad_abs_mean

        wd_analysis[pname] = {
            "param_norm": param_norm,
            "grad_norm": grad_norm,
            "grad_abs_mean": grad_abs_mean,
            "wd_removal_per_step": wd_removal,
            "grad_addition_rough": grad_addition_rough,
            "wd_dominates": wd_removal > grad_addition_rough,
            "wd_to_grad_ratio": wd_removal / max(grad_addition_rough, 1e-15),
        }

    results["weight_decay_analysis"] = wd_analysis

    print(f"\n  Weight decay analysis (lr={lr}, wd={wd_rate}):")
    for pname, wda in wd_analysis.items():
        print(f"    {pname}:")
        print(f"      param_norm={wda['param_norm']:.6e}")
        print(f"      wd_removal/step={wda['wd_removal_per_step']:.6e}")
        print(f"      grad_addition/step={wda['grad_addition_rough']:.6e}")
        print(f"      wd_dominates={wda['wd_dominates']}")
        print(f"      wd/grad ratio={wda['wd_to_grad_ratio']:.4f}")

    # Also measure: per-sample gradient of loss w.r.t. L output
    # This tells us about the signal strength at the *output* of load_head
    print(f"\n  Additional: L output statistics from diagnostic batch")
    print(f"    L_norm (per sample avg): {L_norm_before:.6f}")
    print(f"    L_std: {L_std_before:.6f}")

    return results


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def main():
    device = "cuda"
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("Loading data...")
    data = np.load("/home/max/Documents/vol-surface-vae-pub/data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25

    TEST_START = 4511
    max_train_idx = TEST_START - H - T
    VAL_SIZE = 441
    train_indices = np.arange(0, max_train_idx - VAL_SIZE)

    surf_tensor = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    # Build training windows
    idx = torch.from_numpy(train_indices).long().to(device)
    offsets_h = torch.arange(H, device=device).unsqueeze(0)
    offsets_f = torch.arange(T, device=device).unsqueeze(0)
    hist_idx = idx.unsqueeze(1) + offsets_h
    fut_idx = idx.unsqueeze(1) + H + offsets_f
    train_hist = surf_tensor[hist_idx]
    train_future = surf_tensor[fut_idx].reshape(len(train_indices), T, C)
    train_last = surf_tensor[idx + H - 1].reshape(len(train_indices), C)

    print(f"Train windows: {len(train_indices)}")
    print(f"Using first 16 for gradient analysis")

    # Analyze both checkpoints
    base_dir = "/home/max/Documents/vol-surface-vae-pub/models/backfill/afcrps_167a"

    results_all = {}
    for ckpt_name in ["checkpoint_epoch_20", "checkpoint_epoch_60"]:
        ckpt_path = f"{base_dir}/{ckpt_name}.pt"
        results = analyze_checkpoint(
            ckpt_path, device, train_hist, train_future, train_last,
            n_members=16, noise_dim=32, n_factors=5,
        )
        results_all[ckpt_name] = results

    # --- Comparison summary ---
    print(f"\n{'='*70}")
    print(f"COMPARISON: epoch 20 vs epoch 60")
    print(f"{'='*70}")

    ep20 = results_all["checkpoint_epoch_20"]
    ep60 = results_all["checkpoint_epoch_60"]

    key_params = [
        "base_head.weight", "base_head.bias",
        "load_head[-1].weight", "load_head[-1].bias",
        "load_head[0].weight", "load_head[0].bias",
        "cond_resid_film[-1].weight",
        "layers[0].cln.scale_proj.weight",
    ]

    comparison = {}
    print(f"\n{'Parameter':45s}  {'Ep20 grad':>12s}  {'Ep60 grad':>12s}  {'Ratio':>10s}  {'Ep20 param':>12s}  {'Ep60 param':>12s}")
    print("-" * 120)
    for p in key_params:
        g20 = ep20["gradients"][p]["grad_norm"]
        g60 = ep60["gradients"][p]["grad_norm"]
        p20 = ep20["gradients"][p]["param_norm"]
        p60 = ep60["gradients"][p]["param_norm"]
        ratio = (g60 / g20) if (g20 and g20 > 0) else None
        ratio_str = f"{ratio:.4f}" if ratio else "N/A"
        print(f"{p:45s}  {g20:12.6e}  {g60:12.6e}  {ratio_str:>10s}  {p20:12.6e}  {p60:12.6e}")
        comparison[p] = {
            "ep20_grad": g20, "ep60_grad": g60, "ratio_60_over_20": ratio,
            "ep20_param": p20, "ep60_param": p60,
        }

    # Group comparison
    print(f"\n{'Group':30s}  {'Ep20':>12s}  {'Ep60':>12s}  {'Ratio':>10s}")
    print("-" * 70)
    group_comparison = {}
    for g in ep20["group_norms"]:
        n20 = ep20["group_norms"][g]
        n60 = ep60["group_norms"][g]
        ratio = n60 / n20 if n20 > 0 else None
        ratio_str = f"{ratio:.4f}" if ratio else "N/A"
        print(f"{g:30s}  {n20:12.6e}  {n60:12.6e}  {ratio_str:>10s}")
        group_comparison[g] = {"ep20": n20, "ep60": n60, "ratio": ratio}

    # Key question: base_head vs load_head gradient ratio
    base_grad_20 = ep20["group_norms"]["base_head_total"]
    load_grad_20 = ep20["group_norms"]["load_head_total"]
    base_grad_60 = ep60["group_norms"]["base_head_total"]
    load_grad_60 = ep60["group_norms"]["load_head_total"]

    print(f"\n--- KEY FINDING ---")
    print(f"Ep20: base_head grad / load_head grad = {base_grad_20 / max(load_grad_20, 1e-15):.4f}")
    print(f"Ep60: base_head grad / load_head grad = {base_grad_60 / max(load_grad_60, 1e-15):.4f}")
    print(f"Ep20: load_head grad magnitude: {load_grad_20:.6e}")
    print(f"Ep60: load_head grad magnitude: {load_grad_60:.6e}")

    # Weight decay verdict
    wd20 = ep20["weight_decay_analysis"]
    wd60 = ep60["weight_decay_analysis"]
    print(f"\n--- WEIGHT DECAY VERDICT ---")
    for pname in ["load_head[-1].weight", "load_head[0].weight"]:
        print(f"\n{pname}:")
        print(f"  Ep20: wd_removal={wd20[pname]['wd_removal_per_step']:.6e}, grad_add={wd20[pname]['grad_addition_rough']:.6e}, wd_dominates={wd20[pname]['wd_dominates']}")
        print(f"  Ep60: wd_removal={wd60[pname]['wd_removal_per_step']:.6e}, grad_add={wd60[pname]['grad_addition_rough']:.6e}, wd_dominates={wd60[pname]['wd_dominates']}")

    # Overall verdict
    claim_verified = (load_grad_20 < base_grad_20) or any(
        wd20[k]["wd_dominates"] for k in wd20
    )

    verdict = {
        "claim": "CRPS doesn't provide enough gradient signal to load_head, causing L_norm to die",
        "gradient_starvation": load_grad_20 < base_grad_20 * 0.1,  # >10x weaker
        "weight_decay_kills": any(wd60[k]["wd_dominates"] for k in wd60),
        "base_to_load_ratio_ep20": base_grad_20 / max(load_grad_20, 1e-15),
        "base_to_load_ratio_ep60": base_grad_60 / max(load_grad_60, 1e-15),
        "L_norm_ep20": ep20["L_norm_before_step"],
        "L_norm_ep60": ep60["L_norm_before_step"],
        "L_norm_decay_ratio": ep60["L_norm_before_step"] / max(ep20["L_norm_before_step"], 1e-15),
    }

    # Determine mechanism
    if verdict["weight_decay_kills"] and verdict["gradient_starvation"]:
        verdict["mechanism"] = "BOTH: gradient starvation + weight decay dominance"
        verdict["verified"] = True
    elif verdict["weight_decay_kills"]:
        verdict["mechanism"] = "Weight decay dominance (removal > gradient addition)"
        verdict["verified"] = True
    elif verdict["gradient_starvation"]:
        verdict["mechanism"] = "Gradient starvation (load_head gets >10x less gradient than base_head)"
        verdict["verified"] = True
    else:
        verdict["mechanism"] = "Neither gradient starvation nor weight decay explains L dying -- claim NOT verified"
        verdict["verified"] = False

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict['mechanism']}")
    print(f"Claim verified: {verdict['verified']}")
    print(f"{'='*70}")

    # Save results
    output_dir = "/home/max/Documents/vol-surface-vae-pub/results/validations/2026-04-04"

    # 1. Full gradient comparison JSON
    full_results = {
        "metadata": {
            "script": "167a_gradient_analysis.py",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "batch_size": 16,
            "n_members": 16,
            "noise_dim": 32,
            "n_factors": 5,
            "loss_config": {
                "lambda_vs": 1.0,
                "lambda_is": 0.005,
                "lambda_floor": 2.5,
                "floor_tau": 0.005,
                "bptt_steps": 5,
            },
        },
        "checkpoints": {
            "epoch_20": results_all["checkpoint_epoch_20"],
            "epoch_60": results_all["checkpoint_epoch_60"],
        },
        "comparison": comparison,
        "group_comparison": group_comparison,
        "verdict": verdict,
    }
    full_results = make_serializable(full_results)

    with open(f"{output_dir}/analysis/167a_gradient/gradient_comparison.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nSaved: {output_dir}/analysis/167a_gradient/gradient_comparison.json")

    # 2. Verification result JSON
    verification = {
        "experiment": "167a_factorized",
        "claim": "CRPS doesn't provide enough gradient signal to load_head, causing L_norm to die over training",
        "verified": verdict["verified"],
        "mechanism": verdict["mechanism"],
        "evidence": {
            "gradient_starvation": verdict["gradient_starvation"],
            "weight_decay_kills": verdict["weight_decay_kills"],
            "base_to_load_ratio_ep20": verdict["base_to_load_ratio_ep20"],
            "base_to_load_ratio_ep60": verdict["base_to_load_ratio_ep60"],
            "L_norm_ep20": verdict["L_norm_ep20"],
            "L_norm_ep60": verdict["L_norm_ep60"],
            "L_norm_decay_ratio": verdict["L_norm_decay_ratio"],
        },
        "weight_decay_detail": {
            "load_head_final_ep20": wd20["load_head[-1].weight"],
            "load_head_final_ep60": wd60["load_head[-1].weight"],
            "load_head_first_ep20": wd20["load_head[0].weight"],
            "load_head_first_ep60": wd60["load_head[0].weight"],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    verification = make_serializable(verification)

    with open(f"{output_dir}/verification_results/167a_gradient.json", "w") as f:
        json.dump(verification, f, indent=2, default=str)
    print(f"Saved: {output_dir}/verification_results/167a_gradient.json")


if __name__ == "__main__":
    main()
