#!/usr/bin/env python
"""
167d S3 Worst Cell Diagnostic

Identifies which cell fails S3 conditionality in 167d and analyzes the
FiLM modulation, loading magnitudes, and conditional widths to determine
the root cause of the CLN + factorized failure pattern.

Grid layout: 5x5 (moneyness rows x tenor cols)
  Rows: deep OTM put -> ATM -> deep OTM call  (moneyness)
  Cols: short tenor -> long tenor

Outputs:
  - results/validations/2026-04-04/analysis/167d_followup/s3_worst_cell.json
  - results/validations/2026-04-04/verification_results/167d_s3_worst_cell.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig

# --- Import model classes from training script ---
# Re-define here to avoid import issues

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


class FactorizedDecoderClean(SpatialTransformerDecoder):
    def __init__(self, n_cells=25, d_model=128, n_heads=4, n_layers=4,
                 cond_dim=128, noise_dim=32, n_factors=5):
        super().__init__(n_cells=n_cells, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, cond_dim=cond_dim, noise_dim=noise_dim)
        self.n_factors = n_factors
        self.base_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)
        self.load_head = nn.Linear(d_model, n_factors)
        nn.init.normal_(self.load_head.weight, std=0.01)
        nn.init.zeros_(self.load_head.bias)
        self.cond_resid_film = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        nn.init.zeros_(self.cond_resid_film[-1].weight)
        nn.init.zeros_(self.cond_resid_film[-1].bias)
        self.register_buffer('cond_ref', torch.zeros(cond_dim))

    def forward(self, cond, prev_frame, noise):
        h = self.forward_trunk(cond, prev_frame, noise)
        delta_base = self.base_head(h).squeeze(-1)
        cond_resid = cond - self.cond_ref
        film_params = self.cond_resid_film(cond_resid)
        gamma, beta = film_params.chunk(2, dim=-1)
        h_modulated = (1 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
        L = self.load_head(h_modulated)
        return delta_base, L


class ARFactorizedCleanModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, n_factors=5):
        super().__init__()
        self.encoder = GRUEncoder(encoder_config)
        self.decoder = FactorizedDecoderClean(**decoder_config, n_factors=n_factors)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_factors = n_factors


def normalize_iv(surfaces):
    return surfaces * 2.0 - 1.0


def denormalize_iv(surfaces):
    return (surfaces + 1.0) / 2.0


def reflecting_boundary(x, lo=0.01, hi=1.0):
    below = x < lo
    above = x > hi
    x = torch.where(below, 2 * lo - x, x)
    x = torch.where(above, 2 * hi - x, x)
    return x.clamp(lo, hi)


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
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("167d S3 WORST CELL DIAGNOSTIC")
    print("=" * 70)

    # --- Load model ---
    model_path = "models/backfill/afcrps_167d/best_model.pt"
    print(f"\nLoading model from {model_path}...")
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt['config']
    enc_cfg = EncoderConfig(**cfg['encoder'])
    model = ARFactorizedCleanModel(enc_cfg, cfg['decoder'], n_factors=cfg.get('n_factors', 5))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Loaded. cond_ref norm: {model.decoder.cond_ref.norm().item():.4f}")

    # --- Load data ---
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data['surface']
    returns = data['ret']
    surf_tensor = torch.tensor(surfaces, dtype=torch.float32, device=device)
    N = len(surfaces)
    H, T_fut = 30, 30

    # Test split starts at 4540
    test_start = 4540
    test_indices = np.arange(test_start, N - H - T_fut)
    print(f"  Test windows: {len(test_indices)} (start={test_start}, end={N-H-T_fut})")

    # Compute vol-of-vol for regime classification (matching test_block_ar_requirements_v2.py)
    # VoV = std of daily changes in mean IV over the history window
    def compute_vov(hist_surfaces):
        """hist_surfaces: (B, H, 5, 5) raw [0,1] surfaces"""
        mean_iv = hist_surfaces.mean(axis=(2, 3))  # (B, H)
        daily_ch = np.diff(mean_iv, axis=1)  # (B, H-1)
        return daily_ch.std(axis=1)  # (B,)

    # =========================================================================
    # STEP 1: Identify worst cell from summary.json
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Worst Cell Identification")
    print("=" * 70)

    summary_167d = json.load(open("results/block_ar/167d_best_30d/summary.json"))
    per_cell_mae = summary_167d['conditionality']['per_cell_mae_reduction']
    per_cell_wr = summary_167d['conditionality']['per_cell_width_ratio']

    # Find worst cell
    worst_mae_val = float('inf')
    worst_wr_val = 0
    worst_mae_cell = None
    worst_wr_cell = None
    for r in range(5):
        for c in range(5):
            if per_cell_mae[r][c] < worst_mae_val:
                worst_mae_val = per_cell_mae[r][c]
                worst_mae_cell = (r, c)
            if per_cell_wr[r][c] > worst_wr_val:
                worst_wr_val = per_cell_wr[r][c]
                worst_wr_cell = (r, c)

    print(f"  Worst MAE reduction: cell ({worst_mae_cell[0]},{worst_mae_cell[1]}) = {worst_mae_val:.2f}%")
    print(f"  Worst width ratio:   cell ({worst_wr_cell[0]},{worst_wr_cell[1]}) = {worst_wr_val:.4f}")

    # Print full grids
    print("\n  Per-cell MAE reduction (%):")
    for r in range(5):
        row = [f"{per_cell_mae[r][c]:7.1f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    print("\n  Per-cell width ratio:")
    for r in range(5):
        row = [f"{per_cell_wr[r][c]:7.3f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    # =========================================================================
    # STEP 2: Compare to baseline and 167b
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Cross-model comparison for problem cells")
    print("=" * 70)

    summary_167b = json.load(open("results/block_ar/167b_best_30d/summary.json"))
    summary_base = json.load(open("results/block_ar/164a_v3_percell_bptt_softplus_best_30d/summary.json"))

    for name, smry in [("167d (CLN+factorized)", summary_167d),
                        ("167b (CLN frozen+factorized)", summary_167b),
                        ("164a baseline (softplus)", summary_base)]:
        mae_grid = smry['conditionality']['per_cell_mae_reduction']
        wr_grid = smry['conditionality']['per_cell_width_ratio']
        wc_mae = smry['conditionality']['worst_cell_mae_reduction']
        wc_wr = smry['conditionality']['worst_cell_width_ratio']

        # Find their worst cell
        worst_r, worst_c = 0, 0
        worst_v = float('inf')
        for r in range(5):
            for c in range(5):
                if mae_grid[r][c] < worst_v:
                    worst_v = mae_grid[r][c]
                    worst_r, worst_c = r, c

        print(f"\n  {name}:")
        print(f"    worst_cell_mae_reduction = {wc_mae:.2f}% at cell ({worst_r},{worst_c})")
        print(f"    worst_cell_width_ratio   = {wc_wr:.4f}")
        # Print the specific problem cells from 167d
        r_mae, c_mae = worst_mae_cell
        r_wr, c_wr = worst_wr_cell
        print(f"    Cell ({r_mae},{c_mae}) MAE red: {mae_grid[r_mae][c_mae]:.2f}%, WR: {wr_grid[r_mae][c_mae]:.3f}")
        if (r_wr, c_wr) != (r_mae, c_mae):
            print(f"    Cell ({r_wr},{c_wr}) MAE red: {mae_grid[r_wr][c_wr]:.2f}%, WR: {wr_grid[r_wr][c_wr]:.3f}")

    # Check column 4 pattern (right edge cells)
    print("\n  Column 4 (long tenor / deep OTM call) pattern:")
    for name, smry in [("167d", summary_167d), ("167b", summary_167b), ("164a", summary_base)]:
        mae_grid = smry['conditionality']['per_cell_mae_reduction']
        wr_grid = smry['conditionality']['per_cell_width_ratio']
        col4_mae = [mae_grid[r][4] for r in range(5)]
        col4_wr = [wr_grid[r][4] for r in range(5)]
        print(f"    {name} col4 MAE: {[f'{v:.1f}' for v in col4_mae]}")
        print(f"    {name} col4 WR:  {[f'{v:.3f}' for v in col4_wr]}")

    # =========================================================================
    # STEP 3: FiLM Modulation Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FiLM Modulation + Loading Analysis")
    print("=" * 70)

    n_windows = min(200, len(test_indices))
    batch_size = 32

    all_gammas = []
    all_betas = []
    all_L_norms = []  # per-cell L norms
    all_L_full = []   # full L matrices for factor analysis
    all_delta_base = []
    all_cond_resid_norms = []
    all_vov = []  # vol-of-vol for regime classification

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            idx = test_indices[start:end]
            B = len(idx)

            # Build history
            idx_t = torch.from_numpy(idx).long().to(device)
            hist = surf_tensor[idx_t.unsqueeze(1) + torch.arange(H, device=device).unsqueeze(0)]
            hist_norm = normalize_iv(hist).reshape(B, H, 25)
            last_frame = hist[:, -1].reshape(B, 25)  # raw [0,1]

            # Compute vol-of-vol for regime
            hist_np = hist.cpu().numpy()
            batch_vov = compute_vov(hist_np)
            all_vov.extend(batch_vov.tolist())

            # Encoder
            gru_outputs, _ = model.encoder.gru(hist_norm)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)

            # FiLM decomposition
            cond_resid = cond - model.decoder.cond_ref
            all_cond_resid_norms.extend(cond_resid.norm(dim=-1).cpu().numpy().tolist())

            film_params = model.decoder.cond_resid_film(cond_resid)
            gamma, beta = film_params.chunk(2, dim=-1)
            all_gammas.append(gamma.cpu().numpy())
            all_betas.append(beta.cpu().numpy())

            # Forward trunk for one frame to get L
            z_t = torch.randn(B, model.decoder.noise_dim, device=device)
            h_trunk = model.decoder.forward_trunk(cond, last_frame, z_t)

            # Base head
            delta_base = model.decoder.base_head(h_trunk).squeeze(-1)
            all_delta_base.append(delta_base.cpu().numpy())

            # FiLM modulate
            h_modulated = (1 + gamma.unsqueeze(1)) * h_trunk + beta.unsqueeze(1)
            L = model.decoder.load_head(h_modulated)
            all_L_norms.append(L.norm(dim=-1).cpu().numpy())  # (B, 25)
            all_L_full.append(L.cpu().numpy())  # (B, 25, n_factors)

    gammas = np.concatenate(all_gammas, axis=0)  # (N, d_model)
    betas = np.concatenate(all_betas, axis=0)
    L_norms = np.concatenate(all_L_norms, axis=0)  # (N, 25)
    L_full = np.concatenate(all_L_full, axis=0)  # (N, 25, n_factors)
    delta_base = np.concatenate(all_delta_base, axis=0)  # (N, 25)
    all_vov_arr = np.array(all_vov)

    # Use Q20/Q80 quantile split (matching test harness)
    vov_q20 = np.quantile(all_vov_arr, 0.20)
    vov_q80 = np.quantile(all_vov_arr, 0.80)
    calm_idx = all_vov_arr <= vov_q20
    turb_idx = all_vov_arr >= vov_q80

    print(f"  Analyzed {len(gammas)} windows")
    print(f"  VoV Q20={vov_q20:.6f}, Q80={vov_q80:.6f}")
    print(f"  Regime split: {calm_idx.sum()} calm (Q20), {turb_idx.sum()} turb (Q80)")

    # FiLM statistics
    gamma_mean = gammas.mean(axis=0)
    gamma_std = gammas.std(axis=0)
    beta_mean = betas.mean(axis=0)
    beta_std = betas.std(axis=0)

    print(f"\n  FiLM gamma: mean={gamma_mean.mean():.4f}, std={gamma_std.mean():.4f}")
    print(f"  FiLM gamma range: [{gammas.min():.4f}, {gammas.max():.4f}]")
    print(f"  FiLM beta:  mean={beta_mean.mean():.4f}, std={beta_std.mean():.4f}")
    print(f"  FiLM |1+gamma|: mean={(1+gammas).mean():.4f}, range=[{(1+gammas).min():.4f}, {(1+gammas).max():.4f}]")

    # L_norms per cell
    print(f"\n  Per-cell L norm (loading magnitude):")
    L_norm_grid = L_norms.mean(axis=0).reshape(5, 5)
    for r in range(5):
        row = [f"{L_norm_grid[r, c]:7.4f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    L_norm_mean = L_norm_grid.mean()
    print(f"  Mean L norm: {L_norm_mean:.4f}")

    # Highlight worst cells
    r_mae, c_mae = worst_mae_cell
    r_wr, c_wr = worst_wr_cell
    cell_idx_mae = r_mae * 5 + c_mae
    cell_idx_wr = r_wr * 5 + c_wr
    print(f"\n  Worst MAE cell ({r_mae},{c_mae}): L norm = {L_norm_grid[r_mae, c_mae]:.4f} (ratio to mean: {L_norm_grid[r_mae, c_mae]/L_norm_mean:.2f}x)")
    if (r_wr, c_wr) != (r_mae, c_mae):
        print(f"  Worst WR cell  ({r_wr},{c_wr}): L norm = {L_norm_grid[r_wr, c_wr]:.4f} (ratio to mean: {L_norm_grid[r_wr, c_wr]/L_norm_mean:.2f}x)")

    # delta_base per cell
    print(f"\n  Per-cell delta_base (mean abs):")
    db_grid = np.abs(delta_base).mean(axis=0).reshape(5, 5)
    for r in range(5):
        row = [f"{db_grid[r, c]:7.5f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    # =========================================================================
    # STEP 4: Conditional Width per Cell — Calm vs Turbulent
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Conditional Width (Calm vs Turb) per Cell")
    print("=" * 70)

    # Generate samples for width analysis (inline, no sample_batched needed)
    n_samples = 50
    n_gen_windows = min(200, len(test_indices))

    all_gen_widths = np.zeros((0, 25))
    all_gen_vov = []

    gen_batch = 8
    n_factors = model.n_factors
    noise_dim = model.decoder.noise_dim

    with torch.no_grad():
        for start in range(0, n_gen_windows, gen_batch):
            end = min(start + gen_batch, n_gen_windows)
            idx = test_indices[start:end]
            B = len(idx)

            idx_t = torch.from_numpy(idx).long().to(device)
            hist = surf_tensor[idx_t.unsqueeze(1) + torch.arange(H, device=device).unsqueeze(0)]
            hist_norm = normalize_iv(hist)

            # Compute VoV
            hist_np = hist.cpu().numpy()
            batch_vov = compute_vov(hist_np)
            all_gen_vov.extend(batch_vov.tolist())

            # Encode
            hist_flat = hist_norm.reshape(B, H, 25)
            last_frame = denormalize_iv(hist_norm[:, -1]).reshape(B, 25)

            gru_outputs_base, h_last_base = model.encoder.gru(hist_flat)

            # Generate K samples per window
            k = n_samples
            last_k = last_frame.unsqueeze(1).expand(B, k, -1).reshape(B * k, 25)

            gru_out_k = gru_outputs_base.unsqueeze(1).expand(
                B, k, -1, -1).reshape(B * k, -1, model.encoder_config.gru_hidden_dim)
            h_last_k = h_last_base.unsqueeze(2).expand(
                1, B, k, -1).reshape(1, B * k, -1)

            attn_logits = model.encoder.attn_proj(gru_out_k).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_out_k).sum(dim=1)
            cond_init = model.encoder.bottleneck(h_pooled)

            prev = last_k
            gru_state = h_last_k.contiguous()
            gru_outs = gru_out_k

            for t in range(T_fut):
                z_t = torch.randn(B * k, noise_dim, device=device)
                if t > 0:
                    al = model.encoder.attn_proj(gru_outs).squeeze(-1)
                    aw = F.softmax(al, dim=1)
                    hp = (aw.unsqueeze(-1) * gru_outs).sum(dim=1)
                    cond_t = model.encoder.bottleneck(hp)
                else:
                    cond_t = cond_init

                delta_base_t, L_t = model.decoder(cond_t, prev, z_t)
                eps_t = torch.randn(B * k, n_factors, device=device)
                delta_t = delta_base_t + torch.einsum("bcr,br->bc", L_t, eps_t)
                frame_t = prev + torch.tanh(delta_t)
                frame_t = reflecting_boundary(frame_t)

                fn = normalize_iv(frame_t).unsqueeze(1)
                go, gru_state = model.encoder.gru(fn, gru_state)
                gru_outs = torch.cat([gru_outs, go], dim=1)
                prev = frame_t

            # Last frame is prev (t=29 frame)
            last_horizon = prev.reshape(B, k, 25)  # (B, K, 25)
            p95 = torch.quantile(last_horizon, 0.95, dim=1)
            p05 = torch.quantile(last_horizon, 0.05, dim=1)
            width = (p95 - p05).cpu().numpy()  # (B, 25)
            all_gen_widths = np.concatenate([all_gen_widths, width], axis=0)

    # Regime split using Q20/Q80
    gen_vov = np.array(all_gen_vov)
    gen_vov_q20 = np.quantile(gen_vov, 0.20)
    gen_vov_q80 = np.quantile(gen_vov, 0.80)
    gen_calm = gen_vov <= gen_vov_q20
    gen_turb = gen_vov >= gen_vov_q80
    calm_widths = all_gen_widths[gen_calm]
    turb_widths = all_gen_widths[gen_turb]
    uncond_widths = all_gen_widths

    print(f"  Generated widths for {len(uncond_widths)} windows ({len(calm_widths)} calm, {len(turb_widths)} turb)")

    calm_mean = calm_widths.mean(axis=0).reshape(5, 5)
    turb_mean = turb_widths.mean(axis=0).reshape(5, 5)
    uncond_mean = uncond_widths.mean(axis=0).reshape(5, 5)
    width_ratio = turb_mean / (calm_mean + 1e-8)

    print(f"\n  Width ratio (turb/calm) per cell:")
    for r in range(5):
        row = [f"{width_ratio[r, c]:7.3f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    print(f"\n  Calm width per cell:")
    for r in range(5):
        row = [f"{calm_mean[r, c]:7.5f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    print(f"\n  Turb width per cell:")
    for r in range(5):
        row = [f"{turb_mean[r, c]:7.5f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    print(f"\n  Uncond width per cell:")
    for r in range(5):
        row = [f"{uncond_mean[r, c]:7.5f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    # =========================================================================
    # STEP 5: CLN scale/bias analysis per cell
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CLN Scale/Bias Magnitude per Cell")
    print("=" * 70)

    # Extract CLN parameters across layers
    cln_scale_norms = []
    cln_bias_norms = []
    for li, layer in enumerate(model.decoder.layers):
        for cln_name in ['cln', 'ff_cln']:
            cln = layer[cln_name]
            # Get scale and bias projections
            sw = cln.scale_proj.weight.data  # (25*d_model, noise_dim)
            bw = cln.bias_proj.weight.data

            # Reshape to per-cell
            sw_cells = sw.reshape(25, model.decoder.d_model, -1)  # (25, d_model, noise_dim)
            bw_cells = bw.reshape(25, model.decoder.d_model, -1)

            sw_norms = sw_cells.norm(dim=(1, 2)).cpu().numpy()  # (25,)
            bw_norms = bw_cells.norm(dim=(1, 2)).cpu().numpy()

            cln_scale_norms.append(sw_norms)
            cln_bias_norms.append(bw_norms)

    # Average across all CLN layers
    cln_scale_avg = np.mean(cln_scale_norms, axis=0).reshape(5, 5)
    cln_bias_avg = np.mean(cln_bias_norms, axis=0).reshape(5, 5)

    print(f"  CLN scale weight norm per cell (avg across layers):")
    for r in range(5):
        row = [f"{cln_scale_avg[r, c]:7.3f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    print(f"\n  CLN bias weight norm per cell (avg across layers):")
    for r in range(5):
        row = [f"{cln_bias_avg[r, c]:7.3f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    # =========================================================================
    # STEP 6: L per-cell per-factor breakdown
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Per-Cell Per-Factor Loading Magnitudes")
    print("=" * 70)

    # L_full is (N, 25, n_factors)
    L_mean_abs = np.abs(L_full).mean(axis=0)  # (25, n_factors)
    L_mean_abs_grid = L_mean_abs.reshape(5, 5, -1)

    for f in range(model.n_factors):
        print(f"\n  Factor {f} loading magnitude:")
        for r in range(5):
            row = [f"{L_mean_abs_grid[r, c, f]:7.4f}" for c in range(5)]
            print(f"    row {r}: {' '.join(row)}")

    # =========================================================================
    # STEP 7: Calm vs Turb L norms for worst cell
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Regime-conditional Loading for Worst Cell")
    print("=" * 70)

    # calm_idx and turb_idx already defined from Step 3 (based on VoV Q20/Q80)
    for cell_name, (r, c) in [("worst_mae", worst_mae_cell), ("worst_wr", worst_wr_cell)]:
        cell_flat = r * 5 + c
        L_calm = L_norms[calm_idx, cell_flat]
        L_turb = L_norms[turb_idx, cell_flat]
        print(f"\n  Cell ({r},{c}) [{cell_name}]:")
        print(f"    L norm calm:  mean={L_calm.mean():.4f}, std={L_calm.std():.4f}")
        print(f"    L norm turb:  mean={L_turb.mean():.4f}, std={L_turb.std():.4f}")
        print(f"    L norm ratio (turb/calm): {L_turb.mean()/L_calm.mean():.3f}")

        db_calm = np.abs(delta_base[calm_idx, cell_flat])
        db_turb = np.abs(delta_base[turb_idx, cell_flat])
        print(f"    delta_base calm: mean={db_calm.mean():.5f}")
        print(f"    delta_base turb: mean={db_turb.mean():.5f}")

    # Average cell for comparison
    L_calm_avg = L_norms[calm_idx].mean()
    L_turb_avg = L_norms[turb_idx].mean()
    print(f"\n  Average cell:")
    print(f"    L norm calm:  mean={L_calm_avg:.4f}")
    print(f"    L norm turb:  mean={L_turb_avg:.4f}")
    print(f"    L norm ratio (turb/calm): {L_turb_avg/L_calm_avg:.3f}")

    # =========================================================================
    # STEP 8: Double-diversity test — contribution from CLN vs L@eps
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Double-diversity — CLN vs L@eps variance decomposition")
    print("=" * 70)

    # For a small batch, compare variance from CLN path vs L@eps path
    n_noise = 50
    idx_test = test_indices[:8]
    idx_t = torch.from_numpy(idx_test).long().to(device)
    B = len(idx_test)

    with torch.no_grad():
        hist = surf_tensor[idx_t.unsqueeze(1) + torch.arange(H, device=device).unsqueeze(0)]
        hist_norm = normalize_iv(hist).reshape(B, H, 25)
        last_frame = hist[:, -1].reshape(B, 25)

        # Encoder
        gru_outputs, _ = model.encoder.gru(hist_norm)
        attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
        cond = model.encoder.bottleneck(h_pooled)

        # Fixed condition, vary noise
        all_delta_base_samples = []
        all_L_samples = []
        all_Leps_samples = []
        all_full_delta = []

        for _ in range(n_noise):
            z_t = torch.randn(B, model.decoder.noise_dim, device=device)
            h_trunk = model.decoder.forward_trunk(cond, last_frame, z_t)

            db = model.decoder.base_head(h_trunk).squeeze(-1)

            cond_resid = cond - model.decoder.cond_ref
            film_params = model.decoder.cond_resid_film(cond_resid)
            gamma, beta = film_params.chunk(2, dim=-1)
            h_mod = (1 + gamma.unsqueeze(1)) * h_trunk + beta.unsqueeze(1)
            L = model.decoder.load_head(h_mod)

            eps = torch.randn(B, model.n_factors, device=device)
            Leps = torch.einsum("bcr,br->bc", L, eps)

            full_delta = db + Leps

            all_delta_base_samples.append(db.cpu().numpy())
            all_L_samples.append(L.cpu().numpy())
            all_Leps_samples.append(Leps.cpu().numpy())
            all_full_delta.append(full_delta.cpu().numpy())

    # Shape: (n_noise, B, 25)
    db_samples = np.stack(all_delta_base_samples, axis=0)
    Leps_samples = np.stack(all_Leps_samples, axis=0)
    full_samples = np.stack(all_full_delta, axis=0)

    # Variance decomposition per cell
    # Total variance = Var(delta_base) + Var(L@eps) + 2*Cov(delta_base, L@eps)
    # Since eps is independent of z, Cov should be small but CLN couples them
    var_db = db_samples.var(axis=0).mean(axis=0)  # (25,)
    var_Leps = Leps_samples.var(axis=0).mean(axis=0)
    var_full = full_samples.var(axis=0).mean(axis=0)
    var_interaction = var_full - var_db - var_Leps  # covariance term

    print(f"\n  Variance decomposition per cell (mean across 8 test windows, 50 noise samples):")
    print(f"  {'Cell':>6} {'Var(base)':>10} {'Var(Leps)':>10} {'Var(full)':>10} {'Interact':>10} {'Leps/full':>10}")
    for cell_flat in range(25):
        r, c = cell_flat // 5, cell_flat % 5
        marker = " <<<" if (r, c) in [worst_mae_cell, worst_wr_cell] else ""
        print(f"  ({r},{c}):  {var_db[cell_flat]:10.6f} {var_Leps[cell_flat]:10.6f} "
              f"{var_full[cell_flat]:10.6f} {var_interaction[cell_flat]:10.6f} "
              f"{var_Leps[cell_flat]/(var_full[cell_flat]+1e-10):10.3f}{marker}")

    var_db_grid = var_db.reshape(5, 5)
    var_Leps_grid = var_Leps.reshape(5, 5)
    var_full_grid = var_full.reshape(5, 5)
    Leps_frac = var_Leps_grid / (var_full_grid + 1e-10)

    print(f"\n  L@eps fraction of total variance:")
    for r in range(5):
        row = [f"{Leps_frac[r, c]:7.3f}" for c in range(5)]
        print(f"    row {r}: {' '.join(row)}")

    # =========================================================================
    # STEP 9: Root Cause Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Root Cause Analysis")
    print("=" * 70)

    r_w, c_w = worst_mae_cell
    cell_flat_w = r_w * 5 + c_w
    L_norm_worst = L_norm_grid[r_w, c_w]
    L_norm_avg = L_norm_grid.mean()
    cln_scale_worst = cln_scale_avg[r_w, c_w]
    cln_scale_all = cln_scale_avg.mean()
    wr_worst = per_cell_wr[r_w][c_w]
    Leps_frac_worst = Leps_frac[r_w, c_w]

    print(f"\n  Worst cell: ({r_w},{c_w})")
    print(f"    MAE reduction: {per_cell_mae[r_w][c_w]:.1f}%")
    print(f"    Width ratio:   {wr_worst:.3f}")
    print(f"    L norm:        {L_norm_worst:.4f} ({L_norm_worst/L_norm_avg:.2f}x avg)")
    print(f"    CLN scale:     {cln_scale_worst:.3f} ({cln_scale_worst/cln_scale_all:.2f}x avg)")
    print(f"    Leps variance fraction: {Leps_frac_worst:.3f}")

    # Check hypothesis (a): FiLM amplifies trunk too much for specific cells
    # FiLM is global (per-window, not per-cell), so amplification is uniform across cells
    print(f"\n  Hypothesis (a) - FiLM amplifies trunk for specific cells:")
    print(f"    FiLM is GLOBAL (broadcast to all 25 cells equally)")
    print(f"    |1+gamma| mean: {np.abs(1+gammas).mean():.4f}, std: {np.abs(1+gammas).std():.4f}")
    print(f"    FiLM cannot cause per-cell divergence. REJECTED if L_norm is uniform.")

    # Check hypothesis (b): L@eps too large for specific cells
    print(f"\n  Hypothesis (b) - L@eps too large for specific cells:")
    print(f"    L norm worst cell: {L_norm_worst:.4f}")
    print(f"    L norm mean cell:  {L_norm_avg:.4f}")
    print(f"    L norm ratio:      {L_norm_worst/L_norm_avg:.2f}x")

    # Check hypothesis (c): CLN + factorized noise double-diversity
    print(f"\n  Hypothesis (c) - CLN + L@eps double-diversity:")
    print(f"    Var(base) at worst cell:    {var_db_grid[r_w, c_w]:.6f}")
    print(f"    Var(Leps) at worst cell:    {var_Leps_grid[r_w, c_w]:.6f}")
    print(f"    Var(full) at worst cell:    {var_full_grid[r_w, c_w]:.6f}")
    print(f"    Interaction at worst cell:  {(var_full_grid - var_db_grid - var_Leps_grid)[r_w, c_w]:.6f}")
    print(f"    For comparison, avg cell:")
    print(f"      Var(base): {var_db_grid.mean():.6f}, Var(Leps): {var_Leps_grid.mean():.6f}")
    print(f"      Var(full): {var_full_grid.mean():.6f}")

    # Check hypothesis (d): 20 epochs insufficient
    print(f"\n  Hypothesis (d) - Training epochs:")
    print(f"    167d: 20 epochs, baseline: 80 epochs")
    print(f"    But 167b (also 20 epochs, CLN frozen) PASSES S3")
    print(f"    So epoch count alone doesn't explain CLN+factorized failure")

    # Derive best hypothesis
    print(f"\n  === DERIVED ROOT CAUSE ===")

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "worst_mae_cell": list(worst_mae_cell),
        "worst_mae_value": worst_mae_val,
        "worst_wr_cell": list(worst_wr_cell),
        "worst_wr_value": worst_wr_val,
        "per_cell_mae_reduction_167d": per_cell_mae,
        "per_cell_width_ratio_167d": per_cell_wr,
        "film_gamma_stats": {
            "mean": float(gamma_mean.mean()),
            "std": float(gamma_std.mean()),
            "range": [float(gammas.min()), float(gammas.max())],
            "abs_1_plus_gamma_mean": float(np.abs(1 + gammas).mean()),
        },
        "film_beta_stats": {
            "mean": float(beta_mean.mean()),
            "std": float(beta_std.mean()),
        },
        "per_cell_L_norm": L_norm_grid.tolist(),
        "per_cell_delta_base_abs": db_grid.tolist(),
        "per_cell_cln_scale_norm": cln_scale_avg.tolist(),
        "per_cell_cln_bias_norm": cln_bias_avg.tolist(),
        "per_cell_width_calm": calm_mean.tolist(),
        "per_cell_width_turb": turb_mean.tolist(),
        "per_cell_width_uncond": uncond_mean.tolist(),
        "per_cell_width_ratio_generated": width_ratio.tolist(),
        "variance_decomposition": {
            "var_base": var_db_grid.tolist(),
            "var_Leps": var_Leps_grid.tolist(),
            "var_full": var_full_grid.tolist(),
            "var_interaction": (var_full_grid - var_db_grid - var_Leps_grid).tolist(),
            "Leps_fraction": Leps_frac.tolist(),
        },
        "regime_conditional_L": {
            "worst_mae_cell": {
                "calm_L_mean": float(L_norms[calm_idx, cell_idx_mae].mean()),
                "turb_L_mean": float(L_norms[turb_idx, cell_idx_mae].mean()),
                "ratio": float(L_norms[turb_idx, cell_idx_mae].mean() / L_norms[calm_idx, cell_idx_mae].mean()),
            },
            "avg_cell": {
                "calm_L_mean": float(L_calm_avg),
                "turb_L_mean": float(L_turb_avg),
                "ratio": float(L_turb_avg / L_calm_avg),
            },
        },
        "cross_model_comparison": {
            "167d_worst_mae": worst_mae_val,
            "167b_worst_mae": summary_167b['conditionality']['worst_cell_mae_reduction'],
            "164a_worst_mae": summary_base['conditionality']['worst_cell_mae_reduction'],
            "167d_worst_wr": worst_wr_val,
            "167b_worst_wr": summary_167b['conditionality']['worst_cell_width_ratio'],
            "164a_worst_wr": summary_base['conditionality']['worst_cell_width_ratio'],
        },
        "per_factor_loadings": {
            f"factor_{f}": L_mean_abs.reshape(5, 5, -1)[:, :, f].tolist()
            for f in range(model.n_factors)
        },
    }

    out_path = Path("results/validations/2026-04-04/analysis/167d_followup/s3_worst_cell.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Verification results
    verification = {
        "experiment": "167d_s3_worst_cell_diagnostic",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "n_windows_analyzed": len(gammas),
        "n_windows_generated": len(uncond_widths),
        "worst_mae_cell": list(worst_mae_cell),
        "worst_mae_value": worst_mae_val,
        "worst_wr_cell": list(worst_wr_cell),
        "worst_wr_value": worst_wr_val,
        "key_findings": {
            "film_is_global": True,
            "L_norm_worst_vs_avg": float(L_norm_worst / L_norm_avg),
            "cln_scale_worst_vs_avg": float(cln_scale_worst / cln_scale_all),
            "Leps_variance_fraction_worst": float(Leps_frac_worst),
            "Leps_variance_fraction_avg": float(Leps_frac.mean()),
        },
    }

    verif_path = Path("results/validations/2026-04-04/verification_results/167d_s3_worst_cell.json")
    verif_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verif_path, 'w') as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"  Verification saved to {verif_path}")


if __name__ == "__main__":
    main()
