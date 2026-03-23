#!/usr/bin/env python3
"""Deep analysis of FactorNoiseSkip mechanism in 146b vs plain skip in 148a.

ANALYSES:
1. Factor noise W loadings matrix -- SVD, effective rank, interpretability
2. Skip pathway output magnitude comparison (146b vs 148a)
3. Noise pathway decomposition at inference (CLN vs skip contributions)
4. Per-cell spread contribution (which cells benefit most from factor noise)

Output: results/validations/2026-03-23/analysis/146b_factor_deep/
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import json

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig,
    CausalARTransformerDecoder,
    normalize_iv, denormalize_iv,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "results/validations/2026-03-23/analysis/146b_factor_deep"
os.makedirs(OUT_DIR, exist_ok=True)

# Moneyness and tenor labels for the 5x5 grid
MONEYNESS = ["0.90", "0.95", "1.00", "1.05", "1.10"]
TENORS = ["30d", "60d", "90d", "120d", "180d"]

# ──────────────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────────────

def load_model(path):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    cfg = SinglePassConfig(**ckpt["config"])
    model = SinglePassBlockAR(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    return model, cfg

print("Loading 146b (factor noise)...")
model_146b, cfg_146b = load_model("models/backfill/afcrps_146b/best_model.pt")
print("Loading 148a (plain skip)...")
model_148a, cfg_148a = load_model("models/backfill/afcrps_148a/best_model.pt")

# ──────────────────────────────────────────────────────────────────────
# Load test data
# ──────────────────────────────────────────────────────────────────────

data = np.load("data/vol_surface_with_ret.npz")
surfaces = data["surface"]  # (N, 5, 5)
N = len(surfaces)
TEST_START = 4540

# Build 5 test windows: each is (30 history, 30 future)
test_windows = []
for i in range(5):
    idx = TEST_START + i * 30
    if idx + 60 > N:
        break
    hist = surfaces[idx:idx+30]      # (30, 5, 5) raw IV in [0,1]
    future = surfaces[idx+30:idx+60]  # (30, 5, 5)
    test_windows.append((hist, future))

print(f"Built {len(test_windows)} test windows starting at index {TEST_START}")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Factor Noise W Loadings Matrix
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ANALYSIS 1: Factor Noise W Loadings Matrix")
print("="*70)

# Extract W from 146b's noise_skip_proj (FactorNoiseSkip)
W = model_146b.frame_decoder.noise_skip_proj.W.detach().cpu().numpy()  # (25, 5)
print(f"W shape: {W.shape}")
print(f"W stats: mean={W.mean():.4f}, std={W.std():.4f}, min={W.min():.4f}, max={W.max():.4f}")

# SVD of W
U, S, Vt = np.linalg.svd(W, full_matrices=False)
print(f"\nSingular values of W: {S}")
print(f"Singular value ratios (S_i / S_0): {S / S[0]}")

# Effective rank (exponential of entropy)
S_norm = S / S.sum()
S_norm_nz = S_norm[S_norm > 1e-10]
eff_rank = np.exp(-np.sum(S_norm_nz * np.log(S_norm_nz)))
print(f"Effective rank of W: {eff_rank:.3f} (out of {len(S)})")

# Nuclear/spectral
nuc_over_spec = S.sum() / S[0]
print(f"Nuclear/spectral ratio: {nuc_over_spec:.3f}")

# Residual projection
residual_w = model_146b.frame_decoder.noise_skip_proj.residual_proj.weight.detach().cpu().numpy()
print(f"\nResidual proj weight shape: {residual_w.shape}")
res_frob = np.linalg.norm(residual_w)
print(f"Residual proj Frobenius norm: {res_frob:.6f}")
print(f"Residual proj max abs: {np.abs(residual_w).max():.6f}")

# Visualize W as 5 heatmaps (one per factor), each reshaped to 5x5 grid
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Factor Noise W Loadings (25 cells x 5 factors)\nEach factor = column of W reshaped to 5x5 grid", fontsize=14)

for f in range(5):
    loadings = W[:, f].reshape(5, 5)
    vmax = max(abs(W[:, f].max()), abs(W[:, f].min()))
    if vmax < 1e-6:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = axes[f].imshow(loadings, cmap="RdBu_r", norm=norm, aspect="auto")
    axes[f].set_title(f"Factor {f+1}\n(sv={S[f]:.4f})")
    axes[f].set_xticks(range(5))
    axes[f].set_xticklabels(TENORS, rotation=45, fontsize=8)
    axes[f].set_yticks(range(5))
    axes[f].set_yticklabels(MONEYNESS, fontsize=8)
    if f == 0:
        axes[f].set_ylabel("Moneyness")
    plt.colorbar(im, ax=axes[f], shrink=0.8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/analysis1_W_loadings.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_DIR}/analysis1_W_loadings.png")

# Print loadings numerically
for f in range(5):
    print(f"\nFactor {f+1} loadings (5x5 grid, rows=moneyness, cols=tenor):")
    grid = W[:, f].reshape(5, 5)
    for i, m in enumerate(MONEYNESS):
        vals = "  ".join(f"{grid[i,j]:+.4f}" for j in range(5))
        print(f"  K={m}: {vals}")

# Interpret factors: correlation with known surface patterns
patterns = {}
patterns["level"] = np.ones((5, 5)).flatten()
patterns["m_slope"] = np.outer(np.linspace(-1, 1, 5), np.ones(5)).flatten()
patterns["t_slope"] = np.outer(np.ones(5), np.linspace(-1, 1, 5)).flatten()
patterns["smile"] = np.outer(np.array([1, 0.25, 0, 0.25, 1]), np.ones(5)).flatten()
patterns["twist"] = np.outer(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)).flatten()

print("\n\nFactor vs pattern correlations (cosine similarity for level, Pearson for rest):")
header = f"{'Factor':<10}" + "".join(f" {p:>8}" for p in patterns.keys())
print(header)
for f in range(5):
    row = f"Factor {f+1:<3}"
    col = W[:, f]
    for pname, pvec in patterns.items():
        if pname == "level":
            # Cosine similarity (level pattern = all 1s, so cosine = mean/norm)
            cos = np.dot(col, pvec) / (np.linalg.norm(col) * np.linalg.norm(pvec) + 1e-10)
            row += f" {cos:+.4f}"
        else:
            corr = np.corrcoef(col, pvec)[0, 1]
            row += f" {corr:+.4f}"
    print(row)

# WW^T = covariance structure imposed by factor noise
WWT = W @ W.T  # (25, 25)
print(f"\nW @ W.T (factor covariance) shape: {WWT.shape}")
print(f"WW^T diagonal (per-cell variance from factors): min={WWT.diagonal().min():.6f}, max={WWT.diagonal().max():.6f}")

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
im = ax.imshow(WWT, cmap="RdBu_r", aspect="auto")
ax.set_title("W @ W^T -- Factor Covariance Structure\n(25x25 implied cell-cell covariance from 5 factors)")
cell_labels = [f"{m},{t}" for m in MONEYNESS for t in TENORS]
ax.set_xticks(range(25))
ax.set_yticks(range(25))
ax.set_xticklabels(cell_labels, rotation=90, fontsize=5)
ax.set_yticklabels(cell_labels, fontsize=5)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/analysis1_WWT_covariance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/analysis1_WWT_covariance.png")

# Compare WWT correlation with ground truth cross-cell correlation
# (from the training data)
# Compute GT cross-cell correlation from daily changes
daily_changes = np.diff(surfaces, axis=0).reshape(-1, 25)  # (N-1, 25)
gt_corr = np.corrcoef(daily_changes.T)  # (25, 25)

# Normalize WWT to correlation
ww_diag = np.sqrt(np.diag(WWT))
ww_diag[ww_diag < 1e-10] = 1e-10
WWT_corr = WWT / np.outer(ww_diag, ww_diag)

# Correlation between implied and GT correlation matrices (upper tri only)
triu_idx = np.triu_indices(25, k=1)
gt_triu = gt_corr[triu_idx]
ww_triu = WWT_corr[triu_idx]
mask = np.isfinite(gt_triu) & np.isfinite(ww_triu)
corr_gt_vs_factor = np.corrcoef(gt_triu[mask], ww_triu[mask])[0, 1]
print(f"\nCorrelation between factor-implied and GT cross-cell correlation: {corr_gt_vs_factor:.4f}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Compare Skip Pathway Output Magnitude
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ANALYSIS 2: Skip Pathway Output Magnitude Comparison")
print("="*70)

# 146b: FactorNoiseSkip
W_frob = np.linalg.norm(W)
print(f"\n146b FactorNoiseSkip:")
print(f"  W (factor loadings) Frobenius norm: {W_frob:.6f}")
print(f"  Residual proj Frobenius norm: {res_frob:.6f}")
print(f"  W column norms: {np.linalg.norm(W, axis=0)}")
print(f"  Total effective Frobenius: ~{np.sqrt(W_frob**2 + res_frob**2):.6f}")

# 148a: plain Linear(32, 25, bias=False)
skip_148a = model_148a.frame_decoder.noise_skip_proj
w_148a = skip_148a.weight.detach().cpu().numpy()  # (25, 32)
print(f"\n148a plain Linear skip:")
print(f"  Weight shape: {w_148a.shape}")
print(f"  Frobenius norm: {np.linalg.norm(w_148a):.6f}")
print(f"  Max abs weight: {np.abs(w_148a).max():.6f}")
print(f"  Mean abs weight: {np.abs(w_148a).mean():.6f}")

# SVD of 148a's skip weights
U_148a, S_148a, Vt_148a = np.linalg.svd(w_148a, full_matrices=False)
print(f"  Singular values (all): {S_148a}")
S_148a_norm = S_148a / S_148a.sum()
S_148a_nz = S_148a_norm[S_148a_norm > 1e-10]
eff_rank_148a = np.exp(-np.sum(S_148a_nz * np.log(S_148a_nz)))
print(f"  Effective rank: {eff_rank_148a:.3f}")
print(f"  S[0]/S[1] ratio: {S_148a[0]/max(S_148a[1], 1e-10):.3f}")

# Empirical: push random noise through both and compare output magnitudes
torch.manual_seed(42)
n_test = 1000
z = torch.randn(n_test, 32).to(DEVICE)

with torch.no_grad():
    skip_146b_module = model_146b.frame_decoder.noise_skip_proj
    out_146b = skip_146b_module(z)       # (n_test, 25)
    out_148a = skip_148a(z)              # (n_test, 25)
    out_146b_tanh = torch.tanh(out_146b)
    out_148a_tanh = torch.tanh(out_148a)

print(f"\nEmpirical output (pre-tanh, {n_test} random noise samples):")
print(f"  146b: mean abs={out_146b.abs().mean():.6f}, std={out_146b.std():.6f}, max abs={out_146b.abs().max():.6f}")
print(f"  148a: mean abs={out_148a.abs().mean():.6f}, std={out_148a.std():.6f}, max abs={out_148a.abs().max():.6f}")
print(f"\nEmpirical output (post-tanh):")
print(f"  146b: mean abs={out_146b_tanh.abs().mean():.6f}, std={out_146b_tanh.std():.6f}")
print(f"  148a: mean abs={out_148a_tanh.abs().mean():.6f}, std={out_148a_tanh.std():.6f}")
print(f"  Ratio 148a/146b: {out_148a_tanh.abs().mean() / out_146b_tanh.abs().mean():.2f}x")

# Per-cell output std (post-tanh)
std_146b_cells = out_146b_tanh.std(dim=0).cpu().numpy()
std_148a_cells = out_148a_tanh.std(dim=0).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, std_vals, title in [
    (axes[0], std_146b_cells, "146b (FactorNoiseSkip)"),
    (axes[1], std_148a_cells, "148a (plain Linear)"),
]:
    im = ax.imshow(std_vals.reshape(5, 5), cmap="viridis", aspect="auto")
    ax.set_title(f"Per-cell skip output std -- {title}")
    ax.set_xticks(range(5))
    ax.set_xticklabels(TENORS, rotation=45)
    ax.set_yticks(range(5))
    ax.set_yticklabels(MONEYNESS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{std_vals[i*5+j]:.4f}", ha="center", va="center", fontsize=8,
                    color="white" if std_vals[i*5+j] < std_vals.max()*0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Skip Pathway Output Std (1000 random noise samples, post-tanh)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/analysis2_skip_magnitude.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_DIR}/analysis2_skip_magnitude.png")

# Cross-cell correlation of skip output
corr_146b_skip = np.corrcoef(out_146b_tanh.cpu().numpy().T)  # (25, 25)
corr_148a_skip = np.corrcoef(out_148a_tanh.cpu().numpy().T)  # (25, 25)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, corr, title in [
    (axes[0], corr_146b_skip, "146b (FactorNoiseSkip)"),
    (axes[1], corr_148a_skip, "148a (plain Linear)"),
]:
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"Cross-cell correlation -- {title}")
    ax.set_xticks(range(25))
    ax.set_yticks(range(25))
    ax.set_xticklabels(cell_labels, rotation=90, fontsize=4)
    ax.set_yticklabels(cell_labels, fontsize=4)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Skip Output Cross-Cell Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/analysis2_skip_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/analysis2_skip_correlation.png")

# Effective rank of correlation matrices
def eff_rank_of_corr(corr):
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.maximum(eigvals, 0)
    eigvals_n = eigvals / eigvals.sum()
    eigvals_nz = eigvals_n[eigvals_n > 1e-10]
    return np.exp(-np.sum(eigvals_nz * np.log(eigvals_nz)))

eff_rank_corr_146b = eff_rank_of_corr(corr_146b_skip)
eff_rank_corr_148a = eff_rank_of_corr(corr_148a_skip)

print(f"\nEffective rank of skip output correlation matrix:")
print(f"  146b (factor): {eff_rank_corr_146b:.3f}")
print(f"  148a (plain):  {eff_rank_corr_148a:.3f}")
print(f"  Ratio 146b/148a: {eff_rank_corr_146b / max(eff_rank_corr_148a, 1e-10):.3f}x")

# Compare skip correlation with GT
corr_gt_vs_146b_skip = np.corrcoef(gt_corr[triu_idx], corr_146b_skip[triu_idx])[0, 1]
corr_gt_vs_148a_skip = np.corrcoef(gt_corr[triu_idx], corr_148a_skip[triu_idx])[0, 1]
print(f"\nCorrelation of skip output cross-cell structure with GT:")
print(f"  146b (factor): {corr_gt_vs_146b_skip:.4f}")
print(f"  148a (plain):  {corr_gt_vs_148a_skip:.4f}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Noise Pathway Decomposition at Inference
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ANALYSIS 3: Noise Pathway Decomposition at Inference")
print("="*70)

def probe_skip_vs_decoder(model, history_raw, n_samples=50, seed=42):
    """Probe skip vs decoder output magnitudes with full trajectory generation.

    For the CausalARTransformerDecoder:
    - forward() returns delta = softplus(log_vol_scale) * out_proj(last_hidden)
      This is the CLN/attention path output.
    - noise_skip_proj(noise) is applied SEPARATELY in the AR loop (skip_bypass_spread=True)
      as: tanh(noise_skip_proj(noise_input))

    So: total_delta_in_iv_space = vol_scale * (cell_spread * decoder_delta + skip_out)
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config
    H, W_dim = cfg.surface_h, cfg.surface_w

    # Prepare normalized history
    hist_tensor = torch.tensor(history_raw, dtype=torch.float32).unsqueeze(0).to(device)
    history_norm = normalize_iv(hist_tensor)  # (1, 30, 5, 5) in [-1, 1]

    # Encode
    with torch.no_grad():
        condition = model.encoder(history_norm, mask=None)
        gru_outputs, h_last = model._init_gru_state(history_norm)

    _, vol_scale = model._compute_vol_scale(history_norm)
    vol_scale_cell = None
    if cfg.ar_frame_percell_vol_scale:
        vol_scale_cell = model._compute_percell_vol_scale(history_norm)

    torch.manual_seed(seed)

    all_decoder_deltas = []   # CLN/transformer output
    all_skip_outputs = []     # skip pathway output (tanh(noise_skip_proj(z)))

    for s in range(n_samples):
        z0 = torch.randn(1, cfg.noise_dim, device=device)
        z_t = z0.clone()
        rho = cfg.ar_frame_rho

        # Re-initialize decoder context for each sample
        cond_s = condition.clone()
        gru_out_s = gru_outputs.clone()
        h_s = h_last.clone()

        if isinstance(model.frame_decoder, CausalARTransformerDecoder):
            hist_flat = denormalize_iv(history_norm).reshape(1, 30, 25)
            model.frame_decoder.init_context(cond_s, hist_flat)

        prev_frame = denormalize_iv(history_norm[:, -1])  # (1, 5, 5) in [0,1]

        floor = cfg.ar_frame_floor_clamp if cfg.ar_frame_floor_clamp > 0 else 0.0
        s_decoder = []
        s_skip = []

        for t in range(30):
            if t > 0:
                eps_t = torch.randn_like(z_t)
                z_t = rho * z_t + math.sqrt(1 - rho**2) * eps_t

            local_pos, horizon_bucket = model._get_ar_frame_positions(
                step_idx=t, batch_size=1, device=device, position_mode="native"
            )
            prev_flat = prev_frame.reshape(1, H * W_dim)
            noise_input = model._get_noise_for_decoder(z_t)

            cln_wf = 1.0
            if cfg.ar_cln_warmup > 0:
                cln_wf = min(1.0, t / max(cfg.ar_cln_warmup, 1))

            # Decoder output (CLN/attention path)
            dec_delta = model.frame_decoder(
                prev_flat, cond_s, noise_input, local_pos, horizon_bucket, cln_wf
            )  # (1, 25) -- this is the decoder's own output, BEFORE skip is added

            # Skip output (computed separately, bypass cell_spread)
            skip_raw = model.frame_decoder.noise_skip_proj(noise_input)  # (1, 25)
            skip_tanh = torch.tanh(skip_raw)  # (1, 25)

            s_decoder.append(dec_delta.detach().cpu())
            s_skip.append(skip_tanh.detach().cpu())

            # Now compute the actual IV update for proper trajectory
            delta_grid = dec_delta.reshape(1, H, W_dim)
            skip_grid = skip_tanh.reshape(1, H, W_dim)

            # Cell spread on decoder delta
            if hasattr(model, 'cell_scale'):
                cs = model.cell_scale.clamp(0.3, 3.0).view(H, W_dim)
                delta_grid = cs * delta_grid
            cs = model._get_cell_spread(cond_s, local_pos)
            if cs is not None:
                delta_grid = cs * delta_grid

            # Skip bypass: add skip AFTER cell_spread
            if cfg.ar_skip_bypass_spread and model.frame_decoder.noise_skip_proj is not None:
                noise_scale = model._get_noise_scale(cond_s)
                if noise_scale is not None:
                    skip_grid = skip_grid * noise_scale.view(1, H, W_dim)
                delta_grid = delta_grid + skip_grid

            vs = model._get_ar_frame_vol_scale(cond_s, vol_scale, vol_scale_cell)

            if cfg.ar_frame_log_space:
                iv_t = (prev_frame * torch.exp(vs * delta_grid)).clamp(floor, 1.0)
            elif cfg.ar_frame_reflect:
                mr = model._get_mean_revert(cond_s, prev_frame)
                raw = prev_frame + vs * delta_grid + mr
                width = 1.0 - floor
                shifted = raw - floor
                shifted = shifted % (2 * width)
                iv_t = torch.where(shifted > width, 2 * width - shifted, shifted) + floor
            else:
                mr = model._get_mean_revert(cond_s, prev_frame)
                iv_t = (prev_frame + vs * delta_grid + mr).clamp(floor, 1.0)

            prev_frame = iv_t

            # GRU step
            if not cfg.ar_freeze_gru_state:
                cond_s, gru_out_s, h_s = model._gru_step(iv_t, gru_out_s, h_s)

        all_decoder_deltas.append(torch.stack(s_decoder, dim=1))  # (1, 30, 25)
        all_skip_outputs.append(torch.stack(s_skip, dim=1))

    decoder_deltas = torch.cat(all_decoder_deltas, dim=0).numpy()  # (n_samples, 30, 25)
    skip_outputs = torch.cat(all_skip_outputs, dim=0).numpy()

    return decoder_deltas, skip_outputs


# Run probe on each test window for both models
N_SAMPLES = 50

print("\nRunning probe on 146b (factor noise)...")
all_decoder_146b = []
all_skip_146b = []
for w_idx, (hist, future) in enumerate(test_windows):
    dec, skip = probe_skip_vs_decoder(model_146b, hist, n_samples=N_SAMPLES, seed=42+w_idx)
    all_decoder_146b.append(dec)
    all_skip_146b.append(skip)
    print(f"  Window {w_idx}: decoder abs mean={np.abs(dec).mean():.6f}, skip abs mean={np.abs(skip).mean():.6f}")

print("\nRunning probe on 148a (plain skip)...")
all_decoder_148a = []
all_skip_148a = []
for w_idx, (hist, future) in enumerate(test_windows):
    dec, skip = probe_skip_vs_decoder(model_148a, hist, n_samples=N_SAMPLES, seed=42+w_idx)
    all_decoder_148a.append(dec)
    all_skip_148a.append(skip)
    print(f"  Window {w_idx}: decoder abs mean={np.abs(dec).mean():.6f}, skip abs mean={np.abs(skip).mean():.6f}")

# Aggregate across windows
dec_146b = np.concatenate(all_decoder_146b, axis=0)   # (250, 30, 25)
skip_146b = np.concatenate(all_skip_146b, axis=0)
dec_148a = np.concatenate(all_decoder_148a, axis=0)
skip_148a = np.concatenate(all_skip_148a, axis=0)

print(f"\n{'='*70}")
print(f"PATHWAY MAGNITUDE SUMMARY")
print(f"{'='*70}")
print(f"{'Metric':<35} {'146b (factor)':>14} {'148a (plain)':>14}")
print(f"-" * 65)
print(f"{'Decoder |delta| mean':35} {np.abs(dec_146b).mean():14.6f} {np.abs(dec_148a).mean():14.6f}")
print(f"{'Skip |output| mean':35} {np.abs(skip_146b).mean():14.6f} {np.abs(skip_148a).mean():14.6f}")
print(f"{'Skip/Decoder ratio':35} {np.abs(skip_146b).mean()/max(np.abs(dec_146b).mean(), 1e-10):14.4f} {np.abs(skip_148a).mean()/max(np.abs(dec_148a).mean(), 1e-10):14.4f}")
print(f"{'Decoder std across samples':35} {dec_146b.std(axis=0).mean():14.6f} {dec_148a.std(axis=0).mean():14.6f}")
print(f"{'Skip std across samples':35} {skip_146b.std(axis=0).mean():14.6f} {skip_148a.std(axis=0).mean():14.6f}")

# Variance decomposition
dec_var_146b_total = dec_146b.var(axis=0).mean()
skip_var_146b_total = skip_146b.var(axis=0).mean()
cov_term_146b = np.mean([np.cov(dec_146b[:, t, c], skip_146b[:, t, c])[0, 1]
                          for t in range(30) for c in range(25)])
total_var_146b = dec_var_146b_total + skip_var_146b_total + 2 * cov_term_146b

dec_var_148a_total = dec_148a.var(axis=0).mean()
skip_var_148a_total = skip_148a.var(axis=0).mean()
cov_term_148a = np.mean([np.cov(dec_148a[:, t, c], skip_148a[:, t, c])[0, 1]
                          for t in range(30) for c in range(25)])
total_var_148a = dec_var_148a_total + skip_var_148a_total + 2 * cov_term_148a

print(f"\n{'='*70}")
print(f"VARIANCE DECOMPOSITION (across 50 ensemble members)")
print(f"{'='*70}")
print(f"\n146b (factor noise):")
print(f"  Decoder variance:    {dec_var_146b_total:.8f} ({100*dec_var_146b_total/total_var_146b:.1f}%)")
print(f"  Skip variance:       {skip_var_146b_total:.8f} ({100*skip_var_146b_total/total_var_146b:.1f}%)")
print(f"  2*Covariance:        {2*cov_term_146b:.8f} ({100*2*cov_term_146b/total_var_146b:.1f}%)")
print(f"  Total:               {total_var_146b:.8f}")

print(f"\n148a (plain skip):")
print(f"  Decoder variance:    {dec_var_148a_total:.8f} ({100*dec_var_148a_total/total_var_148a:.1f}%)")
print(f"  Skip variance:       {skip_var_148a_total:.8f} ({100*skip_var_148a_total/total_var_148a:.1f}%)")
print(f"  2*Covariance:        {2*cov_term_148a:.8f} ({100*2*cov_term_148a/total_var_148a:.1f}%)")
print(f"  Total:               {total_var_148a:.8f}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Per-Cell Diversity Decomposition
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ANALYSIS 4: Per-Cell Diversity Decomposition")
print("="*70)

# Per-cell variance decomposition
dec_var_cell_146b = dec_146b.var(axis=0).mean(axis=0)    # (25,)
skip_var_cell_146b = skip_146b.var(axis=0).mean(axis=0)
cov_cell_146b = np.zeros(25)
for c in range(25):
    cov_cell_146b[c] = np.mean([np.cov(dec_146b[:, t, c], skip_146b[:, t, c])[0, 1]
                                 for t in range(30)])
total_var_cell_146b = dec_var_cell_146b + skip_var_cell_146b + 2 * cov_cell_146b
skip_frac_146b = skip_var_cell_146b / np.maximum(total_var_cell_146b, 1e-10)

dec_var_cell_148a = dec_148a.var(axis=0).mean(axis=0)
skip_var_cell_148a = skip_148a.var(axis=0).mean(axis=0)
cov_cell_148a = np.zeros(25)
for c in range(25):
    cov_cell_148a[c] = np.mean([np.cov(dec_148a[:, t, c], skip_148a[:, t, c])[0, 1]
                                 for t in range(30)])
total_var_cell_148a = dec_var_cell_148a + skip_var_cell_148a + 2 * cov_cell_148a
skip_frac_148a = skip_var_cell_148a / np.maximum(total_var_cell_148a, 1e-10)

# Also: decoder std per cell
dec_std_per_cell_146b = dec_146b.std(axis=0).mean(axis=0)
skip_std_per_cell_146b = skip_146b.std(axis=0).mean(axis=0)
dec_std_per_cell_148a = dec_148a.std(axis=0).mean(axis=0)
skip_std_per_cell_148a = skip_148a.std(axis=0).mean(axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Row 1: Skip fraction of total variance
for ax, frac, title in [
    (axes[0, 0], skip_frac_146b, "146b (factor): skip % of total var"),
    (axes[0, 1], skip_frac_148a, "148a (plain): skip % of total var"),
]:
    grid = frac.reshape(5, 5)
    im = ax.imshow(grid * 100, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xticks(range(5))
    ax.set_xticklabels(TENORS, rotation=45)
    ax.set_yticks(range(5))
    ax.set_yticklabels(MONEYNESS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{grid[i,j]*100:.1f}%", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="%")

# Row 2: Bar chart of decoder vs skip std
for ax, dec_v, skip_v, title in [
    (axes[1, 0], dec_std_per_cell_146b, skip_std_per_cell_146b, "146b: per-cell std (dec vs skip)"),
    (axes[1, 1], dec_std_per_cell_148a, skip_std_per_cell_148a, "148a: per-cell std (dec vs skip)"),
]:
    cells = np.arange(25)
    ax.bar(cells - 0.2, dec_v, 0.4, label="Decoder (CLN)", alpha=0.8, color="steelblue")
    ax.bar(cells + 0.2, skip_v, 0.4, label="Skip", alpha=0.8, color="coral")
    ax.set_title(title)
    ax.set_xlabel("Cell index")
    ax.set_ylabel("Std across samples (avg over horizons)")
    ax.legend()
    ax.set_xticks(range(0, 25, 5))

fig.suptitle("Per-Cell Diversity Decomposition: Decoder vs Skip Pathway", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/analysis4_per_cell_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/analysis4_per_cell_decomposition.png")

# Print table
print(f"\n{'Cell':<15} {'146b dec_std':>12} {'146b skip_std':>13} {'146b skip%':>10} | {'148a dec_std':>12} {'148a skip_std':>13} {'148a skip%':>10}")
print("-" * 95)
for c in range(25):
    row = int(c / 5)
    col = c % 5
    label = f"K={MONEYNESS[row]},T={TENORS[col]}"
    print(f"{label:<15} {dec_std_per_cell_146b[c]:12.6f} {skip_std_per_cell_146b[c]:13.6f} {skip_frac_146b[c]*100:9.1f}% | {dec_std_per_cell_148a[c]:12.6f} {skip_std_per_cell_148a[c]:13.6f} {skip_frac_148a[c]*100:9.1f}%")

# Horizon evolution of skip fraction
dec_var_h_146b = dec_146b.var(axis=0).mean(axis=1)    # (30,)
skip_var_h_146b = skip_146b.var(axis=0).mean(axis=1)
total_h_146b = dec_var_h_146b + skip_var_h_146b
skip_frac_h_146b = skip_var_h_146b / np.maximum(total_h_146b, 1e-10)

dec_var_h_148a = dec_148a.var(axis=0).mean(axis=1)
skip_var_h_148a = skip_148a.var(axis=0).mean(axis=1)
total_h_148a = dec_var_h_148a + skip_var_h_148a
skip_frac_h_148a = skip_var_h_148a / np.maximum(total_h_148a, 1e-10)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1, 31), skip_frac_h_146b * 100, 'b-o', label="146b (factor)", markersize=3)
axes[0].plot(range(1, 31), skip_frac_h_148a * 100, 'r-s', label="148a (plain)", markersize=3)
axes[0].set_xlabel("Horizon step")
axes[0].set_ylabel("Skip % of total variance")
axes[0].set_title("Skip Fraction vs Horizon")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, 31), dec_var_h_146b, 'b-o', label="146b decoder var", markersize=3)
axes[1].plot(range(1, 31), skip_var_h_146b, 'b--s', label="146b skip var", markersize=3)
axes[1].plot(range(1, 31), dec_var_h_148a, 'r-o', label="148a decoder var", markersize=3)
axes[1].plot(range(1, 31), skip_var_h_148a, 'r--s', label="148a skip var", markersize=3)
axes[1].set_xlabel("Horizon step")
axes[1].set_ylabel("Variance")
axes[1].set_title("Decoder vs Skip Variance by Horizon")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

# Effective rank of ensemble correlation at each horizon
eff_rank_146b_by_h = []
eff_rank_148a_by_h = []
for t in range(30):
    # Total delta = decoder + skip
    total_146b_t = dec_146b[:, t, :] + skip_146b[:, t, :]  # (250, 25)
    total_148a_t = dec_148a[:, t, :] + skip_148a[:, t, :]

    corr_146b_t = np.corrcoef(total_146b_t.T)
    corr_148a_t = np.corrcoef(total_148a_t.T)

    # Handle NaN correlations
    corr_146b_t = np.nan_to_num(corr_146b_t, nan=0.0)
    corr_148a_t = np.nan_to_num(corr_148a_t, nan=0.0)

    eff_rank_146b_by_h.append(eff_rank_of_corr(corr_146b_t))
    eff_rank_148a_by_h.append(eff_rank_of_corr(corr_148a_t))

axes[2].plot(range(1, 31), eff_rank_146b_by_h, 'b-o', label="146b (factor)", markersize=3)
axes[2].plot(range(1, 31), eff_rank_148a_by_h, 'r-s', label="148a (plain)", markersize=3)
axes[2].axhline(y=eff_rank_of_corr(gt_corr), color='green', linestyle='--', label="GT")
axes[2].set_xlabel("Horizon step")
axes[2].set_ylabel("Effective rank")
axes[2].set_title("Effective Rank of Ensemble Cross-Cell Corr by Horizon")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/analysis4_horizon_evolution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/analysis4_horizon_evolution.png")


# ══════════════════════════════════════════════════════════════════════
# BONUS: Gradient flow analysis -- why doesn't CRPS collapse factor noise?
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BONUS: Why factor noise resists CRPS collapse")
print("="*70)

# Key insight: in 148a (plain Linear(32, 25, bias=False)):
# - CRPS can collapse ALL 800 weights (32*25) toward zero
# - Each cell's skip output is w_c @ z, where w_c is row c of the weight matrix
# - CRPS minimizer can make w_c -> 0 for all c independently
# - The 25 cells are FULLY coupled by the CRPS loss, so CRPS drives them
#   to rank-1 (all cells proportional)

# In 146b (FactorNoiseSkip):
# - Factor part: z_factors @ W.T where W is (25, 5)
# - Only 125 parameters (25*5) but STRUCTURED: cell c's output is sum_f W[c,f] * z_f
# - To collapse diversity, CRPS must set ALL of W -> 0
# - But W columns are INDEPENDENT factors -- collapsing one factor still leaves 4 others
# - More importantly: the factor structure FORCES multi-factor output
#   Even if CRPS collapses some factors, the remaining ones maintain diversity
# - Residual proj starts at zero and stays near zero (Frobenius {res_frob:.6f})

# Measure: effective number of active factors
print(f"\nFactor activity analysis:")
factor_norms = np.linalg.norm(W, axis=0)
print(f"  Factor column norms: {factor_norms}")
print(f"  All factors active (>0.005): {np.sum(factor_norms > 0.005)} / 5")
print(f"  Min/Max ratio: {factor_norms.min()/factor_norms.max():.3f}")

# What fraction of skip output comes from factors vs residual?
torch.manual_seed(42)
z_test = torch.randn(10000, 32).to(DEVICE)
with torch.no_grad():
    z_factors = z_test[:, :5]
    z_residual = z_test[:, 5:]
    W_tensor = model_146b.frame_decoder.noise_skip_proj.W
    factor_out = z_factors @ W_tensor.T  # (10000, 25)
    residual_out = model_146b.frame_decoder.noise_skip_proj.residual_proj(z_residual)

print(f"\n  Factor output var:   {factor_out.var().item():.8f}")
print(f"  Residual output var: {residual_out.var().item():.8f}")
print(f"  Factor / Total:      {factor_out.var().item() / (factor_out.var().item() + residual_out.var().item()) * 100:.1f}%")

# Jacobian analysis: d(skip_output) / d(z) for both models
# For 146b: Jacobian = [W | residual_proj.weight.T] (approx, ignoring tanh)
# For 148a: Jacobian = weight.T
# The structure of the Jacobian determines collapse resistance

# 146b: factor part maps 5 noise dims -> 25 cells via W (25x5)
# residual part maps 27 noise dims -> 25 cells via residual_proj (25x27)
# Combined Jacobian: [W | residual_proj.weight] is (25, 32) total

print(f"\n  146b: W rank (numerical, tol=1e-4): {np.linalg.matrix_rank(W, tol=1e-4)}")
print(f"  148a: weight rank (numerical, tol=1e-4): {np.linalg.matrix_rank(w_148a, tol=1e-4)}")

# Combined Jacobian for 146b
J_146b = np.hstack([W, residual_w])  # (25, 5+27=32)
J_146b_sv = np.linalg.svd(J_146b, compute_uv=False)
J_148a_sv = S_148a  # already computed

print(f"\n  146b combined Jacobian SVD (top 5): {J_146b_sv[:5]}")
print(f"  148a skip weight SVD (top 5):       {J_148a_sv[:5]}")

# Condition number (ratio of largest to smallest nonzero SV)
J_146b_nz = J_146b_sv[J_146b_sv > 1e-6]
J_148a_nz = J_148a_sv[J_148a_sv > 1e-6]
print(f"\n  146b condition number (max/min SV): {J_146b_nz[0]/J_146b_nz[-1]:.2f}")
print(f"  148a condition number (max/min SV): {J_148a_nz[0]/J_148a_nz[-1]:.2f}")

# Key structural difference: 148a has rank ~1-2 after training
# meaning CRPS collapsed most singular vectors
# 146b keeps 5 factors active because they're STRUCTURALLY independent

# Per-cell contribution of each factor
print(f"\n  Per-cell factor contribution (variance from each factor):")
print(f"  {'Cell':<15} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8} {'Total':>8}")
for c in range(25):
    row = int(c / 5)
    col = c % 5
    label = f"K={MONEYNESS[row]},T={TENORS[col]}"
    factor_vars = W[c, :]**2  # variance from each factor for this cell
    total_fv = factor_vars.sum()
    vals = "  ".join(f"{fv:8.6f}" for fv in factor_vars)
    print(f"  {label:<15} {vals} {total_fv:8.6f}")


# ══════════════════════════════════════════════════════════════════════
# Summary JSON
# ══════════════════════════════════════════════════════════════════════

summary = {
    "analysis1_W_loadings": {
        "W_shape": list(W.shape),
        "singular_values": S.tolist(),
        "effective_rank_W": float(eff_rank),
        "residual_proj_frobenius": float(res_frob),
        "W_frobenius": float(W_frob),
        "gt_corr_vs_factor_corr": float(corr_gt_vs_factor),
    },
    "analysis2_skip_magnitude": {
        "146b_factor": {
            "frobenius_norm_W": float(W_frob),
            "frobenius_norm_residual": float(res_frob),
            "skip_output_mean_abs_random": float(out_146b_tanh.abs().mean().item()),
            "skip_output_mean_abs_inference": float(np.abs(skip_146b).mean()),
            "eff_rank_skip_correlation": float(eff_rank_corr_146b),
            "gt_corr_vs_skip_corr": float(corr_gt_vs_146b_skip),
        },
        "148a_plain": {
            "frobenius_norm": float(np.linalg.norm(w_148a)),
            "skip_output_mean_abs_random": float(out_148a_tanh.abs().mean().item()),
            "skip_output_mean_abs_inference": float(np.abs(skip_148a).mean()),
            "eff_rank_skip_correlation": float(eff_rank_corr_148a),
            "effective_rank_weights": float(eff_rank_148a),
            "S0_S1_ratio": float(S_148a[0]/max(S_148a[1], 1e-10)),
            "gt_corr_vs_skip_corr": float(corr_gt_vs_148a_skip),
        },
    },
    "analysis3_variance_decomposition": {
        "146b": {
            "decoder_var": float(dec_var_146b_total),
            "skip_var": float(skip_var_146b_total),
            "cov_2x": float(2 * cov_term_146b),
            "decoder_var_pct": float(100 * dec_var_146b_total / total_var_146b),
            "skip_var_pct": float(100 * skip_var_146b_total / total_var_146b),
        },
        "148a": {
            "decoder_var": float(dec_var_148a_total),
            "skip_var": float(skip_var_148a_total),
            "cov_2x": float(2 * cov_term_148a),
            "decoder_var_pct": float(100 * dec_var_148a_total / total_var_148a),
            "skip_var_pct": float(100 * skip_var_148a_total / total_var_148a),
        },
    },
    "analysis4_per_cell": {
        "146b_skip_frac_mean": float(skip_frac_146b.mean()),
        "148a_skip_frac_mean": float(skip_frac_148a.mean()),
        "eff_rank_by_horizon_146b": [float(x) for x in eff_rank_146b_by_h],
        "eff_rank_by_horizon_148a": [float(x) for x in eff_rank_148a_by_h],
    },
    "collapse_resistance": {
        "146b_all_factors_active": int(np.sum(factor_norms > 0.005)),
        "146b_factor_min_max_ratio": float(factor_norms.min()/factor_norms.max()),
        "146b_factor_var_pct": float(factor_out.var().item() / (factor_out.var().item() + residual_out.var().item()) * 100),
        "148a_weight_rank": int(np.linalg.matrix_rank(w_148a, tol=1e-4)),
        "148a_S0_S1_ratio": float(S_148a[0]/max(S_148a[1], 1e-10)),
    },
}

with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT_DIR}/summary.json")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
