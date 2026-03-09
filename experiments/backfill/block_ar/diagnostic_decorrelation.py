#!/usr/bin/env python
"""
Diagnostic analysis: Why does decorrelation hurt per-cell CI calibration?

Compares model 97a (no decorrelation, CI=93.6%) vs 99g (decorrelated, CI=85.7%).
Analyzes per-cell spread, coverage, noise sensitivity, spread/MAE ratio, and GT variance.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnostic_decorrelation.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
DEVICE = "cuda"
N_SAMPLES = 50
MAX_BATCHES = 20
BATCH_SIZE = 32
HORIZONS = [0, 6, 13, 29]  # 0-indexed: h=1, h=7, h=14, h=30
HORIZON_LABELS = ["h=1", "h=7", "h=14", "h=30"]
DATA_PATH = "data/vol_surface_with_ret.npz"
TEST_START = 4540
HISTORY_LEN = 30
FUTURE_LEN = 30

MODEL_97A = "models/backfill/afcrps_97a/best_model.pt"
MODEL_99G = "models/backfill/afcrps_99g/best_model.pt"

OUTPUT_FILE = "/tmp/decorrelation_analysis.txt"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_model(path: str, device: str) -> SinglePassBlockAR:
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    config = SinglePassConfig(**checkpoint["config"])
    model = SinglePassBlockAR(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model


def get_test_loader(batch_size: int) -> DataLoader:
    data = np.load(DATA_PATH)
    surfaces = data["surface"]
    # Normalize surfaces to [0, 1]
    surf_min, surf_max = surfaces.min(), surfaces.max()
    surfaces_norm = (surfaces - surf_min) / (surf_max - surf_min + 1e-8)
    dataset = VolSurfaceDataset(
        surfaces=surfaces_norm,
        history_len=HISTORY_LEN,
        future_len=FUTURE_LEN,
        start_idx=TEST_START,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def generate_samples(model, loader, n_samples, max_batches, device):
    """Generate samples and collect GT.

    Returns:
        samples: (N, n_samples, T, 5, 5) in [0, 1]
        gt: (N, T, 5, 5) in [0, 1]
        history: (N, H, 5, 5) in [-1, 1]
    """
    all_samples = []
    all_gt = []
    all_hist = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            samples = model.sample(history, n_samples=n_samples)
            # samples: (B, n_samples, T, 5, 5) in [0, 1]

            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_hist.append(history.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{max_batches}")

    return (
        np.concatenate(all_samples, axis=0),
        np.concatenate(all_gt, axis=0),
        np.concatenate(all_hist, axis=0),
    )


# ──────────────────────────────────────────────────────────────────────
# Analysis 1: Per-cell spread comparison
# ──────────────────────────────────────────────────────────────────────

def analysis_per_cell_spread(samples_97a, samples_99g, out):
    """For each cell and horizon, compute spread (std across samples)."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("ANALYSIS 1: Per-Cell Spread (std across ensemble members)\n")
    out.write("=" * 80 + "\n")

    for hi, (h_idx, h_label) in enumerate(zip(HORIZONS, HORIZON_LABELS)):
        out.write(f"\n--- {h_label} ---\n")

        # (N, n_samples, 5, 5) at this horizon
        s97 = samples_97a[:, :, h_idx, :, :]
        s99 = samples_99g[:, :, h_idx, :, :]

        # Per-window spread: std across n_samples dim, then mean across windows
        spread_97 = s97.std(axis=1).mean(axis=0)  # (5, 5)
        spread_99 = s99.std(axis=1).mean(axis=0)  # (5, 5)

        ratio = spread_99 / (spread_97 + 1e-10)

        out.write(f"  97a spread (mean across windows, x1000):\n")
        for r in range(5):
            vals = " ".join(f"{spread_97[r, c]*1000:6.2f}" for c in range(5))
            out.write(f"    [{vals}]\n")

        out.write(f"  99g spread (mean across windows, x1000):\n")
        for r in range(5):
            vals = " ".join(f"{spread_99[r, c]*1000:6.2f}" for c in range(5))
            out.write(f"    [{vals}]\n")

        out.write(f"  Ratio 99g/97a:\n")
        for r in range(5):
            vals = " ".join(f"{ratio[r, c]:6.3f}" for c in range(5))
            out.write(f"    [{vals}]\n")

        out.write(f"  Summary: 97a mean={spread_97.mean()*1000:.3f}, "
                  f"99g mean={spread_99.mean()*1000:.3f}, "
                  f"ratio mean={ratio.mean():.3f}, min={ratio.min():.3f}, max={ratio.max():.3f}\n")

    # Flat summary
    out.write(f"\n--- Flat spread summary (averaged over all horizons) ---\n")
    for model_name, samples in [("97a", samples_97a), ("99g", samples_99g)]:
        spreads = []
        for h_idx in HORIZONS:
            s = samples[:, :, h_idx, :, :]
            sp = s.std(axis=1).mean(axis=0)
            spreads.append(sp)
        mean_spread = np.mean(spreads, axis=0)
        out.write(f"  {model_name} avg spread (x1000):\n")
        for r in range(5):
            vals = " ".join(f"{mean_spread[r, c]*1000:6.2f}" for c in range(5))
            out.write(f"    [{vals}]\n")
        out.write(f"  mean={mean_spread.mean()*1000:.3f}, std={mean_spread.std()*1000:.3f}, "
                  f"CV={mean_spread.std()/mean_spread.mean():.3f}\n")


# ──────────────────────────────────────────────────────────────────────
# Analysis 2: Per-cell CI coverage heatmap
# ──────────────────────────────────────────────────────────────────────

def analysis_per_cell_ci(samples_97a, gt_97a, samples_99g, gt_99g, out):
    """90% CI coverage per (cell, horizon). Flag cells outside [70%, 95%]."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("ANALYSIS 2: Per-Cell 90% CI Coverage\n")
    out.write("=" * 80 + "\n")

    for model_name, samples, gt in [
        ("97a", samples_97a, gt_97a),
        ("99g", samples_99g, gt_99g),
    ]:
        out.write(f"\n{'=' * 40}\n")
        out.write(f"Model: {model_name}\n")
        out.write(f"{'=' * 40}\n")

        n_out_total = 0
        n_total = 0

        for hi, (h_idx, h_label) in enumerate(zip(HORIZONS, HORIZON_LABELS)):
            # samples: (N, n_samples, T, 5, 5), gt: (N, T, 5, 5)
            s = samples[:, :, h_idx, :, :]  # (N, n_samples, 5, 5)
            g = gt[:, h_idx, :, :]            # (N, 5, 5)

            lo = np.percentile(s, 5, axis=1)   # (N, 5, 5)
            hi_q = np.percentile(s, 95, axis=1)  # (N, 5, 5)

            covered = (g >= lo) & (g <= hi_q)  # (N, 5, 5)
            coverage = covered.mean(axis=0)  # (5, 5)

            out.write(f"\n  {h_label} coverage (%):\n")
            for r in range(5):
                vals = " ".join(f"{coverage[r, c]*100:5.1f}" for c in range(5))
                out.write(f"    [{vals}]\n")

            # Flag out-of-range cells
            out_cells = []
            for r in range(5):
                for c in range(5):
                    cov = coverage[r, c]
                    if cov < 0.70 or cov > 0.95:
                        direction = "LOW" if cov < 0.70 else "HIGH"
                        out_cells.append((r, c, cov, direction))
                        n_out_total += 1
                    n_total += 1

            if out_cells:
                out.write(f"  OUT OF RANGE [70%, 95%]:\n")
                for r, c, cov, direction in out_cells:
                    out.write(f"    cell({r},{c}): {cov*100:.1f}% ({direction})\n")
            else:
                out.write(f"  All cells in range.\n")

        out.write(f"\n  TOTAL: {n_out_total}/{n_total} (cell,horizon) pairs out of range\n")


# ──────────────────────────────────────────────────────────────────────
# Analysis 3: Noise sensitivity per cell (Jacobian analysis)
# ──────────────────────────────────────────────────────────────────────

def analysis_noise_sensitivity(model_97a, model_99g, loader, device, out):
    """Compute d(output)/d(noise) Jacobian norm for each cell.

    For a batch of histories, we run the frame_decoder with the same condition
    and prev_frame but enable gradients on the noise vector. We compute the
    Jacobian norm per cell to measure how sensitive each cell is to noise.
    """
    out.write("\n" + "=" * 80 + "\n")
    out.write("ANALYSIS 3: Noise Sensitivity per Cell (Jacobian |d(delta)/d(noise)|)\n")
    out.write("=" * 80 + "\n")

    n_batches = min(5, MAX_BATCHES)

    for model_name, model in [("97a", model_97a), ("99g", model_99g)]:
        out.write(f"\n--- Model: {model_name} ---\n")

        all_jac_norms = []  # per cell

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            history = batch["history"].to(device)
            B = history.shape[0]

            # Get condition and vol_scale
            with torch.no_grad():
                _, vol_scale = model._compute_vol_scale(history)
                condition = model.encoder(history, mask=None)
                prev_frame = denormalize_iv(history[:, -1])  # (B, 5, 5)
                prev_flat = prev_frame.reshape(B, 25)

            # Compute Jacobian: d(delta)/d(noise) for step 0
            noise_dim = model.config.noise_dim
            z = torch.randn(B, noise_dim, device=device, requires_grad=True)

            local_pos = torch.zeros(B, device=device, dtype=torch.long)
            horizon_bucket = None

            noise_input = model._get_noise_for_decoder(z)

            # Need condition without no_grad for the forward
            condition_for_jac = condition.detach()
            prev_flat_for_jac = prev_flat.detach()

            delta = model.frame_decoder(
                prev_flat_for_jac, condition_for_jac, noise_input, local_pos, horizon_bucket
            )  # (B, 25)

            # Compute per-cell Jacobian norms
            cell_jac_norms = torch.zeros(B, 25, device=device)
            for cell_idx in range(25):
                # Sum over batch to get scalar for backward
                grad_outputs = torch.zeros_like(delta)
                grad_outputs[:, cell_idx] = 1.0
                grads = torch.autograd.grad(
                    delta, z, grad_outputs=grad_outputs,
                    retain_graph=(cell_idx < 24), create_graph=False
                )[0]  # (B, noise_dim)
                cell_jac_norms[:, cell_idx] = grads.norm(dim=-1)

            all_jac_norms.append(cell_jac_norms.detach().cpu().numpy())

        jac_norms = np.concatenate(all_jac_norms, axis=0)  # (N, 25)
        mean_jac = jac_norms.mean(axis=0).reshape(5, 5)
        std_jac = jac_norms.std(axis=0).reshape(5, 5)

        out.write(f"  Mean Jacobian norm per cell:\n")
        for r in range(5):
            vals = " ".join(f"{mean_jac[r, c]:.4f}" for c in range(5))
            out.write(f"    [{vals}]\n")

        out.write(f"  Std of Jacobian norm per cell:\n")
        for r in range(5):
            vals = " ".join(f"{std_jac[r, c]:.4f}" for c in range(5))
            out.write(f"    [{vals}]\n")

        # Effective rank of Jacobian (how many independent noise directions matter)
        # Use singular values of the mean Jacobian across cells
        flat_mean = mean_jac.flatten()
        out.write(f"  Jacobian norm: mean={flat_mean.mean():.4f}, std={flat_mean.std():.4f}, "
                  f"CV={flat_mean.std()/flat_mean.mean():.4f}\n")
        out.write(f"  Range: min={flat_mean.min():.4f}, max={flat_mean.max():.4f}, "
                  f"max/min={flat_mean.max()/flat_mean.min():.3f}\n")

    # Also compute full Jacobian matrix and its singular values for rank analysis
    out.write(f"\n--- Full Jacobian rank analysis ---\n")
    for model_name, model in [("97a", model_97a), ("99g", model_99g)]:
        # Use first batch only
        batch = next(iter(loader))
        history = batch["history"].to(device)
        B = history.shape[0]

        with torch.no_grad():
            condition = model.encoder(history, mask=None)
            prev_frame = denormalize_iv(history[:, -1])
            prev_flat = prev_frame.reshape(B, 25)

        noise_dim = model.config.noise_dim
        # Compute full B x 25 x noise_dim Jacobian for first 8 samples
        n_samp = min(8, B)
        all_svs = []

        for bi in range(n_samp):
            z = torch.randn(1, noise_dim, device=device, requires_grad=True)
            local_pos = torch.zeros(1, device=device, dtype=torch.long)
            noise_input = model._get_noise_for_decoder(z)

            delta = model.frame_decoder(
                prev_flat[bi:bi+1].detach(), condition[bi:bi+1].detach(),
                noise_input, local_pos, None
            )

            jac = torch.zeros(25, noise_dim, device=device)
            for cell_idx in range(25):
                grad_outputs = torch.zeros_like(delta)
                grad_outputs[0, cell_idx] = 1.0
                grads = torch.autograd.grad(
                    delta, z, grad_outputs=grad_outputs,
                    retain_graph=(cell_idx < 24), create_graph=False
                )[0]
                jac[cell_idx] = grads[0]

            svs = torch.linalg.svdvals(jac).cpu().numpy()
            all_svs.append(svs)

        svs_arr = np.array(all_svs)  # (n_samp, min(25, noise_dim))
        mean_svs = svs_arr.mean(axis=0)
        # Participation ratio
        pr = (mean_svs.sum() ** 2) / (mean_svs ** 2).sum()

        out.write(f"  {model_name}: Singular values (top 8): {', '.join(f'{s:.4f}' for s in mean_svs[:8])}\n")
        out.write(f"  {model_name}: Participation ratio = {pr:.3f}\n")
        out.write(f"  {model_name}: SV ratio (s1/s2) = {mean_svs[0]/mean_svs[1]:.3f}\n")

    # Also check noise_skip_proj weights for 99g
    out.write(f"\n--- Noise skip projection analysis (99g only) ---\n")
    if hasattr(model_99g.frame_decoder, 'noise_skip_proj') and model_99g.frame_decoder.noise_skip_proj is not None:
        skip_w = model_99g.frame_decoder.noise_skip_proj.weight.detach().cpu().numpy()  # (25, noise_dim)
        out.write(f"  noise_skip_proj weight shape: {skip_w.shape}\n")
        skip_norms = np.linalg.norm(skip_w, axis=1)
        out.write(f"  Per-cell projection norms:\n")
        for r in range(5):
            vals = " ".join(f"{skip_norms[r*5+c]:.4f}" for c in range(5))
            out.write(f"    [{vals}]\n")
        out.write(f"  Norm range: min={skip_norms.min():.4f}, max={skip_norms.max():.4f}, "
                  f"max/min={skip_norms.max()/skip_norms.min():.3f}\n")

        # Cross-cell cosine similarity of skip projections
        skip_w_norm = skip_w / (np.linalg.norm(skip_w, axis=1, keepdims=True) + 1e-10)
        cos_sim = skip_w_norm @ skip_w_norm.T
        # Upper triangle mean
        triu_idx = np.triu_indices(25, k=1)
        mean_cos = cos_sim[triu_idx].mean()
        out.write(f"  Mean pairwise cosine similarity: {mean_cos:.4f}\n")

        # SVD of skip weights
        skip_svs = np.linalg.svd(skip_w, compute_uv=False)
        skip_pr = (skip_svs.sum() ** 2) / ((skip_svs ** 2).sum() + 1e-10)
        out.write(f"  SVD of skip weights: top 5 = {', '.join(f'{s:.4f}' for s in skip_svs[:5])}\n")
        out.write(f"  Skip weight participation ratio: {skip_pr:.3f}\n")
    else:
        out.write(f"  No noise_skip_proj found in 99g\n")

    # Check cell_spread_linear for 99g
    out.write(f"\n--- Cell spread analysis (99g only) ---\n")
    if hasattr(model_99g, 'cell_spread_linear'):
        # Evaluate cell_spread on a batch of conditions
        batch = next(iter(loader))
        history = batch["history"].to(device)
        B = history.shape[0]
        with torch.no_grad():
            condition = model_99g.encoder(history, mask=None)
            pos_emb = model_99g.frame_decoder.pos_embed(
                torch.zeros(B, device=device, dtype=torch.long)
            )
            cs_in = torch.cat([condition, pos_emb], dim=-1)
            cs = F.softplus(model_99g.cell_spread_linear(cs_in))  # (B, 25)

        cs_np = cs.cpu().numpy()  # (B, 25)
        mean_cs = cs_np.mean(axis=0).reshape(5, 5)
        std_cs = cs_np.std(axis=0).reshape(5, 5)

        out.write(f"  Mean cell spread multiplier:\n")
        for r in range(5):
            vals = " ".join(f"{mean_cs[r, c]:.4f}" for c in range(5))
            out.write(f"    [{vals}]\n")
        out.write(f"  Std of cell spread multiplier:\n")
        for r in range(5):
            vals = " ".join(f"{std_cs[r, c]:.4f}" for c in range(5))
            out.write(f"    [{vals}]\n")
        out.write(f"  Range: min={mean_cs.min():.4f}, max={mean_cs.max():.4f}\n")
    else:
        out.write(f"  No cell_spread_linear in 99g\n")


# ──────────────────────────────────────────────────────────────────────
# Analysis 4: Spread-to-MAE ratio per cell
# ──────────────────────────────────────────────────────────────────────

def analysis_spread_mae_ratio(samples_97a, gt_97a, samples_99g, gt_99g, out):
    """Spread/MAE per cell. Well-calibrated model has uniform ratio."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("ANALYSIS 4: Spread / MAE Ratio Per Cell\n")
    out.write("=" * 80 + "\n")

    for model_name, samples, gt in [
        ("97a", samples_97a, gt_97a),
        ("99g", samples_99g, gt_99g),
    ]:
        out.write(f"\n--- Model: {model_name} ---\n")

        for hi, (h_idx, h_label) in enumerate(zip(HORIZONS, HORIZON_LABELS)):
            s = samples[:, :, h_idx, :, :]  # (N, n_samples, 5, 5)
            g = gt[:, h_idx, :, :]            # (N, 5, 5)

            # Spread: std across ensemble
            spread = s.std(axis=1)  # (N, 5, 5)
            mean_spread = spread.mean(axis=0)  # (5, 5)

            # MAE: |ensemble_mean - gt|
            ens_mean = s.mean(axis=1)  # (N, 5, 5)
            mae = np.abs(ens_mean - g).mean(axis=0)  # (5, 5)

            ratio = mean_spread / (mae + 1e-10)

            out.write(f"\n  {h_label}:\n")
            out.write(f"    Spread/MAE ratio:\n")
            for r in range(5):
                vals = " ".join(f"{ratio[r, c]:6.3f}" for c in range(5))
                out.write(f"      [{vals}]\n")
            out.write(f"    mean={ratio.mean():.3f}, std={ratio.std():.3f}, "
                      f"CV={ratio.std()/ratio.mean():.3f}\n")

        # Summary across all horizons
        all_spreads = []
        all_maes = []
        for h_idx in HORIZONS:
            s = samples[:, :, h_idx, :, :]
            g = gt[:, h_idx, :, :]
            all_spreads.append(s.std(axis=1).mean(axis=0))
            all_maes.append(np.abs(s.mean(axis=1) - g).mean(axis=0))
        avg_spread = np.mean(all_spreads, axis=0)
        avg_mae = np.mean(all_maes, axis=0)
        avg_ratio = avg_spread / (avg_mae + 1e-10)
        out.write(f"\n  Average across horizons:\n")
        out.write(f"    Spread/MAE ratio:\n")
        for r in range(5):
            vals = " ".join(f"{avg_ratio[r, c]:6.3f}" for c in range(5))
            out.write(f"      [{vals}]\n")
        out.write(f"    mean={avg_ratio.mean():.3f}, CV={avg_ratio.std()/avg_ratio.mean():.3f}\n")


# ──────────────────────────────────────────────────────────────────────
# Analysis 5: GT per-cell daily change variance
# ──────────────────────────────────────────────────────────────────────

def analysis_gt_variance(gt_97a, samples_97a, samples_99g, out):
    """GT per-cell daily change variance. Check if 99g failures correlate with extreme cells."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("ANALYSIS 5: Ground Truth Per-Cell Daily Change Variance\n")
    out.write("=" * 80 + "\n")

    # GT daily changes across all windows
    # gt shape: (N, T, 5, 5) where T=30
    gt_daily = gt_97a[:, 1:, :, :] - gt_97a[:, :-1, :, :]  # (N, 29, 5, 5)

    # Per-cell variance of daily changes
    gt_var = gt_daily.var(axis=(0, 1))  # (5, 5)
    gt_std = np.sqrt(gt_var)

    out.write(f"\n  GT daily change std (x1000):\n")
    for r in range(5):
        vals = " ".join(f"{gt_std[r, c]*1000:6.3f}" for c in range(5))
        out.write(f"    [{vals}]\n")

    out.write(f"\n  GT daily change variance (x1e6):\n")
    for r in range(5):
        vals = " ".join(f"{gt_var[r, c]*1e6:6.3f}" for c in range(5))
        out.write(f"    [{vals}]\n")

    # Rank cells by GT variance
    flat_var = gt_var.flatten()
    ranked = np.argsort(flat_var)
    out.write(f"\n  Cells ranked by GT variance (low to high):\n")
    for i, idx in enumerate(ranked):
        r, c = divmod(idx, 5)
        out.write(f"    {i+1}. cell({r},{c}): var={flat_var[idx]*1e6:.3f}e-6, "
                  f"std={np.sqrt(flat_var[idx])*1000:.3f}e-3\n")

    # Now compare model spread / GT std ratio for each cell
    out.write(f"\n  Model spread / GT std ratio (should be ~1.0 for well-calibrated):\n")
    for model_name, samples in [("97a", samples_97a), ("99g", samples_99g)]:
        spreads = []
        for h_idx in HORIZONS:
            s = samples[:, :, h_idx, :, :]
            sp = s.std(axis=1).mean(axis=0)
            spreads.append(sp)
        avg_spread = np.mean(spreads, axis=0)

        # GT std at various horizons (std of cumulative changes)
        gt_cum_std = []
        for h_idx in HORIZONS:
            cum_chg = gt_97a[:, h_idx, :, :] - gt_97a[:, 0, :, :]
            gt_cum_std.append(cum_chg.std(axis=0))
        avg_gt_std = np.mean(gt_cum_std, axis=0)

        ratio = avg_spread / (avg_gt_std + 1e-10)
        out.write(f"\n    {model_name}:\n")
        for r in range(5):
            vals = " ".join(f"{ratio[r, c]:6.3f}" for c in range(5))
            out.write(f"      [{vals}]\n")
        out.write(f"    mean={ratio.mean():.3f}, CV={ratio.std()/ratio.mean():.3f}\n")

    # Correlation between GT variance rank and coverage failures
    out.write(f"\n--- Correlation: GT variance vs CI coverage ---\n")
    for model_name, samples, gt in [
        ("97a", samples_97a, gt_97a),
        ("99g", samples_99g, gt_97a),
    ]:
        # Per-cell average coverage across horizons
        coverages = []
        for h_idx in HORIZONS:
            s = samples[:, :, h_idx, :, :]
            g = gt[:, h_idx, :, :]
            lo = np.percentile(s, 5, axis=1)
            hi_q = np.percentile(s, 95, axis=1)
            covered = ((g >= lo) & (g <= hi_q)).mean(axis=0)
            coverages.append(covered)
        avg_cov = np.mean(coverages, axis=0).flatten()
        gt_var_flat = gt_var.flatten()

        # Compute rank correlation
        from scipy.stats import spearmanr
        corr, pval = spearmanr(gt_var_flat, avg_cov)
        out.write(f"  {model_name}: Spearman corr(GT_var, coverage) = {corr:.3f} (p={pval:.4f})\n")

    # Also check which cells are high/low variance AND out of range for 99g
    out.write(f"\n--- 99g: Out-of-range cells mapped to GT variance rank ---\n")
    gt_var_rank = np.argsort(np.argsort(gt_var.flatten()))  # rank (0=lowest var)
    for h_idx, h_label in zip(HORIZONS, HORIZON_LABELS):
        s = samples_99g[:, :, h_idx, :, :]
        g = gt_97a[:, h_idx, :, :]
        lo = np.percentile(s, 5, axis=1)
        hi_q = np.percentile(s, 95, axis=1)
        covered = ((g >= lo) & (g <= hi_q)).mean(axis=0)
        for r in range(5):
            for c in range(5):
                cov = covered[r, c]
                if cov < 0.70 or cov > 0.95:
                    direction = "UNDER" if cov < 0.70 else "OVER"
                    rank = gt_var_rank[r * 5 + c]
                    out.write(f"    {h_label} cell({r},{c}): cov={cov*100:.1f}% ({direction}), "
                              f"GT var rank={rank+1}/25 "
                              f"({'HIGH var' if rank >= 20 else 'MED var' if rank >= 10 else 'LOW var'})\n")


# ──────────────────────────────────────────────────────────────────────
# Analysis 6: Cross-cell correlation comparison
# ──────────────────────────────────────────────────────────────────────

def analysis_cross_cell_correlation(samples_97a, samples_99g, gt_97a, out):
    """Compare cross-cell correlation structure between models and GT."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("ANALYSIS 6: Cross-Cell Correlation of Daily Changes\n")
    out.write("=" * 80 + "\n")

    # GT correlation: daily changes across cells
    gt_daily = gt_97a[:, 1:, :, :] - gt_97a[:, :-1, :, :]  # (N, 29, 5, 5)
    gt_flat = gt_daily.reshape(-1, 25)  # (N*29, 25)
    gt_corr = np.corrcoef(gt_flat.T)  # (25, 25)
    triu = np.triu_indices(25, k=1)
    gt_mean_corr = gt_corr[triu].mean()

    out.write(f"\n  GT mean cross-cell correlation: {gt_mean_corr:.4f}\n")

    for model_name, samples in [("97a", samples_97a), ("99g", samples_99g)]:
        # Use daily changes from generated samples
        # samples: (N, n_samples, T, 5, 5)
        daily = samples[:, :, 1:, :, :] - samples[:, :, :-1, :, :]  # (N, ns, 29, 5, 5)
        # Flatten to get all daily changes: (N*ns*29, 25)
        flat = daily.reshape(-1, 25)
        model_corr = np.corrcoef(flat.T)
        model_mean_corr = model_corr[triu].mean()
        out.write(f"  {model_name} mean cross-cell correlation: {model_mean_corr:.4f}\n")

    # Per-cell pair correlation differences
    out.write(f"\n  Correlation difference from GT (selected pairs):\n")
    pairs = [(0, 0, 4, 4), (0, 0, 0, 4), (2, 2, 0, 0), (2, 2, 4, 4), (0, 2, 4, 2)]
    for r1, c1, r2, c2 in pairs:
        i1, i2 = r1 * 5 + c1, r2 * 5 + c2
        gt_c = gt_corr[i1, i2]

        daily_97 = samples_97a[:, :, 1:, :, :] - samples_97a[:, :, :-1, :, :]
        flat_97 = daily_97.reshape(-1, 25)
        c97 = np.corrcoef(flat_97.T)[i1, i2]

        daily_99 = samples_99g[:, :, 1:, :, :] - samples_99g[:, :, :-1, :, :]
        flat_99 = daily_99.reshape(-1, 25)
        c99 = np.corrcoef(flat_99.T)[i1, i2]

        out.write(f"    ({r1},{c1})-({r2},{c2}): GT={gt_c:.3f}, 97a={c97:.3f}, 99g={c99:.3f}\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    model_97a = load_model(MODEL_97A, DEVICE)
    model_99g = load_model(MODEL_99G, DEVICE)
    print(f"  97a config diffs: ar_frame_cell_spread={model_97a.config.ar_frame_cell_spread}, "
          f"ar_noise_skip={getattr(model_97a.config, 'ar_noise_skip', False)}")
    print(f"  99g config diffs: ar_frame_cell_spread={model_99g.config.ar_frame_cell_spread}, "
          f"ar_noise_skip={getattr(model_99g.config, 'ar_noise_skip', False)}")

    print("\nLoading test data...")
    loader = get_test_loader(BATCH_SIZE)
    print(f"  Test loader: {len(loader)} batches, batch_size={BATCH_SIZE}")

    print("\nGenerating samples for 97a...")
    samples_97a, gt_97a, hist_97a = generate_samples(
        model_97a, loader, N_SAMPLES, MAX_BATCHES, DEVICE
    )
    print(f"  97a: samples={samples_97a.shape}, gt={gt_97a.shape}")

    # Re-create loader (iterator exhausted)
    loader = get_test_loader(BATCH_SIZE)
    print("\nGenerating samples for 99g...")
    samples_99g, gt_99g, hist_99g = generate_samples(
        model_99g, loader, N_SAMPLES, MAX_BATCHES, DEVICE
    )
    print(f"  99g: samples={samples_99g.shape}, gt={gt_99g.shape}")

    # Truncate to same size (in case different loader lengths)
    n_min = min(samples_97a.shape[0], samples_99g.shape[0])
    samples_97a = samples_97a[:n_min]
    gt_97a = gt_97a[:n_min]
    samples_99g = samples_99g[:n_min]
    gt_99g = gt_99g[:n_min]

    print(f"\nUsing {n_min} windows for analysis")

    # Write results
    with open(OUTPUT_FILE, "w") as out:
        out.write("DECORRELATION DIAGNOSTIC ANALYSIS\n")
        out.write(f"97a: {MODEL_97A}\n")
        out.write(f"99g: {MODEL_99G}\n")
        out.write(f"Windows: {n_min}, Samples: {N_SAMPLES}\n")

        # Overall CI coverage for both models
        out.write(f"\n{'=' * 80}\n")
        out.write("OVERALL 90% CI COVERAGE\n")
        out.write(f"{'=' * 80}\n")
        for model_name, samples, gt in [
            ("97a", samples_97a, gt_97a),
            ("99g", samples_99g, gt_99g),
        ]:
            lo = np.percentile(samples, 5, axis=1)
            hi_q = np.percentile(samples, 95, axis=1)
            covered = (gt >= lo) & (gt <= hi_q)
            overall = covered.mean()
            out.write(f"  {model_name}: {overall*100:.1f}%\n")

            for h_idx, h_label in zip(HORIZONS, HORIZON_LABELS):
                lo_h = np.percentile(samples[:, :, h_idx, :, :], 5, axis=1)
                hi_h = np.percentile(samples[:, :, h_idx, :, :], 95, axis=1)
                cov_h = ((gt[:, h_idx, :, :] >= lo_h) & (gt[:, h_idx, :, :] <= hi_h)).mean()
                out.write(f"    {h_label}: {cov_h*100:.1f}%\n")

        print("\nRunning Analysis 1: Per-cell spread...")
        analysis_per_cell_spread(samples_97a, samples_99g, out)

        print("Running Analysis 2: Per-cell CI coverage...")
        analysis_per_cell_ci(samples_97a, gt_97a, samples_99g, gt_99g, out)

        # Reload loader for Jacobian analysis
        loader = get_test_loader(BATCH_SIZE)
        print("Running Analysis 3: Noise sensitivity (Jacobian)...")
        analysis_noise_sensitivity(model_97a, model_99g, loader, DEVICE, out)

        print("Running Analysis 4: Spread/MAE ratio...")
        analysis_spread_mae_ratio(samples_97a, gt_97a, samples_99g, gt_99g, out)

        print("Running Analysis 5: GT variance...")
        analysis_gt_variance(gt_97a, samples_97a, samples_99g, out)

        print("Running Analysis 6: Cross-cell correlation...")
        analysis_cross_cell_correlation(samples_97a, samples_99g, gt_97a, out)

        # Final summary
        out.write("\n" + "=" * 80 + "\n")
        out.write("SUMMARY\n")
        out.write("=" * 80 + "\n")

        # Compute key stats for summary
        # 1. Spread uniformity (CV of spread across cells)
        for model_name, samples in [("97a", samples_97a), ("99g", samples_99g)]:
            all_sp = []
            for h_idx in HORIZONS:
                sp = samples[:, :, h_idx, :, :].std(axis=1).mean(axis=0)
                all_sp.append(sp)
            avg_sp = np.mean(all_sp, axis=0)
            cv = avg_sp.std() / avg_sp.mean()
            out.write(f"  {model_name} spread CV across cells: {cv:.4f}\n")

        # 2. Coverage uniformity
        for model_name, samples, gt in [
            ("97a", samples_97a, gt_97a),
            ("99g", samples_99g, gt_99g),
        ]:
            all_cov = []
            for h_idx in HORIZONS:
                s = samples[:, :, h_idx, :, :]
                g = gt[:, h_idx, :, :]
                lo = np.percentile(s, 5, axis=1)
                hi_q = np.percentile(s, 95, axis=1)
                cov = ((g >= lo) & (g <= hi_q)).mean(axis=0)
                all_cov.append(cov)
            avg_cov = np.mean(all_cov, axis=0)
            cv_cov = avg_cov.std() / avg_cov.mean()
            out.write(f"  {model_name} coverage CV across cells: {cv_cov:.4f}\n")
            out.write(f"  {model_name} coverage range: [{avg_cov.min()*100:.1f}%, {avg_cov.max()*100:.1f}%]\n")

    print(f"\nResults saved to {OUTPUT_FILE}")
    print("\n--- Quick summary ---")
    with open(OUTPUT_FILE) as f:
        # Print first 50 lines and last 30 lines
        lines = f.readlines()
        for line in lines[:60]:
            print(line, end="")
        if len(lines) > 90:
            print(f"\n... ({len(lines) - 90} lines omitted) ...\n")
        for line in lines[-30:]:
            print(line, end="")


if __name__ == "__main__":
    main()
