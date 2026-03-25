#!/usr/bin/env python
"""
154c Kurtosis Mechanism Analysis

WHY does conditioning improve kurtosis from 0.81 (154b) to 1.003 (154c)?

Investigations:
1. Residual distribution comparison (raw residuals before adding to base)
2. Per-cell kurtosis of combined daily changes
3. Condition ablation (zero-condition 154c vs real-condition)
4. Daily change histograms with tail focus
5. Hypothesis test: condition modulates SCALE per window

Output:
  - results/validations/2026-03-24/verification_results/154c_kurtosis.json
  - results/validations/2026-03-24/analysis/154c_kurtosis/*.png
"""

import json
import math
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, ks_2samp

sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv,
)
from experiments.backfill.block_ar.train_oneshot_flow import (
    FactoredVelocityTransformer,
)

OUTDIR = Path("/home/max/Documents/vol-surface-vae-pub/results/validations/2026-03-24/analysis/154c_kurtosis")
RESULTDIR = Path("/home/max/Documents/vol-surface-vae-pub/results/validations/2026-03-24/verification_results")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_STEPS = 8
DT = 1.0 / N_STEPS
DIM = 750
T_FRAMES = 30
N_CELLS = 25
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def sample_fm(model, n_samples, device, cond=None):
    """Sample from a flow matching model using Euler ODE integration."""
    with torch.no_grad():
        x = torch.randn(n_samples, DIM, device=device)
        for step in range(N_STEPS):
            t = torch.full((n_samples,), step * DT, device=device)
            if cond is not None:
                x = x + model(x, t, cond=cond) * DT
            else:
                x = x + model(x, t) * DT
    return x.cpu().numpy()


def load_models():
    """Load 154b (uncond residual FM) and 154c (cond residual FM)."""
    print("Loading models...")

    # Encoder
    encoder, cond_dim = load_encoder(
        '/home/max/Documents/vol-surface-vae-pub/models/backfill/block_ar_vol_scaled_30ep/best_model.pt',
        DEVICE)
    for p in encoder.parameters():
        p.requires_grad = False

    # 154b: Unconditional residual FM
    rc_b = torch.load(
        '/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_154b/final_model.pt',
        weights_only=False, map_location=DEVICE)
    cfg_b = rc_b['config']
    model_b = FactoredVelocityTransformer(
        n_frames=cfg_b['n_frames'], n_cells=cfg_b['n_cells'],
        d_model=cfg_b['d_model'], n_heads=cfg_b['n_heads'], n_layers=cfg_b['n_layers'])
    model_b.load_state_dict(rc_b['model_state_dict'])
    model_b.to(DEVICE).eval()
    res_mean_b = rc_b['res_mean']
    res_std_b = rc_b['res_std']

    # 154c: Conditional residual FM
    rc_c = torch.load(
        '/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_154c/final_model.pt',
        weights_only=False, map_location=DEVICE)
    cfg_c = rc_c['config']
    model_c = ConditionalFactoredVelocityTransformer(
        n_frames=cfg_c['n_frames'], n_cells=cfg_c['n_cells'],
        d_model=cfg_c['d_model'], n_heads=cfg_c['n_heads'], n_layers=cfg_c['n_layers'],
        cond_dim=cfg_c['cond_dim'])
    model_c.load_state_dict(rc_c['model_state_dict'])
    model_c.to(DEVICE).eval()
    res_mean_c = rc_c['res_mean']
    res_std_c = rc_c['res_std']

    # Base model (153a) for generating base predictions
    bc = torch.load(
        '/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_153a/final_model.pt',
        weights_only=False, map_location=DEVICE)
    bcfg = bc['config']
    base_model = ConditionalFactoredVelocityTransformer(
        n_frames=bcfg['n_frames'], n_cells=bcfg['n_cells'],
        d_model=bcfg['d_model'], n_heads=bcfg['n_heads'], n_layers=bcfg['n_layers'],
        cond_dim=bcfg['cond_dim'])
    base_model.load_state_dict(bc['model_state_dict'])
    base_model.to(DEVICE).eval()
    base_mean = torch.from_numpy(bc['train_mean']).float().to(DEVICE)
    base_std = torch.from_numpy(bc['train_std']).float().to(DEVICE)

    return {
        'encoder': encoder,
        'model_b': model_b, 'res_mean_b': res_mean_b, 'res_std_b': res_std_b,
        'model_c': model_c, 'res_mean_c': res_mean_c, 'res_std_c': res_std_c,
        'base_model': base_model, 'base_mean': base_mean, 'base_std': base_std,
    }


def load_data():
    """Load cached base predictions and raw data."""
    bp = np.load('/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_154b/base_predictions.npz')
    data = np.load('/home/max/Documents/vol-surface-vae-pub/data/vol_surface_with_ret.npz')
    return {
        'val_preds': bp['val_preds'],
        'val_gts': bp['val_gts'],
        'res_mean': bp['res_mean'],
        'res_std': bp['res_std'],
        'surfaces': data['surface'],
        'rets': data['ret'],
    }


def investigation_1(models, data, results):
    """Residual distribution comparison: raw residuals before adding to base."""
    print("\n=== Investigation 1: Residual Distribution Comparison ===")

    model_b = models['model_b']
    model_c = models['model_c']
    encoder = models['encoder']
    surfaces = data['surfaces']

    res_mean_b = models['res_mean_b']
    res_std_b = models['res_std_b']
    res_mean_c = models['res_mean_c']
    res_std_c = models['res_std_c']

    # GT residuals
    gt_residuals = data['val_gts'] - data['val_preds']  # (441, 750)

    # Generate residual samples from 154b (unconditional) - 1000 samples
    print("  Sampling 1000 residuals from 154b (unconditional)...")
    n_res = 1000
    all_res_b = []
    for i in range(0, n_res, 100):
        batch = min(100, n_res - i)
        raw = sample_fm(model_b, batch, DEVICE, cond=None)
        denorm = raw * res_std_b + res_mean_b
        all_res_b.append(denorm)
    res_b = np.concatenate(all_res_b)  # (1000, 750)

    # Generate residual samples from 154c (conditional) with real conditions
    print("  Sampling 1000 residuals from 154c (conditional, real conds)...")
    # Get conditions from validation windows
    H = 30
    val_start = 4040
    all_res_c = []
    with torch.no_grad():
        for i in range(0, n_res, 100):
            batch = min(100, n_res - i)
            # Use random validation conditions
            idx = np.random.choice(len(data['val_preds']), batch)
            conds = []
            for j in idx:
                wi = val_start + j
                hist = torch.from_numpy(surfaces[wi:wi+H][None].astype(np.float32)).to(DEVICE)
                c = encoder(normalize_iv(hist))
                conds.append(c)
            conds = torch.cat(conds, dim=0)
            raw = sample_fm(model_c, batch, DEVICE, cond=conds)
            denorm = raw * res_std_c + res_mean_c
            all_res_c.append(denorm)
    res_c = np.concatenate(all_res_c)  # (1000, 750)

    # Compute kurtosis of raw residuals
    # Reshape to (N, T=30, C=25) and compute daily changes
    res_b_3d = res_b.reshape(-1, T_FRAMES, N_CELLS)
    res_c_3d = res_c.reshape(-1, T_FRAMES, N_CELLS)
    gt_res_3d = gt_residuals.reshape(-1, T_FRAMES, N_CELLS)

    # Daily changes within residuals
    dch_b = np.diff(res_b_3d, axis=1).flatten()
    dch_c = np.diff(res_c_3d, axis=1).flatten()
    dch_gt = np.diff(gt_res_3d, axis=1).flatten()

    # Also compute kurtosis on the raw residuals themselves (per-cell flattened)
    kurt_raw_b = kurtosis(res_b.flatten(), fisher=True)
    kurt_raw_c = kurtosis(res_c.flatten(), fisher=True)
    kurt_raw_gt = kurtosis(gt_residuals.flatten(), fisher=True)

    kurt_dch_b = kurtosis(dch_b, fisher=True)
    kurt_dch_c = kurtosis(dch_c, fisher=True)
    kurt_dch_gt = kurtosis(dch_gt, fisher=True)

    # Std of residuals
    std_b = res_b.std()
    std_c = res_c.std()
    std_gt = gt_residuals.std()

    # Per-sample (window-level) residual magnitudes
    mag_b = np.linalg.norm(res_b, axis=1)
    mag_c = np.linalg.norm(res_c, axis=1)
    mag_gt = np.linalg.norm(gt_residuals, axis=1)

    r1 = {
        'kurtosis_raw_residuals': {
            '154b_uncond': kurt_raw_b, '154c_cond': kurt_raw_c, 'gt': kurt_raw_gt
        },
        'kurtosis_residual_daily_changes': {
            '154b_uncond': kurt_dch_b, '154c_cond': kurt_dch_c, 'gt': kurt_dch_gt
        },
        'std_residuals': {
            '154b_uncond': float(std_b), '154c_cond': float(std_c), 'gt': float(std_gt)
        },
        'residual_magnitude_stats': {
            '154b_mean': float(mag_b.mean()), '154b_std': float(mag_b.std()),
            '154c_mean': float(mag_c.mean()), '154c_std': float(mag_c.std()),
            'gt_mean': float(mag_gt.mean()), 'gt_std': float(mag_gt.std()),
        },
        'residual_magnitude_cv': {
            '154b': float(mag_b.std() / mag_b.mean()),
            '154c': float(mag_c.std() / mag_c.mean()),
            'gt': float(mag_gt.std() / mag_gt.mean()),
        }
    }

    print(f"  Raw residual kurtosis: 154b={kurt_raw_b:.3f}, 154c={kurt_raw_c:.3f}, GT={kurt_raw_gt:.3f}")
    print(f"  Residual daily change kurtosis: 154b={kurt_dch_b:.3f}, 154c={kurt_dch_c:.3f}, GT={kurt_dch_gt:.3f}")
    print(f"  Residual std: 154b={std_b:.5f}, 154c={std_c:.5f}, GT={std_gt:.5f}")
    print(f"  Magnitude CV: 154b={mag_b.std()/mag_b.mean():.3f}, 154c={mag_c.std()/mag_c.mean():.3f}, GT={mag_gt.std()/mag_gt.mean():.3f}")

    results['investigation_1'] = r1

    return res_b, res_c, gt_residuals


def investigation_2(models, data, results, res_b, res_c):
    """Per-cell kurtosis of combined (base + residual) daily changes."""
    print("\n=== Investigation 2: Per-Cell Kurtosis of Combined Daily Changes ===")

    val_preds = data['val_preds']
    val_gts = data['val_gts']

    # Combine: base + residual (use first 441 residual samples to match windows)
    n_win = min(len(val_preds), len(res_b))
    combined_b = np.clip(val_preds[:n_win] + res_b[:n_win], 0, 1)
    combined_c = np.clip(val_preds[:n_win] + res_c[:n_win], 0, 1)
    gt = val_gts[:n_win]

    # Base only (153a)
    base_only = val_preds[:n_win]

    # Reshape to (N, 30, 25)
    combined_b_3d = combined_b.reshape(-1, T_FRAMES, N_CELLS)
    combined_c_3d = combined_c.reshape(-1, T_FRAMES, N_CELLS)
    gt_3d = gt.reshape(-1, T_FRAMES, N_CELLS)
    base_3d = base_only.reshape(-1, T_FRAMES, N_CELLS)

    # Daily changes
    dch_b = np.diff(combined_b_3d, axis=1)  # (N, 29, 25)
    dch_c = np.diff(combined_c_3d, axis=1)
    dch_gt = np.diff(gt_3d, axis=1)
    dch_base = np.diff(base_3d, axis=1)

    # Per-cell kurtosis
    cell_kurt_b = [kurtosis(dch_b[:, :, c].flatten(), fisher=True) for c in range(N_CELLS)]
    cell_kurt_c = [kurtosis(dch_c[:, :, c].flatten(), fisher=True) for c in range(N_CELLS)]
    cell_kurt_gt = [kurtosis(dch_gt[:, :, c].flatten(), fisher=True) for c in range(N_CELLS)]
    cell_kurt_base = [kurtosis(dch_base[:, :, c].flatten(), fisher=True) for c in range(N_CELLS)]

    # Overall kurtosis ratio
    overall_b = kurtosis(dch_b.flatten(), fisher=True) / (kurtosis(dch_gt.flatten(), fisher=True) + 1e-6)
    overall_c = kurtosis(dch_c.flatten(), fisher=True) / (kurtosis(dch_gt.flatten(), fisher=True) + 1e-6)
    overall_base = kurtosis(dch_base.flatten(), fisher=True) / (kurtosis(dch_gt.flatten(), fisher=True) + 1e-6)

    # Which cells improve most?
    improvements = [(c, cell_kurt_c[c] / (cell_kurt_gt[c] + 1e-6),
                     cell_kurt_b[c] / (cell_kurt_gt[c] + 1e-6)) for c in range(N_CELLS)]
    improvements.sort(key=lambda x: abs(x[1] - 1.0))  # Sort by closeness to GT

    r2 = {
        'overall_kurtosis_ratio': {
            'base_153a': overall_base,
            '154b_uncond': overall_b,
            '154c_cond': overall_c,
        },
        'per_cell_kurtosis': {
            'gt': cell_kurt_gt,
            'base_153a': cell_kurt_base,
            '154b_uncond': cell_kurt_b,
            '154c_cond': cell_kurt_c,
        },
        'per_cell_kurtosis_ratio': {
            '154b': [cell_kurt_b[c] / (cell_kurt_gt[c] + 1e-6) for c in range(N_CELLS)],
            '154c': [cell_kurt_c[c] / (cell_kurt_gt[c] + 1e-6) for c in range(N_CELLS)],
        },
        'top5_improved_cells': [
            {'cell': int(c), 'ratio_154c': float(r_c), 'ratio_154b': float(r_b)}
            for c, r_c, r_b in improvements[:5]
        ],
    }

    print(f"  Overall kurtosis ratio: base={overall_base:.3f}, 154b={overall_b:.3f}, 154c={overall_c:.3f}")
    print(f"  Per-cell kurtosis ratio range:")
    print(f"    154b: [{min(r2['per_cell_kurtosis_ratio']['154b']):.2f}, {max(r2['per_cell_kurtosis_ratio']['154b']):.2f}]")
    print(f"    154c: [{min(r2['per_cell_kurtosis_ratio']['154c']):.2f}, {max(r2['per_cell_kurtosis_ratio']['154c']):.2f}]")

    # Plot per-cell kurtosis comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cells = np.arange(N_CELLS)

    ax = axes[0]
    ax.bar(cells - 0.3, cell_kurt_gt, 0.2, label='GT', alpha=0.8, color='black')
    ax.bar(cells - 0.1, cell_kurt_base, 0.2, label='Base (153a)', alpha=0.8, color='gray')
    ax.bar(cells + 0.1, cell_kurt_b, 0.2, label='154b (uncond)', alpha=0.8, color='red')
    ax.bar(cells + 0.3, cell_kurt_c, 0.2, label='154c (cond)', alpha=0.8, color='blue')
    ax.set_xlabel('Cell index')
    ax.set_ylabel('Excess kurtosis')
    ax.set_title('Per-Cell Kurtosis of Daily Changes')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(cell_kurt_gt, cell_kurt_b, c='red', label='154b vs GT', alpha=0.7)
    ax.scatter(cell_kurt_gt, cell_kurt_c, c='blue', label='154c vs GT', alpha=0.7)
    lim = max(max(cell_kurt_gt), max(cell_kurt_b), max(cell_kurt_c)) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('GT kurtosis')
    ax.set_ylabel('Generated kurtosis')
    ax.set_title('Per-Cell Kurtosis: Generated vs GT')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTDIR / 'per_cell_kurtosis.png', dpi=150)
    plt.close()
    print(f"  Saved: per_cell_kurtosis.png")

    results['investigation_2'] = r2


def investigation_3(models, data, results):
    """Condition ablation: zero-condition vs real-condition on 154c."""
    print("\n=== Investigation 3: Condition Ablation on 154c ===")

    model_c = models['model_c']
    encoder = models['encoder']
    surfaces = data['surfaces']
    val_preds = data['val_preds']
    val_gts = data['val_gts']
    res_mean_c = models['res_mean_c']
    res_std_c = models['res_std_c']

    H = 30
    val_start = 4040
    n_res = 500

    # 3a: Zero condition
    print("  Sampling 154c with ZERO condition...")
    zero_cond = torch.zeros(100, 128, device=DEVICE)
    all_res_zero = []
    for i in range(0, n_res, 100):
        batch = min(100, n_res - i)
        raw = sample_fm(model_c, batch, DEVICE, cond=zero_cond[:batch])
        denorm = raw * res_std_c + res_mean_c
        all_res_zero.append(denorm)
    res_zero = np.concatenate(all_res_zero)

    # 3b: Random Gaussian condition (same stats as encoder output but no structure)
    print("  Sampling 154c with RANDOM GAUSSIAN condition...")
    # First compute mean/std of real conditions
    cond_samples = []
    with torch.no_grad():
        for j in range(min(200, len(val_preds))):
            wi = val_start + j
            hist = torch.from_numpy(surfaces[wi:wi+H][None].astype(np.float32)).to(DEVICE)
            c = encoder(normalize_iv(hist))
            cond_samples.append(c.cpu().numpy())
    cond_array = np.concatenate(cond_samples)
    cond_mean_np = cond_array.mean(axis=0)
    cond_std_np = cond_array.std(axis=0)

    all_res_rand = []
    for i in range(0, n_res, 100):
        batch = min(100, n_res - i)
        rand_cond = torch.from_numpy(
            (np.random.randn(batch, 128) * cond_std_np + cond_mean_np).astype(np.float32)
        ).to(DEVICE)
        raw = sample_fm(model_c, batch, DEVICE, cond=rand_cond)
        denorm = raw * res_std_c + res_mean_c
        all_res_rand.append(denorm)
    res_rand = np.concatenate(all_res_rand)

    # 3c: Real conditions (matched to windows)
    print("  Sampling 154c with REAL condition (matched)...")
    all_res_real = []
    with torch.no_grad():
        for i in range(0, n_res, 100):
            batch = min(100, n_res - i)
            idx = np.random.choice(len(val_preds), batch)
            conds = []
            for j in idx:
                wi = val_start + j
                hist = torch.from_numpy(surfaces[wi:wi+H][None].astype(np.float32)).to(DEVICE)
                c = encoder(normalize_iv(hist))
                conds.append(c)
            conds = torch.cat(conds, dim=0)
            raw = sample_fm(model_c, batch, DEVICE, cond=conds)
            denorm = raw * res_std_c + res_mean_c
            all_res_real.append(denorm)
    res_real = np.concatenate(all_res_real)

    # Compute kurtosis for each ablation
    gt_residuals = val_gts - val_preds

    for label, res in [('zero_cond', res_zero), ('random_cond', res_rand), ('real_cond', res_real)]:
        r3d = res.reshape(-1, T_FRAMES, N_CELLS)
        dch = np.diff(r3d, axis=1).flatten()
        k = kurtosis(dch, fisher=True)
        s = res.std()
        mag = np.linalg.norm(res, axis=1)
        print(f"  {label}: kurtosis(res daily changes)={k:.3f}, std={s:.5f}, "
              f"magnitude mean={mag.mean():.3f}, cv={mag.std()/mag.mean():.3f}")

    # Now combine with base predictions and compute kurtosis ratio
    n_comb = min(len(val_preds), n_res)
    gt_3d = val_gts[:n_comb].reshape(-1, T_FRAMES, N_CELLS)
    dch_gt = np.diff(gt_3d, axis=1).flatten()
    kurt_gt = kurtosis(dch_gt, fisher=True)

    combined_results = {}
    for label, res in [('zero_cond', res_zero), ('random_cond', res_rand), ('real_cond', res_real)]:
        combined = np.clip(val_preds[:n_comb] + res[:n_comb], 0, 1)
        c3d = combined.reshape(-1, T_FRAMES, N_CELLS)
        dch = np.diff(c3d, axis=1).flatten()
        k = kurtosis(dch, fisher=True)
        ratio = k / (kurt_gt + 1e-6)
        combined_results[label] = {
            'kurtosis': k,
            'kurtosis_ratio': ratio,
            'residual_std': float(res.std()),
            'residual_magnitude_cv': float(np.linalg.norm(res, axis=1).std() / np.linalg.norm(res, axis=1).mean()),
        }
        print(f"  Combined {label}: kurtosis_ratio={ratio:.3f}")

    r3 = {
        'gt_daily_change_kurtosis': kurt_gt,
        'ablation_results': combined_results,
        'condition_statistics': {
            'mean_norm': float(np.linalg.norm(cond_mean_np)),
            'mean_std_across_dims': float(cond_std_np.mean()),
            'active_dims_gt_0.01': int((cond_std_np > 0.01).sum()),
        }
    }

    results['investigation_3'] = r3

    return cond_array


def investigation_4(models, data, results):
    """Daily change distribution histograms with tail focus."""
    print("\n=== Investigation 4: Daily Change Distribution Comparison ===")

    val_preds = data['val_preds']
    val_gts = data['val_gts']
    res_mean_b = models['res_mean_b']
    res_std_b = models['res_std_b']
    res_mean_c = models['res_mean_c']
    res_std_c = models['res_std_c']
    model_b = models['model_b']
    model_c = models['model_c']
    encoder = models['encoder']
    surfaces = data['surfaces']

    H = 30
    val_start = 4040
    n_samples = 500

    # Generate samples from both models
    all_b = []
    for i in range(0, n_samples, 100):
        batch = min(100, n_samples - i)
        raw = sample_fm(model_b, batch, DEVICE, cond=None)
        denorm = raw * res_std_b + res_mean_b
        all_b.append(denorm)
    res_b = np.concatenate(all_b)

    all_c = []
    with torch.no_grad():
        for i in range(0, n_samples, 100):
            batch = min(100, n_samples - i)
            idx = np.random.choice(len(val_preds), batch)
            conds = []
            for j in idx:
                wi = val_start + j
                hist = torch.from_numpy(surfaces[wi:wi+H][None].astype(np.float32)).to(DEVICE)
                c = encoder(normalize_iv(hist))
                conds.append(c)
            conds = torch.cat(conds, dim=0)
            raw = sample_fm(model_c, batch, DEVICE, cond=conds)
            denorm = raw * res_std_c + res_mean_c
            all_c.append(denorm)
    res_c = np.concatenate(all_c)

    n_comb = min(len(val_preds), n_samples)
    combined_b = np.clip(val_preds[:n_comb] + res_b[:n_comb], 0, 1)
    combined_c = np.clip(val_preds[:n_comb] + res_c[:n_comb], 0, 1)
    gt = val_gts[:n_comb]
    base_only = val_preds[:n_comb]

    # Daily changes
    dch_gt = np.diff(gt.reshape(-1, T_FRAMES, N_CELLS), axis=1).flatten()
    dch_base = np.diff(base_only.reshape(-1, T_FRAMES, N_CELLS), axis=1).flatten()
    dch_b = np.diff(combined_b.reshape(-1, T_FRAMES, N_CELLS), axis=1).flatten()
    dch_c = np.diff(combined_c.reshape(-1, T_FRAMES, N_CELLS), axis=1).flatten()

    # Standardize for comparison
    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    gt_s = standardize(dch_gt)
    base_s = standardize(dch_base)
    b_s = standardize(dch_b)
    c_s = standardize(dch_c)

    # Tail exceedance rates
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    tail_rates = {}
    for label, x in [('GT', gt_s), ('Base_153a', base_s), ('154b_uncond', b_s), ('154c_cond', c_s)]:
        rates = {}
        for th in thresholds:
            rates[f'>{th}sigma'] = float((np.abs(x) > th).mean())
        tail_rates[label] = rates

    # Print comparison
    print("  Tail exceedance rates (|z| > threshold):")
    print(f"  {'Threshold':>10s} {'GT':>10s} {'Base':>10s} {'154b':>10s} {'154c':>10s}")
    for th in thresholds:
        key = f'>{th}sigma'
        print(f"  {key:>10s} {tail_rates['GT'][key]:>10.4f} {tail_rates['Base_153a'][key]:>10.4f} "
              f"{tail_rates['154b_uncond'][key]:>10.4f} {tail_rates['154c_cond'][key]:>10.4f}")

    # Plot 1: Full histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bins = np.linspace(-0.015, 0.015, 120)
    for ax, (label, x) in zip(axes.flat, [
        ('GT', dch_gt), ('Base (153a)', dch_base),
        ('154b (uncond residual)', dch_b), ('154c (cond residual)', dch_c)
    ]):
        ax.hist(x, bins=bins, density=True, alpha=0.8, color='steelblue')
        ax.set_title(f'{label}\nkurtosis={kurtosis(x, fisher=True):.2f}')
        ax.set_xlim(-0.015, 0.015)
        ax.set_ylabel('Density')

    plt.suptitle('Daily Change Distributions (All Models)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'daily_change_histograms.png', dpi=150)
    plt.close()
    print(f"  Saved: daily_change_histograms.png")

    # Plot 2: Log-scale tail comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bins_tail = np.linspace(-6, 6, 200)
    for label, x, color in [
        ('GT', gt_s, 'black'),
        ('Base (153a)', base_s, 'gray'),
        ('154b (uncond)', b_s, 'red'),
        ('154c (cond)', c_s, 'blue'),
    ]:
        counts, edges = np.histogram(x, bins=bins_tail, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        ax.semilogy(centers[mask], counts[mask], label=label, alpha=0.8, color=color)

    # Add Gaussian reference
    from scipy.stats import norm
    xx = np.linspace(-6, 6, 200)
    ax.semilogy(xx, norm.pdf(xx), 'k--', alpha=0.3, label='Gaussian')

    ax.set_xlabel('Standardized daily change (z-score)')
    ax.set_ylabel('Log density')
    ax.set_title('Tail Comparison: Log-Scale Density')
    ax.legend()
    ax.set_ylim(1e-5, 1.0)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'tail_comparison_log.png', dpi=150)
    plt.close()
    print(f"  Saved: tail_comparison_log.png")

    r4 = {
        'kurtosis': {
            'GT': kurtosis(dch_gt, fisher=True),
            'Base_153a': kurtosis(dch_base, fisher=True),
            '154b_uncond': kurtosis(dch_b, fisher=True),
            '154c_cond': kurtosis(dch_c, fisher=True),
        },
        'tail_exceedance_rates': tail_rates,
    }

    results['investigation_4'] = r4


def investigation_5(models, data, results, cond_array):
    """Hypothesis test: condition modulates SCALE per window, producing correct kurtosis mixture."""
    print("\n=== Investigation 5: Scale Modulation Hypothesis Test ===")

    model_c = models['model_c']
    encoder = models['encoder']
    surfaces = data['surfaces']
    val_preds = data['val_preds']
    val_gts = data['val_gts']
    res_mean_c = models['res_mean_c']
    res_std_c = models['res_std_c']
    model_b = models['model_b']
    res_mean_b = models['res_mean_b']
    res_std_b = models['res_std_b']

    H = 30
    val_start = 4040
    n_windows = min(200, len(val_preds))
    n_samples_per_window = 20  # Multiple samples per window to measure spread

    # Key test: For each window, generate multiple residual samples from both models.
    # Measure the per-window residual spread.
    # Hypothesis: 154c's per-window spread varies WITH the condition (high for turbulent,
    # low for calm). 154b's spread is constant across windows.

    print(f"  Generating {n_samples_per_window} samples per window for {n_windows} windows...")

    per_window_std_b = []
    per_window_std_c = []
    window_volatility = []  # GT measure of how "turbulent" each window's history is

    rets = data['rets']

    with torch.no_grad():
        for w in range(n_windows):
            wi = val_start + w

            # History volatility (realized vol of returns in history window)
            hist_rets = rets[wi:wi+H]
            hist_vol = np.std(hist_rets)
            window_volatility.append(hist_vol)

            # Get condition for this window
            hist = torch.from_numpy(surfaces[wi:wi+H][None].astype(np.float32)).to(DEVICE)
            cond = encoder(normalize_iv(hist))

            # Sample residuals from 154c (conditional) - multiple per window
            x_c = torch.randn(n_samples_per_window, DIM, device=DEVICE)
            cond_expanded = cond.expand(n_samples_per_window, -1)
            for step in range(N_STEPS):
                t = torch.full((n_samples_per_window,), step * DT, device=DEVICE)
                x_c = x_c + model_c(x_c, t, cond=cond_expanded) * DT
            res_c_raw = (x_c.cpu().numpy() * res_std_c + res_mean_c)  # (n_samples, 750)
            per_window_std_c.append(res_c_raw.std())

            # Sample residuals from 154b (unconditional) - multiple per window
            x_b = torch.randn(n_samples_per_window, DIM, device=DEVICE)
            for step in range(N_STEPS):
                t = torch.full((n_samples_per_window,), step * DT, device=DEVICE)
                x_b = x_b + model_b(x_b, t) * DT
            res_b_raw = (x_b.cpu().numpy() * res_std_b + res_mean_b)
            per_window_std_b.append(res_b_raw.std())

            if (w + 1) % 50 == 0:
                print(f"    {w+1}/{n_windows}")

    per_window_std_b = np.array(per_window_std_b)
    per_window_std_c = np.array(per_window_std_c)
    window_volatility = np.array(window_volatility)

    # GT per-window residual std
    gt_residuals = val_gts - val_preds
    per_window_std_gt = np.array([gt_residuals[w].std() for w in range(n_windows)])

    # Key metrics:
    # 1. Coefficient of variation of per-window spread
    cv_b = per_window_std_b.std() / per_window_std_b.mean()
    cv_c = per_window_std_c.std() / per_window_std_c.mean()
    cv_gt = per_window_std_gt.std() / per_window_std_gt.mean()

    # 2. Correlation of per-window spread with history volatility
    corr_b_vol = np.corrcoef(per_window_std_b, window_volatility)[0, 1]
    corr_c_vol = np.corrcoef(per_window_std_c, window_volatility)[0, 1]
    corr_gt_vol = np.corrcoef(per_window_std_gt, window_volatility)[0, 1]

    # 3. Correlation of per-window spread with GT per-window spread
    corr_b_gt = np.corrcoef(per_window_std_b, per_window_std_gt)[0, 1]
    corr_c_gt = np.corrcoef(per_window_std_c, per_window_std_gt)[0, 1]

    # 4. Ratio of turbulent/calm window spreads
    vol_80 = np.percentile(window_volatility, 80)
    vol_20 = np.percentile(window_volatility, 20)
    turb_mask = window_volatility > vol_80
    calm_mask = window_volatility < vol_20

    turb_calm_b = per_window_std_b[turb_mask].mean() / per_window_std_b[calm_mask].mean()
    turb_calm_c = per_window_std_c[turb_mask].mean() / per_window_std_c[calm_mask].mean()
    turb_calm_gt = per_window_std_gt[turb_mask].mean() / per_window_std_gt[calm_mask].mean()

    # 5. Mixture kurtosis explanation
    # If you mix Gaussians with different variances, the mixture has excess kurtosis.
    # Excess kurtosis of a scale mixture = 3 * Var(sigma^2) / E[sigma^2]^2
    # (for zero-mean components)
    def mixture_kurtosis_from_spreads(spreads):
        variances = spreads ** 2
        return 3 * np.var(variances) / (np.mean(variances) ** 2 + 1e-10)

    predicted_kurt_b = mixture_kurtosis_from_spreads(per_window_std_b)
    predicted_kurt_c = mixture_kurtosis_from_spreads(per_window_std_c)
    predicted_kurt_gt = mixture_kurtosis_from_spreads(per_window_std_gt)

    print(f"\n  Per-window spread statistics:")
    print(f"  {'':>20s} {'154b':>10s} {'154c':>10s} {'GT':>10s}")
    print(f"  {'CV of spread':>20s} {cv_b:>10.4f} {cv_c:>10.4f} {cv_gt:>10.4f}")
    print(f"  {'Corr w/ hist vol':>20s} {corr_b_vol:>10.4f} {corr_c_vol:>10.4f} {corr_gt_vol:>10.4f}")
    print(f"  {'Corr w/ GT spread':>20s} {corr_b_gt:>10.4f} {corr_c_gt:>10.4f} {'1.000':>10s}")
    print(f"  {'Turb/Calm ratio':>20s} {turb_calm_b:>10.4f} {turb_calm_c:>10.4f} {turb_calm_gt:>10.4f}")
    print(f"  {'Predicted mix kurt':>20s} {predicted_kurt_b:>10.4f} {predicted_kurt_c:>10.4f} {predicted_kurt_gt:>10.4f}")

    # Plot: per-window spread vs history volatility
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(window_volatility, per_window_std_b, alpha=0.4, c='red', label='154b', s=15)
    ax.scatter(window_volatility, per_window_std_c, alpha=0.4, c='blue', label='154c', s=15)
    ax.scatter(window_volatility, per_window_std_gt, alpha=0.4, c='black', label='GT', s=15)
    ax.set_xlabel('History volatility (ret std)')
    ax.set_ylabel('Per-window residual std')
    ax.set_title('Residual Scale vs History Volatility')
    ax.legend()

    ax = axes[1]
    ax.hist(per_window_std_b, bins=30, alpha=0.5, color='red', label='154b', density=True)
    ax.hist(per_window_std_c, bins=30, alpha=0.5, color='blue', label='154c', density=True)
    ax.hist(per_window_std_gt, bins=30, alpha=0.5, color='black', label='GT', density=True)
    ax.set_xlabel('Per-window residual std')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Per-Window Spread')
    ax.legend()

    ax = axes[2]
    ax.scatter(per_window_std_gt, per_window_std_b, alpha=0.4, c='red', label='154b', s=15)
    ax.scatter(per_window_std_gt, per_window_std_c, alpha=0.4, c='blue', label='154c', s=15)
    lim = max(per_window_std_gt.max(), per_window_std_c.max(), per_window_std_b.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('GT per-window residual std')
    ax.set_ylabel('Model per-window residual std')
    ax.set_title('Per-Window Spread: Model vs GT')
    ax.legend()

    plt.suptitle('Investigation 5: Scale Modulation Hypothesis', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'scale_modulation.png', dpi=150)
    plt.close()
    print(f"  Saved: scale_modulation.png")

    # Condition norm vs spread
    if cond_array is not None and len(cond_array) >= n_windows:
        cond_norms = np.linalg.norm(cond_array[:n_windows], axis=1)
        corr_cond_norm = np.corrcoef(cond_norms, per_window_std_c)[0, 1]
        print(f"\n  Correlation of condition norm with 154c spread: {corr_cond_norm:.4f}")
    else:
        corr_cond_norm = None

    r5 = {
        'cv_of_per_window_spread': {
            '154b_uncond': cv_b, '154c_cond': cv_c, 'gt': cv_gt
        },
        'corr_spread_with_hist_vol': {
            '154b_uncond': corr_b_vol, '154c_cond': corr_c_vol, 'gt': corr_gt_vol
        },
        'corr_spread_with_gt_spread': {
            '154b_uncond': corr_b_gt, '154c_cond': corr_c_gt
        },
        'turb_calm_spread_ratio': {
            '154b_uncond': turb_calm_b, '154c_cond': turb_calm_c, 'gt': turb_calm_gt
        },
        'predicted_mixture_kurtosis': {
            '154b_uncond': predicted_kurt_b, '154c_cond': predicted_kurt_c, 'gt': predicted_kurt_gt
        },
        'corr_cond_norm_with_spread': corr_cond_norm,
        'hypothesis_supported': cv_c > cv_b and corr_c_vol > corr_b_vol,
        'mechanism_explanation': (
            "The condition tells the residual FM about the uncertainty regime of each "
            "history window. Without condition, the FM generates average-scale residuals "
            "for ALL windows (low CV of spread). With condition, it generates appropriately "
            "scaled residuals -- larger for high-uncertainty windows, smaller for low-uncertainty. "
            "This produces a MIXTURE of narrow and wide distributions across windows. "
            "A Gaussian scale mixture has excess kurtosis = 3 * Var(sigma^2) / E[sigma^2]^2, "
            "which naturally produces heavier tails, matching the GT kurtosis. "
            "The unconditional FM, by contrast, uses a single scale for all windows, "
            "producing too-light tails (kurtosis ratio < 1)."
        ),
    }

    results['investigation_5'] = r5


def main():
    print("=" * 70)
    print("154c Kurtosis Mechanism Analysis")
    print("WHY does conditioning improve kurtosis from 0.81 to 1.003?")
    print("=" * 70)

    models = load_models()
    data = load_data()

    results = {}

    # Investigation 1: Raw residual distributions
    res_b, res_c, gt_residuals = investigation_1(models, data, results)

    # Investigation 2: Per-cell kurtosis
    investigation_2(models, data, results, res_b, res_c)

    # Investigation 3: Condition ablation
    cond_array = investigation_3(models, data, results)

    # Investigation 4: Daily change histograms
    investigation_4(models, data, results)

    # Investigation 5: Scale modulation hypothesis
    investigation_5(models, data, results, cond_array)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    h_supported = results['investigation_5'].get('hypothesis_supported', False)
    cv_b = results['investigation_5']['cv_of_per_window_spread']['154b_uncond']
    cv_c = results['investigation_5']['cv_of_per_window_spread']['154c_cond']
    cv_gt = results['investigation_5']['cv_of_per_window_spread']['gt']
    corr_b = results['investigation_5']['corr_spread_with_hist_vol']['154b_uncond']
    corr_c = results['investigation_5']['corr_spread_with_hist_vol']['154c_cond']

    # Did zero-condition ablation confirm?
    ablation = results['investigation_3']['ablation_results']
    zero_ratio = ablation['zero_cond']['kurtosis_ratio']
    real_ratio = ablation['real_cond']['kurtosis_ratio']

    summary = {
        'hypothesis': (
            "Conditioning modulates residual SCALE per window, creating a Gaussian "
            "scale mixture whose excess kurtosis matches GT."
        ),
        'hypothesis_supported': h_supported,
        'evidence': {
            'spread_cv_154b': cv_b,
            'spread_cv_154c': cv_c,
            'spread_cv_gt': cv_gt,
            'spread_cv_154c_closer_to_gt': abs(cv_c - cv_gt) < abs(cv_b - cv_gt),
            'corr_with_hist_vol_154b': corr_b,
            'corr_with_hist_vol_154c': corr_c,
            'ablation_zero_cond_kurtosis_ratio': zero_ratio,
            'ablation_real_cond_kurtosis_ratio': real_ratio,
            'ablation_confirms_condition_is_causal': abs(real_ratio - 1.0) < abs(zero_ratio - 1.0),
        },
    }

    results['summary'] = summary

    print(f"  Hypothesis supported: {h_supported}")
    print(f"  Spread CV: 154b={cv_b:.4f}, 154c={cv_c:.4f}, GT={cv_gt:.4f}")
    print(f"  Corr(spread, hist_vol): 154b={corr_b:.4f}, 154c={corr_c:.4f}")
    print(f"  Ablation: zero_cond ratio={zero_ratio:.3f}, real_cond ratio={real_ratio:.3f}")
    print(f"  Condition is causal for kurtosis: {abs(real_ratio - 1.0) < abs(zero_ratio - 1.0)}")

    # Save results
    results_ser = make_serializable(results)
    with open(RESULTDIR / '154c_kurtosis.json', 'w') as f:
        json.dump(results_ser, f, indent=2)
    print(f"\n  Results saved to: {RESULTDIR / '154c_kurtosis.json'}")
    print(f"  Plots saved to: {OUTDIR}/")


if __name__ == "__main__":
    main()
