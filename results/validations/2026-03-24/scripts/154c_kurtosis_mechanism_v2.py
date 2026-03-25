#!/usr/bin/env python
"""
154c Kurtosis Mechanism Analysis v2 — Correct Evaluation Protocol

The v1 analysis showed the initial hypothesis (scale modulation) was WRONG.
This v2 uses the CORRECT evaluation protocol: per-window matched generation,
matching eval_154c.py methodology.

Key insight from v1: both models have similar per-window spread (~0.05-0.06 CV).
The condition does NOT modulate residual scale in the way hypothesized.

This v2 focuses on:
1. Reproduce the exact kurtosis numbers (0.81 vs 1.003) using correct eval
2. Check if the difference comes from the SHAPE of residuals, not just scale
3. Test whether condition changes the DIRECTION/STRUCTURE of residuals per window
4. Per-horizon kurtosis breakdown
5. Deeper ablation: what aspects of the condition drive the improvement?
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


def load_all():
    """Load all models and data."""
    print("Loading models and data...")

    # Encoder
    encoder, cond_dim = load_encoder(
        '/home/max/Documents/vol-surface-vae-pub/models/backfill/block_ar_vol_scaled_30ep/best_model.pt',
        DEVICE)
    for p in encoder.parameters(): p.requires_grad = False

    # Base model (153a)
    bc = torch.load('/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_153a/final_model.pt',
                     weights_only=False, map_location=DEVICE)
    bcfg = bc['config']
    base_model = ConditionalFactoredVelocityTransformer(
        n_frames=bcfg['n_frames'], n_cells=bcfg['n_cells'],
        d_model=bcfg['d_model'], n_heads=bcfg['n_heads'], n_layers=bcfg['n_layers'],
        cond_dim=bcfg['cond_dim'])
    base_model.load_state_dict(bc['model_state_dict'])
    base_model.to(DEVICE).eval()
    bm = torch.from_numpy(bc['train_mean']).float().to(DEVICE)
    bs = torch.from_numpy(bc['train_std']).float().to(DEVICE)

    # 154b: Unconditional residual FM
    rc_b = torch.load('/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_154b/final_model.pt',
                       weights_only=False, map_location=DEVICE)
    cfg_b = rc_b['config']
    model_b = FactoredVelocityTransformer(
        n_frames=cfg_b['n_frames'], n_cells=cfg_b['n_cells'],
        d_model=cfg_b['d_model'], n_heads=cfg_b['n_heads'], n_layers=cfg_b['n_layers'])
    model_b.load_state_dict(rc_b['model_state_dict'])
    model_b.to(DEVICE).eval()
    rm_b = torch.from_numpy(rc_b['res_mean']).float().to(DEVICE)
    rs_b = torch.from_numpy(rc_b['res_std']).float().to(DEVICE)

    # 154c: Conditional residual FM
    rc_c = torch.load('/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_154c/final_model.pt',
                       weights_only=False, map_location=DEVICE)
    cfg_c = rc_c['config']
    model_c = ConditionalFactoredVelocityTransformer(
        n_frames=cfg_c['n_frames'], n_cells=cfg_c['n_cells'],
        d_model=cfg_c['d_model'], n_heads=cfg_c['n_heads'], n_layers=cfg_c['n_layers'],
        cond_dim=cfg_c['cond_dim'])
    model_c.load_state_dict(rc_c['model_state_dict'])
    model_c.to(DEVICE).eval()
    rm_c = torch.from_numpy(rc_c['res_mean']).float().to(DEVICE)
    rs_c = torch.from_numpy(rc_c['res_std']).float().to(DEVICE)

    # Data
    data = np.load('/home/max/Documents/vol-surface-vae-pub/data/vol_surface_with_ret.npz')
    surfaces = data['surface']
    rets = data['ret']

    return {
        'encoder': encoder, 'base_model': base_model, 'bm': bm, 'bs': bs,
        'model_b': model_b, 'rm_b': rm_b, 'rs_b': rs_b,
        'model_c': model_c, 'rm_c': rm_c, 'rs_c': rs_c,
        'surfaces': surfaces, 'rets': rets,
    }


def generate_matched_samples(M, n_windows=160, n_samples=50):
    """
    Generate matched per-window samples exactly like eval_154c.py.
    Returns: (samples_base, samples_154b, samples_154c, gt) all per-window.
    """
    H, T = 30, 30
    ts = 4540  # Test set start
    surfaces = M['surfaces']
    encoder = M['encoder']
    base = M['base_model']
    bm, bs = M['bm'], M['bs']
    model_b = M['model_b']; rm_b, rs_b = M['rm_b'], M['rs_b']
    model_c = M['model_c']; rm_c, rs_c = M['rm_c'], M['rs_c']

    max_i = min(ts + n_windows, len(surfaces) - H - T + 1)
    windows = [(surfaces[i:i+H], surfaces[i+H:i+H+T])
               for i in range(ts, max_i)]
    N = len(windows)
    print(f"  Generating for {N} windows, {n_samples} samples each...")

    all_base = []      # (N, n_samples, T, 5, 5)
    all_154b = []
    all_154c = []
    all_gt = []
    all_conds = []

    with torch.no_grad():
        for w in range(N):
            h = torch.from_numpy(windows[w][0][None].astype(np.float32)).to(DEVICE)
            cond = encoder(normalize_iv(h))
            all_conds.append(cond.cpu().numpy())

            # Base prediction (average of 3 ODE runs)
            bp_runs = []
            for _ in range(3):
                x = torch.randn(1, DIM, device=DEVICE)
                for s in range(N_STEPS):
                    t = torch.full((1,), s * DT, device=DEVICE)
                    x = x + base(x, t, cond=cond) * DT
                bp_runs.append((x * bs + bm).clamp(0, 1))
            bp = torch.stack(bp_runs).mean(0)  # (1, 750)

            samp_base = []
            samp_b = []
            samp_c = []

            # Generate n_samples from each model
            for _ in range(n_samples):
                # 153a base: single ODE sample (no residual)
                x0 = torch.randn(1, DIM, device=DEVICE)
                for s in range(N_STEPS):
                    t = torch.full((1,), s * DT, device=DEVICE)
                    x0 = x0 + base(x0, t, cond=cond) * DT
                base_sample = (x0 * bs + bm).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
                samp_base.append(base_sample)

                # 154b: base_pred + uncond residual
                xb = torch.randn(1, DIM, device=DEVICE)
                for s in range(N_STEPS):
                    t = torch.full((1,), s * DT, device=DEVICE)
                    xb = xb + model_b(xb, t) * DT
                combined_b = (bp + xb * rs_b + rm_b).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
                samp_b.append(combined_b)

                # 154c: base_pred + cond residual
                xc = torch.randn(1, DIM, device=DEVICE)
                for s in range(N_STEPS):
                    t = torch.full((1,), s * DT, device=DEVICE)
                    xc = xc + model_c(xc, t, cond=cond) * DT
                combined_c = (bp + xc * rs_c + rm_c).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
                samp_c.append(combined_c)

            all_base.append(np.array(samp_base))   # (n_samples, T, 5, 5)
            all_154b.append(np.array(samp_b))
            all_154c.append(np.array(samp_c))
            all_gt.append(windows[w][1])             # (T, 5, 5)

            if (w + 1) % 20 == 0:
                print(f"    {w+1}/{N}")

    return {
        'base': np.array(all_base),   # (N, ns, T, 5, 5)
        '154b': np.array(all_154b),
        '154c': np.array(all_154c),
        'gt': np.array(all_gt),        # (N, T, 5, 5)
        'conds': np.concatenate(all_conds),  # (N, 128)
    }


def compute_kurtosis_ratio(samples, gt):
    """Compute kurtosis ratio using 1st sample daily changes (like eval_154c.py)."""
    # Use first sample per window (as in eval_154c.py: cs[:,0])
    gen = samples[:, 0]  # (N, T, 5, 5)
    gen_ch = np.diff(gen, axis=1).reshape(-1, N_CELLS)  # (N*29, 25)
    gt_ch = np.diff(gt, axis=1).reshape(-1, N_CELLS)
    kr = kurtosis(gen_ch.flatten(), fisher=True) / (kurtosis(gt_ch.flatten(), fisher=True) + 1e-6)
    return kr


def main():
    print("=" * 70)
    print("154c Kurtosis Mechanism Analysis v2 — Correct Evaluation Protocol")
    print("=" * 70)

    M = load_all()
    results = {}

    # Step 1: Reproduce exact kurtosis numbers with correct eval
    print("\n=== Step 1: Reproduce Kurtosis Numbers ===")

    # Use same parameters as eval_154c.py: 160 windows, 50 samples
    S = generate_matched_samples(M, n_windows=160, n_samples=50)

    kr_base = compute_kurtosis_ratio(S['base'], S['gt'])
    kr_b = compute_kurtosis_ratio(S['154b'], S['gt'])
    kr_c = compute_kurtosis_ratio(S['154c'], S['gt'])

    gt_ch = np.diff(S['gt'], axis=1).reshape(-1, N_CELLS)
    gt_kurt = kurtosis(gt_ch.flatten(), fisher=True)

    base_ch = np.diff(S['base'][:, 0], axis=1).reshape(-1, N_CELLS)
    b_ch = np.diff(S['154b'][:, 0], axis=1).reshape(-1, N_CELLS)
    c_ch = np.diff(S['154c'][:, 0], axis=1).reshape(-1, N_CELLS)

    print(f"  GT excess kurtosis: {gt_kurt:.3f}")
    print(f"  Base (153a) kurtosis ratio: {kr_base:.3f}")
    print(f"  154b (uncond) kurtosis ratio: {kr_b:.3f}")
    print(f"  154c (cond) kurtosis ratio: {kr_c:.3f}")

    results['step1_kurtosis_reproduction'] = {
        'gt_excess_kurtosis': gt_kurt,
        'base_153a_ratio': kr_base,
        '154b_uncond_ratio': kr_b,
        '154c_cond_ratio': kr_c,
    }

    # Step 2: Per-horizon kurtosis breakdown
    print("\n=== Step 2: Per-Horizon Kurtosis Breakdown ===")

    # For each horizon h, compute kurtosis of daily change at that horizon
    per_hz_kurt = {'gt': [], 'base': [], '154b': [], '154c': []}

    for h in range(29):  # 29 daily changes
        gt_h = (S['gt'][:, h+1] - S['gt'][:, h]).reshape(-1, N_CELLS).flatten()
        base_h = (S['base'][:, 0, h+1] - S['base'][:, 0, h]).reshape(-1, N_CELLS).flatten()
        b_h = (S['154b'][:, 0, h+1] - S['154b'][:, 0, h]).reshape(-1, N_CELLS).flatten()
        c_h = (S['154c'][:, 0, h+1] - S['154c'][:, 0, h]).reshape(-1, N_CELLS).flatten()

        per_hz_kurt['gt'].append(kurtosis(gt_h, fisher=True))
        per_hz_kurt['base'].append(kurtosis(base_h, fisher=True))
        per_hz_kurt['154b'].append(kurtosis(b_h, fisher=True))
        per_hz_kurt['154c'].append(kurtosis(c_h, fisher=True))

    # Plot per-horizon kurtosis
    fig, ax = plt.subplots(figsize=(12, 5))
    hz = np.arange(1, 30)
    ax.plot(hz, per_hz_kurt['gt'], 'k-o', label='GT', markersize=4)
    ax.plot(hz, per_hz_kurt['base'], '--', color='gray', label='Base (153a)', alpha=0.7)
    ax.plot(hz, per_hz_kurt['154b'], 'r-s', label='154b (uncond)', markersize=3, alpha=0.7)
    ax.plot(hz, per_hz_kurt['154c'], 'b-^', label='154c (cond)', markersize=3, alpha=0.7)
    ax.set_xlabel('Horizon (day)')
    ax.set_ylabel('Excess kurtosis')
    ax.set_title('Per-Horizon Kurtosis of Daily Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'per_horizon_kurtosis.png', dpi=150)
    plt.close()
    print(f"  Saved: per_horizon_kurtosis.png")

    # Which horizons show the biggest 154b→154c improvement?
    improvements = [(h, per_hz_kurt['154c'][h] / (per_hz_kurt['gt'][h] + 1e-6),
                     per_hz_kurt['154b'][h] / (per_hz_kurt['gt'][h] + 1e-6))
                    for h in range(29)]

    results['step2_per_horizon'] = {
        'per_horizon_kurtosis': per_hz_kurt,
        'mean_ratio_154b': float(np.mean([per_hz_kurt['154b'][h] / (per_hz_kurt['gt'][h] + 1e-6) for h in range(29)])),
        'mean_ratio_154c': float(np.mean([per_hz_kurt['154c'][h] / (per_hz_kurt['gt'][h] + 1e-6) for h in range(29)])),
    }

    # Step 3: Per-window residual properties with MATCHED conditions
    print("\n=== Step 3: Per-Window Matched Residual Analysis ===")

    N = len(S['gt'])
    rets = M['rets']
    ts = 4540; H = 30

    # Per-window: measure the magnitude of residuals
    # For each window, the residual = sample - base_prediction
    # In 154b, this is driven only by noise z; in 154c, by noise z + condition

    # Extract the actual residuals from generated samples
    # base_pred is the average of 3 ODE runs (not stored, but we can approximate)
    # Actually, we have: 154b_sample = base_pred + res_b, 154c_sample = base_pred + res_c
    # The spread across n_samples IS the residual spread

    per_window_ensemble_spread_b = []
    per_window_ensemble_spread_c = []
    per_window_ensemble_spread_base = []
    per_window_gt_deviation = []
    window_vol = []

    for w in range(N):
        # Ensemble spread: std across n_samples, averaged over dims
        spread_base = S['base'][w].std(axis=0).mean()  # std across 50 samples
        spread_b = S['154b'][w].std(axis=0).mean()
        spread_c = S['154c'][w].std(axis=0).mean()

        per_window_ensemble_spread_base.append(spread_base)
        per_window_ensemble_spread_b.append(spread_b)
        per_window_ensemble_spread_c.append(spread_c)

        # GT deviation from ensemble mean
        gt_dev = np.abs(S['gt'][w] - S['154c'][w].mean(axis=0)).mean()
        per_window_gt_deviation.append(gt_dev)

        # History volatility
        wi = ts + w
        window_vol.append(np.std(rets[wi:wi+H]))

    spread_base = np.array(per_window_ensemble_spread_base)
    spread_b = np.array(per_window_ensemble_spread_b)
    spread_c = np.array(per_window_ensemble_spread_c)
    gt_dev = np.array(per_window_gt_deviation)
    window_vol = np.array(window_vol)

    # Correlation of ensemble spread with history volatility
    corr_base_vol = np.corrcoef(spread_base, window_vol)[0, 1]
    corr_b_vol = np.corrcoef(spread_b, window_vol)[0, 1]
    corr_c_vol = np.corrcoef(spread_c, window_vol)[0, 1]
    corr_gt_dev_vol = np.corrcoef(gt_dev, window_vol)[0, 1]

    # Correlation of ensemble spread with GT deviation
    corr_base_gt = np.corrcoef(spread_base, gt_dev)[0, 1]
    corr_b_gt = np.corrcoef(spread_b, gt_dev)[0, 1]
    corr_c_gt = np.corrcoef(spread_c, gt_dev)[0, 1]

    # CV of ensemble spread
    cv_base = spread_base.std() / spread_base.mean()
    cv_b = spread_b.std() / spread_b.mean()
    cv_c = spread_c.std() / spread_c.mean()
    cv_gt = gt_dev.std() / gt_dev.mean()

    # Turb/calm ratio of spread
    vol80 = np.percentile(window_vol, 80)
    vol20 = np.percentile(window_vol, 20)
    turb = window_vol > vol80
    calm = window_vol < vol20

    tc_base = spread_base[turb].mean() / spread_base[calm].mean()
    tc_b = spread_b[turb].mean() / spread_b[calm].mean()
    tc_c = spread_c[turb].mean() / spread_c[calm].mean()
    tc_gt = gt_dev[turb].mean() / gt_dev[calm].mean()

    print(f"  {'':>25s} {'Base':>8s} {'154b':>8s} {'154c':>8s} {'GT dev':>8s}")
    print(f"  {'Spread CV':>25s} {cv_base:>8.4f} {cv_b:>8.4f} {cv_c:>8.4f} {cv_gt:>8.4f}")
    print(f"  {'Corr(spread, hist_vol)':>25s} {corr_base_vol:>8.4f} {corr_b_vol:>8.4f} {corr_c_vol:>8.4f} {corr_gt_dev_vol:>8.4f}")
    print(f"  {'Corr(spread, GT_dev)':>25s} {corr_base_gt:>8.4f} {corr_b_gt:>8.4f} {corr_c_gt:>8.4f} {'1.000':>8s}")
    print(f"  {'Turb/Calm spread':>25s} {tc_base:>8.4f} {tc_b:>8.4f} {tc_c:>8.4f} {tc_gt:>8.4f}")
    print(f"  {'Mean spread':>25s} {spread_base.mean():>8.5f} {spread_b.mean():>8.5f} {spread_c.mean():>8.5f} {gt_dev.mean():>8.5f}")

    results['step3_per_window'] = {
        'spread_cv': {'base': cv_base, '154b': cv_b, '154c': cv_c, 'gt_dev': cv_gt},
        'corr_with_hist_vol': {'base': corr_base_vol, '154b': corr_b_vol, '154c': corr_c_vol, 'gt_dev': corr_gt_dev_vol},
        'corr_with_gt_dev': {'base': corr_base_gt, '154b': corr_b_gt, '154c': corr_c_gt},
        'turb_calm_ratio': {'base': tc_base, '154b': tc_b, '154c': tc_c, 'gt_dev': tc_gt},
        'mean_spread': {'base': float(spread_base.mean()), '154b': float(spread_b.mean()), '154c': float(spread_c.mean())},
    }

    # Step 4: Residual structure analysis
    print("\n=== Step 4: Residual Structure — Shape vs Scale ===")

    # The residuals from the 2 models may differ not in SCALE but in STRUCTURE.
    # Key insight: the condition may not change the overall scale,
    # but may change the TEMPORAL PATTERN of residuals (e.g., larger perturbations
    # at early horizons vs late horizons, or different cell-weighting).

    # Compute per-horizon spread for each model
    # spread_by_hz[model][horizon] = mean spread at that horizon
    spread_by_hz_base = S['base'].std(axis=1).mean(axis=(0, 2, 3))  # (T,)
    spread_by_hz_b = S['154b'].std(axis=1).mean(axis=(0, 2, 3))
    spread_by_hz_c = S['154c'].std(axis=1).mean(axis=(0, 2, 3))

    print(f"  Spread growth (h1 -> h30):")
    print(f"    Base: {spread_by_hz_base[0]:.5f} -> {spread_by_hz_base[-1]:.5f} (ratio {spread_by_hz_base[-1]/spread_by_hz_base[0]:.2f})")
    print(f"    154b: {spread_by_hz_b[0]:.5f} -> {spread_by_hz_b[-1]:.5f} (ratio {spread_by_hz_b[-1]/spread_by_hz_b[0]:.2f})")
    print(f"    154c: {spread_by_hz_c[0]:.5f} -> {spread_by_hz_c[-1]:.5f} (ratio {spread_by_hz_c[-1]/spread_by_hz_c[0]:.2f})")

    # Per-cell spread
    spread_by_cell_base = S['base'].std(axis=1).mean(axis=(0, 1))  # (5, 5) averaged over windows and horizons
    spread_by_cell_b = S['154b'].std(axis=1).mean(axis=(0, 1))
    spread_by_cell_c = S['154c'].std(axis=1).mean(axis=(0, 1))

    print(f"\n  Cell-level spread (reshaped as 5x5 grid):")
    print(f"    Base range: [{spread_by_cell_base.min():.5f}, {spread_by_cell_base.max():.5f}]")
    print(f"    154b range: [{spread_by_cell_b.min():.5f}, {spread_by_cell_b.max():.5f}]")
    print(f"    154c range: [{spread_by_cell_c.min():.5f}, {spread_by_cell_c.max():.5f}]")

    # Cross-cell correlation of daily changes
    # This is the key metric for Suite 9
    gen_ch_b = np.diff(S['154b'][:, 0], axis=1).reshape(-1, N_CELLS)
    gen_ch_c = np.diff(S['154c'][:, 0], axis=1).reshape(-1, N_CELLS)
    gen_ch_base = np.diff(S['base'][:, 0], axis=1).reshape(-1, N_CELLS)

    corr_b = np.corrcoef(gen_ch_b.T)
    corr_c = np.corrcoef(gen_ch_c.T)
    corr_base = np.corrcoef(gen_ch_base.T)
    corr_gt = np.corrcoef(gt_ch.T)

    def eff_rank(corr_mat):
        ev = np.linalg.eigvalsh(corr_mat)[::-1]
        ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10); p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    er_b = eff_rank(corr_b)
    er_c = eff_rank(corr_c)
    er_base = eff_rank(corr_base)
    er_gt = eff_rank(corr_gt)

    print(f"\n  Effective rank: base={er_base:.2f}, 154b={er_b:.2f}, 154c={er_c:.2f}, GT={er_gt:.2f}")
    print(f"  Mean |corr|: base={np.abs(corr_base).mean():.3f}, 154b={np.abs(corr_b).mean():.3f}, "
          f"154c={np.abs(corr_c).mean():.3f}, GT={np.abs(corr_gt).mean():.3f}")

    results['step4_structure'] = {
        'spread_growth': {
            'base': {'h1': float(spread_by_hz_base[0]), 'h30': float(spread_by_hz_base[-1])},
            '154b': {'h1': float(spread_by_hz_b[0]), 'h30': float(spread_by_hz_b[-1])},
            '154c': {'h1': float(spread_by_hz_c[0]), 'h30': float(spread_by_hz_c[-1])},
        },
        'eff_rank': {'base': er_base, '154b': er_b, '154c': er_c, 'gt': er_gt},
        'mean_abs_corr': {
            'base': float(np.abs(corr_base).mean()),
            '154b': float(np.abs(corr_b).mean()),
            '154c': float(np.abs(corr_c).mean()),
            'gt': float(np.abs(corr_gt).mean()),
        },
    }

    # Step 5: Condition ablation with correct eval protocol
    print("\n=== Step 5: Condition Ablation (Correct Protocol) ===")

    # Generate 154c with different conditions per window
    ablation_results = {}
    H = 30; ts = 4540
    n_abl_windows = 80  # Fewer for speed
    n_abl_samples = 1  # Just 1 sample for kurtosis (uses all windows)

    ablation_configs = {
        'real_cond': 'real',
        'zero_cond': 'zero',
        'mean_cond': 'mean',
        'shuffled_cond': 'shuffled',
    }

    # Compute mean condition
    mean_cond = torch.from_numpy(S['conds'].mean(axis=0, keepdims=True).astype(np.float32)).to(DEVICE)

    for label, mode in ablation_configs.items():
        print(f"  Ablation: {label}...")
        all_samp = []

        with torch.no_grad():
            for w in range(min(n_abl_windows, N)):
                h = torch.from_numpy(M['surfaces'][ts+w:ts+w+H][None].astype(np.float32)).to(DEVICE)
                real_cond = M['encoder'](normalize_iv(h))

                # Base pred
                bp_runs = []
                for _ in range(3):
                    x = torch.randn(1, DIM, device=DEVICE)
                    for s in range(N_STEPS):
                        t = torch.full((1,), s * DT, device=DEVICE)
                        x = x + M['base_model'](x, t, cond=real_cond) * DT
                    bp_runs.append((x * M['bs'] + M['bm']).clamp(0, 1))
                bp = torch.stack(bp_runs).mean(0)

                # Choose condition based on mode
                if mode == 'real':
                    cond = real_cond
                elif mode == 'zero':
                    cond = torch.zeros_like(real_cond)
                elif mode == 'mean':
                    cond = mean_cond
                elif mode == 'shuffled':
                    # Random permutation of window indices
                    rand_w = np.random.randint(0, N)
                    cond = torch.from_numpy(S['conds'][rand_w:rand_w+1].astype(np.float32)).to(DEVICE)

                # Generate residual
                xc = torch.randn(1, DIM, device=DEVICE)
                for s in range(N_STEPS):
                    t = torch.full((1,), s * DT, device=DEVICE)
                    xc = xc + M['model_c'](xc, t, cond=cond) * DT
                combined = (bp + xc * M['rs_c'] + M['rm_c']).clamp(0, 1).cpu().numpy().reshape(T_FRAMES, 5, 5)
                all_samp.append(combined)

        samp_arr = np.array(all_samp)  # (n_abl_windows, T, 5, 5)
        gt_abl = S['gt'][:n_abl_windows]

        gen_ch_abl = np.diff(samp_arr, axis=1).reshape(-1, N_CELLS)
        gt_ch_abl = np.diff(gt_abl, axis=1).reshape(-1, N_CELLS)
        kr = kurtosis(gen_ch_abl.flatten(), fisher=True) / (kurtosis(gt_ch_abl.flatten(), fisher=True) + 1e-6)

        ablation_results[label] = {
            'kurtosis_ratio': kr,
            'kurtosis_abs': float(kurtosis(gen_ch_abl.flatten(), fisher=True)),
        }
        print(f"    {label}: kurtosis_ratio = {kr:.3f}")

    results['step5_ablation'] = ablation_results

    # Step 6: Deeper analysis — what the condition actually changes
    print("\n=== Step 6: What the Condition Changes ===")

    # The condition might not change SCALE but might change which CELLS/HORIZONS
    # get perturbed more or less, leading to a different daily change distribution.

    # Compare per-cell daily change distributions between 154b and 154c
    cell_ks = {'154b_vs_gt': [], '154c_vs_gt': [], '154b_vs_154c': []}
    cell_std_ratio = {'154b': [], '154c': []}

    for c in range(N_CELLS):
        ks_b, _ = ks_2samp(b_ch[:, c], gt_ch[:, c])
        ks_c, _ = ks_2samp(c_ch[:, c], gt_ch[:, c])
        ks_bc, _ = ks_2samp(b_ch[:, c], c_ch[:, c])
        cell_ks['154b_vs_gt'].append(ks_b)
        cell_ks['154c_vs_gt'].append(ks_c)
        cell_ks['154b_vs_154c'].append(ks_bc)
        cell_std_ratio['154b'].append(b_ch[:, c].std() / (gt_ch[:, c].std() + 1e-8))
        cell_std_ratio['154c'].append(c_ch[:, c].std() / (gt_ch[:, c].std() + 1e-8))

    # How many cells does 154c improve vs 154b?
    improved_cells = sum(1 for c in range(N_CELLS)
                        if cell_ks['154c_vs_gt'][c] < cell_ks['154b_vs_gt'][c])

    print(f"  KS(model, GT) per cell:")
    print(f"    154b mean: {np.mean(cell_ks['154b_vs_gt']):.4f}")
    print(f"    154c mean: {np.mean(cell_ks['154c_vs_gt']):.4f}")
    print(f"    154c better for {improved_cells}/25 cells")
    print(f"  KS(154b, 154c) per cell: mean={np.mean(cell_ks['154b_vs_154c']):.4f}")
    print(f"  Std ratio (model/GT) per cell:")
    print(f"    154b: mean={np.mean(cell_std_ratio['154b']):.3f}, range=[{min(cell_std_ratio['154b']):.3f}, {max(cell_std_ratio['154b']):.3f}]")
    print(f"    154c: mean={np.mean(cell_std_ratio['154c']):.3f}, range=[{min(cell_std_ratio['154c']):.3f}, {max(cell_std_ratio['154c']):.3f}]")

    results['step6_cell_analysis'] = {
        'ks_vs_gt': {
            '154b': cell_ks['154b_vs_gt'],
            '154c': cell_ks['154c_vs_gt'],
        },
        'ks_between_models': cell_ks['154b_vs_154c'],
        'improved_cells': improved_cells,
        'std_ratio': cell_std_ratio,
    }

    # Plot: KS per cell
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cells = np.arange(N_CELLS)

    ax = axes[0]
    ax.bar(cells - 0.2, cell_ks['154b_vs_gt'], 0.35, label='154b vs GT', alpha=0.7, color='red')
    ax.bar(cells + 0.2, cell_ks['154c_vs_gt'], 0.35, label='154c vs GT', alpha=0.7, color='blue')
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='KS threshold')
    ax.set_xlabel('Cell index')
    ax.set_ylabel('KS statistic')
    ax.set_title('Per-Cell KS(model, GT) for Daily Changes')
    ax.legend()

    ax = axes[1]
    ax.bar(cells - 0.2, cell_std_ratio['154b'], 0.35, label='154b/GT', alpha=0.7, color='red')
    ax.bar(cells + 0.2, cell_std_ratio['154c'], 0.35, label='154c/GT', alpha=0.7, color='blue')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Cell index')
    ax.set_ylabel('Std ratio (model/GT)')
    ax.set_title('Per-Cell Std Ratio')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTDIR / 'per_cell_ks_and_std.png', dpi=150)
    plt.close()
    print(f"  Saved: per_cell_ks_and_std.png")

    # Step 7: Ensemble-level analysis (multi-sample kurtosis)
    print("\n=== Step 7: Multi-Sample Kurtosis (All Ensemble Members) ===")

    # Instead of just sample 0, use ALL n_samples. This pools daily changes
    # across windows AND ensemble members.
    for label, arr in [('base', S['base']), ('154b', S['154b']), ('154c', S['154c'])]:
        all_ch = np.diff(arr[:, :, :, :, :], axis=2)  # (N, ns, 29, 5, 5)
        all_ch_flat = all_ch.reshape(-1, N_CELLS)  # (N*ns*29, 25)
        k = kurtosis(all_ch_flat.flatten(), fisher=True)
        kr = k / (gt_kurt + 1e-6)
        print(f"  {label} all-member kurtosis ratio: {kr:.3f} (abs kurtosis: {k:.3f})")

    # Save final results
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    summary = {
        'reproduced_kurtosis_ratios': {
            'base_153a': kr_base,
            '154b_uncond': kr_b,
            '154c_cond': kr_c,
        },
        'v1_hypothesis_scale_modulation': {
            'supported': False,
            'reason': (
                "Per-window spread CV is nearly identical between 154b and 154c "
                f"(154b={cv_b:.4f} vs 154c={cv_c:.4f} vs GT={cv_gt:.4f}). "
                "Condition does NOT significantly modulate residual scale."
            ),
        },
        'ablation_summary': {k: v['kurtosis_ratio'] for k, v in ablation_results.items()},
        'per_window_spread': results['step3_per_window'],
        'factor_structure': results['step4_structure'],
        'cell_analysis_improved': improved_cells,
    }

    results['summary'] = summary

    # Save
    results_ser = make_serializable(results)
    with open(RESULTDIR / '154c_kurtosis.json', 'w') as f:
        json.dump(results_ser, f, indent=2)
    print(f"\n  Results saved to: {RESULTDIR / '154c_kurtosis.json'}")
    print(f"  Plots saved to: {OUTDIR}/")

    print(f"\n  Key findings:")
    print(f"    1. Kurtosis ratios: base={kr_base:.3f}, 154b={kr_b:.3f}, 154c={kr_c:.3f}")
    print(f"    2. Scale modulation hypothesis: REJECTED")
    print(f"    3. Ablation: zero={ablation_results['zero_cond']['kurtosis_ratio']:.3f}, "
          f"mean={ablation_results['mean_cond']['kurtosis_ratio']:.3f}, "
          f"shuffled={ablation_results['shuffled_cond']['kurtosis_ratio']:.3f}, "
          f"real={ablation_results['real_cond']['kurtosis_ratio']:.3f}")
    print(f"    4. Eff rank: base={er_base:.2f}, 154b={er_b:.2f}, 154c={er_c:.2f}, GT={er_gt:.2f}")
    print(f"    5. Cells improved by conditioning: {improved_cells}/25")


if __name__ == "__main__":
    main()
