#!/usr/bin/env python
"""
154c Kurtosis Mechanism — v4: Focus on Within-Cell Shape

v3 findings:
- Per-cell std calibration NOT the full story (rescaling 154b -> worse, not better)
- Per-cell heterogeneity (CV of cell stds) similar between all models
- The mechanism is NOT about SCALE but about SHAPE of within-cell distributions

New hypothesis: conditioning changes the SHAPE of per-cell residual distributions.
Without condition, the FM learns to produce residuals that are too Gaussian-like
(thin-tailed per cell). With condition, the FM can produce fat-tailed residuals
for windows that need them.

Key insight: kurtosis is NOT just about mixing different scales.
It's about the WITHIN-CELL tail behavior. If each cell's distribution is
too thin-tailed, the aggregate is thin-tailed regardless of cell heterogeneity.
"""

import json
import sys
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
from experiments.backfill.block_ar.train_oneshot_flow import FactoredVelocityTransformer

OUTDIR = Path("/home/max/Documents/vol-surface-vae-pub/results/validations/2026-03-24/analysis/154c_kurtosis")
RESULTDIR = Path("/home/max/Documents/vol-surface-vae-pub/results/validations/2026-03-24/verification_results")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_STEPS = 8; DT = 1.0 / N_STEPS; DIM = 750; T = 30; C = 25
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)


def ms(obj):
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: ms(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [ms(v) for v in obj]
    return obj


def load():
    enc, cd = load_encoder(
        '/home/max/Documents/vol-surface-vae-pub/models/backfill/block_ar_vol_scaled_30ep/best_model.pt', DEVICE)
    for p in enc.parameters(): p.requires_grad = False

    bc = torch.load('/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_153a/final_model.pt',
                     weights_only=False, map_location=DEVICE)
    bcfg = bc['config']
    base = ConditionalFactoredVelocityTransformer(
        n_frames=bcfg['n_frames'], n_cells=bcfg['n_cells'], d_model=bcfg['d_model'],
        n_heads=bcfg['n_heads'], n_layers=bcfg['n_layers'], cond_dim=bcfg['cond_dim'])
    base.load_state_dict(bc['model_state_dict']); base.to(DEVICE).eval()
    bm = torch.from_numpy(bc['train_mean']).float().to(DEVICE)
    bs = torch.from_numpy(bc['train_std']).float().to(DEVICE)

    rb = torch.load('/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_154b/final_model.pt',
                     weights_only=False, map_location=DEVICE)
    cfg = rb['config']
    mb = FactoredVelocityTransformer(n_frames=cfg['n_frames'], n_cells=cfg['n_cells'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'])
    mb.load_state_dict(rb['model_state_dict']); mb.to(DEVICE).eval()
    rmb = torch.from_numpy(rb['res_mean']).float().to(DEVICE)
    rsb = torch.from_numpy(rb['res_std']).float().to(DEVICE)

    rc = torch.load('/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_154c/final_model.pt',
                     weights_only=False, map_location=DEVICE)
    cfg = rc['config']
    mc = ConditionalFactoredVelocityTransformer(n_frames=cfg['n_frames'], n_cells=cfg['n_cells'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'], cond_dim=cfg['cond_dim'])
    mc.load_state_dict(rc['model_state_dict']); mc.to(DEVICE).eval()
    rmc = torch.from_numpy(rc['res_mean']).float().to(DEVICE)
    rsc = torch.from_numpy(rc['res_std']).float().to(DEVICE)

    data = np.load('/home/max/Documents/vol-surface-vae-pub/data/vol_surface_with_ret.npz')
    return dict(enc=enc, base=base, bm=bm, bs=bs, mb=mb, rmb=rmb, rsb=rsb,
                mc=mc, rmc=rmc, rsc=rsc, surfaces=data['surface'], rets=data['ret'])


def ode_sample(model, cond=None):
    x = torch.randn(1, DIM, device=DEVICE)
    for s in range(N_STEPS):
        t = torch.full((1,), s * DT, device=DEVICE)
        if cond is not None:
            x = x + model(x, t, cond=cond) * DT
        else:
            x = x + model(x, t) * DT
    return x


def main():
    print("=" * 70)
    print("154c Kurtosis Mechanism — v4: Within-Cell Shape Analysis")
    print("=" * 70)

    M = load()
    results = {}

    # Use 160 windows, 50 samples — exact match to eval_154c.py
    ts = 4540; H = 30; ns = 50; nw = 160
    max_i = min(ts + nw, len(M['surfaces']) - H - T + 1)
    N = max_i - ts

    print(f"\n  Generating {N} windows x {ns} samples (matching eval_154c.py)...")

    # Collect per-window, per-sample daily changes
    all_dch_b = []   # list of (ns, 29, 25) per window
    all_dch_c = []
    all_dch_gt = []  # (29, 25) per window

    with torch.no_grad():
        for w in range(N):
            wi = ts + w
            h = torch.from_numpy(M['surfaces'][wi:wi+H][None].astype(np.float32)).to(DEVICE)
            cond = M['enc'](normalize_iv(h))

            bp = torch.stack([
                (ode_sample(M['base'], cond=cond) * M['bs'] + M['bm']).clamp(0, 1)
                for _ in range(3)
            ]).mean(0)

            sb = []; sc = []
            for _ in range(ns):
                xb = ode_sample(M['mb'])
                combined_b = (bp + xb * M['rsb'] + M['rmb']).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
                sb.append(combined_b)

                xc = ode_sample(M['mc'], cond=cond)
                combined_c = (bp + xc * M['rsc'] + M['rmc']).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
                sc.append(combined_c)

            sb = np.array(sb)  # (ns, T, 5, 5)
            sc = np.array(sc)
            gt = M['surfaces'][wi+H:wi+H+T]  # (T, 5, 5)

            # Daily changes
            dch_b = np.diff(sb, axis=1).reshape(ns, 29, C)  # (ns, 29, 25)
            dch_c = np.diff(sc, axis=1).reshape(ns, 29, C)
            dch_gt = np.diff(gt, axis=0).reshape(29, C)       # (29, 25)

            all_dch_b.append(dch_b)
            all_dch_c.append(dch_c)
            all_dch_gt.append(dch_gt)

            if (w+1) % 20 == 0: print(f"    {w+1}/{N}")

    # Stack arrays
    dch_b_all = np.array(all_dch_b)   # (N, ns, 29, 25)
    dch_c_all = np.array(all_dch_c)
    dch_gt_all = np.array(all_dch_gt) # (N, 29, 25)

    # =========================================================================
    # 1. Reproduce exact kurtosis (sample 0 only, as in eval_154c.py)
    # =========================================================================
    print("\n=== 1: Reproduce Exact Kurtosis ===")

    dch_b_s0 = dch_b_all[:, 0, :, :].reshape(-1, C)  # (N*29, 25)
    dch_c_s0 = dch_c_all[:, 0, :, :].reshape(-1, C)
    dch_gt_flat = dch_gt_all.reshape(-1, C)            # (N*29, 25)

    k_gt = kurtosis(dch_gt_flat.flatten(), fisher=True)
    kr_b = kurtosis(dch_b_s0.flatten(), fisher=True) / (k_gt + 1e-6)
    kr_c = kurtosis(dch_c_s0.flatten(), fisher=True) / (k_gt + 1e-6)

    # Also with all samples (pooled across ensemble)
    dch_b_all_flat = dch_b_all.reshape(-1, C)  # (N*ns*29, 25)
    dch_c_all_flat = dch_c_all.reshape(-1, C)
    kr_b_ens = kurtosis(dch_b_all_flat.flatten(), fisher=True) / (k_gt + 1e-6)
    kr_c_ens = kurtosis(dch_c_all_flat.flatten(), fisher=True) / (k_gt + 1e-6)

    print(f"  GT excess kurtosis: {k_gt:.3f}")
    print(f"  Sample-0 kurtosis ratio: 154b={kr_b:.3f}, 154c={kr_c:.3f}")
    print(f"  All-ensemble kurtosis ratio: 154b={kr_b_ens:.3f}, 154c={kr_c_ens:.3f}")

    results['kurtosis_reproduction'] = {
        'gt_excess_kurtosis': k_gt,
        'sample_0': {'154b': kr_b, '154c': kr_c},
        'all_ensemble': {'154b': kr_b_ens, '154c': kr_c_ens},
    }

    # =========================================================================
    # 2. PER-CELL kurtosis (within each cell)
    # =========================================================================
    print("\n=== 2: Per-Cell Kurtosis (Within-Cell Tail Behavior) ===")

    # Using sample 0, compute kurtosis WITHIN each cell
    cell_kurt_gt = []
    cell_kurt_b = []
    cell_kurt_c = []

    for c_idx in range(C):
        cell_kurt_gt.append(kurtosis(dch_gt_flat[:, c_idx], fisher=True))
        cell_kurt_b.append(kurtosis(dch_b_s0[:, c_idx], fisher=True))
        cell_kurt_c.append(kurtosis(dch_c_s0[:, c_idx], fisher=True))

    cell_kurt_gt = np.array(cell_kurt_gt)
    cell_kurt_b = np.array(cell_kurt_b)
    cell_kurt_c = np.array(cell_kurt_c)

    # Per-cell kurtosis RATIO (model / GT)
    cell_kr_b = cell_kurt_b / (cell_kurt_gt + 1e-6)
    cell_kr_c = cell_kurt_c / (cell_kurt_gt + 1e-6)

    print(f"  Per-cell kurtosis (excess, within each cell):")
    print(f"    GT:   mean={cell_kurt_gt.mean():.2f}, range=[{cell_kurt_gt.min():.2f}, {cell_kurt_gt.max():.2f}]")
    print(f"    154b: mean={cell_kurt_b.mean():.2f}, range=[{cell_kurt_b.min():.2f}, {cell_kurt_b.max():.2f}]")
    print(f"    154c: mean={cell_kurt_c.mean():.2f}, range=[{cell_kurt_c.min():.2f}, {cell_kurt_c.max():.2f}]")
    print(f"\n  Per-cell kurtosis ratio (model/GT):")
    print(f"    154b: mean={cell_kr_b.mean():.3f}, cells within [0.5, 2.0]: {((cell_kr_b >= 0.5) & (cell_kr_b <= 2.0)).sum()}/25")
    print(f"    154c: mean={cell_kr_c.mean():.3f}, cells within [0.5, 2.0]: {((cell_kr_c >= 0.5) & (cell_kr_c <= 2.0)).sum()}/25")

    # Which cells does 154c improve?
    improved = cell_kr_c_closer = np.abs(cell_kr_c - 1.0) < np.abs(cell_kr_b - 1.0)
    print(f"    154c closer to GT kurtosis for: {improved.sum()}/25 cells")

    results['per_cell_kurtosis'] = {
        'gt': cell_kurt_gt.tolist(),
        '154b': cell_kurt_b.tolist(),
        '154c': cell_kurt_c.tolist(),
        'ratio_154b': cell_kr_b.tolist(),
        'ratio_154c': cell_kr_c.tolist(),
        '154c_closer_cells': int(improved.sum()),
    }

    # =========================================================================
    # 3. Per-cell std analysis (from v3 but with 160 windows)
    # =========================================================================
    print("\n=== 3: Per-Cell Std Calibration (160 windows) ===")

    std_gt = dch_gt_flat.std(axis=0)
    std_b = dch_b_s0.std(axis=0)
    std_c = dch_c_s0.std(axis=0)

    ratio_b = std_b / (std_gt + 1e-8)
    ratio_c = std_c / (std_gt + 1e-8)

    print(f"  Mean std ratio: 154b={ratio_b.mean():.3f}, 154c={ratio_c.mean():.3f}")

    # =========================================================================
    # 4. Decompose kurtosis into within-cell and between-cell components
    # =========================================================================
    print("\n=== 4: Kurtosis Decomposition ===")

    # Pooled kurtosis = f(within-cell kurtosis, between-cell heterogeneity)
    # For standardized variables pooled together:
    # If X = mixture of X_c, c=1..25, each standardized to mean 0:
    # kurt(X) depends on:
    #   (a) Individual kurt(X_c)
    #   (b) Relative weights (proportional to sigma_c^4 for excess kurtosis)
    #   (c) Fourth moments of the mixing distribution

    # Let's compute the pooled kurtosis from the cell-level statistics
    # and compare to the actual pooled kurtosis

    # Method: weighted average of cell kurtoses, weighted by sigma^4
    for label, dch, cell_k in [('154b', dch_b_s0, cell_kurt_b),
                                ('154c', dch_c_s0, cell_kurt_c),
                                ('GT', dch_gt_flat, cell_kurt_gt)]:
        stds = dch.std(axis=0)  # (25,)
        means = dch.mean(axis=0)

        # Fourth central moment of pooled distribution
        pooled = dch.flatten()
        mu_pool = pooled.mean()
        s_pool = pooled.std()

        # Direct kurtosis
        k_direct = kurtosis(pooled, fisher=True)

        # Predicted from cells (assuming independence):
        # E[X^4] = sum_c p_c * E[X_c^4]  where p_c = 1/C
        # E[X_c^4] = mu4_c = sigma_c^4 * (kurt_c + 3) (raw kurtosis)
        # E[X^2] = sum_c p_c * sigma_c^2 (assuming mean=0 per cell, approx)
        mu4_cells = np.mean(stds**4 * (cell_k + 3))
        mu2_cells = np.mean(stds**2 + means**2)
        k_predicted = mu4_cells / (mu2_cells**2 + 1e-10) - 3

        print(f"  {label}:")
        print(f"    Direct pooled kurtosis:    {k_direct:.3f}")
        print(f"    Predicted from cell stats: {k_predicted:.3f}")
        print(f"    Ratio (pred/direct):       {k_predicted / (k_direct + 1e-6):.3f}")

    results['kurtosis_decomposition'] = {}

    # =========================================================================
    # 5. The REAL test: standardize per cell, then measure kurtosis
    # =========================================================================
    print("\n=== 5: Cell-Standardized Kurtosis ===")

    # If we standardize each cell to have std=1, the REMAINING kurtosis comes
    # purely from within-cell shape (not between-cell heterogeneity)

    def cell_standardized_kurtosis(dch):
        """Standardize each cell to mean=0 std=1, then compute pooled kurtosis."""
        means = dch.mean(axis=0, keepdims=True)
        stds = dch.std(axis=0, keepdims=True) + 1e-8
        standardized = (dch - means) / stds
        return kurtosis(standardized.flatten(), fisher=True)

    ck_gt = cell_standardized_kurtosis(dch_gt_flat)
    ck_b = cell_standardized_kurtosis(dch_b_s0)
    ck_c = cell_standardized_kurtosis(dch_c_s0)

    print(f"  Cell-standardized kurtosis (pure within-cell shape):")
    print(f"    GT:   {ck_gt:.3f}")
    print(f"    154b: {ck_b:.3f}")
    print(f"    154c: {ck_c:.3f}")
    print(f"  Ratios: 154b={ck_b/ck_gt:.3f}, 154c={ck_c/ck_gt:.3f}")

    results['cell_standardized_kurtosis'] = {
        'gt': ck_gt, '154b': ck_b, '154c': ck_c,
        'ratio_154b': ck_b / ck_gt, 'ratio_154c': ck_c / ck_gt,
    }

    # =========================================================================
    # 6. Per-WINDOW kurtosis variation
    # =========================================================================
    print("\n=== 6: Per-Window Kurtosis (Across Ensemble Members) ===")

    # For each window, compute kurtosis across the 50 ensemble members
    # This tells us whether the model produces leptokurtic ensembles per window

    per_win_kurt_b = []
    per_win_kurt_c = []
    per_win_gt_volatility = []

    for w in range(N):
        # Daily changes from all ensemble members for this window
        # dch_b_all[w] shape: (ns, 29, 25)
        flat_b = dch_b_all[w].reshape(-1)  # (ns*29*25,)
        flat_c = dch_c_all[w].reshape(-1)

        per_win_kurt_b.append(kurtosis(flat_b, fisher=True))
        per_win_kurt_c.append(kurtosis(flat_c, fisher=True))
        per_win_gt_volatility.append(np.std(M['rets'][ts+w:ts+w+H]))

    per_win_kurt_b = np.array(per_win_kurt_b)
    per_win_kurt_c = np.array(per_win_kurt_c)
    per_win_gt_volatility = np.array(per_win_gt_volatility)

    # Does 154c produce higher per-window kurtosis for turbulent windows?
    vol80 = np.percentile(per_win_gt_volatility, 80)
    vol20 = np.percentile(per_win_gt_volatility, 20)
    turb = per_win_gt_volatility > vol80
    calm = per_win_gt_volatility < vol20

    print(f"  Per-window ensemble kurtosis:")
    print(f"    154b: mean={per_win_kurt_b.mean():.2f}, turb={per_win_kurt_b[turb].mean():.2f}, calm={per_win_kurt_b[calm].mean():.2f}")
    print(f"    154c: mean={per_win_kurt_c.mean():.2f}, turb={per_win_kurt_c[turb].mean():.2f}, calm={per_win_kurt_c[calm].mean():.2f}")
    print(f"  Corr(window kurt, hist vol): 154b={np.corrcoef(per_win_kurt_b, per_win_gt_volatility)[0,1]:.3f}, "
          f"154c={np.corrcoef(per_win_kurt_c, per_win_gt_volatility)[0,1]:.3f}")

    # The MIXTURE of per-window distributions determines overall kurtosis
    # Windows with higher kurtosis contribute more to aggregate heavy tails

    results['per_window_kurtosis'] = {
        '154b_mean': float(per_win_kurt_b.mean()),
        '154c_mean': float(per_win_kurt_c.mean()),
        '154b_turb': float(per_win_kurt_b[turb].mean()),
        '154c_turb': float(per_win_kurt_c[turb].mean()),
        '154b_calm': float(per_win_kurt_b[calm].mean()),
        '154c_calm': float(per_win_kurt_c[calm].mean()),
        'corr_kurt_vol_154b': float(np.corrcoef(per_win_kurt_b, per_win_gt_volatility)[0,1]),
        'corr_kurt_vol_154c': float(np.corrcoef(per_win_kurt_c, per_win_gt_volatility)[0,1]),
    }

    # =========================================================================
    # 7. The DEFINITIVE synthetic test
    # =========================================================================
    print("\n=== 7: Definitive Synthetic Tests ===")

    # Test 1: Keep 154b's within-cell SHAPE but use 154c's per-cell STD
    dch_b_shape = dch_b_s0.copy()
    for c_idx in range(C):
        # Standardize to z-score
        m = dch_b_shape[:, c_idx].mean()
        s = dch_b_shape[:, c_idx].std() + 1e-8
        z = (dch_b_shape[:, c_idx] - m) / s
        # Re-scale to 154c std
        dch_b_shape[:, c_idx] = z * std_c[c_idx] + dch_c_s0[:, c_idx].mean()

    kr_shape_swap = kurtosis(dch_b_shape.flatten(), fisher=True) / (k_gt + 1e-6)

    # Test 2: Keep 154c's within-cell SHAPE but use 154b's per-cell STD
    dch_c_shape = dch_c_s0.copy()
    for c_idx in range(C):
        m = dch_c_shape[:, c_idx].mean()
        s = dch_c_shape[:, c_idx].std() + 1e-8
        z = (dch_c_shape[:, c_idx] - m) / s
        dch_c_shape[:, c_idx] = z * std_b[c_idx] + dch_b_s0[:, c_idx].mean()

    kr_scale_swap = kurtosis(dch_c_shape.flatten(), fisher=True) / (k_gt + 1e-6)

    # Test 3: Keep 154b's within-cell SHAPE but use GT per-cell STD
    dch_b_gt_std = dch_b_s0.copy()
    for c_idx in range(C):
        m = dch_b_gt_std[:, c_idx].mean()
        s = dch_b_gt_std[:, c_idx].std() + 1e-8
        z = (dch_b_gt_std[:, c_idx] - m) / s
        dch_b_gt_std[:, c_idx] = z * std_gt[c_idx]

    kr_b_gt_std = kurtosis(dch_b_gt_std.flatten(), fisher=True) / (k_gt + 1e-6)

    # Test 4: Keep 154c's within-cell SHAPE but use GT per-cell STD
    dch_c_gt_std = dch_c_s0.copy()
    for c_idx in range(C):
        m = dch_c_gt_std[:, c_idx].mean()
        s = dch_c_gt_std[:, c_idx].std() + 1e-8
        z = (dch_c_gt_std[:, c_idx] - m) / s
        dch_c_gt_std[:, c_idx] = z * std_gt[c_idx]

    kr_c_gt_std = kurtosis(dch_c_gt_std.flatten(), fisher=True) / (k_gt + 1e-6)

    print(f"  Synthetic swap experiments (kurtosis ratio):")
    print(f"    154b original:                          {kr_b:.3f}")
    print(f"    154c original:                          {kr_c:.3f}")
    print(f"    154b shape + 154c std (scale):          {kr_shape_swap:.3f}")
    print(f"    154c shape + 154b std (scale):          {kr_scale_swap:.3f}")
    print(f"    154b shape + GT std:                    {kr_b_gt_std:.3f}")
    print(f"    154c shape + GT std:                    {kr_c_gt_std:.3f}")

    # Attribution
    delta_total = kr_c - kr_b
    delta_shape = kr_shape_swap - kr_b  # Effect of changing STD only (keeping 154b shape)
    delta_from_c_shape = kr_c - kr_scale_swap  # Effect of changing shape only (keeping 154b std)

    print(f"\n  Attribution:")
    print(f"    Total improvement (154c - 154b): {delta_total:.3f}")
    print(f"    Due to SCALE change (154b shape + 154c std): {delta_shape:.3f} ({delta_shape/max(delta_total,1e-6)*100:.0f}%)")
    print(f"    Due to SHAPE change (154c shape + 154b std): {delta_from_c_shape:.3f} ({delta_from_c_shape/max(delta_total,1e-6)*100:.0f}%)")

    results['synthetic_tests'] = {
        '154b_original': kr_b,
        '154c_original': kr_c,
        '154b_shape_154c_std': kr_shape_swap,
        '154c_shape_154b_std': kr_scale_swap,
        '154b_shape_gt_std': kr_b_gt_std,
        '154c_shape_gt_std': kr_c_gt_std,
        'total_improvement': delta_total,
        'scale_contribution': delta_shape,
        'shape_contribution': delta_from_c_shape,
        'scale_pct': float(delta_shape / max(abs(delta_total), 1e-6) * 100),
        'shape_pct': float(delta_from_c_shape / max(abs(delta_total), 1e-6) * 100),
    }

    # =========================================================================
    # 8. Per-cell QQ analysis: which cells are most different?
    # =========================================================================
    print("\n=== 8: Per-Cell QQ Divergence ===")

    # For each cell, compute KS and kurtosis difference
    cell_analysis = []
    for c_idx in range(C):
        row = c_idx // 5  # moneyness
        col = c_idx % 5   # tenor
        ks_b, _ = ks_2samp(dch_b_s0[:, c_idx], dch_gt_flat[:, c_idx])
        ks_c, _ = ks_2samp(dch_c_s0[:, c_idx], dch_gt_flat[:, c_idx])

        cell_analysis.append({
            'cell': c_idx, 'row': row, 'col': col,
            'gt_kurt': cell_kurt_gt[c_idx],
            'b_kurt': cell_kurt_b[c_idx], 'c_kurt': cell_kurt_c[c_idx],
            'b_std_ratio': float(ratio_b[c_idx]), 'c_std_ratio': float(ratio_c[c_idx]),
            'ks_b': ks_b, 'ks_c': ks_c,
            'kurt_improved': abs(cell_kr_c[c_idx] - 1.0) < abs(cell_kr_b[c_idx] - 1.0),
            'ks_improved': ks_c < ks_b,
        })

    # Summary
    n_kurt_improved = sum(1 for r in cell_analysis if r['kurt_improved'])
    n_ks_improved = sum(1 for r in cell_analysis if r['ks_improved'])
    print(f"  Cells with kurtosis closer to GT: {n_kurt_improved}/25")
    print(f"  Cells with lower KS:              {n_ks_improved}/25")

    # Print top 5 most improved and worst 5
    sorted_by_improvement = sorted(cell_analysis,
        key=lambda r: (abs(r['b_kurt'] - r['gt_kurt']) - abs(r['c_kurt'] - r['gt_kurt'])),
        reverse=True)

    print(f"\n  Top 5 most improved cells (kurtosis):")
    for r in sorted_by_improvement[:5]:
        print(f"    Cell {r['cell']:2d} (m{r['row']}t{r['col']}): "
              f"GT_k={r['gt_kurt']:.1f}, 154b_k={r['b_kurt']:.1f}, 154c_k={r['c_kurt']:.1f}")

    print(f"\n  Top 5 worst cells (kurtosis):")
    for r in sorted_by_improvement[-5:]:
        print(f"    Cell {r['cell']:2d} (m{r['row']}t{r['col']}): "
              f"GT_k={r['gt_kurt']:.1f}, 154b_k={r['b_kurt']:.1f}, 154c_k={r['c_kurt']:.1f}")

    results['cell_analysis'] = cell_analysis

    # =========================================================================
    # FINAL PLOTS
    # =========================================================================
    print("\n=== Final Plots ===")

    # Plot 1: synthetic swap results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    labels = ['154b', '154c', '154b shape\n154c std', '154c shape\n154b std',
              '154b shape\nGT std', '154c shape\nGT std']
    vals = [kr_b, kr_c, kr_shape_swap, kr_scale_swap, kr_b_gt_std, kr_c_gt_std]
    colors = ['red', 'blue', 'salmon', 'lightblue', 'darksalmon', 'cornflowerblue']
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.8)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel('Kurtosis ratio')
    ax.set_title('Shape vs Scale Swap Experiments')

    # Plot 2: per-cell kurtosis comparison
    ax = axes[1]
    cells = np.arange(C)
    ax.scatter(cell_kurt_gt, cell_kurt_b, c='red', label='154b', alpha=0.6, s=40)
    ax.scatter(cell_kurt_gt, cell_kurt_c, c='blue', label='154c', alpha=0.6, s=40)
    lim = max(cell_kurt_gt.max(), cell_kurt_b.max(), cell_kurt_c.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('GT per-cell kurtosis')
    ax.set_ylabel('Model per-cell kurtosis')
    ax.set_title('Per-Cell Kurtosis: Model vs GT')
    ax.legend()

    # Plot 3: cell-standardized kurtosis + per-window kurtosis
    ax = axes[2]
    x_pos = [0, 1, 2]
    ax.bar(x_pos, [ck_gt, ck_b, ck_c],
           color=['black', 'red', 'blue'], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['GT', '154b', '154c'])
    ax.set_ylabel('Cell-standardized excess kurtosis')
    ax.set_title('Within-Cell Shape (Cell-Normalized)')

    plt.suptitle('154c Kurtosis Mechanism: Shape vs Scale Decomposition', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'v4_shape_vs_scale.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: v4_shape_vs_scale.png")

    # Plot 4: Per-window kurtosis distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(per_win_kurt_b, bins=30, alpha=0.5, color='red', label='154b', density=True)
    ax.hist(per_win_kurt_c, bins=30, alpha=0.5, color='blue', label='154c', density=True)
    ax.axvline(per_win_kurt_b.mean(), color='red', linestyle='--', alpha=0.7)
    ax.axvline(per_win_kurt_c.mean(), color='blue', linestyle='--', alpha=0.7)
    ax.set_xlabel('Per-window ensemble kurtosis')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Per-Window Kurtosis Across 160 Windows')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / 'v4_per_window_kurtosis_dist.png', dpi=150)
    plt.close()
    print(f"  Saved: v4_per_window_kurtosis_dist.png")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Determine mechanism from evidence
    if abs(delta_shape) > abs(delta_from_c_shape):
        dominant = "SCALE (per-cell std calibration)"
        scale_dominant = True
    else:
        dominant = "SHAPE (within-cell tail behavior)"
        scale_dominant = False

    print(f"\n  Kurtosis ratio: 154b={kr_b:.3f}, 154c={kr_c:.3f}")
    print(f"  Total improvement: {delta_total:.3f}")
    print(f"  Scale contribution: {delta_shape:.3f} ({abs(delta_shape/max(abs(delta_total),1e-6))*100:.0f}%)")
    print(f"  Shape contribution: {delta_from_c_shape:.3f} ({abs(delta_from_c_shape/max(abs(delta_total),1e-6))*100:.0f}%)")
    print(f"  Dominant mechanism: {dominant}")
    print(f"\n  Cell-standardized kurtosis: GT={ck_gt:.2f}, 154b={ck_b:.2f}, 154c={ck_c:.2f}")
    print(f"  Per-cell kurtosis closer to GT: {n_kurt_improved}/25 cells")
    print(f"  Per-cell KS improved: {n_ks_improved}/25 cells")

    verdict = {
        'dominant_mechanism': dominant,
        'scale_contribution_pct': float(abs(delta_shape / max(abs(delta_total), 1e-6)) * 100),
        'shape_contribution_pct': float(abs(delta_from_c_shape / max(abs(delta_total), 1e-6)) * 100),
        'kurtosis_ratios': {'154b': kr_b, '154c': kr_c, 'improvement': delta_total},
        'cell_standardized_kurtosis': {'gt': ck_gt, '154b': ck_b, '154c': ck_c},
        'per_cell_improvements': {
            'kurtosis_closer': n_kurt_improved,
            'ks_improved': n_ks_improved,
        },
        'explanation': (
            f"The conditioning in 154c improves kurtosis through {dominant}. "
            f"Scale (per-cell std calibration) accounts for ~{abs(delta_shape/max(abs(delta_total),1e-6))*100:.0f}% "
            f"of the improvement, while shape (within-cell tail behavior) accounts for "
            f"~{abs(delta_from_c_shape/max(abs(delta_total),1e-6))*100:.0f}%. "
            f"The cell-standardized kurtosis shows that within-cell tail behavior is "
            f"{'similar' if abs(ck_b - ck_c) < 5 else 'different'} between models "
            f"(154b={ck_b:.1f} vs 154c={ck_c:.1f}), confirming that "
            f"{'both scale and shape contribute' if abs(delta_shape) > 0.01 and abs(delta_from_c_shape) > 0.01 else dominant + ' dominates'}."
        ),
    }

    results['verdict'] = verdict

    # Save
    with open(RESULTDIR / '154c_kurtosis.json', 'w') as f:
        json.dump(ms(results), f, indent=2)
    print(f"\n  Saved: {RESULTDIR / '154c_kurtosis.json'}")


if __name__ == "__main__":
    main()
