"""
Diagnostic experiments for Block-AR model failure modes.
Runs 4 experiments: arbitrage decomposition, kurtosis by horizon,
max_residual effect, and grid correlation structure.
"""

import sys
import torch
import numpy as np
from scipy import stats

# ============================================================================
# Load model and data
# ============================================================================
print("=" * 70)
print("LOADING MODEL AND DATA")
print("=" * 70)

from diffusion.block_ar.block_ar_ddpm import BlockARConfig, ConditionalBlockARDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

checkpoint = torch.load('models/backfill/block_ar/final_model.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']
if isinstance(config, dict):
    config = BlockARConfig(**config)
model = ConditionalBlockARDDPM(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

data = np.load('data/vol_surface_with_ret.npz')
surfaces = data['surface']
dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)

print(f"Model loaded. Config: bottleneck_dim={config.bottleneck_dim}, "
      f"block_size={config.block_size}, n_steps={config.n_steps}")
print(f"Dataset: {len(dataset)} windows")

# Tenor multipliers for calendar arbitrage (relative tenor values)
# 5 tenors ordered short to long; using index+1 as proxy for tenor
TENORS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def check_calendar_arb(surfaces_5x5):
    """
    Check calendar arbitrage per tenor pair.
    surfaces_5x5: (..., 5, 5) where dim -2 is tenor (short to long), dim -1 is strike.
    Total variance = IV^2 * tenor must be monotonically increasing in tenor.
    Returns: violations per tenor pair (4 pairs), shape (..., 4, 5) -> per pair, per strike
    """
    # surfaces_5x5: (..., 5, 5)
    iv = surfaces_5x5  # (..., 5_tenor, 5_strike)
    # Total variance
    tenors = torch.tensor(TENORS, dtype=iv.dtype, device=iv.device)
    # Reshape tenors to broadcast: (5, 1)
    total_var = iv ** 2 * tenors.view(*([1] * (iv.dim() - 2)), 5, 1)
    # Check monotonicity: total_var[i+1] >= total_var[i]
    diffs = total_var[..., 1:, :] - total_var[..., :-1, :]  # (..., 4, 5)
    violations = (diffs < 0)  # True where violated
    return violations


def check_butterfly_arb(surfaces_5x5):
    """
    Check butterfly arbitrage per tenor row.
    surfaces_5x5: (..., 5, 5) where dim -2 is tenor, dim -1 is strike (low to high).
    Convexity: IV[j-1] - 2*IV[j] + IV[j+1] >= 0 for each tenor row.
    Returns: violations per tenor, per interior strike (5 tenors, 3 interior strikes)
    """
    iv = surfaces_5x5  # (..., 5_tenor, 5_strike)
    # Second differences along strike dimension
    second_diff = iv[..., :-2] - 2 * iv[..., 1:-1] + iv[..., 2:]  # (..., 5, 3)
    violations = (second_diff < 0)
    return violations


def generate_samples(model, dataset, n_windows, n_samples, max_residual=20):
    """Generate samples from model for given number of test windows."""
    all_samples = []
    all_gt = []
    all_hist = []
    for i in range(min(n_windows, len(dataset))):
        item = dataset[i]
        history = item["history"].unsqueeze(0)  # (1, 30, 5, 5)
        future = item["future"].unsqueeze(0)

        with torch.no_grad():
            samples = model.sample(history, n_samples=n_samples, max_residual=max_residual)
        # samples: (1, n_samples, 30, 5, 5) in [0, 1]

        # Also denormalize ground truth
        gt = denormalize_iv(future)  # (1, 30, 5, 5) in [0, 1]

        all_samples.append(samples.squeeze(0))  # (n_samples, 30, 5, 5)
        all_gt.append(gt.squeeze(0))  # (30, 5, 5)
        all_hist.append(history.squeeze(0))  # (30, 5, 5)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i+1}/{n_windows} windows")

    return all_samples, all_gt, all_hist


# ============================================================================
# EXPERIMENT 1: Arbitrage Decomposition
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: ARBITRAGE DECOMPOSITION")
print("=" * 70)

n_windows_exp1 = 5
n_samples_exp1 = 20

print(f"Generating samples: {n_windows_exp1} windows x {n_samples_exp1} samples...")
samples_1, gt_1, hist_1 = generate_samples(model, dataset, n_windows_exp1, n_samples_exp1)

# Calendar arbitrage decomposition
print("\n--- Calendar Arbitrage (per tenor pair) ---")
print("Tenor pairs: 0→1, 1→2, 2→3, 3→4")

# Generated samples
gen_cal_violations = []
for win_idx in range(n_windows_exp1):
    samples = samples_1[win_idx]  # (n_samples, 30, 5, 5)
    for s in range(n_samples_exp1):
        for t in range(30):
            surf = samples[s, t]  # (5, 5)
            viol = check_calendar_arb(surf)  # (4, 5)
            gen_cal_violations.append(viol.numpy())

gen_cal_violations = np.array(gen_cal_violations)  # (N, 4, 5)
gen_cal_rate_per_pair = gen_cal_violations.mean(axis=(0, 2))  # Per tenor pair
gen_cal_rate_per_strike = gen_cal_violations.mean(axis=(0, 1))  # Per strike
gen_cal_rate_total = gen_cal_violations.mean()

print(f"Generated - Overall calendar arb rate: {gen_cal_rate_total:.4f} ({gen_cal_rate_total*100:.1f}%)")
print(f"  Per tenor pair: {['%.4f' % v for v in gen_cal_rate_per_pair]}")
print(f"  Per strike:     {['%.4f' % v for v in gen_cal_rate_per_strike]}")

# Ground truth
gt_cal_violations = []
for win_idx in range(n_windows_exp1):
    gt = gt_1[win_idx]  # (30, 5, 5)
    for t in range(30):
        viol = check_calendar_arb(gt[t])  # (4, 5)
        gt_cal_violations.append(viol.numpy())

gt_cal_violations = np.array(gt_cal_violations)
gt_cal_rate_per_pair = gt_cal_violations.mean(axis=(0, 2))
gt_cal_rate_per_strike = gt_cal_violations.mean(axis=(0, 1))
gt_cal_rate_total = gt_cal_violations.mean()

print(f"\nGround truth - Overall calendar arb rate: {gt_cal_rate_total:.4f} ({gt_cal_rate_total*100:.1f}%)")
print(f"  Per tenor pair: {['%.4f' % v for v in gt_cal_rate_per_pair]}")
print(f"  Per strike:     {['%.4f' % v for v in gt_cal_rate_per_strike]}")

# Butterfly arbitrage decomposition
print("\n--- Butterfly Arbitrage (per tenor row) ---")
print("Tenors: 0 (shortest) to 4 (longest)")

# Generated samples
gen_bfly_violations = []
for win_idx in range(n_windows_exp1):
    samples = samples_1[win_idx]
    for s in range(n_samples_exp1):
        for t in range(30):
            surf = samples[s, t]
            viol = check_butterfly_arb(surf)  # (5, 3)
            gen_bfly_violations.append(viol.numpy())

gen_bfly_violations = np.array(gen_bfly_violations)  # (N, 5, 3)
gen_bfly_rate_per_tenor = gen_bfly_violations.mean(axis=(0, 2))
gen_bfly_rate_per_strike = gen_bfly_violations.mean(axis=(0, 1))
gen_bfly_rate_total = gen_bfly_violations.mean()

print(f"Generated - Overall butterfly arb rate: {gen_bfly_rate_total:.4f} ({gen_bfly_rate_total*100:.1f}%)")
print(f"  Per tenor:  {['%.4f' % v for v in gen_bfly_rate_per_tenor]}")
print(f"  Per strike: {['%.4f' % v for v in gen_bfly_rate_per_strike]}")

# Ground truth
gt_bfly_violations = []
for win_idx in range(n_windows_exp1):
    gt = gt_1[win_idx]
    for t in range(30):
        viol = check_butterfly_arb(gt[t])
        gt_bfly_violations.append(viol.numpy())

gt_bfly_violations = np.array(gt_bfly_violations)
gt_bfly_rate_per_tenor = gt_bfly_violations.mean(axis=(0, 2))
gt_bfly_rate_per_strike = gt_bfly_violations.mean(axis=(0, 1))
gt_bfly_rate_total = gt_bfly_violations.mean()

print(f"\nGround truth - Overall butterfly arb rate: {gt_bfly_rate_total:.4f} ({gt_bfly_rate_total*100:.1f}%)")
print(f"  Per tenor:  {['%.4f' % v for v in gt_bfly_rate_per_tenor]}")
print(f"  Per strike: {['%.4f' % v for v in gt_bfly_rate_per_strike]}")


# ============================================================================
# EXPERIMENT 2: Kurtosis by Horizon
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: KURTOSIS BY HORIZON")
print("=" * 70)

n_windows_exp2 = 10
n_samples_exp2 = 20

print(f"Generating samples: {n_windows_exp2} windows x {n_samples_exp2} samples...")
samples_2, gt_2, hist_2 = generate_samples(model, dataset, n_windows_exp2, n_samples_exp2)

# Compute one-step changes (diff along time axis)
horizon_bins = [(0, 5), (5, 10), (10, 20), (20, 30)]
block_boundaries = [9, 10, 19, 20]  # Block boundaries at frames 10, 20

print("\n--- Kurtosis of one-step changes by horizon ---")
print(f"{'Horizon':>12} | {'Gen Kurtosis':>14} | {'GT Kurtosis':>14} | {'Ratio':>8}")
print("-" * 60)

for h_start, h_end in horizon_bins:
    # Generated: collect all one-step changes in this horizon range
    gen_diffs = []
    for win_idx in range(n_windows_exp2):
        samples = samples_2[win_idx]  # (n_samples, 30, 5, 5)
        # Compute diff: (n_samples, 29, 5, 5)
        diffs = samples[:, 1:, :, :] - samples[:, :-1, :, :]
        # Select horizon range (diffs are indexed 0..28, corresponding to transitions 0→1, 1→2, etc.)
        gen_diffs.append(diffs[:, h_start:min(h_end, 29), :, :].numpy().flatten())

    gen_diffs = np.concatenate(gen_diffs)
    gen_kurt = stats.kurtosis(gen_diffs, fisher=True)  # excess kurtosis

    # Ground truth
    gt_diffs = []
    for win_idx in range(n_windows_exp2):
        gt = gt_2[win_idx]  # (30, 5, 5)
        diffs = gt[1:, :, :] - gt[:-1, :, :]
        gt_diffs.append(diffs[h_start:min(h_end, 29), :, :].numpy().flatten())

    gt_diffs = np.concatenate(gt_diffs)
    gt_kurt = stats.kurtosis(gt_diffs, fisher=True)

    ratio = gen_kurt / gt_kurt if gt_kurt != 0 else float('inf')
    label = f"h={h_start+1}-{h_end}"
    print(f"{label:>12} | {gen_kurt:>14.3f} | {gt_kurt:>14.3f} | {ratio:>8.4f}")

# Check block boundary kurtosis specifically
print("\n--- Kurtosis at block boundaries vs non-boundaries ---")
for boundary in block_boundaries:
    if boundary >= 29:
        continue
    gen_boundary_diffs = []
    gen_nonboundary_diffs = []
    for win_idx in range(n_windows_exp2):
        samples = samples_2[win_idx]
        diffs = samples[:, 1:, :, :] - samples[:, :-1, :, :]
        gen_boundary_diffs.append(diffs[:, boundary, :, :].numpy().flatten())
        # Non-boundary: the frame before
        if boundary > 0:
            gen_nonboundary_diffs.append(diffs[:, boundary-1, :, :].numpy().flatten())

    gen_boundary_diffs = np.concatenate(gen_boundary_diffs)
    gen_nonboundary_diffs = np.concatenate(gen_nonboundary_diffs)
    kurt_boundary = stats.kurtosis(gen_boundary_diffs, fisher=True)
    kurt_nonboundary = stats.kurtosis(gen_nonboundary_diffs, fisher=True)
    print(f"  Transition {boundary}→{boundary+1}: kurtosis={kurt_boundary:.3f}  "
          f"(prev transition {boundary-1}→{boundary}: {kurt_nonboundary:.3f})")


# ============================================================================
# EXPERIMENT 3: Effect of max_residual on kurtosis
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: EFFECT OF MAX_RESIDUAL ON METRICS")
print("=" * 70)

n_windows_exp3 = 3
n_samples_exp3 = 20
max_residuals = [0, 5, 10, 20, 50]

print(f"Testing max_residual values: {max_residuals}")
print(f"Using {n_windows_exp3} windows x {n_samples_exp3} samples each\n")

print(f"{'max_residual':>12} | {'Kurtosis':>10} | {'Kurt Ratio':>10} | "
      f"{'Cal Arb %':>10} | {'Bfly Arb %':>10} | {'90% CI Cov':>10}")
print("-" * 80)

# Compute GT kurtosis once
gt_diffs_all = []
for i in range(n_windows_exp3):
    item = dataset[i]
    gt = denormalize_iv(item["future"].unsqueeze(0)).squeeze(0)  # (30, 5, 5)
    diffs = gt[1:] - gt[:-1]
    gt_diffs_all.append(diffs.numpy().flatten())
gt_diffs_all = np.concatenate(gt_diffs_all)
gt_kurtosis = stats.kurtosis(gt_diffs_all, fisher=True)

for mr in max_residuals:
    samples_mr, gt_mr, _ = generate_samples(model, dataset, n_windows_exp3, n_samples_exp3, max_residual=mr)

    # Kurtosis
    gen_diffs_mr = []
    for win_idx in range(n_windows_exp3):
        s = samples_mr[win_idx]
        d = s[:, 1:] - s[:, :-1]
        gen_diffs_mr.append(d.numpy().flatten())
    gen_diffs_mr = np.concatenate(gen_diffs_mr)
    gen_kurt_mr = stats.kurtosis(gen_diffs_mr, fisher=True)
    kurt_ratio = gen_kurt_mr / gt_kurtosis if gt_kurtosis != 0 else float('inf')

    # Calendar arbitrage
    cal_viols = 0
    cal_total = 0
    for win_idx in range(n_windows_exp3):
        s = samples_mr[win_idx]
        for si in range(n_samples_exp3):
            for t in range(30):
                v = check_calendar_arb(s[si, t])
                cal_viols += v.sum().item()
                cal_total += v.numel()
    cal_rate = cal_viols / cal_total if cal_total > 0 else 0

    # Butterfly arbitrage
    bfly_viols = 0
    bfly_total = 0
    for win_idx in range(n_windows_exp3):
        s = samples_mr[win_idx]
        for si in range(n_samples_exp3):
            for t in range(30):
                v = check_butterfly_arb(s[si, t])
                bfly_viols += v.sum().item()
                bfly_total += v.numel()
    bfly_rate = bfly_viols / bfly_total if bfly_total > 0 else 0

    # CI coverage (90%)
    ci_hits = 0
    ci_total = 0
    for win_idx in range(n_windows_exp3):
        s = samples_mr[win_idx]  # (n_samples, 30, 5, 5)
        gt = gt_mr[win_idx]  # (30, 5, 5)
        p5 = np.percentile(s.numpy(), 5, axis=0)
        p95 = np.percentile(s.numpy(), 95, axis=0)
        within = (gt.numpy() >= p5) & (gt.numpy() <= p95)
        ci_hits += within.sum()
        ci_total += within.size
    ci_cov = ci_hits / ci_total if ci_total > 0 else 0

    print(f"{mr:>12} | {gen_kurt_mr:>10.3f} | {kurt_ratio:>10.4f} | "
          f"{cal_rate*100:>9.1f}% | {bfly_rate*100:>9.1f}% | {ci_cov*100:>9.1f}%")


# ============================================================================
# EXPERIMENT 4: Grid Correlation Structure
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: GRID CORRELATION STRUCTURE")
print("=" * 70)

n_windows_exp4 = 50
n_samples_exp4 = 1  # Just need 1 sample per window for correlation

print(f"Computing correlations from {n_windows_exp4} windows...")

# Ground truth: flatten each 5x5 surface to 25-dim, compute correlations
gt_surfaces_flat = []
for i in range(min(n_windows_exp4, len(dataset))):
    item = dataset[i]
    gt = denormalize_iv(item["future"].unsqueeze(0)).squeeze(0)  # (30, 5, 5)
    for t in range(30):
        gt_surfaces_flat.append(gt[t].numpy().flatten())  # 25-dim

gt_surfaces_flat = np.array(gt_surfaces_flat)  # (N, 25)
gt_corr = np.corrcoef(gt_surfaces_flat.T)  # (25, 25)

# Generated: same
print("Generating samples for correlation analysis...")
gen_surfaces_flat = []
for i in range(min(n_windows_exp4, len(dataset))):
    item = dataset[i]
    history = item["history"].unsqueeze(0)
    with torch.no_grad():
        samples = model.sample(history, n_samples=1, max_residual=20)
    # samples: (1, 1, 30, 5, 5) in [0, 1]
    s = samples.squeeze(0).squeeze(0)  # (30, 5, 5)
    for t in range(30):
        gen_surfaces_flat.append(s[t].numpy().flatten())

    if (i + 1) % 10 == 0:
        print(f"  Generated {i+1}/{n_windows_exp4} windows")

gen_surfaces_flat = np.array(gen_surfaces_flat)  # (N, 25)
gen_corr = np.corrcoef(gen_surfaces_flat.T)  # (25, 25)

# Compare correlation matrices
corr_diff = gen_corr - gt_corr
corr_mae = np.abs(corr_diff).mean()
corr_max = np.abs(corr_diff).max()

# Frobenius norm of difference
corr_fro = np.linalg.norm(corr_diff, 'fro')

print(f"\n--- Correlation Matrix Comparison ---")
print(f"MAE between GT and Gen correlation matrices: {corr_mae:.4f}")
print(f"Max absolute difference: {corr_max:.4f}")
print(f"Frobenius norm of difference: {corr_fro:.4f}")

# Block-diagonal analysis: within-tenor (5 blocks of 5) and within-strike
# For a 5x5 grid flattened row-major: indices 0-4 = tenor 0, 5-9 = tenor 1, etc.
print("\n--- Within-tenor correlations (should be high) ---")
for tenor in range(5):
    start = tenor * 5
    end = start + 5
    gt_within = gt_corr[start:end, start:end]
    gen_within = gen_corr[start:end, start:end]
    # Off-diagonal mean
    mask = ~np.eye(5, dtype=bool)
    gt_mean = gt_within[mask].mean()
    gen_mean = gen_within[mask].mean()
    print(f"  Tenor {tenor}: GT within-tenor corr = {gt_mean:.4f}, Gen = {gen_mean:.4f}, diff = {gen_mean - gt_mean:.4f}")

print("\n--- Within-strike correlations (across tenors, should be high) ---")
for strike in range(5):
    indices = [strike + 5 * t for t in range(5)]
    gt_within = gt_corr[np.ix_(indices, indices)]
    gen_within = gen_corr[np.ix_(indices, indices)]
    mask = ~np.eye(5, dtype=bool)
    gt_mean = gt_within[mask].mean()
    gen_mean = gen_within[mask].mean()
    print(f"  Strike {strike}: GT within-strike corr = {gt_mean:.4f}, Gen = {gen_mean:.4f}, diff = {gen_mean - gt_mean:.4f}")

# Cross-block correlations (between different tenors at different strikes)
print("\n--- Cross-block correlations (between different tenors & strikes) ---")
cross_gt = []
cross_gen = []
for i in range(25):
    for j in range(i+1, 25):
        tenor_i, strike_i = i // 5, i % 5
        tenor_j, strike_j = j // 5, j % 5
        if tenor_i != tenor_j and strike_i != strike_j:
            cross_gt.append(gt_corr[i, j])
            cross_gen.append(gen_corr[i, j])

cross_gt = np.array(cross_gt)
cross_gen = np.array(cross_gen)
print(f"  GT cross-block mean corr: {cross_gt.mean():.4f} (std={cross_gt.std():.4f})")
print(f"  Gen cross-block mean corr: {cross_gen.mean():.4f} (std={cross_gen.std():.4f})")
print(f"  Diff: {cross_gen.mean() - cross_gt.mean():.4f}")

# Overall correlation of correlations
flat_gt = gt_corr[np.triu_indices(25, k=1)]
flat_gen = gen_corr[np.triu_indices(25, k=1)]
corr_of_corr = np.corrcoef(flat_gt, flat_gen)[0, 1]
print(f"\nCorrelation of correlation structures (GT vs Gen): {corr_of_corr:.4f}")

# Print actual correlation matrices (condensed)
print("\n--- Ground Truth Correlation Matrix (5x5 grid, row-major) ---")
print("       ", "  ".join([f"t{i}s{j}" for i in range(5) for j in range(5)][:10]))
for i in range(5):
    row = gt_corr[i*5:(i+1)*5, :10]
    print(f"t{i//5}s{i%5}: ", " ".join([f"{v:+.2f}" for v in row.flatten()[:10]]))

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
