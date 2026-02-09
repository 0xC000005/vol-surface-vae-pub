"""
Follow-up diagnostic: Quantify within-block vs cross-block quality degradation.
Tests the hypothesis that staggered sampling is the dominant quality degradation factor.
"""

import torch
import numpy as np
from scipy import stats

from diffusion.block_ar.block_ar_ddpm import BlockARConfig, ConditionalBlockARDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

# Load model
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

TENORS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

def check_calendar_arb(surfaces_5x5):
    iv = surfaces_5x5
    tenors = torch.tensor(TENORS, dtype=iv.dtype, device=iv.device)
    total_var = iv ** 2 * tenors.view(*([1] * (iv.dim() - 2)), 5, 1)
    diffs = total_var[..., 1:, :] - total_var[..., :-1, :]
    return (diffs < 0)

def check_butterfly_arb(surfaces_5x5):
    iv = surfaces_5x5
    second_diff = iv[..., :-2] - 2 * iv[..., 1:-1] + iv[..., 2:]
    return (second_diff < 0)


# ============================================================================
# Experiment: Per-frame quality within blocks vs at boundaries
# ============================================================================
print("=" * 70)
print("WITHIN-BLOCK vs CROSS-BLOCK QUALITY ANALYSIS")
print("=" * 70)

n_windows = 10
n_samples = 20

print(f"Generating {n_windows} windows x {n_samples} samples...")

all_samples = []
all_gt = []

for i in range(n_windows):
    item = dataset[i]
    history = item["history"].unsqueeze(0)
    future = item["future"].unsqueeze(0)

    with torch.no_grad():
        samples = model.sample(history, n_samples=n_samples, max_residual=20)

    gt = denormalize_iv(future).squeeze(0)
    all_samples.append(samples.squeeze(0))  # (n_samples, 30, 5, 5)
    all_gt.append(gt)  # (30, 5, 5)

    if (i + 1) % 5 == 0:
        print(f"  Generated {i+1}/{n_windows}")


# ============================================================================
# 1. Calendar & butterfly arb rates PER FRAME (time index 0..29)
# ============================================================================
print("\n--- Calendar Arbitrage Rate Per Frame ---")
print(f"{'Frame':>6} | {'Gen Cal%':>8} | {'GT Cal%':>8} | {'Block':>6}")
print("-" * 40)

gen_cal_by_frame = []
gt_cal_by_frame = []

for t in range(30):
    gen_viols = 0
    gen_total = 0
    for win in range(n_windows):
        for s in range(n_samples):
            v = check_calendar_arb(all_samples[win][s, t])
            gen_viols += v.sum().item()
            gen_total += v.numel()
    gen_rate = gen_viols / gen_total
    gen_cal_by_frame.append(gen_rate)

    gt_viols = 0
    gt_total = 0
    for win in range(n_windows):
        v = check_calendar_arb(all_gt[win][t])
        gt_viols += v.sum().item()
        gt_total += v.numel()
    gt_rate = gt_viols / gt_total
    gt_cal_by_frame.append(gt_rate)

    block = t // 10
    marker = " <-- boundary" if t in [9, 10, 19, 20] else ""
    print(f"{t:>6} | {gen_rate*100:>7.1f}% | {gt_rate*100:>7.1f}% | {block:>6}{marker}")

# Summary by block
print("\n--- Calendar Arb Summary by Block ---")
for block in range(3):
    start, end = block * 10, (block + 1) * 10
    gen_mean = np.mean(gen_cal_by_frame[start:end])
    gt_mean = np.mean(gt_cal_by_frame[start:end])
    print(f"  Block {block} (frames {start}-{end-1}): Gen={gen_mean*100:.1f}%, GT={gt_mean*100:.1f}%")


# ============================================================================
# 2. Per-frame kurtosis of one-step changes
# ============================================================================
print("\n--- Kurtosis of One-Step Changes Per Frame ---")
print(f"{'Trans':>8} | {'Gen Kurt':>10} | {'GT Kurt':>10} | {'Block':>6}")
print("-" * 50)

gen_kurt_by_frame = []
gt_kurt_by_frame = []

for t in range(29):
    gen_diffs = []
    for win in range(n_windows):
        d = all_samples[win][:, t+1] - all_samples[win][:, t]  # (n_samples, 5, 5)
        gen_diffs.append(d.numpy().flatten())
    gen_diffs = np.concatenate(gen_diffs)
    gen_k = stats.kurtosis(gen_diffs, fisher=True)
    gen_kurt_by_frame.append(gen_k)

    gt_diffs = []
    for win in range(n_windows):
        d = all_gt[win][t+1] - all_gt[win][t]  # (5, 5)
        gt_diffs.append(d.numpy().flatten())
    gt_diffs = np.concatenate(gt_diffs)
    gt_k = stats.kurtosis(gt_diffs, fisher=True)
    gt_kurt_by_frame.append(gt_k)

    block_from = t // 10
    block_to = (t + 1) // 10
    if block_from != block_to:
        marker = " <-- BLOCK BOUNDARY"
    else:
        marker = ""
    print(f"  {t}->{t+1} | {gen_k:>10.3f} | {gt_k:>10.3f} | {block_from}->{block_to}{marker}")

# Summary
print("\n--- Kurtosis Summary ---")
within_block_kurt = [gen_kurt_by_frame[t] for t in range(29) if t // 10 == (t+1) // 10]
boundary_kurt = [gen_kurt_by_frame[t] for t in range(29) if t // 10 != (t+1) // 10]
gt_all_kurt = np.mean(gt_kurt_by_frame)

print(f"  Within-block mean kurtosis: {np.mean(within_block_kurt):.3f} (n={len(within_block_kurt)} transitions)")
print(f"  Block-boundary mean kurtosis: {np.mean(boundary_kurt):.3f} (n={len(boundary_kurt)} transitions)")
print(f"  GT mean kurtosis: {gt_all_kurt:.3f}")
print(f"  Within-block / GT ratio: {np.mean(within_block_kurt) / gt_all_kurt:.4f}")
print(f"  Boundary / GT ratio: {np.mean(boundary_kurt) / gt_all_kurt:.4f}")


# ============================================================================
# 3. Staggered noise level analysis: what's the actual per-frame noise?
# ============================================================================
print("\n--- Staggered Noise Level Analysis ---")
print("For max_residual=20, the per-frame t_min schedule is:")
T_fut = 10  # block size
max_residual = 20
frame_idx = torch.arange(T_fut, dtype=torch.float32)
t_min = (max_residual * frame_idx / (T_fut - 1)).long()
print(f"  t_min per frame within block: {t_min.tolist()}")

# What does this mean in terms of alpha_bar (signal preservation)?
scheduler = model.scheduler
alpha_bars_at_tmin = [scheduler.alpha_bar[t].item() for t in t_min]
print(f"  alpha_bar at t_min: {['%.4f' % a for a in alpha_bars_at_tmin]}")
print(f"  signal fraction:    {['%.1f%%' % (a*100) for a in alpha_bars_at_tmin]}")
noise_fracs = [1.0 - a for a in alpha_bars_at_tmin]
print(f"  noise fraction:     {['%.1f%%' % (n*100) for n in noise_fracs]}")


# ============================================================================
# 4. Per-frame variance of generated samples (does it grow within blocks?)
# ============================================================================
print("\n--- Per-Frame Sample Variance (across n_samples) ---")
print(f"{'Frame':>6} | {'Gen Var':>10} | {'Block':>6}")
print("-" * 35)

for t in range(30):
    variances = []
    for win in range(n_windows):
        v = all_samples[win][:, t].var(dim=0).mean().item()  # var across samples, mean across grid
        variances.append(v)
    mean_var = np.mean(variances)
    block = t // 10
    marker = ""
    if t == 0:
        marker = " <-- block 0 start"
    elif t == 10:
        marker = " <-- block 1 start (fresh noise)"
    elif t == 20:
        marker = " <-- block 2 start (fresh noise)"
    print(f"{t:>6} | {mean_var:>10.6f} | {block:>6}{marker}")


# ============================================================================
# 5. ACF lag-1 within blocks vs across boundaries
# ============================================================================
print("\n--- ACF Lag-1 Within Blocks vs Across Boundaries ---")

def compute_acf_lag1(series):
    """Compute lag-1 autocorrelation of a 1D series."""
    if len(series) < 3:
        return float('nan')
    x = series - series.mean()
    c0 = np.sum(x**2)
    if c0 == 0:
        return 0.0
    c1 = np.sum(x[:-1] * x[1:])
    return c1 / c0

# For each sample, compute ACF within blocks and across
within_acfs = []
across_acfs = []

for win in range(n_windows):
    for s in range(n_samples):
        traj = all_samples[win][s].numpy()  # (30, 5, 5)
        # Flatten spatial dims, compute per-element ACF
        traj_flat = traj.reshape(30, -1)  # (30, 25)

        for dim in range(25):
            series = traj_flat[:, dim]

            # Within-block ACFs (3 blocks of 10)
            for block in range(3):
                block_series = series[block*10:(block+1)*10]
                acf = compute_acf_lag1(block_series)
                if not np.isnan(acf):
                    within_acfs.append(acf)

            # Across-boundary ACFs (using 5 frames around each boundary)
            for boundary in [10, 20]:
                cross_series = series[boundary-3:boundary+3]
                acf = compute_acf_lag1(cross_series)
                if not np.isnan(acf):
                    across_acfs.append(acf)

# GT ACFs
gt_within_acfs = []
gt_across_acfs = []
for win in range(n_windows):
    traj = all_gt[win].numpy().reshape(30, -1)
    for dim in range(25):
        series = traj[:, dim]
        for block in range(3):
            block_series = series[block*10:(block+1)*10]
            acf = compute_acf_lag1(block_series)
            if not np.isnan(acf):
                gt_within_acfs.append(acf)
        for boundary in [10, 20]:
            cross_series = series[boundary-3:boundary+3]
            acf = compute_acf_lag1(cross_series)
            if not np.isnan(acf):
                gt_across_acfs.append(acf)

print(f"  Generated within-block ACF lag-1: {np.mean(within_acfs):.4f} (std={np.std(within_acfs):.4f})")
print(f"  Generated across-boundary ACF lag-1: {np.mean(across_acfs):.4f} (std={np.std(across_acfs):.4f})")
print(f"  GT within-block ACF lag-1: {np.mean(gt_within_acfs):.4f} (std={np.std(gt_within_acfs):.4f})")
print(f"  GT across-boundary ACF lag-1: {np.mean(gt_across_acfs):.4f} (std={np.std(gt_across_acfs):.4f})")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
