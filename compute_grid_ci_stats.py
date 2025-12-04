"""
Compute best and worst grid point CI violations for each method.
"""
import numpy as np

def compute_ci_violations_per_grid(predictions, ground_truth):
    """
    Compute CI violations for each grid point.

    Args:
        predictions: (N, 3, 5, 5) - [p05, p50, p95]
        ground_truth: (N, 5, 5)

    Returns:
        violations_per_grid: (5, 5) - violation rate for each grid point
    """
    N = predictions.shape[0]
    violations = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            p05 = predictions[:, 0, i, j]
            p95 = predictions[:, 2, i, j]
            gt = ground_truth[:, i, j]

            # Count violations
            below = np.sum(gt < p05)
            above = np.sum(gt > p95)
            total_violations = below + above

            violations[i, j] = (total_violations / N) * 100

    return violations

# ============================================================================
# Oracle - In-Sample
# ============================================================================

print("=" * 80)
print("ORACLE - IN-SAMPLE (2004-2019)")
print("=" * 80)

oracle_data = np.load("models_backfill/insample_reconstruction_16yr.npz")

for h in [1, 7, 14, 30]:
    preds = oracle_data[f'recon_h{h}']  # (N, 3, 5, 5)
    indices = oracle_data[f'indices_h{h}']

    # Load ground truth
    data = np.load("data/vol_surface_with_ret.npz")
    gt = data["surface"][indices]

    # Compute per-grid violations
    grid_violations = compute_ci_violations_per_grid(preds, gt)

    best = grid_violations.min()
    worst = grid_violations.max()
    avg = grid_violations.mean()

    print(f"\nH{h}:")
    print(f"  Average: {avg:.1f}%")
    print(f"  Best grid point: {best:.1f}%")
    print(f"  Worst grid point: {worst:.1f}%")
    print(f"  Range: {worst - best:.1f} pp")

# ============================================================================
# Oracle - OOS
# ============================================================================

print("\n" + "=" * 80)
print("ORACLE - OOS (2019-2023)")
print("=" * 80)

oracle_oos_data = np.load("models_backfill/oos_reconstruction_16yr.npz")

for h in [1, 7, 14, 30]:
    preds = oracle_oos_data[f'recon_h{h}']
    indices = oracle_oos_data[f'indices_h{h}']

    gt = data["surface"][indices]
    grid_violations = compute_ci_violations_per_grid(preds, gt)

    best = grid_violations.min()
    worst = grid_violations.max()
    avg = grid_violations.mean()

    print(f"\nH{h}:")
    print(f"  Average: {avg:.1f}%")
    print(f"  Best grid point: {best:.1f}%")
    print(f"  Worst grid point: {worst:.1f}%")
    print(f"  Range: {worst - best:.1f} pp")

# ============================================================================
# VAE Prior - In-Sample
# ============================================================================

print("\n" + "=" * 80)
print("VAE PRIOR - IN-SAMPLE (2004-2019)")
print("=" * 80)

vae_prior_data = np.load("models_backfill/vae_prior_insample_16yr.npz")

for h in [1, 7, 14, 30]:
    preds = vae_prior_data[f'recon_h{h}']
    indices = vae_prior_data[f'indices_h{h}']

    gt = data["surface"][indices]
    grid_violations = compute_ci_violations_per_grid(preds, gt)

    best = grid_violations.min()
    worst = grid_violations.max()
    avg = grid_violations.mean()

    print(f"\nH{h}:")
    print(f"  Average: {avg:.1f}%")
    print(f"  Best grid point: {best:.1f}%")
    print(f"  Worst grid point: {worst:.1f}%")
    print(f"  Range: {worst - best:.1f} pp")

# ============================================================================
# Oracle - Crisis Period (2008-2010)
# ============================================================================

print("\n" + "=" * 80)
print("ORACLE - CRISIS (2008-2010)")
print("=" * 80)

crisis_start = 2000
crisis_end = 2765

for h in [1, 7, 14, 30]:
    preds = oracle_data[f'recon_h{h}']
    indices = oracle_data[f'indices_h{h}']

    # Filter for crisis period
    crisis_mask = (indices >= crisis_start) & (indices <= crisis_end)
    crisis_preds = preds[crisis_mask]
    crisis_indices = indices[crisis_mask]

    gt = data["surface"][crisis_indices]
    grid_violations = compute_ci_violations_per_grid(crisis_preds, gt)

    best = grid_violations.min()
    worst = grid_violations.max()
    avg = grid_violations.mean()

    print(f"\nH{h}:")
    print(f"  Average: {avg:.1f}%")
    print(f"  Best grid point: {best:.1f}%")
    print(f"  Worst grid point: {worst:.1f}%")
    print(f"  Range: {worst - best:.1f} pp")

# ============================================================================
# VAE Prior - Crisis Period
# ============================================================================

print("\n" + "=" * 80)
print("VAE PRIOR - CRISIS (2008-2010)")
print("=" * 80)

for h in [1, 7, 14, 30]:
    preds = vae_prior_data[f'recon_h{h}']
    indices = vae_prior_data[f'indices_h{h}']

    crisis_mask = (indices >= crisis_start) & (indices <= crisis_end)
    crisis_preds = preds[crisis_mask]
    crisis_indices = indices[crisis_mask]

    gt = data["surface"][crisis_indices]
    grid_violations = compute_ci_violations_per_grid(crisis_preds, gt)

    best = grid_violations.min()
    worst = grid_violations.max()
    avg = grid_violations.mean()

    print(f"\nH{h}:")
    print(f"  Average: {avg:.1f}%")
    print(f"  Best grid point: {best:.1f}%")
    print(f"  Worst grid point: {worst:.1f}%")
    print(f"  Range: {worst - best:.1f} pp")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
