"""Evaluate baselines on 38-d joint IV + factor forecasting (Benchmark A).

Generates (B, K, 30, 38) daily changes, then:
  Panel A: Reconstruct IV surfaces, run existing 9 IV test suites
  Panel B: Factor CRPS, standardized Energy Score, Variogram Score,
           correlation Frobenius error, factor KS, per-dim raw CRPS

Usage:
    # All 6 classical baselines:
    PYTHONPATH=. python experiments/backfill/baselines/evaluate_baselines_38d.py \
        --baselines random_walk bootstrap historical_sim pca_var garch_ccc filtered_hs \
        --n_samples 50 --max_batches 20

    # Quick test:
    PYTHONPATH=. python experiments/backfill/baselines/evaluate_baselines_38d.py \
        --baselines random_walk --n_samples 10 --max_batches 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.backfill.baselines.data_loader_38d import load_aligned_38d_data, make_windows
from experiments.backfill.baselines.joint_classical_baselines import (
    JointRandomWalk, JointBootstrap, JointHistoricalSim,
    JointPCAVAR, JointGARCHCCC, JointFilteredHS,
)


CLASSICAL_BASELINES = [
    "random_walk", "bootstrap", "historical_sim",
    "pca_var", "garch_ccc", "filtered_hs",
]
DEEP_BASELINES = ["deepvar", "csdi", "timegrad"]
ALL_BASELINES = CLASSICAL_BASELINES + DEEP_BASELINES


# ---------------------------------------------------------------------------
# S3: Conditionality from pre-generated samples
# ---------------------------------------------------------------------------

def run_conditionality_from_samples(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict:
    """Test conditionality using pre-generated samples (no model needed).

    Classifies windows into turbulent/calm by vol-of-vol of history,
    then checks whether CI widths are wider for turbulent windows.

    Primary gate: turb/calm width ratio > 1.15 (same as v2 S3).
    Secondary: growing uncertainty (monotonic variance across horizons).

    Note: MAE reduction vs unconditional baseline is not computed here
    since baselines don't have a "zero-history" generation mode. The
    turb/calm regime differentiation is the primary test.

    Args:
        cond_samples: (N, K, T, 5, 5) generated IV surfaces in [0, 1]
        ground_truth: (N, T, 5, 5) ground truth IV surfaces in [0, 1]
        history: (N, H, 5, 5) history IV surfaces in [0, 1]
    """
    print("\n" + "=" * 60)
    print("TEST SUITE 3: CONDITIONALITY (from pre-generated samples)")
    print("=" * 60)

    N = cond_samples.shape[0]

    # Regime classification from history vol-of-vol
    mean_iv = history.mean(axis=(2, 3))  # (N, H)
    daily_ch = np.diff(mean_iv, axis=1)  # (N, H-1)
    vol_of_vol = daily_ch.std(axis=1)  # (N,)

    q20 = np.quantile(vol_of_vol, 0.20)
    q80 = np.quantile(vol_of_vol, 0.80)
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80
    n_calm = int(calm_mask.sum())
    n_turb = int(turb_mask.sum())

    print(f"  Windows: {N} total, {n_calm} calm (Q20), {n_turb} turb (Q80)")
    print(f"  Vol-of-vol thresholds: Q20={q20:.5f}, Q80={q80:.5f}")

    # 90% CI width per window
    lower = np.quantile(cond_samples, 0.05, axis=1)  # (N, T, 5, 5)
    upper = np.quantile(cond_samples, 0.95, axis=1)  # (N, T, 5, 5)
    ci_width = (upper - lower).mean(axis=(1, 2, 3))  # (N,) avg over T, cells

    # Turb/calm width ratio
    calm_width = float(ci_width[calm_mask].mean()) if n_calm > 0 else 0.0
    turb_width = float(ci_width[turb_mask].mean()) if n_turb > 0 else 0.0
    turb_calm_ratio = turb_width / calm_width if calm_width > 0 else 1.0
    turb_calm_pass = turb_calm_ratio > 1.15

    print(f"  Turb/Calm width ratio: {turb_calm_ratio:.3f} "
          f"(target >1.15) {'PASS' if turb_calm_pass else 'FAIL'}")
    print(f"    Calm avg width: {calm_width:.4f}")
    print(f"    Turb avg width: {turb_width:.4f}")

    # Per-cell turb/calm width ratio
    per_cell_width = (upper - lower).mean(axis=1)  # (N, 5, 5)
    calm_cell_width = per_cell_width[calm_mask].mean(axis=0) if n_calm > 0 else np.ones((5, 5))
    turb_cell_width = per_cell_width[turb_mask].mean(axis=0) if n_turb > 0 else np.ones((5, 5))
    cell_turb_calm = turb_cell_width / np.maximum(calm_cell_width, 1e-8)
    worst_cell_wr = float(cell_turb_calm.min())
    worst_cell_wr_pass = worst_cell_wr > 1.0  # at minimum, turb should be wider than calm

    print(f"  Worst cell turb/calm ratio: {worst_cell_wr:.3f} "
          f"(target >1.0) {'PASS' if worst_cell_wr_pass else 'FAIL'}")

    # Growing uncertainty (monotonic variance across horizons)
    horizons = [1, 10, 20, 30]
    avg_var = {}
    for h in horizons:
        if h <= cond_samples.shape[2]:
            var_h = cond_samples[:, :, h - 1].var(axis=1).mean()
            avg_var[h] = float(var_h)

    monotonic = True
    hkeys = sorted(avg_var.keys())
    for i in range(len(hkeys) - 1):
        if avg_var[hkeys[i]] >= avg_var[hkeys[i + 1]]:
            monotonic = False
            break

    print(f"  Growing uncertainty (monotonic): {'PASS' if monotonic else 'FAIL'} (informational)")
    for h in hkeys:
        print(f"    Var(h={h:2d}): {avg_var[h]:.6f}")

    overall_pass = turb_calm_pass and worst_cell_wr_pass

    return {
        "turb_calm_ratio": float(turb_calm_ratio),
        "turb_calm_pass": turb_calm_pass,
        "calm_avg_width": calm_width,
        "turb_avg_width": turb_width,
        "n_calm": n_calm,
        "n_turb": n_turb,
        "worst_cell_turb_calm_ratio": worst_cell_wr,
        "worst_cell_wr_pass": worst_cell_wr_pass,
        "growing_uncertainty_monotonic": monotonic,
        "per_horizon_var": avg_var,
        "note": "MAE vs unconditional not computed (baselines lack zero-history mode)",
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def compute_factor_crps(samples, ground_truth, horizons=(1, 7, 14, 30)):
    """Per-factor CRPS on daily changes.

    Args:
        samples: (N, K, T, D) generated daily changes
        ground_truth: (N, T, D) actual daily changes
    Returns:
        dict with per_factor, per_horizon, overall
    """
    N, K, T, D = samples.shape
    results = {}

    for h in horizons:
        if h > T:
            continue
        t = h - 1
        s = samples[:, :, t]  # (N, K, D)
        y = ground_truth[:, t]  # (N, D)

        mae_term = np.abs(s - y[:, None, :]).mean(axis=1)  # (N, D)

        # E|X - X'| via subsampling
        n_pairs = min(500, K * (K - 1) // 2)
        rng = np.random.RandomState(42)
        spread = np.zeros((N, D))
        for _ in range(n_pairs):
            i, j = rng.choice(K, size=2, replace=False)
            spread += np.abs(s[:, i] - s[:, j])
        spread /= n_pairs

        crps = mae_term - 0.5 * spread  # (N, D)
        results[str(h)] = crps.mean(axis=0)  # (D,)

    # Per-factor average across horizons
    per_factor = np.mean(list(results.values()), axis=0)  # (D,)

    return {
        "per_horizon": {h: vals.tolist() for h, vals in results.items()},
        "per_factor": per_factor.tolist(),
        "overall": float(per_factor.mean()),
        "iv_crps": float(per_factor[:25].mean()),
        "factor_crps": float(per_factor[25:].mean()),
    }


def compute_energy_score_standardized(
    samples, ground_truth, train_mean, train_std, horizons=(1, 7, 14, 30)
):
    """Energy Score on z-scored 38-d coordinates.

    Args:
        samples: (N, K, T, D) raw daily changes
        ground_truth: (N, T, D) raw daily changes
        train_mean, train_std: (D,) for z-scoring
    """
    N, K, T, D = samples.shape

    # Standardize
    s_z = (samples - train_mean) / train_std
    y_z = (ground_truth - train_mean) / train_std

    es_per_horizon = {}
    for h in horizons:
        if h > T:
            continue
        t = h - 1
        s = s_z[:, :, t]  # (N, K, D)
        y = y_z[:, t]  # (N, D)

        # E||X - y||_2
        mae = np.linalg.norm(s - y[:, None], axis=2).mean(axis=1)  # (N,)

        # E||X - X'||_2 subsampled
        n_pairs = min(500, K * (K - 1) // 2)
        rng = np.random.RandomState(42)
        spread_sum = np.zeros(N)
        for _ in range(n_pairs):
            i, j = rng.choice(K, size=2, replace=False)
            spread_sum += np.linalg.norm(s[:, i] - s[:, j], axis=1)
        spread = spread_sum / n_pairs

        es = mae - 0.5 * spread
        es_per_horizon[str(h)] = float(es.mean())

    overall = float(np.mean(list(es_per_horizon.values())))
    return {"per_horizon": es_per_horizon, "overall": overall}


def compute_variogram_score(samples, ground_truth, train_std, p=0.5):
    """Variogram Score — dependence-sensitive complement to Energy Score.

    VS = mean_{i<j} (E|X_i - X_j|^p - |y_i - y_j|^p)^2

    Computed on standardized coordinates at horizon 30 (full path).
    """
    N, K, T, D = samples.shape
    t = min(T, 30) - 1

    s = (samples[:, :, t] - samples.mean(axis=2, keepdims=True)[:, :, 0:1].squeeze(2)) / train_std
    y = (ground_truth[:, t]) / train_std

    # Compute pairwise |X_i - X_j|^p for samples (averaged over K)
    sample_variogram = np.zeros((N, D, D))
    for k in range(K):
        diff = np.abs(s[:, k, :, None] - s[:, k, None, :])  # (N, D, D)
        sample_variogram += diff ** p
    sample_variogram /= K  # E|X_i - X_j|^p

    # Ground truth pairwise
    gt_variogram = np.abs(y[:, :, None] - y[:, None, :]) ** p  # (N, D, D)

    # VS = mean over (i, j) pairs of (E|X_i-X_j|^p - |y_i-y_j|^p)^2
    diff_sq = (sample_variogram - gt_variogram) ** 2  # (N, D, D)
    # Average over upper triangle (i < j) and over windows
    triu_mask = np.triu(np.ones((D, D), dtype=bool), k=1)
    vs = diff_sq[:, triu_mask].mean()

    return {"variogram_score": float(vs), "p": p, "horizon": t + 1}


def compute_correlation_frobenius(samples, ground_truth):
    """Cross-factor correlation matrix Frobenius error.

    Computes on factor dims only (25:38).
    """
    N, K, T, D = samples.shape

    # Ground truth factor correlation
    gt_factors = ground_truth[:, :, 25:].reshape(-1, 13)
    gt_corr = np.corrcoef(gt_factors.T)

    # Generated factor correlation (pool all samples and timesteps)
    gen_factors = samples[:, :, :, 25:].reshape(-1, 13)
    gen_corr = np.corrcoef(gen_factors.T)

    fro_error = np.linalg.norm(gen_corr - gt_corr, "fro")
    fro_gt = np.linalg.norm(gt_corr, "fro")

    return {
        "frobenius_error": float(fro_error),
        "frobenius_ratio": float(fro_error / fro_gt),
        "corr_score": float(1 - fro_error / fro_gt),
    }


def compute_factor_ks(samples, ground_truth, factor_columns):
    """Per-factor KS test on daily change distribution."""
    N, K, T, D = samples.shape

    results = {}
    n_pass = 0
    for d in range(25, D):
        gen_vals = samples[:, :, :, d].flatten()
        gt_vals = ground_truth[:, :, d].flatten()

        stat, pval = sp_stats.ks_2samp(gen_vals, gt_vals)
        passed = pval > 0.05
        if passed:
            n_pass += 1
        col_name = factor_columns[d - 25] if d - 25 < len(factor_columns) else f"dim_{d}"
        results[col_name] = {
            "statistic": float(stat), "p_value": float(pval), "pass": bool(passed),
        }

    return {
        "per_factor": results,
        "n_pass": n_pass,
        "n_total": D - 25,
        "pass_rate": float(n_pass / (D - 25)),
    }


def reconstruct_iv_surfaces(iv_changes, anchor_surfaces):
    """Reconstruct IV surfaces from daily changes + anchor.

    Args:
        iv_changes: (N, K, T, 25) IV daily changes
        anchor_surfaces: (N, 5, 5) last known surface at forecast origin
    Returns:
        surfaces: (N, K, T, 5, 5) in [0, 1]
    """
    N, K, T = iv_changes.shape[:3]
    anchor_flat = anchor_surfaces.reshape(N, 1, 1, 25)
    cum_changes = np.cumsum(iv_changes, axis=2)
    surfaces_flat = anchor_flat + cum_changes
    surfaces = surfaces_flat.reshape(N, K, T, 5, 5)
    return np.clip(surfaces, 0, 1)


# ---------------------------------------------------------------------------
# Baseline construction
# ---------------------------------------------------------------------------

def create_classical_baseline(name, data, train_changes):
    """Create a joint classical baseline from name."""
    if name == "random_walk":
        return JointRandomWalk(train_changes)
    elif name == "bootstrap":
        return JointBootstrap(train_changes)
    elif name == "historical_sim":
        return JointHistoricalSim(
            train_changes, data["surfaces"], data["factor_levels_13"],
        )
    elif name == "pca_var":
        return JointPCAVAR(train_changes)
    elif name == "garch_ccc":
        return JointGARCHCCC(train_changes)
    elif name == "filtered_hs":
        return JointFilteredHS(train_changes)
    else:
        raise ValueError(f"Unknown classical baseline: {name}")


def load_deep_baseline(name, device="cuda"):
    """Load a trained 38-d deep baseline."""
    from experiments.backfill.baselines.joint_deep_baselines import (
        load_joint_deepvar, load_joint_timegrad, load_joint_csdi,
    )
    base = Path("models/backfill/baselines")
    if name == "deepvar":
        return load_joint_deepvar(base / "deepvar_38d" / "best_model.pt", device)
    elif name == "timegrad":
        return load_joint_timegrad(base / "timegrad_38d" / "best_model.pt", device)
    elif name == "csdi":
        return load_joint_csdi(base / "csdi_38d" / "best_model.pt", device)
    else:
        raise ValueError(f"Unknown deep baseline: {name}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    name, model, test_windows, data,
    n_samples, max_batches, output_dir,
):
    """Run Benchmark A evaluation for one baseline."""
    print("\n" + "=" * 70)
    print(f"  EVALUATING: {name} (38-d joint)")
    print("=" * 70)

    # Select windows
    n_windows = min(len(test_windows["history_changes"]), max_batches * 16)
    hist_chg = test_windows["history_changes"][:n_windows]
    fut_chg = test_windows["future_changes"][:n_windows]
    anchors = test_windows["anchor_surfaces"][:n_windows]
    hist_surf = test_windows["history_surfaces"][:n_windows, 1:]  # exclude t=0
    hist_flev = test_windows["history_factor_levels"][:n_windows]

    print(f"  Windows: {n_windows}, Samples: {n_samples}")

    # Generate samples
    t0 = time.time()
    batch_size = 32
    all_samples = []
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch_hist = hist_chg[start:end]
        kwargs = {}
        if name == "historical_sim":
            kwargs["surfaces"] = hist_surf[start:end]
            kwargs["factor_levels"] = hist_flev[start:end]

        batch_out = model.sample_joint(batch_hist, n_samples=n_samples, **kwargs)
        all_samples.append(batch_out)

        if (start // batch_size + 1) % 5 == 0:
            print(f"  Batch {start // batch_size + 1}/{(n_windows + batch_size - 1) // batch_size}")

    samples = np.concatenate(all_samples, axis=0)  # (N, K, 30, 38)
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s, shape: {samples.shape}")

    # Split IV and factor components
    iv_changes = samples[:, :, :, :25]  # (N, K, 30, 25)
    factor_changes = samples[:, :, :, 25:]  # (N, K, 30, 13)

    # --- Panel A: IV evaluation ---
    print("\n  --- Panel A: IV Test Suites ---")
    iv_surfaces = reconstruct_iv_surfaces(iv_changes, anchors)  # (N, K, 30, 5, 5)

    # Ground truth IV surfaces
    gt_iv_changes = fut_chg[:, :, :25]  # (N, 30, 25)
    gt_iv_surfaces = reconstruct_iv_surfaces(
        gt_iv_changes[:, None], anchors
    ).squeeze(1)  # (N, 30, 5, 5)

    # Run IV test suites using the v2 functions (they expect numpy arrays)
    from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
        run_surface_validity_tests,
        run_ci_coverage_tests,
        run_time_series_tests,
        run_distributional_fidelity_tests,
        run_cross_cell_correlation_tests,
        run_regime_coverage_tests,
    )

    iv_surfaces_np = iv_surfaces.astype(np.float64)
    gt_iv_np = gt_iv_surfaces.astype(np.float64)
    hist_surf_01 = test_windows["history_surfaces"][:n_windows, 1:].astype(np.float64)

    suite_results = {}

    # S1: Surface Validity
    try:
        s1 = run_surface_validity_tests(iv_surfaces_np, gt_iv_np)
        suite_results["S1_surface_validity"] = s1
        print(f"    S1: {'PASS' if s1.get('overall_pass') else 'FAIL'}")
    except Exception as e:
        print(f"    S1: ERROR - {e}")

    # S2: CI Coverage
    try:
        s2 = run_ci_coverage_tests(iv_surfaces_np, gt_iv_np)
        suite_results["S2_ci_coverage"] = s2
        print(f"    S2: {'PASS' if s2.get('overall_pass') else 'FAIL'}")
    except Exception as e:
        print(f"    S2: ERROR - {e}")

    # S3: Conditionality (regime differentiation from pre-generated samples)
    try:
        s3 = run_conditionality_from_samples(iv_surfaces_np, gt_iv_np, hist_surf_01)
        suite_results["S3_conditionality"] = s3
        print(f"    S3: {'PASS' if s3.get('overall_pass') else 'FAIL'}")
    except Exception as e:
        print(f"    S3: ERROR - {e}")

    # S4: Time Series
    try:
        s4 = run_time_series_tests(iv_surfaces_np, gt_iv_np)
        suite_results["S4_time_series"] = s4
        print(f"    S4: {'PASS' if s4.get('overall_pass') else 'FAIL'}")
    except Exception as e:
        print(f"    S4: ERROR - {e}")

    # S7: Regime Coverage (three-layer)
    try:
        s7 = run_regime_coverage_tests(iv_surfaces_np, gt_iv_np, hist_surf_01)
        suite_results["S7_regime_coverage"] = s7
        print(f"    S7: {'PASS' if s7.get('overall_pass') else 'FAIL'}")
    except Exception as e:
        print(f"    S7: ERROR - {e}")

    # S8: Distributional (needs history)
    try:
        s8 = run_distributional_fidelity_tests(iv_surfaces_np, gt_iv_np, hist_surf_01)
        suite_results["S8_distributional"] = s8
        print(f"    S8: {'PASS' if s8.get('overall_pass') else 'FAIL'}")
    except Exception as e:
        print(f"    S8: ERROR - {e}")

    # S9: Cross-Cell Correlation
    try:
        s9 = run_cross_cell_correlation_tests(iv_surfaces_np, gt_iv_np)
        suite_results["S9_cross_cell"] = s9
        print(f"    S9: {'PASS' if s9.get('overall_pass') else 'FAIL'}")
    except Exception as e:
        print(f"    S9: ERROR - {e}")

    # Count passes
    iv_passes = sum(
        1 for v in suite_results.values() if v.get("overall_pass", False)
    )
    print(f"\n  Panel A IV result: {iv_passes}/{len(suite_results)} suites passed")

    # --- Panel B: Factor evaluation ---
    print("\n  --- Panel B: Factor Metrics ---")

    factor_columns = data.get("factor_return_columns", [f"f{i}" for i in range(13)])

    # Per-dim CRPS
    crps = compute_factor_crps(samples, fut_chg, horizons=(1, 7, 14, 30))
    print(f"    CRPS overall: {crps['overall']:.6f}")
    print(f"    CRPS IV: {crps['iv_crps']:.6f}, Factor: {crps['factor_crps']:.6f}")

    # Standardized Energy Score
    es = compute_energy_score_standardized(
        samples, fut_chg, data["train_mean_38"], data["train_std_38"],
    )
    print(f"    Energy Score (std): {es['overall']:.4f}")

    # Variogram Score
    vs = compute_variogram_score(samples, fut_chg, data["train_std_38"])
    print(f"    Variogram Score: {vs['variogram_score']:.6f}")

    # Correlation Frobenius
    corr = compute_correlation_frobenius(samples, fut_chg)
    print(f"    Correlation score: {corr['corr_score']:.4f} (Frobenius ratio: {corr['frobenius_ratio']:.4f})")

    # Factor KS
    ks = compute_factor_ks(samples, fut_chg, factor_columns)
    print(f"    Factor KS: {ks['n_pass']}/{ks['n_total']} pass")

    # --- Save results ---
    results = {
        "baseline_name": name,
        "mode": "joint_38d",
        "generation_time_s": gen_time,
        "n_windows": n_windows,
        "n_samples": n_samples,
        "eval_config": {
            "seed": int(np.random.get_state()[1][0]),
            "train_end_surface": 4040,
            "train_end_changes": 4039,
            "test_start_surface": 4540,
            "test_start_changes": 4539,
            "history_len": 30,
            "future_len": 30,
            "column_order": data["column_names"],
            "normalization_source": "train_changes[:4039]",
            "iv_clipping": "[0, 1] after cumsum+anchor reconstruction",
            "factor_nan_fill": "returns→0.0, levels→forward-fill",
            "suites_run": list(suite_results.keys()),
        },
        "panel_a_iv_suites": suite_results,
        "panel_a_iv_passes": iv_passes,
        "panel_a_iv_total": len(suite_results),
        "panel_b_crps": crps,
        "panel_b_energy_score": es,
        "panel_b_variogram_score": vs,
        "panel_b_correlation": corr,
        "panel_b_factor_ks": ks,
    }

    out_path = output_dir / name
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "results_38d.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}/results_38d.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines on 38-d joint benchmark")
    parser.add_argument("--baselines", nargs="+", default=CLASSICAL_BASELINES,
                        choices=ALL_BASELINES)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/baselines_38d")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)

    # Load data
    print("Loading aligned 38-d data...")
    data = load_aligned_38d_data()
    train_changes = data["joint_changes_38"][:4039]

    print("Creating test windows...")
    test_windows = make_windows(data, start_idx=4540)
    print(f"  {len(test_windows['history_changes'])} test windows")

    all_results = {}

    for name in args.baselines:
        if name in CLASSICAL_BASELINES:
            print(f"\nCreating {name}...")
            model = create_classical_baseline(name, data, train_changes)
        elif name in DEEP_BASELINES:
            print(f"\nLoading {name}...")
            model = load_deep_baseline(name, args.device)
        else:
            print(f"Skipping unknown baseline: {name}")
            continue

        results = evaluate_baseline(
            name, model, test_windows, data,
            args.n_samples, args.max_batches, output_dir,
        )
        all_results[name] = results

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: 38-d Joint Benchmark")
    print("=" * 70)
    print(f"{'Baseline':20s} {'IV Pass':>8s} {'CRPS':>8s} {'ES(std)':>8s} {'VS':>10s} {'CorrF':>8s} {'KS':>6s}")
    print("-" * 70)
    for name, r in all_results.items():
        iv_pass = f"{r['panel_a_iv_passes']}/{len(r['panel_a_iv_suites'])}"
        crps = f"{r['panel_b_crps']['overall']:.5f}"
        es = f"{r['panel_b_energy_score']['overall']:.4f}"
        vs_val = f"{r['panel_b_variogram_score']['variogram_score']:.6f}"
        corr = f"{r['panel_b_correlation']['corr_score']:.3f}"
        ks = f"{r['panel_b_factor_ks']['n_pass']}/{r['panel_b_factor_ks']['n_total']}"
        print(f"{name:20s} {iv_pass:>8s} {crps:>8s} {es:>8s} {vs_val:>10s} {corr:>8s} {ks:>6s}")


if __name__ == "__main__":
    main()
