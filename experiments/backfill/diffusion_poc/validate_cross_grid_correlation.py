#!/usr/bin/env python
"""
Validate whether cross-grid correlation under-estimation is a real problem.

Tests:
1. Portfolio-Level CI Coverage - does under-correlation affect aggregate forecasts?
2. Extreme Co-movement Events - does model capture joint tail events?
3. ATM-Wing Correlation - do ATM and wings move together correctly?
4. History vs Future Correlation - is generated correlation just using prior?

Usage:
    python experiments/backfill/diffusion_poc/validate_cross_grid_correlation.py \
        --model_path models/backfill/ddpm_poc/checkpoint_epoch_50.pt \
        --max_batches 20 --n_samples 50
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.simple_denoiser import ConditionalDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config


# Grid convention: rows=tenors (1M,2M,3M,6M,1Y), cols=strikes (90%,95%,ATM,105%,110%)
ATM_IDX = (slice(None), 2)  # All tenors, ATM strike
OTM_PUT_IDX = (slice(None), 0)  # All tenors, 90% strike
OTM_CALL_IDX = (slice(None), 4)  # All tenors, 110% strike


def compute_correlation_matrix(surfaces: np.ndarray) -> np.ndarray:
    """Compute 25x25 correlation matrix from (T, 5, 5) surfaces."""
    T = surfaces.shape[0]
    flat = surfaces.reshape(T, 25)
    return np.corrcoef(flat.T)


def test_portfolio_ci_coverage(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    device: str,
    n_samples: int = 50,
    max_batches: int = 20,
) -> dict:
    """
    Test 1: Portfolio-Level CI Coverage

    Compare CI coverage at:
    - Per-grid-point level (25 separate coverages)
    - Portfolio level (sum of all grid points)

    If under-correlation is a problem, portfolio CI coverage will be worse
    because the model underestimates how grid points move together.
    """
    print("\n" + "="*60)
    print("Test 1: Portfolio-Level CI Coverage")
    print("="*60)

    per_point_in_ci = []  # Per grid point
    portfolio_in_ci = []  # Sum of all points

    ci_level = 0.90

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Portfolio CI")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            # Generate samples
            samples = model.sample(
                history, n_samples=n_samples,
                sampler='ddim', n_inference_steps=20
            )  # (B, n_samples, T_fut, 5, 5)

            # Denormalize
            future_gt = denormalize_iv(future_gt).cpu().numpy()
            samples = samples.cpu().numpy()

            B = samples.shape[0]
            for b in range(B):
                gt = future_gt[b]  # (T_fut, 5, 5)
                gen = samples[b]  # (n_samples, T_fut, 5, 5)

                # Per-point CI coverage (average across all 25 grid points)
                lower = np.percentile(gen, (1 - ci_level) / 2 * 100, axis=0)
                upper = np.percentile(gen, (1 + ci_level) / 2 * 100, axis=0)
                in_ci = (gt >= lower) & (gt <= upper)
                per_point_in_ci.append(in_ci.mean())

                # Portfolio CI coverage (sum across grid)
                gt_portfolio = gt.sum(axis=(1, 2))  # (T_fut,)
                gen_portfolio = gen.sum(axis=(2, 3))  # (n_samples, T_fut)

                lower_port = np.percentile(gen_portfolio, (1 - ci_level) / 2 * 100, axis=0)
                upper_port = np.percentile(gen_portfolio, (1 + ci_level) / 2 * 100, axis=0)
                in_ci_port = (gt_portfolio >= lower_port) & (gt_portfolio <= upper_port)
                portfolio_in_ci.append(in_ci_port.mean())

    per_point_coverage = np.mean(per_point_in_ci)
    portfolio_coverage = np.mean(portfolio_in_ci)

    print(f"\n  Per-Grid-Point 90% CI Coverage: {per_point_coverage:.1%}")
    print(f"  Portfolio-Level 90% CI Coverage: {portfolio_coverage:.1%}")
    print(f"  Difference: {(portfolio_coverage - per_point_coverage):.1%}")

    if portfolio_coverage < per_point_coverage - 0.10:
        print("  WARNING: Portfolio coverage significantly worse - correlation matters!")
    else:
        print("  OK: Portfolio coverage similar to per-point - correlation may not matter")

    return {
        'per_point_coverage': float(per_point_coverage),
        'portfolio_coverage': float(portfolio_coverage),
        'difference': float(portfolio_coverage - per_point_coverage),
    }


def test_extreme_comovement(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    device: str,
    n_samples: int = 50,
    max_batches: int = 20,
) -> dict:
    """
    Test 2: Extreme Co-movement Events

    Find days where GT has extreme moves in multiple grid points.
    Check if generated samples produce similar joint extremes.

    Under-correlation → underestimate probability of joint tail events.
    """
    print("\n" + "="*60)
    print("Test 2: Extreme Co-movement Events")
    print("="*60)

    # Collect all GT and generated day-changes
    gt_changes_all = []
    gen_changes_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Extreme events")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            samples = model.sample(
                history, n_samples=n_samples,
                sampler='ddim', n_inference_steps=20
            )

            future_gt = denormalize_iv(future_gt).cpu().numpy()
            samples = samples.cpu().numpy()

            B = samples.shape[0]
            for b in range(B):
                gt = future_gt[b]  # (T_fut, 5, 5)
                gen = samples[b]  # (n_samples, T_fut, 5, 5)

                # Day-to-day changes
                gt_changes = np.diff(gt, axis=0)  # (T_fut-1, 5, 5)
                gen_changes = np.diff(gen, axis=1)  # (n_samples, T_fut-1, 5, 5)

                gt_changes_all.append(gt_changes)
                gen_changes_all.append(gen_changes)

    gt_changes_all = np.concatenate(gt_changes_all, axis=0)  # (N, 5, 5)
    gen_changes_all = np.concatenate(gen_changes_all, axis=1)  # (n_samples, N, 5, 5)

    # Flatten to (N, 25) for GT and (n_samples, N, 25) for generated
    N = gt_changes_all.shape[0]
    gt_flat = gt_changes_all.reshape(N, 25)
    gen_flat = gen_changes_all.reshape(gen_changes_all.shape[0], N, 25)

    # Define "extreme" as |change| > 2*std for that grid point
    gt_std = gt_flat.std(axis=0)  # (25,)
    threshold = 2.0 * gt_std

    # Count days with joint extremes (>= 10 grid points moving extremely)
    gt_extreme = np.abs(gt_flat) > threshold  # (N, 25)
    gt_joint_extreme_days = (gt_extreme.sum(axis=1) >= 10)  # (N,)
    gt_joint_extreme_rate = gt_joint_extreme_days.mean()

    # For generated: check if samples produce similar joint extreme rate
    gen_extreme = np.abs(gen_flat) > threshold  # (n_samples, N, 25)
    gen_joint_extreme_days = (gen_extreme.sum(axis=2) >= 10)  # (n_samples, N)
    gen_joint_extreme_rate = gen_joint_extreme_days.mean()

    print(f"\n  GT joint extreme rate (>=10 points moving >2σ): {gt_joint_extreme_rate:.2%}")
    print(f"  Generated joint extreme rate: {gen_joint_extreme_rate:.2%}")
    print(f"  Ratio (Gen/GT): {gen_joint_extreme_rate / max(gt_joint_extreme_rate, 1e-8):.2f}x")

    if gen_joint_extreme_rate < gt_joint_extreme_rate * 0.5:
        print("  WARNING: Model underestimates joint extreme events by >50%!")
    else:
        print("  OK: Joint extreme rate reasonably captured")

    return {
        'gt_joint_extreme_rate': float(gt_joint_extreme_rate),
        'gen_joint_extreme_rate': float(gen_joint_extreme_rate),
        'ratio': float(gen_joint_extreme_rate / max(gt_joint_extreme_rate, 1e-8)),
    }


def test_atm_wing_correlation(
    model: ConditionalDDPM,
    test_loader: DataLoader,
    device: str,
    n_samples: int = 50,
    max_batches: int = 20,
) -> dict:
    """
    Test 3: ATM-Wing Correlation

    Check the most important correlations:
    - ATM vs OTM Put (left wing)
    - ATM vs OTM Call (right wing)

    These should be strongly positive (when ATM moves, wings move too).
    """
    print("\n" + "="*60)
    print("Test 3: ATM-Wing Correlation")
    print("="*60)

    gt_corrs_atm_put = []
    gt_corrs_atm_call = []
    gen_corrs_atm_put = []
    gen_corrs_atm_call = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="ATM-Wing corr")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            samples = model.sample(
                history, n_samples=n_samples,
                sampler='ddim', n_inference_steps=20
            )

            future_gt = denormalize_iv(future_gt).cpu().numpy()
            samples = samples.cpu().numpy()

            B = samples.shape[0]
            for b in range(B):
                gt = future_gt[b]  # (T_fut, 5, 5)
                gen = samples[b]  # (n_samples, T_fut, 5, 5)

                # For 3M tenor (index 2) - most liquid
                tenor_idx = 2

                # GT correlations
                gt_atm = gt[:, tenor_idx, 2]  # ATM
                gt_put = gt[:, tenor_idx, 0]  # OTM put
                gt_call = gt[:, tenor_idx, 4]  # OTM call

                if gt_atm.std() > 1e-8 and gt_put.std() > 1e-8:
                    gt_corrs_atm_put.append(np.corrcoef(gt_atm, gt_put)[0, 1])
                if gt_atm.std() > 1e-8 and gt_call.std() > 1e-8:
                    gt_corrs_atm_call.append(np.corrcoef(gt_atm, gt_call)[0, 1])

                # Generated correlations (average across samples)
                sample_corrs_put = []
                sample_corrs_call = []
                for s in range(n_samples):
                    gen_atm = gen[s, :, tenor_idx, 2]
                    gen_put = gen[s, :, tenor_idx, 0]
                    gen_call = gen[s, :, tenor_idx, 4]

                    if gen_atm.std() > 1e-8 and gen_put.std() > 1e-8:
                        sample_corrs_put.append(np.corrcoef(gen_atm, gen_put)[0, 1])
                    if gen_atm.std() > 1e-8 and gen_call.std() > 1e-8:
                        sample_corrs_call.append(np.corrcoef(gen_atm, gen_call)[0, 1])

                if sample_corrs_put:
                    gen_corrs_atm_put.append(np.mean(sample_corrs_put))
                if sample_corrs_call:
                    gen_corrs_atm_call.append(np.mean(sample_corrs_call))

    gt_atm_put = np.mean(gt_corrs_atm_put) if gt_corrs_atm_put else 0
    gt_atm_call = np.mean(gt_corrs_atm_call) if gt_corrs_atm_call else 0
    gen_atm_put = np.mean(gen_corrs_atm_put) if gen_corrs_atm_put else 0
    gen_atm_call = np.mean(gen_corrs_atm_call) if gen_corrs_atm_call else 0

    print(f"\n  3M Tenor Correlations:")
    print(f"  ATM-OTM_Put:  GT={gt_atm_put:.3f}, Gen={gen_atm_put:.3f}, Diff={gen_atm_put-gt_atm_put:+.3f}")
    print(f"  ATM-OTM_Call: GT={gt_atm_call:.3f}, Gen={gen_atm_call:.3f}, Diff={gen_atm_call-gt_atm_call:+.3f}")

    avg_diff = (abs(gen_atm_put - gt_atm_put) + abs(gen_atm_call - gt_atm_call)) / 2
    if avg_diff > 0.3:
        print(f"  WARNING: ATM-Wing correlation differs by >{avg_diff:.2f} on average!")
    else:
        print(f"  OK: ATM-Wing correlations reasonably close (avg diff={avg_diff:.2f})")

    return {
        'gt_atm_put': float(gt_atm_put),
        'gt_atm_call': float(gt_atm_call),
        'gen_atm_put': float(gen_atm_put),
        'gen_atm_call': float(gen_atm_call),
    }


def test_history_vs_future_correlation(
    test_loader: DataLoader,
    device: str,
    max_batches: int = 20,
) -> dict:
    """
    Test 4: History vs Future Correlation

    Compare correlation matrices:
    - History correlation (what model sees as input)
    - Future GT correlation (what we want to predict)

    If they're very different, the model needs to learn the transition.
    If they're similar, using history correlation as prior is reasonable.
    """
    print("\n" + "="*60)
    print("Test 4: History vs Future Correlation Structure")
    print("="*60)

    history_corrs = []
    future_corrs = []

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="History vs Future")):
        if batch_idx >= max_batches:
            break

        history = batch["history"].numpy()  # (B, T_hist, 5, 5)
        future = batch["future"].numpy()  # (B, T_fut, 5, 5)

        # Denormalize (dataset stores in [-1, 1])
        history = (history + 1) / 2  # to [0, 1]
        future = (future + 1) / 2

        B = history.shape[0]
        for b in range(B):
            hist_corr = compute_correlation_matrix(history[b])
            fut_corr = compute_correlation_matrix(future[b])

            if not np.isnan(hist_corr).any() and not np.isnan(fut_corr).any():
                history_corrs.append(hist_corr)
                future_corrs.append(fut_corr)

    history_corr_mean = np.mean(history_corrs, axis=0)
    future_corr_mean = np.mean(future_corrs, axis=0)

    # Frobenius distance between history and future correlation
    frob_hist_fut = np.linalg.norm(history_corr_mean - future_corr_mean, 'fro')

    # Mean absolute difference
    mean_abs_diff = np.abs(history_corr_mean - future_corr_mean).mean()

    print(f"\n  Frobenius(History_corr, Future_corr): {frob_hist_fut:.2f}")
    print(f"  Mean absolute correlation difference: {mean_abs_diff:.3f}")

    if frob_hist_fut < 5.0:
        print("  History and Future correlations are SIMILAR")
        print("  → Model could reasonably use history correlation as prior")
    else:
        print("  History and Future correlations are DIFFERENT")
        print("  → Model needs to learn correlation transition")

    return {
        'frobenius_hist_fut': float(frob_hist_fut),
        'mean_abs_diff': float(mean_abs_diff),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate cross-grid correlation")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/ddpm_poc/checkpoint_epoch_50.pt")
    parser.add_argument("--data_path", type=str,
                        default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print("="*60)
    print("Cross-Grid Correlation Validation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Samples per history: {args.n_samples}")
    print(f"Max batches: {args.max_batches}")

    # Load data
    config = get_default_config()
    data = np.load(args.data_path)
    surfaces = data["surface"]

    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = ConditionalDDPM(checkpoint["config"], scheduler_config={"device": device})
    # Use strict=False to ignore removed cross-attention keys from old checkpoints
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Run all tests
    results = {}

    results['test1_portfolio'] = test_portfolio_ci_coverage(
        model, test_loader, device, args.n_samples, args.max_batches
    )

    results['test2_extreme'] = test_extreme_comovement(
        model, test_loader, device, args.n_samples, args.max_batches
    )

    results['test3_atm_wing'] = test_atm_wing_correlation(
        model, test_loader, device, args.n_samples, args.max_batches
    )

    results['test4_history'] = test_history_vs_future_correlation(
        test_loader, device, args.max_batches
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    issues = []

    # Test 1: Portfolio coverage
    port_diff = results['test1_portfolio']['difference']
    if port_diff < -0.10:
        issues.append(f"Portfolio CI coverage {port_diff:+.1%} worse than per-point")

    # Test 2: Joint extremes
    extreme_ratio = results['test2_extreme']['ratio']
    if extreme_ratio < 0.5:
        issues.append(f"Joint extreme events underestimated by {(1-extreme_ratio):.0%}")

    # Test 3: ATM-Wing correlation
    atm_put_diff = abs(results['test3_atm_wing']['gen_atm_put'] - results['test3_atm_wing']['gt_atm_put'])
    atm_call_diff = abs(results['test3_atm_wing']['gen_atm_call'] - results['test3_atm_wing']['gt_atm_call'])
    if (atm_put_diff + atm_call_diff) / 2 > 0.3:
        issues.append(f"ATM-Wing correlation off by {(atm_put_diff + atm_call_diff) / 2:.2f}")

    if issues:
        print("\nCross-grid correlation IS a problem:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendation: Consider U-Net or other architecture changes")
    else:
        print("\nCross-grid correlation is NOT a significant problem for forecasting.")
        print("The model achieves good CI coverage at both per-point and portfolio levels.")
        print("\nRecommendation: Focus on other improvements instead of U-Net")

    return results


if __name__ == "__main__":
    main()
