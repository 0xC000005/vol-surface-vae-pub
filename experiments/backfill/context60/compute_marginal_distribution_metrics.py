"""
Marginal Distribution Metrics for VAE Scenario Generation

Production validation: Does stacking all model predictions at horizon H
match stacking all ground truth observations at horizon H?

Metrics:
---------
1. Wasserstein-1 Distance: Earth mover distance (in data units)
2. KS Statistic: Max CDF difference (with p-value)
3. Quantile RMSE: How well percentiles [5, 25, 50, 75, 95] match
4. Moment Matching: Mean, Variance, Skewness, Kurtosis differences

Usage:
------
python experiments/backfill/context60/compute_marginal_distribution_metrics.py \
    --model_path models/backfill/context60_experiment/checkpoints/model.pt \
    --horizons 1 7 14 30 60 90 \
    --n_samples 100 \
    --output_dir results/marginal_metrics

Author: Research
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class MarginalMetrics:
    """Container for marginal distribution metrics at a single horizon."""
    horizon: int

    # Primary metrics
    wasserstein: float
    ks_statistic: float
    ks_pvalue: float
    quantile_rmse: float

    # Moment metrics
    mean_gt: float
    mean_model: float
    mean_bias: float

    var_gt: float
    var_model: float
    var_ratio: float

    skew_gt: float
    skew_model: float
    skew_diff: float

    kurt_gt: float
    kurt_model: float
    kurt_diff: float

    # Sample sizes
    n_gt: int
    n_model: int

    def passes_thresholds(self,
                          wasserstein_thresh: float = 0.01,
                          ks_pvalue_thresh: float = 0.05,
                          mean_bias_thresh: float = 0.005,
                          var_ratio_range: Tuple[float, float] = (0.8, 1.2)) -> Dict[str, bool]:
        """Check if metrics pass production thresholds."""
        return {
            'wasserstein': self.wasserstein < wasserstein_thresh,
            'ks_test': self.ks_pvalue > ks_pvalue_thresh,
            'mean_bias': abs(self.mean_bias) < mean_bias_thresh,
            'var_ratio': var_ratio_range[0] < self.var_ratio < var_ratio_range[1],
        }

    def to_dict(self) -> dict:
        return {
            'horizon': self.horizon,
            'wasserstein': self.wasserstein,
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'quantile_rmse': self.quantile_rmse,
            'mean_gt': self.mean_gt,
            'mean_model': self.mean_model,
            'mean_bias': self.mean_bias,
            'var_gt': self.var_gt,
            'var_model': self.var_model,
            'var_ratio': self.var_ratio,
            'skew_gt': self.skew_gt,
            'skew_model': self.skew_model,
            'skew_diff': self.skew_diff,
            'kurt_gt': self.kurt_gt,
            'kurt_model': self.kurt_model,
            'kurt_diff': self.kurt_diff,
            'n_gt': self.n_gt,
            'n_model': self.n_model,
        }


def compute_marginal_metrics(gt_samples: np.ndarray,
                              model_samples: np.ndarray,
                              horizon: int) -> MarginalMetrics:
    """
    Compute all marginal distribution metrics.

    Args:
        gt_samples: Ground truth samples (N,) - e.g., all actual H-day changes
        model_samples: Model samples (M,) - e.g., all generated H-day changes
        horizon: Forecast horizon in days

    Returns:
        MarginalMetrics object with all computed metrics
    """
    # Flatten to 1D
    gt = gt_samples.flatten()
    model = model_samples.flatten()

    # Primary metrics
    wasserstein = wasserstein_distance(gt, model)
    ks_stat, ks_pval = ks_2samp(gt, model)

    # Quantile comparison
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    gt_quantiles = np.percentile(gt, [q*100 for q in quantiles])
    model_quantiles = np.percentile(model, [q*100 for q in quantiles])
    quantile_rmse = np.sqrt(np.mean((gt_quantiles - model_quantiles)**2))

    # Moments
    mean_gt = np.mean(gt)
    mean_model = np.mean(model)

    var_gt = np.var(gt)
    var_model = np.var(model)

    skew_gt = stats.skew(gt)
    skew_model = stats.skew(model)

    kurt_gt = stats.kurtosis(gt)
    kurt_model = stats.kurtosis(model)

    return MarginalMetrics(
        horizon=horizon,
        wasserstein=wasserstein,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        quantile_rmse=quantile_rmse,
        mean_gt=mean_gt,
        mean_model=mean_model,
        mean_bias=mean_model - mean_gt,
        var_gt=var_gt,
        var_model=var_model,
        var_ratio=var_model / var_gt if var_gt > 0 else float('inf'),
        skew_gt=skew_gt,
        skew_model=skew_model,
        skew_diff=skew_model - skew_gt,
        kurt_gt=kurt_gt,
        kurt_model=kurt_model,
        kurt_diff=kurt_model - kurt_gt,
        n_gt=len(gt),
        n_model=len(model),
    )


def extract_ground_truth_changes(data: dict,
                                  indices: np.ndarray,
                                  context_len: int,
                                  horizon: int,
                                  grid_point: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Extract ground truth H-day changes for a specific grid point.

    Args:
        data: Data dict with 'surface' key
        indices: Starting indices for sequences
        context_len: Context length
        horizon: Forecast horizon
        grid_point: (row, col) of grid point to analyze (default: ATM 6M)

    Returns:
        changes: (N,) array of H-day changes
    """
    surfaces = data['surface']
    r, c = grid_point

    changes = []
    for start_idx in indices:
        context_end = start_idx
        target_end = start_idx + horizon

        if target_end >= len(surfaces):
            continue

        # Change from end of context to end of horizon
        start_val = surfaces[context_end - 1, r, c]  # Last context day
        end_val = surfaces[target_end - 1, r, c]      # Last target day
        changes.append(end_val - start_val)

    return np.array(changes)


def generate_model_samples(model,
                           data: dict,
                           indices: np.ndarray,
                           context_len: int,
                           horizon: int,
                           n_samples: int,
                           grid_point: Tuple[int, int] = (2, 2),
                           device: str = 'cuda') -> np.ndarray:
    """
    Generate model samples at specified horizon.

    Args:
        model: Trained VAE model
        data: Data dict
        indices: Starting indices
        context_len: Context length
        horizon: Forecast horizon
        n_samples: Number of samples per sequence
        grid_point: Grid point to analyze
        device: Device to use

    Returns:
        samples: (N * n_samples,) array of generated H-day changes
    """
    model.eval()
    surfaces = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    r, c = grid_point
    all_changes = []

    with torch.no_grad():
        for start_idx in indices:
            context_start = start_idx - context_len
            context_end = start_idx

            if context_start < 0 or context_end + horizon > len(surfaces):
                continue

            # Prepare context
            ctx_surface = torch.tensor(
                surfaces[context_start:context_end],
                dtype=torch.float32, device=device
            ).unsqueeze(0)
            ctx_ex = torch.tensor(
                ex_data[context_start:context_end],
                dtype=torch.float32, device=device
            ).unsqueeze(0)

            context = {"surface": ctx_surface, "ex_feats": ctx_ex}
            anchor_val = surfaces[context_end - 1, r, c]

            # Generate multiple samples
            for _ in range(n_samples):
                # Generate prediction with explicit latent sampling (float32 dtype)
                if hasattr(model, 'get_surface_given_conditions'):
                    # Sample latent z with correct dtype (float32 to match model)
                    T = context_len + horizon
                    z_sample = torch.randn(
                        (1, T, model.config["latent_dim"]),
                        dtype=torch.float32,  # Match model's dtype
                        device=device
                    )
                    pred = model.get_surface_given_conditions(context, z=z_sample, horizon=horizon)
                    if isinstance(pred, tuple):
                        pred_surface = pred[0]
                    else:
                        pred_surface = pred
                else:
                    # Fallback
                    pred_surface = model(context)

                # Extract median prediction at horizon
                # pred_surface shape: (1, horizon, 3, 5, 5) for quantile model
                # or (1, horizon, 5, 5) for non-quantile
                if len(pred_surface.shape) == 5:
                    # Quantile model - take median (index 1)
                    pred_val = pred_surface[0, horizon-1, 1, r, c].cpu().numpy()
                else:
                    pred_val = pred_surface[0, horizon-1, r, c].cpu().numpy()

                change = pred_val - anchor_val
                all_changes.append(change)

    return np.array(all_changes)


def print_report(metrics_by_horizon: Dict[int, MarginalMetrics]):
    """Print formatted report card."""
    print("\n" + "="*80)
    print("MARGINAL DISTRIBUTION METRICS - PRODUCTION VALIDATION")
    print("="*80)
    print("\nCriterion: Model distribution should match Ground Truth distribution")
    print("           (stacking all predictions at horizon H vs all actuals at H)")
    print()

    # Thresholds
    print("Thresholds:")
    print("  Wasserstein: < 0.01 (lower = better)")
    print("  KS p-value:  > 0.05 (higher = distributions match)")
    print("  Mean Bias:   < 0.005 (absolute)")
    print("  Var Ratio:   0.8 - 1.2 (model variance / GT variance)")
    print()

    # Per-horizon results
    for h in sorted(metrics_by_horizon.keys()):
        m = metrics_by_horizon[h]
        passes = m.passes_thresholds()

        print(f"\n{'─'*40}")
        print(f"Horizon: {h} days")
        print(f"{'─'*40}")
        print(f"  Samples: GT={m.n_gt:,}, Model={m.n_model:,}")
        print()
        print("  PRIMARY METRICS:")
        status = "✓" if passes['wasserstein'] else "✗"
        print(f"    Wasserstein:   {m.wasserstein:.6f}  {status}")
        status = "✓" if passes['ks_test'] else "✗"
        print(f"    KS Statistic:  {m.ks_statistic:.4f} (p={m.ks_pvalue:.4f})  {status}")
        print(f"    Quantile RMSE: {m.quantile_rmse:.6f}")
        print()
        print("  MOMENT MATCHING:")
        status = "✓" if passes['mean_bias'] else "✗"
        print(f"    Mean:     GT={m.mean_gt:+.6f}, Model={m.mean_model:+.6f}, Bias={m.mean_bias:+.6f}  {status}")
        status = "✓" if passes['var_ratio'] else "✗"
        print(f"    Variance: GT={m.var_gt:.6f}, Model={m.var_model:.6f}, Ratio={m.var_ratio:.3f}  {status}")
        print(f"    Skewness: GT={m.skew_gt:+.3f}, Model={m.skew_model:+.3f}, Δ={m.skew_diff:+.3f}")
        print(f"    Kurtosis: GT={m.kurt_gt:+.3f}, Model={m.kurt_model:+.3f}, Δ={m.kurt_diff:+.3f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_pass = True
    for h in sorted(metrics_by_horizon.keys()):
        m = metrics_by_horizon[h]
        passes = m.passes_thresholds()
        n_pass = sum(passes.values())
        status = "PASS" if n_pass == len(passes) else f"WARN ({n_pass}/{len(passes)})"
        if n_pass < len(passes):
            all_pass = False
        print(f"  H={h:2d}: {status}")

    print()
    if all_pass:
        print("  OVERALL: ✓ All horizons pass production thresholds")
    else:
        print("  OVERALL: ⚠ Some horizons need attention")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compute marginal distribution metrics')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 7, 14, 30],
                       help='Horizons to evaluate')
    parser.add_argument('--n_samples', type=int, default=50,
                       help='Number of samples per sequence')
    parser.add_argument('--context_len', type=int, default=60,
                       help='Context length')
    parser.add_argument('--grid_point', type=int, nargs=2, default=[2, 2],
                       help='Grid point to analyze (row col)')
    parser.add_argument('--output_dir', type=str, default='results/marginal_metrics',
                       help='Output directory')
    parser.add_argument('--period', type=str, default='full',
                       choices=['full', 'crisis', 'post_crisis', 'recent'],
                       help='Data period to evaluate')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")

    # Define periods
    periods = {
        'full': (1000, 5000),
        'crisis': (2000, 2750),
        'post_crisis': (2750, 3500),
        'recent': (4500, 5500),
    }
    start_idx, end_idx = periods[args.period]

    # Create indices
    indices = np.arange(start_idx + args.context_len,
                        end_idx - max(args.horizons))
    print(f"Period: {args.period}, Indices: {len(indices)}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Determine model type
    model_config = checkpoint.get('model_config', checkpoint.get('config', {}))

    if model_config.get('use_conditional_prior', False):
        from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior
        model = CVAEMemRandConditionalPrior(model_config)
    else:
        from vae.cvae_with_mem_randomized import CVAEMemRand
        model = CVAEMemRand(model_config)

    # Load weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_weights(checkpoint)

    model = model.to(device)
    model.eval()
    print("✓ Model loaded")

    # Compute metrics for each horizon
    grid_point = tuple(args.grid_point)
    metrics_by_horizon = {}

    for horizon in args.horizons:
        print(f"\nProcessing horizon {horizon}...")

        # Ground truth
        gt_changes = extract_ground_truth_changes(
            data, indices, args.context_len, horizon, grid_point
        )
        print(f"  GT samples: {len(gt_changes)}")

        # Model samples
        model_changes = generate_model_samples(
            model, data, indices, args.context_len, horizon,
            args.n_samples, grid_point, device
        )
        print(f"  Model samples: {len(model_changes)}")

        # Compute metrics
        metrics = compute_marginal_metrics(gt_changes, model_changes, horizon)
        metrics_by_horizon[horizon] = metrics

    # Print report
    print_report(metrics_by_horizon)

    # Save results
    results = {
        'period': args.period,
        'context_len': args.context_len,
        'n_samples': args.n_samples,
        'grid_point': list(grid_point),
        'model_path': args.model_path,
        'metrics': {h: m.to_dict() for h, m in metrics_by_horizon.items()}
    }

    output_file = output_dir / f"marginal_metrics_{args.period}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
