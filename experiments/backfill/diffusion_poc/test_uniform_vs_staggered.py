#!/usr/bin/env python
"""
EXPERIMENT: Uniform vs Staggered Sampling for Block-AR BiGRU Model.

This is the CRITICAL experiment that tests whether the quality degradation
seen in Block-AR is due to the staggered sampling strategy or the BiGRU
architecture itself.

Previous evidence from the DDPM POC (3D ConvNet) showed:
- Uniform: kurtosis 0.663, cal arb 9.4%, ACF 0.842
- Staggered: kurtosis 0.023, cal arb 30.4%, ACF 0.200

But that was a DIFFERENT model. This experiment tests the SAME Block-AR BiGRU
model with both sampling strategies to isolate the cause.

Three sampling variants tested:
1. Staggered (default): max_residual=20, frames denoise to different levels
2. Staggered (zero residual): max_residual=0, equivalent to uniform-like staggered
3. True uniform: DDPMScheduler.sample() where all frames share identical timestep

Usage:
    PYTHONPATH=/home/max/Documents/vol-surface-vae-pub python \
        experiments/backfill/diffusion_poc/test_uniform_vs_staggered.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import kurtosis

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    _DenoiserAdapter,
    denormalize_iv,
)
from diffusion.ddpm_scheduler import DDPMScheduler
from experiments.backfill.diffusion_poc.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# =============================================================================
# Uniform Sampling Implementation
# =============================================================================

class _UniformDenoiserAdapter(nn.Module):
    """Wraps BiGRUDenoiser for UNIFORM sampling (scalar t -> per-frame t).

    DDPMScheduler.sample() passes t as (B,) scalar, but BiGRUDenoiser expects
    per-frame timesteps (B, T). This adapter expands the scalar t to all frames.
    """

    def __init__(
        self,
        denoiser,
        condition: torch.Tensor,
        positions: torch.Tensor,
        n_frames: int,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.condition = condition
        self.positions = positions
        self.n_frames = n_frames

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition_unused: torch.Tensor
    ) -> torch.Tensor:
        # x_t: (B, T, 5, 5) -> flatten to (B, T, 25)
        B, T, H, W = x_t.shape
        x_flat = x_t.reshape(B, T, H * W)

        # Expand scalar t (B,) -> per-frame t (B, T) — uniform across frames
        t_per_frame = t.unsqueeze(1).expand(B, T)

        # Call denoiser with per-frame timesteps (all the same value)
        noise_pred_flat = self.denoiser(
            x_flat, self.condition, self.positions, t_per_frame
        )  # (B, T, 25)

        # Reshape back to (B, T, 5, 5)
        return noise_pred_flat.reshape(B, T, H, W)


@torch.no_grad()
def sample_uniform(
    model: ConditionalBlockARDDPM,
    history: torch.Tensor,
    n_samples: int = 1,
) -> torch.Tensor:
    """
    Block-AR generation using UNIFORM DDPM sampling (all frames same timestep).

    Instead of staggered per-frame timesteps, all frames in a block share
    the same global timestep during reverse diffusion.

    Args:
        model: Trained Block-AR model
        history: (B, history_len, 5, 5) in [-1, 1]
        n_samples: number of independent samples per history

    Returns:
        (B, n_samples, future_len, 5, 5) denormalized to [0, 1]
    """
    B = history.shape[0]
    device = history.device
    bs = model.config.block_size
    n_blocks = model.config.future_len // bs

    model._ensure_scheduler_device(device)

    all_samples = []

    for _ in range(n_samples):
        current_cond_surfaces = history
        blocks = []

        for block_idx in range(n_blocks):
            # Encode growing context
            condition = model.encoder(current_cond_surfaces, mask=None)

            # Create positions for this block
            positions = (
                torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                + block_idx * bs
            )

            # Create uniform adapter (expands scalar t to per-frame)
            adapter = _UniformDenoiserAdapter(
                model.denoiser, condition, positions, bs
            )

            # Generate block with UNIFORM DDPM (DDPMScheduler.sample)
            # This calls p_sample with a scalar t for ALL frames at each step
            shape = (B, bs, model.config.surface_h, model.config.surface_w)
            block = model.scheduler.sample(
                adapter,
                history,  # dummy condition (ignored by adapter)
                shape,
            )  # (B, bs, 5, 5) in [-1, 1]

            blocks.append(block)

            # Grow conditioning surfaces
            current_cond_surfaces = torch.cat(
                [current_cond_surfaces, block], dim=1
            )

        # Concatenate all blocks
        full_trajectory = torch.cat(blocks, dim=1)
        all_samples.append(full_trajectory)

    samples = torch.stack(all_samples, dim=1)
    samples = denormalize_iv(samples)
    samples = samples.clamp(0.0, 1.0)
    return samples


# =============================================================================
# Metrics
# =============================================================================

def compute_acf(series: np.ndarray, max_lag: int = 20) -> np.ndarray:
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        acf.append(cov / var)
    return np.array(acf)


def compute_calendar_arb(samples: np.ndarray) -> float:
    """Calendar arbitrage violation rate."""
    tenors = np.array([1, 2, 4, 8, 12])
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]
        total_var = surf ** 2 * tenors[:, None]
        for i in range(4):
            violation = (total_var[:, i, :] > total_var[:, i + 1, :] * 1.001)
            violations.append(violation.mean())
    return float(np.mean(violations))


def compute_butterfly_arb(samples: np.ndarray) -> float:
    """Butterfly arbitrage violation rate."""
    violations = []
    for t_idx in range(samples.shape[1]):
        surf = samples[:, t_idx]
        d2_dk2 = surf[:, :, :-2] - 2 * surf[:, :, 1:-1] + surf[:, :, 2:]
        violation = (d2_dk2 < -0.005).mean()
        violations.append(float(violation))
    return float(np.mean(violations))


def compute_kurtosis_ratio(gen_samples: np.ndarray, gt: np.ndarray) -> float:
    """Kurtosis ratio (gen/gt) of frame-to-frame changes."""
    gt_diff = np.diff(gt, axis=1).flatten()
    gen_diff = np.diff(gen_samples, axis=1).flatten()
    gt_kurt = float(kurtosis(gt_diff, fisher=True))
    gen_kurt = float(kurtosis(gen_diff, fisher=True))
    return gen_kurt / gt_kurt if gt_kurt != 0 else float("inf")


def compute_acf_lag1(gen_samples: np.ndarray, gt: np.ndarray) -> float:
    """ACF correlation between GT and generated ACFs."""
    gt_atm = gt[:, :, 2, 2].flatten()
    gen_atm = gen_samples[:, :, 2, 2].flatten()
    gt_acf = compute_acf(gt_atm, 20)
    gen_acf = compute_acf(gen_atm, 20)
    return float(np.corrcoef(gt_acf, gen_acf)[0, 1])


def compute_ci_coverage(
    cond_samples: np.ndarray, gt: np.ndarray, level: float = 0.9
) -> float:
    """CI coverage at given level."""
    alpha = (1 - level) / 2
    lower = np.quantile(cond_samples, alpha, axis=1)
    upper = np.quantile(cond_samples, 1 - alpha, axis=1)
    covered = (gt >= lower) & (gt <= upper)
    return float(covered.mean())


def compute_grid_correlation(gen_samples: np.ndarray, gt: np.ndarray) -> float:
    """Correlation between GT and generated grid patterns (averaged across time)."""
    correlations = []
    for t in range(gt.shape[1]):
        gt_grid = gt[:, t].mean(axis=0).flatten()  # (25,)
        gen_grid = gen_samples[:, t].mean(axis=0).flatten()  # (25,)
        if gt_grid.std() > 0 and gen_grid.std() > 0:
            corr = np.corrcoef(gt_grid, gen_grid)[0, 1]
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 0.0


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    config = get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples = 20
    max_batches = 10

    # Load model
    model_path = "models/backfill/block_ar/final_model.pt"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    print("=" * 70)
    print("EXPERIMENT: Uniform vs Staggered Sampling — Block-AR BiGRU")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"N samples: {n_samples}")
    print(f"Max batches: {max_batches}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)

    model = ConditionalBlockARDDPM(model_config)
    # Use regular weights (no EMA) based on memory that EMA destroys conditionality
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Block size: {model_config.block_size}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Load test data
    data = np.load(config.data_path)
    surfaces = data["surface"]
    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len, start_idx=config.test_start
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )
    print(f"Test windows: {len(test_dataset)}")

    # =========================================================================
    # Generate samples with all three methods
    # =========================================================================
    methods = {
        "staggered_mr20": {"max_residual": 20, "method": "staggered"},
        "staggered_mr0": {"max_residual": 0, "method": "staggered"},
        "uniform": {"method": "uniform"},
    }

    results = {}

    for method_name, method_config in methods.items():
        print(f"\n{'='*70}")
        print(f"Generating with: {method_name}")
        print(f"{'='*70}")

        all_samples = []
        all_gt = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(test_loader, desc=f"  {method_name}", total=max_batches)
            ):
                if batch_idx >= max_batches:
                    break

                history = batch["history"].to(device)
                future_gt = denormalize_iv(batch["future"].to(device))

                if method_config["method"] == "staggered":
                    samples = model.sample(
                        history,
                        n_samples=n_samples,
                        max_residual=method_config["max_residual"],
                    )
                else:  # uniform
                    samples = sample_uniform(model, history, n_samples=n_samples)

                all_samples.append(samples.cpu().numpy())
                all_gt.append(future_gt.cpu().numpy())

        cond_samples = np.concatenate(all_samples, axis=0)
        ground_truth = np.concatenate(all_gt, axis=0)

        N, S, T, H, W = cond_samples.shape
        all_gen = cond_samples.reshape(N * S, T, H, W)
        first_sample = cond_samples[:, 0]  # (N, T, 5, 5)

        print(f"  Shape: cond_samples={cond_samples.shape}, gt={ground_truth.shape}")

        # Compute metrics
        cal_arb = compute_calendar_arb(all_gen)
        but_arb = compute_butterfly_arb(all_gen)
        kurt_ratio = compute_kurtosis_ratio(first_sample, ground_truth)
        acf_corr = compute_acf_lag1(first_sample, ground_truth)
        ci_90 = compute_ci_coverage(cond_samples, ground_truth, 0.9)
        grid_corr = compute_grid_correlation(first_sample, ground_truth)

        # Compute raw kurtosis values
        gt_diff = np.diff(ground_truth, axis=1).flatten()
        gen_diff = np.diff(first_sample, axis=1).flatten()
        gt_kurt = float(kurtosis(gt_diff, fisher=True))
        gen_kurt = float(kurtosis(gen_diff, fisher=True))

        results[method_name] = {
            "calendar_arb": cal_arb,
            "butterfly_arb": but_arb,
            "kurtosis_ratio": kurt_ratio,
            "gt_kurtosis": gt_kurt,
            "gen_kurtosis": gen_kurt,
            "acf_correlation": acf_corr,
            "ci_90_coverage": ci_90,
            "grid_correlation": grid_corr,
        }

        print(f"\n  Results for {method_name}:")
        print(f"    Calendar arb:    {cal_arb:.1%}")
        print(f"    Butterfly arb:   {but_arb:.1%}")
        print(f"    Kurtosis ratio:  {kurt_ratio:.3f} (GT={gt_kurt:.3f}, Gen={gen_kurt:.3f})")
        print(f"    ACF correlation: {acf_corr:.3f}")
        print(f"    90% CI coverage: {ci_90:.1%}")
        print(f"    Grid correlation:{grid_corr:.3f}")

    # =========================================================================
    # Comparative Summary
    # =========================================================================
    print("\n\n" + "=" * 90)
    print("COMPARATIVE SUMMARY: Uniform vs Staggered Sampling")
    print("=" * 90)
    print(f"{'Metric':<25} {'Staggered(mr=20)':<20} {'Staggered(mr=0)':<20} {'Uniform':<20} {'Target':<15}")
    print("-" * 90)

    metrics = [
        ("Calendar Arb", "calendar_arb", "<10%"),
        ("Butterfly Arb", "butterfly_arb", "<20%"),
        ("Kurtosis Ratio", "kurtosis_ratio", "0.5-2.0"),
        ("GT Kurtosis", "gt_kurtosis", "-"),
        ("Gen Kurtosis", "gen_kurtosis", "-"),
        ("ACF Correlation", "acf_correlation", ">0.5"),
        ("90% CI Coverage", "ci_90_coverage", ">65%"),
        ("Grid Correlation", "grid_correlation", ">0.5"),
    ]

    for label, key, target in metrics:
        vals = []
        for method_name in ["staggered_mr20", "staggered_mr0", "uniform"]:
            v = results[method_name][key]
            if key in ["calendar_arb", "butterfly_arb", "ci_90_coverage"]:
                vals.append(f"{v:.1%}")
            else:
                vals.append(f"{v:.3f}")
        print(f"{label:<25} {vals[0]:<20} {vals[1]:<20} {vals[2]:<20} {target:<15}")

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("=" * 90)

    # Compute ratios for interpretation
    stag = results["staggered_mr20"]
    unif = results["uniform"]

    print(f"\nUniform vs Staggered(mr=20):")
    print(f"  Calendar arb:    {stag['calendar_arb']:.1%} -> {unif['calendar_arb']:.1%}  "
          f"({'improved' if unif['calendar_arb'] < stag['calendar_arb'] else 'worsened'})")
    print(f"  Butterfly arb:   {stag['butterfly_arb']:.1%} -> {unif['butterfly_arb']:.1%}  "
          f"({'improved' if unif['butterfly_arb'] < stag['butterfly_arb'] else 'worsened'})")
    print(f"  Kurtosis ratio:  {stag['kurtosis_ratio']:.3f} -> {unif['kurtosis_ratio']:.3f}  "
          f"(closer to 1.0 is better)")
    print(f"  ACF correlation: {stag['acf_correlation']:.3f} -> {unif['acf_correlation']:.3f}  "
          f"({'improved' if unif['acf_correlation'] > stag['acf_correlation'] else 'worsened'})")
    print(f"  CI coverage:     {stag['ci_90_coverage']:.1%} -> {unif['ci_90_coverage']:.1%}  "
          f"({'improved' if unif['ci_90_coverage'] > stag['ci_90_coverage'] else 'worsened'})")
    print(f"  Grid corr:       {stag['grid_correlation']:.3f} -> {unif['grid_correlation']:.3f}  "
          f"({'improved' if unif['grid_correlation'] > stag['grid_correlation'] else 'worsened'})")

    # Key conclusion
    print(f"\n{'='*90}")
    print("KEY QUESTION: Is the problem the sampling strategy or the BiGRU architecture?")
    print("="*90)

    # Count improvements
    improvements = 0
    total = 0
    for key, better_fn in [
        ("calendar_arb", lambda u, s: u < s),
        ("butterfly_arb", lambda u, s: u < s),
        ("kurtosis_ratio", lambda u, s: abs(u - 1.0) < abs(s - 1.0)),
        ("acf_correlation", lambda u, s: u > s),
        ("grid_correlation", lambda u, s: u > s),
    ]:
        if better_fn(unif[key], stag[key]):
            improvements += 1
        total += 1

    if improvements >= 4:
        print(f"\nUNIFORM SAMPLING FIXES MOST ISSUES ({improvements}/{total} metrics improved)")
        print("=> Staggered sampling is the dominant degradation factor")
        print("=> BiGRU architecture may be adequate with proper sampling")
    elif improvements >= 2:
        print(f"\nMIXED RESULTS ({improvements}/{total} metrics improved)")
        print("=> Both sampling strategy AND architecture contribute to degradation")
        print("=> Need both fixes: uniform sampling + better denoiser")
    else:
        print(f"\nUNIFORM SAMPLING DOES NOT HELP ({improvements}/{total} metrics improved)")
        print("=> BiGRU architecture is the dominant degradation factor")
        print("=> Fixing sampling alone is insufficient; need architectural change")


if __name__ == "__main__":
    main()
