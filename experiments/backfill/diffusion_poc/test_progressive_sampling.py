#!/usr/bin/env python
"""
Test progressive noise sampling for CI calibration improvement.

This script tests whether inference-time progressive noise scheduling
improves CI coverage at far horizons without retraining the model.

Hypothesis:
- h=1 coverage should stay similar (~90%)
- h=30 coverage should improve (currently 80.4% → target >85%)
- CI width should naturally widen for far horizons

Usage:
    python experiments/backfill/diffusion_poc/test_progressive_sampling.py
    python experiments/backfill/diffusion_poc/test_progressive_sampling.py --max_batches 10
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig, denormalize_iv
from experiments.backfill.diffusion_poc.metrics import (
    compute_fsd,
    extract_encoder_features,
    extract_domain_features,
    FrechetSurfaceDistance,
)


# IV normalization constants (must match training)
IV_MIN = 0.0
IV_MAX = 1.0


def normalize_iv(iv: torch.Tensor) -> torch.Tensor:
    """Normalize IV from [0, 1] to [-1, 1]."""
    return 2.0 * (iv - IV_MIN) / (IV_MAX - IV_MIN) - 1.0


class TestDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""

    def __init__(self, surfaces: np.ndarray, history_len: int, future_len: int,
                 start_idx: int = 0, end_idx: int = None):
        self.history_len = history_len
        self.future_len = future_len
        self.total_len = history_len + future_len

        end_idx = end_idx or len(surfaces)
        self.surfaces = surfaces[start_idx:end_idx]
        self.valid_starts = list(range(len(self.surfaces) - self.total_len + 1))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        seq = self.surfaces[start:start + self.total_len]

        history = torch.from_numpy(seq[:self.history_len]).float()
        future = torch.from_numpy(seq[self.history_len:]).float()

        # Normalize to [-1, 1]
        history = normalize_iv(history)
        future = normalize_iv(future)

        return {"history": history, "future": future}


def sample_uniform(
    model: ConditionalDDPM,
    history: torch.Tensor,
    n_samples: int = 50,
) -> torch.Tensor:
    """Standard uniform DDPM/DDIM sampling (baseline)."""
    return model.sample(history, n_samples=n_samples, sampler='ddim', n_inference_steps=20)


def sample_progressive_denoising(
    model: ConditionalDDPM,
    history: torch.Tensor,
    n_samples: int = 50,
    max_residual_noise_level: int = 15,
) -> torch.Tensor:
    """
    Progressive denoising: early frames denoise fully, late frames keep some noise.

    Frame 0: denoise to t=0 (fully clean)
    Frame T-1: denoise to t=max_residual_noise_level (some noise remains)

    This creates natural uncertainty growth for far horizons.
    """
    B = history.shape[0]
    device = history.device
    T_fut = model.config.future_len
    H, W = model.config.surface_h, model.config.surface_w
    n_steps = model.config.n_steps

    all_samples = []

    for _ in range(n_samples):
        # Start from pure noise
        x_t = torch.randn(B, T_fut, H, W, device=device)

        # Encode history once
        condition = model.denoiser.history_encoder(history)

        # Progressive denoising
        for t_val in reversed(range(n_steps)):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Compute per-frame minimum noise level
            # Frame 0: min_level = 0 (denoise fully)
            # Frame T-1: min_level = max_residual_noise_level
            frame_indices = torch.arange(T_fut, device=device).float()
            min_noise_levels = (frame_indices / (T_fut - 1) * max_residual_noise_level).long()

            # Only denoise frames where current t >= their minimum level
            for frame_idx in range(T_fut):
                if t_val >= min_noise_levels[frame_idx].item():
                    # Denoise this frame
                    frame_x = x_t[:, frame_idx:frame_idx+1, :, :]  # (B, 1, H, W)

                    # Get noise prediction for this frame
                    # We need to handle single-frame prediction
                    noise_pred = model.denoiser(
                        x_t,  # Full sequence for context
                        t,
                        history
                    )

                    # Update only this frame using scheduler
                    alpha_bar = model.scheduler.alpha_bar[t_val]
                    alpha_bar_prev = model.scheduler.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0)
                    beta = model.scheduler.betas[t_val]

                    # DDPM update for this frame
                    noise = noise_pred[:, frame_idx, :, :]
                    x_0_pred = (x_t[:, frame_idx, :, :] - torch.sqrt(1 - alpha_bar) * noise) / torch.sqrt(alpha_bar)

                    if t_val > 0:
                        posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                        x_t[:, frame_idx, :, :] = (
                            torch.sqrt(alpha_bar_prev) * x_0_pred +
                            torch.sqrt(1 - alpha_bar_prev - posterior_var) * noise +
                            torch.sqrt(posterior_var) * torch.randn_like(x_t[:, frame_idx, :, :])
                        )
                    else:
                        x_t[:, frame_idx, :, :] = x_0_pred

        all_samples.append(x_t)

    samples = torch.stack(all_samples, dim=1)  # (B, n_samples, T_fut, H, W)

    # Denormalize
    samples = denormalize_iv(samples)

    return samples


def sample_posthoc_noise(
    model: ConditionalDDPM,
    history: torch.Tensor,
    n_samples: int = 50,
    max_noise_scale: float = 0.03,
) -> torch.Tensor:
    """
    Simple post-hoc approach: Generate with standard DDPM, then add
    progressive noise to later frames.

    This is the simplest test - just adds uncertainty to far horizons.
    """
    # Standard sampling
    samples = model.sample(history, n_samples=n_samples, sampler='ddim', n_inference_steps=20)
    # samples: (B, n_samples, T_fut, H, W) already denormalized

    T_fut = samples.shape[2]

    # Add progressive noise
    for frame_idx in range(T_fut):
        noise_scale = max_noise_scale * (frame_idx / (T_fut - 1))
        samples[:, :, frame_idx] += torch.randn_like(samples[:, :, frame_idx]) * noise_scale

    # Clamp to valid range
    samples = samples.clamp(0.0, 1.0)

    return samples


def compute_ci_coverage_by_horizon(
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    ci_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
    horizons: List[int] = [1, 7, 14, 30],
) -> Dict:
    """
    Compute CI coverage per horizon.

    Args:
        samples: (B, n_samples, T_fut, H, W) generated samples
        ground_truth: (B, T_fut, H, W) actual future surfaces
        ci_levels: CI levels to compute
        horizons: Which horizons to report (1-indexed days)

    Returns:
        dict with coverage per horizon and CI widths
    """
    results = {}

    for level in ci_levels:
        results[f"coverage_{int(level*100)}"] = {}
        results[f"ci_width_{int(level*100)}"] = {}

        alpha = (1 - level) / 2
        lower = torch.quantile(samples, alpha, dim=1)
        upper = torch.quantile(samples, 1 - alpha, dim=1)

        for h in horizons:
            if h <= samples.shape[2]:
                frame_idx = h - 1  # Convert to 0-indexed

                # Coverage at this horizon
                covered = (ground_truth[:, frame_idx] >= lower[:, frame_idx]) & \
                          (ground_truth[:, frame_idx] <= upper[:, frame_idx])
                coverage = covered.float().mean().item()
                results[f"coverage_{int(level*100)}"][f"h={h}"] = coverage

                # CI width at this horizon
                width = (upper[:, frame_idx] - lower[:, frame_idx]).mean().item()
                results[f"ci_width_{int(level*100)}"][f"h={h}"] = width

    return results


def run_comparison(
    model: ConditionalDDPM,
    dataloader: DataLoader,
    device: str,
    n_samples: int = 50,
    max_batches: int = 20,
    compute_fsd_metric: bool = False,
) -> Dict:
    """Run comparison of all three sampling methods."""

    methods = {
        "uniform": lambda h: sample_uniform(model, h, n_samples),
        "posthoc_noise": lambda h: sample_posthoc_noise(model, h, n_samples, max_noise_scale=0.03),
        # Progressive denoising is slower, use fewer samples
        # "progressive": lambda h: sample_progressive_denoising(model, h, min(n_samples, 20)),
    }

    results = {method: {"coverages": [], "widths": []} for method in methods}

    # For FSD computation - collect features across batches
    if compute_fsd_metric:
        real_features_encoder = []
        real_features_domain = []
        gen_features = {method: {"encoder": [], "domain": []} for method in methods}

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            # Denormalize ground truth for comparison
            future_gt_denorm = denormalize_iv(future_gt)

            # Collect real features for FSD
            if compute_fsd_metric:
                # Encoder features from real sequences
                real_enc = extract_encoder_features(model, future_gt)
                real_features_encoder.append(real_enc.cpu())

                # Domain features from real sequences (use denormalized)
                real_dom = extract_domain_features(future_gt_denorm)
                real_features_domain.append(real_dom.cpu())

            for method_name, sample_fn in methods.items():
                samples = sample_fn(history)

                coverage_results = compute_ci_coverage_by_horizon(
                    samples, future_gt_denorm,
                    ci_levels=[0.9],
                    horizons=[1, 7, 14, 30]
                )

                results[method_name]["coverages"].append(coverage_results)

                # Collect generated features for FSD
                if compute_fsd_metric:
                    # Use mean of samples for feature extraction
                    samples_mean = samples.mean(dim=1)  # (B, T_fut, H, W)

                    # Normalize back for encoder features
                    samples_norm = normalize_iv(samples_mean)
                    gen_enc = extract_encoder_features(model, samples_norm)
                    gen_features[method_name]["encoder"].append(gen_enc.cpu())

                    # Domain features from denormalized samples
                    gen_dom = extract_domain_features(samples_mean)
                    gen_features[method_name]["domain"].append(gen_dom.cpu())

    # Aggregate results
    aggregated = {}
    for method_name in methods:
        aggregated[method_name] = {}

        # Average coverage across batches
        for key in results[method_name]["coverages"][0]:
            if "coverage" in key:
                aggregated[method_name][key] = {}
                for horizon_key in results[method_name]["coverages"][0][key]:
                    values = [r[key][horizon_key] for r in results[method_name]["coverages"]]
                    aggregated[method_name][key][horizon_key] = np.mean(values)

            if "ci_width" in key:
                aggregated[method_name][key] = {}
                for horizon_key in results[method_name]["coverages"][0][key]:
                    values = [r[key][horizon_key] for r in results[method_name]["coverages"]]
                    aggregated[method_name][key][horizon_key] = np.mean(values)

    # Compute FSD metrics
    if compute_fsd_metric:
        # Concatenate all features
        all_real_encoder = torch.cat(real_features_encoder, dim=0)
        all_real_domain = torch.cat(real_features_domain, dim=0)

        for method_name in methods:
            all_gen_encoder = torch.cat(gen_features[method_name]["encoder"], dim=0)
            all_gen_domain = torch.cat(gen_features[method_name]["domain"], dim=0)

            # Compute FSD
            fsd_encoder = compute_fsd(all_real_encoder, all_gen_encoder)
            fsd_domain = compute_fsd(all_real_domain, all_gen_domain)

            aggregated[method_name]["fsd_encoder"] = fsd_encoder
            aggregated[method_name]["fsd_domain"] = fsd_domain

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Test progressive sampling")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/ddpm_poc/checkpoint_epoch_50.pt",
                        help="Path to trained model")
    parser.add_argument("--data_path", type=str,
                        default="data/vol_surface_with_ret.npz",
                        help="Path to data")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Samples per history")
    parser.add_argument("--max_batches", type=int, default=20,
                        help="Max batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--compute_fsd", action="store_true",
                        help="Compute Fréchet Surface Distance metric")
    args = parser.parse_args()

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print("=" * 70)
    print("Progressive Sampling Test")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ConditionalDDPM(config, scheduler_config={"device": device})
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (history={config.history_len}, future={config.future_len})")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data = np.load(args.data_path)
    surfaces = data["surface"]

    # Use test split (last portion)
    test_start = int(len(surfaces) * 0.9)
    test_dataset = TestDataset(
        surfaces,
        history_len=config.history_len,
        future_len=config.future_len,
        start_idx=test_start
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    print(f"Test dataset: {len(test_dataset)} sequences")

    # Run comparison
    print(f"\nRunning comparison (n_samples={args.n_samples}, max_batches={args.max_batches}, compute_fsd={args.compute_fsd})...")
    results = run_comparison(
        model, test_loader, device,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
        compute_fsd_metric=args.compute_fsd
    )

    # Print results
    print("\n" + "=" * 70)
    print("Results: 90% CI Coverage by Horizon")
    print("=" * 70)

    print(f"\n{'Method':<20} | {'h=1':>8} | {'h=7':>8} | {'h=14':>8} | {'h=30':>8} |")
    print("-" * 70)

    for method_name, method_results in results.items():
        cov = method_results.get("coverage_90", {})
        print(f"{method_name:<20} | {cov.get('h=1', 0)*100:>7.1f}% | {cov.get('h=7', 0)*100:>7.1f}% | "
              f"{cov.get('h=14', 0)*100:>7.1f}% | {cov.get('h=30', 0)*100:>7.1f}% |")

    print("\n" + "=" * 70)
    print("Results: 90% CI Width by Horizon")
    print("=" * 70)

    print(f"\n{'Method':<20} | {'h=1':>8} | {'h=7':>8} | {'h=14':>8} | {'h=30':>8} | {'Ratio h30/h1':>12} |")
    print("-" * 85)

    for method_name, method_results in results.items():
        width = method_results.get("ci_width_90", {})
        w1 = width.get('h=1', 0.001)
        w30 = width.get('h=30', 0.001)
        ratio = w30 / w1 if w1 > 0 else 0
        print(f"{method_name:<20} | {w1:>8.4f} | {width.get('h=7', 0):>8.4f} | "
              f"{width.get('h=14', 0):>8.4f} | {w30:>8.4f} | {ratio:>12.2f} |")

    # FSD Results (if computed)
    if args.compute_fsd:
        print("\n" + "=" * 70)
        print("Results: Fréchet Surface Distance (lower is better)")
        print("=" * 70)

        print(f"\n{'Method':<20} | {'FSD-Encoder':>12} | {'FSD-Domain':>12} |")
        print("-" * 52)

        baseline_fsd_enc = results.get("uniform", {}).get("fsd_encoder", 1.0)
        baseline_fsd_dom = results.get("uniform", {}).get("fsd_domain", 1.0)

        for method_name, method_results in results.items():
            fsd_enc = method_results.get("fsd_encoder", 0)
            fsd_dom = method_results.get("fsd_domain", 0)
            print(f"{method_name:<20} | {fsd_enc:>12.3f} | {fsd_dom:>12.3f} |")

        print("\nFSD Improvement vs Baseline (uniform):")
        for method_name, method_results in results.items():
            if method_name == "uniform":
                continue
            fsd_enc = method_results.get("fsd_encoder", baseline_fsd_enc)
            fsd_dom = method_results.get("fsd_domain", baseline_fsd_dom)
            enc_change = ((fsd_enc - baseline_fsd_enc) / baseline_fsd_enc) * 100 if baseline_fsd_enc > 0 else 0
            dom_change = ((fsd_dom - baseline_fsd_dom) / baseline_fsd_dom) * 100 if baseline_fsd_dom > 0 else 0
            print(f"  {method_name}: FSD-Encoder {enc_change:+.1f}%, FSD-Domain {dom_change:+.1f}%")

        print("\nInterpretation:")
        print("  - FSD measures distributional similarity (lower = more realistic)")
        print("  - FSD-Encoder: Uses learned 128-dim representation")
        print("  - FSD-Domain: Uses hand-crafted features (level, skew, convexity, term slope)")
        print("  - If both agree, high confidence in assessment")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    uniform_h30 = results.get("uniform", {}).get("coverage_90", {}).get("h=30", 0)
    posthoc_h30 = results.get("posthoc_noise", {}).get("coverage_90", {}).get("h=30", 0)

    improvement = (posthoc_h30 - uniform_h30) * 100

    if improvement > 5:
        print(f"\n✅ Post-hoc noise significantly improves h=30 coverage (+{improvement:.1f}%)")
        print("   Recommendation: Implement full Diffusion Forcing (Option B)")
    elif improvement > 2:
        print(f"\n⚪ Post-hoc noise marginally improves h=30 coverage (+{improvement:.1f}%)")
        print("   Recommendation: Consider, but prioritize regime sampling (Option C)")
    else:
        print(f"\n❌ Post-hoc noise doesn't significantly help ({improvement:+.1f}%)")
        print("   Recommendation: Focus on regime sampling (Option C)")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
