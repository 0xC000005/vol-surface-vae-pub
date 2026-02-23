"""Diagnose x_0 prediction skewness — the clean image predictions at each step.

During reverse diffusion, the model predicts x_0 = (x_t - sqrt(1-alpha_bar)*noise_pred) / sqrt(alpha_bar).
This x_0 prediction is what gets clamped to [-1, 1] and determines the output distribution.

If x_0 predictions are skewed, the clamping + iterative refinement will preserve that skew.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_x0_prediction.py \
        --max_batches 20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import skew
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import dataclasses
from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig, denormalize_iv
from diffusion.block_ar import ConditionalBlockARDDPM, BlockARConfig
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config as get_ddpm_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def compute_x0_pred_skewness(noise_pred, x_t, t_val, sqrt_recip_ab, sqrt_recip_ab_m1):
    """Compute x_0 prediction and its skewness."""
    x_0_pred = sqrt_recip_ab * x_t - sqrt_recip_ab_m1 * noise_pred
    # Clamp like the actual sampling does
    x_0_clamped = x_0_pred.clamp(-1.0, 1.0)
    return x_0_pred, x_0_clamped


def analyze_x0_ddpm(model, test_loader, max_batches, device, timesteps):
    """Analyze x_0 prediction skewness for DDPM POC."""
    results = {}

    for t_val in timesteps:
        all_x0_raw = []
        all_x0_clamped = []
        all_gt = []
        all_diff_raw = []

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future = batch["future"].to(device)

            B = future.shape[0]
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(future)

            sqrt_ab = model.scheduler.sqrt_alpha_bar[t_val].to(device)
            sqrt_1mab = model.scheduler.sqrt_one_minus_alpha_bar[t_val].to(device)
            noisy_future = sqrt_ab * future + sqrt_1mab * noise

            with torch.no_grad():
                noise_pred = model.denoiser(noisy_future, t, history)

            # Compute x_0 prediction
            sqrt_recip = model.scheduler.sqrt_recip_alpha_bar[t_val].to(device)
            sqrt_recip_m1 = model.scheduler.sqrt_recip_alpha_bar_minus_one[t_val].to(device)

            x_0_raw = sqrt_recip * noisy_future - sqrt_recip_m1 * noise_pred
            x_0_clamped = x_0_raw.clamp(-1.0, 1.0)

            # Difference from ground truth
            diff = x_0_clamped - future

            all_x0_raw.append(x_0_raw.cpu().numpy())
            all_x0_clamped.append(x_0_clamped.cpu().numpy())
            all_gt.append(future.cpu().numpy())
            all_diff_raw.append(diff.cpu().numpy())

        x0_raw = np.concatenate(all_x0_raw)
        x0_clamp = np.concatenate(all_x0_clamped)
        gt = np.concatenate(all_gt)
        diffs = np.concatenate(all_diff_raw)

        # Compute changes (diff between frames) - this is what skewness metric measures
        x0_changes = np.diff(x0_clamp, axis=1)  # (N, T-1, H, W)
        gt_changes = np.diff(gt, axis=1)

        results[f"t={t_val}"] = {
            "x0_raw_skew": float(skew(x0_raw.flatten())),
            "x0_clamped_skew": float(skew(x0_clamp.flatten())),
            "x0_changes_skew": float(skew(x0_changes.flatten())),
            "gt_changes_skew": float(skew(gt_changes.flatten())),
            "diff_skew": float(skew(diffs.flatten())),
            "x0_raw_mean": float(np.mean(x0_raw)),
            "x0_clamped_mean": float(np.mean(x0_clamp)),
            "pct_clamped": float(np.mean(np.abs(x0_raw) > 1.0) * 100),
        }

    return results


def analyze_x0_block_ar(model, test_loader, max_batches, device, timesteps):
    """Analyze x_0 prediction skewness for Block-AR."""
    results = {}

    for t_val in timesteps:
        all_x0_raw = []
        all_x0_clamped = []
        all_gt = []
        all_diff_raw = []

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future = batch["future"].to(device)

            B = future.shape[0]
            T = model.config.block_size
            H, W = model.config.surface_h, model.config.surface_w

            future_block = future[:, :T]

            with torch.no_grad():
                condition = model.encoder(history)

            t_block = torch.full((B, T), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(future_block)

            sqrt_ab = model.scheduler.sqrt_alpha_bar[t_val].to(device)
            sqrt_1mab = model.scheduler.sqrt_one_minus_alpha_bar[t_val].to(device)
            noisy_block = sqrt_ab * future_block + sqrt_1mab * noise

            noisy_flat = noisy_block.reshape(B, T, H * W)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

            with torch.no_grad():
                noise_pred_flat = model.denoiser(noisy_flat, condition, positions, t_block)

            noise_pred = noise_pred_flat.reshape(B, T, H, W)

            # Compute x_0 prediction
            sqrt_recip = model.scheduler.sqrt_recip_alpha_bar[t_val].to(device)
            sqrt_recip_m1 = model.scheduler.sqrt_recip_alpha_bar_minus_one[t_val].to(device)

            x_0_raw = sqrt_recip * noisy_block - sqrt_recip_m1 * noise_pred
            x_0_clamped = x_0_raw.clamp(-1.0, 1.0)

            diff = x_0_clamped - future_block

            all_x0_raw.append(x_0_raw.cpu().numpy())
            all_x0_clamped.append(x_0_clamped.cpu().numpy())
            all_gt.append(future_block.cpu().numpy())
            all_diff_raw.append(diff.cpu().numpy())

        x0_raw = np.concatenate(all_x0_raw)
        x0_clamp = np.concatenate(all_x0_clamped)
        gt = np.concatenate(all_gt)
        diffs = np.concatenate(all_diff_raw)

        x0_changes = np.diff(x0_clamp, axis=1)
        gt_changes = np.diff(gt, axis=1)

        results[f"t={t_val}"] = {
            "x0_raw_skew": float(skew(x0_raw.flatten())),
            "x0_clamped_skew": float(skew(x0_clamp.flatten())),
            "x0_changes_skew": float(skew(x0_changes.flatten())),
            "gt_changes_skew": float(skew(gt_changes.flatten())),
            "diff_skew": float(skew(diffs.flatten())),
            "x0_raw_mean": float(np.mean(x0_raw)),
            "x0_clamped_mean": float(np.mean(x0_clamp)),
            "pct_clamped": float(np.mean(np.abs(x0_raw) > 1.0) * 100),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/skewness_diagnosis/x0_predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    ddpm_config = get_ddpm_config()
    data = np.load(ddpm_config.data_path)
    surfaces = data["surface"]

    test_dataset = VolSurfaceDataset(
        surfaces,
        ddpm_config.history_len, ddpm_config.future_len,
        start_idx=ddpm_config.test_start,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    timesteps = [1, 5, 10, 20, 50, 80, 95, 99]

    print("=" * 70)
    print("X_0 PREDICTION SKEWNESS ANALYSIS")
    print("=" * 70)

    # DDPM POC
    print("\n--- DDPM POC ---")
    ckpt = torch.load("models/backfill/ddpm_poc/baseline_uniform_epoch_50.pt",
                       weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model_config = cfg if dataclasses.is_dataclass(cfg) else DenoiserConfig(**cfg)
    ddpm_model = ConditionalDDPM(model_config)
    ddpm_model.load_state_dict(ckpt["model_state_dict"])
    ddpm_model = ddpm_model.to(device).eval()

    ddpm_results = analyze_x0_ddpm(ddpm_model, test_loader, args.max_batches, device, timesteps)

    print(f"\n{'t':>4} {'x0_raw_sk':>10} {'x0_clamp_sk':>12} {'changes_sk':>11} {'gt_ch_sk':>9} {'diff_sk':>8} {'%clamp':>7}")
    for key in sorted(ddpm_results.keys(), key=lambda x: int(x.split("=")[1])):
        r = ddpm_results[key]
        print(f"{key:>5} {r['x0_raw_skew']:>10.4f} {r['x0_clamped_skew']:>12.4f} "
              f"{r['x0_changes_skew']:>11.4f} {r['gt_changes_skew']:>9.4f} "
              f"{r['diff_skew']:>8.4f} {r['pct_clamped']:>6.1f}%")

    # Block-AR
    print("\n--- Block-AR Config B ---")
    ckpt_bar = torch.load("models/backfill/block_ar_taskprob_B/best_coverage_model.pt",
                           weights_only=False, map_location=device)
    bar_config = BlockARConfig(**ckpt_bar["config"])
    bar_model = ConditionalBlockARDDPM(bar_config)
    bar_model.load_state_dict(ckpt_bar["model_state_dict"])
    bar_model = bar_model.to(device).eval()

    bar_results = analyze_x0_block_ar(bar_model, test_loader, args.max_batches, device, timesteps)

    print(f"\n{'t':>4} {'x0_raw_sk':>10} {'x0_clamp_sk':>12} {'changes_sk':>11} {'gt_ch_sk':>9} {'diff_sk':>8} {'%clamp':>7}")
    for key in sorted(bar_results.keys(), key=lambda x: int(x.split("=")[1])):
        r = bar_results[key]
        print(f"{key:>5} {r['x0_raw_skew']:>10.4f} {r['x0_clamped_skew']:>12.4f} "
              f"{r['x0_changes_skew']:>11.4f} {r['gt_changes_skew']:>9.4f} "
              f"{r['diff_skew']:>8.4f} {r['pct_clamped']:>6.1f}%")

    # Side-by-side comparison of changes skewness
    print("\n" + "=" * 70)
    print("COMPARISON: x_0 Changes Skewness (frame-to-frame diffs)")
    print("=" * 70)
    print(f"\n{'t':>4} {'DDPM_ch_sk':>11} {'BAR_ch_sk':>10} {'Ratio':>8} {'DDPM_%clamp':>12} {'BAR_%clamp':>11}")
    for t_val in timesteps:
        key = f"t={t_val}"
        dr = ddpm_results[key]
        br = bar_results[key]
        ratio = dr['x0_changes_skew'] / br['x0_changes_skew'] if abs(br['x0_changes_skew']) > 0.001 else float('inf')
        print(f"{key:>5} {dr['x0_changes_skew']:>11.4f} {br['x0_changes_skew']:>10.4f} "
              f"{ratio:>8.2f} {dr['pct_clamped']:>11.1f}% {br['pct_clamped']:>10.1f}%")

    # Save
    all_results = {
        "ddpm_poc": ddpm_results,
        "block_ar_config_b": bar_results,
        "eval_config": {
            "max_batches": args.max_batches,
            "timesteps": timesteps,
            "ddpm_model": "models/backfill/ddpm_poc/baseline_uniform_epoch_50.pt",
            "block_ar_model": "models/backfill/block_ar_taskprob_B/best_coverage_model.pt",
        },
    }
    output_path = os.path.join(args.output_dir, "x0_prediction_skewness.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
