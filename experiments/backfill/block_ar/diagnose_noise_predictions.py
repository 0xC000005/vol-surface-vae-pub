"""Diagnose noise prediction skewness — are the learned noise predictions asymmetric?

For each model, compute:
1. Forward pass on test data at various timesteps
2. Compute (predicted_noise - actual_noise) residuals
3. Measure skewness of residuals and predictions

If the denoiser learns asymmetric noise predictions, the skew should appear here.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_noise_predictions.py \
        --output_dir results/skewness_diagnosis/noise_predictions
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig, denormalize_iv
from diffusion.block_ar import ConditionalBlockARDDPM, BlockARConfig
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config as get_ddpm_config
from experiments.backfill.block_ar.config_block_ar import get_default_config as get_block_ar_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def analyze_noise_predictions_ddpm(model, test_loader, max_batches, device, timesteps=[10, 30, 50, 70, 90]):
    """Analyze noise prediction skewness for DDPM POC."""
    results = {}

    for t_val in timesteps:
        all_noise_pred = []
        all_noise_actual = []
        all_residuals = []

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future = batch["future"].to(device)  # normalized [-1, 1]

            B = future.shape[0]

            # Add noise at timestep t
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(future)
            sqrt_ab = model.scheduler.sqrt_alpha_bar[t_val].to(device)
            sqrt_1mab = model.scheduler.sqrt_one_minus_alpha_bar[t_val].to(device)
            noisy_future = sqrt_ab * future + sqrt_1mab * noise

            # Predict noise
            with torch.no_grad():
                noise_pred = model.denoiser(noisy_future, t, history)

            residual = noise_pred - noise

            all_noise_pred.append(noise_pred.cpu().numpy())
            all_noise_actual.append(noise.cpu().numpy())
            all_residuals.append(residual.cpu().numpy())

        preds = np.concatenate(all_noise_pred)
        actual = np.concatenate(all_noise_actual)
        resids = np.concatenate(all_residuals)

        results[f"t={t_val}"] = {
            "noise_pred_skew": float(skew(preds.flatten())),
            "noise_actual_skew": float(skew(actual.flatten())),
            "residual_skew": float(skew(resids.flatten())),
            "noise_pred_mean": float(np.mean(preds)),
            "residual_mean": float(np.mean(resids)),
            "noise_pred_std": float(np.std(preds)),
            "residual_std": float(np.std(resids)),
            "n_values": int(preds.size),
        }

    return results


def analyze_noise_predictions_block_ar(model, test_loader, max_batches, device, timesteps=[10, 30, 50, 70, 90]):
    """Analyze noise prediction skewness for Block-AR."""
    results = {}

    for t_val in timesteps:
        all_noise_pred = []
        all_noise_actual = []
        all_residuals = []

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future = batch["future"].to(device)  # normalized [-1, 1]

            B = future.shape[0]
            T = model.config.block_size  # 10 frames per block
            H, W = model.config.surface_h, model.config.surface_w

            # Use just the first block (frames 0-9)
            future_block = future[:, :T]  # (B, T, H, W)

            # Encode history
            with torch.no_grad():
                condition = model.encoder(history)  # (B, bottleneck_dim)

            # Add noise at timestep t (uniform across block)
            t_block = torch.full((B, T), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(future_block)

            # Use scheduler to add noise
            sqrt_ab = model.scheduler.sqrt_alpha_bar[t_val]
            sqrt_1mab = model.scheduler.sqrt_one_minus_alpha_bar[t_val]
            noisy_block = sqrt_ab * future_block + sqrt_1mab * noise

            # Flatten spatial for denoiser
            noisy_flat = noisy_block.reshape(B, T, H * W)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

            # Predict noise
            with torch.no_grad():
                noise_pred_flat = model.denoiser(noisy_flat, condition, positions, t_block)

            noise_pred = noise_pred_flat.reshape(B, T, H, W)
            residual = noise_pred - noise

            all_noise_pred.append(noise_pred.cpu().numpy())
            all_noise_actual.append(noise.cpu().numpy())
            all_residuals.append(residual.cpu().numpy())

        preds = np.concatenate(all_noise_pred)
        actual = np.concatenate(all_noise_actual)
        resids = np.concatenate(all_residuals)

        results[f"t={t_val}"] = {
            "noise_pred_skew": float(skew(preds.flatten())),
            "noise_actual_skew": float(skew(actual.flatten())),
            "residual_skew": float(skew(resids.flatten())),
            "noise_pred_mean": float(np.mean(preds)),
            "residual_mean": float(np.mean(resids)),
            "noise_pred_std": float(np.std(preds)),
            "residual_std": float(np.std(resids)),
            "n_values": int(preds.size),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/skewness_diagnosis/noise_predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # Load test data (shared between both models)
    ddpm_config = get_ddpm_config()
    data = np.load(ddpm_config.data_path)
    surfaces = data["surface"]

    # Test dataset
    test_dataset = VolSurfaceDataset(
        surfaces,
        ddpm_config.history_len, ddpm_config.future_len,
        start_idx=ddpm_config.test_start,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    timesteps = [5, 20, 50, 80, 95]

    print("=" * 60)
    print("NOISE PREDICTION SKEWNESS ANALYSIS")
    print("=" * 60)

    # --- DDPM POC ---
    print("\n--- DDPM POC (baseline_uniform_epoch_50) ---")
    import dataclasses
    ckpt = torch.load("models/backfill/ddpm_poc/baseline_uniform_epoch_50.pt",
                       weights_only=False, map_location=device)
    cfg = ckpt["config"]
    if dataclasses.is_dataclass(cfg):
        model_config = cfg
    else:
        model_config = DenoiserConfig(**cfg)
    ddpm_model = ConditionalDDPM(model_config)
    ddpm_model.load_state_dict(ckpt["model_state_dict"])
    ddpm_model = ddpm_model.to(device).eval()

    ddpm_results = analyze_noise_predictions_ddpm(
        ddpm_model, test_loader, args.max_batches, device, timesteps
    )

    print(f"\n{'Timestep':>10} {'PredSkew':>10} {'ResidSkew':>10} {'PredMean':>10} {'ResidMean':>10}")
    for key in sorted(ddpm_results.keys(), key=lambda x: int(x.split("=")[1])):
        r = ddpm_results[key]
        print(f"{key:>10} {r['noise_pred_skew']:>10.4f} {r['residual_skew']:>10.4f} "
              f"{r['noise_pred_mean']:>10.6f} {r['residual_mean']:>10.6f}")

    # --- Block-AR Config B ---
    print("\n--- Block-AR Config B (best_coverage_model) ---")
    ckpt_bar = torch.load("models/backfill/block_ar_taskprob_B/best_coverage_model.pt",
                           weights_only=False, map_location=device)
    bar_config = BlockARConfig(**ckpt_bar["config"])
    bar_model = ConditionalBlockARDDPM(bar_config)
    bar_model.load_state_dict(ckpt_bar["model_state_dict"])
    bar_model = bar_model.to(device).eval()

    bar_results = analyze_noise_predictions_block_ar(
        bar_model, test_loader, args.max_batches, device, timesteps
    )

    print(f"\n{'Timestep':>10} {'PredSkew':>10} {'ResidSkew':>10} {'PredMean':>10} {'ResidMean':>10}")
    for key in sorted(bar_results.keys(), key=lambda x: int(x.split("=")[1])):
        r = bar_results[key]
        print(f"{key:>10} {r['noise_pred_skew']:>10.4f} {r['residual_skew']:>10.4f} "
              f"{r['noise_pred_mean']:>10.6f} {r['residual_mean']:>10.6f}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON: Noise Prediction Skewness")
    print("=" * 60)
    print(f"\n{'Timestep':>10} {'DDPM_pred':>10} {'BAR_pred':>10} {'DDPM_resid':>10} {'BAR_resid':>10}")
    for t in timesteps:
        key = f"t={t}"
        dr = ddpm_results[key]
        br = bar_results[key]
        print(f"{key:>10} {dr['noise_pred_skew']:>10.4f} {br['noise_pred_skew']:>10.4f} "
              f"{dr['residual_skew']:>10.4f} {br['residual_skew']:>10.4f}")

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
    output_path = os.path.join(args.output_dir, "noise_prediction_skewness.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
