"""
Exp D2: Per-Cell Denoiser Noise Prediction Analysis.

Analyzes what the denoiser actually predicts per cell:
- Per-cell |epsilon_theta| (noise prediction magnitude)
- Per-cell |epsilon_theta - epsilon| (prediction error)
- Split by calm/turb regime
- Comparison to GT per-cell spread

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_percell_noise.py \
        --model_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --n_windows 500 --device cuda
"""

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig, ConditionalBlockARDDPM, denormalize_iv,
    sample_pyoco_noise,
)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)
    model_config.device = device
    model = ConditionalBlockARDDPM(model_config).to(device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, model_config


def load_test_data(config):
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
    test_start = getattr(config, 'test_start', 4540)
    dataset = VolSurfaceDataset(surfaces, config.history_len, config.future_len,
                                start_idx=test_start)
    return dataset


def compute_vol_of_vol(history_norm):
    """Compute vol_of_vol for regime splitting."""
    past_abs = denormalize_iv(history_norm)  # (B, T, 5, 5)
    mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T)
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1)  # (B,)
    return vov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--n_windows", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/block_ar/percell_noise_diagnostic")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    model, config = load_model(args.model_path, device)
    dataset = load_test_data(config)

    # Collect batches
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    H, W = config.surface_h, config.surface_w
    bs = config.block_size

    # Storage for per-cell statistics
    all_noise_pred_mag = []  # |epsilon_theta| per cell
    all_noise_true_mag = []  # |epsilon| per cell
    all_error_mag = []  # |epsilon_theta - epsilon| per cell
    all_vov = []  # vol_of_vol per window

    n_collected = 0
    timesteps_to_test = [25, 50, 75]  # early, mid, late diffusion

    print(f"Analyzing {args.n_windows} windows at timesteps {timesteps_to_test}...")

    with torch.no_grad():
        for batch in loader:
            if n_collected >= args.n_windows:
                break

            history = batch["history"].to(device)
            future = batch["future"].to(device)
            B = history.shape[0]

            # Compute vol_of_vol for regime splitting
            vov = compute_vol_of_vol(history)
            all_vov.append(vov.cpu().numpy())

            # Ensure scheduler is on device
            model._ensure_scheduler_device(device)

            # Get the first block for analysis
            target_block = future[:, :bs]  # (B, bs, 5, 5)
            past_ctx = history

            # Encode condition
            condition = model.encoder(past_ctx.reshape(B, -1, H * W))

            # Compute ratio target (vol_scaled mode)
            past_abs = denormalize_iv(past_ctx)
            target_abs = denormalize_iv(target_block)
            baseline = model._compute_baseline(past_ctx)  # (B, 1, 5, 5)
            baseline_abs = denormalize_iv(baseline)

            # Vol scale
            mean_iv = past_abs.mean(dim=(-1, -2))
            mean_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
            vol = mean_chg.std(dim=1, keepdim=True)
            vol_scale = (vol / config.global_mean_vol).clamp(
                config.vol_scale_min, config.vol_scale_max)
            vol_scale = vol_scale.pow(config.vol_scale_power)

            # Target in ratio space
            log_ratio = torch.log(target_abs / baseline_abs.clamp(min=1e-6))
            target_ratio = (log_ratio / vol_scale.unsqueeze(-1).unsqueeze(-1)).clamp(-3, 3)

            # Test at multiple timesteps
            for t_val in timesteps_to_test:
                k = torch.full((B, bs), t_val, device=device, dtype=torch.long)

                # Sample noise
                noise = sample_pyoco_noise(target_ratio.shape, config.noise_rho, device)

                # Forward diffusion
                noisy, _ = model.scheduler.q_sample_per_frame(target_ratio, k, noise)

                # Denoiser prediction
                noisy_flat = noisy.reshape(B, bs, -1)
                positions = torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)

                noise_pred = model.denoiser(
                    noisy_flat, condition, positions, k.float(),
                )
                if isinstance(noise_pred, tuple):
                    noise_pred = noise_pred[0]

                # Reshape to spatial
                noise_pred_spatial = noise_pred.reshape(B, bs, H, W)
                noise_spatial = noise.reshape(B, bs, H, W)

                # Per-cell magnitudes (mean over time dimension)
                pred_mag = noise_pred_spatial.abs().mean(dim=1)  # (B, H, W)
                true_mag = noise_spatial.abs().mean(dim=1)  # (B, H, W)
                error_mag = (noise_pred_spatial - noise_spatial).abs().mean(dim=1)  # (B, H, W)

                all_noise_pred_mag.append(pred_mag.cpu().numpy())
                all_noise_true_mag.append(true_mag.cpu().numpy())
                all_error_mag.append(error_mag.cpu().numpy())

            n_collected += B

    # Concatenate
    vov = np.concatenate(all_vov)[:args.n_windows]
    n_t = len(timesteps_to_test)
    # Each timestep contributes one entry per window, reshape accordingly
    pred_mag = np.concatenate(all_noise_pred_mag)  # (n_windows * n_t, H, W)
    true_mag = np.concatenate(all_noise_true_mag)
    error_mag = np.concatenate(all_error_mag)

    # Reshape: (n_windows, n_timesteps, H, W)
    n = min(args.n_windows, len(vov))
    pred_mag = pred_mag[:n * n_t].reshape(n, n_t, H, W)
    true_mag = true_mag[:n * n_t].reshape(n, n_t, H, W)
    error_mag = error_mag[:n * n_t].reshape(n, n_t, H, W)

    # Average across timesteps for summary stats
    pred_mag_avg = pred_mag.mean(axis=1)  # (n, H, W)
    error_mag_avg = error_mag.mean(axis=1)

    # Regime split (Q20/Q80)
    vov_q20 = np.percentile(vov[:n], 20)
    vov_q80 = np.percentile(vov[:n], 80)
    calm_mask = vov[:n] < vov_q20
    turb_mask = vov[:n] > vov_q80

    # Per-cell statistics
    print("\n" + "=" * 60)
    print("PER-CELL NOISE PREDICTION ANALYSIS")
    print("=" * 60)

    # 1. Per-cell noise prediction magnitude
    overall_mean = pred_mag_avg.mean()
    cell_pred_ratio = pred_mag_avg.mean(axis=0) / overall_mean
    print("\n--- Per-cell |epsilon_theta| / mean(|epsilon_theta|) ---")
    print("(>1 = denoiser predicts larger noise, <1 = smaller noise)")
    for r in range(H):
        row = "  ".join(f"{cell_pred_ratio[r, c]:5.3f}" for c in range(W))
        print(f"  {row}")

    # 2. Per-cell prediction error
    overall_err = error_mag_avg.mean()
    cell_err_ratio = error_mag_avg.mean(axis=0) / overall_err
    print("\n--- Per-cell |epsilon_theta - epsilon| / mean (prediction error ratio) ---")
    print("(>1 = harder to predict, <1 = easier)")
    for r in range(H):
        row = "  ".join(f"{cell_err_ratio[r, c]:5.3f}" for c in range(W))
        print(f"  {row}")

    # 3. Per-cell noise prediction accuracy (1 - |error|/|true|)
    cell_accuracy = 1.0 - (error_mag_avg.mean(axis=0) / true_mag.mean(axis=(0, 1)).clip(min=1e-6))
    print("\n--- Per-cell noise prediction accuracy (1 - |error|/|true|) ---")
    for r in range(H):
        row = "  ".join(f"{cell_accuracy[r, c]:5.3f}" for c in range(W))
        print(f"  {row}")

    # 4. Calm vs Turb comparison
    print("\n--- Calm vs Turb noise prediction magnitude ---")
    calm_pred = pred_mag_avg[calm_mask].mean(axis=0)
    turb_pred = pred_mag_avg[turb_mask].mean(axis=0)
    turb_calm_ratio = turb_pred / calm_pred.clip(min=1e-6)
    print("Turb/Calm |epsilon_theta| ratio per cell:")
    for r in range(H):
        row = "  ".join(f"{turb_calm_ratio[r, c]:5.3f}" for c in range(W))
        print(f"  {row}")
    print(f"  Mean turb/calm ratio: {turb_calm_ratio.mean():.3f}")

    # 5. Calm vs Turb prediction error
    print("\n--- Calm vs Turb prediction error ---")
    calm_err = error_mag_avg[calm_mask].mean(axis=0)
    turb_err = error_mag_avg[turb_mask].mean(axis=0)
    turb_calm_err = turb_err / calm_err.clip(min=1e-6)
    print("Turb/Calm |error| ratio per cell:")
    for r in range(H):
        row = "  ".join(f"{turb_calm_err[r, c]:5.3f}" for c in range(W))
        print(f"  {row}")
    print(f"  Mean turb/calm error ratio: {turb_calm_err.mean():.3f}")

    # 6. Per-timestep analysis
    print("\n--- Per-timestep noise prediction magnitude (averaged across cells) ---")
    for ti, t_val in enumerate(timesteps_to_test):
        mean_pred = pred_mag[:, ti].mean()
        mean_err = error_mag[:, ti].mean()
        mean_true = true_mag[:, ti].mean()
        print(f"  t={t_val}: |pred|={mean_pred:.4f}, |true|={mean_true:.4f}, "
              f"|error|={mean_err:.4f}, accuracy={1-mean_err/mean_true:.3f}")

    # 7. GT per-cell spread for comparison
    print("\n--- Reference: GT per-cell daily change std (from training data) ---")
    gmcv = np.array([
        [0.1560, 0.0591, 0.0208, 0.0288, 0.1044],
        [0.0547, 0.0281, 0.0137, 0.0079, 0.0616],
        [0.0230, 0.0147, 0.0085, 0.0056, 0.0265],
        [0.0079, 0.0064, 0.0049, 0.0044, 0.0047],
        [0.0058, 0.0048, 0.0046, 0.0046, 0.0052],
    ])
    gmcv_ratio = gmcv / gmcv.mean()
    print("GT cell_std / mean(cell_std) ratio:")
    for r in range(H):
        row = "  ".join(f"{gmcv_ratio[r, c]:5.3f}" for c in range(W))
        print(f"  {row}")

    # 8. Correlation: does noise prediction error correlate with GT cell volatility?
    from scipy.stats import spearmanr
    cell_err_flat = cell_err_ratio.flatten()
    gmcv_flat = gmcv_ratio.flatten()
    rho, p = spearmanr(cell_err_flat, gmcv_flat)
    print(f"\nSpearman(prediction_error_ratio, GT_cell_volatility): {rho:.3f} (p={p:.3e})")

    rho2, p2 = spearmanr(cell_pred_ratio.flatten(), gmcv_flat)
    print(f"Spearman(prediction_magnitude_ratio, GT_cell_volatility): {rho2:.3f} (p={p2:.3e})")

    # Save results
    results = {
        "n_windows": n,
        "timesteps": timesteps_to_test,
        "cell_pred_ratio": cell_pred_ratio.tolist(),
        "cell_err_ratio": cell_err_ratio.tolist(),
        "cell_accuracy": cell_accuracy.tolist(),
        "turb_calm_pred_ratio": turb_calm_ratio.tolist(),
        "turb_calm_err_ratio": turb_calm_err.tolist(),
        "spearman_err_vs_gt_vol": {"rho": float(rho), "p": float(p)},
        "spearman_pred_vs_gt_vol": {"rho": float(rho2), "p": float(p2)},
        "vov_q20": float(vov_q20),
        "vov_q80": float(vov_q80),
    }

    out_path = f"{args.output_dir}/percell_noise_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
