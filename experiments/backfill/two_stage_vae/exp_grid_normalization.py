"""
Experiment: Per-Grid-Point Normalization

Goal: Address variance heterogeneity across grid points by standardizing each point.

Problem:
- GT log-return std varies 27x across grid (0.03 to 0.83)
- MSE loss treats all points equally, favoring easy (low-var) points
- Model "gives up" on high-variance points

Solution:
- Normalize each grid point by its historical mean/std before training
- All points become std=1, equally weighted in MSE
- Denormalize predictions for evaluation

Experiments:
- Baseline: No normalization (using exp_c config: latent_dim=8, z_dropout=0.3)
- Normalized: Per-grid-point standardization

Metrics:
- z_logvar mean and variance
- Decoder gain
- CI violations per grid point
- CI violations for center region (0.95-1.05, 60d-180d)

Usage:
    python experiments/backfill/two_stage_vae/exp_grid_normalization.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage
from config.two_stage_config import TwoStageConfig


def to_log_returns(surfaces):
    """Transform IV surfaces to log-returns."""
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]
    return log_returns, log_surfaces


def create_sequences(data, seq_len):
    """Create overlapping sequences from data."""
    n_sequences = len(data) - seq_len + 1
    sequences = torch.stack([data[i:i+seq_len] for i in range(n_sequences)])
    return sequences


def train_epoch(model, train_sequences, optimizer, batch_size, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    z_logvar_sum = 0
    n_batches = 0

    indices = torch.randperm(len(train_sequences))

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = train_sequences[batch_idx].to(device)

        losses = model.train_step_autoencoder({"surface": batch}, optimizer)

        total_loss += losses["loss"].item()
        total_recon += losses["re_surface"].item()
        total_kl += losses["kl_loss"].item()
        z_logvar_sum += losses.get("z_logvar_mean", 0)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "z_logvar_mean": z_logvar_sum / n_batches,
    }


def measure_z_stats(model, val_sequences, device, n_samples=20):
    """Measure z statistics."""
    model.eval()

    z_logvars = []
    z_vars = []

    with torch.no_grad():
        for i in range(min(n_samples, len(val_sequences))):
            batch = val_sequences[i:i+1].to(device)

            # Get z stats from encoder
            encoder_input = {"surface": batch}
            z_mean, z_logvar, z = model.encoder(encoder_input)

            z_logvars.append(z_logvar.mean().item())
            z_vars.append(torch.exp(z_logvar).mean().item())

    return {
        "z_logvar_mean": np.mean(z_logvars),
        "z_logvar_std": np.std(z_logvars),
        "z_variance_mean": np.mean(z_vars),
    }


def measure_decoder_gain(model, val_sequences, device, config, num_contexts=30, num_samples=30):
    """Measure decoder gain: output_variance / z_variance."""
    model.eval()

    z_vars = []
    out_vars = []

    with torch.no_grad():
        for i in range(min(num_contexts, len(val_sequences))):
            batch = val_sequences[i:i+1].to(device)

            # Get context embedding and z distribution
            ctx_emb = model.ctx_encoder({"surface": batch})
            z_mean, z_logvar, _ = model.encoder({"surface": batch})

            # Sample multiple z and decode
            samples = []
            z_samples = []

            for _ in range(num_samples):
                eps = torch.randn_like(z_logvar)
                z = z_mean + torch.exp(0.5 * z_logvar) * eps
                z_samples.append(z.cpu())

                decoded = model.decoder(ctx_emb, z)
                samples.append(decoded.cpu())

            samples_tensor = torch.stack(samples).squeeze()
            z_tensor = torch.stack(z_samples).squeeze()

            # Variance at last position
            out_var = samples_tensor[:, -1, :, :].var(dim=0).mean().item()
            z_var = z_tensor[:, -1, :].var(dim=0).mean().item()

            out_vars.append(out_var)
            z_vars.append(z_var)

    mean_out_var = np.mean(out_vars)
    mean_z_var = np.mean(z_vars)
    decoder_gain = mean_out_var / (mean_z_var + 1e-10)

    return decoder_gain, mean_z_var, mean_out_var


def measure_ci_violations_per_grid(model, log_returns, device, config,
                                    grid_mean=None, grid_std=None,
                                    horizons=[1, 7, 14, 30], n_samples=100, n_test=50):
    """
    Measure CI violations per grid point in log-return space.

    If grid_mean/grid_std provided, denormalizes predictions before evaluation.
    """
    model.eval()
    context_len = config["context_len"]

    # Grid labels
    moneyness = ['0.90', '0.95', '1.00', '1.05', '1.10']
    maturity = ['30d', '60d', '90d', '180d', '365d']

    results = {}

    for H in horizons:
        seq_len = context_len + H

        all_violations = []  # (n_test, 5, 5)

        for start_idx in range(0, len(log_returns) - seq_len, max(1, (len(log_returns) - seq_len) // n_test)):
            if len(all_violations) >= n_test:
                break

            gt_log_seq = log_returns[start_idx:start_idx + seq_len]
            gt_cumsum = gt_log_seq[context_len:context_len + H].sum(axis=0)  # (5, 5)

            # Normalize input if stats provided
            if grid_mean is not None and grid_std is not None:
                input_seq = (gt_log_seq - grid_mean) / grid_std
            else:
                input_seq = gt_log_seq

            batch = torch.tensor(input_seq[None], dtype=torch.float32).to(device)

            # Sample predictions
            samples_cumsum = []
            with torch.no_grad():
                for _ in range(n_samples):
                    output = model({"surface": batch}, return_full_sequence=True)
                    recon = output[0].cpu().numpy()[0]

                    # Denormalize if needed
                    if grid_mean is not None and grid_std is not None:
                        recon = recon * grid_std + grid_mean

                    pred_cumsum = recon[context_len:context_len + H].sum(axis=0)
                    samples_cumsum.append(pred_cumsum)

            samples_cumsum = np.array(samples_cumsum)
            p05 = np.percentile(samples_cumsum, 5, axis=0)
            p95 = np.percentile(samples_cumsum, 95, axis=0)

            in_ci = (gt_cumsum >= p05) & (gt_cumsum <= p95)
            all_violations.append(~in_ci)

        all_violations = np.array(all_violations)
        violation_rate = all_violations.mean(axis=0) * 100  # (5, 5)

        # Center region (0.95-1.05, 60d-180d) = indices [1:4, 1:4]
        center_violations = violation_rate[1:4, 1:4].mean()
        overall_violations = violation_rate.mean()

        results[H] = {
            "per_grid": violation_rate.tolist(),
            "center": center_violations,
            "overall": overall_violations,
        }

    return results


def train_and_evaluate(exp_name, config, train_sequences, val_sequences,
                       log_returns, output_dir, device,
                       grid_mean=None, grid_std=None, n_epochs=100):
    """Train a model and evaluate it."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")

    batch_size = 256
    learning_rate = 1e-4

    # Build model
    print(f"\nBuilding model...")
    model = CVAETwoStage(config)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  z_dropout: {config.get('z_dropout', 0.0)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    print(f"\nTraining for {n_epochs} epochs...")
    best_val_loss = float('inf')
    history = {"train": [], "val": [], "z_stats": []}

    for epoch in tqdm(range(n_epochs), desc=exp_name):
        train_metrics = train_epoch(model, train_sequences, optimizer, batch_size, device)

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for i in range(0, len(val_sequences), batch_size):
                batch = val_sequences[i:i+batch_size].to(device)
                losses = model.test_step({"surface": batch})
                val_loss += losses["loss"].item()
                n_val += 1
        val_loss /= n_val

        history["train"].append(train_metrics)
        history["val"].append({"loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        # Periodic z stats measurement
        if (epoch + 1) % 20 == 0:
            z_stats = measure_z_stats(model, val_sequences, device)
            history["z_stats"].append({"epoch": epoch + 1, **z_stats})
            print(f"\n  Epoch {epoch+1}: val_loss={val_loss:.4f}, z_logvar={z_stats['z_logvar_mean']:.3f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    print(f"\nEvaluating...")

    # z stats
    z_stats = measure_z_stats(model, val_sequences, device)
    print(f"  z_logvar mean: {z_stats['z_logvar_mean']:.4f}")
    print(f"  z_variance mean: {z_stats['z_variance_mean']:.4f}")

    # Decoder gain
    decoder_gain, z_var, out_var = measure_decoder_gain(model, val_sequences, device, config)
    print(f"  Decoder Gain: {decoder_gain:.2e}")

    # CI violations per grid
    ci_results = measure_ci_violations_per_grid(
        model, log_returns, device, config,
        grid_mean=grid_mean, grid_std=grid_std
    )

    print(f"\n  CI Violations (Log-Return Space):")
    for H in [1, 7, 14, 30]:
        r = ci_results[H]
        print(f"    H={H:2d}: Center={r['center']:.1f}%, Overall={r['overall']:.1f}%")

    # Save results
    results = {
        "exp_name": exp_name,
        "best_val_loss": best_val_loss,
        "z_stats": z_stats,
        "decoder_gain": decoder_gain,
        "z_variance": z_var,
        "output_variance": out_var,
        "ci_results": ci_results,
    }

    # Save checkpoint
    checkpoint_path = output_dir / f"{exp_name}_best.pt"
    torch.save({
        "model_config": config,
        "model_state_dict": best_state,
        "results": results,
        "history": history,
        "grid_mean": grid_mean,
        "grid_std": grid_std,
    }, checkpoint_path)
    print(f"\n  Saved: {checkpoint_path}")

    return results, model


def main():
    print("="*70)
    print("PER-GRID-POINT NORMALIZATION EXPERIMENT")
    print("="*70)
    print()
    print("Goal: Address variance heterogeneity by standardizing each grid point")
    print()

    # Output directory
    output_dir = Path("models/backfill/two_stage/grid_normalization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)
    print(f"  Surfaces: {surfaces.shape}")
    print(f"  Log-returns: {log_returns.shape}")

    # Train/val split (same as z_dropout experiment)
    train_log_returns = log_returns[:4000]
    val_log_returns = log_returns[4000:]

    # Compute normalization stats from TRAINING data only
    grid_mean = train_log_returns.mean(axis=0)  # (5, 5)
    grid_std = train_log_returns.std(axis=0)    # (5, 5)

    print(f"\nNormalization stats (from training data):")
    print(f"  Grid mean range: [{grid_mean.min():.6f}, {grid_mean.max():.6f}]")
    print(f"  Grid std range: [{grid_std.min():.4f}, {grid_std.max():.4f}]")

    # Create sequences
    config = TwoStageConfig.get_model_config()
    # Use exp_c config: latent_dim=8, z_dropout=0.3
    config["latent_dim"] = 8
    config["z_dropout"] = 0.3

    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon

    device = config["device"]

    print(f"\nConfig: latent_dim={config['latent_dim']}, z_dropout={config['z_dropout']}")
    print(f"Sequence length: {seq_len} (context={context_len} + horizon={horizon})")

    # Create normalized and unnormalized sequences
    print("\nCreating sequences...")

    # Unnormalized
    train_tensor = torch.tensor(train_log_returns, dtype=torch.float32)
    train_sequences = create_sequences(train_tensor, seq_len)

    val_tensor = torch.tensor(val_log_returns, dtype=torch.float32)
    val_sequences = create_sequences(val_tensor, seq_len)

    # Normalized
    train_normalized = (train_log_returns - grid_mean) / grid_std
    val_normalized = (val_log_returns - grid_mean) / grid_std

    train_norm_tensor = torch.tensor(train_normalized, dtype=torch.float32)
    train_norm_sequences = create_sequences(train_norm_tensor, seq_len)

    val_norm_tensor = torch.tensor(val_normalized, dtype=torch.float32)
    val_norm_sequences = create_sequences(val_norm_tensor, seq_len)

    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences: {len(val_sequences)}")

    # Run experiments
    all_results = {}

    # Experiment 1: Baseline (no normalization)
    results_baseline, _ = train_and_evaluate(
        exp_name="baseline_no_norm",
        config=config.copy(),
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        log_returns=log_returns,
        output_dir=output_dir,
        device=device,
        grid_mean=None,
        grid_std=None,
        n_epochs=100,
    )
    all_results["baseline_no_norm"] = results_baseline

    # Experiment 2: With per-grid-point normalization
    results_norm, _ = train_and_evaluate(
        exp_name="with_normalization",
        config=config.copy(),
        train_sequences=train_norm_sequences,
        val_sequences=val_norm_sequences,
        log_returns=log_returns,
        output_dir=output_dir,
        device=device,
        grid_mean=grid_mean,
        grid_std=grid_std,
        n_epochs=100,
    )
    all_results["with_normalization"] = results_norm

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n| Experiment | z_logvar | Decoder Gain | H=1 Center | H=30 Center |")
    print("|------------|----------|--------------|------------|-------------|")
    for name, res in all_results.items():
        z_lv = res["z_stats"]["z_logvar_mean"]
        dg = res["decoder_gain"]
        h1_c = res["ci_results"][1]["center"]
        h30_c = res["ci_results"][30]["center"]
        print(f"| {name:18s} | {z_lv:8.3f} | {dg:12.2e} | {h1_c:10.1f}% | {h30_c:11.1f}% |")

    # Detailed per-grid comparison for H=1
    print("\n" + "="*70)
    print("PER-GRID-POINT VIOLATIONS AT H=1")
    print("="*70)

    moneyness = ['0.90', '0.95', '1.00', '1.05', '1.10']
    maturity = ['30d', '60d', '90d', '180d', '365d']

    for name, res in all_results.items():
        print(f"\n{name}:")
        grid_viol = np.array(res["ci_results"][1]["per_grid"])
        print(f"{'':>6s}  " + "  ".join([f"{m:>5s}" for m in maturity]))
        for i, m in enumerate(moneyness):
            row = "  ".join([f"{grid_viol[i,j]:5.1f}" for j in range(5)])
            print(f"{m:>6s}  {row}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    baseline = all_results["baseline_no_norm"]
    normed = all_results["with_normalization"]

    print(f"\nz_logvar change: {baseline['z_stats']['z_logvar_mean']:.3f} -> {normed['z_stats']['z_logvar_mean']:.3f}")

    dg_change = normed["decoder_gain"] / baseline["decoder_gain"]
    print(f"Decoder gain change: {dg_change:.2f}x")

    h1_center_change = normed["ci_results"][1]["center"] - baseline["ci_results"][1]["center"]
    h30_center_change = normed["ci_results"][30]["center"] - baseline["ci_results"][30]["center"]
    print(f"H=1 center violations change: {h1_center_change:+.1f}%")
    print(f"H=30 center violations change: {h30_center_change:+.1f}%")

    # Success criteria check
    print("\n" + "-"*70)
    print("SUCCESS CRITERIA CHECK:")
    criteria = [
        (normed["ci_results"][1]["center"] < 25, f"H=1 center < 25%: {normed['ci_results'][1]['center']:.1f}%"),
        (normed["z_stats"]["z_logvar_mean"] > -1.0, f"z_logvar > -1.0: {normed['z_stats']['z_logvar_mean']:.3f}"),
        (normed["decoder_gain"] > 1e-4, f"Decoder gain > 1e-4: {normed['decoder_gain']:.2e}"),
    ]

    all_passed = True
    for passed, desc in criteria:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nALL CRITERIA PASSED!")
    else:
        print("\nSome criteria not met.")

    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
