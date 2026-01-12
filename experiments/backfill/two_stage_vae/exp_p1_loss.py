"""
Experiment: P1 Loss for CVAETwoStage

Goal: Force decoder to produce output variance when z varies.

P1 loss = -log(CV) where CV = std/|mean| (coefficient of variation)
This forces the decoder to respond to z variation without targeting
a specific variance level (avoids conditional/unconditional problem).

Experiments:
- Baseline: p1_weight=0.0 (no P1 loss)
- P1 low: p1_weight=0.01
- P1 high: p1_weight=0.1

Config base: latent_dim=8, z_dropout=0.3 (exp_c config)

Metrics:
- Decoder gain
- z_logvar mean
- CI violations per grid point (log-return space)
- Reconstruction MSE

Usage:
    python experiments/backfill/two_stage_vae/exp_p1_loss.py
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


def train_epoch(model, train_sequences, optimizer, batch_size, device, p1_weight=0.0):
    """Train for one epoch with optional P1 loss."""
    model.train()
    total_loss = 0
    total_mse = 0
    total_kl = 0
    total_p1 = 0
    total_cv = 0
    z_logvar_sum = 0
    n_batches = 0

    indices = torch.randperm(len(train_sequences))

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = train_sequences[batch_idx].to(device)

        if p1_weight > 0:
            losses = model.train_step_with_p1_loss(
                {"surface": batch}, optimizer, p1_weight=p1_weight, n_z_samples=5
            )
            total_p1 += losses["p1_loss"].item() if torch.is_tensor(losses["p1_loss"]) else losses["p1_loss"]
            total_cv += losses["cv"]
        else:
            losses = model.train_step_autoencoder({"surface": batch}, optimizer)

        total_loss += losses["loss"].item()
        total_mse += losses["re_surface"].item()
        total_kl += losses["kl_loss"].item()
        z_logvar_sum += losses.get("z_logvar_mean", 0)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "kl": total_kl / n_batches,
        "p1": total_p1 / n_batches if p1_weight > 0 else 0,
        "cv": total_cv / n_batches if p1_weight > 0 else 0,
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

            ctx_emb = model.ctx_encoder({"surface": batch})
            z_mean, z_logvar, _ = model.encoder({"surface": batch})

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

            out_var = samples_tensor[:, -1, :, :].var(dim=0).mean().item()
            z_var = z_tensor[:, -1, :].var(dim=0).mean().item()

            out_vars.append(out_var)
            z_vars.append(z_var)

    mean_out_var = np.mean(out_vars)
    mean_z_var = np.mean(z_vars)
    decoder_gain = mean_out_var / (mean_z_var + 1e-10)

    return decoder_gain, mean_z_var, mean_out_var


def measure_ci_violations_per_grid(model, log_returns, device, config,
                                    horizons=[1, 7, 14, 30], n_samples=100, n_test=50):
    """Measure CI violations per grid point in log-return space."""
    model.eval()
    context_len = config["context_len"]

    moneyness = ['0.90', '0.95', '1.00', '1.05', '1.10']
    maturity = ['30d', '60d', '90d', '180d', '365d']

    results = {}

    for H in horizons:
        seq_len = context_len + H

        all_violations = []

        for start_idx in range(0, len(log_returns) - seq_len, max(1, (len(log_returns) - seq_len) // n_test)):
            if len(all_violations) >= n_test:
                break

            gt_log_seq = log_returns[start_idx:start_idx + seq_len]
            gt_cumsum = gt_log_seq[context_len:context_len + H].sum(axis=0)

            batch = torch.tensor(gt_log_seq[None], dtype=torch.float32).to(device)

            samples_cumsum = []
            with torch.no_grad():
                for _ in range(n_samples):
                    output = model({"surface": batch}, return_full_sequence=True)
                    recon = output[0].cpu().numpy()[0]
                    pred_cumsum = recon[context_len:context_len + H].sum(axis=0)
                    samples_cumsum.append(pred_cumsum)

            samples_cumsum = np.array(samples_cumsum)
            p05 = np.percentile(samples_cumsum, 5, axis=0)
            p95 = np.percentile(samples_cumsum, 95, axis=0)

            in_ci = (gt_cumsum >= p05) & (gt_cumsum <= p95)
            all_violations.append(~in_ci)

        all_violations = np.array(all_violations)
        violation_rate = all_violations.mean(axis=0) * 100

        center_violations = violation_rate[1:4, 1:4].mean()
        overall_violations = violation_rate.mean()

        results[H] = {
            "per_grid": violation_rate.tolist(),
            "center": center_violations,
            "overall": overall_violations,
        }

    return results


def train_and_evaluate(exp_name, config, train_sequences, val_sequences,
                       log_returns, output_dir, device, p1_weight=0.0, n_epochs=100):
    """Train a model and evaluate it."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")

    batch_size = 256
    learning_rate = 1e-4

    print(f"\nBuilding model...")
    model = CVAETwoStage(config)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  z_dropout: {config.get('z_dropout', 0.0)}")
    print(f"  p1_weight: {p1_weight}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining for {n_epochs} epochs...")
    best_val_loss = float('inf')
    best_state = model.state_dict().copy()  # Initialize with starting weights
    history = {"train": [], "val": [], "z_stats": []}

    for epoch in tqdm(range(n_epochs), desc=exp_name):
        train_metrics = train_epoch(model, train_sequences, optimizer, batch_size, device, p1_weight)

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

        # Track best model (skip NaN)
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            z_stats = measure_z_stats(model, val_sequences, device)
            history["z_stats"].append({"epoch": epoch + 1, **z_stats})
            print(f"\n  Epoch {epoch+1}: val_loss={val_loss:.4f}, z_logvar={z_stats['z_logvar_mean']:.3f}, "
                  f"mse={train_metrics['mse']:.4f}, p1={train_metrics['p1']:.4f}, cv={train_metrics['cv']:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    print(f"\nEvaluating...")

    z_stats = measure_z_stats(model, val_sequences, device)
    print(f"  z_logvar mean: {z_stats['z_logvar_mean']:.4f}")
    print(f"  z_variance mean: {z_stats['z_variance_mean']:.4f}")

    decoder_gain, z_var, out_var = measure_decoder_gain(model, val_sequences, device, config)
    print(f"  Decoder Gain: {decoder_gain:.2e}")
    print(f"  Output Variance: {out_var:.6f}")

    ci_results = measure_ci_violations_per_grid(model, log_returns, device, config)

    print(f"\n  CI Violations (Log-Return Space):")
    for H in [1, 7, 14, 30]:
        r = ci_results[H]
        print(f"    H={H:2d}: Center={r['center']:.1f}%, Overall={r['overall']:.1f}%")

    # Save results
    results = {
        "exp_name": exp_name,
        "p1_weight": p1_weight,
        "best_val_loss": best_val_loss,
        "final_mse": history["train"][-1]["mse"],
        "z_stats": z_stats,
        "decoder_gain": decoder_gain,
        "z_variance": z_var,
        "output_variance": out_var,
        "ci_results": ci_results,
    }

    checkpoint_path = output_dir / f"{exp_name}_best.pt"
    torch.save({
        "model_config": config,
        "model_state_dict": best_state,
        "results": results,
        "history": history,
    }, checkpoint_path)
    print(f"\n  Saved: {checkpoint_path}")

    return results, model


def main():
    print("="*70)
    print("P1 LOSS EXPERIMENT FOR CVAETwoStage")
    print("="*70)
    print()
    print("Goal: Force decoder to produce output variance when z varies")
    print("Method: P1 loss = -log(CV) where CV = std/|mean|")
    print()

    output_dir = Path("models/backfill/two_stage/p1_loss")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)
    print(f"  Surfaces: {surfaces.shape}")
    print(f"  Log-returns: {log_returns.shape}")

    # Train/val split
    train_log_returns = log_returns[:4000]
    val_log_returns = log_returns[4000:]

    # Create sequences
    config = TwoStageConfig.get_model_config()
    # Use exp_c config
    config["latent_dim"] = 8
    config["z_dropout"] = 0.3

    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon
    device = config["device"]

    print(f"\nConfig: latent_dim={config['latent_dim']}, z_dropout={config['z_dropout']}")
    print(f"Sequence length: {seq_len} (context={context_len} + horizon={horizon})")

    train_tensor = torch.tensor(train_log_returns, dtype=torch.float32)
    train_sequences = create_sequences(train_tensor, seq_len)

    val_tensor = torch.tensor(val_log_returns, dtype=torch.float32)
    val_sequences = create_sequences(val_tensor, seq_len)

    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences: {len(val_sequences)}")

    # Define experiments
    experiments = [
        ("baseline", 0.0),
        ("p1_0.01", 0.01),
        ("p1_0.1", 0.1),
    ]

    all_results = {}

    for exp_name, p1_weight in experiments:
        results, _ = train_and_evaluate(
            exp_name=exp_name,
            config=config.copy(),
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            log_returns=log_returns,
            output_dir=output_dir,
            device=device,
            p1_weight=p1_weight,
            n_epochs=100,
        )
        all_results[exp_name] = results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n| Experiment | p1_weight | z_logvar | Decoder Gain | H=1 Center | H=30 Center | MSE |")
    print("|------------|-----------|----------|--------------|------------|-------------|-----|")
    for name, res in all_results.items():
        z_lv = res["z_stats"]["z_logvar_mean"]
        dg = res["decoder_gain"]
        h1_c = res["ci_results"][1]["center"]
        h30_c = res["ci_results"][30]["center"]
        mse = res["final_mse"]
        print(f"| {name:10s} | {res['p1_weight']:9.2f} | {z_lv:8.3f} | {dg:12.2e} | "
              f"{h1_c:10.1f}% | {h30_c:11.1f}% | {mse:.4f} |")

    # Per-grid comparison for H=1
    print("\n" + "="*70)
    print("PER-GRID-POINT VIOLATIONS AT H=1")
    print("="*70)

    moneyness = ['0.90', '0.95', '1.00', '1.05', '1.10']
    maturity = ['30d', '60d', '90d', '180d', '365d']

    for name, res in all_results.items():
        print(f"\n{name} (p1_weight={res['p1_weight']}):")
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

    baseline = all_results["baseline"]
    for name, res in all_results.items():
        if name == "baseline":
            continue

        dg_change = res["decoder_gain"] / baseline["decoder_gain"]
        h1_change = res["ci_results"][1]["center"] - baseline["ci_results"][1]["center"]
        mse_change = (res["final_mse"] - baseline["final_mse"]) / baseline["final_mse"] * 100

        print(f"\n{name}:")
        print(f"  Decoder gain change: {dg_change:.2f}x")
        print(f"  H=1 center violations change: {h1_change:+.1f}%")
        print(f"  MSE change: {mse_change:+.1f}%")

    # Success criteria check
    print("\n" + "-"*70)
    print("SUCCESS CRITERIA CHECK:")

    best_exp = max(all_results.items(), key=lambda x: x[1]["decoder_gain"])
    best_name, best_res = best_exp

    criteria = [
        (best_res["decoder_gain"] > 0.01, f"Decoder gain > 0.01: {best_res['decoder_gain']:.2e}"),
        (best_res["ci_results"][1]["center"] < 25, f"H=1 center < 25%: {best_res['ci_results'][1]['center']:.1f}%"),
        (best_res["final_mse"] < baseline["final_mse"] * 1.5, f"MSE < 1.5x baseline: {best_res['final_mse']:.4f}"),
    ]

    all_passed = True
    for passed, desc in criteria:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\nALL CRITERIA PASSED! Best experiment: {best_name}")
    else:
        print(f"\nSome criteria not met. Best experiment: {best_name}")

    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
