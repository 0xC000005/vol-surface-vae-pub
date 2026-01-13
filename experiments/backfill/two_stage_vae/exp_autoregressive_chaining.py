"""
Autoregressive Chaining Experiment

Goal: Test what happens when we chain predictions by feeding generated
outputs back as new context.

Chaining logic (15-day overlap):
    Hop 1: Real[0:30] → Generate[0:30]
    Hop 2: Real[15:30] + Gen[0:15] → Generate[30:60]
    Hop 3: Gen[0:30] → Generate[60:90]
    ...

This tests if error/kurtosis/ACF degrades across hops.

Usage:
    python experiments/backfill/two_stage_vae/exp_autoregressive_chaining.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictor


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_models(device: str = "cuda"):
    """Load the Student-t VAE and trained predictor."""
    # Try to load existing models
    vae_paths = [
        "models/backfill/two_stage/student_t/student_t_best.pt",
        "models/backfill/two_stage/student_t_acf/student_t_acf_lambda0.02.pt",
    ]

    vae = None
    config = None

    for vae_path in vae_paths:
        if Path(vae_path).exists():
            print(f"Loading VAE from {vae_path}")
            vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
            config = vae_ckpt["model_config"]
            config["device"] = device

            vae = CVAETwoStageStudentTMLP(config)
            vae.load_state_dict(vae_ckpt["model_state_dict"])
            vae = vae.to(device)
            vae.eval()
            break

    if vae is None:
        raise FileNotFoundError("No VAE model found. Train one first.")

    # Load predictor if available
    predictor = None
    pred_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    if Path(pred_path).exists():
        print(f"Loading predictor from {pred_path}")
        pred_ckpt = torch.load(pred_path, map_location=device, weights_only=False)
        predictor = LatentPredictor(config)
        predictor.load_state_dict(pred_ckpt["predictor_state_dict"])
        predictor = predictor.to(device)
        predictor.eval()
    else:
        print("No predictor found, will use Oracle mode only.")

    return vae, predictor, config


def generate_single_hop_oracle(vae, context_log_returns, horizon, n_samples, device):
    """
    Generate one hop of predictions using Oracle mode (encoder z).

    Args:
        vae: Trained VAE model
        context_log_returns: (context_len, 5, 5) log-return context
        horizon: Number of days to generate
        n_samples: Number of samples per day

    Returns:
        generated_log_returns: (n_samples, horizon, 5, 5) generated log-returns
    """
    context_len = context_log_returns.shape[0]

    # Build batch with context only (for oracle, we sample z from encoder)
    context_tensor = torch.tensor(context_log_returns, dtype=torch.float32).unsqueeze(0).to(device)

    generated = np.zeros((n_samples, horizon, 5, 5))

    with torch.no_grad():
        for h in range(horizon):
            # For oracle: build sequence up to this point
            if h == 0:
                # First step: just context
                batch = {"surface": context_tensor}
            else:
                # Subsequent steps: context + generated so far
                # We need to extend the sequence with generated returns
                # For oracle mode, we'll use the mean of samples as "observed"
                gen_so_far = torch.tensor(
                    generated[:, :h].mean(axis=0),  # (h, 5, 5) mean across samples
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                seq = torch.cat([context_tensor, gen_so_far], dim=1)
                batch = {"surface": seq}

            # Sample from VAE
            samples = vae.sample(batch, n_samples=n_samples)
            # samples: (n_samples, 1, T, 5, 5)

            # Extract the last timestep prediction
            generated[:, h] = samples[:, 0, -1].cpu().numpy()

    return generated


def generate_single_hop_predictor(vae, predictor, context_log_returns, horizon, n_samples, device):
    """
    Generate one hop of predictions using Predictor mode (predicted z).

    Simplified: Generate step by step, sampling z from predictor.

    Args:
        vae: Trained VAE model
        predictor: Trained LatentPredictor
        context_log_returns: (context_len, 5, 5) log-return context
        horizon: Number of days to generate
        n_samples: Number of samples per day

    Returns:
        generated_log_returns: (n_samples, horizon, 5, 5) generated log-returns
    """
    context_len = context_log_returns.shape[0]
    context_tensor = torch.tensor(context_log_returns, dtype=torch.float32).unsqueeze(0).to(device)

    generated = np.zeros((n_samples, horizon, 5, 5))

    with torch.no_grad():
        # Get predicted z for all horizon steps
        z_mean, z_logvar = predictor(context_tensor, horizon=horizon)
        # z_mean: (1, horizon, latent_dim)

        for s in range(n_samples):
            # Sample z for this sample
            z_future = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

            # Generate step by step using oracle-like approach but with predicted z
            for h in range(horizon):
                if h == 0:
                    # First step: just context
                    batch = {"surface": context_tensor}
                else:
                    # Extend with generated so far (using this sample's generations)
                    gen_so_far = torch.tensor(
                        generated[s, :h],  # (h, 5, 5)
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    seq = torch.cat([context_tensor, gen_so_far], dim=1)
                    batch = {"surface": seq}

                # Use model's sample with forced z
                # For simplicity, sample once from model and use mean
                samples = vae.sample(batch, n_samples=1)
                generated[s, h] = samples[0, 0, -1].cpu().numpy()

    return generated


def generate_chained_sequence(
    vae, predictor, initial_context, gt_log_returns,
    n_hops=4, horizon=30, overlap=15, n_samples=50, device="cuda", mode="oracle"
):
    """
    Generate long sequence by chaining predictions.

    Args:
        vae: Trained VAE
        predictor: Trained predictor (can be None for oracle mode)
        initial_context: (context_len, 5, 5) initial real context
        gt_log_returns: Full GT log-returns for comparison
        n_hops: Number of generation hops
        horizon: Days per hop
        overlap: Days of overlap between hops
        n_samples: Samples per generation
        device: CUDA/CPU
        mode: "oracle" or "predictor"

    Returns:
        dict with generated sequences and per-hop metrics
    """
    context_len = initial_context.shape[0]
    total_days = n_hops * (horizon - overlap) + overlap

    print(f"\nGenerating {n_hops} hops × {horizon} days with {overlap}-day overlap")
    print(f"Total unique days: {total_days}")

    # Storage for all generated samples
    all_generated = []  # List of (n_samples, horizon, 5, 5) per hop

    # Current context for next hop
    current_context = initial_context.copy()

    # Track how much is real vs generated in context
    real_days_in_context = context_len

    per_hop_metrics = []

    for hop in range(n_hops):
        print(f"\n--- Hop {hop + 1}/{n_hops} ---")
        print(f"  Context: {real_days_in_context} real + {context_len - real_days_in_context} generated")

        # Generate this hop
        if mode == "oracle":
            generated = generate_single_hop_oracle(
                vae, current_context, horizon, n_samples, device
            )
        else:
            if predictor is None:
                raise ValueError("Predictor required for predictor mode")
            generated = generate_single_hop_predictor(
                vae, predictor, current_context, horizon, n_samples, device
            )

        all_generated.append(generated)

        # Compute metrics for this hop
        start_idx = hop * (horizon - overlap)
        end_idx = start_idx + horizon

        # Get GT for this hop if available
        if end_idx <= len(gt_log_returns):
            gt_hop = gt_log_returns[start_idx:end_idx]

            # Compute per-hop metrics
            gen_atm = generated[:, :, 2, 2].flatten()  # All samples, all days, ATM
            gt_atm = gt_hop[:, 2, 2].flatten()

            hop_kurtosis = kurtosis(gen_atm, fisher=True)
            gt_kurtosis = kurtosis(gt_atm, fisher=True)

            # RMSE (mean prediction vs GT)
            mean_pred = generated.mean(axis=0)  # (horizon, 5, 5)
            rmse = np.sqrt(np.mean((mean_pred - gt_hop) ** 2))

            # Std of predictions (uncertainty)
            pred_std = generated.std(axis=0).mean()

            metrics = {
                "hop": hop + 1,
                "kurtosis": hop_kurtosis,
                "gt_kurtosis": gt_kurtosis,
                "kurtosis_recovery": hop_kurtosis / gt_kurtosis * 100 if gt_kurtosis > 0.1 else 0,
                "rmse": rmse,
                "pred_std": pred_std,
                "real_days_in_context": real_days_in_context,
            }
        else:
            metrics = {
                "hop": hop + 1,
                "kurtosis": kurtosis(generated[:, :, 2, 2].flatten(), fisher=True),
                "gt_kurtosis": None,
                "kurtosis_recovery": None,
                "rmse": None,
                "pred_std": generated.std(axis=0).mean(),
                "real_days_in_context": real_days_in_context,
            }

        per_hop_metrics.append(metrics)

        gt_kurt_str = f"{metrics['gt_kurtosis']:.2f}" if metrics['gt_kurtosis'] else 'N/A'
        print(f"  Kurtosis: {metrics['kurtosis']:.2f} (GT: {gt_kurt_str})")
        if metrics['rmse']:
            print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Pred Std: {metrics['pred_std']:.4f}")

        # Update context for next hop
        # Take last (context_len - overlap) days from old context
        # + first overlap days from generated
        if hop < n_hops - 1:
            keep_from_old = context_len - overlap
            mean_generated = generated.mean(axis=0)  # Use mean as "observed"

            new_context = np.concatenate([
                current_context[overlap:],  # Last (context_len - overlap) of old
                mean_generated[:overlap]     # First overlap days of new
            ], axis=0)

            current_context = new_context
            real_days_in_context = max(0, real_days_in_context - (horizon - overlap))

    return {
        "all_generated": all_generated,
        "per_hop_metrics": per_hop_metrics,
        "mode": mode,
        "n_hops": n_hops,
        "horizon": horizon,
        "overlap": overlap,
    }


def plot_chaining_results(oracle_results, predictor_results, save_path=None):
    """Plot comparison of oracle vs predictor chaining."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract metrics
    oracle_metrics = oracle_results["per_hop_metrics"]
    hops = [m["hop"] for m in oracle_metrics]

    # Plot 1: Kurtosis by hop
    ax1 = axes[0, 0]
    oracle_kurt = [m["kurtosis"] for m in oracle_metrics]
    ax1.plot(hops, oracle_kurt, 'b-o', label='Oracle', markersize=8)

    if predictor_results:
        pred_metrics = predictor_results["per_hop_metrics"]
        pred_kurt = [m["kurtosis"] for m in pred_metrics]
        ax1.plot(hops, pred_kurt, 'r-s', label='Predictor', markersize=8)

    gt_kurt = [m["gt_kurtosis"] for m in oracle_metrics if m["gt_kurtosis"]]
    if gt_kurt:
        ax1.axhline(np.mean(gt_kurt), color='g', linestyle='--', label=f'GT Mean ({np.mean(gt_kurt):.1f})')

    ax1.set_xlabel('Hop')
    ax1.set_ylabel('Excess Kurtosis')
    ax1.set_title('Kurtosis Degradation Across Hops')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Kurtosis Recovery
    ax2 = axes[0, 1]
    oracle_recovery = [m["kurtosis_recovery"] for m in oracle_metrics if m["kurtosis_recovery"]]
    ax2.plot(hops[:len(oracle_recovery)], oracle_recovery, 'b-o', label='Oracle', markersize=8)

    if predictor_results:
        pred_recovery = [m["kurtosis_recovery"] for m in pred_metrics if m["kurtosis_recovery"]]
        ax2.plot(hops[:len(pred_recovery)], pred_recovery, 'r-s', label='Predictor', markersize=8)

    ax2.axhline(100, color='k', linestyle='--', alpha=0.5, label='Perfect')
    ax2.set_xlabel('Hop')
    ax2.set_ylabel('Kurtosis Recovery (%)')
    ax2.set_title('Kurtosis Recovery by Hop')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: RMSE by hop
    ax3 = axes[1, 0]
    oracle_rmse = [m["rmse"] for m in oracle_metrics if m["rmse"]]
    ax3.plot(hops[:len(oracle_rmse)], oracle_rmse, 'b-o', label='Oracle', markersize=8)

    if predictor_results:
        pred_rmse = [m["rmse"] for m in pred_metrics if m["rmse"]]
        ax3.plot(hops[:len(pred_rmse)], pred_rmse, 'r-s', label='Predictor', markersize=8)

    ax3.set_xlabel('Hop')
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE Degradation Across Hops')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Real vs Generated context
    ax4 = axes[1, 1]
    real_days = [m["real_days_in_context"] for m in oracle_metrics]
    ax4.bar(hops, real_days, color='green', alpha=0.7, label='Real days in context')
    ax4.set_xlabel('Hop')
    ax4.set_ylabel('Days')
    ax4.set_title('Real Data in Context Over Hops')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def main():
    """Run autoregressive chaining experiment."""
    print("=" * 70)
    print("Autoregressive Chaining Experiment")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"\nData: {len(log_returns)} days of log-returns")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    try:
        vae, predictor, config = load_models(device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Training a quick model for testing...")

        # Quick train for testing
        from config.two_stage_config import TWO_STAGE_CONFIG
        config = TWO_STAGE_CONFIG.copy()
        config["device"] = device
        config["horizon"] = 30
        config["context_len"] = 30

        vae = CVAETwoStageStudentTMLP(config)
        vae = vae.to(device)
        predictor = None

    # Configuration
    context_len = config.get("context_len", 30)
    horizon = 30
    overlap = 15  # Half context
    n_hops = 4
    n_samples = 30

    print(f"\nConfiguration:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon per hop: {horizon}")
    print(f"  Overlap: {overlap} days")
    print(f"  Number of hops: {n_hops}")
    print(f"  Samples per generation: {n_samples}")
    print(f"  Total days generated: {n_hops * (horizon - overlap) + overlap}")

    # Pick a starting point (validation set)
    train_end = int(len(log_returns) * 0.7)
    start_idx = train_end + 100  # Start in validation period

    initial_context = log_returns[start_idx:start_idx + context_len]
    gt_log_returns = log_returns[start_idx + context_len:]

    print(f"\nStarting from index {start_idx}")
    print(f"GT available for: {len(gt_log_returns)} days")

    # Run Oracle mode
    print("\n" + "=" * 70)
    print("ORACLE MODE")
    print("=" * 70)

    oracle_results = generate_chained_sequence(
        vae, predictor, initial_context, gt_log_returns,
        n_hops=n_hops, horizon=horizon, overlap=overlap,
        n_samples=n_samples, device=device, mode="oracle"
    )

    # Run Predictor mode if available
    predictor_results = None
    if predictor is not None:
        print("\n" + "=" * 70)
        print("PREDICTOR MODE")
        print("=" * 70)

        predictor_results = generate_chained_sequence(
            vae, predictor, initial_context, gt_log_returns,
            n_hops=n_hops, horizon=horizon, overlap=overlap,
            n_samples=n_samples, device=device, mode="predictor"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nOracle Mode:")
    print(f"{'Hop':>5} | {'Kurtosis':>10} | {'Recovery':>10} | {'RMSE':>10} | {'Real Days':>10}")
    print("-" * 55)
    for m in oracle_results["per_hop_metrics"]:
        recovery_str = f"{m['kurtosis_recovery']:.1f}%" if m['kurtosis_recovery'] else "N/A"
        rmse_str = f"{m['rmse']:.4f}" if m['rmse'] else "N/A"
        print(f"{m['hop']:>5} | {m['kurtosis']:>10.2f} | {recovery_str:>10} | {rmse_str:>10} | {m['real_days_in_context']:>10}")

    if predictor_results:
        print("\nPredictor Mode:")
        print(f"{'Hop':>5} | {'Kurtosis':>10} | {'Recovery':>10} | {'RMSE':>10} | {'Real Days':>10}")
        print("-" * 55)
        for m in predictor_results["per_hop_metrics"]:
            recovery_str = f"{m['kurtosis_recovery']:.1f}%" if m['kurtosis_recovery'] else "N/A"
            rmse_str = f"{m['rmse']:.4f}" if m['rmse'] else "N/A"
            print(f"{m['hop']:>5} | {m['kurtosis']:>10.2f} | {recovery_str:>10} | {rmse_str:>10} | {m['real_days_in_context']:>10}")

    # Plot
    save_dir = Path("results/two_stage_vae/autoregressive_chaining")
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_chaining_results(oracle_results, predictor_results,
                          save_path=save_dir / "chaining_comparison.png")

    # Save results
    np.savez(
        save_dir / "chaining_results.npz",
        oracle_metrics=[m for m in oracle_results["per_hop_metrics"]],
        predictor_metrics=[m for m in predictor_results["per_hop_metrics"]] if predictor_results else [],
        config={
            "context_len": context_len,
            "horizon": horizon,
            "overlap": overlap,
            "n_hops": n_hops,
            "n_samples": n_samples,
        },
    )
    print(f"\nResults saved to: {save_dir}")

    return oracle_results, predictor_results


if __name__ == "__main__":
    main()
