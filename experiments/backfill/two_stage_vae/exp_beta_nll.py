"""
Train Student-t VAE with β-NLL Loss for Unbiased Mean Estimation

This addresses the heteroscedastic regression pathology where standard NLL
allows the model to trade mean accuracy for variance, resulting in biased
mean predictions that compound exponentially during autoregressive chaining.

β-NLL (Seitzer 2022, ICLR) weights loss by variance^β to prevent this:
- β=0: Standard NLL (variance dominates gradients)
- β=0.5: Balanced (recommended)
- β=1: MSE-like (variance ignored)

Reference: https://arxiv.org/abs/2203.09168

Usage:
    python experiments/backfill/two_stage_vae/exp_beta_nll.py
    python experiments/backfill/two_stage_vae/exp_beta_nll.py --beta 0.5
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from config.two_stage_config import TwoStageConfig


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_sequences(data, context_len, horizon=1):
    """Create overlapping sequences for training."""
    sequences = []
    for i in range(len(data) - context_len - horizon + 1):
        seq = data[i:i + context_len + horizon]
        sequences.append(seq)
    return np.array(sequences)


def train_with_beta_nll(
    model, train_loader, val_loader, config,
    epochs=150, beta=0.5, kl_weight=0.001
):
    """
    Train Student-t VAE with β-NLL loss.

    Args:
        model: CVAETwoStageStudentTMLP model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model config dict
        epochs: Number of training epochs
        beta: β-NLL parameter (0.5 recommended)
        kl_weight: Weight for KL divergence
    """
    device = config.get("device", "cuda")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-5
    )

    best_loss = float('inf')
    best_state = None

    print(f"\n{'='*70}")
    print(f"Training with β-NLL (β={beta})")
    print(f"{'='*70}")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_nlls = []
        train_kls = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Forward pass
            mean, z_mean, z_logvar, factor, log_diag = model(
                batch, return_full_sequence=True
            )

            # Target and prediction
            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # β-NLL loss
            nll_loss = model.decoder.compute_student_t_nll(
                pred, target, factor, log_diag, beta=beta
            )

            # KL loss
            kl_loss = -0.5 * (
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp()
            ).mean()

            # Total loss
            total_loss = nll_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())
            train_nlls.append(nll_loss.item())
            train_kls.append(kl_loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}

                mean, z_mean, z_logvar, factor, log_diag = model(
                    batch, return_full_sequence=True
                )
                target = batch_data[:, 1:]
                pred = mean[:, :-1]

                # Use same β for validation
                nll_loss = model.decoder.compute_student_t_nll(
                    pred, target, factor, log_diag, beta=beta
                )
                val_losses.append(nll_loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{epochs}: "
                f"NLL={np.mean(train_nlls):.4f}, "
                f"KL={np.mean(train_kls):.4f}, "
                f"Val={val_loss:.4f}"
            )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_mean_bias(model, val_loader, config):
    """
    Evaluate mean bias at different z values.

    Returns dict with bias statistics.
    """
    device = config.get("device", "cuda")
    model = model.to(device)
    model.eval()

    # Collect z values and decoded means
    z_values = []
    decoded_means = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            z_mean, z_logvar, _ = model.main_encoder(batch)
            z = z_mean[:, -1]  # Last timestep

            # Decode each z
            for i in range(z.shape[0]):
                z_i = z[i:i+1].unsqueeze(1)  # (1, 1, latent_dim)
                mean = model.decoder.mean_net(z_i.view(1, -1))
                decoded_means.append(mean.cpu().numpy())
                z_values.append(z[i].cpu().numpy())

    z_values = np.array(z_values)
    decoded_means = np.array(decoded_means).reshape(-1, 5, 5)

    # Statistics
    mean_bias = decoded_means.mean(axis=0)
    mean_bias_abs = np.abs(mean_bias)

    # Check canonical z values
    z_zero = torch.zeros(1, config.get("latent_dim", 8)).to(device)
    with torch.no_grad():
        mean_at_zero = model.decoder.mean_net(z_zero).view(5, 5).cpu().numpy()

    return {
        "mean_bias_grid": mean_bias,
        "mean_bias_abs_avg": mean_bias_abs.mean(),
        "mean_at_z_zero": mean_at_zero,
        "mean_at_z_zero_00": mean_at_zero[0, 0],
        "mean_at_z_zero_22": mean_at_zero[2, 2],
        "z_norm_avg": np.linalg.norm(z_values, axis=1).mean(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.5, help="β-NLL parameter")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--compare", action="store_true", help="Compare β=0 vs β=0.5")
    args = parser.parse_args()

    print("=" * 70)
    print("β-NLL Training Experiment")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    # Split data
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)

    train_data = log_returns[:train_end]
    val_data = log_returns[train_end:val_end]

    print(f"\nData: {len(log_returns)} days")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")

    # Create sequences
    context_len = 30
    train_seqs = create_sequences(train_data, context_len, horizon=1)
    val_seqs = create_sequences(val_data, context_len, horizon=1)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_seqs, dtype=torch.float32)),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_seqs, dtype=torch.float32)),
        batch_size=64, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Get config
    config = TwoStageConfig.get_model_config()
    config["device"] = device
    config["latent_dim"] = 8  # Match existing model

    if args.compare:
        # Compare β=0 (standard) vs β=0.5 (recommended)
        results = {}

        for beta in [0.0, 0.5]:
            print(f"\n{'='*70}")
            print(f"Training with β={beta}")
            print(f"{'='*70}")

            model = CVAETwoStageStudentTMLP(config)
            model = train_with_beta_nll(
                model, train_loader, val_loader, config,
                epochs=args.epochs, beta=beta
            )

            # Evaluate bias
            bias_results = evaluate_mean_bias(model, val_loader, config)

            results[beta] = bias_results

            print(f"\n  Results for β={beta}:")
            print(f"    Mean at z=0 (0,0): {bias_results['mean_at_z_zero_00']:.4f}")
            print(f"    Mean at z=0 (2,2): {bias_results['mean_at_z_zero_22']:.4f}")
            print(f"    Avg |mean bias|: {bias_results['mean_bias_abs_avg']:.4f}")

        # Summary comparison
        print("\n" + "=" * 70)
        print("COMPARISON: β=0 (standard) vs β=0.5 (β-NLL)")
        print("=" * 70)
        print(f"\n{'Metric':<30} | {'β=0':>12} | {'β=0.5':>12} | {'Improvement':>12}")
        print("-" * 70)

        for metric in ['mean_at_z_zero_00', 'mean_at_z_zero_22', 'mean_bias_abs_avg']:
            v0 = results[0.0][metric]
            v05 = results[0.5][metric]
            imp = (abs(v0) - abs(v05)) / abs(v0) * 100 if v0 != 0 else 0
            print(f"{metric:<30} | {v0:>12.4f} | {v05:>12.4f} | {imp:>+11.1f}%")

    else:
        # Single training with specified β
        print(f"\nTraining with β={args.beta}")

        model = CVAETwoStageStudentTMLP(config)
        model = train_with_beta_nll(
            model, train_loader, val_loader, config,
            epochs=args.epochs, beta=args.beta
        )

        # Evaluate
        bias_results = evaluate_mean_bias(model, val_loader, config)

        print("\n" + "=" * 70)
        print("MEAN BIAS EVALUATION")
        print("=" * 70)
        print(f"\nMean at z=0:")
        print(f"  Grid point (0,0): {bias_results['mean_at_z_zero_00']:.4f}")
        print(f"  Grid point (2,2): {bias_results['mean_at_z_zero_22']:.4f}")
        print(f"\nAverage |mean bias|: {bias_results['mean_bias_abs_avg']:.4f}")

        # Save model
        save_dir = Path("models/backfill/two_stage/beta_nll")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"student_t_beta{args.beta}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "beta": args.beta,
            "bias_results": bias_results,
        }, save_path)
        print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
