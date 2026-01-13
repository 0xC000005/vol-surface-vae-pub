"""
Predictor Hidden Size Experiment

Test if removing the 8-dim bottleneck improves z prediction.

Current architecture:
    Conv2D → 50-dim → LSTM(hidden=8) → proj(8→50) → LSTM → z
                            ↑               ↑
                       bottleneck     redundant if hidden=50

This experiment compares:
    - hidden_size=8  (baseline, with projection)
    - hidden_size=32 (intermediate)
    - hidden_size=50 (match embed_dim, no projection needed)

Usage:
    python experiments/backfill/two_stage_vae/exp_predictor_hidden_size.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictor


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, horizon, batch_size, shuffle=True):
    """Create dataloader with specified horizon."""
    seq_len = context_len + horizon
    sequences = []

    for i in range(len(log_returns) - seq_len):
        seq = log_returns[i:i + seq_len + 1]
        sequences.append(seq)

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_vae(device: str = "cuda"):
    """Load the frozen VAE model."""
    vae_paths = [
        "models/backfill/two_stage/student_t/student_t_best.pt",
        "models/backfill/two_stage/student_t_acf/student_t_acf_lambda0.02.pt",
    ]

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

            # Freeze VAE
            for param in vae.parameters():
                param.requires_grad = False

            return vae, config

    raise FileNotFoundError("No VAE model found")


def train_predictor(predictor, vae, train_loader, val_loader, config, epochs=50):
    """Train predictor to predict z from context only."""
    device = config["device"]
    context_len = config["context_len"]
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        predictor.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            B, T = batch_data.shape[:2]

            # Get ground truth z from frozen VAE encoder
            with torch.no_grad():
                z_mean_gt, z_logvar_gt, z_gt = vae.main_encoder({"surface": batch_data})
                z_target = z_mean_gt[:, context_len:]  # Only horizon positions

            # Predict z from context only
            context = batch_data[:, :context_len]
            horizon = T - context_len
            z_pred_mean, z_pred_logvar = predictor(context, horizon)

            # MSE loss
            loss = F.mse_loss(z_pred_mean, z_target.detach())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        predictor.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                B, T = batch_data.shape[:2]

                z_mean_gt, _, _ = vae.main_encoder({"surface": batch_data})
                z_target = z_mean_gt[:, context_len:]

                context = batch_data[:, :context_len]
                horizon = T - context_len
                z_pred_mean, _ = predictor(context, horizon)

                val_loss += F.mse_loss(z_pred_mean, z_target).item()
                n_val += 1

        val_loss = val_loss / n_val
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

    return best_val_loss, train_losses, val_losses


def evaluate_generation(predictor, vae, val_loader, config, n_samples=30):
    """Evaluate generation quality with this predictor."""
    device = config["device"]
    context_len = config["context_len"]
    predictor.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 10:  # Limit for speed
                break

            batch_data = batch_data.to(device)
            B, T = batch_data.shape[:2]
            horizon = T - context_len

            # Get context
            context = batch_data[:, :context_len]
            gt_horizon = batch_data[:, context_len:]

            # Predict z
            z_mean, z_logvar = predictor(context, horizon)
            z_std = torch.exp(0.5 * z_logvar)

            # Sample multiple times
            batch_samples = []
            for _ in range(n_samples):
                z = z_mean + z_std * torch.randn_like(z_std)

                # Get ctx_emb from VAE
                ctx_emb = vae.ctx_encoder({"surface": batch_data})
                ctx_emb_horizon = ctx_emb[:, context_len:]

                # Decode
                mean, samples, _, _ = vae.decoder(ctx_emb_horizon, z, sample=True)
                batch_samples.append(samples.cpu().numpy())

            all_samples.append(np.stack(batch_samples, axis=0))
            all_gt.append(gt_horizon.cpu().numpy())

    # Concatenate
    all_samples = np.concatenate(all_samples, axis=1)  # (n_samples, total_B, H, 5, 5)
    all_gt = np.concatenate(all_gt, axis=0)  # (total_B, H, 5, 5)

    # Compute metrics
    samples_atm = all_samples[:, :, :, 2, 2].flatten()
    gt_atm = all_gt[:, :, 2, 2].flatten()

    sample_kurt = kurtosis(samples_atm, fisher=True)
    gt_kurt = kurtosis(gt_atm, fisher=True)
    kurt_recovery = (sample_kurt / gt_kurt * 100) if abs(gt_kurt) > 0.1 else 0

    return {
        "kurtosis_recovery": kurt_recovery,
        "sample_kurtosis": sample_kurt,
        "gt_kurtosis": gt_kurt,
    }


def run_experiment():
    """Run hidden size comparison experiment."""
    print("=" * 70)
    print("Predictor Hidden Size Experiment")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"\nData: {len(log_returns)} days of log-returns")

    # Split data
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)
    train_data = log_returns[:train_end]
    val_data = log_returns[train_end:val_end]

    print(f"Train: {len(train_data)} days, Val: {len(val_data)} days")

    # Config
    context_len = 30
    horizon = 30
    batch_size = 32

    train_loader = create_dataloader(train_data, context_len, horizon, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, context_len, horizon, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load frozen VAE
    vae, vae_config = load_vae(device)

    # Hidden sizes to test
    hidden_sizes = [8, 32, 50]

    results = {}

    for hidden_size in hidden_sizes:
        print(f"\n{'=' * 70}")
        print(f"Testing hidden_size = {hidden_size}")
        print("=" * 70)

        # Create predictor config
        predictor_config = {
            "feat_dim": (5, 5),
            "latent_dim": vae_config.get("latent_dim", 16),
            "hidden_size": hidden_size,
            "num_layers": 1,
            "dropout": 0.2,
            "surface_hidden": [2, 4, 2],
            "max_horizon": horizon,
            "context_len": context_len,
            "device": device,
        }

        # Create predictor
        predictor = LatentPredictor(predictor_config)
        predictor = predictor.to(device)

        # Check if feedback projection is used
        print(f"  embed_dim: 50 (from Conv2D)")
        print(f"  hidden_size: {hidden_size}")
        print(f"  feedback_proj used: {predictor.use_feedback_proj}")
        print(f"  Total params: {sum(p.numel() for p in predictor.parameters()):,}")

        # Train
        print(f"\nTraining for 50 epochs...")
        best_val_loss, train_losses, val_losses = train_predictor(
            predictor, vae, train_loader, val_loader,
            {"device": device, "context_len": context_len},
            epochs=50
        )

        print(f"\nBest val MSE: {best_val_loss:.6f}")

        # Evaluate generation
        print("\nEvaluating generation quality...")
        gen_metrics = evaluate_generation(
            predictor, vae, val_loader,
            {"device": device, "context_len": context_len},
            n_samples=30
        )

        print(f"  Kurtosis recovery: {gen_metrics['kurtosis_recovery']:.1f}%")
        print(f"  Sample kurtosis: {gen_metrics['sample_kurtosis']:.2f}")
        print(f"  GT kurtosis: {gen_metrics['gt_kurtosis']:.2f}")

        results[hidden_size] = {
            "val_mse": best_val_loss,
            "use_feedback_proj": predictor.use_feedback_proj,
            "n_params": sum(p.numel() for p in predictor.parameters()),
            **gen_metrics,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Hidden':>8} | {'Params':>10} | {'feedback_proj':>13} | {'Val MSE':>10} | {'Kurt %':>8}")
    print("-" * 60)

    for h, r in results.items():
        proj_str = "Yes" if r["use_feedback_proj"] else "No"
        print(f"{h:>8} | {r['n_params']:>10,} | {proj_str:>13} | {r['val_mse']:>10.6f} | {r['kurtosis_recovery']:>7.1f}%")

    print("-" * 60)

    # Find best
    best_h = min(results.keys(), key=lambda h: results[h]["val_mse"])
    print(f"\nBest hidden_size: {best_h} (lowest val MSE)")

    if best_h == 50:
        print("CONCLUSION: Removing bottleneck (h=50) improves z prediction!")
    elif best_h == 8:
        print("CONCLUSION: Bottleneck (h=8) is optimal, compression helps.")
    else:
        print(f"CONCLUSION: Intermediate hidden_size ({best_h}) is optimal.")

    # Save results
    save_dir = Path("results/two_stage_vae/predictor_hidden_size")
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        save_dir / "results.npz",
        hidden_sizes=hidden_sizes,
        results=results,
    )
    print(f"\nResults saved to: {save_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
