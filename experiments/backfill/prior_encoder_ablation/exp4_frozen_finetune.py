"""
Experiment 4: Frozen Encoder Fine-tuning

Hypothesis: If we freeze context encoder and only train prior encoder, we can quickly
assess Architecture B's potential.

Method:
1. Load trained V4 model
2. Create CVAEWithPriorEncoderFullCov and copy frozen weights
3. Train ONLY prior encoder for 20 epochs
4. Measure E[Var(X|C)] / Var(X) before and after

Decision Point:
- If conditional variance increases significantly: Architecture B promising
- If no change: Problem lies elsewhere

Time: ~30-60 minutes (20 epochs)
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.cvae_prior_encoder import CVAEWithPriorEncoderFullCov
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


def measure_conditional_variance(model, val_surface, num_contexts=200, num_samples=100):
    """Quick conditional variance measurement."""
    C = model.context_len
    device = model.device
    model.eval()

    all_samples_per_context = []

    with torch.no_grad():
        for i in range(num_contexts):
            if i + C + 1 > len(val_surface):
                break

            context = val_surface[i:i+C].unsqueeze(0).to(device)

            # Sample from prior
            samples = []
            for _ in range(num_samples):
                # Get prior distribution
                ctx_input = {"surface": context}

                if isinstance(model, CVAEWithPriorEncoderFullCov):
                    # Prior encoder
                    mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=1)
                else:
                    # Original prior network
                    ctx_out = model.ctx_encoder(ctx_input)
                    context_summary = ctx_out[:, -1, :]
                    mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

                # Sample z ~ N(mu_p, Sigma_p)
                L = torch.linalg.cholesky(Sigma_p + 1e-6 * torch.eye(Sigma_p.shape[0], device=device))
                epsilon = torch.randn(1, 1, model.latent_dim, device=device)
                L_expanded = L.unsqueeze(0)
                z = mu_p + torch.bmm(L_expanded, epsilon.transpose(1, 2)).transpose(1, 2)

                # Decode
                decoded = model.decoder(z.reshape(1, model.latent_dim))
                samples.append(decoded.cpu())

            samples = torch.stack(samples, dim=0).squeeze()  # (num_samples, 5, 5)
            all_samples_per_context.append(samples)

    all_samples = torch.stack(all_samples_per_context).numpy()  # (N, num_samples, 5, 5)

    # Compute E[Var(X|C)]
    var_given_context = np.var(all_samples, axis=1)
    expected_conditional_var = np.mean(var_given_context)

    # Compute Var(X)
    all_gt = val_surface[C:C+num_contexts].numpy()
    total_var = np.var(all_gt.reshape(-1, 5, 5), axis=0).mean()

    ratio = expected_conditional_var / total_var

    return {
        'expected_conditional_var': expected_conditional_var,
        'total_var': total_var,
        'ratio': ratio
    }


def create_model_with_frozen_encoder(old_model, config):
    """
    Create new model with Prior Encoder and copy frozen weights from old model.
    """
    print("Creating new model with Prior Encoder...")

    # Create new model with Prior Encoder
    new_model = CVAEWithPriorEncoderFullCov(config)
    new_model = new_model.to(old_model.device)
    new_model.device = old_model.device

    print("Copying weights from trained model...")

    # Copy context encoder weights
    new_model.ctx_encoder.load_state_dict(old_model.ctx_encoder.state_dict())

    # Copy decoder weights
    new_model.decoder.load_state_dict(old_model.decoder.state_dict())

    # Freeze context encoder
    for param in new_model.ctx_encoder.parameters():
        param.requires_grad = False

    # Freeze decoder
    for param in new_model.decoder.parameters():
        param.requires_grad = False

    print("✓ Context encoder and decoder frozen")
    print("✓ Prior encoder will be trained from scratch")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in new_model.parameters())

    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    return new_model


def train_prior_encoder(model, train_data, config, num_epochs=20):
    """Train only the prior encoder with frozen context encoder and decoder."""

    # Optimizer only for prior encoder parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.learning_rate)

    model.train()

    surface = train_data["surface"]
    ex_data = train_data["ex_data"]

    batch_size = config.batch_size
    C = config.context_len
    seq_len = C + 1  # Context + 1 horizon

    print(f"\nTraining prior encoder for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print()

    for epoch in range(num_epochs):
        epoch_losses = []

        # Shuffle indices
        num_samples = len(surface) - seq_len
        indices = torch.randperm(num_samples)

        # Create batches
        num_batches = num_samples // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_indices = indices[batch_idx*batch_size:(batch_idx+1)*batch_size]

            # Get batch
            batch_surface = torch.stack([surface[i:i+seq_len] for i in batch_indices])
            batch_ex = torch.stack([ex_data[i:i+seq_len] for i in batch_indices])

            x = {
                "surface": batch_surface,
                "ex_feats": batch_ex
            }

            # Training step
            loss_dict = model.train_step(x, optimizer, scaler=None)

            epoch_losses.append(loss_dict['total_loss'])

        # Print epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

    print("\n✓ Training complete")

    return model


def main():
    print("=" * 80)
    print("EXPERIMENT 4: Frozen Encoder Fine-tuning")
    print("=" * 80)
    print()

    # Load configuration
    config = BackfillContext60ConfigV4FullCov

    # Find the trained model
    checkpoint_path = Path("models/backfill/context60_v4_full_cov/checkpoints/backfill_context60_latent12_v4_full_cov_phase1_ep99.pt")

    if not checkpoint_path.exists():
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading trained model from: {checkpoint_path}")
    model_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Strip _orig_mod. prefix from compiled models
    if "model" in model_data and any(k.startswith("_orig_mod.") for k in model_data["model"].keys()):
        model_data["model"] = {k.replace("_orig_mod.", ""): v for k, v in model_data["model"].items()}

    # Initialize old model
    old_model = CVAEFullCovPrior(model_data["model_config"])
    old_model.load_weights(dict_to_load=model_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    old_model = old_model.to(device)
    old_model.device = device
    print(f"Old model loaded on device: {device}")
    print()

    # Load data
    print("Loading data...")
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)

    surface = torch.tensor(data["surface"], dtype=torch.float32)
    # Construct ex_data from components
    ex_data = np.stack([data["ret"], data["skews"], data["slopes"]], axis=1)
    ex_data = torch.tensor(ex_data, dtype=torch.float32)

    # Split train/val
    train_start = config.train_start_idx
    train_end = config.train_end_idx

    train_surface = surface[train_start:train_end]
    train_ex_data = ex_data[train_start:train_end]

    val_surface = surface[train_end:train_end+1000]

    print(f"Train data: {train_surface.shape}")
    print(f"Val data: {val_surface.shape}")
    print()

    # =========================================================================
    # BASELINE: Measure conditional variance with original model
    # =========================================================================

    print("=" * 80)
    print("BASELINE: Measuring conditional variance with original model")
    print("=" * 80)
    print()

    baseline_stats = measure_conditional_variance(old_model, val_surface, num_contexts=200, num_samples=100)

    print(f"E[Var(X|C)] / Var(X) = {baseline_stats['ratio']:.4%}")
    print()

    # =========================================================================
    # Create new model with Prior Encoder
    # =========================================================================

    print("=" * 80)
    print("Creating model with Prior Encoder")
    print("=" * 80)
    print()

    # Update config to include prior encoder settings
    model_config = model_data["model_config"].copy()
    model_config["prior_surface_hidden"] = model_config.get("surface_hidden", [5, 5, 5])  # Same as context encoder
    model_config["prior_mem_hidden"] = 64
    model_config["prior_mem_layers"] = 1
    model_config["prior_dropout"] = 0.1
    model_config["prior_pos_dim"] = 32

    new_model = create_model_with_frozen_encoder(old_model, model_config)
    print()

    # =========================================================================
    # BEFORE TRAINING: Measure conditional variance with untrained prior encoder
    # =========================================================================

    print("=" * 80)
    print("BEFORE TRAINING: Measuring with untrained Prior Encoder")
    print("=" * 80)
    print()

    before_stats = measure_conditional_variance(new_model, val_surface, num_contexts=200, num_samples=100)

    print(f"E[Var(X|C)] / Var(X) = {before_stats['ratio']:.4%}")
    print()

    # =========================================================================
    # TRAIN: Train only the prior encoder
    # =========================================================================

    print("=" * 80)
    print("TRAINING: Prior Encoder only (20 epochs)")
    print("=" * 80)

    train_data = {
        "surface": train_surface,
        "ex_data": train_ex_data
    }

    trained_model = train_prior_encoder(new_model, train_data, config, num_epochs=20)

    # =========================================================================
    # AFTER TRAINING: Measure conditional variance
    # =========================================================================

    print("=" * 80)
    print("AFTER TRAINING: Measuring with trained Prior Encoder")
    print("=" * 80)
    print()

    after_stats = measure_conditional_variance(trained_model, val_surface, num_contexts=200, num_samples=100)

    print(f"E[Var(X|C)] / Var(X) = {after_stats['ratio']:.4%}")
    print()

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Configuration':<30} {'E[Var(X|C)] / Var(X)':<25} {'Change':<15}")
    print("-" * 80)
    print(f"{'Baseline (Original Model)':<30} {baseline_stats['ratio']:<25.4%} {'-':<15}")
    print(f"{'Before Training (Untrained)':<30} {before_stats['ratio']:<25.4%} {(before_stats['ratio']/baseline_stats['ratio']-1)*100:+.1f}%")
    print(f"{'After Training (20 epochs)':<30} {after_stats['ratio']:<25.4%} {(after_stats['ratio']/baseline_stats['ratio']-1)*100:+.1f}%")
    print()

    improvement = after_stats['ratio'] / baseline_stats['ratio']

    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if improvement > 4.0:
        print(f"✓ MAJOR IMPROVEMENT: {improvement:.1f}× increase in conditional variance")
        print("  Prior Encoder significantly improves uncertainty estimation")
        print("  → Architecture B is very promising, proceed with full training")
    elif improvement > 2.0:
        print(f"✓ SIGNIFICANT IMPROVEMENT: {improvement:.1f}× increase in conditional variance")
        print("  Prior Encoder helps with uncertainty estimation")
        print("  → Architecture B is promising, consider full training")
    elif improvement > 1.5:
        print(f"⚠ MODERATE IMPROVEMENT: {improvement:.1f}× increase in conditional variance")
        print("  Prior Encoder provides some benefit")
        print("  → May need longer training or architecture tuning")
    else:
        print(f"⚠ MINIMAL IMPROVEMENT: {improvement:.1f}× increase in conditional variance")
        print("  Prior Encoder does not significantly help")
        print("  → Problem may lie elsewhere (decoder, KL weight, etc.)")

    print()

    # Save results
    output_dir = Path("results/prior_encoder_ablation/frozen_finetune")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "frozen_finetune_results.npz"
    np.savez(
        output_file,
        baseline_ratio=baseline_stats['ratio'],
        before_ratio=before_stats['ratio'],
        after_ratio=after_stats['ratio'],
        improvement=improvement
    )

    print(f"Results saved to: {output_file}")
    print()

    # Save trained model
    model_save_path = output_dir / "trained_prior_encoder_ep20.pt"
    torch.save({
        "model_config": model_config,
        "state_dict": trained_model.state_dict(),
        "optimizer_state": None,
        "epoch": 20
    }, model_save_path)

    print(f"Trained model saved to: {model_save_path}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
