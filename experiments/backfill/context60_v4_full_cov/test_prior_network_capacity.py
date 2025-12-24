"""
Test Prior Network Capacity for V4 Full Covariance Prior

This script tests different hidden_dims configurations to determine if the prior
network should be simplified to prevent over-discrimination.

Problem 1 from CONDITIONAL_VARIANCE_SOLUTIONS.md:
- Prior network too context-specific (r=0.89 correlation)
- Need to test if simpler network forces coarser context groupings

Usage:
    python experiments/backfill/context60_v4_full_cov/test_prior_network_capacity.py

Results saved to:
    results/context60_v4_full_cov/capacity_sweep/
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import json
from datetime import datetime

from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.utils import set_seeds, model_eval
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov
from torch.amp import autocast, GradScaler


# ============================================================================
# Configurations to Test
# ============================================================================

CONFIGS_TO_TEST = [
    {"hidden_dims": [256, 128], "name": "large", "desc": "Current baseline"},
    {"hidden_dims": [128, 64], "name": "medium", "desc": "50% reduction"},
    {"hidden_dims": [64, 32], "name": "small", "desc": "Conservative 86% reduction"},
    {"hidden_dims": [32], "name": "minimal", "desc": "Aggressive 95% reduction"},
]

# Quick training settings
QUICK_EPOCHS = 50  # Phase 1 only (teacher forcing)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Helper Functions
# ============================================================================

def compute_param_count(hidden_dims, input_dim=76, output_dim=12):
    """Compute parameter count for MLP with given hidden_dims."""
    param_count = 0
    prev_dim = input_dim
    for h_dim in hidden_dims:
        param_count += prev_dim * h_dim + h_dim  # weights + bias
        prev_dim = h_dim
    param_count += prev_dim * output_dim + output_dim  # output layer
    return param_count


def compute_prior_context_correlation(model, data_loader, device):
    """
    Measure correlation between context embeddings and prior μ.

    High correlation (>0.8) = over-discrimination
    Low correlation (<0.6) = coarser groupings
    """
    model.eval()

    all_context_emb = []
    all_prior_mu = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing correlations", leave=False):
            surface = batch["surface"].to(device)
            B = surface.shape[0]
            T = surface.shape[1]
            C = T - model.horizon

            # Get context embedding
            ctx_surface = surface[:, :C, :, :]
            ctx_input = {"surface": ctx_surface}
            if "ex_feats" in batch:
                ctx_input["ex_feats"] = batch["ex_feats"][:, :C, :].to(device)

            ctx_embedding = model.ctx_encoder(ctx_input)  # (B, C, latent_dim)
            context_summary = ctx_embedding[:, -1, :]  # (B, latent_dim)

            # Get prior μ (just for first timestep)
            mu_p, _ = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

            all_context_emb.append(context_summary.cpu().numpy())
            all_prior_mu.append(mu_p[:, 0, :].cpu().numpy())  # First timestep

    # Concatenate all batches
    context_emb = np.concatenate(all_context_emb, axis=0)  # (N, latent_dim)
    prior_mu = np.concatenate(all_prior_mu, axis=0)  # (N, latent_dim)

    # Compute correlation per latent dimension
    correlations = []
    for d in range(context_emb.shape[1]):
        corr = np.corrcoef(context_emb[:, d], prior_mu[:, d])[0, 1]
        correlations.append(corr)

    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)

    return mean_corr, std_corr, correlations


def measure_sample_diversity(model, data_loader, device, num_samples=10):
    """
    Measure diversity of samples from the prior for each context.

    Returns std of samples across latent dimensions.
    Higher diversity = less over-discrimination
    """
    model.eval()

    all_sample_stds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Measuring diversity", leave=False):
            surface = batch["surface"].to(device)
            B = surface.shape[0]
            T = surface.shape[1]
            C = T - model.horizon

            # Get context embedding
            ctx_surface = surface[:, :C, :, :]
            ctx_input = {"surface": ctx_surface}
            if "ex_feats" in batch:
                ctx_input["ex_feats"] = batch["ex_feats"][:, :C, :].to(device)

            ctx_embedding = model.ctx_encoder(ctx_input)
            context_summary = ctx_embedding[:, -1, :]  # (B, latent_dim)

            # Sample multiple times from prior
            samples = []
            for _ in range(num_samples):
                z = model.full_cov_prior.sample(context_summary, horizon=1)
                samples.append(z[:, 0, :])  # (B, latent_dim)

            samples = torch.stack(samples, dim=1)  # (B, num_samples, latent_dim)

            # Compute std across samples for each context
            sample_std = torch.std(samples, dim=1).mean(dim=1)  # (B,)
            all_sample_stds.extend(sample_std.cpu().numpy().tolist())

    return np.mean(all_sample_stds), np.std(all_sample_stds)


def compute_roughness_ratio(model, data_loader, device, ground_truth_roughness=0.0364):
    """
    Compute roughness ratio of generated trajectories.

    Target: >40% (0.40)
    Oracle: 75% (0.75)
    Baseline: 9.7% (0.097)
    """
    model.eval()

    all_pred_roughness = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing roughness", leave=False):
            surface = batch["surface"].to(device)
            B = surface.shape[0]
            T = surface.shape[1]
            C = T - model.horizon

            # Generate predictions
            ctx_surface = surface[:, :C, :, :]
            ctx_input = {"surface": ctx_surface}
            if "ex_feats" in batch:
                ctx_input["ex_feats"] = batch["ex_feats"][:, :C, :].to(device)

            # Generate with prior sampling
            if "ex_feats" in batch:
                surf_pred, _ = model.get_surface_given_conditions(
                    ctx_input, horizon=model.horizon, prior_mode="full_cov"
                )
            else:
                surf_pred = model.get_surface_given_conditions(
                    ctx_input, horizon=model.horizon, prior_mode="full_cov"
                )

            # Compute roughness (std of daily changes)
            # Use ATM 6M point (2, 2)
            pred_series = surf_pred[:, :, 2, 2].cpu().numpy()  # (B, H)
            pred_changes = np.diff(pred_series, axis=1)  # (B, H-1)
            roughness = np.std(pred_changes)
            all_pred_roughness.append(roughness)

    mean_roughness = np.mean(all_pred_roughness)
    roughness_ratio = mean_roughness / ground_truth_roughness

    return roughness_ratio, mean_roughness


# ============================================================================
# Training Function
# ============================================================================

def train_quick(model, train_loader, valid_loader, device, num_epochs=50):
    """Quick training for 50 epochs (Phase 1 only) with proper mixed precision."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda')

    history = {
        'train_loss': [],
        'val_loss': [],
        'phi': [],
        'sigma_sq': []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            losses = model.train_step(batch, optimizer, scaler=scaler)
            train_losses.append(losses['loss'].item())

        # Validation
        val_metrics = model_eval(model, valid_loader)

        # Get prior params
        phi = model.full_cov_prior.get_phi().item()
        sigma_sq = model.full_cov_prior.get_sigma_sq().item()

        # Record
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_metrics['loss'])
        history['phi'].append(phi)
        history['sigma_sq'].append(sigma_sq)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train={history['train_loss'][-1]:.6f}, "
                  f"Val={history['val_loss'][-1]:.6f}, φ={phi:.4f}, σ²={sigma_sq:.4f}")

    return history


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(config_entry):
    """Run full experiment for one configuration."""
    print("\n" + "="*80)
    print(f"Testing: {config_entry['name']} - {config_entry['desc']}")
    print("="*80)

    hidden_dims = config_entry['hidden_dims']
    param_count = compute_param_count(hidden_dims)
    print(f"Hidden dims: {hidden_dims}")
    print(f"Estimated params: {param_count:,}")
    print()

    # Set seed
    set_seeds(42)

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    cfg = BackfillContext60ConfigV4FullCov
    surfaces = data['surface'][cfg.train_start_idx:cfg.train_end_idx]
    ex_data = np.stack([
        data['ret'][cfg.train_start_idx:cfg.train_end_idx],
        data['skews'][cfg.train_start_idx:cfg.train_end_idx],
        data['slopes'][cfg.train_start_idx:cfg.train_end_idx]
    ], axis=1)

    # Create datasets (Phase 1 only: seq_len = context+1)
    split_idx = int(0.8 * len(surfaces))
    train_dataset = VolSurfaceDataSetRand(
        (surfaces[:split_idx], ex_data[:split_idx]),
        min_seq_len=61, max_seq_len=61,  # 60 context + 1 target
        dtype=torch.float32
    )
    valid_dataset = VolSurfaceDataSetRand(
        (surfaces[split_idx:], ex_data[split_idx:]),
        min_seq_len=61, max_seq_len=61,
        dtype=torch.float32
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=CustomBatchSampler(train_dataset, cfg.batch_size, 61),
        pin_memory=True, num_workers=2
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=CustomBatchSampler(valid_dataset, cfg.valid_batch_size, 61),
        pin_memory=True, num_workers=2
    )

    # Create model with this hidden_dims
    print("Initializing model...")
    model_config = {
        "seq_len": 200,
        "feat_dim": (5, 5),
        "latent_dim": cfg.latent_dim,
        "kl_weight": cfg.kl_weight,
        "re_feat_weight": cfg.re_feat_weight,
        "surface_hidden": cfg.surface_hidden,
        "ctx_surface_hidden": cfg.surface_hidden,
        "ex_feats_dim": 3,
        "ex_feats_hidden": None,
        "ctx_ex_feats_hidden": None,
        "mem_type": "lstm",
        "mem_hidden": cfg.mem_hidden,
        "mem_layers": cfg.mem_layers,
        "mem_dropout": cfg.mem_dropout,
        "interaction_layers": 2,
        "use_dense_surface": False,
        "compress_context": True,
        "ex_loss_on_ret_only": cfg.ex_loss_on_ret_only,
        "ex_feats_loss_type": cfg.ex_feats_loss_type,
        "device": DEVICE,
        "horizon": 1,  # Phase 1 only
        "context_len": cfg.context_len,
        "max_horizon": 90,
        "full_cov_pos_dim": cfg.full_cov_pos_dim,
        "full_cov_hidden_dims": hidden_dims,  # TEST THIS
        "full_cov_dropout": cfg.full_cov_dropout,
        "full_cov_init_phi": cfg.full_cov_init_phi,
        "full_cov_init_sigma_sq": cfg.full_cov_init_sigma_sq,
    }

    model = CVAEFullCovPrior(model_config).to(DEVICE)
    actual_params = sum(p.numel() for p in model.full_cov_prior.mean_network.parameters())
    print(f"✓ Model initialized (actual mean network params: {actual_params:,})")

    # Train
    print("\nTraining...")
    history = train_quick(model, train_loader, valid_loader, DEVICE, num_epochs=QUICK_EPOCHS)

    # Compute metrics
    print("\nComputing metrics...")

    # 1. Prior-context correlation
    print("  1. Prior-context correlation...")
    mean_corr, std_corr, _ = compute_prior_context_correlation(model, valid_loader, DEVICE)
    print(f"     Mean correlation: {mean_corr:.4f} ± {std_corr:.4f}")

    # 2. Sample diversity
    print("  2. Sample diversity...")
    mean_div, std_div = measure_sample_diversity(model, valid_loader, DEVICE, num_samples=10)
    print(f"     Mean diversity: {mean_div:.6f} ± {std_div:.6f}")

    # 3. Roughness ratio (skip for quick test - requires longer sequences)
    # print("  3. Roughness ratio...")
    # roughness_ratio, _ = compute_roughness_ratio(model, valid_loader, DEVICE)
    # print(f"     Roughness ratio: {roughness_ratio:.2%}")

    # Collect results
    results = {
        "config": config_entry,
        "param_count": actual_params,
        "final_val_loss": history['val_loss'][-1],
        "final_phi": history['phi'][-1],
        "final_sigma_sq": history['sigma_sq'][-1],
        "prior_context_correlation": {
            "mean": float(mean_corr),
            "std": float(std_corr)
        },
        "sample_diversity": {
            "mean": float(mean_div),
            "std": float(std_div)
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.items()}
    }

    return results


def main():
    print("="*80)
    print("PRIOR NETWORK CAPACITY SWEEP EXPERIMENT")
    print("="*80)
    print()
    print("Testing configurations:")
    for cfg in CONFIGS_TO_TEST:
        params = compute_param_count(cfg['hidden_dims'])
        print(f"  {cfg['name']:10s}: {str(cfg['hidden_dims']):15s} ({params:6,} params) - {cfg['desc']}")
    print()

    # Create output directory
    output_dir = Path("results/context60_v4_full_cov/capacity_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    for config_entry in CONFIGS_TO_TEST:
        results = run_experiment(config_entry)
        all_results.append(results)

        # Save individual result
        result_file = output_dir / f"{config_entry['name']}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to {result_file}")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_{timestamp}.json"
    summary = {
        "experiment": "Prior Network Capacity Sweep",
        "date": timestamp,
        "configs_tested": len(CONFIGS_TO_TEST),
        "epochs_per_config": QUICK_EPOCHS,
        "results": all_results
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"{'Config':<10} {'Params':>10} {'Val Loss':>10} {'φ':>8} {'σ²':>8} {'Corr':>8} {'Diversity':>10}")
    print("-"*80)
    for r in all_results:
        print(f"{r['config']['name']:<10} "
              f"{r['param_count']:>10,} "
              f"{r['final_val_loss']:>10.6f} "
              f"{r['final_phi']:>8.4f} "
              f"{r['final_sigma_sq']:>8.4f} "
              f"{r['prior_context_correlation']['mean']:>8.4f} "
              f"{r['sample_diversity']['mean']:>10.6f}")
    print()
    print(f"✓ Summary saved to {summary_file}")
    print()
    print("Interpretation:")
    print("  - Lower correlation = less over-discrimination (target: <0.6)")
    print("  - Higher diversity = more diverse samples from similar contexts")
    print("  - Similar val loss = simpler network doesn't hurt performance")


if __name__ == "__main__":
    main()
