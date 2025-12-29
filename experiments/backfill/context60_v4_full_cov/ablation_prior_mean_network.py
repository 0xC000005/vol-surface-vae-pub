"""
Ablation Study: MLP vs RNN Prior Mean Networks

Compares 5 configurations to test the hypothesis that RNN-based prior means
with temporal dependency produce smoother predictions than MLP + position encoding.

Configurations:
1. mlp_pos: MLP [64,32] + position encoding (baseline)
2. lstm_nopos: LSTM h=32 without position encoding
3. lstm_pos: LSTM h=32 with position encoding
4. gru_nopos: GRU h=32 without position encoding
5. gru_pos: GRU h=32 with position encoding

Quick Test: 20 epochs per config (~50 min total)
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import defaultdict
from pathlib import Path
import json
import pandas as pd

from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.utils import set_seeds, model_eval
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov
from torch.amp import autocast, GradScaler


# ==============================================================================
# Configuration
# ==============================================================================

CONFIGS = [
    {"name": "mlp_pos", "type": "mlp", "use_pos_enc": True, "desc": "MLP + Position Encoding (Baseline)"},
    {"name": "lstm_nopos", "type": "lstm", "use_pos_enc": False, "desc": "LSTM without Position Encoding"},
    {"name": "lstm_pos", "type": "lstm", "use_pos_enc": True, "desc": "LSTM + Position Encoding"},
    {"name": "gru_nopos", "type": "gru", "use_pos_enc": False, "desc": "GRU without Position Encoding"},
    {"name": "gru_pos", "type": "gru", "use_pos_enc": True, "desc": "GRU + Position Encoding"},
]

parser = argparse.ArgumentParser(description='Prior Mean Network Ablation Study')
parser.add_argument('--epochs', type=int, default=20, help='Total epochs to train (default: 20)')
parser.add_argument('--phase1_end', type=int, default=10, help='Phase 1 end epoch (default: 10)')
parser.add_argument('--output_dir', type=str, default='results/prior_mean_ablation',
                   help='Output directory for results (default: results/prior_mean_ablation)')
parser.add_argument('--run_configs', type=str, default='all',
                   help='Comma-separated config names to run (default: all)')
args = parser.parse_args()


# ==============================================================================
# Dataset Creation
# ==============================================================================

def create_datasets(vol_data, ex_data, seq_len_range):
    """Create train/valid datasets with specified sequence length range."""
    min_len, max_len = seq_len_range
    split_idx = int(0.8 * len(vol_data))

    train_dataset = VolSurfaceDataSetRand(
        (vol_data[:split_idx], ex_data[:split_idx]),
        min_seq_len=min_len,
        max_seq_len=max_len,
        dtype=torch.float32
    )

    valid_dataset = VolSurfaceDataSetRand(
        (vol_data[split_idx:], ex_data[split_idx:]),
        min_seq_len=min_len,
        max_seq_len=max_len,
        dtype=torch.float32
    )

    return train_dataset, valid_dataset


def create_dataloaders(train_dataset, valid_dataset, batch_size, valid_batch_size):
    """Create dataloaders with custom batch sampler."""
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=CustomBatchSampler(
            train_dataset,
            batch_size,
            train_dataset.seq_lens[0]
        ),
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=CustomBatchSampler(
            valid_dataset,
            valid_batch_size,
            valid_dataset.seq_lens[0]
        ),
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2
    )

    return train_loader, valid_loader


# ==============================================================================
# Training Functions
# ==============================================================================

def train_one_epoch(model, optimizer, train_loader, device, scaler):
    """Train one epoch."""
    model.train()
    metrics = defaultdict(float)
    num_batches = 0

    for batch in train_loader:
        losses = model.train_step(batch, optimizer, scaler=scaler)
        for k, v in losses.items():
            metrics[k] += v.item() if hasattr(v, 'item') else v
        num_batches += 1

    for key in metrics:
        metrics[key] /= num_batches

    return dict(metrics)


def validate(model, valid_loader, device):
    """Validate model."""
    return model_eval(model, valid_loader)


# ==============================================================================
# Single Configuration Training
# ==============================================================================

def train_config(config, cfg, surfaces, ex_data, device, total_epochs, phase1_end, output_dir):
    """Train a single configuration."""
    config_name = config["name"]
    print("\n" + "=" * 80)
    print(f"Training: {config_name} - {config['desc']}")
    print("=" * 80)

    # Set seeds for reproducibility
    set_seeds(42)

    # Create model config
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
        "device": device,
        "horizon": 90,
        "context_len": cfg.context_len,
        "max_horizon": cfg.max_horizon,
        # Prior mean network configuration
        "mean_network_type": config["type"],
        "use_position_encoding": config["use_pos_enc"],
        "full_cov_pos_dim": cfg.full_cov_pos_dim,
        "full_cov_hidden_dims": cfg.full_cov_hidden_dims,
        "rnn_hidden_dim": 32,  # Match MLP capacity (~6K params)
        "rnn_num_layers": 1,
        "full_cov_dropout": cfg.full_cov_dropout,
        "full_cov_init_phi": cfg.full_cov_init_phi,
        "full_cov_init_sigma_sq": cfg.full_cov_init_sigma_sq,
    }

    # Create model
    model = CVAEFullCovPrior(model_config)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    prior_params = sum(p.numel() for p in model.full_cov_prior.parameters())
    mean_net_params = sum(p.numel() for p in model.full_cov_prior.mean_network.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Prior parameters: {prior_params:,}")
    print(f"  Mean network parameters: {mean_net_params:,}")
    print(f"  Initial φ: {model.full_cov_prior.get_phi().item():.4f}")
    print(f"  Initial σ²: {model.full_cov_prior.get_sigma_sq().item():.4f}")

    # Optimizer and scaler
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler('cuda')

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'kl_loss': [],
        'phi': [],
        'sigma_sq': []
    }

    # Training loop
    for epoch in range(total_epochs):
        # Determine phase
        if epoch < phase1_end:
            seq_len = cfg.phase1_seq_len
            horizon = 1
        else:
            seq_len = cfg.phase2_seq_len
            horizon = max(cfg.phase2_horizons)

        # Create datasets for current phase
        train_dataset, valid_dataset = create_datasets(surfaces, ex_data, seq_len)
        train_loader, valid_loader = create_dataloaders(
            train_dataset, valid_dataset, cfg.batch_size, cfg.valid_batch_size
        )

        # Set model horizon
        model.horizon = horizon

        # KL annealing
        kl_anneal_epochs = min(10, phase1_end)  # Faster annealing for quick test
        if epoch < kl_anneal_epochs:
            model.kl_weight = cfg.kl_weight * (epoch / kl_anneal_epochs)
        else:
            model.kl_weight = cfg.kl_weight

        # Train and validate
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, scaler)
        val_metrics = validate(model, valid_loader, device)

        # Get current φ and σ²
        phi = model.full_cov_prior.get_phi().item()
        sigma_sq = model.full_cov_prior.get_sigma_sq().item()

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['kl_loss'].append(val_metrics['kl_loss'])
        history['phi'].append(phi)
        history['sigma_sq'].append(sigma_sq)

        # Print progress
        phase_name = "Phase 1" if epoch < phase1_end else "Phase 2"
        print(f"Epoch {epoch}/{total_epochs-1} ({phase_name}): "
              f"Train Loss={train_metrics['loss']:.6f}, "
              f"Val Loss={val_metrics['loss']:.6f}, "
              f"KL={val_metrics['kl_loss']:.6f}, "
              f"φ={phi:.4f}, σ²={sigma_sq:.4f}")

    # Save final metrics
    final_metrics = {
        'config_name': config_name,
        'network_type': config['type'],
        'use_position_encoding': config['use_pos_enc'],
        'total_params': total_params,
        'mean_net_params': mean_net_params,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_kl_loss': history['kl_loss'][-1],
        'final_phi': history['phi'][-1],
        'final_sigma_sq': history['sigma_sq'][-1],
        'mean_val_loss': np.mean(history['val_loss'][-5:]),  # Last 5 epochs
        'history': history
    }

    return final_metrics


# ==============================================================================
# Main Experiment
# ==============================================================================

def main():
    print("=" * 80)
    print("PRIOR MEAN NETWORK ABLATION STUDY")
    print("=" * 80)
    print(f"Total epochs: {args.epochs}")
    print(f"Phase 1 end: {args.phase1_end}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load config
    cfg = BackfillContext60ConfigV4FullCov

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Enable TF32
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for CUDA")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data['surface'][cfg.train_start_idx:cfg.train_end_idx]
    ex_data = np.stack([
        data['ret'][cfg.train_start_idx:cfg.train_end_idx],
        data['skews'][cfg.train_start_idx:cfg.train_end_idx],
        data['slopes'][cfg.train_start_idx:cfg.train_end_idx]
    ], axis=1)
    print(f"Loaded {len(surfaces)} days of data")

    # Filter configs to run
    if args.run_configs != 'all':
        config_names = args.run_configs.split(',')
        configs_to_run = [c for c in CONFIGS if c['name'] in config_names]
    else:
        configs_to_run = CONFIGS

    print(f"\nRunning {len(configs_to_run)} configurations:")
    for c in configs_to_run:
        print(f"  - {c['name']}: {c['desc']}")

    # Train all configurations
    results = []
    for config in configs_to_run:
        metrics = train_config(
            config, cfg, surfaces, ex_data, device,
            args.epochs, args.phase1_end, args.output_dir
        )
        results.append(metrics)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    # Create comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    df_data = []
    for r in results:
        df_data.append({
            'Config': r['config_name'],
            'Network': r['network_type'],
            'Pos Enc': 'Yes' if r['use_position_encoding'] else 'No',
            'Mean Net Params': f"{r['mean_net_params']:,}",
            'Final Val Loss': f"{r['final_val_loss']:.6f}",
            'Final KL': f"{r['final_kl_loss']:.6f}",
            'φ': f"{r['final_phi']:.4f}",
            'σ²': f"{r['final_sigma_sq']:.4f}",
        })

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))

    # Save table
    table_file = output_dir / "comparison_table.csv"
    df.to_csv(table_file, index=False)
    print(f"\n✓ Comparison table saved to {table_file}")

    # Identify best configuration
    best_config = min(results, key=lambda x: x['final_val_loss'])
    print(f"\n🏆 Best Configuration: {best_config['config_name']}")
    print(f"   Validation Loss: {best_config['final_val_loss']:.6f}")
    print(f"   Network: {best_config['network_type']}, Position Encoding: {best_config['use_position_encoding']}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
