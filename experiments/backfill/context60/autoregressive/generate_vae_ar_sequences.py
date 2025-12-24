"""
Generate VAE Autoregressive Multi-Step Sequences for Context60 Model

Uses model.generate_autoregressive_sequence() for TRUE autoregressive generation:
- Initial context: 60 days of real data
- Iterative: Each prediction feeds back into context
- Long horizons: 180-day (3×60) and 270-day (3×90) sequences

Supports two sampling modes:
- oracle: Posterior sampling (approximate oracle for AR)
- prior: Realistic AR generation (no future knowledge)

Output: NPZ files in results/context60_baseline/predictions/autoregressive_multi_step/{sampling_mode}/
- vae_ar_crisis_{180,270}day.npz
- vae_ar_insample_{180,270}day.npz
- vae_ar_oos_{180,270}day.npz
- vae_ar_gap_{180,270}day.npz

Usage:
    python experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py --period crisis --sampling_mode prior
    python experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py --period oos --sampling_mode oracle
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds


def get_period_config(period_name):
    """Get start/end indices for each period."""
    configs = {
        'crisis': (2000, 2765),      # 2008-2010 financial crisis
        'insample': (0, 3971),        # Full in-sample training data
        'oos': (5000, 5792),          # Out-of-sample test data
        'gap': (3972, 4999),          # Gap period 2015-2019
    }

    if period_name not in configs:
        raise ValueError(f"Unknown period: {period_name}. Choose from: {list(configs.keys())}")

    return configs[period_name]


def generate_and_save_ar_sequences(model, vol_surf, ex_data,
                                    period_start, period_end,
                                    total_horizon, period_name,
                                    sampling_mode):
    """
    Generate autoregressive sequences for all valid dates in period.

    Args:
        model: Loaded CVAEMemRand model
        vol_surf: Volatility surface data (N, 5, 5)
        ex_data: Extra features data (N, 3)
        period_start: Start index of period
        period_end: End index of period (inclusive)
        total_horizon: Total days to generate autoregressively (180 or 270)
        period_name: Period identifier ('crisis', 'insample', 'oos', 'gap')
        sampling_mode: Sampling strategy ('oracle' or 'prior')
    """
    context_len = 60

    # Calculate valid range
    min_start = period_start + context_len
    max_start = period_end - total_horizon + 1
    num_sequences = max_start - min_start + 1

    print(f"\n{'='*80}")
    print(f"PERIOD: {period_name.upper()} | HORIZON: AR-{total_horizon} days | MODE: {sampling_mode.upper()}")
    print(f"{'='*80}")
    print(f"  Period range: [{period_start}, {period_end}]")
    print(f"  Valid start indices: [{min_start}, {max_start}]")
    print(f"  Number of sequences: {num_sequences}")
    print(f"  Sampling strategy: {sampling_mode}")
    print(f"  Method: Autoregressive (iterative day-by-day)")
    print()

    if num_sequences <= 0:
        print(f"  ⚠ WARNING: No valid sequences for {period_name} H={total_horizon}")
        print(f"    Period not long enough for {total_horizon}-day sequences with {context_len}-day context")
        return

    sequences = []
    indices = []

    with torch.no_grad():
        for seq_start in tqdm(range(min_start, max_start + 1),
                              desc=f"  {period_name} AR-{total_horizon} ({sampling_mode})"):
            # Extract 60-day context
            context_start = seq_start - context_len
            context_surface = vol_surf[context_start:seq_start]  # (60, 5, 5)
            context_ex_feats = ex_data[context_start:seq_start]  # (60, 3)

            context = {
                "surface": torch.from_numpy(context_surface).unsqueeze(0).double(),
                "ex_feats": torch.from_numpy(context_ex_feats).unsqueeze(0).double()
            }

            # Determine AR parameters based on total_horizon
            # Training used: Phase 3 (H=60, offset=60, steps=3 → 180 days)
            #                Phase 4 (H=90, offset=90, steps=3 → 270 days)
            if total_horizon == 180:
                ar_steps, horizon, offset = 3, 60, 60
            elif total_horizon == 270:
                ar_steps, horizon, offset = 3, 90, 90
            else:
                raise ValueError(f"Unsupported total_horizon: {total_horizon}")

            # Generate autoregressive sequence using offset-based method (3 model calls)
            # This matches training approach: generate in chunks, not day-by-day
            surf_pred, ex_pred = model.generate_autoregressive_offset(
                initial_context=context,
                ar_steps=ar_steps,
                horizon=horizon,
                offset=offset,
                z=None  # Let model sample z internally
            )

            # Remove batch dimension and convert to numpy
            forecast_np = surf_pred[0].cpu().numpy()  # (total_horizon, 3, 5, 5)

            sequences.append(forecast_np)
            indices.append(seq_start)

    # Convert to numpy arrays
    sequences_array = np.array(sequences)  # (n_days, total_horizon, 3, 5, 5)
    indices_array = np.array(indices)       # (n_days,)

    print(f"  ✓ Generated {len(sequences)} sequences")
    print(f"  Output shape: {sequences_array.shape}")

    # Save to NPZ file (subdirectory by sampling mode)
    output_dir = Path(f"results/context60_baseline/predictions/autoregressive_multi_step/{sampling_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"vae_ar_{period_name}_{total_horizon}day.npz"

    np.savez(
        output_file,
        surfaces=sequences_array,           # (n_days, total_horizon, 3, 5, 5)
        indices=indices_array,               # (n_days,)
        quantiles=[0.05, 0.50, 0.95],       # Quantile levels
        total_horizon=total_horizon,         # Total days generated
        period_start=period_start,           # Period metadata
        period_end=period_end,
        method='autoregressive_multistep',   # Method identifier
        sampling_mode=sampling_mode,         # Sampling strategy (oracle/prior)
        context_len=60                       # Context length
    )

    print(f"  ✓ Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")


def main():
    """Main generation pipeline."""
    parser = argparse.ArgumentParser(
        description='Generate VAE autoregressive multi-step sequences for context60 model'
    )
    parser.add_argument('--period', type=str, required=True,
                       choices=['crisis', 'insample', 'oos', 'gap'],
                       help='Period to generate (crisis/insample/oos/gap)')
    parser.add_argument('--sampling_mode', type=str, default='prior',
                       choices=['oracle', 'prior'],
                       help='Sampling strategy: oracle or prior (default: prior)')
    args = parser.parse_args()

    # Set seeds and dtype
    set_seeds(0)
    torch.set_default_dtype(torch.float64)

    print("=" * 80)
    print("VAE AUTOREGRESSIVE MULTI-STEP SEQUENCE GENERATION (CONTEXT60)")
    print("=" * 80)
    print(f"Period: {args.period}")
    print(f"Sampling mode: {args.sampling_mode}")
    print()

    # ========================================================================
    # Load Model
    # ========================================================================

    print("Loading model...")
    model_file = "models/backfill/context60_experiment/checkpoints/backfill_context60_best.pt"
    model_data = torch.load(model_file, weights_only=False)
    model_config = model_data["model_config"]

    print(f"Model config:")
    print(f"  Context length: {model_config['context_len']}")
    print(f"  Latent dim: {model_config['latent_dim']}")
    print(f"  Quantiles: {model_config['quantiles']}")
    print(f"  Quantile weights: {model_config.get('quantile_loss_weights', [1,1,1])}")

    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()
    print("✓ Model loaded")

    # ========================================================================
    # Load Data
    # ========================================================================

    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    vol_surf = data["surface"]       # (5822, 5, 5)
    ret_data = data["ret"]
    skew_data = data["skews"]
    slope_data = data["slopes"]

    # Concatenate extra features [return, skew, slope]
    ex_data = np.concatenate([
        ret_data[..., np.newaxis],
        skew_data[..., np.newaxis],
        slope_data[..., np.newaxis]
    ], axis=-1)  # (5822, 3)

    print(f"  Volatility surfaces: {vol_surf.shape}")
    print(f"  Extra features: {ex_data.shape}")

    # ========================================================================
    # Get Period Configuration
    # ========================================================================

    period_start, period_end = get_period_config(args.period)
    print(f"\nPeriod configuration for '{args.period}':")
    print(f"  Start index: {period_start}")
    print(f"  End index: {period_end}")
    print(f"  Total days: {period_end - period_start + 1}")

    # ========================================================================
    # Generate All Horizons
    # ========================================================================

    horizons = [180, 270]  # 3×60 and 3×90 day sequences

    print(f"\n{'='*80}")
    print(f"GENERATING FOR ALL AR HORIZONS: {horizons}")
    print(f"{'='*80}")

    for total_horizon in horizons:
        generate_and_save_ar_sequences(
            model, vol_surf, ex_data,
            period_start, period_end,
            total_horizon, args.period, args.sampling_mode
        )

    # ========================================================================
    # Summary
    # ========================================================================

    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Generated {len(horizons)} files for period '{args.period}' with sampling mode '{args.sampling_mode}'")
    print(f"\nOutput directory: results/context60_baseline/predictions/autoregressive_multi_step/{args.sampling_mode}/")
    print(f"Files created:")
    for h in horizons:
        print(f"  - vae_ar_{args.period}_{h}day.npz")
    print()
    print("Next steps:")
    print(f"  1. Validate outputs: python experiments/backfill/context60/autoregressive/validate_vae_ar_sequences.py --sampling_mode {args.sampling_mode}")
    print(f"  2. Compare with teacher forcing results")
    print()


if __name__ == "__main__":
    main()
