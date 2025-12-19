"""
Generate VAE Teacher Forcing Sequences for Context60 Model

Generates full in-sequence data for all horizons (H=1, 7, 14, 30, 60, 90) using teacher forcing.
For each horizon H, generates (n_days, H, 3, 5, 5) arrays containing ALL days in the forecast.

Supports two sampling modes:
- **oracle**: Posterior sampling q(z|context,target) via model.forward()
             Requires future target data (upper bound performance)
             Uses: z ~ q(z|context, target) - samples from posterior
             Generation: Single-pass, H days in one forward call

- **prior**: Prior sampling via model.get_surface_given_conditions()
            Context only, realistic deployment scenario (no future knowledge)
            Uses: z[:,:C] = posterior mean (context), z[:,C:] ~ N(0,1) (future)
            Generation: Single-pass teacher forcing, H days in one forward call
            Always conditions on real 60-day context (no prediction feedback)

Both modes use teacher forcing: ALL forecasts condition on real historical data,
not on previous predictions. This ensures fair comparison where the only difference
is the latent variable sampling strategy (posterior vs prior).

Output: NPZ files in results/context60_baseline/predictions/teacher_forcing/{sampling_mode}/
- vae_tf_crisis_h{1,7,14,30,60,90}.npz
- vae_tf_insample_h{1,7,14,30,60,90}.npz
- vae_tf_oos_h{1,7,14,30,60,90}.npz

Usage:
    # Oracle sampling (posterior, with future knowledge)
    python experiments/backfill/context60/teacher_forcing/generate_vae_tf_sequences.py --period crisis --sampling_mode oracle

    # Prior sampling (realistic, no future knowledge)
    python experiments/backfill/context60/teacher_forcing/generate_vae_tf_sequences.py --period crisis --sampling_mode prior
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


def generate_and_save_horizon(model, vol_surf, ex_data,
                              period_start, period_end,
                              horizon, period_name, sampling_mode, prior_mode="standard"):
    """
    Generate full in-sequence data for a single horizon using teacher forcing.

    Supports two sampling strategies:
    - oracle: model.forward(context+target) → z ~ q(z|context,target)
    - prior: model.get_surface_given_conditions(context) → z[:,:C]=posterior_mean, z[:,C:]~N(0,1)

    Args:
        model: Loaded CVAEMemRand model
        vol_surf: Volatility surface data (N, 5, 5)
        ex_data: Extra features data (N, 3)
        period_start: Start index of period
        period_end: End index of period (inclusive)
        horizon: Forecast horizon (1, 7, 14, 30, 60, or 90)
        period_name: Period identifier ('crisis', 'insample', 'oos')
        sampling_mode: Sampling strategy ('oracle' or 'prior')
    """
    context_len = 60  # CHANGED from 20

    # Calculate valid range
    # Need: context_len days before + horizon days after
    min_start = period_start + context_len
    max_start = period_end - horizon + 1
    num_sequences = max_start - min_start + 1

    print(f"\n{'='*80}")
    print(f"PERIOD: {period_name.upper()} | HORIZON: H={horizon} days | MODE: {sampling_mode.upper()}")
    print(f"{'='*80}")
    print(f"  Period range: [{period_start}, {period_end}]")
    print(f"  Valid start indices: [{min_start}, {max_start}]")
    print(f"  Number of sequences: {num_sequences}")
    print(f"  Sampling strategy: {sampling_mode}")
    print()

    # Storage
    sequences = []
    indices = []

    # Temporarily set model horizon
    original_horizon = model.horizon
    model.horizon = horizon

    with torch.no_grad():
        for seq_start in tqdm(range(min_start, max_start + 1),
                              desc=f"  {period_name} H={horizon} ({sampling_mode})"):
            full_seq_start = seq_start - context_len

            if sampling_mode == "oracle":
                # ORACLE/POSTERIOR SAMPLING: Use full sequence (context + target)
                # Model encodes entire sequence and samples z ~ q(z|context, target)
                full_seq_end = seq_start + horizon
                seq_surface = vol_surf[full_seq_start:full_seq_end]  # (context_len+H, 5, 5)
                seq_ex_feats = ex_data[full_seq_start:full_seq_end]  # (context_len+H, 3)

                x = {
                    "surface": torch.from_numpy(seq_surface).unsqueeze(0).double(),
                    "ex_feats": torch.from_numpy(seq_ex_feats).unsqueeze(0).double()
                }

                # Forward pass with posterior sampling
                surf_recon, ex_recon, z_mean, z_logvar, z = model.forward(x)
                forecast = surf_recon  # (1, H, 3, 5, 5)

            elif sampling_mode == "prior":
                # PRIOR/REALISTIC SAMPLING: Teacher forcing with hybrid latent
                # z[:,:C] = posterior mean from context (deterministic)
                # z[:,C:] ~ N(0,1) (stochastic future, sampled from prior)
                context_end = seq_start
                context_surface = vol_surf[full_seq_start:context_end]  # (context_len, 5, 5)
                context_ex_feats = ex_data[full_seq_start:context_end]  # (context_len, 3)

                context = {
                    "surface": torch.from_numpy(context_surface).unsqueeze(0).double(),
                    "ex_feats": torch.from_numpy(context_ex_feats).unsqueeze(0).double()
                }

                # Single-pass teacher forcing generation for H days
                # Always conditions on real 60-day context (no prediction feedback)
                # Only difference from oracle: z sampled from prior for future steps
                surf_pred, ex_pred = model.get_surface_given_conditions(
                    context, z=None, mu=0, std=1, horizon=horizon, prior_mode=prior_mode
                )
                forecast = surf_pred  # (1, H, 3, 5, 5)

            else:
                raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

            # Remove batch dimension and convert to numpy
            forecast_np = forecast[0].cpu().numpy()  # (H, 3, 5, 5)

            sequences.append(forecast_np)
            indices.append(seq_start)

    # Restore original horizon
    model.horizon = original_horizon

    # Convert to numpy arrays
    sequences_array = np.array(sequences)  # (n_days, H, 3, 5, 5)
    indices_array = np.array(indices)       # (n_days,)

    print(f"  ✓ Generated {len(sequences)} sequences")
    print(f"  Output shape: {sequences_array.shape}")

    # Save to NPZ file (subdirectory by sampling mode)
    output_dir = Path(f"results/context60_latent12_v2/predictions/teacher_forcing/{sampling_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"vae_tf_{period_name}_h{horizon}.npz"

    np.savez(
        output_file,
        surfaces=sequences_array,      # (n_days, H, 3, 5, 5)
        indices=indices_array,          # (n_days,)
        quantiles=[0.05, 0.50, 0.95],  # Quantile levels
        horizon=horizon,                # Horizon value
        period_start=period_start,      # Period metadata
        period_end=period_end,
        method='teacher_forcing',       # Method identifier
        sampling_mode=sampling_mode,    # Sampling strategy (oracle/prior)
        context_len=60                  # ADDED: Context length metadata
    )

    print(f"  ✓ Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")


def main():
    """Main generation pipeline."""
    parser = argparse.ArgumentParser(
        description='Generate VAE teacher forcing sequences for context60 model'
    )
    parser.add_argument('--period', type=str, required=True,
                       choices=['crisis', 'insample', 'oos', 'gap'],
                       help='Period to generate (crisis/insample/oos/gap)')
    parser.add_argument('--sampling_mode', type=str, default='oracle',
                       choices=['oracle', 'prior'],
                       help='Sampling strategy: oracle (posterior, with future) or prior (realistic, context only)')
    parser.add_argument('--prior_mode', type=str, default='standard',
                       choices=['standard', 'fitted'],
                       help='Prior sampling mode (only for --sampling_mode prior): standard=N(0,1), fitted=GMM')
    parser.add_argument('--fitted_prior_path', type=str,
                       default='models/backfill/context60_experiment/fitted_prior_gmm.pt',
                       help='Path to fitted prior parameters (only for --prior_mode fitted)')
    args = parser.parse_args()

    # Set seeds and dtype
    set_seeds(0)
    torch.set_default_dtype(torch.float64)

    print("=" * 80)
    print("VAE TEACHER FORCING SEQUENCE GENERATION (CONTEXT60)")
    print("=" * 80)
    print(f"Period: {args.period}")
    print(f"Sampling mode: {args.sampling_mode}")
    if args.sampling_mode == 'prior':
        print(f"Prior mode: {args.prior_mode}")
        if args.prior_mode == 'fitted':
            print(f"Fitted prior path: {args.fitted_prior_path}")
    print()

    # ========================================================================
    # Load Model
    # ========================================================================

    print("Loading model...")
    model_file = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_best.pt"
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

    # Load fitted prior if using fitted mode
    if args.sampling_mode == 'prior' and args.prior_mode == 'fitted':
        print(f"\nLoading fitted prior from {args.fitted_prior_path}...")
        model.load_fitted_prior(args.fitted_prior_path)
        print("✓ Fitted prior loaded")

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

    horizons = [1, 7, 14, 30, 60, 90]  # CHANGED: Added 60, 90

    print(f"\n{'='*80}")
    print(f"GENERATING FOR ALL HORIZONS: {horizons}")
    print(f"{'='*80}")

    for horizon in horizons:
        generate_and_save_horizon(
            model, vol_surf, ex_data,
            period_start, period_end,
            horizon, args.period, args.sampling_mode,
            prior_mode=args.prior_mode if args.sampling_mode == 'prior' else 'standard'
        )

    # ========================================================================
    # Summary
    # ========================================================================

    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Generated {len(horizons)} files for period '{args.period}' with sampling mode '{args.sampling_mode}'")
    print(f"\nOutput directory: results/context60_baseline/predictions/teacher_forcing/{args.sampling_mode}/")
    print(f"Files created:")
    for h in horizons:
        print(f"  - vae_tf_{args.period}_h{h}.npz")
    print()
    print("Next steps:")
    print(f"  1. Validate outputs: python experiments/backfill/context60/teacher_forcing/validate_vae_tf_sequences.py --sampling_mode {args.sampling_mode}")
    print(f"  2. Run analysis scripts with context60 data")
    print()


if __name__ == "__main__":
    main()
