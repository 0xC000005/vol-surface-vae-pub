"""
Oracle Z Autocorrelation Analysis and Solution Viability Investigation

Investigates whether temporal structure in latent z is essential for achieving
oracle's 75% roughness ratio, and evaluates different horizon lengths.

Key Questions:
1. Do oracle z values have temporal correlation?
2. What's the optimal horizon length (H=30, 60, 90)?
3. Which prior solution is viable for each horizon?

Author: Claude
Date: 2025-12-23
"""

import numpy as np
import torch
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_phase2_ep599.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"

CONTEXT_LEN = 60
HORIZONS = [30, 60, 90]  # Test different horizon lengths
NUM_SEQUENCES = 100
NUM_SAMPLES = 100

OUTPUT_FILE = "/tmp/prior_solution_viability_results.txt"

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

def measure_autocorrelation(z_values, lag=1):
    """
    Measure autocorrelation of z values across time.

    Args:
        z_values: (B, T, latent_dim) tensor
        lag: lag for autocorrelation

    Returns:
        float: autocorrelation coefficient
    """
    if z_values.shape[1] <= lag:
        return 0.0

    z1 = z_values[:, :-lag, :].flatten()
    z2 = z_values[:, lag:, :].flatten()

    # Convert to numpy for correlation
    z1 = z1.cpu().numpy()
    z2 = z2.cpu().numpy()

    corr = np.corrcoef(z1, z2)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def measure_temporal_variance(z_values):
    """
    Measure how much z varies across time vs across samples.

    Args:
        z_values: (B, T, latent_dim) tensor

    Returns:
        dict with temporal and cross-sample variance ratios
    """
    # Variance across time (for each sample)
    temporal_var = z_values.var(dim=1).mean().item()

    # Variance across samples (for each timestep)
    cross_sample_var = z_values.var(dim=0).mean().item()

    return {
        'temporal_var': temporal_var,
        'cross_sample_var': cross_sample_var,
        'ratio': temporal_var / (cross_sample_var + 1e-8)
    }


def load_model_and_data():
    """Load model and data"""
    print(f"Loading model from {MODEL_PATH}...")
    model_data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    model = CVAEMemRandConditionalPrior(model_data["model_config"])

    # Handle torch.compile prefix
    state_dict = model_data["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"✓ Model loaded")

    print(f"\nLoading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    surfaces = torch.tensor(data['surface'], dtype=torch.float32)

    # Construct ex_data
    ret = data['ret']
    skews = data['skews']
    slopes = data['slopes']
    ex_data = np.stack([ret, skews, slopes], axis=1)
    ex_feats = torch.tensor(ex_data, dtype=torch.float32)

    print(f"✓ Loaded {len(surfaces)} days of data")

    return model, surfaces, ex_feats


def get_oracle_z_for_horizon(model, surfaces, ex_feats, horizon, num_sequences=100):
    """
    Get oracle (posterior) z values for a specific horizon.

    Args:
        model: The VAE model
        surfaces: All surface data
        ex_feats: All extra features
        horizon: Forecast horizon
        num_sequences: Number of sequences to analyze

    Returns:
        z_future: (num_sequences, horizon, latent_dim) - oracle z values
    """
    T = CONTEXT_LEN + horizon
    valid_starts = len(surfaces) - T

    # Sample random sequences
    indices = np.random.choice(valid_starts, num_sequences, replace=False)

    z_futures = []

    with torch.no_grad():
        for idx in indices:
            # Full sequence (context + target)
            full_surf = surfaces[idx:idx+T].unsqueeze(0).to(DEVICE)
            full_ex = ex_feats[idx:idx+T].unsqueeze(0).to(DEVICE)

            full_input = {
                'surface': full_surf,
                'ex_feats': full_ex
            }

            # Get posterior z (oracle - sees target)
            z_mean, z_logvar, z = model.encoder(full_input)

            # Extract future portion
            z_future = z_mean[0, CONTEXT_LEN:, :].cpu()  # Use mean for deterministic analysis
            z_futures.append(z_future)

    return torch.stack(z_futures)  # (num_sequences, horizon, latent_dim)


#------------------------------------------------------------------------------
# Main Investigation
#------------------------------------------------------------------------------

def main():
    print("="*80)
    print("ORACLE Z AUTOCORRELATION & SOLUTION VIABILITY INVESTIGATION")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Horizons: {HORIZONS}")
    print(f"Sequences: {NUM_SEQUENCES}")

    # Load model and data
    model, surfaces, ex_feats = load_model_and_data()

    results = {}

    # Analyze each horizon
    for horizon in HORIZONS:
        print(f"\n{'='*80}")
        print(f"HORIZON: {horizon} days")
        print(f"{'='*80}")

        # Get oracle z values
        print(f"\nGetting oracle z values...")
        z_oracle = get_oracle_z_for_horizon(model, surfaces, ex_feats, horizon, NUM_SEQUENCES)
        print(f"Oracle z shape: {z_oracle.shape}")

        # Measure autocorrelation at different lags
        print(f"\nMeasuring autocorrelation...")
        autocorrs = {}
        for lag in [1, 5, 10]:
            if horizon > lag:
                autocorr = measure_autocorrelation(z_oracle, lag=lag)
                autocorrs[f'lag_{lag}'] = autocorr
                print(f"  Lag {lag}: {autocorr:.4f}")

        # Measure temporal variance
        print(f"\nMeasuring variance structure...")
        var_stats = measure_temporal_variance(z_oracle)
        print(f"  Temporal variance: {var_stats['temporal_var']:.6f}")
        print(f"  Cross-sample variance: {var_stats['cross_sample_var']:.6f}")
        print(f"  Ratio (temporal/cross-sample): {var_stats['ratio']:.4f}")

        # Determine recommendation
        lag1_autocorr = autocorrs.get('lag_1', 0.0)

        if lag1_autocorr > 0.3:
            recommendation = "P1 (Autoregressive) - High temporal correlation"
        elif lag1_autocorr > 0.1:
            recommendation = "P5 (Hierarchical) or P2+regularization - Moderate correlation"
        else:
            recommendation = "P2 (Position-Encoded) may suffice - Low correlation"

        print(f"\n  → Recommendation: {recommendation}")

        # Estimate computational costs
        print(f"\nEstimating computational costs...")

        # P1 (Autoregressive): sequential
        p1_cost_per_sample = horizon * 0.1  # 0.1ms per RNN pass (estimate)
        p1_total_ms = p1_cost_per_sample * NUM_SAMPLES * NUM_SEQUENCES

        # P2 (Position-Encoded): parallel
        p2_cost_per_sample = 0.1  # Single forward pass (estimate)
        p2_total_ms = p2_cost_per_sample * NUM_SAMPLES * NUM_SEQUENCES

        print(f"  P1 (Autoregressive): ~{p1_total_ms/1000:.1f}s for {NUM_SAMPLES}×{NUM_SEQUENCES} samples")
        print(f"  P2 (Position-Encoded): ~{p2_total_ms/1000:.1f}s for {NUM_SAMPLES}×{NUM_SEQUENCES} samples")
        print(f"  Slowdown factor: {p1_total_ms/p2_total_ms:.1f}×")

        # Store results
        results[horizon] = {
            'autocorrs': autocorrs,
            'var_stats': var_stats,
            'recommendation': recommendation,
            'p1_cost_seconds': p1_total_ms / 1000,
            'p2_cost_seconds': p2_total_ms / 1000
        }

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    with open(OUTPUT_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ORACLE Z AUTOCORRELATION & SOLUTION VIABILITY INVESTIGATION\n")
        f.write("="*80 + "\n\n")

        f.write(f"Context length: {CONTEXT_LEN}\n")
        f.write(f"Sequences analyzed: {NUM_SEQUENCES}\n\n")

        for horizon in HORIZONS:
            r = results[horizon]
            f.write(f"\n{'='*80}\n")
            f.write(f"HORIZON: {horizon} days\n")
            f.write(f"{'='*80}\n\n")

            f.write("Autocorrelation:\n")
            for lag, autocorr in r['autocorrs'].items():
                f.write(f"  {lag}: {autocorr:.4f}\n")

            f.write(f"\nVariance Structure:\n")
            f.write(f"  Temporal variance: {r['var_stats']['temporal_var']:.6f}\n")
            f.write(f"  Cross-sample variance: {r['var_stats']['cross_sample_var']:.6f}\n")
            f.write(f"  Ratio: {r['var_stats']['ratio']:.4f}\n")

            f.write(f"\nComputational Cost (for {NUM_SAMPLES}×{NUM_SEQUENCES} samples):\n")
            f.write(f"  P1 (Autoregressive): {r['p1_cost_seconds']:.1f}s\n")
            f.write(f"  P2 (Position-Encoded): {r['p2_cost_seconds']:.1f}s\n")
            f.write(f"  Slowdown: {r['p1_cost_seconds']/r['p2_cost_seconds']:.1f}×\n")

            f.write(f"\nRecommendation: {r['recommendation']}\n")

        # Summary and recommendations
        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY & RECOMMENDATIONS\n")
        f.write(f"{'='*80}\n\n")

        # Find horizon with best trade-off
        lag1_autocorrs = {h: results[h]['autocorrs'].get('lag_1', 0.0) for h in HORIZONS}

        f.write("Key Findings:\n\n")
        f.write("1. Temporal Correlation in Oracle Z:\n")
        for h, autocorr in lag1_autocorrs.items():
            if autocorr > 0.3:
                level = "HIGH"
            elif autocorr > 0.1:
                level = "MODERATE"
            else:
                level = "LOW"
            f.write(f"   H={h}: lag-1 autocorr = {autocorr:.4f} ({level})\n")

        f.write("\n2. Recommended Approaches by Horizon:\n\n")
        for h in HORIZONS:
            f.write(f"   H={h}:\n")
            f.write(f"     {results[h]['recommendation']}\n")
            f.write(f"     Cost: P1={results[h]['p1_cost_seconds']:.1f}s vs P2={results[h]['p2_cost_seconds']:.1f}s\n\n")

        f.write("3. Horizon Selection Strategy:\n\n")
        f.write("   Option A: Shorter Horizon + Autoregressive Extension\n")
        f.write("   - Use H=30 for direct prediction (faster training, fewer 'modes')\n")
        f.write("   - Chain predictions autoregressively for longer sequences\n")
        f.write("   - Example: 90-day forecast = 3× sequential 30-day predictions\n")
        f.write(f"   - Pros: {30} 'modes' to learn vs {90} for H=90\n")
        f.write("   - Cons: Error accumulation in chained predictions\n\n")

        f.write("   Option B: Direct Long Horizon\n")
        f.write("   - Use H=90 for single-shot prediction\n")
        f.write("   - No error accumulation\n")
        f.write(f"   - Cons: {90} 'modes' to learn, harder training\n\n")

        # Final recommendation
        avg_lag1 = np.mean(list(lag1_autocorrs.values()))

        f.write("4. Final Recommendation:\n\n")

        if avg_lag1 > 0.3:
            f.write("   ⭐ P1 (Autoregressive Prior) with H=30\n")
            f.write("   - Oracle z has HIGH temporal correlation (avg {:.3f})\n".format(avg_lag1))
            f.write("   - Temporal structure is essential for realistic paths\n")
            f.write("   - Use H=30 to reduce sequential cost (30× vs 90×)\n")
            f.write("   - Extend to longer horizons via autoregressive chaining if needed\n")
        elif avg_lag1 > 0.1:
            f.write("   ⭐ P5 (Hierarchical/Chunked Prior) with H=30 or H=60\n")
            f.write("   - Oracle z has MODERATE temporal correlation (avg {:.3f})\n".format(avg_lag1))
            f.write("   - Chunk into smaller units (e.g., 6×5 or 6×10 days)\n")
            f.write("   - AR within chunks, parallel across chunks\n")
            f.write("   - Balances temporal structure and computational efficiency\n")
        else:
            f.write("   ⭐ P2 (Position-Encoded) with smoothness regularization\n")
            f.write("   - Oracle z has LOW temporal correlation (avg {:.3f})\n".format(avg_lag1))
            f.write("   - Position encoding may suffice with proper regularization\n")
            f.write("   - Add loss term: λ * ||μ_{t+1} - μ_t||²\n")
            f.write("   - Start with H=30 or H=60 for easier learning\n")

    print(f"✓ Results saved to {OUTPUT_FILE}")
    print("\nDone!")


if __name__ == "__main__":
    main()
