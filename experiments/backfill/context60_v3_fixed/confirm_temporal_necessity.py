"""
Additional Experiments to Confirm Temporal Structure Necessity

Tests whether autoregressive prior is TRULY necessary by:
1. Breaking temporal structure (shuffled z) and measuring impact
2. Creating synthetic z with varying autocorrelation levels
3. Testing cheaper alternatives (smoothing, chunking)

Author: Claude
Date: 2025-12-23
"""

import numpy as np
import torch
from pathlib import Path
import sys
from scipy import signal

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
HORIZON = 30  # Use H=30 for faster experiments
NUM_SEQUENCES = 50
NUM_SAMPLES = 100

ATM_6M = (2, 2)  # ATM strike, 6M maturity

OUTPUT_FILE = "/tmp/temporal_necessity_confirmation.txt"

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

def measure_roughness(paths):
    """
    Measure path roughness = std of day-to-day changes.

    Args:
        paths: (num_samples, horizon, H, W) tensor

    Returns:
        float: roughness value
    """
    if len(paths.shape) == 3:
        # (num_samples, horizon, features)
        daily_changes = paths[:, 1:, :] - paths[:, :-1, :]
    else:
        # (num_samples, horizon, H, W)
        daily_changes = paths[:, 1:, :, :] - paths[:, :-1, :, :]

    return daily_changes.std().item()


def measure_autocorr(values, lag=1):
    """Measure autocorrelation at given lag"""
    if values.shape[1] <= lag:
        return 0.0

    v1 = values[:, :-lag, :].flatten()
    v2 = values[:, lag:, :].flatten()

    v1 = v1.cpu().numpy()
    v2 = v2.cpu().numpy()

    corr = np.corrcoef(v1, v2)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def load_model_and_data():
    """Load model and data"""
    print(f"Loading model...")
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

    print(f"\nLoading data...")
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


def get_test_sequences(surfaces, ex_feats, num_sequences):
    """Get random test sequences"""
    T = CONTEXT_LEN + HORIZON
    valid_starts = len(surfaces) - T

    indices = np.random.choice(valid_starts, num_sequences, replace=False)

    sequences = []
    for idx in indices:
        seq = {
            'context_surf': surfaces[idx:idx+CONTEXT_LEN],
            'context_ex': ex_feats[idx:idx+CONTEXT_LEN],
            'target_surf': surfaces[idx+CONTEXT_LEN:idx+T],
            'target_ex': ex_feats[idx+CONTEXT_LEN:idx+T]
        }
        sequences.append(seq)

    return sequences


def generate_with_z(model, sequences, z_values, label=""):
    """
    Generate surfaces using custom z values.

    Args:
        model: VAE model
        sequences: List of context sequences
        z_values: (num_sequences, horizon, latent_dim) - z values to use
        label: Label for logging

    Returns:
        paths: (num_sequences*num_samples, horizon, H, W)
        roughness: float
    """
    print(f"  Generating with {label}...")

    all_paths = []

    for seq_idx, seq in enumerate(sequences):
        context_input = {
            'surface': seq['context_surf'].unsqueeze(0).to(DEVICE),
            'ex_feats': seq['context_ex'].unsqueeze(0).to(DEVICE)
        }

        # Get context encoding
        with torch.no_grad():
            ctx_latent_mean, _, _ = model.encoder(context_input)

        # Use provided z for future
        z_future = z_values[seq_idx].to(DEVICE)  # (horizon, latent_dim)

        for _ in range(NUM_SAMPLES):
            # Build full z (context + future)
            T = CONTEXT_LEN + HORIZON
            z_full = torch.zeros((1, T, model.config['latent_dim']), device=DEVICE)
            z_full[:, :CONTEXT_LEN, :] = ctx_latent_mean
            z_full[:, CONTEXT_LEN:, :] = z_future.unsqueeze(0)

            # Generate
            with torch.no_grad():
                output = model.get_surface_given_conditions(
                    context_input,
                    z=z_full,
                    horizon=HORIZON
                )
                # Unpack if tuple (with ex_feats) or use directly
                surface = output[0] if isinstance(output, tuple) else output

            all_paths.append(surface[0, :, :, :].cpu())

    paths = torch.stack(all_paths)
    roughness = measure_roughness(paths)

    print(f"    Roughness: {roughness:.6f}")

    return paths, roughness


#------------------------------------------------------------------------------
# Experiments
#------------------------------------------------------------------------------

def experiment_1_shuffled_oracle(model, sequences, oracle_z):
    """
    Experiment 1: Shuffle oracle z across time to break temporal structure.

    If roughness drops significantly, confirms temporal correlation is essential.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: SHUFFLED ORACLE Z")
    print("="*80)
    print("\nQuestion: Does breaking temporal structure reduce roughness?")

    # Original oracle
    _, roughness_original = generate_with_z(model, sequences, oracle_z, "Original Oracle Z")

    # Shuffled oracle - shuffle each sequence independently across time
    shuffled_z = oracle_z.clone()
    for i in range(len(shuffled_z)):
        perm = torch.randperm(HORIZON)
        shuffled_z[i] = shuffled_z[i][perm]

    _, roughness_shuffled = generate_with_z(model, sequences, shuffled_z, "Shuffled Oracle Z")

    # Measure autocorrelation
    autocorr_original = measure_autocorr(oracle_z, lag=1)
    autocorr_shuffled = measure_autocorr(shuffled_z, lag=1)

    print(f"\nResults:")
    print(f"  Original oracle - Autocorr: {autocorr_original:.4f}, Roughness: {roughness_original:.6f}")
    print(f"  Shuffled oracle - Autocorr: {autocorr_shuffled:.4f}, Roughness: {roughness_shuffled:.6f}")
    print(f"  Roughness drop: {(1 - roughness_shuffled/roughness_original)*100:.1f}%")

    if roughness_shuffled < roughness_original * 0.7:
        conclusion = "✅ TEMPORAL STRUCTURE IS ESSENTIAL (>30% drop)"
    elif roughness_shuffled < roughness_original * 0.9:
        conclusion = "⚠️ TEMPORAL STRUCTURE HELPS (10-30% drop)"
    else:
        conclusion = "❌ TEMPORAL STRUCTURE NOT CRITICAL (<10% drop)"

    print(f"\n  → {conclusion}")

    return {
        'roughness_original': roughness_original,
        'roughness_shuffled': roughness_shuffled,
        'autocorr_original': autocorr_original,
        'autocorr_shuffled': autocorr_shuffled,
        'drop_percent': (1 - roughness_shuffled/roughness_original) * 100,
        'conclusion': conclusion
    }


def experiment_2_synthetic_autocorr(model, sequences, oracle_z):
    """
    Experiment 2: Create synthetic z with varying autocorrelation levels.

    Tests if there's a relationship between autocorr and roughness.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: SYNTHETIC Z WITH VARYING AUTOCORRELATION")
    print("="*80)
    print("\nQuestion: Is there a relationship between z autocorr and roughness?")

    autocorr_targets = [0.0, 0.3, 0.5, 0.7]
    results = []

    for target_autocorr in autocorr_targets:
        print(f"\n  Testing autocorr = {target_autocorr}")

        # Generate AR(1) process: z_t = phi * z_{t-1} + sqrt(1-phi^2) * eps
        phi = target_autocorr

        synthetic_z = torch.zeros_like(oracle_z)
        for i in range(len(synthetic_z)):
            # Start with random initial value
            z_t = torch.randn(model.config['latent_dim'])
            synthetic_z[i, 0, :] = z_t

            # Generate AR(1) sequence
            for t in range(1, HORIZON):
                innovation = torch.randn(model.config['latent_dim'])
                z_t = phi * z_t + np.sqrt(1 - phi**2) * innovation
                synthetic_z[i, t, :] = z_t

        # Generate and measure
        _, roughness = generate_with_z(model, sequences, synthetic_z, f"Synthetic autocorr={target_autocorr}")
        actual_autocorr = measure_autocorr(synthetic_z, lag=1)

        results.append({
            'target_autocorr': target_autocorr,
            'actual_autocorr': actual_autocorr,
            'roughness': roughness
        })

    print(f"\n  Results:")
    print(f"  {'Target':<10} {'Actual':<10} {'Roughness':<15}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['target_autocorr']:<10.2f} {r['actual_autocorr']:<10.4f} {r['roughness']:<15.6f}")

    # Check if monotonic relationship
    roughnesses = [r['roughness'] for r in results]
    is_increasing = all(roughnesses[i] <= roughnesses[i+1] for i in range(len(roughnesses)-1))

    if is_increasing:
        conclusion = "✅ HIGHER AUTOCORR → HIGHER ROUGHNESS (monotonic)"
    else:
        conclusion = "❌ NO CLEAR RELATIONSHIP between autocorr and roughness"

    print(f"\n  → {conclusion}")

    return {'results': results, 'conclusion': conclusion}


def experiment_3_smoothed_iid(model, sequences, oracle_z):
    """
    Experiment 3: Generate iid z, then apply smoothing.

    Tests if post-hoc smoothing can match oracle without autoregressive generation.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: POST-HOC SMOOTHING OF IID Z")
    print("="*80)
    print("\nQuestion: Can we match oracle roughness by smoothing iid z?")

    # IID (no correlation)
    iid_z = torch.randn_like(oracle_z)
    _, roughness_iid = generate_with_z(model, sequences, iid_z, "IID Z (no smoothing)")
    autocorr_iid = measure_autocorr(iid_z, lag=1)

    # Apply moving average smoothing
    window_sizes = [3, 5, 7]
    results = [{'method': 'iid', 'autocorr': autocorr_iid, 'roughness': roughness_iid}]

    for window in window_sizes:
        smoothed_z = torch.zeros_like(iid_z)

        # Apply moving average to each sequence
        for i in range(len(iid_z)):
            for d in range(model.config['latent_dim']):
                z_series = iid_z[i, :, d].numpy()
                # Simple moving average
                smoothed = np.convolve(z_series, np.ones(window)/window, mode='same')
                smoothed_z[i, :, d] = torch.tensor(smoothed, dtype=torch.float32)

        _, roughness = generate_with_z(model, sequences, smoothed_z, f"Smoothed (window={window})")
        autocorr = measure_autocorr(smoothed_z, lag=1)

        results.append({
            'method': f'smooth_{window}',
            'autocorr': autocorr,
            'roughness': roughness
        })

    # Oracle for comparison
    _, roughness_oracle = generate_with_z(model, sequences, oracle_z, "Oracle Z")
    autocorr_oracle = measure_autocorr(oracle_z, lag=1)

    print(f"\n  Results:")
    print(f"  {'Method':<15} {'Autocorr':<12} {'Roughness':<15} {'vs Oracle':<15}")
    print(f"  {'-'*60}")
    for r in results:
        ratio = r['roughness'] / roughness_oracle * 100
        print(f"  {r['method']:<15} {r['autocorr']:<12.4f} {r['roughness']:<15.6f} {ratio:<15.1f}%")
    print(f"  {'oracle':<15} {autocorr_oracle:<12.4f} {roughness_oracle:<15.6f} {'100.0%':<15}")

    # Find best smoothing (exclude oracle and iid)
    smoothing_results = [r for r in results if r['method'].startswith('smooth_')]
    best = max(smoothing_results, key=lambda x: x['roughness'])
    best_ratio = best['roughness'] / roughness_oracle

    window = best['method'].split('_')[1]
    if best_ratio > 0.9:
        conclusion = f"✅ SMOOTHING WORKS! (window={window} achieves {best_ratio*100:.1f}%)"
    elif best_ratio > 0.7:
        conclusion = f"⚠️ SMOOTHING HELPS (window={window} achieves {best_ratio*100:.1f}%, not quite oracle)"
    else:
        conclusion = f"❌ SMOOTHING INSUFFICIENT (window={window} only {best_ratio*100:.1f}% of oracle)"

    print(f"\n  → {conclusion}")

    return {'results': results, 'oracle': roughness_oracle, 'conclusion': conclusion}


def experiment_4_hierarchical_chunks(model, sequences, oracle_z):
    """
    Experiment 4: Hierarchical chunks with simple interpolation.

    Tests middle-ground: chunk-level variation + interpolation within chunks.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: HIERARCHICAL CHUNKS")
    print("="*80)
    print("\nQuestion: Can chunked approach match oracle roughness?")

    chunk_configs = [
        {'num_chunks': 3, 'chunk_size': 10},  # 3×10
        {'num_chunks': 5, 'chunk_size': 6},   # 5×6
        {'num_chunks': 6, 'chunk_size': 5},   # 6×5
    ]

    results = []

    for config in chunk_configs:
        num_chunks = config['num_chunks']
        chunk_size = config['chunk_size']

        print(f"\n  Testing {num_chunks} chunks × {chunk_size} days")

        # Generate chunk endpoints (iid)
        chunk_z = torch.zeros((len(sequences), num_chunks, model.config['latent_dim']))
        for i in range(len(sequences)):
            chunk_z[i] = torch.randn(num_chunks, model.config['latent_dim'])

        # Interpolate within chunks
        chunked_z = torch.zeros_like(oracle_z)
        for i in range(len(sequences)):
            for chunk_idx in range(num_chunks):
                start_t = chunk_idx * chunk_size
                end_t = min((chunk_idx + 1) * chunk_size, HORIZON)

                if chunk_idx < num_chunks - 1:
                    # Linear interpolation to next chunk
                    z_start = chunk_z[i, chunk_idx]
                    z_end = chunk_z[i, chunk_idx + 1]

                    for t in range(start_t, end_t):
                        alpha = (t - start_t) / chunk_size
                        chunked_z[i, t] = (1 - alpha) * z_start + alpha * z_end
                else:
                    # Last chunk: constant
                    chunked_z[i, start_t:end_t] = chunk_z[i, chunk_idx]

        _, roughness = generate_with_z(model, sequences, chunked_z, f"{num_chunks}×{chunk_size} chunks")
        autocorr = measure_autocorr(chunked_z, lag=1)

        results.append({
            'config': f'{num_chunks}×{chunk_size}',
            'num_chunks': num_chunks,
            'autocorr': autocorr,
            'roughness': roughness
        })

    # Oracle for comparison
    _, roughness_oracle = generate_with_z(model, sequences, oracle_z, "Oracle Z")

    print(f"\n  Results:")
    print(f"  {'Config':<12} {'Autocorr':<12} {'Roughness':<15} {'vs Oracle':<15}")
    print(f"  {'-'*60}")
    for r in results:
        ratio = r['roughness'] / roughness_oracle * 100
        print(f"  {r['config']:<12} {r['autocorr']:<12.4f} {r['roughness']:<15.6f} {ratio:<15.1f}%")
    print(f"  {'oracle':<12} {measure_autocorr(oracle_z):<12.4f} {roughness_oracle:<15.6f} {'100.0%':<15}")

    # Find best chunking
    best = max(results, key=lambda x: x['roughness'])
    best_ratio = best['roughness'] / roughness_oracle

    if best_ratio > 0.9:
        conclusion = f"✅ CHUNKS WORK! ({best['config']} achieves {best_ratio*100:.1f}%)"
    elif best_ratio > 0.7:
        conclusion = f"⚠️ CHUNKS HELP ({best['config']} achieves {best_ratio*100:.1f}%)"
    else:
        conclusion = f"❌ CHUNKS INSUFFICIENT (best {best_ratio*100:.1f}%)"

    print(f"\n  → {conclusion}")

    return {'results': results, 'oracle': roughness_oracle, 'conclusion': conclusion}


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

def main():
    print("="*80)
    print("CONFIRMING TEMPORAL STRUCTURE NECESSITY")
    print("="*80)

    # Load model and data
    model, surfaces, ex_feats = load_model_and_data()

    # Get test sequences
    print(f"\nPreparing test sequences...")
    sequences = get_test_sequences(surfaces, ex_feats, NUM_SEQUENCES)
    print(f"✓ {len(sequences)} sequences prepared")

    # Get oracle z values
    print(f"\nGetting oracle z values (H={HORIZON})...")
    oracle_z_list = []

    for seq in sequences:
        full_surf = torch.cat([seq['context_surf'], seq['target_surf']], dim=0).unsqueeze(0).to(DEVICE)
        full_ex = torch.cat([seq['context_ex'], seq['target_ex']], dim=0).unsqueeze(0).to(DEVICE)

        full_input = {'surface': full_surf, 'ex_feats': full_ex}

        with torch.no_grad():
            z_mean, _, _ = model.encoder(full_input)
            z_future = z_mean[0, CONTEXT_LEN:, :].cpu()
            oracle_z_list.append(z_future)

    oracle_z = torch.stack(oracle_z_list)  # (num_sequences, horizon, latent_dim)
    print(f"✓ Oracle z shape: {oracle_z.shape}")
    print(f"✓ Oracle autocorr (lag-1): {measure_autocorr(oracle_z, lag=1):.4f}")

    # Run experiments
    exp1_results = experiment_1_shuffled_oracle(model, sequences, oracle_z)
    exp2_results = experiment_2_synthetic_autocorr(model, sequences, oracle_z)
    exp3_results = experiment_3_smoothed_iid(model, sequences, oracle_z)
    exp4_results = experiment_4_hierarchical_chunks(model, sequences, oracle_z)

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    with open(OUTPUT_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONFIRMING TEMPORAL STRUCTURE NECESSITY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Horizon: {HORIZON}\n")
        f.write(f"Sequences: {NUM_SEQUENCES}\n")
        f.write(f"Samples per sequence: {NUM_SAMPLES}\n\n")

        f.write("="*80 + "\n")
        f.write("EXPERIMENT 1: SHUFFLED ORACLE Z\n")
        f.write("="*80 + "\n")
        f.write(f"Original roughness: {exp1_results['roughness_original']:.6f}\n")
        f.write(f"Shuffled roughness: {exp1_results['roughness_shuffled']:.6f}\n")
        f.write(f"Drop: {exp1_results['drop_percent']:.1f}%\n")
        f.write(f"Conclusion: {exp1_results['conclusion']}\n\n")

        f.write("="*80 + "\n")
        f.write("EXPERIMENT 2: SYNTHETIC AUTOCORRELATION\n")
        f.write("="*80 + "\n")
        for r in exp2_results['results']:
            f.write(f"Autocorr {r['target_autocorr']:.2f}: roughness {r['roughness']:.6f}\n")
        f.write(f"Conclusion: {exp2_results['conclusion']}\n\n")

        f.write("="*80 + "\n")
        f.write("EXPERIMENT 3: POST-HOC SMOOTHING\n")
        f.write("="*80 + "\n")
        for r in exp3_results['results']:
            ratio = r['roughness'] / exp3_results['oracle'] * 100
            f.write(f"{r['method']}: roughness {r['roughness']:.6f} ({ratio:.1f}% of oracle)\n")
        f.write(f"Conclusion: {exp3_results['conclusion']}\n\n")

        f.write("="*80 + "\n")
        f.write("EXPERIMENT 4: HIERARCHICAL CHUNKS\n")
        f.write("="*80 + "\n")
        for r in exp4_results['results']:
            ratio = r['roughness'] / exp4_results['oracle'] * 100
            f.write(f"{r['config']}: roughness {r['roughness']:.6f} ({ratio:.1f}% of oracle)\n")
        f.write(f"Conclusion: {exp4_results['conclusion']}\n\n")

        f.write("="*80 + "\n")
        f.write("FINAL VERDICT\n")
        f.write("="*80 + "\n\n")

        # Determine if autoregressive is necessary
        shuffled_drop = exp1_results['drop_percent']
        smoothing_works = any(r['roughness'] / exp3_results['oracle'] > 0.9 for r in exp3_results['results'])
        chunks_work = any(r['roughness'] / exp4_results['oracle'] > 0.9 for r in exp4_results['results'])

        if shuffled_drop > 30 and not smoothing_works and not chunks_work:
            verdict = "⚠️ AUTOREGRESSIVE LIKELY NECESSARY"
            f.write("Evidence:\n")
            f.write(f"- Shuffling drops roughness by {shuffled_drop:.1f}% (temporal structure essential)\n")
            f.write("- Post-hoc smoothing cannot recover oracle performance\n")
            f.write("- Hierarchical chunks insufficient\n")
            f.write("\nRecommendation: Proceed with P1 (Autoregressive Prior)\n")
        elif smoothing_works:
            verdict = "✅ POST-HOC SMOOTHING SUFFICIENT"
            f.write("Evidence:\n")
            f.write("- Post-hoc smoothing of iid z can match oracle roughness\n")
            f.write("\nRecommendation: Use P2 (Position-Encoded) + smoothing filter\n")
        elif chunks_work:
            verdict = "✅ HIERARCHICAL CHUNKS SUFFICIENT"
            f.write("Evidence:\n")
            f.write("- Chunked approach with interpolation matches oracle\n")
            f.write("\nRecommendation: Use P5 (Hierarchical Chunks)\n")
        else:
            verdict = "⚠️ UNCERTAIN - FURTHER INVESTIGATION NEEDED"

        f.write(f"\nFINAL VERDICT: {verdict}\n")

    print(f"✓ Results saved to {OUTPUT_FILE}")
    print("\nDone!")


if __name__ == "__main__":
    main()
