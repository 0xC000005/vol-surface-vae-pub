"""
Verification Experiments: Prior Network Parameter Reuse Problem

This script tests whether reusing a single (μ, σ) for all H future timesteps
causes unrealistically smooth generated paths (19% of GT roughness).

Three experiments:
A. Path roughness under different prior schemes
B. Oracle vs constant prior comparison
C. Variance scaling impact

Model: backfill_context60_latent12_v3_conditional_prior_phase2_ep599.pt
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_phase2_ep599.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_PATH = "/tmp/prior_reuse_verification_results.txt"

CONTEXT_LEN = 60
HORIZON = 90
NUM_CONTEXTS = 50
NUM_SAMPLES_PER_CONTEXT = 100
ATM_6M = (2, 2)  # Grid indices for ATM 6M point


def measure_roughness(paths):
    """
    Roughness = std of day-to-day changes across all samples

    Args:
        paths: (num_samples, horizon, 5, 5)
    Returns:
        float: roughness metric
    """
    daily_changes = paths[:, 1:, :, :] - paths[:, :-1, :, :]
    return daily_changes.std().item()


def measure_autocorr(paths, lag=1):
    """
    Autocorrelation of daily changes

    Args:
        paths: (num_samples, horizon, 5, 5)
        lag: int, time lag for autocorrelation
    Returns:
        float: autocorrelation coefficient
    """
    changes = paths[:, 1:, :, :] - paths[:, :-1, :, :]
    changes_flat = changes.reshape(changes.shape[0], -1)

    if changes_flat.shape[1] <= lag:
        return 0.0

    try:
        corr = np.corrcoef(changes_flat[:, :-lag].flatten(),
                          changes_flat[:, lag:].flatten())[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def load_model_and_data():
    """Load model and data"""
    print("Loading model...")
    model_data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    model = CVAEMemRandConditionalPrior(model_data["model_config"])

    # Remove "_orig_mod." prefix from state_dict keys (from torch.compile)
    state_dict = model_data["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"✓ Model loaded from {MODEL_PATH}")

    print("\nLoading data...")
    data = np.load(DATA_PATH)
    surfaces = torch.tensor(data['surface'], dtype=torch.float32)

    # Construct ex_data from individual components [ret, skews, slopes]
    ret = data['ret']
    skews = data['skews']
    slopes = data['slopes']
    ex_data = np.stack([ret, skews, slopes], axis=1)  # (N, 3)
    ex_feats = torch.tensor(ex_data, dtype=torch.float32)

    print(f"✓ Loaded {len(surfaces)} days of data")
    print(f"✓ Extra features shape: {ex_feats.shape}")

    return model, surfaces, ex_feats


def get_test_sequences(surfaces, ex_feats, num_contexts):
    """Get test sequences with context and target"""
    sequences = []
    T = CONTEXT_LEN + HORIZON

    # Use middle portion of data for test
    start_idx = len(surfaces) // 3
    end_idx = len(surfaces) - T - 10

    indices = np.random.choice(range(start_idx, end_idx), size=num_contexts, replace=False)

    for idx in indices:
        context_surf = surfaces[idx:idx+CONTEXT_LEN]
        target_surf = surfaces[idx+CONTEXT_LEN:idx+T]
        context_ex = ex_feats[idx:idx+CONTEXT_LEN]
        target_ex = ex_feats[idx+CONTEXT_LEN:idx+T]

        sequences.append({
            'context_surf': context_surf,
            'target_surf': target_surf,
            'context_ex': context_ex,
            'target_ex': target_ex,
        })

    return sequences


#------------------------------------------------------------------------------
# Experiment A: Path Roughness Under Different Prior Schemes
#------------------------------------------------------------------------------

def experiment_a(model, test_sequences):
    """
    Test whether varying prior parameters per timestep increases roughness

    Conditions:
    1. baseline: Same (μ, σ) for all H days (current)
    2. random_walk_mu: μ_t = μ_base + cumsum(small_noise)
    3. scaled_sigma: σ_t = σ_base * sqrt(1 + t/H)
    4. both: random walk μ + scaled σ
    """
    print("\n" + "="*80)
    print("EXPERIMENT A: PATH ROUGHNESS UNDER DIFFERENT PRIOR SCHEMES")
    print("="*80)

    results = {}

    # Get ground truth roughness
    gt_paths = []
    for seq in test_sequences:
        gt_paths.append(seq['target_surf'][: HORIZON, :, :].numpy())
    gt_paths = np.stack(gt_paths)
    gt_roughness = measure_roughness(torch.tensor(gt_paths))

    print(f"\nGround Truth roughness: {gt_roughness:.6f}")

    # Test each condition
    for condition_name in ['baseline', 'random_walk_mu', 'scaled_sigma', 'both']:
        print(f"\nTesting condition: {condition_name}")

        all_paths = []

        for seq in test_sequences:
            context_input = {
                'surface': seq['context_surf'].unsqueeze(0).to(DEVICE),
                'ex_feats': seq['context_ex'].unsqueeze(0).to(DEVICE)
            }

            # Get prior parameters
            with torch.no_grad():
                prior_mean, prior_logvar = model.prior_network(context_input)

            B, C, latent_dim = prior_mean.shape

            # Apply condition-specific modifications
            if condition_name == 'baseline':
                mu = prior_mean[:, -1:, :].expand(1, HORIZON, -1)
                logvar = prior_logvar[:, -1:, :].expand(1, HORIZON, -1)

            elif condition_name == 'random_walk_mu':
                mu_base = prior_mean[:, -1:, :]
                noise_steps = torch.cumsum(0.01 * torch.randn(1, HORIZON, latent_dim, device=DEVICE), dim=1)
                mu = mu_base + noise_steps
                logvar = prior_logvar[:, -1:, :].expand(1, HORIZON, -1)

            elif condition_name == 'scaled_sigma':
                t = torch.arange(HORIZON, device=DEVICE).float()
                scale = 0.5 * torch.log(1 + t / HORIZON)  # sqrt scaling
                mu = prior_mean[:, -1:, :].expand(1, HORIZON, -1)
                logvar = prior_logvar[:, -1:, :].expand(1, HORIZON, -1) + scale.view(1, -1, 1)

            elif condition_name == 'both':
                mu_base = prior_mean[:, -1:, :]
                noise_steps = torch.cumsum(0.01 * torch.randn(1, HORIZON, latent_dim, device=DEVICE), dim=1)
                mu = mu_base + noise_steps

                t = torch.arange(HORIZON, device=DEVICE).float()
                scale = 0.5 * torch.log(1 + t / HORIZON)
                logvar = prior_logvar[:, -1:, :].expand(1, HORIZON, -1) + scale.view(1, -1, 1)

            # Generate samples
            for _ in range(NUM_SAMPLES_PER_CONTEXT):
                eps = torch.randn_like(mu)
                z_future = mu + torch.exp(0.5 * logvar) * eps

                # Build full z (context + future)
                T = C + HORIZON
                z_full = torch.zeros((1, T, latent_dim), device=DEVICE)

                # Context part: use encoder posterior mean (deterministic)
                with torch.no_grad():
                    ctx_latent_mean, _, _ = model.encoder(context_input)
                    z_full[:, :C, :] = ctx_latent_mean

                # Future part: use modified prior samples
                z_full[:, C:, :] = z_future

                # Generate using get_surface_given_conditions with custom z
                with torch.no_grad():
                    if model.config.get("ex_feats_dim", 0) > 0:
                        surface, _ = model.get_surface_given_conditions(context_input, z=z_full, horizon=HORIZON)
                    else:
                        surface = model.get_surface_given_conditions(context_input, z=z_full, horizon=HORIZON)

                all_paths.append(surface[0, :, :, :].cpu())

        # Measure metrics
        all_paths = torch.stack(all_paths)
        roughness = measure_roughness(all_paths)
        roughness_ratio = roughness / gt_roughness
        autocorr = measure_autocorr(all_paths)
        marginal_std = all_paths[:, -1, ATM_6M[0], ATM_6M[1]].std().item()

        results[condition_name] = {
            'roughness': roughness,
            'roughness_ratio': roughness_ratio,
            'autocorr': autocorr,
            'marginal_std': marginal_std
        }

        print(f"  Roughness: {roughness:.6f} ({roughness_ratio:.1%} of GT)")
        print(f"  Autocorr:  {autocorr:.4f}")
        print(f"  Marginal std: {marginal_std:.6f}")

    return results, gt_roughness


#------------------------------------------------------------------------------
# Experiment B: Oracle vs Constant Prior Comparison
#------------------------------------------------------------------------------

def experiment_b(model, test_sequences):
    """
    Compare roughness: posterior z (oracle) vs constant prior z
    """
    print("\n" + "="*80)
    print("EXPERIMENT B: ORACLE VS CONSTANT PRIOR COMPARISON")
    print("="*80)

    # Generate oracle paths (using posterior z - sees target)
    print("\nGenerating oracle paths (posterior z)...")
    oracle_paths = []

    for seq in test_sequences:
        full_seq_surf = torch.cat([seq['context_surf'], seq['target_surf']], dim=0).unsqueeze(0).to(DEVICE)
        full_seq_ex = torch.cat([seq['context_ex'], seq['target_ex']], dim=0).unsqueeze(0).to(DEVICE)

        full_input = {
            'surface': full_seq_surf,
            'ex_feats': full_seq_ex
        }

        # Get posterior z (different z_t for each timestep)
        with torch.no_grad():
            z_mean, z_logvar, _ = model.encoder(full_input)

        # Generate samples from posterior
        for _ in range(NUM_SAMPLES_PER_CONTEXT):
            eps = torch.randn_like(z_mean)
            z_sample = z_mean + torch.exp(0.5 * z_logvar) * eps

            # Use get_surface_given_conditions with full posterior z
            context_only_input = {
                'surface': seq['context_surf'].unsqueeze(0).to(DEVICE),
                'ex_feats': seq['context_ex'].unsqueeze(0).to(DEVICE)
            }

            with torch.no_grad():
                if model.config.get("ex_feats_dim", 0) > 0:
                    surface, _ = model.get_surface_given_conditions(context_only_input, z=z_sample, horizon=HORIZON)
                else:
                    surface = model.get_surface_given_conditions(context_only_input, z=z_sample, horizon=HORIZON)

            oracle_paths.append(surface[0, :, :, :].cpu())

    oracle_paths = torch.stack(oracle_paths)

    # Generate constant prior paths (current approach)
    print("Generating constant prior paths...")
    constant_paths = []

    for seq in test_sequences:
        context_input = {
            'surface': seq['context_surf'].unsqueeze(0).to(DEVICE),
            'ex_feats': seq['context_ex'].unsqueeze(0).to(DEVICE)
        }

        with torch.no_grad():
            prior_mu, prior_logvar = model.prior_network(context_input)

        # Expand last timestep to all H future positions (CURRENT APPROACH)
        B, C, latent_dim = prior_mu.shape
        mu = prior_mu[:, -1:, :].expand(1, HORIZON, -1)
        logvar = prior_logvar[:, -1:, :].expand(1, HORIZON, -1)

        for _ in range(NUM_SAMPLES_PER_CONTEXT):
            eps = torch.randn_like(mu)
            z_future = mu + torch.exp(0.5 * logvar) * eps

            # Build full z (context + future)
            T = C + HORIZON
            z_full = torch.zeros((1, T, latent_dim), device=DEVICE)

            # Context part: use encoder posterior mean
            with torch.no_grad():
                ctx_latent_mean, _, _ = model.encoder(context_input)
                z_full[:, :C, :] = ctx_latent_mean

            # Future part: constant prior
            z_full[:, C:, :] = z_future

            with torch.no_grad():
                if model.config.get("ex_feats_dim", 0) > 0:
                    surface, _ = model.get_surface_given_conditions(context_input, z=z_full, horizon=HORIZON)
                else:
                    surface = model.get_surface_given_conditions(context_input, z=z_full, horizon=HORIZON)

            constant_paths.append(surface[0, :, :, :].cpu())

    constant_paths = torch.stack(constant_paths)

    # Get ground truth
    gt_paths = []
    for seq in test_sequences:
        gt_paths.append(seq['target_surf'][:HORIZON, :, :].numpy())
    gt_paths = np.stack(gt_paths)
    gt_roughness = measure_roughness(torch.tensor(gt_paths))

    # Compare
    oracle_roughness = measure_roughness(oracle_paths)
    constant_roughness = measure_roughness(constant_paths)

    oracle_ratio = oracle_roughness / gt_roughness
    constant_ratio = constant_roughness / gt_roughness

    print(f"\nResults:")
    print(f"  Oracle roughness:   {oracle_roughness:.6f} ({oracle_ratio:.1%} of GT)")
    print(f"  Constant roughness: {constant_roughness:.6f} ({constant_ratio:.1%} of GT)")
    print(f"  GT roughness:       {gt_roughness:.6f}")
    print(f"\nGap: {oracle_roughness - constant_roughness:.6f} ({(oracle_ratio - constant_ratio)*100:.1f}%)")

    return {
        'oracle_roughness': oracle_roughness,
        'constant_roughness': constant_roughness,
        'gt_roughness': gt_roughness,
        'oracle_ratio': oracle_ratio,
        'constant_ratio': constant_ratio
    }


#------------------------------------------------------------------------------
# Experiment C: Variance Scaling Impact
#------------------------------------------------------------------------------

def experiment_c(model, test_sequences):
    """
    Test variance scaling functions
    """
    print("\n" + "="*80)
    print("EXPERIMENT C: VARIANCE SCALING IMPACT")
    print("="*80)

    # Get ground truth
    gt_paths = []
    for seq in test_sequences:
        gt_paths.append(seq['target_surf'][:HORIZON, :, :].numpy())
    gt_paths = np.stack(gt_paths)
    gt_roughness = measure_roughness(torch.tensor(gt_paths))
    gt_marginal_std = gt_paths[:, -1, ATM_6M[0], ATM_6M[1]].std()

    print(f"\nGround Truth:")
    print(f"  Roughness: {gt_roughness:.6f}")
    print(f"  Marginal std: {gt_marginal_std:.6f}")

    # Define scaling functions
    scale_configs = {
        'none':   lambda t, H: 0,
        'sqrt':   lambda t, H: 0.5 * np.log(1 + t/H),
        'linear': lambda t, H: np.log(1 + 0.5 * t/H),
        'log':    lambda t, H: np.log(np.log(2 + t)) if t > 0 else 0,
        'strong': lambda t, H: np.log(1 + t/H),
    }

    results = {}

    for scale_name, scale_fn in scale_configs.items():
        print(f"\nTesting scale: {scale_name}")

        all_paths = []

        for seq in test_sequences:
            context_input = {
                'surface': seq['context_surf'].unsqueeze(0).to(DEVICE),
                'ex_feats': seq['context_ex'].unsqueeze(0).to(DEVICE)
            }

            with torch.no_grad():
                prior_mu, prior_logvar = model.prior_network(context_input)

            B, C, latent_dim = prior_mu.shape
            mu = prior_mu[:, -1:, :].expand(1, HORIZON, -1)

            # Apply horizon-dependent scaling to logvar
            t = torch.arange(HORIZON, device=DEVICE).float()
            scale = torch.tensor([scale_fn(ti.item(), HORIZON) for ti in t], device=DEVICE)
            logvar = prior_logvar[:, -1:, :].expand(1, HORIZON, -1) + scale.view(1, -1, 1)

            for _ in range(NUM_SAMPLES_PER_CONTEXT):
                eps = torch.randn_like(mu)
                z_future = mu + torch.exp(0.5 * logvar) * eps

                # Build full z (context + future)
                T = C + HORIZON
                z_full = torch.zeros((1, T, latent_dim), device=DEVICE)

                # Context part: use encoder posterior mean
                with torch.no_grad():
                    ctx_latent_mean, _, _ = model.encoder(context_input)
                    z_full[:, :C, :] = ctx_latent_mean

                # Future part: scaled variance prior
                z_full[:, C:, :] = z_future

                with torch.no_grad():
                    if model.config.get("ex_feats_dim", 0) > 0:
                        surface, _ = model.get_surface_given_conditions(context_input, z=z_full, horizon=HORIZON)
                    else:
                        surface = model.get_surface_given_conditions(context_input, z=z_full, horizon=HORIZON)

                all_paths.append(surface[0, :, :, :].cpu())

        all_paths = torch.stack(all_paths)
        roughness = measure_roughness(all_paths)
        roughness_ratio = roughness / gt_roughness
        autocorr = measure_autocorr(all_paths)
        marginal_std = all_paths[:, -1, ATM_6M[0], ATM_6M[1]].std().item()
        marginal_match = abs(marginal_std - gt_marginal_std) / gt_marginal_std

        results[scale_name] = {
            'roughness': roughness,
            'roughness_ratio': roughness_ratio,
            'autocorr': autocorr,
            'marginal_std': marginal_std,
            'marginal_match': marginal_match
        }

        print(f"  Roughness: {roughness:.6f} ({roughness_ratio:.1%} of GT)")
        print(f"  Autocorr:  {autocorr:.4f}")
        print(f"  Marginal std: {marginal_std:.6f} ({marginal_match:.1%} error)")

    return results, gt_roughness, gt_marginal_std


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

def main():
    print("="*80)
    print("PRIOR NETWORK PARAMETER REUSE VERIFICATION")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Horizon: {HORIZON}")
    print(f"Num contexts: {NUM_CONTEXTS}")
    print(f"Samples per context: {NUM_SAMPLES_PER_CONTEXT}")

    # Load model and data
    model, surfaces, ex_feats = load_model_and_data()

    # Get test sequences
    print(f"\nPreparing {NUM_CONTEXTS} test sequences...")
    test_sequences = get_test_sequences(surfaces, ex_feats, NUM_CONTEXTS)
    print(f"✓ Test sequences prepared")

    # Run experiments
    results_a, gt_roughness_a = experiment_a(model, test_sequences)
    results_b = experiment_b(model, test_sequences)
    results_c, gt_roughness_c, gt_marginal_std_c = experiment_c(model, test_sequences)

    # Save results
    print(f"\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    with open(OUTPUT_PATH, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PRIOR NETWORK PARAMETER REUSE VERIFICATION RESULTS\n")
        f.write("="*80 + "\n\n")

        # Experiment A
        f.write("EXPERIMENT A: PATH ROUGHNESS UNDER DIFFERENT PRIOR SCHEMES\n")
        f.write("-"*80 + "\n")
        f.write(f"Ground Truth roughness: {gt_roughness_a:.6f}\n\n")
        f.write(f"{'Condition':<20} {'Roughness':<12} {'Ratio':<12} {'Autocorr':<12} {'Marginal Std':<12}\n")
        f.write("-"*80 + "\n")
        for name, r in results_a.items():
            f.write(f"{name:<20} {r['roughness']:.6f}    {r['roughness_ratio']:.1%}       "
                   f"{r['autocorr']:.4f}       {r['marginal_std']:.6f}\n")

        # Experiment B
        f.write("\n\nEXPERIMENT B: ORACLE VS CONSTANT PRIOR COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"Oracle roughness:   {results_b['oracle_roughness']:.6f} ({results_b['oracle_ratio']:.1%} of GT)\n")
        f.write(f"Constant roughness: {results_b['constant_roughness']:.6f} ({results_b['constant_ratio']:.1%} of GT)\n")
        f.write(f"GT roughness:       {results_b['gt_roughness']:.6f}\n")
        f.write(f"Gap: {results_b['oracle_roughness'] - results_b['constant_roughness']:.6f} "
               f"({(results_b['oracle_ratio'] - results_b['constant_ratio'])*100:.1f}%)\n")

        # Experiment C
        f.write("\n\nEXPERIMENT C: VARIANCE SCALING IMPACT\n")
        f.write("-"*80 + "\n")
        f.write(f"Ground Truth roughness: {gt_roughness_c:.6f}\n")
        f.write(f"Ground Truth marginal std: {gt_marginal_std_c:.6f}\n\n")
        f.write(f"{'Scale Type':<12} {'Roughness':<12} {'Ratio':<12} {'Autocorr':<12} {'Marginal Std':<12} {'Error':<12}\n")
        f.write("-"*80 + "\n")
        for name, r in results_c.items():
            f.write(f"{name:<12} {r['roughness']:.6f}    {r['roughness_ratio']:.1%}       "
                   f"{r['autocorr']:.4f}       {r['marginal_std']:.6f}       {r['marginal_match']:.1%}\n")

        # Success criteria
        f.write("\n\nSUCCESS CRITERIA\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<30} {'Target':<15} {'Baseline':<15} {'Status':<10}\n")
        f.write("-"*80 + "\n")

        baseline_ratio = results_a['baseline']['roughness_ratio']
        baseline_autocorr = results_a['baseline']['autocorr']

        f.write(f"{'Roughness ratio':<30} {'>40%':<15} {f'{baseline_ratio:.1%}':<15} "
               f"{'✓ PASS' if baseline_ratio > 0.4 else '✗ FAIL'}\n")
        f.write(f"{'Autocorrelation':<30} {'>0.3':<15} {f'{baseline_autocorr:.3f}':<15} "
               f"{'✓ PASS' if baseline_autocorr > 0.3 else '✗ FAIL'}\n")

        # Check if any condition improves
        best_condition = max(results_a.items(), key=lambda x: x[1]['roughness_ratio'])
        f.write(f"\nBest condition: {best_condition[0]} with {best_condition[1]['roughness_ratio']:.1%} roughness ratio\n")

        if best_condition[1]['roughness_ratio'] > baseline_ratio:
            f.write(f"\n✅ CONCLUSION: Varying prior parameters DOES improve roughness\n")
            f.write(f"   Improvement: {(best_condition[1]['roughness_ratio'] - baseline_ratio)*100:.1f}%\n")
        else:
            f.write(f"\n❌ CONCLUSION: Varying prior parameters does NOT improve roughness\n")

    print(f"✓ Results saved to {OUTPUT_PATH}")
    print("\nDone!")


if __name__ == "__main__":
    main()
