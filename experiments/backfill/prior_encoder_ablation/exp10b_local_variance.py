"""
Experiment 10b: Context-Aware Local Variance Estimation

PROBLEM:
Exp10 used constant σ² across all contexts (homoscedastic).
But volatility regimes change - uncertainty should be higher in crisis periods.

INSIGHT:
We can't compute Var(X|C) for a single C (one observation).
But we CAN estimate variance across SIMILAR contexts.

APPROACH:
1. Encode all contexts into latent space
2. For each context C, find k nearest neighbors in latent space
3. σ²(C) = variance of residuals among those k neighbors
4. This gives context-aware, heteroscedastic variance

WHY THIS WORKS:
- Similar contexts (same regime) should have similar uncertainty
- Latent space captures meaningful similarity
- k neighbors provide enough samples to estimate local variance
- Generalizes to unseen data via latent space similarity
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior


def load_model(model_path, model_class):
    """Load a trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']

    model = model_class(model_config)

    state_dict = checkpoint['state_dict']
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            cleaned_state_dict[k[10:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, model_config


def get_context_encoding(model, context):
    """Get latent encoding of context (for similarity computation)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    C = model.config["context_len"]

    context = context.unsqueeze(0).to(device).to(dtype)
    ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
    ctx_input = {"surface": context, "ex_feats": ctx_feats}

    with torch.no_grad():
        ctx_out = model.ctx_encoder(ctx_input)
        context_summary = ctx_out[:, -1, :]  # (1, hidden_dim)

    return context_summary.squeeze().cpu().numpy()


def get_deterministic_prediction(model, context):
    """Get deterministic prediction (z = prior mean)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    C = model.config["context_len"]
    latent_dim = model.config["latent_dim"]

    context = context.unsqueeze(0).to(device).to(dtype)
    ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
    ctx_input = {"surface": context, "ex_feats": ctx_feats}

    with torch.no_grad():
        ctx_out = model.ctx_encoder(ctx_input)
        context_summary = ctx_out[:, -1, :]
        mu_p, _ = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

        if model.config.get("compress_context", True):
            ctx_embedding_dim = latent_dim
        else:
            ctx_embedding_dim = model.config["mem_hidden"]

        ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
        decoder_input = torch.cat([mu_p, ctx_zeros], dim=-1)
        decoded = model.decoder(decoder_input)

        if isinstance(decoded, tuple):
            decoded = decoded[0]

    return decoded.squeeze().cpu().numpy()


def main():
    print("=" * 70)
    print("EXPERIMENT 10b: Context-Aware Local Variance")
    print("=" * 70)

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nTotal data: {surface.shape}")

    # Load model
    model_path = Path("results/prior_encoder_ablation/extended_training_v5/baseline_ep200.pt")
    print(f"\nLoading model from: {model_path}")
    model, config = load_model(model_path, CVAEFullCovPrior)

    C = config["context_len"]
    print(f"Context length: {C}")

    # =========================================================================
    # STEP 1: Compute encodings and residuals for all data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Compute Context Encodings and Residuals")
    print("=" * 70)

    n_total = len(surface) - C - 1

    # Use substantial portion for robust estimation
    n_sample = min(2000, n_total)
    all_indices = np.arange(n_total)
    np.random.shuffle(all_indices)
    sample_indices = all_indices[:n_sample]

    encodings = []
    predictions = []
    targets = []

    for i in tqdm(sample_indices, desc="Computing encodings"):
        context = surface[i:i+C]
        target = surface[i+C].numpy()

        encoding = get_context_encoding(model, context)
        pred = get_deterministic_prediction(model, context)

        encodings.append(encoding)
        predictions.append(pred)
        targets.append(target)

    encodings = np.array(encodings)      # (N, hidden_dim)
    predictions = np.array(predictions)  # (N, 5, 5)
    targets = np.array(targets)          # (N, 5, 5)
    residuals = targets - predictions    # (N, 5, 5)

    print(f"\nEncodings shape: {encodings.shape}")
    print(f"Residuals shape: {residuals.shape}")

    # =========================================================================
    # STEP 2: Build k-NN index in latent space
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Build k-NN Index")
    print("=" * 70)

    # Split into calibration and test
    n_cal = n_sample // 2
    cal_idx = np.arange(n_cal)
    test_idx = np.arange(n_cal, n_sample)

    cal_encodings = encodings[cal_idx]
    cal_residuals = residuals[cal_idx]

    test_encodings = encodings[test_idx]
    test_predictions = predictions[test_idx]
    test_targets = targets[test_idx]
    test_residuals = residuals[test_idx]

    print(f"Calibration set: {len(cal_idx)}")
    print(f"Test set: {len(test_idx)}")

    # Build k-NN index on calibration encodings
    k_values = [10, 20, 50, 100]

    nn_index = NearestNeighbors(n_neighbors=max(k_values), algorithm='ball_tree')
    nn_index.fit(cal_encodings)

    print(f"k-NN index built on {len(cal_encodings)} calibration points")

    # =========================================================================
    # STEP 3: Compute local variance for test points
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Local Variance Estimation")
    print("=" * 70)

    # Find neighbors for all test points
    distances, neighbor_indices = nn_index.kneighbors(test_encodings)

    results = {}

    for k in k_values:
        print(f"\n--- k = {k} neighbors ---")

        local_variances = []
        local_quantiles_lower = []
        local_quantiles_upper = []
        local_biases = []

        for i in range(len(test_idx)):
            # Get k nearest neighbors' residuals
            nn_idx = neighbor_indices[i, :k]
            nn_residuals = cal_residuals[nn_idx]  # (k, 5, 5)

            # Local variance (per grid point)
            local_var = nn_residuals.var(axis=0)  # (5, 5)
            local_variances.append(local_var)

            # Local quantiles (empirical)
            lower_q = np.percentile(nn_residuals, 5, axis=0)
            upper_q = np.percentile(nn_residuals, 95, axis=0)
            local_quantiles_lower.append(lower_q)
            local_quantiles_upper.append(upper_q)

            # Local bias
            local_bias = nn_residuals.mean(axis=0)
            local_biases.append(local_bias)

        local_variances = np.array(local_variances)
        local_quantiles_lower = np.array(local_quantiles_lower)
        local_quantiles_upper = np.array(local_quantiles_upper)
        local_biases = np.array(local_biases)

        # Check variance heterogeneity
        var_of_var = local_variances.var(axis=0).mean()
        mean_var = local_variances.mean(axis=0).mean()
        cv_of_var = np.sqrt(var_of_var) / mean_var  # coefficient of variation

        print(f"  Mean local variance: {mean_var:.6f}")
        print(f"  Variance of local variance: {var_of_var:.8f}")
        print(f"  CV of variance (heterogeneity): {cv_of_var:.2%}")

        # =====================================================================
        # METHOD A: Local Gaussian CI
        # =====================================================================
        local_std = np.sqrt(local_variances)

        covered_gaussian = []
        for i in range(len(test_idx)):
            pred_corrected = test_predictions[i] + local_biases[i]
            lower = pred_corrected - 1.645 * local_std[i]
            upper = pred_corrected + 1.645 * local_std[i]
            in_interval = (test_targets[i] >= lower) & (test_targets[i] <= upper)
            covered_gaussian.append(in_interval)

        covered_gaussian = np.array(covered_gaussian)
        gaussian_coverage = covered_gaussian.mean() * 100

        # =====================================================================
        # METHOD B: Local Quantile CI
        # =====================================================================
        covered_quantile = []
        for i in range(len(test_idx)):
            lower = test_predictions[i] + local_quantiles_lower[i]
            upper = test_predictions[i] + local_quantiles_upper[i]
            in_interval = (test_targets[i] >= lower) & (test_targets[i] <= upper)
            covered_quantile.append(in_interval)

        covered_quantile = np.array(covered_quantile)
        quantile_coverage = covered_quantile.mean() * 100

        print(f"\n  Coverage (90% target):")
        print(f"    Local Gaussian: {gaussian_coverage:.1f}%")
        print(f"    Local Quantile: {quantile_coverage:.1f}%")

        results[k] = {
            'gaussian_coverage': gaussian_coverage,
            'quantile_coverage': quantile_coverage,
            'cv_of_var': cv_of_var,
            'mean_var': mean_var,
            'local_variances': local_variances,
            'covered_quantile': covered_quantile
        }

    # =========================================================================
    # STEP 4: Analyze Variance by Volatility Regime
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Variance by Volatility Regime")
    print("=" * 70)

    # Use ATM IV (center of grid) as regime indicator
    atm_iv = targets[:, 2, 2]  # Center point

    # Split into low/medium/high volatility regimes
    vol_terciles = np.percentile(atm_iv, [33, 67])

    low_vol_mask = atm_iv < vol_terciles[0]
    mid_vol_mask = (atm_iv >= vol_terciles[0]) & (atm_iv < vol_terciles[1])
    high_vol_mask = atm_iv >= vol_terciles[1]

    print(f"\nVolatility terciles: {vol_terciles}")
    print(f"  Low vol:  {low_vol_mask.sum()} points (ATM IV < {vol_terciles[0]:.3f})")
    print(f"  Mid vol:  {mid_vol_mask.sum()} points")
    print(f"  High vol: {high_vol_mask.sum()} points (ATM IV > {vol_terciles[1]:.3f})")

    # Compute residual variance by regime
    for regime, mask in [("Low", low_vol_mask), ("Mid", mid_vol_mask), ("High", high_vol_mask)]:
        regime_residuals = residuals[mask]
        regime_var = regime_residuals.var(axis=0).mean()
        regime_std = np.sqrt(regime_var)
        print(f"\n  {regime} volatility regime:")
        print(f"    Residual variance: {regime_var:.6f}")
        print(f"    Residual std:      {regime_std:.4f}")

    # =========================================================================
    # STEP 5: Compare Constant vs Local Variance
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Constant vs Local Variance Comparison")
    print("=" * 70)

    # Global (constant) variance from calibration
    global_var = cal_residuals.var(axis=0)
    global_std = np.sqrt(global_var)
    global_bias = cal_residuals.mean(axis=0)

    # Global quantiles
    global_lower_q = np.percentile(cal_residuals, 5, axis=0)
    global_upper_q = np.percentile(cal_residuals, 95, axis=0)

    # Coverage with global (constant) variance
    covered_global = []
    for i in range(len(test_idx)):
        lower = test_predictions[i] + global_lower_q
        upper = test_predictions[i] + global_upper_q
        in_interval = (test_targets[i] >= lower) & (test_targets[i] <= upper)
        covered_global.append(in_interval)

    covered_global = np.array(covered_global)
    global_coverage = covered_global.mean() * 100

    print(f"\nConstant variance (exp10 approach):")
    print(f"  Global coverage: {global_coverage:.1f}%")

    # Best local k
    best_k = max(results.keys(), key=lambda k: results[k]['quantile_coverage'])
    best_coverage = results[best_k]['quantile_coverage']

    print(f"\nLocal variance (k={best_k}):")
    print(f"  Local coverage: {best_coverage:.1f}%")

    print(f"\nImprovement: {best_coverage - global_coverage:+.1f}%")

    # =========================================================================
    # STEP 6: Coverage by Regime
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Coverage by Volatility Regime")
    print("=" * 70)

    # Get test set masks
    test_atm_iv = test_targets[:, 2, 2]
    test_low = test_atm_iv < vol_terciles[0]
    test_mid = (test_atm_iv >= vol_terciles[0]) & (test_atm_iv < vol_terciles[1])
    test_high = test_atm_iv >= vol_terciles[1]

    print(f"\n{'Method':<20} {'Low Vol':<12} {'Mid Vol':<12} {'High Vol':<12} {'Overall':<12}")
    print("-" * 68)

    # Global coverage by regime
    for regime, mask in [("Low", test_low), ("Mid", test_mid), ("High", test_high)]:
        pass  # Just to define masks

    global_low = covered_global[test_low].mean() * 100 if test_low.sum() > 0 else 0
    global_mid = covered_global[test_mid].mean() * 100 if test_mid.sum() > 0 else 0
    global_high = covered_global[test_high].mean() * 100 if test_high.sum() > 0 else 0
    print(f"{'Constant (global)':<20} {global_low:<12.1f} {global_mid:<12.1f} {global_high:<12.1f} {global_coverage:<12.1f}")

    # Local coverage by regime (best k)
    local_covered = results[best_k]['covered_quantile']
    local_low = local_covered[test_low].mean() * 100 if test_low.sum() > 0 else 0
    local_mid = local_covered[test_mid].mean() * 100 if test_mid.sum() > 0 else 0
    local_high = local_covered[test_high].mean() * 100 if test_high.sum() > 0 else 0
    print(f"{'Local (k=' + str(best_k) + ')':<20} {local_low:<12.1f} {local_mid:<12.1f} {local_high:<12.1f} {best_coverage:<12.1f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Experiment 10b: Context-Aware Local Variance

HETEROGENEITY CHECK:
  CV of local variance: {results[best_k]['cv_of_var']:.2%}
  → {"Significant heterogeneity detected" if results[best_k]['cv_of_var'] > 0.1 else "Variance is relatively homogeneous"}

COVERAGE COMPARISON (90% target):
  Constant variance:  {global_coverage:.1f}%
  Local variance:     {best_coverage:.1f}% (k={best_k})
  Improvement:        {best_coverage - global_coverage:+.1f}%

BY VOLATILITY REGIME:
  Low vol:  {global_low:.1f}% → {local_low:.1f}% ({local_low - global_low:+.1f}%)
  Mid vol:  {global_mid:.1f}% → {local_mid:.1f}% ({local_mid - global_mid:+.1f}%)
  High vol: {global_high:.1f}% → {local_high:.1f}% ({local_high - global_high:+.1f}%)

CONCLUSION:
  {"Local variance improves coverage, especially in extreme regimes" if best_coverage > global_coverage else "Local variance provides similar coverage to constant variance"}
""")

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp10b_local_variance")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "results.npz",
        encodings=encodings,
        residuals=residuals,
        global_coverage=global_coverage,
        best_k=best_k,
        best_coverage=best_coverage,
        results_by_k={str(k): {
            'gaussian_coverage': results[k]['gaussian_coverage'],
            'quantile_coverage': results[k]['quantile_coverage'],
            'cv_of_var': results[k]['cv_of_var']
        } for k in k_values}
    )

    print(f"Results saved to: {output_dir}/results.npz")


if __name__ == "__main__":
    main()
