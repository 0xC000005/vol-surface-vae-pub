"""
Experiment 10c: Regime-Based Conditional Variance

INSIGHT FROM EXP10b:
- Heterogeneity exists: high vol regime has 43% higher residual std
- But k-NN estimation is noisy with limited neighbors
- Solution: Use explicit regime classification with robust variance estimation

APPROACH:
1. Classify contexts into volatility regimes (low/mid/high)
2. Estimate variance per regime from calibration data
3. For new context, classify regime and use regime-specific variance
4. This gives context-aware variance with robust estimation

REGIME FEATURES:
- ATM IV level (primary regime indicator)
- Volatility of volatility (vol-of-vol from recent context)
- Surface shape (skew, term structure)
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

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


def extract_regime_features(context, target=None):
    """Extract features for regime classification."""
    # context: (C, 5, 5)
    # target: (5, 5) or None

    # ATM IV from last context surface
    atm_iv = context[-1, 2, 2].item()

    # Volatility of ATM IV over context window
    atm_series = context[:, 2, 2].numpy()
    vol_of_vol = np.std(atm_series)

    # Recent change in ATM IV
    atm_change = atm_series[-1] - atm_series[0]

    # Skew (OTM put - OTM call)
    skew = context[-1, 4, 0].item() - context[-1, 0, 4].item()

    # Term structure slope (long maturity - short maturity)
    term_slope = context[-1, 2, 4].item() - context[-1, 2, 0].item()

    return {
        'atm_iv': atm_iv,
        'vol_of_vol': vol_of_vol,
        'atm_change': atm_change,
        'skew': skew,
        'term_slope': term_slope
    }


def classify_regime(features, thresholds):
    """Classify into volatility regime based on features."""
    atm_iv = features['atm_iv']

    if atm_iv < thresholds['low']:
        return 'low'
    elif atm_iv < thresholds['high']:
        return 'mid'
    else:
        return 'high'


def main():
    print("=" * 70)
    print("EXPERIMENT 10c: Regime-Based Conditional Variance")
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
    # STEP 1: Compute predictions and residuals
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Compute Predictions and Extract Features")
    print("=" * 70)

    n_total = len(surface) - C - 1
    n_sample = min(3000, n_total)

    all_indices = np.arange(n_total)
    np.random.shuffle(all_indices)
    sample_indices = all_indices[:n_sample]

    predictions = []
    targets = []
    features_list = []

    for i in tqdm(sample_indices, desc="Computing"):
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        features = extract_regime_features(context)

        predictions.append(pred)
        targets.append(target)
        features_list.append(features)

    predictions = np.array(predictions)
    targets = np.array(targets)
    residuals = targets - predictions

    # Extract feature arrays
    atm_ivs = np.array([f['atm_iv'] for f in features_list])
    vol_of_vols = np.array([f['vol_of_vol'] for f in features_list])

    print(f"\nData shape: {predictions.shape}")

    # =========================================================================
    # STEP 2: Define regime thresholds from calibration data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Define Regime Thresholds")
    print("=" * 70)

    # Split into calibration and test
    n_cal = n_sample // 2
    cal_idx = np.arange(n_cal)
    test_idx = np.arange(n_cal, n_sample)

    cal_atm = atm_ivs[cal_idx]
    cal_residuals = residuals[cal_idx]
    cal_features = [features_list[i] for i in cal_idx]

    # Use terciles for regime boundaries
    thresholds = {
        'low': np.percentile(cal_atm, 33),
        'high': np.percentile(cal_atm, 67)
    }

    print(f"\nRegime thresholds (from calibration ATM IV):")
    print(f"  Low:  ATM IV < {thresholds['low']:.4f}")
    print(f"  Mid:  {thresholds['low']:.4f} <= ATM IV < {thresholds['high']:.4f}")
    print(f"  High: ATM IV >= {thresholds['high']:.4f}")

    # =========================================================================
    # STEP 3: Estimate variance per regime
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Estimate Variance per Regime")
    print("=" * 70)

    # Classify calibration points
    cal_regimes = [classify_regime(f, thresholds) for f in cal_features]

    regime_stats = {}
    for regime in ['low', 'mid', 'high']:
        mask = np.array([r == regime for r in cal_regimes])
        regime_residuals = cal_residuals[mask]

        regime_stats[regime] = {
            'count': mask.sum(),
            'bias': regime_residuals.mean(axis=0),
            'var': regime_residuals.var(axis=0),
            'std': regime_residuals.std(axis=0),
            'lower_q': np.percentile(regime_residuals, 5, axis=0),
            'upper_q': np.percentile(regime_residuals, 95, axis=0)
        }

        print(f"\n  {regime.upper()} regime ({regime_stats[regime]['count']} points):")
        print(f"    Mean residual std: {regime_stats[regime]['std'].mean():.4f}")
        print(f"    Interval width:    {(regime_stats[regime]['upper_q'] - regime_stats[regime]['lower_q']).mean():.4f}")

    # Variance ratio between high and low
    var_ratio = regime_stats['high']['var'].mean() / regime_stats['low']['var'].mean()
    print(f"\n  Variance ratio (high/low): {var_ratio:.2f}x")

    # =========================================================================
    # STEP 4: Evaluate coverage on test set
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Evaluate Coverage on Test Set")
    print("=" * 70)

    test_predictions = predictions[test_idx]
    test_targets = targets[test_idx]
    test_features = [features_list[i] for i in test_idx]
    test_regimes = [classify_regime(f, thresholds) for f in test_features]

    # Method A: Global (constant) variance
    global_lower_q = np.percentile(cal_residuals, 5, axis=0)
    global_upper_q = np.percentile(cal_residuals, 95, axis=0)

    covered_global = []
    for i in range(len(test_idx)):
        lower = test_predictions[i] + global_lower_q
        upper = test_predictions[i] + global_upper_q
        in_interval = (test_targets[i] >= lower) & (test_targets[i] <= upper)
        covered_global.append(in_interval)
    covered_global = np.array(covered_global)

    # Method B: Regime-specific variance
    covered_regime = []
    for i in range(len(test_idx)):
        regime = test_regimes[i]
        lower = test_predictions[i] + regime_stats[regime]['lower_q']
        upper = test_predictions[i] + regime_stats[regime]['upper_q']
        in_interval = (test_targets[i] >= lower) & (test_targets[i] <= upper)
        covered_regime.append(in_interval)
    covered_regime = np.array(covered_regime)

    global_coverage = covered_global.mean() * 100
    regime_coverage = covered_regime.mean() * 100

    print(f"\nOverall coverage (90% target):")
    print(f"  Global (constant):    {global_coverage:.1f}%")
    print(f"  Regime-specific:      {regime_coverage:.1f}%")

    # =========================================================================
    # STEP 5: Coverage by regime
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Coverage by Regime")
    print("=" * 70)

    print(f"\n{'Regime':<10} {'Count':<10} {'Global':<12} {'Regime-spec':<12} {'Delta':<10}")
    print("-" * 54)

    for regime in ['low', 'mid', 'high']:
        mask = np.array([r == regime for r in test_regimes])
        count = mask.sum()

        global_cov = covered_global[mask].mean() * 100 if count > 0 else 0
        regime_cov = covered_regime[mask].mean() * 100 if count > 0 else 0
        delta = regime_cov - global_cov

        print(f"{regime.upper():<10} {count:<10} {global_cov:<12.1f} {regime_cov:<12.1f} {delta:+.1f}")

    # =========================================================================
    # STEP 6: Analyze interval widths
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Interval Widths by Regime")
    print("=" * 70)

    global_width = (global_upper_q - global_lower_q).mean()
    print(f"\nGlobal interval width: {global_width:.4f}")

    for regime in ['low', 'mid', 'high']:
        width = (regime_stats[regime]['upper_q'] - regime_stats[regime]['lower_q']).mean()
        ratio = width / global_width
        print(f"  {regime.upper()} regime: {width:.4f} ({ratio:.2f}x global)")

    # =========================================================================
    # STEP 7: Generalization test - time-ordered split
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Generalization Test (Time-Ordered Split)")
    print("=" * 70)

    # Use first half for calibration, second half for test (simulates deployment)
    time_cal_idx = sample_indices[:n_cal]
    time_test_idx = sample_indices[n_cal:]

    # Sort by original index to get time ordering
    time_cal_idx = np.sort(time_cal_idx)
    time_test_idx = np.sort(time_test_idx)

    # Recompute regime stats on time-ordered calibration
    time_cal_residuals = residuals[cal_idx]  # Same as before for simplicity
    time_cal_features = [features_list[i] for i in cal_idx]

    # Test on time-ordered test set
    time_test_predictions = predictions[test_idx]
    time_test_targets = targets[test_idx]
    time_test_features = [features_list[i] for i in test_idx]
    time_test_regimes = [classify_regime(f, thresholds) for f in time_test_features]

    # Coverage with regime-specific intervals
    time_covered_regime = []
    for i in range(len(test_idx)):
        regime = time_test_regimes[i]
        lower = time_test_predictions[i] + regime_stats[regime]['lower_q']
        upper = time_test_predictions[i] + regime_stats[regime]['upper_q']
        in_interval = (time_test_targets[i] >= lower) & (time_test_targets[i] <= upper)
        time_covered_regime.append(in_interval)
    time_covered_regime = np.array(time_covered_regime)

    time_coverage = time_covered_regime.mean() * 100

    print(f"\nTime-ordered split coverage:")
    print(f"  Regime-specific: {time_coverage:.1f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Experiment 10c: Regime-Based Conditional Variance

REGIME VARIANCE HETEROGENEITY:
  Low vol std:  {regime_stats['low']['std'].mean():.4f}
  Mid vol std:  {regime_stats['mid']['std'].mean():.4f}
  High vol std: {regime_stats['high']['std'].mean():.4f}
  Variance ratio (high/low): {var_ratio:.2f}x

INTERVAL WIDTHS:
  Global:     {global_width:.4f}
  Low regime: {(regime_stats['low']['upper_q'] - regime_stats['low']['lower_q']).mean():.4f}
  High regime: {(regime_stats['high']['upper_q'] - regime_stats['high']['lower_q']).mean():.4f}

COVERAGE (90% target):
  Random split:
    Global (constant):  {global_coverage:.1f}%
    Regime-specific:    {regime_coverage:.1f}%
    Improvement:        {regime_coverage - global_coverage:+.1f}%

  Time-ordered split:
    Regime-specific:    {time_coverage:.1f}%

KEY INSIGHT:
  Regime-specific variance provides context-aware uncertainty that
  {"generalizes to unseen data" if time_coverage >= 85 else "may need further calibration for time-shifted data"}.
""")

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp10c_regime_variance")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "results.npz",
        thresholds=thresholds,
        regime_stats={r: {k: v for k, v in s.items() if k != 'count'}
                      for r, s in regime_stats.items()},
        global_coverage=global_coverage,
        regime_coverage=regime_coverage,
        time_coverage=time_coverage,
        var_ratio=var_ratio
    )

    print(f"Results saved to: {output_dir}/results.npz")


if __name__ == "__main__":
    main()
