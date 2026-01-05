"""
Experiment 3: Information Bottleneck Probe

Hypothesis: 12-dim context_summary loses information needed for uncertainty.

Method:
1. Extract context_summary (12-dim) vs raw_context (1500-dim) for test set
2. Train simple linear probes to predict:
   - Next-day volatility regime (high/low)
   - Next-day surface curvature
   - Next-day ATM volatility
3. Compare probe accuracy

Decision Point:
- If raw_context probe >> summary probe: Information lost, Arch B justified
- If similar: 12-dim is sufficient

Time: ~10 minutes, NO TRAINING
"""

import torch
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


def extract_representations(model, val_surface, num_samples=1000):
    """
    Extract context_summary (12-dim) and raw_context (flattened 1500-dim).

    Returns:
        context_summary: (N, 12)
        raw_context: (N, 1500)
        targets: dict with various prediction targets
    """
    C = model.context_len
    device = model.device
    model.eval()

    context_summaries = []
    raw_contexts = []

    # Target variables
    next_day_atm_vol = []  # Center grid point volatility
    next_day_avg_vol = []  # Average volatility
    next_day_skew = []  # Right-Left skew
    next_day_smile = []  # ATM - Average

    with torch.no_grad():
        for i in range(num_samples):
            if i + C + 1 > len(val_surface):
                break

            # Get context
            context = val_surface[i:i+C].unsqueeze(0).to(device)  # (1, C, 5, 5)
            next_day = val_surface[i+C].to(device)  # (5, 5)

            # Extract context summary (from context encoder)
            ctx_input = {"surface": context}
            ctx_out = model.ctx_encoder(ctx_input)
            context_summary = ctx_out[0, -1, :].cpu().numpy()  # (12,)

            # Extract raw context (flatten)
            raw_ctx = context.squeeze(0).cpu().numpy()  # (C, 5, 5)
            raw_ctx_flat = raw_ctx.flatten()  # (C*5*5 = 1500,)

            context_summaries.append(context_summary)
            raw_contexts.append(raw_ctx_flat)

            # Compute target variables
            # 1. ATM volatility (center point)
            atm_vol = next_day[2, 2].item()
            next_day_atm_vol.append(atm_vol)

            # 2. Average volatility
            avg_vol = next_day.mean().item()
            next_day_avg_vol.append(avg_vol)

            # 3. Skew: right side - left side
            right = next_day[:, 4].mean().item()  # High strike
            left = next_day[:, 0].mean().item()  # Low strike
            skew = right - left
            next_day_skew.append(skew)

            # 4. Smile: ATM - Average (convexity)
            smile = atm_vol - avg_vol
            next_day_smile.append(smile)

    context_summaries = np.array(context_summaries)
    raw_contexts = np.array(raw_contexts)

    targets = {
        'atm_vol': np.array(next_day_atm_vol),
        'avg_vol': np.array(next_day_avg_vol),
        'skew': np.array(next_day_skew),
        'smile': np.array(next_day_smile)
    }

    # Create binary classification targets
    # Volatility regime: high (above median) vs low (below median)
    median_vol = np.median(targets['atm_vol'])
    targets['vol_regime'] = (targets['atm_vol'] > median_vol).astype(int)

    # Skew regime: positive vs negative
    targets['skew_regime'] = (targets['skew'] > 0).astype(int)

    print(f"Extracted {len(context_summaries)} samples")
    print(f"Context summary shape: {context_summaries.shape}")
    print(f"Raw context shape: {raw_contexts.shape}")
    print()

    return context_summaries, raw_contexts, targets


def train_probes(X_summary, X_raw, targets, test_ratio=0.2):
    """
    Train linear probes on both representations and compare performance.

    Returns:
        dict with probe results
    """
    n_samples = len(X_summary)
    n_train = int(n_samples * (1 - test_ratio))

    # Split train/test
    X_summary_train = X_summary[:n_train]
    X_summary_test = X_summary[n_train:]
    X_raw_train = X_raw[:n_train]
    X_raw_test = X_raw[n_train:]

    # Standardize features
    scaler_summary = StandardScaler()
    scaler_raw = StandardScaler()

    X_summary_train = scaler_summary.fit_transform(X_summary_train)
    X_summary_test = scaler_summary.transform(X_summary_test)
    X_raw_train = scaler_raw.fit_transform(X_raw_train)
    X_raw_test = scaler_raw.transform(X_raw_test)

    results = {}

    # =========================================================================
    # Classification Tasks
    # =========================================================================

    classification_tasks = ['vol_regime', 'skew_regime']

    for task in classification_tasks:
        y = targets[task]
        y_train = y[:n_train]
        y_test = y[n_train:]

        # Train on context summary
        clf_summary = LogisticRegression(max_iter=1000, random_state=42)
        clf_summary.fit(X_summary_train, y_train)
        pred_summary = clf_summary.predict(X_summary_test)
        acc_summary = accuracy_score(y_test, pred_summary)

        # Train on raw context
        clf_raw = LogisticRegression(max_iter=1000, random_state=42)
        clf_raw.fit(X_raw_train, y_train)
        pred_raw = clf_raw.predict(X_raw_test)
        acc_raw = accuracy_score(y_test, pred_raw)

        results[task] = {
            'summary_accuracy': acc_summary,
            'raw_accuracy': acc_raw,
            'improvement': acc_raw - acc_summary,
            'relative_improvement': (acc_raw - acc_summary) / acc_summary if acc_summary > 0 else 0
        }

    # =========================================================================
    # Regression Tasks
    # =========================================================================

    regression_tasks = ['atm_vol', 'avg_vol', 'skew', 'smile']

    for task in regression_tasks:
        y = targets[task]
        y_train = y[:n_train]
        y_test = y[n_train:]

        # Train on context summary
        reg_summary = Ridge(alpha=1.0, random_state=42)
        reg_summary.fit(X_summary_train, y_train)
        pred_summary = reg_summary.predict(X_summary_test)
        r2_summary = r2_score(y_test, pred_summary)

        # Train on raw context
        reg_raw = Ridge(alpha=1.0, random_state=42)
        reg_raw.fit(X_raw_train, y_train)
        pred_raw = reg_raw.predict(X_raw_test)
        r2_raw = r2_score(y_test, pred_raw)

        results[task] = {
            'summary_r2': r2_summary,
            'raw_r2': r2_raw,
            'improvement': r2_raw - r2_summary,
            'relative_improvement': (r2_raw - r2_summary) / r2_summary if r2_summary > 0 else 0
        }

    return results


def main():
    print("=" * 80)
    print("EXPERIMENT 3: Information Bottleneck Probe")
    print("=" * 80)
    print()

    # Load configuration
    config = BackfillContext60ConfigV4FullCov

    # Find the trained model
    checkpoint_path = Path("models/backfill/context60_v4_full_cov/checkpoints/backfill_context60_latent12_v4_full_cov_phase1_ep99.pt")

    if not checkpoint_path.exists():
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        print("Please ensure the model has been trained.")
        return

    print(f"Loading model from: {checkpoint_path}")
    model_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Strip _orig_mod. prefix from compiled models
    if "model" in model_data and any(k.startswith("_orig_mod.") for k in model_data["model"].keys()):
        model_data["model"] = {k.replace("_orig_mod.", ""): v for k, v in model_data["model"].items()}

    # Initialize model
    model = CVAEFullCovPrior(model_data["model_config"])
    model.load_weights(dict_to_load=model_data)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.device = device
    print(f"Model loaded on device: {device}")
    print()

    # Load data
    print("Loading data...")
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)

    surface = torch.tensor(data["surface"], dtype=torch.float32)

    # Use validation set
    train_end = config.train_end_idx
    val_surface = surface[train_end:train_end+2000]

    print(f"Validation data shape: {val_surface.shape}")
    print()

    # Extract representations
    print("Extracting representations...")
    context_summaries, raw_contexts, targets = extract_representations(
        model, val_surface, num_samples=1000
    )

    # Train probes
    print("Training linear probes...")
    results = train_probes(context_summaries, raw_contexts, targets, test_ratio=0.2)

    print()
    print("=" * 80)
    print("RESULTS: Information Bottleneck Analysis")
    print("=" * 80)
    print()

    # Classification results
    print("CLASSIFICATION TASKS (Accuracy)")
    print("-" * 80)
    print(f"{'Task':<20} {'Summary (12d)':<15} {'Raw (1500d)':<15} {'Improvement':<15} {'Relative %':<15}")
    print("-" * 80)

    for task in ['vol_regime', 'skew_regime']:
        r = results[task]
        rel_pct = r['relative_improvement'] * 100
        print(f"{task:<20} {r['summary_accuracy']:<15.4f} {r['raw_accuracy']:<15.4f} {r['improvement']:<15.4f} {rel_pct:<15.2f}%")

    print()

    # Regression results
    print("REGRESSION TASKS (R² Score)")
    print("-" * 80)
    print(f"{'Task':<20} {'Summary (12d)':<15} {'Raw (1500d)':<15} {'Improvement':<15} {'Relative %':<15}")
    print("-" * 80)

    for task in ['atm_vol', 'avg_vol', 'skew', 'smile']:
        r = results[task]
        rel_pct = r['relative_improvement'] * 100
        print(f"{task:<20} {r['summary_r2']:<15.4f} {r['raw_r2']:<15.4f} {r['improvement']:<15.4f} {rel_pct:<15.2f}%")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Compute average relative improvement
    all_rel_improvements = [results[task]['relative_improvement'] for task in results.keys()]
    avg_rel_improvement = np.mean(all_rel_improvements) * 100

    print()
    print(f"Average Relative Improvement: {avg_rel_improvement:.1f}%")
    print()

    if avg_rel_improvement > 20:
        print("✓ SIGNIFICANT INFORMATION LOSS DETECTED")
        print(f"  Raw context provides {avg_rel_improvement:.1f}% better predictions on average")
        print("  12-dim bottleneck loses critical information")
        print("  → Architecture B (Prior Encoder) justified - use full 1500-dim context")
    elif avg_rel_improvement > 10:
        print("⚠ MODERATE INFORMATION LOSS")
        print(f"  Raw context provides {avg_rel_improvement:.1f}% better predictions")
        print("  Some information lost in compression")
        print("  → Architecture B may help but gains may be modest")
    else:
        print("⚠ MINIMAL INFORMATION LOSS")
        print(f"  Raw context provides only {avg_rel_improvement:.1f}% improvement")
        print("  12-dim summary captures most relevant information")
        print("  → Information bottleneck is not the primary issue")

    print()

    # Detailed analysis
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()

    # Find tasks with biggest gaps
    improvements = [(task, results[task]['relative_improvement']) for task in results.keys()]
    improvements.sort(key=lambda x: x[1], reverse=True)

    print("Tasks with largest information gap:")
    for task, imp in improvements[:3]:
        print(f"  - {task}: {imp*100:.1f}% improvement with raw context")

    print()

    # Save results
    output_dir = Path("results/prior_encoder_ablation/information_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "information_probe_results.npz"

    # Convert results dict to saveable format
    save_dict = {}
    for task, task_results in results.items():
        for key, value in task_results.items():
            save_dict[f"{task}_{key}"] = value

    save_dict['avg_relative_improvement'] = avg_rel_improvement / 100

    np.savez(output_file, **save_dict)

    print(f"Results saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
