"""
Verify Evaluation Alignment for Two-Stage VAE

This script tests the hypothesis that negative direction correlations are due to
an off-by-one error in evaluation: the model predicts x[t+1] but we compare to x[t].

With mean-reverting data (ACF = -0.325), comparing prediction-for-x[t+1] to x[t]
gives negative correlation because when x[t] is high, x[t+1] tends to be low.

Tests two alignments:
1. CURRENT: pred[h] vs gt[h] (what we do now)
2. FIXED: pred[h] vs gt[h+1] (if pred is for next step)

Expected outcome: If hypothesis is correct, direction accuracy should jump from
~29% to ~60%+ after fixing alignment.

Usage:
    python experiments/backfill/two_stage_vae/verify_evaluation_alignment.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.backfill.two_stage_vae.exp_student_t_decoder import (
    CVAETwoStageStudentT,
    to_log_returns,
)


# ============================================================================
# Configuration
# ============================================================================

PERIODS = {
    "Vol Spike (Sep 2008)": 2100,
    "Crisis Peak (Oct 2008)": 2150,
    "Recovery (Mar 2009)": 2280,
    "Debt Ceiling (Aug 2011)": 2900,
    "Calm (2017)": 4300,
}

CONTEXT_LEN = 20
HORIZON = 30  # Will test up to HORIZON-1 since we need gt[h+1]


def load_model(device: str = "cuda"):
    """Load the Student-t model."""
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def load_data():
    """Load and prepare data."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)
    return surfaces, log_returns


def verify_alignment(model, log_returns, device):
    """
    Compare two evaluation alignments:
    1. CURRENT: pred[h] vs gt[h]
    2. FIXED: pred[h] vs gt[h+1]
    """
    print("\n" + "="*70)
    print("EVALUATION ALIGNMENT VERIFICATION")
    print("="*70)

    print("\n  HYPOTHESIS:")
    print("    Training: mean[t] predicts x[t+1] (next step)")
    print("    Current eval: compares mean[-1] to gt[h]")
    print("    If mean[-1] predicts gt[h+1], comparing to gt[h] gives wrong sign!")
    print("    With mean-reverting data (ACF=-0.325), this explains negative correlations.")

    results_current = defaultdict(list)
    results_fixed = defaultdict(list)

    for period_name, start_idx in PERIODS.items():
        print(f"\n  Processing {period_name}...")

        # Get context and extended ground truth (need h+1)
        context = log_returns[start_idx:start_idx + CONTEXT_LEN]
        gt_returns = log_returns[start_idx + CONTEXT_LEN:start_idx + CONTEXT_LEN + HORIZON + 1]

        if len(gt_returns) < HORIZON + 1:
            print(f"    Skipping - insufficient data")
            continue

        pred_means = []

        with torch.no_grad():
            for h in range(HORIZON):
                if h == 0:
                    full_seq = np.concatenate([context, gt_returns[:1]], axis=0)
                else:
                    full_seq = np.concatenate([context, gt_returns[:h+1]], axis=0)

                seq_tensor = torch.tensor(full_seq, dtype=torch.float32).unsqueeze(0).to(device)
                batch = {"surface": seq_tensor}

                mean, _, _, _, _ = model(batch, return_full_sequence=True)
                pred_h = mean[0, -1, 2, 2].cpu().numpy()  # ATM, last position
                pred_means.append(pred_h)

        pred_means = np.array(pred_means)

        # ATM ground truth
        gt_atm = gt_returns[:, 2, 2]

        # ===============================================================
        # CURRENT ALIGNMENT: pred[h] vs gt[h]
        # ===============================================================
        gt_current = gt_atm[:HORIZON]

        # Direction accuracy
        current_direction_acc = (np.sign(pred_means) == np.sign(gt_current)).mean()

        # Correlation
        current_corr, _ = pearsonr(pred_means, gt_current)

        # Magnitude
        current_magnitude_ratio = np.abs(pred_means).mean() / np.abs(gt_current).mean()

        results_current["direction_acc"].append(current_direction_acc)
        results_current["correlation"].append(current_corr)
        results_current["magnitude_ratio"].append(current_magnitude_ratio)

        # ===============================================================
        # FIXED ALIGNMENT: pred[h] vs gt[h+1]
        # ===============================================================
        # pred[h] is the model's prediction at position (context_len + h)
        # In training: mean[t] predicts x[t+1]
        # So pred[h] should predict x[context_len + h + 1] = gt_returns[h+1]
        gt_fixed = gt_atm[1:HORIZON+1]  # gt[1], gt[2], ..., gt[HORIZON]

        # Direction accuracy
        fixed_direction_acc = (np.sign(pred_means) == np.sign(gt_fixed)).mean()

        # Correlation
        fixed_corr, _ = pearsonr(pred_means, gt_fixed)

        # Magnitude
        fixed_magnitude_ratio = np.abs(pred_means).mean() / np.abs(gt_fixed).mean()

        results_fixed["direction_acc"].append(fixed_direction_acc)
        results_fixed["correlation"].append(fixed_corr)
        results_fixed["magnitude_ratio"].append(fixed_magnitude_ratio)

        # Print comparison for this period
        print(f"\n    {period_name}:")
        print(f"      CURRENT (pred[h] vs gt[h]):")
        print(f"        Direction Accuracy: {current_direction_acc*100:.1f}%")
        print(f"        Correlation:        {current_corr:.3f}")
        print(f"        Magnitude Ratio:    {current_magnitude_ratio*100:.1f}%")
        print(f"      FIXED (pred[h] vs gt[h+1]):")
        print(f"        Direction Accuracy: {fixed_direction_acc*100:.1f}%")
        print(f"        Correlation:        {fixed_corr:.3f}")
        print(f"        Magnitude Ratio:    {fixed_magnitude_ratio*100:.1f}%")
        print(f"      IMPROVEMENT:")
        print(f"        Direction Accuracy: {(fixed_direction_acc - current_direction_acc)*100:+.1f}%")
        print(f"        Correlation:        {fixed_corr - current_corr:+.3f}")

    # ===============================================================
    # SUMMARY
    # ===============================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n  CURRENT ALIGNMENT (pred[h] vs gt[h]):")
    print(f"    Avg Direction Accuracy: {np.mean(results_current['direction_acc'])*100:.1f}%")
    print(f"    Avg Correlation:        {np.mean(results_current['correlation']):.3f}")
    print(f"    Avg Magnitude Ratio:    {np.mean(results_current['magnitude_ratio'])*100:.1f}%")

    print("\n  FIXED ALIGNMENT (pred[h] vs gt[h+1]):")
    print(f"    Avg Direction Accuracy: {np.mean(results_fixed['direction_acc'])*100:.1f}%")
    print(f"    Avg Correlation:        {np.mean(results_fixed['correlation']):.3f}")
    print(f"    Avg Magnitude Ratio:    {np.mean(results_fixed['magnitude_ratio'])*100:.1f}%")

    # Calculate improvement
    dir_improvement = np.mean(results_fixed['direction_acc']) - np.mean(results_current['direction_acc'])
    corr_improvement = np.mean(results_fixed['correlation']) - np.mean(results_current['correlation'])

    print("\n  IMPROVEMENT (FIXED - CURRENT):")
    print(f"    Direction Accuracy: {dir_improvement*100:+.1f}%")
    print(f"    Correlation:        {corr_improvement:+.3f}")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if corr_improvement > 0.3 and dir_improvement > 0.15:
        print("\n  ✓ HYPOTHESIS CONFIRMED!")
        print("    The off-by-one error explains the negative correlations.")
        print("    The model IS predicting correctly for the next step.")
        print("\n  RECOMMENDATION:")
        print("    Fix the evaluation code to compare pred[h] with gt[h+1].")
        print("    No architectural changes needed for direction accuracy.")
        verdict = "CONFIRMED"
    elif corr_improvement > 0.1:
        print("\n  ~ HYPOTHESIS PARTIALLY CONFIRMED")
        print("    Some improvement with fixed alignment, but not dramatic.")
        print("    May need additional fixes beyond alignment.")
        verdict = "PARTIAL"
    else:
        print("\n  ✗ HYPOTHESIS NOT CONFIRMED")
        print("    Alignment fix doesn't significantly improve metrics.")
        print("    Root cause is elsewhere - consider architectural changes.")
        verdict = "NOT_CONFIRMED"

    return {
        "current": {
            "avg_direction_acc": float(np.mean(results_current["direction_acc"])),
            "avg_correlation": float(np.mean(results_current["correlation"])),
            "avg_magnitude_ratio": float(np.mean(results_current["magnitude_ratio"])),
            "per_period_direction_acc": [float(x) for x in results_current["direction_acc"]],
            "per_period_correlation": [float(x) for x in results_current["correlation"]],
        },
        "fixed": {
            "avg_direction_acc": float(np.mean(results_fixed["direction_acc"])),
            "avg_correlation": float(np.mean(results_fixed["correlation"])),
            "avg_magnitude_ratio": float(np.mean(results_fixed["magnitude_ratio"])),
            "per_period_direction_acc": [float(x) for x in results_fixed["direction_acc"]],
            "per_period_correlation": [float(x) for x in results_fixed["correlation"]],
        },
        "improvement": {
            "direction_acc": float(dir_improvement),
            "correlation": float(corr_improvement),
        },
        "verdict": verdict,
    }


def test_training_prediction_semantics(model, log_returns, device):
    """
    Directly verify what the model predicts.

    Training setup:
        target = batch_data[:, 1:]   # x[1:T]
        pred = mean[:, :-1]          # mean[0:T-1]
        loss = MSE(pred, target)

    So mean[t] is trained to minimize MSE with x[t+1].
    This means mean[t] predicts x[t+1], NOT x[t].
    """
    print("\n" + "="*70)
    print("TRAINING PREDICTION SEMANTICS VERIFICATION")
    print("="*70)

    # Use a simple test: pass sequence [x0, x1, x2, ..., xN]
    # mean[0] should predict x1, mean[1] should predict x2, etc.

    test_idx = 100
    test_len = 30

    test_seq = log_returns[test_idx:test_idx + test_len]
    seq_tensor = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0).to(device)
    batch = {"surface": seq_tensor}

    with torch.no_grad():
        mean, _, _, _, _ = model(batch, return_full_sequence=True)

    mean_np = mean[0].cpu().numpy()  # (T, 5, 5)

    # Check correlation patterns at ATM
    print("\n  Testing: Is mean[t] closer to x[t] or x[t+1]?")
    print("  (Using ATM grid point [2,2])")

    gt_current = test_seq[:-1, 2, 2]    # x[0:T-1] for comparison with mean[0:T-1]
    gt_next = test_seq[1:, 2, 2]        # x[1:T] for comparison with mean[0:T-1]
    pred = mean_np[:-1, 2, 2]           # mean[0:T-1]

    # Correlation with current vs next
    corr_current, _ = pearsonr(pred, gt_current)
    corr_next, _ = pearsonr(pred, gt_next)

    # MSE with current vs next
    mse_current = ((pred - gt_current) ** 2).mean()
    mse_next = ((pred - gt_next) ** 2).mean()

    print(f"\n    mean[0:T-1] vs x[0:T-1] (current):")
    print(f"      Correlation: {corr_current:.3f}")
    print(f"      MSE:         {mse_current:.6f}")

    print(f"\n    mean[0:T-1] vs x[1:T] (next):")
    print(f"      Correlation: {corr_next:.3f}")
    print(f"      MSE:         {mse_next:.6f}")

    print("\n  INTERPRETATION:")
    if corr_next > corr_current and mse_next < mse_current:
        print("    ✓ mean[t] predicts x[t+1] (next step)")
        print("    This confirms training semantics: mean[t] → x[t+1]")
        confirmed = True
    elif corr_current > corr_next and mse_current < mse_next:
        print("    ✗ mean[t] predicts x[t] (current step)")
        print("    Evaluation alignment is correct")
        confirmed = False
    else:
        print("    ? Ambiguous - correlations and MSEs don't align")
        print(f"      Correlation difference (next - current): {corr_next - corr_current:.3f}")
        print(f"      MSE difference (current - next): {mse_current - mse_next:.6f}")
        confirmed = corr_next > corr_current

    return {
        "corr_with_current": float(corr_current),
        "corr_with_next": float(corr_next),
        "mse_with_current": float(mse_current),
        "mse_with_next": float(mse_next),
        "predicts_next_step": confirmed,
    }


def main():
    print("="*70)
    print("EVALUATION ALIGNMENT VERIFICATION FOR TWO-STAGE VAE")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nLoading model and data...")
    model, config = load_model(device)
    surfaces, log_returns = load_data()
    print(f"Data shape: {log_returns.shape}")

    # First: verify training semantics
    semantics_results = test_training_prediction_semantics(model, log_returns, device)

    # Second: compare alignments on all periods
    alignment_results = verify_alignment(model, log_returns, device)

    # Save results
    import json
    output_dir = Path("results/two_stage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "training_semantics": semantics_results,
        "alignment_comparison": alignment_results,
    }

    with open(output_dir / "alignment_verification_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'alignment_verification_results.json'}")

    # Final conclusion
    print("\n" + "="*70)
    print("FINAL CONCLUSION")
    print("="*70)

    if semantics_results["predicts_next_step"] and alignment_results["verdict"] == "CONFIRMED":
        print("\n  The off-by-one hypothesis is CONFIRMED by both tests.")
        print("\n  NEXT STEPS:")
        print("    1. Fix comprehensive_oracle_analysis.py to use gt[h+1]")
        print("    2. Re-run comprehensive analysis")
        print("    3. Focus on remaining issues (magnitude attenuation, ACF)")
    elif alignment_results["verdict"] == "PARTIAL":
        print("\n  Partial evidence for off-by-one error.")
        print("  Some additional issues may exist.")
    else:
        print("\n  Off-by-one hypothesis NOT confirmed.")
        print("  Need to investigate other causes.")

    return results


if __name__ == "__main__":
    results = main()
