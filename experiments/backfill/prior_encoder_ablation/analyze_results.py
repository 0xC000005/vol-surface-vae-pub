"""
Analyze and Visualize Results from Prior Encoder Ablation Experiments

This script aggregates results from all 5 experiments and creates:
1. Summary table with key findings
2. Visualizations comparing architectures
3. Recommendations for next steps
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def load_experiment_results():
    """Load results from all experiments."""
    base_dir = Path("results/prior_encoder_ablation")

    results = {}

    # Experiment 1: Gradient Flow
    exp1_file = base_dir / "gradient_flow/gradient_flow_results.npz"
    if exp1_file.exists():
        data = np.load(exp1_file)
        results['gradient_flow'] = {
            'total_recon_grad': float(data['total_recon_grad']),
            'total_kl_grad': float(data['total_kl_grad']),
            'overall_ratio': float(data['overall_ratio'])
        }
        print("✓ Loaded Experiment 1: Gradient Flow")
    else:
        print("✗ Experiment 1 results not found")
        results['gradient_flow'] = None

    # Experiment 2: Variance Baseline
    exp2_file = base_dir / "variance_baseline/variance_baseline_results.npz"
    if exp2_file.exists():
        data = np.load(exp2_file)
        results['variance_baseline'] = {
            'conditional_var_ratio': float(data['conditional_var_ratio']),
            'expected_conditional_var': float(data['expected_conditional_var']),
            'mean_total_var': float(data['mean_total_var'])
        }
        print("✓ Loaded Experiment 2: Variance Baseline")
    else:
        print("✗ Experiment 2 results not found")
        results['variance_baseline'] = None

    # Experiment 3: Information Probe
    exp3_file = base_dir / "information_probe/information_probe_results.npz"
    if exp3_file.exists():
        data = np.load(exp3_file)
        results['information_probe'] = {
            'avg_relative_improvement': float(data['avg_relative_improvement'])
        }
        print("✓ Loaded Experiment 3: Information Probe")
    else:
        print("✗ Experiment 3 results not found")
        results['information_probe'] = None

    # Experiment 4: Frozen Finetune
    exp4_file = base_dir / "frozen_finetune/frozen_finetune_results.npz"
    if exp4_file.exists():
        data = np.load(exp4_file)
        results['frozen_finetune'] = {
            'baseline_ratio': float(data['baseline_ratio']),
            'before_ratio': float(data['before_ratio']),
            'after_ratio': float(data['after_ratio']),
            'improvement': float(data['improvement'])
        }
        print("✓ Loaded Experiment 4: Frozen Finetune")
    else:
        print("✗ Experiment 4 results not found")
        results['frozen_finetune'] = None

    # Experiment 5: Short Training
    exp5_file = base_dir / "short_training/comparison_results.npz"
    if exp5_file.exists():
        data = np.load(exp5_file)

        # Extract final conditional variance ratios for each variant
        baseline_cond_var = [x for x in data['baseline_conditional_var_ratio'] if x is not None]
        diagonal_cond_var = [x for x in data['prior_encoder_diagonal_conditional_var_ratio'] if x is not None]
        full_cov_cond_var = [x for x in data['prior_encoder_full_cov_conditional_var_ratio'] if x is not None]

        results['short_training'] = {
            'baseline_best': float(max(baseline_cond_var)) if baseline_cond_var else 0.0,
            'diagonal_best': float(max(diagonal_cond_var)) if diagonal_cond_var else 0.0,
            'full_cov_best': float(max(full_cov_cond_var)) if full_cov_cond_var else 0.0,
            'baseline_final': float(baseline_cond_var[-1]) if baseline_cond_var else 0.0,
            'diagonal_final': float(diagonal_cond_var[-1]) if diagonal_cond_var else 0.0,
            'full_cov_final': float(full_cov_cond_var[-1]) if full_cov_cond_var else 0.0,
        }
        print("✓ Loaded Experiment 5: Short Training")
    else:
        print("✗ Experiment 5 results not found")
        results['short_training'] = None

    return results


def print_summary(results):
    """Print comprehensive summary of all experiments."""
    print("\n" + "=" * 80)
    print("PRIOR ENCODER ABLATION: COMPREHENSIVE RESULTS")
    print("=" * 80)
    print()

    # Experiment 1: Gradient Flow
    if results['gradient_flow']:
        print("EXPERIMENT 1: Gradient Flow Analysis")
        print("-" * 80)
        r = results['gradient_flow']
        print(f"Total Reconstruction Gradient: {r['total_recon_grad']:.6e}")
        print(f"Total KL Gradient:             {r['total_kl_grad']:.6e}")
        print(f"Ratio (KL / Recon):            {r['overall_ratio']:.2f}")
        print()
        if r['overall_ratio'] > 2.0:
            print("→ GRADIENT CONFOUNDING CONFIRMED")
        elif r['overall_ratio'] > 0.5:
            print("→ Gradients are balanced")
        else:
            print("→ Reconstruction dominates")
        print()

    # Experiment 2: Variance Baseline
    if results['variance_baseline']:
        print("EXPERIMENT 2: Variance Baseline")
        print("-" * 80)
        r = results['variance_baseline']
        print(f"E[Var(X|C)] / Var(X):          {r['conditional_var_ratio']:.4%}")
        print(f"E[Var(X|C)]:                   {r['expected_conditional_var']:.8f}")
        print(f"Var(X):                        {r['mean_total_var']:.8f}")
        print()
        if r['conditional_var_ratio'] < 0.01:
            print("→ VERY LOW conditional variance (< 1%)")
        elif r['conditional_var_ratio'] < 0.05:
            print("→ LOW conditional variance (< 5%)")
        else:
            print("→ HEALTHY conditional variance")
        print()

    # Experiment 3: Information Probe
    if results['information_probe']:
        print("EXPERIMENT 3: Information Bottleneck Probe")
        print("-" * 80)
        r = results['information_probe']
        improvement_pct = r['avg_relative_improvement'] * 100
        print(f"Avg Relative Improvement (Raw vs Summary): {improvement_pct:.1f}%")
        print()
        if improvement_pct > 20:
            print("→ SIGNIFICANT information loss in 12-dim bottleneck")
        elif improvement_pct > 10:
            print("→ MODERATE information loss")
        else:
            print("→ MINIMAL information loss")
        print()

    # Experiment 4: Frozen Finetune
    if results['frozen_finetune']:
        print("EXPERIMENT 4: Frozen Encoder Fine-tuning")
        print("-" * 80)
        r = results['frozen_finetune']
        print(f"Baseline (Original Model):     {r['baseline_ratio']:.4%}")
        print(f"Before Training (Untrained):   {r['before_ratio']:.4%}")
        print(f"After Training (20 epochs):    {r['after_ratio']:.4%}")
        print(f"Improvement Factor:            {r['improvement']:.2f}×")
        print()
        if r['improvement'] > 4.0:
            print("→ MAJOR improvement")
        elif r['improvement'] > 2.0:
            print("→ SIGNIFICANT improvement")
        elif r['improvement'] > 1.5:
            print("→ MODERATE improvement")
        else:
            print("→ MINIMAL improvement")
        print()

    # Experiment 5: Short Training
    if results['short_training']:
        print("EXPERIMENT 5: Short Training Comparison (30 epochs)")
        print("-" * 80)
        r = results['short_training']
        print(f"{'Model':<30} {'Best Ratio':<15} {'Final Ratio':<15} {'vs Baseline':<15}")
        print("-" * 80)
        print(f"{'Baseline':<30} {r['baseline_best']:<15.4%} {r['baseline_final']:<15.4%} {'1.00×':<15}")
        diag_improvement = r['diagonal_best'] / r['baseline_best']
        print(f"{'Prior Encoder Diagonal':<30} {r['diagonal_best']:<15.4%} {r['diagonal_final']:<15.4%} {diag_improvement:<15.2f}×")
        full_improvement = r['full_cov_best'] / r['baseline_best']
        print(f"{'Prior Encoder Full Cov':<30} {r['full_cov_best']:<15.4%} {r['full_cov_final']:<15.4%} {full_improvement:<15.2f}×")
        print()

        best_improvement = max(diag_improvement, full_improvement)
        if best_improvement > 4.0:
            print("→ STRONG evidence for Prior Encoder")
        elif best_improvement > 2.0:
            print("→ MODERATE evidence for Prior Encoder")
        else:
            print("→ WEAK evidence for Prior Encoder")
        print()

    # Overall Conclusion
    print("=" * 80)
    print("OVERALL CONCLUSION")
    print("=" * 80)
    print()

    # Count evidence for Prior Encoder
    evidence_points = []

    if results['gradient_flow'] and results['gradient_flow']['overall_ratio'] > 2.0:
        evidence_points.append("✓ Gradient confounding confirmed")

    if results['variance_baseline'] and results['variance_baseline']['conditional_var_ratio'] < 0.01:
        evidence_points.append("✓ Very low conditional variance (< 1%)")

    if results['information_probe'] and results['information_probe']['avg_relative_improvement'] > 0.2:
        evidence_points.append("✓ Significant information loss in bottleneck")

    if results['frozen_finetune'] and results['frozen_finetune']['improvement'] > 2.0:
        evidence_points.append("✓ Prior Encoder improves conditional variance")

    if results['short_training']:
        r = results['short_training']
        best_imp = max(r['diagonal_best'] / r['baseline_best'], r['full_cov_best'] / r['baseline_best'])
        if best_imp > 2.0:
            evidence_points.append("✓ Short training shows improvement")

    print(f"Evidence for Prior Encoder Architecture: {len(evidence_points)}/5 experiments")
    print()
    for point in evidence_points:
        print(f"  {point}")
    print()

    if len(evidence_points) >= 4:
        print("RECOMMENDATION: STRONG support for Prior Encoder")
        print("→ Proceed with full 400-epoch training")
        print("→ Prioritize full covariance variant")
    elif len(evidence_points) >= 3:
        print("RECOMMENDATION: MODERATE support for Prior Encoder")
        print("→ Consider full training with hyperparameter tuning")
    elif len(evidence_points) >= 2:
        print("RECOMMENDATION: WEAK support for Prior Encoder")
        print("→ May need architecture modifications")
    else:
        print("RECOMMENDATION: INSUFFICIENT evidence")
        print("→ Problem likely lies elsewhere (decoder, KL weight, training schedule)")

    print()
    print("=" * 80)


def create_visualizations(results):
    """Create visualization plots."""
    output_dir = Path("results/prior_encoder_ablation/summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Conditional Variance Comparison
    if results['variance_baseline'] and results['frozen_finetune'] and results['short_training']:
        fig, ax = plt.subplots(figsize=(10, 6))

        models = ['Baseline\n(Original)', 'Untrained\nPrior Enc', 'Trained\nPrior Enc\n(20 ep)',
                  'Baseline\n(30 ep)', 'Diagonal\n(30 ep)', 'Full Cov\n(30 ep)']
        ratios = [
            results['variance_baseline']['conditional_var_ratio'],
            results['frozen_finetune']['before_ratio'],
            results['frozen_finetune']['after_ratio'],
            results['short_training']['baseline_best'],
            results['short_training']['diagonal_best'],
            results['short_training']['full_cov_best']
        ]

        bars = ax.bar(models, [r * 100 for r in ratios], color=['red', 'orange', 'yellow', 'lightblue', 'blue', 'darkblue'])
        ax.axhline(y=2.0, color='green', linestyle='--', label='Target: 2%')
        ax.set_ylabel('E[Var(X|C)] / Var(X) (%)')
        ax.set_title('Conditional Variance Ratio Comparison')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "conditional_variance_comparison.png", dpi=150)
        print(f"✓ Saved: {output_dir / 'conditional_variance_comparison.png'}")
        plt.close()

    # Figure 2: Gradient Flow
    if results['gradient_flow']:
        fig, ax = plt.subplots(figsize=(8, 6))

        r = results['gradient_flow']
        bars = ax.bar(['Reconstruction\nGradient', 'KL\nGradient'],
                      [r['total_recon_grad'], r['total_kl_grad']])
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Context Encoder Gradient Sources')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "gradient_flow.png", dpi=150)
        print(f"✓ Saved: {output_dir / 'gradient_flow.png'}")
        plt.close()


def main():
    print("=" * 80)
    print("PRIOR ENCODER ABLATION: Result Analysis")
    print("=" * 80)
    print()

    # Load all results
    print("Loading experiment results...")
    results = load_experiment_results()
    print()

    # Print summary
    print_summary(results)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results)

    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
