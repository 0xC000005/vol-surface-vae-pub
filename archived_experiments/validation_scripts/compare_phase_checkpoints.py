"""
Compare Phase Checkpoints for Context=60 Training

This script loads all 4 phase checkpoints and compares their performance
to understand how the model evolved through the 4-phase curriculum.

Usage:
    python compare_phase_checkpoints.py

Outputs:
    - Terminal table showing metrics progression
    - Plot: models_backfill/phase_comparison.png
    - Detailed text report: models_backfill/phase_comparison.txt
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config.backfill_context60_config import BackfillContext60Config as cfg


def load_checkpoint(checkpoint_path):
    """
    Load checkpoint and extract key information.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        dict with keys:
            - epoch: Epoch number
            - phase: Phase number
            - phase_name: Phase name
            - val_metrics: Validation metrics dict
            - timestamp: Training timestamp
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return {
        "epoch": checkpoint.get("epoch", -1),
        "phase": checkpoint.get("phase", -1),
        "phase_name": checkpoint.get("phase_name", "Unknown"),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "timestamp": checkpoint.get("timestamp", 0),
    }


def format_metric(value, precision=6):
    """Format metric value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.{precision}f}"


def print_comparison_table(phase_data):
    """
    Print formatted comparison table.

    Args:
        phase_data: List of dicts with checkpoint data
    """
    print("\n" + "=" * 100)
    print("PHASE CHECKPOINT COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Phase':<8} {'Epoch':<8} {'Description':<35} {'Val Loss':<12} {'Recon Loss':<12} {'KL Loss':<12}"
    print(header)
    print("-" * 100)

    # Rows
    for data in phase_data:
        phase = data['phase']
        epoch = data['epoch']
        phase_name = data['phase_name']
        metrics = data['val_metrics']

        val_loss = format_metric(metrics.get('total_loss'))
        recon_loss = format_metric(metrics.get('re_loss'))
        kl_loss = format_metric(metrics.get('kl_loss'))

        row = f"{phase:<8} {epoch:<8} {phase_name:<35} {val_loss:<12} {recon_loss:<12} {kl_loss:<12}"
        print(row)

    print("=" * 100)
    print()


def analyze_metric_changes(phase_data, metric_key, metric_name):
    """
    Analyze how a metric changed across phases.

    Args:
        phase_data: List of checkpoint data dicts
        metric_key: Key in val_metrics dict
        metric_name: Human-readable metric name
    """
    print(f"\n{metric_name} Progression:")
    print("-" * 60)

    values = []
    for data in phase_data:
        val = data['val_metrics'].get(metric_key)
        values.append(val)

        # Print value and change
        if len(values) == 1:
            print(f"  Phase {data['phase']}: {format_metric(val)}")
        else:
            prev_val = values[-2]
            if val is not None and prev_val is not None:
                change = val - prev_val
                pct_change = (change / prev_val) * 100 if prev_val != 0 else 0
                change_str = f"{change:+.6f}" if abs(change) >= 1e-6 else f"{change:+.2e}"
                pct_str = f"({pct_change:+.2f}%)"
                print(f"  Phase {data['phase']}: {format_metric(val)} [{change_str} {pct_str}]")
            else:
                print(f"  Phase {data['phase']}: {format_metric(val)}")

    # Summary
    if all(v is not None for v in values):
        total_change = values[-1] - values[0]
        pct_total = (total_change / values[0]) * 100 if values[0] != 0 else 0
        print(f"\n  Total change: {total_change:+.6f} ({pct_total:+.2f}%)")

        if total_change < 0:
            print(f"  ✅ {metric_name} IMPROVED (decreased by {abs(pct_total):.2f}%)")
        elif total_change > 0:
            print(f"  ⚠️  {metric_name} WORSENED (increased by {pct_total:.2f}%)")
        else:
            print(f"  ➡️  {metric_name} UNCHANGED")


def plot_metrics_progression(phase_data, output_path):
    """
    Create visualization of metrics progression across phases.

    Args:
        phase_data: List of checkpoint data dicts
        output_path: Path to save plot
    """
    phases = [d['phase'] for d in phase_data]
    epochs = [d['epoch'] for d in phase_data]

    # Extract metrics
    total_loss = [d['val_metrics'].get('total_loss') for d in phase_data]
    re_loss = [d['val_metrics'].get('re_loss') for d in phase_data]
    kl_loss = [d['val_metrics'].get('kl_loss') for d in phase_data]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Total Loss
    if all(v is not None for v in total_loss):
        axes[0].plot(phases, total_loss, 'o-', linewidth=2, markersize=8, color='tab:blue')
        axes[0].set_title('Total Validation Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Phase')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(phases)

        # Annotate values
        for p, e, v in zip(phases, epochs, total_loss):
            axes[0].annotate(f'{v:.6f}\n(ep {e})',
                           xy=(p, v),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8)

    # Plot 2: Reconstruction Loss
    if all(v is not None for v in re_loss):
        axes[1].plot(phases, re_loss, 'o-', linewidth=2, markersize=8, color='tab:orange')
        axes[1].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Phase')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(phases)

        # Annotate values
        for p, e, v in zip(phases, epochs, re_loss):
            axes[1].annotate(f'{v:.6f}\n(ep {e})',
                           xy=(p, v),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8)

    # Plot 3: KL Loss
    if all(v is not None for v in kl_loss):
        axes[2].plot(phases, kl_loss, 'o-', linewidth=2, markersize=8, color='tab:green')
        axes[2].set_title('KL Divergence', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Phase')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(phases)

        # Annotate values
        for p, e, v in zip(phases, epochs, kl_loss):
            axes[2].annotate(f'{v:.2f}\n(ep {e})',
                           xy=(p, v),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    plt.close()


def generate_text_report(phase_data, output_path):
    """
    Generate detailed text report.

    Args:
        phase_data: List of checkpoint data dicts
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CONTEXT=60 PHASE CHECKPOINT COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Context length: {cfg.context_len}\n")
        f.write(f"  Latent dimension: {cfg.latent_dim}\n")
        f.write(f"  Total epochs: {cfg.total_epochs}\n")
        f.write(f"  Batch size: {cfg.batch_size}\n")
        f.write(f"  Learning rate: {cfg.learning_rate}\n")
        f.write("\n")

        f.write("Phase Schedule:\n")
        for i, data in enumerate(phase_data, 1):
            f.write(f"  Phase {i} ({data['epoch']}): {data['phase_name']}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("VALIDATION METRICS\n")
        f.write("=" * 80 + "\n\n")

        # Detailed metrics table
        f.write(f"{'Phase':<8} {'Epoch':<8} {'Total Loss':<15} {'Recon Loss':<15} {'KL Loss':<15}\n")
        f.write("-" * 80 + "\n")

        for data in phase_data:
            phase = data['phase']
            epoch = data['epoch']
            metrics = data['val_metrics']

            total_loss = format_metric(metrics.get('total_loss'))
            recon_loss = format_metric(metrics.get('re_loss'))
            kl_loss = format_metric(metrics.get('kl_loss'))

            f.write(f"{phase:<8} {epoch:<8} {total_loss:<15} {recon_loss:<15} {kl_loss:<15}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("METRIC PROGRESSION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Analyze each metric
        for metric_key, metric_name in [
            ('total_loss', 'Total Loss'),
            ('re_loss', 'Reconstruction Loss'),
            ('kl_loss', 'KL Divergence')
        ]:
            f.write(f"{metric_name}:\n")
            values = [d['val_metrics'].get(metric_key) for d in phase_data]

            for i, (data, val) in enumerate(zip(phase_data, values)):
                if i == 0:
                    f.write(f"  Phase {data['phase']}: {format_metric(val)}\n")
                else:
                    prev_val = values[i-1]
                    if val is not None and prev_val is not None:
                        change = val - prev_val
                        pct_change = (change / prev_val) * 100 if prev_val != 0 else 0
                        change_str = f"{change:+.6f}"
                        f.write(f"  Phase {data['phase']}: {format_metric(val)} [{change_str} ({pct_change:+.2f}%)]\n")
                    else:
                        f.write(f"  Phase {data['phase']}: {format_metric(val)}\n")

            if all(v is not None for v in values):
                total_change = values[-1] - values[0]
                pct_total = (total_change / values[0]) * 100 if values[0] != 0 else 0
                f.write(f"  → Total change: {total_change:+.6f} ({pct_total:+.2f}%)\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("TRAINING PHASES SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for data in phase_data:
            f.write(f"Phase {data['phase']}: {data['phase_name']}\n")
            f.write(f"  Final epoch: {data['epoch']}\n")

            if data['phase'] == 1:
                f.write(f"  Training mode: Teacher forcing (H=1)\n")
                f.write(f"  Sequence length: {cfg.phase1_seq_len}\n")
            elif data['phase'] == 2:
                f.write(f"  Training mode: Multi-horizon {cfg.phase2_horizons}\n")
                f.write(f"  Sequence length: {cfg.phase2_seq_len}\n")
            elif data['phase'] == 3:
                f.write(f"  Training mode: Autoregressive H={cfg.phase3_horizon}\n")
                f.write(f"  Offsets: {cfg.phase3_offsets}\n")
                f.write(f"  AR steps: {cfg.phase3_ar_steps}\n")
                f.write(f"  Sequence length: {cfg.phase3_seq_len}\n")
            elif data['phase'] == 4:
                f.write(f"  Training mode: Autoregressive H={cfg.phase4_horizon}\n")
                f.write(f"  Offsets: {cfg.phase4_offsets}\n")
                f.write(f"  AR steps: {cfg.phase4_ar_steps}\n")
                f.write(f"  Sequence length: {cfg.phase4_seq_len}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"✅ Report saved to: {output_path}")


def main():
    """Main comparison function."""

    # Create output directory if needed
    output_dir = Path(cfg.checkpoint_dir)
    output_dir.mkdir(exist_ok=True)

    # Load all 4 phase checkpoints
    print("\n" + "=" * 80)
    print("LOADING PHASE CHECKPOINTS")
    print("=" * 80)

    phase_data = []
    for phase_num in [1, 2, 3, 4]:
        epoch = cfg.get_phase_end_epoch(phase_num) - 1
        checkpoint_name = cfg.get_checkpoint_name(phase_num, epoch)
        checkpoint_path = output_dir / checkpoint_name

        print(f"\nPhase {phase_num}:")
        print(f"  Loading: {checkpoint_path}")

        try:
            data = load_checkpoint(checkpoint_path)
            phase_data.append(data)
            print(f"  ✅ Loaded successfully")
            print(f"     Epoch: {data['epoch']}")
            print(f"     Phase name: {data['phase_name']}")

            # Print key metrics
            metrics = data['val_metrics']
            if metrics:
                print(f"     Val loss: {format_metric(metrics.get('total_loss'))}")
                print(f"     Recon loss: {format_metric(metrics.get('re_loss'))}")
                print(f"     KL loss: {format_metric(metrics.get('kl_loss'))}")

        except FileNotFoundError as e:
            print(f"  ❌ Error: {e}")
            print(f"  Skipping Phase {phase_num}")

    if len(phase_data) == 0:
        print("\n❌ No checkpoints found. Please run training first.")
        return

    # Print comparison table
    print_comparison_table(phase_data)

    # Analyze metric changes
    print("\n" + "=" * 80)
    print("METRIC PROGRESSION ANALYSIS")
    print("=" * 80)

    analyze_metric_changes(phase_data, 'total_loss', 'Total Validation Loss')
    analyze_metric_changes(phase_data, 're_loss', 'Reconstruction Loss')
    analyze_metric_changes(phase_data, 'kl_loss', 'KL Divergence')

    # Generate plot
    if len(phase_data) >= 2:
        plot_path = output_dir / "phase_comparison.png"
        plot_metrics_progression(phase_data, plot_path)

    # Generate text report
    report_path = output_dir / "phase_comparison.txt"
    generate_text_report(phase_data, report_path)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nAnalyzed {len(phase_data)} / 4 phase checkpoints")
    print(f"Checkpoint directory: {output_dir}")
    print(f"\nOutputs:")
    print(f"  - Visualization: {output_dir / 'phase_comparison.png'}")
    print(f"  - Text report: {output_dir / 'phase_comparison.txt'}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
