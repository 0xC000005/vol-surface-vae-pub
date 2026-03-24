#!/usr/bin/env python
"""
152f Training History Reconstruction

The original training_history.json was corrupted by numpy float32
serialization error (crashed at epoch 100 during json.dump).

This script parses the training log (flow_152f_training.log) to
reconstruct the complete training history.

Log format:
  Ep   1  train=0.3435  val=0.3091  (8.5s)  eff_rank=2.77(GT=7.61)  PC1=0.949  ...
  Ep   2  train=0.2260  val=0.2042  (8.3s)
"""
import json
import re
from pathlib import Path

LOG_PATH = Path("/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_152f_training.log")
OUTPUT_PATH = Path("/home/max/Documents/vol-surface-vae-pub/models/backfill/flow_152f/training_history.json")
RESULT_PATH = Path("/home/max/Documents/vol-surface-vae-pub/results/validations/2026-03-24/verification_results/152f_history_fix.json")


def parse_training_log(log_path):
    """Parse 152f training log and extract evaluation checkpoint data.

    The training_history.json only contains rows from evaluation epochs
    (every 20 epochs + epoch 1), matching the original script's behavior.
    """
    with open(log_path) as f:
        lines = f.readlines()

    all_epochs = []
    eval_epochs = []

    # Pattern for basic epoch line: Ep  XX  train=0.XXXX  val=0.XXXX  (X.Xs)
    basic_re = re.compile(
        r'Ep\s+(\d+)\s+train=([\d.]+)\s+val=([\d.]+)\s+\(([\d.]+)s\)'
    )

    # Pattern for evaluation metrics (appended on same line)
    metrics_re = re.compile(
        r'eff_rank=([\d.]+)\(GT=([\d.]+)\)\s+'
        r'PC1=([\d.]+)\s+PC2=([\d.]+)\s+frob=([\d.]+)\s+'
        r'KS=(\d+)/25\s+kurt=([\d.]+)\s+'
        r'spread h1=([\d.]+) h30=([\d.]+)\s*'
        r'vel_var_t05=([\d.]+)'
    )

    for line in lines:
        line = line.strip()
        basic_match = basic_re.search(line)
        if not basic_match:
            continue

        epoch = int(basic_match.group(1))
        train_loss = float(basic_match.group(2))
        val_loss = float(basic_match.group(3))
        elapsed = float(basic_match.group(4))

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        # Check if this line also has evaluation metrics
        metrics_match = metrics_re.search(line)
        if metrics_match:
            epoch_data.update({
                "vel_var_t05": float(metrics_match.group(10)),
                "eff_rank": float(metrics_match.group(1)),
                "gt_eff_rank": float(metrics_match.group(2)),
                "pc1": float(metrics_match.group(3)),
                "pc2": float(metrics_match.group(4)),
                "frob": float(metrics_match.group(5)),
                "ks_pass": int(metrics_match.group(6)),
                "kurt_ratio": float(metrics_match.group(7)),
                "spread_h1": float(metrics_match.group(8)),
                "spread_h30": float(metrics_match.group(9)),
            })
            eval_epochs.append(epoch_data)

        all_epochs.append(epoch_data)

    return all_epochs, eval_epochs


def main():
    print("=" * 60)
    print("152f Training History Reconstruction")
    print("=" * 60)

    if not LOG_PATH.exists():
        print(f"ERROR: Log file not found at {LOG_PATH}")
        return

    all_epochs, eval_epochs = parse_training_log(LOG_PATH)

    print(f"  Total epochs parsed: {len(all_epochs)}")
    print(f"  Evaluation epochs: {len(eval_epochs)}")
    print(f"  Eval epoch numbers: {[e['epoch'] for e in eval_epochs]}")

    # The original training_history.json only contains eval epoch rows
    # (the script appends to history[] only inside the eval block)
    history = eval_epochs

    # Print summary of eval epochs
    print("\n  Evaluation Checkpoints:")
    for e in history:
        print(f"    Ep {e['epoch']:3d}: val={e['val_loss']:.4f}  "
              f"eff_rank={e['eff_rank']:.2f}  PC1={e['pc1']:.3f}  "
              f"KS={e['ks_pass']}/25  kurt={e['kurt_ratio']:.3f}")

    # Save reconstructed history
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  Saved reconstructed history to {OUTPUT_PATH}")
    print(f"  Entries: {len(history)}")

    # Verify by re-reading
    with open(OUTPUT_PATH) as f:
        verify = json.load(f)
    assert len(verify) == len(history), "Verification failed: length mismatch"
    print(f"  Verification: JSON re-read OK ({len(verify)} entries)")

    # Save verification result
    verification = {
        "task": "152f_training_history_fix",
        "status": "COMPLETE",
        "log_file": str(LOG_PATH),
        "output_file": str(OUTPUT_PATH),
        "total_epochs_in_log": len(all_epochs),
        "eval_checkpoints": len(eval_epochs),
        "eval_epoch_numbers": [e["epoch"] for e in eval_epochs],
        "corruption_cause": "numpy float32 not JSON serializable (line 235 of train_oneshot_flow_152f.py)",
        "corruption_details": (
            "Original script crashed at epoch 100 final save. The json.dump "
            "received a dict with numpy.float32 values and failed with "
            "TypeError: Object of type float32 is not JSON serializable. "
            "Only first partial entry was written before crash."
        ),
        "reconstruction_method": "Parsed training log (flow_152f_training.log) with regex",
        "final_metrics": eval_epochs[-1] if eval_epochs else {},
        "all_epochs_summary": {
            "first_epoch": all_epochs[0] if all_epochs else {},
            "last_epoch": all_epochs[-1] if all_epochs else {},
            "best_val_loss": min(e["val_loss"] for e in all_epochs) if all_epochs else None,
            "best_val_epoch": min(all_epochs, key=lambda e: e["val_loss"])["epoch"] if all_epochs else None,
        },
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(verification, f, indent=2)
    print(f"  Verification result saved to {RESULT_PATH}")

    return verification


if __name__ == "__main__":
    main()
