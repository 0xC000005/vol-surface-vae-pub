"""Diagnose skewness gap between Block-AR and DDPM POC.

Ablations:
1. Block boundary exclusion: compute skewness with/without boundary frames
2. Per-horizon skewness decomposition
3. Per-cell skewness decomposition
4. Multiple samples (not just sample 0)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_skewness.py \
        --model_path models/backfill/block_ar_taskprob_B/best_coverage_model.pt \
        --no_ema --max_batches 20 \
        --output_dir results/skewness_diagnosis
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from diffusion.block_ar.block_ar_ddpm import ConditionalBlockARDDPM, BlockARConfig, denormalize_iv
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.block_ar.train_block_ar import VolSurfaceDataset


def generate_samples(model, test_loader, n_samples, max_batches, device):
    """Generate samples and return (samples, ground_truth) arrays."""
    all_samples = []
    all_gt = []

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break

        history = batch["history"].to(device)
        future_gt = denormalize_iv(batch["future"].to(device))

        with torch.no_grad():
            samples = model.sample_batched(
                history, n_samples=n_samples,
                max_global_residual=0,
            )

        all_samples.append(samples.cpu().numpy())
        all_gt.append(future_gt.cpu().numpy())

        if (batch_idx + 1) % 5 == 0:
            print(f"  Generated {batch_idx + 1}/{max_batches} batches")

    return np.concatenate(all_samples), np.concatenate(all_gt)


def compute_skewness_decomposition(samples, gt, block_size=10):
    """Compute skewness with multiple decompositions.

    Args:
        samples: (N, n_samples, T, 5, 5)
        gt: (N, T, 5, 5)
        block_size: frames per block

    Returns dict with all decomposition results.
    """
    N, n_samp, T, H, W = samples.shape
    results = {}

    # --- 1. Standard (matches test script): sample 0, all transitions ---
    gen_diff_s0 = np.diff(samples[:, 0], axis=1)  # (N, T-1, 5, 5)
    gt_diff = np.diff(gt, axis=1)

    results['standard'] = {
        'gen_skewness': float(skew(gen_diff_s0.flatten())),
        'gt_skewness': float(skew(gt_diff.flatten())),
        'gen_kurtosis': float(kurtosis(gen_diff_s0.flatten(), fisher=True)),
        'gt_kurtosis': float(kurtosis(gt_diff.flatten(), fisher=True)),
        'n_changes': int(gen_diff_s0.size),
    }

    # --- 2. Block boundary exclusion ---
    boundary_frames = set()
    for b in range(1, T // block_size):
        boundary_frames.add(b * block_size - 1)  # frame index in diff (0-indexed)

    intra_mask = np.ones(T - 1, dtype=bool)
    for bf in boundary_frames:
        intra_mask[bf] = False

    gen_intra = gen_diff_s0[:, intra_mask]  # exclude boundary transitions
    gt_intra = gt_diff[:, intra_mask]
    gen_boundary = gen_diff_s0[:, ~intra_mask]  # only boundary transitions
    gt_boundary = gt_diff[:, ~intra_mask]

    results['intra_block_only'] = {
        'gen_skewness': float(skew(gen_intra.flatten())),
        'gt_skewness': float(skew(gt_intra.flatten())),
        'gen_kurtosis': float(kurtosis(gen_intra.flatten(), fisher=True)),
        'n_changes': int(gen_intra.size),
        'excluded_frames': sorted(boundary_frames),
    }

    results['boundary_only'] = {
        'gen_skewness': float(skew(gen_boundary.flatten())),
        'gt_skewness': float(skew(gt_boundary.flatten())),
        'gen_kurtosis': float(kurtosis(gen_boundary.flatten(), fisher=True)),
        'n_changes': int(gen_boundary.size),
        'boundary_frames': sorted(boundary_frames),
    }

    # --- 3. Per-horizon skewness ---
    per_horizon = {}
    for h in range(T - 1):
        gen_h = gen_diff_s0[:, h].flatten()
        gt_h = gt_diff[:, h].flatten()
        per_horizon[str(h)] = {
            'gen_skewness': float(skew(gen_h)),
            'gt_skewness': float(skew(gt_h)),
            'is_boundary': h in boundary_frames,
        }
    results['per_horizon'] = per_horizon

    # --- 4. Per-cell skewness (ATM vs wings) ---
    cells = {
        'atm': (2, 2),
        'otm_put': (0, 0),
        'otm_call': (4, 0),
        'itm_put': (0, 4),
        'itm_call': (4, 4),
    }
    per_cell = {}
    for name, (m, t) in cells.items():
        gen_c = gen_diff_s0[:, :, m, t].flatten()
        gt_c = gt_diff[:, :, m, t].flatten()
        per_cell[name] = {
            'gen_skewness': float(skew(gen_c)),
            'gt_skewness': float(skew(gt_c)),
            'gen_kurtosis': float(kurtosis(gen_c, fisher=True)),
        }
    results['per_cell'] = per_cell

    # --- 5. Multi-sample skewness (average across samples, not just sample 0) ---
    multi_sample_skew = []
    for s in range(min(n_samp, 10)):
        gen_diff_s = np.diff(samples[:, s], axis=1).flatten()
        multi_sample_skew.append(float(skew(gen_diff_s)))

    results['multi_sample'] = {
        'per_sample_skewness': multi_sample_skew,
        'mean_skewness': float(np.mean(multi_sample_skew)),
        'std_skewness': float(np.std(multi_sample_skew)),
        'n_samples_tested': len(multi_sample_skew),
    }

    # --- 6. Pooled across all samples (not just sample 0) ---
    all_diffs = np.diff(samples[:, :min(n_samp, 10)], axis=2)  # (N, S, T-1, 5, 5)
    results['all_samples_pooled'] = {
        'gen_skewness': float(skew(all_diffs.flatten())),
        'gen_kurtosis': float(kurtosis(all_diffs.flatten(), fisher=True)),
        'n_changes': int(all_diffs.size),
    }

    # --- 7. Block-specific skewness (block 0, 1, 2 separately) ---
    n_blocks = T // block_size
    per_block = {}
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size
        # Diff within block (excluding first frame which connects to prev block)
        if b == 0:
            block_diff = gen_diff_s0[:, start:end-1]
        else:
            # Include the boundary transition as first element
            block_diff_with_boundary = gen_diff_s0[:, start-1:end-1]
            block_diff_no_boundary = gen_diff_s0[:, start:end-1]
            per_block[f'block_{b}_with_boundary'] = {
                'gen_skewness': float(skew(block_diff_with_boundary.flatten())),
                'n_changes': int(block_diff_with_boundary.size),
            }
            block_diff = block_diff_no_boundary

        per_block[f'block_{b}'] = {
            'gen_skewness': float(skew(block_diff.flatten())),
            'n_changes': int(block_diff.size),
        }
    results['per_block'] = per_block

    return results


def main():
    parser = argparse.ArgumentParser(description="Skewness gap diagnosis")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/skewness_diagnosis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SKEWNESS GAP DIAGNOSIS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"N samples: {args.n_samples}")
    print(f"Max batches: {args.max_batches}")

    # Load model
    checkpoint = torch.load(args.model_path, weights_only=False, map_location=args.device)
    model_config = BlockARConfig(**checkpoint["config"])
    model = ConditionalBlockARDDPM(model_config)

    if args.no_ema or "ema_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Loaded regular weights")
    else:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("  Loaded EMA weights")

    model = model.to(args.device).eval()
    epoch = checkpoint.get("epoch", "?")
    print(f"  Epoch: {epoch}")
    print(f"  Block size: {model_config.block_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]
    dataset = VolSurfaceDataset(
        surfaces,
        config.history_len, config.future_len,
        start_idx=config.test_start,
    )
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    print(f"  Test set: {len(dataset)} windows")

    # Generate samples
    print("\nGenerating samples...")
    samples, gt = generate_samples(
        model, test_loader, args.n_samples, args.max_batches, args.device
    )
    print(f"  Samples: {samples.shape}")
    print(f"  Ground truth: {gt.shape}")

    # Run decomposition
    print("\nComputing skewness decomposition...")
    results = compute_skewness_decomposition(samples, gt, model_config.block_size)

    # Add eval metadata
    results['eval_config'] = {
        'model_path': args.model_path,
        'epoch': epoch,
        'n_samples': args.n_samples,
        'max_batches': args.max_batches,
        'block_size': model_config.block_size,
        'no_ema': args.no_ema,
        'device': args.device,
        'model_config': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                         for k, v in vars(model_config).items()},
    }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    std = results['standard']
    print(f"\n1. Standard (sample 0, all transitions):")
    print(f"   Gen skewness: {std['gen_skewness']:.4f}")
    print(f"   GT  skewness: {std['gt_skewness']:.4f}")
    print(f"   Ratio: {std['gen_skewness']/std['gt_skewness']:.3f}")

    intra = results['intra_block_only']
    print(f"\n2. Intra-block only (excluding boundary transitions at frames {intra['excluded_frames']}):")
    print(f"   Gen skewness: {intra['gen_skewness']:.4f}")
    print(f"   GT  skewness: {intra['gt_skewness']:.4f}")
    print(f"   Ratio: {intra['gen_skewness']/intra['gt_skewness']:.3f}")

    bnd = results['boundary_only']
    print(f"\n3. Boundary transitions only (frames {bnd['boundary_frames']}):")
    print(f"   Gen skewness: {bnd['gen_skewness']:.4f}")
    print(f"   GT  skewness: {bnd['gt_skewness']:.4f}")

    print(f"\n4. Per-horizon skewness (gen, boundary marked with *):")
    for h in range(min(29, len(results['per_horizon']))):
        ph = results['per_horizon'][str(h)]
        marker = " *" if ph['is_boundary'] else ""
        print(f"   h={h:2d}: gen={ph['gen_skewness']:+.4f}  gt={ph['gt_skewness']:+.4f}{marker}")

    print(f"\n5. Per-cell skewness:")
    for name, vals in results['per_cell'].items():
        ratio = vals['gen_skewness'] / vals['gt_skewness'] if vals['gt_skewness'] != 0 else float('inf')
        print(f"   {name:10s}: gen={vals['gen_skewness']:+.4f}  gt={vals['gt_skewness']:+.4f}  ratio={ratio:.3f}")

    ms = results['multi_sample']
    print(f"\n6. Multi-sample skewness ({ms['n_samples_tested']} samples):")
    print(f"   Mean: {ms['mean_skewness']:.4f} ± {ms['std_skewness']:.4f}")
    print(f"   Per-sample: {[f'{s:.3f}' for s in ms['per_sample_skewness']]}")

    ap = results['all_samples_pooled']
    print(f"\n7. All samples pooled:")
    print(f"   Gen skewness: {ap['gen_skewness']:.4f}")

    print(f"\n8. Per-block skewness:")
    for name, vals in sorted(results['per_block'].items()):
        print(f"   {name}: gen={vals['gen_skewness']:+.4f}")

    # Delta: boundary effect
    delta = intra['gen_skewness'] - std['gen_skewness']
    print(f"\n{'='*60}")
    print(f"BOUNDARY EFFECT: {delta:+.4f} (intra-block minus standard)")
    if abs(delta) > 0.02:
        print(f"  Block boundaries {'decrease' if delta > 0 else 'increase'} pooled skewness by {abs(delta):.4f}")
    else:
        print(f"  Block boundaries have NEGLIGIBLE effect on skewness (|delta| < 0.02)")
    print(f"{'='*60}")

    # Save
    output_path = os.path.join(args.output_dir, "skewness_diagnosis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
