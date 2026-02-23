"""Diagnose skewness in DDPM POC for comparison with Block-AR.

Uses same decomposition as block_ar/diagnose_skewness.py.

Usage:
    PYTHONPATH=. python experiments/backfill/diffusion_poc/diagnose_skewness_ddpm.py \
        --model_path models/backfill/ddpm_poc/baseline_uniform_epoch_50.pt \
        --sampler ddpm --n_steps 100 --max_batches 20 \
        --output_dir results/skewness_diagnosis/ddpm_poc_ddpm100
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig, denormalize_iv
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def generate_samples(model, test_loader, n_samples, max_batches, device, sampler, n_steps):
    """Generate samples and return (samples, ground_truth) arrays."""
    all_samples = []
    all_gt = []

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break

        history = batch["history"].to(device)
        future_gt = denormalize_iv(batch["future"].to(device))

        with torch.no_grad():
            samples = model.sample(
                history, n_samples=n_samples,
                sampler=sampler, n_inference_steps=n_steps,
            )

        all_samples.append(samples.cpu().numpy())
        all_gt.append(future_gt.cpu().numpy())

        if (batch_idx + 1) % 5 == 0:
            print(f"  Generated {batch_idx + 1}/{max_batches} batches")

    return np.concatenate(all_samples), np.concatenate(all_gt)


def compute_skewness_decomposition(samples, gt):
    """Same decomposition as Block-AR version, but without block structure."""
    N, n_samp, T, H, W = samples.shape
    results = {}

    # --- 1. Standard: sample 0 ---
    gen_diff_s0 = np.diff(samples[:, 0], axis=1)
    gt_diff = np.diff(gt, axis=1)

    results['standard'] = {
        'gen_skewness': float(skew(gen_diff_s0.flatten())),
        'gt_skewness': float(skew(gt_diff.flatten())),
        'gen_kurtosis': float(kurtosis(gen_diff_s0.flatten(), fisher=True)),
        'gt_kurtosis': float(kurtosis(gt_diff.flatten(), fisher=True)),
        'n_changes': int(gen_diff_s0.size),
    }

    # --- 2. Per-horizon skewness ---
    per_horizon = {}
    for h in range(T - 1):
        gen_h = gen_diff_s0[:, h].flatten()
        gt_h = gt_diff[:, h].flatten()
        per_horizon[str(h)] = {
            'gen_skewness': float(skew(gen_h)),
            'gt_skewness': float(skew(gt_h)),
        }
    results['per_horizon'] = per_horizon

    # --- 3. Per-cell skewness ---
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

    # --- 4. Multi-sample ---
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

    # --- 5. Pooled ---
    all_diffs = np.diff(samples[:, :min(n_samp, 10)], axis=2)
    results['all_samples_pooled'] = {
        'gen_skewness': float(skew(all_diffs.flatten())),
        'gen_kurtosis': float(kurtosis(all_diffs.flatten(), fisher=True)),
        'n_changes': int(all_diffs.size),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="DDPM POC skewness diagnosis")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/skewness_diagnosis/ddpm_poc")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DDPM POC SKEWNESS DIAGNOSIS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Sampler: {args.sampler}, steps: {args.n_steps}")

    # Load model
    import dataclasses
    checkpoint = torch.load(args.model_path, weights_only=False, map_location=args.device)
    cfg = checkpoint["config"]
    if dataclasses.is_dataclass(cfg):
        model_config = cfg
    else:
        model_config = DenoiserConfig(**cfg)
    model = ConditionalDDPM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device).eval()
    epoch = checkpoint.get("epoch", "?")
    print(f"  Epoch: {epoch}")
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

    # Generate
    print("\nGenerating samples...")
    samples, gt = generate_samples(
        model, test_loader, args.n_samples, args.max_batches,
        args.device, args.sampler, args.n_steps
    )
    print(f"  Samples: {samples.shape}")
    print(f"  Ground truth: {gt.shape}")

    # Decomposition
    print("\nComputing skewness decomposition...")
    results = compute_skewness_decomposition(samples, gt)

    results['eval_config'] = {
        'model_path': args.model_path,
        'epoch': epoch,
        'sampler': args.sampler,
        'n_steps': args.n_steps,
        'n_samples': args.n_samples,
        'max_batches': args.max_batches,
        'device': args.device,
        'model_config': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                         for k, v in vars(model_config).items()},
    }

    # Print
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    std = results['standard']
    print(f"\n1. Standard (sample 0):")
    print(f"   Gen skewness: {std['gen_skewness']:.4f}")
    print(f"   GT  skewness: {std['gt_skewness']:.4f}")
    gt_s = std['gt_skewness']
    print(f"   Ratio: {std['gen_skewness']/gt_s:.3f}" if gt_s != 0 else "   Ratio: inf")

    print(f"\n2. Per-horizon skewness (gen):")
    for h in range(min(29, len(results['per_horizon']))):
        ph = results['per_horizon'][str(h)]
        print(f"   h={h:2d}: gen={ph['gen_skewness']:+.4f}  gt={ph['gt_skewness']:+.4f}")

    print(f"\n3. Per-cell skewness:")
    for name, vals in results['per_cell'].items():
        ratio = vals['gen_skewness'] / vals['gt_skewness'] if vals['gt_skewness'] != 0 else float('inf')
        print(f"   {name:10s}: gen={vals['gen_skewness']:+.4f}  gt={vals['gt_skewness']:+.4f}  ratio={ratio:.3f}")

    ms = results['multi_sample']
    print(f"\n4. Multi-sample skewness ({ms['n_samples_tested']} samples):")
    print(f"   Mean: {ms['mean_skewness']:.4f} ± {ms['std_skewness']:.4f}")

    ap = results['all_samples_pooled']
    print(f"\n5. All samples pooled:")
    print(f"   Gen skewness: {ap['gen_skewness']:.4f}")

    # Save
    output_path = os.path.join(args.output_dir, "skewness_diagnosis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
