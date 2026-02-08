#!/usr/bin/env python
"""
Pre-generate test samples for MCVD validation.

Generates conditioned and unconditioned samples once, saves to .npz.
Run tests separately with:
    python test_mcvd_requirements.py --precomputed results/mcvd_paper_aligned/test_samples.npz

Usage:
    python experiments/backfill/diffusion_poc/generate_test_samples.py \
        --model_path models/backfill/mcvd_paper_aligned/best_coverage_model.pt \
        --n_samples 10 --max_batches 15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.mcvd_wrapper import MCVDModel, denormalize_iv
from experiments.backfill.diffusion_poc.config_mcvd_poc import MCVDPOCConfig, build_mcvd_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.diffusion_poc.test_mcvd_requirements import (
    load_mcvd_model,
    ARModelWrapper,
    sample_unconditioned,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-generate test samples for MCVD validation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Samples per history sequence")
    parser.add_argument("--max_batches", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path")
    parser.add_argument("--skip_unconditioned", action="store_true",
                        help="Skip unconditioned samples (faster, but no conditionality test)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load model
    print("Loading model...")
    model, poc_config, checkpoint = load_mcvd_model(args.model_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    epoch = checkpoint.get("epoch", "?")
    print(f"  Model: {args.model_path}")
    print(f"  Params: {n_params:,}, Epoch: {epoch}")

    ddim_steps = args.ddim_steps or min(100, poc_config.n_steps)
    print(f"  DDIM steps: {ddim_steps} (T={poc_config.n_steps})")

    # AR wrapper if needed
    eval_future_len = 30
    use_ar = poc_config.future_len < eval_future_len
    if use_ar:
        n_blocks = eval_future_len // poc_config.future_len
        print(f"  AR rollout: {n_blocks} blocks × {poc_config.future_len} frames = {n_blocks * poc_config.future_len} days")
        model = ARModelWrapper(model, n_blocks=n_blocks, n_inference_steps=ddim_steps)

    # Load test data
    print("\nLoading test data...")
    data = np.load(poc_config.data_path)
    surfaces = data["surface"]
    test_dataset = VolSurfaceDataset(
        surfaces, poc_config.history_len, eval_future_len,
        start_idx=poc_config.test_start,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=poc_config.batch_size,
        shuffle=False, num_workers=2,
    )
    print(f"  Dataset: {len(test_dataset)} sequences")

    # === Generate conditioned samples ===
    print(f"\nGenerating conditioned samples ({args.n_samples} per sequence, "
          f"{args.max_batches} batches)...")
    all_cond = []
    all_gt = []
    all_history = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            test_loader, desc="Conditioned", total=args.max_batches
        )):
            if batch_idx >= args.max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))
            samples = model.sample(
                history, n_samples=args.n_samples,
                sampler=args.sampler, n_inference_steps=ddim_steps,
            )
            all_cond.append(samples.cpu().numpy().astype(np.float32))
            all_gt.append(future_gt.cpu().numpy().astype(np.float32))
            all_history.append(history.cpu().numpy().astype(np.float32))

    cond_samples = np.concatenate(all_cond, axis=0)
    ground_truth = np.concatenate(all_gt, axis=0)
    history_arr = np.concatenate(all_history, axis=0)
    print(f"  Conditioned samples: {cond_samples.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    print(f"  History: {history_arr.shape}")

    # === Generate unconditioned samples ===
    uncond_samples = np.array([], dtype=np.float32)
    if not args.skip_unconditioned:
        print(f"\nGenerating unconditioned samples...")
        all_uncond = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(
                test_loader, desc="Unconditioned", total=args.max_batches
            )):
                if batch_idx >= args.max_batches:
                    break
                history = batch["history"].to(device)
                uncond = sample_unconditioned(
                    model, history, n_samples=args.n_samples,
                    sampler=args.sampler, n_inference_steps=ddim_steps,
                )
                all_uncond.append(uncond.cpu().numpy().astype(np.float32))
        uncond_samples = np.concatenate(all_uncond, axis=0)
        print(f"  Unconditioned samples: {uncond_samples.shape}")

    # === Save ===
    output_path = args.output or "results/mcvd_paper_aligned/test_samples.npz"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        cond_samples=cond_samples,
        uncond_samples=uncond_samples,
        ground_truth=ground_truth,
        history=history_arr,
    )

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    n_sequences = cond_samples.shape[0]
    print(f"\nSaved {n_sequences} sequences to {output_path} ({size_mb:.1f} MB)")
    print(f"Run tests with:")
    print(f"  python experiments/backfill/diffusion_poc/test_mcvd_requirements.py "
          f"--precomputed {output_path}")


if __name__ == "__main__":
    main()
