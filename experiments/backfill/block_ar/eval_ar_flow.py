#!/usr/bin/env python
"""
Evaluate AR Flow Matching model with the full 9-suite test battery.

Loads the ARFlowMatchingModel, generates samples, runs all test suites.
Outputs summary.json compatible with existing analysis tools.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/eval_ar_flow.py \
        --model_path models/backfill/flow_152b/best_model.pt \
        --n_samples 50 --max_batches 20 \
        --output_dir results/block_ar/152b_30d --device cuda
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig, normalize_iv, denormalize_iv,
)
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.block_ar.train_ar_flow import (
    ConditionalVelocityMLP, ARFlowMatchingModel,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_surface_validity_tests,
    run_ci_coverage_tests,
    run_conditionality_tests,
    run_time_series_tests,
    run_block_ar_tests,
    run_cointegration_tests,
    run_regime_coverage_tests,
    run_distributional_fidelity_tests,
    run_cross_cell_correlation_tests,
    print_summary,
    convert_to_serializable,
    hash_file,
)


def load_ar_flow_model(model_path, device="cuda"):
    """Load ARFlowMatchingModel from checkpoint."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    enc_cfg = ckpt["encoder_config"]

    # Rebuild encoder
    sp_cfg = {k: v for k, v in enc_cfg.items()
              if k in SinglePassConfig.__dataclass_fields__}
    sp_config = SinglePassConfig(**sp_cfg)
    base_model = SinglePassBlockAR(sp_config)
    base_model.load_state_dict(
        {"encoder." + k: v for k, v in ckpt["encoder_state_dict"].items()},
        strict=False,
    )
    encoder = base_model.encoder

    # Rebuild velocity net
    velocity_net = ConditionalVelocityMLP(
        frame_dim=cfg["frame_dim"],
        cond_dim=cfg["cond_dim"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
    )
    velocity_net.load_state_dict(ckpt["velocity_state_dict"])

    # Build model
    model = ARFlowMatchingModel(
        encoder=encoder,
        velocity_net=velocity_net,
        frame_mean=ckpt["frame_mean"],
        frame_std=ckpt["frame_std"],
        n_steps=cfg["n_steps"],
        future_len=cfg["future_len"],
    )
    model.to(device).eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Evaluate AR Flow Matching Model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--noise_sigma", type=float, default=0.0,
                        help="Post-ODE noise in standardized space (0=deterministic)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Prior noise temperature (>1 = wider ODE starts = more spread)")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("AR Flow Matching Model — Full 9-Suite Evaluation")
    print("=" * 60)

    # Load model
    model, ckpt = load_ar_flow_model(args.model_path, device)
    print(f"  Model: {args.model_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val loss: {ckpt.get('val_loss', '?'):.4f}")

    # Load test data
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]
    returns = data["ret"] if "ret" in data else None

    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0)
    print(f"  Test windows: {len(test_dataset)}")

    # Generate samples
    print("\nGenerating samples...")
    all_samples = []
    all_gt = []
    all_history = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Generating", total=args.max_batches)
        ):
            if batch_idx >= args.max_batches:
                break
            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            samples = model.sample_batched(history, n_samples=args.n_samples,
                                           noise_sigma=args.noise_sigma,
                                           temperature=args.temperature)
            history_denorm = denormalize_iv(history)

            all_samples.append(samples.cpu().numpy())
            all_gt.append(future_gt.cpu().numpy())
            all_history.append(history_denorm.cpu().numpy())

    cond_samples = np.concatenate(all_samples)
    ground_truth = np.concatenate(all_gt)
    history_arr = np.concatenate(all_history)

    print(f"  Samples: {cond_samples.shape}")
    print(f"  GT: {ground_truth.shape}")

    # Run all 9 test suites
    print("\n" + "=" * 60)
    print("FULL 9-SUITE EVALUATION")
    print("=" * 60)

    results = {}
    results['surface'] = run_surface_validity_tests(cond_samples, ground_truth)
    results['coverage'] = run_ci_coverage_tests(cond_samples, ground_truth)

    # Suite 3: conditionality — needs model for uncond generation
    # Use the model with null embedding for unconditional
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    cond_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0)
    results['conditionality'] = run_conditionality_tests(
        model, cond_loader, n_samples=args.n_samples,
        max_batches=min(args.max_batches, 15),
        max_residual=20, device=device,
    )

    results['time_series'] = run_time_series_tests(cond_samples, ground_truth)
    results['block_ar'] = run_block_ar_tests(cond_samples, block_size=config.block_size)

    if returns is not None:
        results['cointegration'] = run_cointegration_tests(
            cond_samples, ground_truth, returns=returns,
            test_start=config.test_start,
            history_len=config.history_len, future_len=config.future_len,
        )

    results['regime_coverage'] = run_regime_coverage_tests(
        cond_samples, ground_truth, history_arr,
    )
    results['distributional'] = run_distributional_fidelity_tests(
        cond_samples, ground_truth, history_arr,
    )
    results['cross_cell_correlation'] = run_cross_cell_correlation_tests(
        cond_samples, ground_truth,
    )

    # Summary
    print_summary(results)

    # Save
    results['eval_config'] = {
        'model_type': 'ARFlowMatching',
        'checkpoint_path': str(Path(args.model_path).resolve()),
        'checkpoint_hash': hash_file(args.model_path),
        'checkpoint_epoch': ckpt.get('epoch', None),
        'n_samples': args.n_samples,
        'max_batches': args.max_batches,
        'n_steps': ckpt['config']['n_steps'],
    }

    results_ser = convert_to_serializable(results)
    json_path = f"{args.output_dir}/summary.json"
    with open(json_path, 'w') as f:
        json.dump(results_ser, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
