"""
Diagnostic script: Why does the dual-path denoiser still violate spatial arb?

Analyzes:
1. Per-cell arb violation heatmap (which tenor/moneyness positions fail?)
2. Ground truth arb violation rates (what's the data baseline?)
3. BiGRU vs spatial stream contribution decomposition
4. Output projection weight analysis per output cell
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def load_model(model_path, device, no_ema=False):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ConditionalBlockARDDPM(config)

    if not no_ema and "ema_params" in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint["ema_params"]:
                param.data.copy_(checkpoint["ema_params"][name])
        print("Loaded EMA parameters")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model parameters (no EMA)")

    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', '?')})")
    return model, config


def per_cell_calendar_violations(samples_np):
    """Compute calendar arb violation rate per (tenor_pair, moneyness) cell.

    samples_np: (N_total, T_future, 5, 5) in [0,1]
    Returns: (4, 5) violation rates — 4 tenor pairs x 5 moneyness points
    """
    tenors = np.array([1, 2, 4, 8, 12])
    # Average over time dimension
    N, T, H, W = samples_np.shape
    violations = np.zeros((4, W))
    counts = 0
    for t_idx in range(T):
        surf = samples_np[:, t_idx]  # (N, 5, 5)
        total_var = surf ** 2 * tenors[:, None]  # (N, 5, 5)
        for i in range(4):
            viol = (total_var[:, i, :] > total_var[:, i + 1, :] * 1.001)  # (N, 5)
            violations[i] += viol.mean(axis=0)
            counts += 1
    violations /= T  # average over time steps
    return violations


def per_cell_butterfly_violations(samples_np):
    """Compute butterfly arb violation rate per (tenor, moneyness_triplet) cell.

    Returns: (5, 3) violation rates — 5 tenors x 3 interior moneyness triplets
    """
    N, T, H, W = samples_np.shape
    violations = np.zeros((H, W - 2))
    for t_idx in range(T):
        surf = samples_np[:, t_idx]  # (N, 5, 5)
        d2 = surf[:, :, :-2] - 2 * surf[:, :, 1:-1] + surf[:, :, 2:]  # (N, 5, 3)
        viol = (d2 < -0.005)  # (N, 5, 3)
        violations += viol.mean(axis=0)
    violations /= T
    return violations


def decompose_output_projection(model):
    """Analyze how output_proj maps BiGRU vs spatial features to each output cell.

    output_proj: Linear(656, 25) where first 256 dims = BiGRU, last 400 = spatial
    Returns per-output-cell fraction of weight magnitude from spatial stream.
    """
    W = model.denoiser.output_proj.weight.data.cpu().numpy()  # (25, 656)
    bigru_weights = W[:, :256]  # (25, 256)
    spatial_weights = W[:, 256:]  # (25, 400)

    bigru_mag = np.abs(bigru_weights).sum(axis=1)  # (25,)
    spatial_mag = np.abs(spatial_weights).sum(axis=1)  # (25,)
    total = bigru_mag + spatial_mag
    spatial_frac = spatial_mag / total  # (25,)

    return spatial_frac.reshape(5, 5), bigru_mag.reshape(5, 5), spatial_mag.reshape(5, 5)


def analyze_spatial_stream_at_noise_levels(model, val_loader, device):
    """Run spatial stream at different noise levels on clean data.

    Shows how spatial features change with noise level.
    """
    model.eval()
    batch = next(iter(val_loader))
    future = batch["future"].to(device)  # (B, 30, 5, 5) in [-1, 1]

    # Take first block
    block = future[:, :10]  # (B, 10, 5, 5)
    frames_flat = block.reshape(block.shape[0], 10, 25)

    # Test at different noise levels
    results = {}
    for t_val in [0, 10, 25, 50, 75, 99]:
        noise_levels = torch.full((block.shape[0], 10), t_val, dtype=torch.long, device=device)
        noise_emb = model.denoiser.noise_embed(noise_levels)

        # Get condition from history
        history = batch["history"].to(device)
        with torch.no_grad():
            condition = model.encoder(history, mask=None)
            spatial_out = model.denoiser.spatial_stream(frames_flat, noise_emb, condition)

        results[t_val] = {
            "spatial_norm": spatial_out.norm(dim=-1).mean().item(),
            "spatial_std": spatial_out.std(dim=-1).mean().item(),
        }

    return results


def run_arb_on_decomposed_predictions(model, val_loader, device, n_batches=10):
    """Generate samples and decompose the noise prediction into BiGRU vs spatial.

    For each denoising step, measure how much each path's contribution
    would violate arb if it were the sole predictor.
    """
    model.eval()
    batch = next(iter(val_loader))
    history = batch["history"].to(device)
    future_gt = batch["future"].to(device)

    B = history.shape[0]
    bs = model.config.block_size

    with torch.no_grad():
        condition = model.encoder(history, mask=None)

        # Use clean future block as input (noise level 0 = nearly clean)
        block = future_gt[:, :bs]  # (B, 10, 5, 5) in [-1, 1]
        frames_flat = block.reshape(B, bs, 25)
        positions = torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)

        # Test at low noise (t=5) where spatial structure should matter most
        noise_levels = torch.full((B, bs), 5, dtype=torch.long, device=device)

        # Forward through both paths separately
        pos_emb = model.denoiser.pos_embed(positions)
        noise_emb = model.denoiser.noise_embed(noise_levels)

        # Spatial path
        spatial_out = model.denoiser.spatial_stream(frames_flat, noise_emb, condition)

        # BiGRU path
        x = torch.cat([frames_flat, pos_emb, noise_emb], dim=-1)
        x = model.denoiser.input_proj(x)
        x = model.denoiser.film_pre(x, condition)
        x, _ = model.denoiser.bigru(x)
        x = model.denoiser.film_post(x, condition)
        bigru_out = x

        # Get output projection weights
        W = model.denoiser.output_proj.weight  # (25, 656)
        b = model.denoiser.output_proj.bias    # (25,)

        W_bigru = W[:, :256]   # (25, 256)
        W_spatial = W[:, 256:]  # (25, 400)

        # Compute each path's contribution to the output
        # noise_pred = W @ [bigru; spatial] + b
        # = W_bigru @ bigru + W_spatial @ spatial + b
        bigru_contrib = torch.einsum('oi,bti->bto', W_bigru, bigru_out)  # (B, T, 25)
        spatial_contrib = torch.einsum('oi,bti->bto', W_spatial, spatial_out)  # (B, T, 25)

        # The full prediction
        full_pred = bigru_contrib + spatial_contrib + b

        # Reshape contributions to 5x5
        bigru_surf = bigru_contrib.reshape(B, bs, 5, 5)
        spatial_surf = spatial_contrib.reshape(B, bs, 5, 5)
        full_surf = full_pred.reshape(B, bs, 5, 5)

    return {
        "bigru_contrib": bigru_surf.cpu().numpy(),
        "spatial_contrib": spatial_surf.cpu().numpy(),
        "full_pred": full_surf.cpu().numpy(),
        "bigru_norm": bigru_contrib.norm(dim=-1).mean().item(),
        "spatial_norm": spatial_contrib.norm(dim=-1).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/block_ar_dual_path/best_coverage_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("SPATIAL ARB DIAGNOSTIC")
    print("=" * 70)

    # Load model
    model, config = load_model(args.model_path, args.device, args.no_ema)

    # Load data
    poc_config = get_default_config()
    data = np.load(poc_config.data_path)
    surfaces = data["surface"]

    val_dataset = VolSurfaceDataset(
        surfaces, poc_config.history_len, poc_config.future_len,
        start_idx=poc_config.val_start, end_idx=poc_config.val_end,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # =========================================================================
    # 1. Ground truth arb violation rates
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. GROUND TRUTH ARB VIOLATIONS")
    print("=" * 70)

    gt_surfaces = []
    for batch in val_loader:
        gt_surfaces.append(batch["future"].numpy())
    gt_surfaces = np.concatenate(gt_surfaces)  # (N, 30, 5, 5) in [-1,1]
    gt_surfaces = (gt_surfaces + 1.0) / 2.0  # denormalize to [0,1]

    gt_cal = per_cell_calendar_violations(gt_surfaces)
    gt_but = per_cell_butterfly_violations(gt_surfaces)

    print("\nCalendar arb violations in GROUND TRUTH (per tenor-pair x moneyness):")
    print("  Tenor pairs: (1→2), (2→4), (4→8), (8→12)")
    print("  Moneyness:   ITM    ITM-   ATM    OTM-   OTM")
    for i in range(4):
        row = " ".join(f"{v:6.1%}" for v in gt_cal[i])
        print(f"  Pair {i}: {row}")
    print(f"  Overall: {gt_cal.mean():.1%}")

    print("\nButterfly arb violations in GROUND TRUTH (per tenor x moneyness-triplet):")
    for i in range(5):
        row = " ".join(f"{v:6.1%}" for v in gt_but[i])
        print(f"  Tenor {i}: {row}")
    print(f"  Overall: {gt_but.mean():.1%}")

    # =========================================================================
    # 2. Generated samples arb per cell
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. GENERATED SAMPLES — PER-CELL ARB VIOLATIONS")
    print("=" * 70)

    all_gen_surfaces = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.max_batches:
                break
            history = batch["history"].to(args.device)
            samples = model.sample_batched(history, n_samples=args.n_samples)
            # (B, n_samples, 30, 5, 5) in [0,1]
            B, S, T, H, W = samples.shape
            all_gen_surfaces.append(samples.reshape(B * S, T, H, W).cpu().numpy())
            print(f"  Batch {batch_idx+1}/{args.max_batches} done ({B*S} samples)")

    gen_surfaces = np.concatenate(all_gen_surfaces)
    print(f"\nTotal generated samples: {len(gen_surfaces)}")

    gen_cal = per_cell_calendar_violations(gen_surfaces)
    gen_but = per_cell_butterfly_violations(gen_surfaces)

    print("\nCalendar arb violations in GENERATED (per tenor-pair x moneyness):")
    print("  Tenor pairs: (1→2), (2→4), (4→8), (8→12)")
    print("  Moneyness:   ITM    ITM-   ATM    OTM-   OTM")
    for i in range(4):
        row = " ".join(f"{v:6.1%}" for v in gen_cal[i])
        print(f"  Pair {i}: {row}")
    print(f"  Overall: {gen_cal.mean():.1%}")

    print("\nButterfly arb violations in GENERATED (per tenor x moneyness-triplet):")
    for i in range(5):
        row = " ".join(f"{v:6.1%}" for v in gen_but[i])
        print(f"  Tenor {i}: {row}")
    print(f"  Overall: {gen_but.mean():.1%}")

    # Excess violations (generated - ground truth)
    print("\n--- EXCESS violations (generated minus ground truth) ---")
    excess_cal = gen_cal - gt_cal
    print("\nExcess calendar arb (positive = model adds violations):")
    for i in range(4):
        row = " ".join(f"{v:+6.1%}" for v in excess_cal[i])
        print(f"  Pair {i}: {row}")
    print(f"  Overall excess: {excess_cal.mean():+.1%}")

    excess_but = gen_but - gt_but
    print("\nExcess butterfly arb:")
    for i in range(5):
        row = " ".join(f"{v:+6.1%}" for v in excess_but[i])
        print(f"  Tenor {i}: {row}")
    print(f"  Overall excess: {excess_but.mean():+.1%}")

    # =========================================================================
    # 3. Output projection decomposition
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. OUTPUT PROJECTION — SPATIAL FRACTION PER CELL")
    print("=" * 70)

    spatial_frac, bigru_mag, spatial_mag = decompose_output_projection(model)
    print("\nSpatial stream weight fraction per output cell (5x5 grid):")
    print("  (higher = spatial stream has more influence on that cell)")
    for i in range(5):
        row = " ".join(f"{v:5.1%}" for v in spatial_frac[i])
        print(f"  Row {i}: {row}")
    print(f"  Mean: {spatial_frac.mean():.1%}")

    # =========================================================================
    # 4. Spatial stream behavior at different noise levels
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. SPATIAL STREAM OUTPUT AT DIFFERENT NOISE LEVELS")
    print("=" * 70)

    noise_results = analyze_spatial_stream_at_noise_levels(model, val_loader, args.device)
    for t_val, metrics in sorted(noise_results.items()):
        print(f"  t={t_val:3d}: norm={metrics['spatial_norm']:.4f}, std={metrics['spatial_std']:.4f}")

    # =========================================================================
    # 5. BiGRU vs Spatial contribution decomposition
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. BIGRU vs SPATIAL CONTRIBUTION AT LOW NOISE (t=5)")
    print("=" * 70)

    decomp = run_arb_on_decomposed_predictions(model, val_loader, args.device)
    print(f"\n  BiGRU contribution norm:   {decomp['bigru_norm']:.4f}")
    print(f"  Spatial contribution norm: {decomp['spatial_norm']:.4f}")
    ratio = decomp['spatial_norm'] / (decomp['bigru_norm'] + decomp['spatial_norm'])
    print(f"  Spatial fraction of output: {ratio:.1%}")

    # Per-cell mean contribution magnitude
    bigru_cell = np.abs(decomp['bigru_contrib']).mean(axis=(0, 1))  # (5,5)
    spatial_cell = np.abs(decomp['spatial_contrib']).mean(axis=(0, 1))  # (5,5)
    total_cell = bigru_cell + spatial_cell
    spatial_cell_frac = spatial_cell / total_cell

    print("\n  Spatial activation fraction per cell (at t=5):")
    for i in range(5):
        row = " ".join(f"{v:5.1%}" for v in spatial_cell_frac[i])
        print(f"    Row {i}: {row}")

    # =========================================================================
    # 6. Cross-analysis: where arb fails vs where spatial influence is weak
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. CROSS-ANALYSIS: ARB VIOLATIONS vs SPATIAL INFLUENCE")
    print("=" * 70)

    # For calendar: correlate excess violations with spatial fraction
    # Calendar violations are per (pair, moneyness) = (4, 5)
    # Spatial fraction is per (tenor, moneyness) = (5, 5)
    # Map pair violations to the LOWER tenor (where violation originates)
    print("\n  Calendar: comparing spatial weight fraction at lower tenor")
    print("  vs excess violation rate for each pair:")
    for i in range(4):
        for j in range(5):
            sf = spatial_frac[i, j]
            ev = excess_cal[i, j]
            marker = " ***" if ev > 0.05 else ""
            print(f"    Pair({i},{i+1}) Money={j}: spatial={sf:.1%}, excess_viol={ev:+.1%}{marker}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
