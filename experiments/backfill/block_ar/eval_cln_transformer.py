#!/usr/bin/env python
"""
Comprehensive evaluation for CLN Residual Transformer models (155d series).

Covers the key metrics from the 9-suite test framework:
- S1: Surface validity (explosion rate)
- S2: CI coverage (per-horizon, per-cell, worst_cell)
- S4: Time series (kurtosis, ACF)
- S5: Growing uncertainty (monotonic spread increase)
- S8: Distributional (KS daily changes, KS levels)
- S9: Cross-cell correlation (corr ratio, eff rank)

Also: per-cell CI breakdown, spread-skill ratio, turb/calm conditionality.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/eval_cln_transformer.py \
        --model_path models/backfill/flow_155d/final_model.pt \
        --base_model models/backfill/flow_153a/final_model.pt \
        --n_samples 50 --output_dir results/block_ar/155d_eval --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, kurtosis

import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    load_encoder, normalize_iv, make_serializable
)
from experiments.backfill.block_ar.train_155d_cln_transformer import (
    CLNResidualTransformer
)


def load_cln_model(model_path, device):
    """Load CLN transformer model (supports 155d and 159a variants)."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt['config']
    model_type = cfg.get('type', 'cln_residual_transformer')

    if model_type == 'no_ln_cln_residual_transformer':
        from experiments.backfill.block_ar.train_159a_no_ln_cln import (
            NoLNCLNResidualTransformer
        )
        model = NoLNCLNResidualTransformer(
            n_frames=cfg['n_frames'], n_cells=cfg['n_cells'],
            d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            cond_dim=cfg['cond_dim'], noise_dim=cfg['noise_dim'],
        )
    else:
        model = CLNResidualTransformer(
            n_frames=cfg['n_frames'], n_cells=cfg['n_cells'],
            d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            cond_dim=cfg['cond_dim'], noise_dim=cfg['noise_dim'],
        )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, cfg, ckpt


def generate_samples(model, encoder, base_preds, surfaces, start_idx, n_windows,
                     n_samples, device, H=30):
    """Generate samples for evaluation windows."""
    all_samples = []
    all_gt = []
    N = min(n_windows, len(base_preds))
    T, C = 30, 25

    with torch.no_grad():
        for i in range(N):
            # Encode history
            hist = torch.from_numpy(
                surfaces[start_idx + i:start_idx + i + H][None].astype(np.float32)
            ).to(device)
            cond = encoder(normalize_iv(hist))  # (1, 128)
            cond_K = cond.expand(n_samples, -1)  # (K, 128)
            noise = torch.randn(n_samples, model.noise_dim, device=device)

            # Generate residuals + combine with base
            residual = model(cond_K, noise).cpu().numpy()  # (K, 750)
            combined = np.clip(base_preds[i] + residual, 0, 1)
            all_samples.append(combined.reshape(n_samples, T, C))

            gt = surfaces[start_idx + i + H:start_idx + i + H + T].reshape(T, C)
            all_gt.append(gt)

            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{N} windows")

    return np.array(all_samples), np.array(all_gt)


def evaluate_full(samples, gt, rets=None):
    """Run comprehensive evaluation."""
    N, K, T, C = samples.shape
    results = {}

    # === S2: CI Coverage ===
    # Per-horizon CI
    ci_per_horizon = []
    for h in range(T):
        lo = np.percentile(samples[:, :, h], 5, axis=1)
        hi = np.percentile(samples[:, :, h], 95, axis=1)
        cov = ((gt[:, h] >= lo) & (gt[:, h] <= hi)).mean()
        ci_per_horizon.append(float(cov))
    results['ci_per_horizon'] = ci_per_horizon
    results['ci_h_pass'] = sum(1 for c in ci_per_horizon if c >= 0.85)

    # Per-cell CI
    ci_per_cell = np.zeros((5, 5))
    ci_per_cell_flat = []
    for c in range(C):
        lo = np.percentile(samples[:, :, :, c], 5, axis=1)
        hi = np.percentile(samples[:, :, :, c], 95, axis=1)
        cov = ((gt[:, :, c] >= lo) & (gt[:, :, c] <= hi)).mean()
        ci_per_cell[c // 5, c % 5] = cov
        ci_per_cell_flat.append(float(cov))
    results['ci_per_cell'] = ci_per_cell_flat
    results['ci_worst_cell'] = float(min(ci_per_cell_flat))
    results['ci_worst_cell_idx'] = int(np.argmin(ci_per_cell_flat))
    results['ci_mean'] = float(np.mean(ci_per_cell_flat))
    results['worst_cell_pass'] = results['ci_worst_cell'] >= 0.80

    # === S1: Surface Validity ===
    # Explosion rate (samples > 1.5 or < -0.5)
    explosion_rate = float(((samples > 1.5) | (samples < -0.5)).mean())
    results['explosion_rate'] = explosion_rate

    # === S4: Time Series ===
    gen_changes = np.diff(samples[:, 0], axis=1).reshape(-1, C)
    gt_changes = np.diff(gt, axis=1).reshape(-1, C)

    # Kurtosis ratio
    gen_kurt = kurtosis(gen_changes.flatten())
    gt_kurt = kurtosis(gt_changes.flatten())
    results['kurtosis_gen'] = float(gen_kurt)
    results['kurtosis_gt'] = float(gt_kurt)
    results['kurtosis_ratio'] = float(gen_kurt / (gt_kurt + 1e-6))
    results['kurtosis_pass'] = 0.5 <= results['kurtosis_ratio'] <= 2.0

    # === S5: Growing Uncertainty ===
    spread_per_horizon = []
    for h in range(T):
        spread_per_horizon.append(float(samples[:, :, h].std(axis=1).mean()))
    results['spread_per_horizon'] = spread_per_horizon

    # Check monotonicity (allowing small violations)
    monotonic_count = sum(1 for i in range(1, T) if spread_per_horizon[i] >= spread_per_horizon[i-1] * 0.95)
    results['growing_unc_ratio'] = monotonic_count / (T - 1)
    results['growing_unc_pass'] = results['growing_unc_ratio'] >= 0.80

    # === S8: Distributional ===
    # KS on daily changes
    ks_daily_pass = 0
    ks_daily_stats = []
    for c in range(C):
        stat, pval = ks_2samp(gen_changes[:, c], gt_changes[:, c])
        ks_daily_stats.append(float(stat))
        if stat < 0.15:
            ks_daily_pass += 1
    results['ks_daily_pass'] = ks_daily_pass
    results['ks_daily_stats'] = ks_daily_stats

    # KS on IV levels
    ks_levels_pass = 0
    for c in range(C):
        stat, _ = ks_2samp(samples[:, 0, :, c].flatten(), gt[:, :, c].flatten())
        if stat < 0.15:
            ks_levels_pass += 1
    results['ks_levels_pass'] = ks_levels_pass

    # Median bias
    ens_median = np.median(samples, axis=1)
    bias = float((ens_median - gt).mean())
    results['median_bias'] = bias

    # === S9: Cross-Cell Correlation ===
    gc = np.corrcoef(gen_changes.T)
    gtc = np.corrcoef(gt_changes.T)
    corr_ratio = float(np.abs(gc).mean() / (np.abs(gtc).mean() + 1e-6))
    results['corr_ratio'] = corr_ratio
    results['corr_pass'] = 0.80 <= corr_ratio <= 1.20

    # Frobenius distance
    frob = float(np.linalg.norm(gc - gtc) / np.linalg.norm(gtc))
    results['frob_rel'] = frob

    # Effective rank
    def eff_rank(corr):
        ev = np.linalg.eigvalsh(corr)[::-1]
        ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10)
        p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))
    er_gen = eff_rank(gc)
    er_gt = eff_rank(gtc)
    results['eff_rank_gen'] = er_gen
    results['eff_rank_gt'] = er_gt
    results['eff_rank_ratio'] = float(er_gen / (er_gt + 1e-6))

    # === Spread-Skill ===
    ens_mean = samples.mean(axis=1)
    skill = float(np.abs(ens_mean - gt).mean())
    spread = float(samples.std(axis=1).mean())
    results['spread'] = spread
    results['skill'] = skill
    results['spread_skill_ratio'] = float(spread / (skill + 1e-8))

    # === Conditionality (turb/calm) ===
    if rets is not None and len(rets) >= N + 60:
        rv = np.array([np.std(rets[i:i+30]) for i in range(len(rets) - 59)])
        # Align with windows
        rv_windows = rv[:N] if len(rv) >= N else rv
        if len(rv_windows) >= 20:
            turb_mask = rv_windows > np.percentile(rv_windows, 80)
            calm_mask = rv_windows < np.percentile(rv_windows, 20)
            if turb_mask.sum() > 5 and calm_mask.sum() > 5:
                spread_turb = float(samples[turb_mask].std(axis=1).mean())
                spread_calm = float(samples[calm_mask].std(axis=1).mean())
                results['turb_calm_ratio'] = float(spread_turb / (spread_calm + 1e-8))
            else:
                results['turb_calm_ratio'] = None
        else:
            results['turb_calm_ratio'] = None
    else:
        results['turb_calm_ratio'] = None

    # === Summary ===
    results['suites_summary'] = {
        'S1_surface': results['explosion_rate'] < 0.01,
        'S2_ci': results['worst_cell_pass'],
        'S4_timeseries': results['kurtosis_pass'],
        'S5_growing_unc': results['growing_unc_pass'],
        'S8_distributional': results['ks_daily_pass'] >= 20,
        'S9_correlation': results['corr_pass'],
    }
    results['suites_passed'] = sum(results['suites_summary'].values())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str,
                        default="models/backfill/flow_153a/final_model.pt")
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["test", "val"],
                        help="test=full test split (4540+), val=validation (4040-4540)")
    args = parser.parse_args()

    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading CLN transformer from {args.model_path}...")
    model, cfg, ckpt = load_cln_model(args.model_path, device)
    print(f"  Config: {cfg}")
    print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")

    # Load encoder
    encoder, _ = load_encoder(args.encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    rets = data["ret"]
    H, T = 30, 30

    # Load base predictions
    cached = np.load("models/backfill/flow_154b/base_predictions.npz")

    if args.eval_split == "test":
        # Test split: windows starting from 4540
        start_idx = 4540
        n_windows = len(surfaces) - start_idx - H - T + 1
        # Need base predictions for test split — generate them
        print(f"\nGenerating base predictions for test split ({n_windows} windows)...")
        from experiments.backfill.block_ar.train_cond_oneshot_flow import (
            ConditionalFactoredVelocityTransformer
        )
        base_ckpt = torch.load(args.base_model, weights_only=False, map_location=device)
        base_cfg = base_ckpt['config']
        base_model = ConditionalFactoredVelocityTransformer(
            n_frames=base_cfg['n_frames'], n_cells=base_cfg['n_cells'],
            d_model=base_cfg['d_model'], n_heads=base_cfg['n_heads'],
            n_layers=base_cfg['n_layers'], cond_dim=base_cfg.get('cond_dim', 128),
        ).to(device).eval()
        base_model.load_state_dict(base_ckpt['model_state_dict'])
        train_mean = torch.from_numpy(base_ckpt['train_mean']).float().to(device)
        train_std = torch.from_numpy(base_ckpt['train_std']).float().to(device)

        base_preds = []
        dt = 1.0 / 8
        with torch.no_grad():
            for i in range(start_idx, start_idx + n_windows):
                hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
                cond = encoder(normalize_iv(hist))
                preds = []
                for _ in range(5):
                    x = torch.randn(1, 750, device=device)
                    for step in range(8):
                        t = torch.full((1,), step * dt, device=device)
                        x = x + base_model(x, t, cond=cond) * dt
                    preds.append((x * train_std + train_mean).clamp(0, 1).cpu().numpy().flatten())
                base_preds.append(np.mean(preds, axis=0))
                if (len(base_preds)) % 50 == 0:
                    print(f"  Base predictions: {len(base_preds)}/{n_windows}")
        base_preds = np.array(base_preds, dtype=np.float32)
    else:
        # Val split: use cached predictions
        start_idx = 4040
        base_preds = cached['val_preds']
        n_windows = len(base_preds)

    print(f"\nEvaluating on {args.eval_split} split: {n_windows} windows, {args.n_samples} samples each")

    # Generate samples
    t0 = time.time()
    samples, gt = generate_samples(
        model, encoder, base_preds, surfaces, start_idx, n_windows,
        args.n_samples, device)
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    # Run evaluation
    print("\nRunning evaluation...")
    eval_rets = rets[start_idx:start_idx + n_windows + 60] if start_idx + n_windows + 60 <= len(rets) else None
    results = evaluate_full(samples, gt, eval_rets)
    results['model_path'] = args.model_path
    results['eval_split'] = args.eval_split
    results['n_windows'] = n_windows
    results['n_samples'] = args.n_samples
    results['generation_time_s'] = gen_time
    results['config'] = cfg

    # Save results
    output_path = Path(args.output_dir) / "summary.json"
    with open(output_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {args.model_path}")
    print(f"{'='*60}")
    print(f"Split: {args.eval_split} ({n_windows} windows, {args.n_samples} samples)")
    print(f"\nSuite results:")
    for suite, passed in results['suites_summary'].items():
        print(f"  {suite}: {'PASS' if passed else 'FAIL'}")
    print(f"\nTotal: {results['suites_passed']}/6 suites passed")
    print(f"\nKey metrics:")
    print(f"  CI worst cell: {results['ci_worst_cell']:.3f} (cell {results['ci_worst_cell_idx']})")
    print(f"  CI mean: {results['ci_mean']:.3f}")
    print(f"  CI horizons pass: {results['ci_h_pass']}/30")
    print(f"  KS daily: {results['ks_daily_pass']}/25")
    print(f"  KS levels: {results['ks_levels_pass']}/25")
    print(f"  Kurtosis ratio: {results['kurtosis_ratio']:.3f}")
    print(f"  Corr ratio: {results['corr_ratio']:.3f}")
    print(f"  Eff rank ratio: {results['eff_rank_ratio']:.3f}")
    print(f"  Spread-skill: {results['spread_skill_ratio']:.3f}")
    print(f"  Growing unc: {results['growing_unc_ratio']:.2f}")
    print(f"  Explosion rate: {results['explosion_rate']:.6f}")
    if results['turb_calm_ratio'] is not None:
        print(f"  Turb/calm ratio: {results['turb_calm_ratio']:.3f}")
    print(f"\nPer-cell CI (worst 5):")
    sorted_cells = sorted(enumerate(results['ci_per_cell']), key=lambda x: x[1])
    for idx, ci in sorted_cells[:5]:
        r, c = idx // 5, idx % 5
        print(f"  cell ({r},{c}): {ci:.3f}")


if __name__ == "__main__":
    main()
