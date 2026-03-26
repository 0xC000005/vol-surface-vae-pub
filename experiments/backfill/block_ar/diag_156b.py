#!/usr/bin/env python
"""
Diagnostic script for Exp 156b: noise_dim=8 CLN transformer (H1-S2).
Same A-F structure as 156a, plus comparison with 156a results.
"""

import numpy as np
import torch
import json
from pathlib import Path
from scipy.stats import ks_2samp, kurtosis

import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import load_encoder, normalize_iv
from experiments.backfill.block_ar.train_155d_cln_transformer import CLNResidualTransformer


def load_model(path, device='cuda'):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    cfg = ckpt['config']
    model = CLNResidualTransformer(
        n_frames=cfg['n_frames'], n_cells=cfg['n_cells'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], cond_dim=cfg['cond_dim'],
        noise_dim=cfg['noise_dim'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def generate_samples(model, conditions, base_preds, n_samples=50, device='cuda'):
    N = len(base_preds)
    T, C = 30, 25
    all_samples = []
    with torch.no_grad():
        for i in range(N):
            cond = torch.from_numpy(conditions[i:i+1]).float().to(device).expand(n_samples, -1)
            noise = torch.randn(n_samples, model.noise_dim, device=device)
            residual = model(cond, noise).cpu().numpy()
            combined = np.clip(base_preds[i] + residual, 0, 1)
            all_samples.append(combined.reshape(n_samples, T, C))
    return np.array(all_samples)


def main():
    device = 'cuda'
    results = {}

    model, cfg = load_model('models/backfill/flow_156b/final_model.pt', device)
    print(f"Model config: noise_dim={cfg['noise_dim']}, d_model={cfg['d_model']}")

    cached = np.load("models/backfill/flow_154b/base_predictions.npz")
    val_preds = cached["val_preds"]
    val_gts = cached["val_gts"]

    encoder, _ = load_encoder("models/backfill/block_ar_vol_scaled_30ep/best_model.pt", device)
    surfaces = np.load("data/vol_surface_with_ret.npz")["surface"]
    val_conds = []
    with torch.no_grad():
        for i in range(4040, 4040 + len(val_preds)):
            hist = torch.from_numpy(surfaces[i:i+30][None].astype(np.float32)).to(device)
            val_conds.append(encoder(normalize_iv(hist)).cpu().numpy())
    val_conds = np.concatenate(val_conds)

    print("Generating samples...")
    samples = generate_samples(model, val_conds, val_preds, n_samples=50, device=device)
    gt = val_gts.reshape(len(val_gts), 30, 25)
    N, K, T, C = samples.shape

    # A. Per-cell CI
    print("\n=== A. Per-cell CI breakdown ===")
    cell_ci = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            cidx = r * 5 + c
            lo = np.percentile(samples[:, :, :, cidx], 5, axis=1)
            hi = np.percentile(samples[:, :, :, cidx], 95, axis=1)
            cov = ((gt[:, :, cidx] >= lo) & (gt[:, :, cidx] <= hi)).mean()
            cell_ci[r, c] = cov
    print("Cell CI (5x5 grid):")
    for r in range(5):
        print("  " + "  ".join(f"{cell_ci[r,c]:.3f}" for c in range(5)))
    results['cell_ci'] = cell_ci.tolist()
    results['ci_worst'] = float(cell_ci.min())
    results['ci_mean'] = float(cell_ci.mean())
    worst_r, worst_c = np.unravel_index(cell_ci.argmin(), cell_ci.shape)
    print(f"Worst cell: ({worst_r},{worst_c}) = {cell_ci[worst_r,worst_c]:.3f}")
    print(f"Mean CI: {cell_ci.mean():.3f}")

    # B. Correlation
    print("\n=== B. Correlation matrix vs GT ===")
    gen_ch = np.diff(samples[:, 0], axis=1).reshape(-1, C)
    gt_ch = np.diff(gt, axis=1).reshape(-1, C)
    gen_corr = np.corrcoef(gen_ch.T)
    gt_corr = np.corrcoef(gt_ch.T)
    corr_ratio = np.abs(gen_corr).mean() / (np.abs(gt_corr).mean() + 1e-6)

    def eff_rank(corr_mat):
        ev = np.linalg.eigvalsh(corr_mat)[::-1]
        ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10)
        p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    gen_er = eff_rank(gen_corr)
    gt_er = eff_rank(gt_corr)
    print(f"Corr ratio: {corr_ratio:.3f}")
    print(f"Eff rank: gen={gen_er:.2f} vs GT={gt_er:.2f} (ratio={gen_er/gt_er:.3f})")
    gen_ev = np.linalg.eigvalsh(gen_corr)[::-1][:5]
    gt_ev = np.linalg.eigvalsh(gt_corr)[::-1][:5]
    print(f"Top 5 eigenvalues:")
    print(f"  Gen: {gen_ev}")
    print(f"  GT:  {gt_ev}")
    results['corr_ratio'] = float(corr_ratio)
    results['eff_rank_gen'] = gen_er
    results['eff_rank_gt'] = gt_er

    # C. Spread trajectory
    print("\n=== C. Spread trajectory ===")
    hist_path = Path('models/backfill/flow_156b/training_history.json')
    if hist_path.exists():
        history = json.load(open(hist_path))
        spreads = [(h['epoch'], h.get('spread', None)) for h in history if 'spread' in h]
        peak_spread = max(s for _, s in spreads)
        peak_ep = [ep for ep, s in spreads if s == peak_spread][0]
        print(f"Peak spread: {peak_spread:.4f} at epoch {peak_ep}")
        print(f"Final spread: {spreads[-1][1]:.4f}")
        print(f"Contraction: {peak_spread / spreads[-1][1]:.2f}x")
        results['spread_peak'] = peak_spread
        results['spread_peak_epoch'] = peak_ep
        results['spread_final'] = spreads[-1][1]

    # D. CLN scales
    print("\n=== D. CLN scale analysis ===")
    scale_norms = []
    for i, layer in enumerate(model.layers):
        for n in ['temp_cln', 'temp_ff_cln', 'spat_cln', 'spat_ff_cln']:
            scale_norms.append(layer[n].scale_proj.weight.detach().norm().item())
    layer_scales = [np.mean([model.layers[i][n].scale_proj.weight.detach().norm().item()
                             for n in ['temp_cln', 'temp_ff_cln', 'spat_cln', 'spat_ff_cln']])
                    for i in range(4)]
    print(f"Mean CLN scale norm: {np.mean(scale_norms):.3f}")
    print(f"Per-layer: {['%.3f' % s for s in layer_scales]}")
    results['cln_mean_scale'] = float(np.mean(scale_norms))

    # E. Scaling tests
    print("\n=== E. Scaling tests ===")
    for scale in [1.3, 1.5, 2.0]:
        all_sc = []
        with torch.no_grad():
            for i in range(N):
                cond = torch.from_numpy(val_conds[i:i+1]).float().to(device).expand(50, -1)
                noise = torch.randn(50, model.noise_dim, device=device)
                residual = model(cond, noise).cpu().numpy()
                combined = np.clip(val_preds[i] + residual * scale, 0, 1)
                all_sc.append(combined.reshape(50, 30, 25))
        sc = np.array(all_sc)
        wci = 1.0
        for cidx in range(C):
            lo = np.percentile(sc[:, :, :, cidx], 5, axis=1)
            hi = np.percentile(sc[:, :, :, cidx], 95, axis=1)
            cov = ((gt[:, :, cidx] >= lo) & (gt[:, :, cidx] <= hi)).mean()
            wci = min(wci, cov)
        sc_ch = np.diff(sc[:, 0], axis=1).reshape(-1, C)
        sc_corr = np.corrcoef(sc_ch.T)
        sc_cr = np.abs(sc_corr).mean() / (np.abs(gt_corr).mean() + 1e-6)
        ks2 = sum(1 for c2 in range(C) if ks_2samp(sc_ch[:, c2], gt_ch[:, c2])[0] < 0.15)
        print(f"{scale}x: CI={wci:.3f}  corr={sc_cr:.3f}  KS={ks2}/25")
        results[f'scaled_{scale}x'] = {'ci_worst': float(wci), 'corr': float(sc_cr), 'ks': ks2}

    # F. Per-horizon
    print("\n=== F. Per-horizon spread and CI ===")
    for h in [0, 4, 9, 14, 19, 24, 29]:
        lo = np.percentile(samples[:, :, h], 5, axis=1)
        hi = np.percentile(samples[:, :, h], 95, axis=1)
        cov = ((gt[:, h] >= lo) & (gt[:, h] <= hi)).mean()
        spread = samples[:, :, h].std(axis=1).mean()
        mae = np.abs(samples[:, :, h].mean(axis=1) - gt[:, h]).mean()
        print(f"  h={h+1:2d}: CI={cov:.3f}  spread={spread:.4f}  mae={mae:.4f}  SS={spread/(mae+1e-8):.3f}")

    spreads_h = [samples[:, :, h].std(axis=1).mean() for h in range(30)]
    gu = sum(1 for i in range(1, 30) if spreads_h[i] >= spreads_h[i-1] * 0.95)
    print(f"\nGrowing uncertainty: {gu}/29")
    print(f"Spread h1={spreads_h[0]:.4f}  h30={spreads_h[-1]:.4f}  ratio={spreads_h[-1]/spreads_h[0]:.2f}x")

    # Comparison with 156a
    print("\n=== COMPARISON: 156a (dim=4) vs 156b (dim=8) vs 155d (dim=32) ===")
    a_diag = Path('results/block_ar/156a_diag/diagnostics.json')
    if a_diag.exists():
        a = json.load(open(a_diag))
        print(f"{'Metric':<20} {'156a (dim=4)':<15} {'156b (dim=8)':<15} {'155d (dim=32)':<15}")
        print(f"{'CI worst':<20} {a['ci_worst']:<15.3f} {results['ci_worst']:<15.3f} {'0.748':<15}")
        print(f"{'CI mean':<20} {a['ci_mean']:<15.3f} {results['ci_mean']:<15.3f} {'~0.85':<15}")
        print(f"{'Corr ratio':<20} {a['corr_ratio']:<15.3f} {results['corr_ratio']:<15.3f} {'0.910':<15}")
        print(f"{'Eff rank gen':<20} {a['eff_rank_gen']:<15.2f} {results['eff_rank_gen']:<15.2f} {'~7.0':<15}")

    out_dir = Path('results/block_ar/156b_diag')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'diagnostics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDiagnostics saved to {out_dir}/diagnostics.json")


if __name__ == "__main__":
    main()
