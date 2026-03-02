"""
Comprehensive diagnostic: Why does the diffusion model fail to learn
conditional uncertainty despite its clear presence in the data?

Tests:
  H1: Denoiser noise prediction error — is it regime-dependent?
  H2: Sample spread from the model — does it vary by regime?
  H3: NLL optimal sigma — is it regime-dependent? (would NLL work in principle?)
  H4: Oracle sigma from data — what's the max achievable Q5/Q1?
  H5: Per-sample training loss by regime — does the model see different loss for turb vs calm?
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.backfill.block_ar.config_block_ar import BlockARPOCConfig
from diffusion.block_ar.block_ar_ddpm import ConditionalBlockARDDPM, denormalize_iv, normalize_iv


def load_model(model_path, device="cuda"):
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    config_dict = checkpoint.get("config", {})
    config = BlockARPOCConfig(**{
        k: v for k, v in config_dict.items()
        if k in BlockARPOCConfig.__dataclass_fields__
    })
    config.device = device
    model = ConditionalBlockARDDPM(config).to(device)
    state = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config


def make_windows(surfaces, test_start, history_len=30, future_len=30, max_windows=400):
    windows = []
    for i in range(test_start, len(surfaces) - history_len - future_len + 1):
        if len(windows) >= max_windows:
            break
        hist = surfaces[i:i + history_len]
        fut = surfaces[i + history_len:i + history_len + future_len]
        daily_mean = hist.mean(axis=(1, 2))
        daily_changes = np.diff(daily_mean)
        vov = np.std(daily_changes)
        windows.append({"history": hist, "future": fut, "vov": vov})
    return windows


def quintile_mask(vovs, q, edges):
    return (vovs >= edges[q]) & (vovs < edges[q + 1] + 1e-10)


# ═══════════════════════════════════════════════════════════════════════
# H1 + H5: Per-sample training loss by regime
# ═══════════════════════════════════════════════════════════════════════
def test_h1_per_sample_loss(model, config, windows, device="cuda"):
    """
    Compute per-sample training loss (MSE on noise prediction) for each window.
    If turb windows have higher loss, the model "knows" they're harder but can't
    translate that into wider output spread.
    """
    print("\n" + "=" * 70)
    print("H1/H5: Per-Sample Training Loss by Regime")
    print("=" * 70)

    vovs = np.array([w["vov"] for w in windows])
    edges = np.percentile(vovs, [0, 20, 40, 60, 80, 100])

    per_window_loss = []
    with torch.no_grad():
        for wi, w in enumerate(windows):
            if wi % 100 == 0:
                print(f"  Window {wi}/{len(windows)}...")

            hist = torch.tensor(normalize_iv(w["history"]), dtype=torch.float32).unsqueeze(0).to(device)
            fut = torch.tensor(normalize_iv(w["future"]), dtype=torch.float32).unsqueeze(0).to(device)

            # Use model's forward to get loss
            result = model(hist, fut)
            per_window_loss.append(result["loss"].item())

    per_window_loss = np.array(per_window_loss)

    print(f"\nPer-Sample Diffusion Loss by Vol-of-Vol Quintile:")
    print(f"{'Quintile':>10} | {'Mean vov':>10} | {'Mean loss':>12} | {'Std loss':>12} | N")
    print("-" * 65)

    q_means = []
    for q in range(5):
        mask = quintile_mask(vovs, q, edges)
        ql = per_window_loss[mask]
        q_means.append(ql.mean())
        print(f"  Q{q + 1}      | {vovs[mask].mean():>10.5f} | {ql.mean():>12.6f} | {ql.std():>12.6f} | {mask.sum()}")

    ratio = q_means[4] / q_means[0] if q_means[0] > 0 else float('inf')
    sr, sp = stats.spearmanr(vovs, per_window_loss)

    print(f"\n  Q5/Q1 ratio: {ratio:.4f}")
    print(f"  Spearman(vov, loss): {sr:.4f} (p={sp:.2e})")

    if ratio > 1.05:
        print("  → CONFIRMED: Model incurs HIGHER loss for turbulent windows")
        print("  → The denoiser 'knows' turb is harder, but MSE doesn't translate to spread")
    else:
        print("  → DENIED: Training loss is regime-INDEPENDENT")
        print("  → MSE has fully equalized prediction difficulty across regimes")

    return {
        "q_means": [float(m) for m in q_means],
        "q5_q1_ratio": float(ratio),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


# ═══════════════════════════════════════════════════════════════════════
# H2: Sample spread by regime
# ═══════════════════════════════════════════════════════════════════════
def test_h2_sample_spread(model, config, windows, device="cuda", n_samples=30, max_windows=200):
    """
    Generate n_samples per window. Measure sample spread (cross-sample std).
    Does it vary with regime?
    """
    print("\n" + "=" * 70)
    print(f"H2: Sample Spread by Regime (n_samples={n_samples})")
    print("=" * 70)

    vovs = []
    spreads_by_h = {0: [], 6: [], 13: [], 29: []}
    overall_spreads = []

    with torch.no_grad():
        for wi, w in enumerate(windows[:max_windows]):
            if wi % 25 == 0:
                print(f"  Window {wi}/{min(len(windows), max_windows)}...")

            hist_np = w["history"]
            hist = torch.tensor(normalize_iv(hist_np), dtype=torch.float32).unsqueeze(0).to(device)

            # model.sample returns (B, n_samples, T, 5, 5) already denormalized to [0,1]
            samp = model.sample(hist, n_samples=n_samples)  # (1, n_samples, T, 5, 5)
            all_samps = samp[0].cpu().numpy()  # (n_samples, T, 5, 5)

            for h in [0, 6, 13, 29]:
                if h < all_samps.shape[1]:
                    cell_std = all_samps[:, h, :, :].std(axis=0).mean()
                    spreads_by_h[h].append(cell_std)

            overall_spreads.append(all_samps.std(axis=0).mean())
            vovs.append(w["vov"])

    vovs = np.array(vovs)
    overall_spreads = np.array(overall_spreads)
    edges = np.percentile(vovs, [0, 20, 40, 60, 80, 100])

    print(f"\nOverall Sample Spread by Quintile:")
    print(f"{'Quintile':>10} | {'Mean vov':>10} | {'Mean spread':>12} | N")
    print("-" * 45)

    q_sp = []
    for q in range(5):
        mask = quintile_mask(vovs, q, edges)
        qs = overall_spreads[mask].mean()
        q_sp.append(qs)
        print(f"  Q{q + 1}      | {vovs[mask].mean():>10.5f} | {qs:>12.6f} | {mask.sum()}")

    ratio = q_sp[4] / q_sp[0] if q_sp[0] > 0 else float('inf')
    sr, sp = stats.spearmanr(vovs, overall_spreads)
    print(f"\n  Q5/Q1: {ratio:.3f}")
    print(f"  Spearman(vov, spread): {sr:.4f} (p={sp:.2e})")

    print(f"\nPer-Horizon Spread Q5/Q1:")
    h_results = {}
    for h in [0, 6, 13, 29]:
        s = np.array(spreads_by_h[h])
        qs = []
        for q in range(5):
            mask = quintile_mask(vovs, q, edges)
            qs.append(s[mask].mean())
        hr = qs[4] / qs[0] if qs[0] > 0 else float('inf')
        hsr, _ = stats.spearmanr(vovs, s)
        print(f"  h={h + 1:>2d}: Q5/Q1={hr:.3f}, Spearman={hsr:.4f}")
        h_results[f"h{h + 1}"] = {"q5_q1": float(hr), "spearman": float(hsr)}

    return {
        "overall_q5_q1": float(ratio),
        "overall_spearman": float(sr),
        "q_spread": [float(s) for s in q_sp],
        "per_horizon": h_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# H3: Is the optimal NLL sigma regime-dependent?
# ═══════════════════════════════════════════════════════════════════════
def test_h3_optimal_sigma(model, config, windows, device="cuda"):
    """
    For each window, compute the model's actual residuals on the training target.
    The optimal NLL sigma for a group of windows is the RMS of those residuals.
    If turb RMS > calm RMS, NLL CAN in principle learn regime-dependent sigma.
    If RMS is flat, NLL fundamentally cannot distinguish regimes.
    """
    print("\n" + "=" * 70)
    print("H3: Optimal NLL Sigma — Is It Regime-Dependent?")
    print("=" * 70)

    vovs = np.array([w["vov"] for w in windows])
    edges = np.percentile(vovs, [0, 20, 40, 60, 80, 100])

    # For each window, compute residuals at 3 different timesteps
    # using the model's internal forward process
    # We manually do: encode, sample t, forward diffuse, predict noise, compute residual

    per_window_residual_sq = {t: [] for t in [10, 50, 90]}
    scheduler = model.scheduler

    with torch.no_grad():
        for wi, w in enumerate(windows):
            if wi % 100 == 0:
                print(f"  Window {wi}/{len(windows)}...")

            hist = torch.tensor(normalize_iv(w["history"]), dtype=torch.float32).unsqueeze(0).to(device)
            fut = torch.tensor(normalize_iv(w["future"]), dtype=torch.float32).unsqueeze(0).to(device)

            # Get the block-level target (first block, as in training)
            target_block = fut[:, :config.block_size]  # (1, bs, 5, 5)

            # Encode condition (forward task: past only)
            condition = model.encoder(hist)
            condition = model._augment_condition(condition, None)
            condition = model._add_regime_features(condition, hist)

            # If ratio target, transform the target as training does
            if config.ratio_target and config.ratio_target_mode == "vol_scaled":
                eps_iv = 1e-4
                baseline = model._compute_baseline(hist)  # (1, 1, 5, 5)
                target_abs = denormalize_iv(target_block).clamp(eps_iv, 1.0 - eps_iv)
                past_abs = denormalize_iv(hist)
                mean_iv = past_abs.mean(dim=(-1, -2))
                daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
                vol = daily_chg.std(dim=1, keepdim=True)
                vol_scale = (vol / config.global_mean_vol).clamp(config.vol_scale_min, config.vol_scale_max)
                vol_scale = vol_scale.pow(config.vol_scale_power).unsqueeze(-1).unsqueeze(-1)
                log_ratio = torch.log(target_abs / baseline.clamp(min=eps_iv))
                train_target = log_ratio / vol_scale
            else:
                train_target = target_block

            # Positions for denoiser
            positions = torch.arange(config.block_size, device=device).unsqueeze(0)

            for t_val in [10, 50, 90]:
                t = torch.tensor([t_val], device=device)
                noise = torch.randn_like(train_target)

                # Forward diffuse
                sqrt_ab = scheduler.sqrt_alpha_bar[t_val]
                sqrt_1mab = scheduler.sqrt_one_minus_alpha_bar[t_val]
                noisy = sqrt_ab * train_target + sqrt_1mab * noise

                # Predict noise
                B, T, H, W = noisy.shape
                x_flat = noisy.reshape(B, T, H * W)
                t_expanded = t.unsqueeze(1).expand(B, T)
                noise_pred_flat = model.denoiser(x_flat, condition, positions, t_expanded)
                noise_pred = noise_pred_flat.reshape(B, T, H, W)

                # Per-element squared residual
                sq_residual = ((noise_pred - noise) ** 2).mean().item()
                per_window_residual_sq[t_val].append(sq_residual)

    print(f"\nOptimal NLL Sigma (sqrt of mean squared residual) by Quintile:")
    print(f"{'Timestep':>10} | {'Q1(calm)':>12} | {'Q3(mid)':>12} | {'Q5(turb)':>12} | {'Q5/Q1':>8} | {'Spearman':>10}")
    print("-" * 80)

    h3_results = {}
    for t_val in [10, 50, 90]:
        mses = np.array(per_window_residual_sq[t_val])
        q_rms = []
        for q in range(5):
            mask = quintile_mask(vovs, q, edges)
            q_rms.append(np.sqrt(mses[mask].mean()))

        ratio = q_rms[4] / q_rms[0] if q_rms[0] > 0 else float('inf')
        sr, sp = stats.spearmanr(vovs, np.sqrt(mses))
        print(f"  t={t_val:>4d} | {q_rms[0]:>12.6f} | {q_rms[2]:>12.6f} | {q_rms[4]:>12.6f} | {ratio:>8.4f} | {sr:>8.4f} (p={sp:.2e})")

        h3_results[t_val] = {
            "q_rms": [float(r) for r in q_rms],
            "q5_q1": float(ratio),
            "spearman": float(sr),
            "spearman_p": float(sp),
        }

    # Interpretation
    t50 = h3_results.get(50, {})
    if t50.get("q5_q1", 1) > 1.02:
        print(f"\n  → Optimal sigma IS regime-dependent (Q5/Q1={t50['q5_q1']:.4f})")
        print(f"  → NLL CAN learn conditional sigma in principle")
        print(f"  → But the signal may be too weak relative to SGD noise")
    else:
        print(f"\n  → Optimal sigma is FLAT (Q5/Q1={t50.get('q5_q1', '?')})")
        print(f"  → MSE training has equalized residuals across regimes")
        print(f"  → NLL fundamentally cannot learn conditional sigma from these residuals")

    return h3_results


# ═══════════════════════════════════════════════════════════════════════
# H4: Oracle sigma from ground truth data
# ═══════════════════════════════════════════════════════════════════════
def test_h4_oracle(windows):
    """
    Pure data analysis: what's the cross-window spread ratio by regime?
    This sets the CEILING for what any model could achieve.
    """
    print("\n" + "=" * 70)
    print("H4: Oracle Sigma from Ground Truth Data")
    print("=" * 70)

    vovs = np.array([w["vov"] for w in windows])
    edges = np.percentile(vovs, [0, 20, 40, 60, 80, 100])

    for h_idx, h in enumerate([0, 6, 13, 29]):
        horizon_label = [1, 7, 14, 30][h_idx]
        future_at_h = np.array([w["future"][h] for w in windows])  # (N, 5, 5)
        baseline = np.array([w["history"][-1] for w in windows])  # (N, 5, 5)
        changes = future_at_h - baseline

        # Per-quintile cross-window std of changes
        print(f"\n  Horizon h={horizon_label}:")
        q_stds = []
        for q in range(5):
            mask = quintile_mask(vovs, q, edges)
            std_per_cell = changes[mask].std(axis=0)  # (5, 5)
            q_stds.append(std_per_cell.mean())
            if q in [0, 4]:
                print(f"    Q{q + 1} cross-window std: {q_stds[-1]:.6f} (mean over cells)")

        ratio = q_stds[4] / q_stds[0] if q_stds[0] > 0 else float('inf')
        print(f"    Q5/Q1: {ratio:.3f}")

    # Per-cell detail at h=1
    print(f"\n  Per-cell Q5/Q1 at h=1:")
    future_h1 = np.array([w["future"][0] for w in windows])
    baseline_all = np.array([w["history"][-1] for w in windows])
    changes_h1 = future_h1 - baseline_all

    cell_ratios = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            ch = changes_h1[:, r, c]
            mask_q1 = quintile_mask(vovs, 0, edges)
            mask_q5 = quintile_mask(vovs, 4, edges)
            std_q1 = ch[mask_q1].std()
            std_q5 = ch[mask_q5].std()
            cell_ratios[r, c] = std_q5 / std_q1 if std_q1 > 0 else 0

    print(np.array2string(cell_ratios, precision=2))
    print(f"    Mean: {cell_ratios.mean():.3f}, Min: {cell_ratios.min():.2f}, Max: {cell_ratios.max():.2f}")

    return {"cell_ratios_h1": cell_ratios.tolist()}


# ═══════════════════════════════════════════════════════════════════════
# NEW: Ground truth vs model residual regime-dependency in OUTPUT SPACE
# ═══════════════════════════════════════════════════════════════════════
def test_output_residuals(model, config, windows, device="cuda", n_samples=30, max_windows=200):
    """
    Generate samples, compute (ensemble_mean - GT) in IV space.
    Are these output-space residuals regime-dependent?
    Compare to ground truth cross-window spread.
    """
    print("\n" + "=" * 70)
    print(f"Output-Space Residuals and Spread by Regime (n={n_samples})")
    print("=" * 70)

    vovs = []
    mae_h1 = []
    spread_h1 = []
    mae_overall = []
    spread_overall = []

    # Per-cell spread tracking at h=1
    cell_spreads = []  # list of (5,5) arrays

    with torch.no_grad():
        for wi, w in enumerate(windows[:max_windows]):
            if wi % 25 == 0:
                print(f"  Window {wi}/{min(len(windows), max_windows)}...")

            hist = torch.tensor(normalize_iv(w["history"]), dtype=torch.float32).unsqueeze(0).to(device)
            gt = w["future"]  # (T, 5, 5) in IV space

            samp = model.sample(hist, n_samples=n_samples)  # (1, n_samples, T, 5, 5)
            all_samps = samp[0].cpu().numpy()  # (n_samples, T, 5, 5)

            ens_mean = all_samps.mean(axis=0)
            mae_h1.append(np.abs(ens_mean[0] - gt[0]).mean())
            mae_overall.append(np.abs(ens_mean - gt).mean())

            spread_h1.append(all_samps[:, 0, :, :].std(axis=0).mean())
            spread_overall.append(all_samps.std(axis=0).mean())

            cell_spreads.append(all_samps[:, 0, :, :].std(axis=0))  # (5,5)
            vovs.append(w["vov"])

    vovs = np.array(vovs)
    mae_h1 = np.array(mae_h1)
    spread_h1 = np.array(spread_h1)
    mae_overall = np.array(mae_overall)
    spread_overall = np.array(spread_overall)
    cell_spreads = np.array(cell_spreads)  # (N, 5, 5)

    edges = np.percentile(vovs, [0, 20, 40, 60, 80, 100])

    print(f"\nOverall Metrics by Quintile:")
    print(f"{'Q':>5} | {'vov':>8} | {'MAE h=1':>10} | {'Spread h=1':>12} | {'MAE all':>10} | {'Spread all':>12}")
    print("-" * 70)

    q_spread_h1 = []
    q_mae_h1 = []
    for q in range(5):
        mask = quintile_mask(vovs, q, edges)
        qs = spread_h1[mask].mean()
        qm = mae_h1[mask].mean()
        q_spread_h1.append(qs)
        q_mae_h1.append(qm)
        print(f"  Q{q + 1} | {vovs[mask].mean():>8.5f} | {qm:>10.6f} | {qs:>12.6f} | {mae_overall[mask].mean():>10.6f} | {spread_overall[mask].mean():>12.6f}")

    ratio_sp = q_spread_h1[4] / q_spread_h1[0] if q_spread_h1[0] > 0 else float('inf')
    ratio_mae = q_mae_h1[4] / q_mae_h1[0] if q_mae_h1[0] > 0 else float('inf')
    sr_sp, sp_sp = stats.spearmanr(vovs, spread_h1)
    sr_mae, sp_mae = stats.spearmanr(vovs, mae_h1)

    print(f"\n  Spread h=1: Q5/Q1={ratio_sp:.3f}, Spearman={sr_sp:.4f} (p={sp_sp:.2e})")
    print(f"  MAE h=1: Q5/Q1={ratio_mae:.3f}, Spearman={sr_mae:.4f} (p={sp_mae:.2e})")

    # Per-cell spread analysis
    print(f"\n  Per-cell spread Q5/Q1 at h=1:")
    cell_q5q1 = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            cs = cell_spreads[:, r, c]
            mask_q1 = quintile_mask(vovs, 0, edges)
            mask_q5 = quintile_mask(vovs, 4, edges)
            s_q1 = cs[mask_q1].mean()
            s_q5 = cs[mask_q5].mean()
            cell_q5q1[r, c] = s_q5 / s_q1 if s_q1 > 0 else 0

    print(np.array2string(cell_q5q1, precision=3))
    print(f"  Mean: {cell_q5q1.mean():.3f}")

    # Compare model cell spread Q5/Q1 to GT cell spread Q5/Q1
    print(f"\n  Comparison: Model cell Q5/Q1 vs GT cell Q5/Q1 at h=1:")
    gt_h1 = np.array([w["future"][0] for w in windows[:max_windows]])
    bl = np.array([w["history"][-1] for w in windows[:max_windows]])
    gt_changes = gt_h1 - bl
    gt_cell_q5q1 = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            ch = gt_changes[:, r, c]
            mask_q1 = quintile_mask(vovs, 0, edges)
            mask_q5 = quintile_mask(vovs, 4, edges)
            gt_cell_q5q1[r, c] = ch[mask_q5].std() / ch[mask_q1].std() if ch[mask_q1].std() > 0 else 0

    recovery = cell_q5q1 / gt_cell_q5q1
    recovery[gt_cell_q5q1 == 0] = 0
    print(f"  GT cell Q5/Q1 mean: {gt_cell_q5q1.mean():.3f}")
    print(f"  Model cell Q5/Q1 mean: {cell_q5q1.mean():.3f}")
    print(f"  Recovery ratio: {cell_q5q1.mean() / gt_cell_q5q1.mean():.3f} ({cell_q5q1.mean() / gt_cell_q5q1.mean() * 100:.1f}%)")

    return {
        "spread_h1_q5q1": float(ratio_sp),
        "spread_h1_spearman": float(sr_sp),
        "mae_h1_q5q1": float(ratio_mae),
        "mae_h1_spearman": float(sr_mae),
        "model_cell_q5q1_mean": float(cell_q5q1.mean()),
        "gt_cell_q5q1_mean": float(gt_cell_q5q1.mean()),
        "recovery_pct": float(cell_q5q1.mean() / gt_cell_q5q1.mean() * 100) if gt_cell_q5q1.mean() > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_label", default="model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--tests", nargs="+", default=["h1", "h3", "h4", "h2", "output"])
    parser.add_argument("--output_dir", default="results/block_ar/conditional_uncertainty_diagnostic")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    print(f"Loading model: {args.model_path}")
    model, config = load_model(args.model_path, args.device)

    print(f"\nModel config:")
    print(f"  ratio_target: {config.ratio_target}")
    print(f"  ratio_target_mode: {getattr(config, 'ratio_target_mode', 'none')}")
    print(f"  forward_only: {config.forward_only}")
    print(f"  denoiser_type: {config.denoiser_type}")

    windows = make_windows(surfaces, config.test_start, max_windows=args.max_windows)
    print(f"Test windows: {len(windows)}")
    print(f"VoV range: [{min(w['vov'] for w in windows):.5f}, {max(w['vov'] for w in windows):.5f}]")

    results = {"model": args.model_path, "label": args.model_label}

    if "h1" in args.tests:
        results["h1_per_sample_loss"] = test_h1_per_sample_loss(model, config, windows, args.device)

    if "h3" in args.tests:
        results["h3_optimal_sigma"] = test_h3_optimal_sigma(model, config, windows, args.device)

    if "h4" in args.tests:
        results["h4_oracle"] = test_h4_oracle(windows)

    if "h2" in args.tests:
        results["h2_sample_spread"] = test_h2_sample_spread(
            model, config, windows, args.device,
            n_samples=args.n_samples,
            max_windows=min(200, args.max_windows),
        )

    if "output" in args.tests:
        results["output_residuals"] = test_output_residuals(
            model, config, windows, args.device,
            n_samples=args.n_samples,
            max_windows=min(200, args.max_windows),
        )

    out_path = Path(args.output_dir) / f"{args.model_label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    for key, val in results.items():
        if isinstance(val, dict) and "q5_q1_ratio" in val:
            print(f"  {key}: Q5/Q1={val['q5_q1_ratio']:.4f}, Spearman={val.get('spearman_r', val.get('spearman', '?'))}")
        elif isinstance(val, dict) and "overall_q5_q1" in val:
            print(f"  {key}: Q5/Q1={val['overall_q5_q1']:.3f}, Spearman={val['overall_spearman']:.4f}")
        elif isinstance(val, dict) and "spread_h1_q5q1" in val:
            print(f"  {key}: Spread Q5/Q1={val['spread_h1_q5q1']:.3f}, Recovery={val.get('recovery_pct', '?'):.1f}%")


if __name__ == "__main__":
    main()
