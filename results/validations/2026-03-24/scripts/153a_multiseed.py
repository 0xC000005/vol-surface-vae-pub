#!/usr/bin/env python
"""
Multi-Seed Evaluation of 153a (seeds 42, 43, 44).

Reproduces the exact eval logic from eval_cond_oneshot_flow.py for 3 seeds,
then produces a comparison table + verification_result.json.

Usage:
    PYTHONPATH=. python results/validations/2026-03-24/scripts/153a_multiseed.py --device cuda
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, kurtosis
from statsmodels.tsa.stattools import coint

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer,
    load_encoder,
    normalize_iv,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def make_serial(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serial(v) for v in obj]
    return obj


def eff_rank(corr):
    ev = np.linalg.eigvalsh(corr)[::-1]
    ev = np.maximum(ev, 0)
    p = ev / (ev.sum() + 1e-10)
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def generate_samples(velocity_net, encoder, history_batch, n_samples, n_steps,
                     train_mean, train_std, device):
    """Generate samples for a batch of history windows.

    Args:
        history_batch: (B, 30, 5, 5) in [0, 1]
    Returns:
        (B, n_samples, 30, 5, 5) in [0, 1]
    """
    B = history_batch.shape[0]
    DIM = 750
    dt = 1.0 / n_steps

    hist_norm = normalize_iv(history_batch)
    cond = encoder(hist_norm)  # (B, cond_dim)
    cond_exp = cond.repeat_interleave(n_samples, dim=0)  # (B*S, cond_dim)

    x = torch.randn(B * n_samples, DIM, device=device)
    for step in range(n_steps):
        t = torch.full((B * n_samples,), step * dt, device=device)
        x = x + velocity_net(x, t, cond=cond_exp) * dt

    mean_t = torch.from_numpy(train_mean).float().to(device)
    std_t = torch.from_numpy(train_std).float().to(device)
    samples = x * std_t + mean_t
    samples = torch.clamp(samples, 0, 1)
    return samples.reshape(B, n_samples, 30, 5, 5)


# ── 9-Suite Evaluation (exact copy of eval_cond_oneshot_flow.py logic) ───

def evaluate_9_suites(cond_samples, ground_truth, history_arr, rets, test_start, n_windows):
    """
    cond_samples: (N, S, 30, 5, 5) in [0,1]
    ground_truth: (N, 30, 5, 5) in [0,1]
    history_arr:  (N, 30, 5, 5)
    rets: full returns array
    test_start: int
    n_windows: int

    Returns: dict with suites, metrics, pass_count
    """
    N, S = cond_samples.shape[:2]
    T = 30
    H = 30

    # ── Suite 1: Surface Validity ──
    explosion_count = 0
    cal_arb_count = 0
    but_arb_count = 0

    for i in range(min(N, 50)):
        for s in range(S):
            for t in range(T):
                surf = cond_samples[i, s, t]
                if surf.max() > 0.99 or surf.min() < 0.001:
                    explosion_count += 1
                for row in range(5):
                    for col in range(4):
                        if surf[row, col + 1] > surf[row, col] + 0.01:
                            cal_arb_count += 1
                for col in range(5):
                    for row in range(1, 4):
                        if surf[row, col] > (surf[row - 1, col] + surf[row + 1, col]) / 2 + 0.01:
                            but_arb_count += 1

    total_checked = min(N, 50) * S * T
    expl_rate = explosion_count / total_checked
    cal_arb_rate = cal_arb_count / (total_checked * 5 * 4)
    but_arb_rate = but_arb_count / (total_checked * 5 * 3)
    s1_pass = expl_rate < 0.05 and cal_arb_rate < 0.10 and but_arb_rate < 0.25

    # ── Suite 2: CI Coverage ──
    ci_pass_horizon = 0
    ci_pass_cell = 0
    worst_cell_cov = 1.0

    for h in range(T):
        gen_h = cond_samples[:, :, h, :, :]
        gt_h = ground_truth[:, h, :, :]
        lo = np.percentile(gen_h, 5, axis=1)
        hi = np.percentile(gen_h, 95, axis=1)
        covered = (gt_h >= lo) & (gt_h <= hi)
        cov_rate = covered.mean()
        if cov_rate >= 0.85:
            ci_pass_horizon += 1

    for r in range(5):
        for c in range(5):
            gen_cell = cond_samples[:, :, :, r, c]
            gt_cell = ground_truth[:, :, r, c]
            lo = np.percentile(gen_cell, 5, axis=1)
            hi = np.percentile(gen_cell, 95, axis=1)
            covered = (gt_cell >= lo) & (gt_cell <= hi)
            cell_cov = covered.mean()
            worst_cell_cov = min(worst_cell_cov, cell_cov)
            if cell_cov >= 0.85:
                ci_pass_cell += 1

    s2_pass = ci_pass_horizon >= 25 and worst_cell_cov >= 0.80

    # ── Suite 3: Conditionality ──
    rv = np.array([np.std(rets[i:i + H]) for i in range(test_start, test_start + n_windows)])
    turb_thresh = np.percentile(rv, 80)
    calm_thresh = np.percentile(rv, 20)
    turb_mask = rv > turb_thresh
    calm_mask = rv < calm_thresh

    if turb_mask.sum() > 0 and calm_mask.sum() > 0:
        turb_spread = cond_samples[turb_mask].std(axis=1).mean()
        calm_spread = cond_samples[calm_mask].std(axis=1).mean()
        turb_calm_ratio = turb_spread / (calm_spread + 1e-8)

        ensemble_mean = cond_samples.mean(axis=1)
        mae_cond = np.abs(ensemble_mean - ground_truth).mean()
        overall_mean = ground_truth.mean(axis=0)
        mae_uncond = np.abs(ground_truth - overall_mean[None]).mean()
        mae_reduction = 1 - mae_cond / mae_uncond
    else:
        turb_calm_ratio = 1.0
        mae_reduction = 0.0

    s3_pass = turb_calm_ratio > 1.15

    # ── Suite 4: Time Series Properties ──
    gen_changes = np.diff(cond_samples[:, 0, :, :, :], axis=1).reshape(-1)
    gt_changes = np.diff(ground_truth, axis=1).reshape(-1)

    def acf(series, max_lag=10):
        m = np.mean(series)
        v = np.var(series)
        if v == 0:
            return np.zeros(max_lag + 1)
        return [1.0] + [np.mean((series[:-l] - m) * (series[l:] - m)) / v for l in range(1, max_lag + 1)]

    gen_acf = acf(gen_changes)
    gt_acf = acf(gt_changes)
    acf_corr = np.corrcoef(gen_acf, gt_acf)[0, 1]

    gen_ch_flat = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch_flat = np.diff(ground_truth, axis=1).reshape(-1, 25)
    gen_kurt = kurtosis(gen_ch_flat.flatten(), fisher=True)
    gt_kurt = kurtosis(gt_ch_flat.flatten(), fisher=True)
    kurt_ratio = gen_kurt / (gt_kurt + 1e-6)

    s4_pass = acf_corr > 0.7 and 0.5 <= kurt_ratio <= 2.0

    # ── Suite 5: Growing Uncertainty ──
    spreads = []
    for h in range(T):
        spread_h = cond_samples[:, :, h].std(axis=1).mean()
        spreads.append(spread_h)
    spreads = np.array(spreads)

    mono_count = sum(1 for i in range(len(spreads) - 1) if spreads[i + 1] >= spreads[i] * 0.97)
    growing = mono_count >= 20
    s5_pass = growing

    # ── Suite 6: Cointegration ──
    coint_pairs = 0
    coint_total = 0
    for i in range(min(N, 30)):
        sample_path = cond_samples[i, 0, :, :, :].reshape(T, 25)
        for c1 in range(0, 25, 5):
            for c2 in range(c1 + 1, min(c1 + 5, 25)):
                try:
                    _, pval, _ = coint(sample_path[:, c1], sample_path[:, c2])
                    if pval < 0.05:
                        coint_pairs += 1
                    coint_total += 1
                except Exception:
                    coint_total += 1

    coint_rate = coint_pairs / max(coint_total, 1)
    s6_pass = coint_rate > 0.50

    # ── Suite 7: Regime Coverage ──
    if turb_mask.sum() > 5 and calm_mask.sum() > 5:
        regime_pass = 0
        regime_detail = {}
        for mask, name in [(turb_mask, "turb"), (calm_mask, "calm")]:
            regime_samp = cond_samples[mask]
            regime_gt = ground_truth[mask]
            cell_pass = 0
            for r in range(5):
                for c in range(5):
                    gen_cell = regime_samp[:, :, :, r, c]
                    gt_cell = regime_gt[:, :, r, c]
                    lo = np.percentile(gen_cell, 5, axis=1)
                    hi = np.percentile(gen_cell, 95, axis=1)
                    covered = (gt_cell >= lo) & (gt_cell <= hi)
                    if covered.mean() >= 0.80:
                        cell_pass += 1
            if cell_pass >= 20:
                regime_pass += 1
            regime_detail[name] = cell_pass
        s7_pass = regime_pass == 2
    else:
        s7_pass = False
        regime_detail = {"turb": 0, "calm": 0}

    # ── Suite 8: Distributional ──
    gen_ch = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch = np.diff(ground_truth, axis=1).reshape(-1, 25)
    ks_daily_pass = sum(1 for c in range(25) if ks_2samp(gen_ch[:, c], gt_ch[:, c])[0] < 0.15)

    gen_levels = cond_samples[:, 0, -1].reshape(-1, 25)
    gt_levels = ground_truth[:, -1].reshape(-1, 25)
    ks_level_pass = sum(1 for c in range(25) if ks_2samp(gen_levels[:, c], gt_levels[:, c])[0] < 0.15)

    gen_median = np.median(cond_samples, axis=1)
    bias = (gen_median - ground_truth).mean()

    s8_pass = ks_daily_pass >= 15 and ks_level_pass >= 15

    # ── Suite 9: Cross-Cell Correlation ──
    gen_ch2 = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch2 = np.diff(ground_truth, axis=1).reshape(-1, 25)
    gen_corr = np.corrcoef(gen_ch2.T)
    gt_corr = np.corrcoef(gt_ch2.T)

    er_gen = eff_rank(gen_corr)
    er_gt = eff_rank(gt_corr)
    rank_ratio = er_gen / er_gt
    corr_ratio = np.abs(gen_corr).mean() / (np.abs(gt_corr).mean() + 1e-6)
    frob = np.linalg.norm(gen_corr - gt_corr, 'fro')

    gt_vecs = np.linalg.eigh(gt_corr)[1][:, ::-1]
    gen_vecs = np.linalg.eigh(gen_corr)[1][:, ::-1]
    pc_aligns = [abs(float(np.dot(gt_vecs[:, i], gen_vecs[:, i]))) for i in range(5)]

    s9_pass = rank_ratio >= 0.50 and corr_ratio >= 0.60

    # ── Assemble ──
    suites = [s1_pass, s2_pass, s3_pass, s4_pass, s5_pass, s6_pass, s7_pass, s8_pass, s9_pass]
    suite_names = [
        "Surface Validity", "CI Coverage", "Conditionality", "Time Series",
        "Growing Uncertainty", "Cointegration", "Regime Coverage",
        "Distributional", "Cross-Cell Correlation",
    ]
    pass_count = sum(suites)

    results = {
        "pass_count": pass_count,
        "total_suites": 9,
        "suites": {name: bool(passed) for name, passed in zip(suite_names, suites)},
        "metrics": {
            "explosion_rate": round(float(expl_rate), 4),
            "calendar_arb_rate": round(float(cal_arb_rate), 4),
            "butterfly_arb_rate": round(float(but_arb_rate), 4),
            "ci_horizon_pass": int(ci_pass_horizon),
            "ci_worst_cell": round(float(worst_cell_cov), 4),
            "ci_cell_pass": int(ci_pass_cell),
            "turb_calm_ratio": round(float(turb_calm_ratio), 4),
            "mae_reduction": round(float(mae_reduction), 4),
            "acf_correlation": round(float(acf_corr), 4),
            "kurt_ratio": round(float(kurt_ratio), 4),
            "spread_h1": round(float(spreads[0]), 5),
            "spread_h30": round(float(spreads[-1]), 5),
            "monotonic_transitions": int(mono_count),
            "coint_rate": round(float(coint_rate), 4),
            "ks_daily_pass": int(ks_daily_pass),
            "ks_level_pass": int(ks_level_pass),
            "median_bias": round(float(bias), 5),
            "eff_rank_gen": round(float(er_gen), 4),
            "eff_rank_gt": round(float(er_gt), 4),
            "rank_ratio": round(float(rank_ratio), 4),
            "corr_ratio": round(float(corr_ratio), 4),
            "frobenius": round(float(frob), 4),
            "pc_alignments": [round(float(a), 4) for a in pc_aligns],
            "regime_detail": regime_detail,
        },
    }
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20)
    args = parser.parse_args()

    device = args.device
    n_samples = args.n_samples
    max_batches = args.max_batches
    seeds = [42, 43, 44]

    model_path = "models/backfill/flow_153a/final_model.pt"
    encoder_path = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"

    output_dir = Path("results/validations/2026-03-24/analysis/153a_multiseed")
    output_dir.mkdir(parents=True, exist_ok=True)
    verif_dir = Path("results/validations/2026-03-24/verification_results")
    verif_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model (once) ──
    print(f"Loading model from {model_path}")
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    velocity_net = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    velocity_net.load_state_dict(ckpt["model_state_dict"])
    velocity_net.to(device).eval()

    encoder, cond_dim = load_encoder(encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False

    train_mean = ckpt["train_mean"]
    train_std = ckpt["train_std"]
    n_steps = cfg.get("n_steps", 8)

    # ── Load data ──
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    rets = data["ret"]
    H, T = 30, 30
    test_start = 4540

    test_windows = []
    for i in range(test_start, len(surfaces) - H - T + 1):
        history = surfaces[i:i + H]
        future = surfaces[i + H:i + H + T]
        test_windows.append((history, future))

    n_windows = min(max_batches * 8, len(test_windows))
    batch_size = 8
    print(f"Test windows: {n_windows}, samples: {n_samples}, seeds: {seeds}")

    # ── Run per-seed ──
    all_seed_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        t0 = time.time()

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        all_samples = []
        all_gt = []
        all_hist = []

        with torch.no_grad():
            for i in range(0, n_windows, batch_size):
                batch_end = min(i + batch_size, n_windows)
                batch_hist = np.array([test_windows[j][0] for j in range(i, batch_end)], dtype=np.float32)
                batch_gt = np.array([test_windows[j][1] for j in range(i, batch_end)], dtype=np.float32)

                hist_t = torch.from_numpy(batch_hist).to(device)
                samp = generate_samples(velocity_net, encoder, hist_t, n_samples,
                                        n_steps, train_mean, train_std, device)

                all_samples.append(samp.cpu().numpy())
                all_gt.append(batch_gt)
                all_hist.append(batch_hist)

                batch_num = i // batch_size + 1
                total_batches = (n_windows + batch_size - 1) // batch_size
                if batch_num % 5 == 0:
                    print(f"  Batch {batch_num}/{total_batches}")

        cond_samples = np.concatenate(all_samples)
        ground_truth = np.concatenate(all_gt)
        history_arr = np.concatenate(all_hist)

        print(f"  Generated: samples={cond_samples.shape}, GT={ground_truth.shape}")

        results = evaluate_9_suites(cond_samples, ground_truth, history_arr, rets, test_start, n_windows)
        results["seed"] = seed
        results["generation_time_s"] = round(time.time() - t0, 1)

        all_seed_results[str(seed)] = results

        # Print summary
        print(f"\n  Seed {seed}: {results['pass_count']}/9 suites PASS")
        for name, passed in results["suites"].items():
            print(f"    {'PASS' if passed else 'FAIL'}: {name}")

        # Save individual seed result
        with open(output_dir / f"seed_{seed}.json", "w") as f:
            json.dump(make_serial(results), f, indent=2)

    # ── Comparison Table ──
    print(f"\n\n{'='*80}")
    print("MULTI-SEED COMPARISON TABLE")
    print(f"{'='*80}")

    suite_names = [
        "Surface Validity", "CI Coverage", "Conditionality", "Time Series",
        "Growing Uncertainty", "Cointegration", "Regime Coverage",
        "Distributional", "Cross-Cell Correlation",
    ]

    # Header
    print(f"\n{'Suite':<25} | {'Seed 42':^10} | {'Seed 43':^10} | {'Seed 44':^10} | Threshold")
    print("-" * 85)

    for sn in suite_names:
        vals = []
        for seed in seeds:
            passed = all_seed_results[str(seed)]["suites"][sn]
            vals.append("PASS" if passed else "FAIL")
        print(f"{sn:<25} | {vals[0]:^10} | {vals[1]:^10} | {vals[2]:^10} |")

    print("-" * 85)
    for seed in seeds:
        pc = all_seed_results[str(seed)]["pass_count"]
        print(f"{'TOTAL':<25} | " if seed == 42 else f"{'':25} | ", end="")
    total_line = " | ".join(f"{all_seed_results[str(s)]['pass_count']}/9" for s in seeds)
    print(f"{'Total Pass':<25} | {total_line.replace(' | ', '    |    ')}")

    # Key borderline metrics
    print(f"\n{'Borderline Metric':<30} | {'Seed 42':>10} | {'Seed 43':>10} | {'Seed 44':>10} | {'Threshold':>10}")
    print("-" * 85)

    borderline_keys = [
        ("calendar_arb_rate", "Cal Arb Rate", "<0.10"),
        ("coint_rate", "Cointegration Rate", ">0.50"),
        ("ks_daily_pass", "KS Daily (of 25)", ">=15"),
        ("ks_level_pass", "KS Levels (of 25)", ">=15"),
        ("ci_worst_cell", "CI Worst Cell", ">=0.80"),
        ("ci_horizon_pass", "CI Horizon Pass (of 30)", ">=25"),
        ("turb_calm_ratio", "Turb/Calm Ratio", ">1.15"),
        ("kurt_ratio", "Kurtosis Ratio", "0.5-2.0"),
        ("rank_ratio", "Rank Ratio", ">=0.50"),
        ("corr_ratio", "Corr Ratio", ">=0.60"),
        ("monotonic_transitions", "Mono Transitions (of 29)", ">=20"),
        ("median_bias", "Median Bias", "-"),
        ("explosion_rate", "Explosion Rate", "<0.05"),
        ("butterfly_arb_rate", "Butterfly Arb Rate", "<0.25"),
    ]

    for key, label, thresh in borderline_keys:
        vals = [all_seed_results[str(s)]["metrics"][key] for s in seeds]
        if isinstance(vals[0], int):
            val_strs = [f"{v:>10d}" for v in vals]
        else:
            val_strs = [f"{v:>10.4f}" for v in vals]
        print(f"{label:<30} | {val_strs[0]} | {val_strs[1]} | {val_strs[2]} | {thresh:>10}")

    # ── Stability Assessment ──
    print(f"\n\n{'='*60}")
    print("STABILITY ASSESSMENT")
    print(f"{'='*60}")

    # Check which suites are stable across seeds
    stable_pass = []
    stable_fail = []
    unstable = []
    for sn in suite_names:
        results_list = [all_seed_results[str(s)]["suites"][sn] for s in seeds]
        if all(results_list):
            stable_pass.append(sn)
        elif not any(results_list):
            stable_fail.append(sn)
        else:
            unstable.append((sn, results_list))

    print(f"\nStable PASS (all 3 seeds):  {len(stable_pass)}")
    for sn in stable_pass:
        print(f"  - {sn}")

    print(f"\nStable FAIL (all 3 seeds):  {len(stable_fail)}")
    for sn in stable_fail:
        print(f"  - {sn}")

    print(f"\nUnstable (varies by seed):  {len(unstable)}")
    for sn, results_list in unstable:
        pattern = ", ".join(f"s{s}={'P' if r else 'F'}" for s, r in zip(seeds, results_list))
        print(f"  - {sn}: [{pattern}]")

    min_pass = min(all_seed_results[str(s)]["pass_count"] for s in seeds)
    max_pass = max(all_seed_results[str(s)]["pass_count"] for s in seeds)
    print(f"\nPass count range: {min_pass}-{max_pass}/9")

    # ── Build verification_result.json ──
    verification = {
        "verification_type": "multi_seed_evaluation",
        "model": "flow_153a",
        "model_path": model_path,
        "seeds": seeds,
        "n_samples": n_samples,
        "n_windows": n_windows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_seed_results": make_serial(all_seed_results),
        "stability": {
            "stable_pass": stable_pass,
            "stable_fail": stable_fail,
            "unstable": [(sn, [bool(r) for r in rl]) for sn, rl in unstable],
            "pass_count_range": [min_pass, max_pass],
            "pass_counts": {str(s): all_seed_results[str(s)]["pass_count"] for s in seeds},
        },
        "comparison_table": {
            sn: {str(s): all_seed_results[str(s)]["suites"][sn] for s in seeds}
            for sn in suite_names
        },
        "borderline_metrics": {
            key: {str(s): all_seed_results[str(s)]["metrics"][key] for s in seeds}
            for key, _, _ in borderline_keys
        },
        "verdict": (
            f"153a achieves {min_pass}-{max_pass}/9 across seeds {seeds}. "
            f"Stable PASS: {len(stable_pass)} suites. "
            f"Stable FAIL: {len(stable_fail)} suites. "
            f"Unstable: {len(unstable)} suites."
        ),
    }

    verif_path = verif_dir / "153a_multiseed.json"
    with open(verif_path, "w") as f:
        json.dump(make_serial(verification), f, indent=2)
    print(f"\nVerification result saved to {verif_path}")

    # Also save to analysis dir
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(make_serial(verification), f, indent=2)
    print(f"Comparison saved to {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
