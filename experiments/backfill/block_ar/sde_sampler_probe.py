#!/usr/bin/env python
"""
RC16-H1-S1: Stochastic Interpolant SDE Sampler Probe (Exp 154a)

Test whether adding diffusion noise to 153a's trained ODE improves per-window
ensemble spread while preserving distributional quality.

Based on: "Probabilistic Forecasting with Stochastic Interpolants and Follmer
Processes" (2403.13724, ICML 2024). Key idea: the score can be approximated from
the trained velocity field, and diffusion g_s is tunable post-training.

The marginal-preserving SDE is:
  dx = [v(x,t) + 0.5 * g^2 * score(x,t)] dt + g * dW

For the score approximation, we use the Tweedie-style estimator:
  score(x_t, t) ≈ -[x_t - (1-t)*mean_pred] / (sigma_t^2)
where mean_pred = x_t + (1-t)*v(x_t, t) is the ODE prediction of x_1,
and sigma_t = sqrt(t*(1-t)) is the interpolant noise scale.

This is a PROBE — no retraining. Uses 153a's frozen velocity field.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/sde_sampler_probe.py \
        --model_path models/backfill/flow_153a/final_model.pt \
        --device cuda
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, kurtosis

sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv
)
from experiments.backfill.block_ar.train_oneshot_flow import evaluate_samples


def sde_sample(model, cond, n_steps, g_s, dim, device, n_samples=1,
               score_method="tweedie"):
    """Sample from the marginal-preserving SDE.

    dx = [v(x,t) + 0.5*g^2*score(x,t)] dt + g*dW

    Args:
        model: velocity network v(x_t, t, cond)
        cond: (B, cond_dim) condition vectors
        n_steps: number of integration steps
        g_s: diffusion coefficient (scalar or callable g(t))
        dim: output dimension (750)
        score_method: "tweedie" or "naive"
    """
    B = cond.shape[0]
    dt = 1.0 / n_steps
    x = torch.randn(B, dim, device=device)

    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)

        # Velocity prediction
        v = model(x, t, cond=cond)

        # Score approximation
        if score_method == "tweedie":
            # Tweedie estimator: predict x_1 from current state
            # x_1_hat = x_t + (1-t)*v (first-order ODE prediction)
            remaining_time = 1.0 - t_val
            if remaining_time > 0.01:  # avoid division by zero near t=1
                x1_hat = x + remaining_time * v
                # score = -(x_t - (1-t)*x_0_hat - t*x_1_hat) / sigma_t^2
                # For linear interpolant with sigma_t = 0:
                # x_t = (1-t)*x_0 + t*x_1, so x_0 = (x_t - t*x_1)/(1-t)
                # The score is the direction from x toward the predicted clean data
                # Simple approximation: score ≈ (x1_hat - x) / remaining_time
                score = (x1_hat - x) / (remaining_time + 1e-6)
            else:
                score = torch.zeros_like(x)
        elif score_method == "naive":
            # Naive: just use velocity direction as score proxy
            score = v
        else:
            score = torch.zeros_like(x)

        # Get g at this timestep
        if callable(g_s):
            g = g_s(t_val)
        else:
            g = g_s

        # SDE step: Euler-Maruyama
        drift = v + 0.5 * g**2 * score
        diffusion = g * math.sqrt(dt) * torch.randn_like(x)
        x = x + drift * dt + diffusion

    return x


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/flow_153a/final_model.pt")
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device

    # Load model
    ckpt = torch.load(args.model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    encoder, _ = load_encoder(args.encoder_path, device)
    mean_t = torch.from_numpy(ckpt["train_mean"]).float().to(device)
    std_t = torch.from_numpy(ckpt["train_std"]).float().to(device)

    # Data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, T, DIM = 30, 30, 750
    test_start = 4540

    # Build test windows
    test_windows = []
    for i in range(test_start, min(test_start + 160, len(surfaces) - H - T + 1)):
        test_windows.append((surfaces[i:i+H], surfaces[i+H:i+H+T]))

    # Also get training data for population metrics
    train_end = 4040
    train_hist = np.array([surfaces[i:i+H] for i in range(train_end - H - T + 1)],
                          dtype=np.float32)
    train_data = np.array([surfaces[i+H:i+H+T].reshape(-1)
                           for i in range(train_end - H - T + 1)], dtype=np.float32)

    # Pre-compute training conditions
    train_conds = []
    with torch.no_grad():
        for i in range(0, len(train_hist), 256):
            bh = torch.from_numpy(train_hist[i:i+256]).to(device)
            c = encoder(normalize_iv(bh))
            train_conds.append(c.cpu().detach().numpy())
    train_conds = np.concatenate(train_conds)

    print("=" * 60)
    print("RC16-H1-S1: Stochastic Interpolant SDE Sampler Probe (154a)")
    print("=" * 60)
    print(f"  Test windows: {len(test_windows)}")
    print(f"  Using 153a final_model (ep{ckpt['epoch']})")

    n_steps = 8
    n_test = min(30, len(test_windows))  # Quick probe on 30 windows
    n_samples = 50

    # === Baseline: ODE (g=0) ===
    print("\n--- Baseline: ODE (g=0) ---")
    ode_ci_covs = []
    ode_spreads = []
    with torch.no_grad():
        for w in range(n_test):
            hist = torch.from_numpy(test_windows[w][0][None].astype(np.float32)).to(device)
            gt = test_windows[w][1]
            cond = encoder(normalize_iv(hist))

            samples = []
            for s in range(n_samples):
                x = torch.randn(1, DIM, device=device)
                dt = 1.0 / n_steps
                for step in range(n_steps):
                    t = torch.full((1,), step * dt, device=device)
                    x = x + model(x, t, cond=cond) * dt
                samp = (x * std_t + mean_t).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
                samples.append(samp)
            samples = np.array(samples)

            lo = np.percentile(samples, 5, axis=0)
            hi = np.percentile(samples, 95, axis=0)
            covered = (gt >= lo) & (gt <= hi)
            ode_ci_covs.append(covered.mean())
            ode_spreads.append(samples[:, 0].std(axis=0).mean())

    ode_ci = np.mean(ode_ci_covs)
    ode_spread = np.mean(ode_spreads)
    print(f"  ODE CI: {ode_ci:.3f}, spread h1: {ode_spread:.5f}")

    # === SDE sweep over g_s values ===
    results_by_g = {}
    for g_val in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
        print(f"\n--- SDE g={g_val} ---")
        sde_ci_covs = []
        sde_spreads = []
        with torch.no_grad():
            for w in range(n_test):
                hist = torch.from_numpy(test_windows[w][0][None].astype(np.float32)).to(device)
                gt = test_windows[w][1]
                cond = encoder(normalize_iv(hist))

                samples = []
                for s in range(n_samples):
                    x = sde_sample(model, cond, n_steps, g_val, DIM, device,
                                   score_method="tweedie")
                    samp = (x * std_t + mean_t).clamp(0, 1).cpu().numpy().reshape(T, 5, 5)
                    samples.append(samp)
                samples = np.array(samples)

                lo = np.percentile(samples, 5, axis=0)
                hi = np.percentile(samples, 95, axis=0)
                covered = (gt >= lo) & (gt <= hi)
                sde_ci_covs.append(covered.mean())
                sde_spreads.append(samples[:, 0].std(axis=0).mean())

        sde_ci = np.mean(sde_ci_covs)
        sde_spread = np.mean(sde_spreads)
        print(f"  SDE CI: {sde_ci:.3f}, spread h1: {sde_spread:.5f}")

        # Population metrics at this g (512 samples with random conditions)
        all_samp = []
        with torch.no_grad():
            for si in range(0, 512, 64):
                eb = min(64, 512 - si)
                idx = np.random.choice(len(train_conds), eb, replace=True)
                c = torch.from_numpy(train_conds[idx]).to(device)
                x = sde_sample(model, c, n_steps, g_val, DIM, device,
                               score_method="tweedie")
                samp = (x * std_t + mean_t).clamp(0, 1).cpu().numpy()
                all_samp.append(samp)
        pop_samples = np.concatenate(all_samp)
        m = evaluate_samples(pop_samples, train_data)

        print(f"  Pop: eff_rank={m['eff_rank']:.2f} PC1={m['pc1']:.3f} "
              f"PC2={m['pc2']:.3f} KS={m['ks_pass']}/25 kurt={m['kurt_ratio']:.3f} "
              f"frob={m['frob']:.1f}")

        results_by_g[str(g_val)] = {
            "g": g_val,
            "ci_coverage": round(float(sde_ci), 4),
            "spread_h1": round(float(sde_spread), 5),
            "eff_rank": round(float(m['eff_rank']), 2),
            "pc1": round(float(m['pc1']), 4),
            "pc2": round(float(m['pc2']), 4),
            "ks_pass": int(m['ks_pass']),
            "kurt_ratio": round(float(m['kurt_ratio']), 3),
            "frob": round(float(m['frob']), 1),
        }

    # Save results
    output = {
        "experiment": "154a_sde_probe",
        "model": args.model_path,
        "n_test_windows": n_test,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "score_method": "tweedie",
        "baseline_ode": {
            "ci_coverage": round(float(ode_ci), 4),
            "spread_h1": round(float(ode_spread), 5),
        },
        "results_by_g": results_by_g,
        "kill_condition": "No g gives CI > 30%",
        "pass_condition": "Some g gives CI > 50% without KS < 15/25",
    }

    # Determine pass/fail
    best_g = None
    best_ci = 0
    for g_str, r in results_by_g.items():
        if r["ci_coverage"] > best_ci and r["ks_pass"] >= 15:
            best_ci = r["ci_coverage"]
            best_g = g_str
    output["best_g"] = best_g
    output["best_ci_with_quality"] = round(best_ci, 4)
    output["passes_kill"] = best_ci > 0.30
    output["passes_target"] = best_ci > 0.50

    out_dir = Path("results/validations/2026-03-24/analysis/154a_sde_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sde_probe_results.json", "w") as f:
        json.dump(make_serial(output), f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  ODE baseline: CI={ode_ci:.3f}, spread={ode_spread:.5f}")
    if best_g:
        r = results_by_g[best_g]
        print(f"  Best SDE (g={best_g}): CI={r['ci_coverage']:.3f}, "
              f"spread={r['spread_h1']:.5f}, KS={r['ks_pass']}/25")
    else:
        print(f"  No g achieves KS >= 15/25")
    print(f"  Kill condition (CI > 30%): {'PASS' if output['passes_kill'] else 'FAIL'}")
    print(f"  Target (CI > 50% + KS >= 15): {'PASS' if output['passes_target'] else 'FAIL'}")
    print(f"\nResults saved to {out_dir}/sde_probe_results.json")


if __name__ == "__main__":
    main()
