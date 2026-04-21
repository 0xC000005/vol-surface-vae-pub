"""
251b time-homogeneity probe (and 251a control).

Fixes z_t and h; sweeps t_idx ∈ {0, 5, 10, 15, 20, 25, 29}. For a truly
time-homogeneous LatentSDE (cfg.latent_sde_time_embed == 0), drift and
diffusion outputs must be identical across all t_idx (floating-point noise
≤ 1e-6). For 251a (cfg.latent_sde_time_embed == 32), the time embedding
alone drives drift/diffusion divergence — this probe is how we reproduce
the amendment's 16.5× finding as a pre-flight hard gate.

Pass condition (251b):
  max |drift(t_i) - drift(t_0)| ≤ 1e-6  AND  same for diffusion.

Hard gate (251a control):
  max |step_out(t_i) - step_out(t_0)| / mean |step_out(t_0)| ≥ 10×
  (amendment reported ≥ 16.5×).

Reports JSON to --output_dir (or prints only if --output_dir absent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import load_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--t_probe_points", type=str, default="0,5,10,15,20,25,29")
    parser.add_argument("--n_samples", type=int, default=8, help="K (batch-multiplier)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, _ = load_model(args.checkpoint, device)
    model.eval()
    if model.latent_sde is None:
        print("ERROR: checkpoint has no LatentSDE. Probe requires 251a or 251b.")
        return 2

    cfg = model.cfg
    L = cfg.latent_dim
    K = args.n_samples
    bottleneck = cfg.bottleneck_dim
    time_embed_dim = cfg.latent_sde_time_embed
    is_stationary_config = time_embed_dim == 0

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  latent_dim={L}  bottleneck_dim={bottleneck}  "
          f"latent_sde_time_embed={time_embed_dim}  "
          f"expected: {'TIME-HOMOGENEOUS' if is_stationary_config else 'TIME-VARYING'}")

    # Fix z_t and h. Use a single sample (B=1), K samples — the probe varies t only.
    z_fixed = torch.randn(1, K, L, device=device)
    h_fixed = torch.randn(1, bottleneck, device=device)

    t_points = [int(x) for x in args.t_probe_points.split(",")]

    with torch.no_grad():
        outputs = {}
        for t_idx in t_points:
            drift, diff = model.latent_sde.step(z_fixed, t_idx, h_fixed)
            outputs[t_idx] = (drift.detach().cpu(), diff.detach().cpu())

    d0, g0 = outputs[t_points[0]]
    max_drift_diff = 0.0
    max_diff_diff = 0.0
    per_t = []
    for t_idx in t_points:
        d_t, g_t = outputs[t_idx]
        delta_d = (d_t - d0).abs().max().item()
        delta_g = (g_t - g0).abs().max().item()
        max_drift_diff = max(max_drift_diff, delta_d)
        max_diff_diff = max(max_diff_diff, delta_g)
        per_t.append({
            "t_idx": t_idx,
            "max_abs_drift_diff_vs_t0": delta_d,
            "max_abs_diffusion_diff_vs_t0": delta_g,
            "mean_abs_drift": d_t.abs().mean().item(),
            "mean_abs_diffusion": g_t.abs().mean().item(),
        })

    # For the 251a control: ratio vs t=0 magnitude.
    d0_mean = d0.abs().mean().item()
    g0_mean = g0.abs().mean().item()
    drift_ratio = max_drift_diff / max(d0_mean, 1e-12)
    diff_ratio = max_diff_diff / max(g0_mean, 1e-12)

    # Pass conditions.
    homo_pass = (max_drift_diff <= 1e-6) and (max_diff_diff <= 1e-6)
    control_10x = (drift_ratio >= 10.0) or (diff_ratio >= 10.0)

    if is_stationary_config:
        verdict = "HOMOGENEITY_PASS" if homo_pass else "HOMOGENEITY_FAIL"
    else:
        verdict = "CONTROL_10X_PASS" if control_10x else "CONTROL_10X_FAIL"

    summary = {
        "checkpoint": str(args.checkpoint),
        "latent_sde_time_embed": time_embed_dim,
        "expected_mode": "time-homogeneous" if is_stationary_config else "time-varying",
        "t_points": t_points,
        "max_drift_diff_vs_t0": max_drift_diff,
        "max_diffusion_diff_vs_t0": max_diff_diff,
        "drift_ratio_vs_t0_mean": drift_ratio,
        "diffusion_ratio_vs_t0_mean": diff_ratio,
        "per_t": per_t,
        "verdict": verdict,
    }

    print(f"max |drift(t) - drift(t0)| = {max_drift_diff:.3e}")
    print(f"max |diffusion(t) - diffusion(t0)| = {max_diff_diff:.3e}")
    print(f"drift ratio (max_diff / mean_abs at t0) = {drift_ratio:.3f}×")
    print(f"diffusion ratio (max_diff / mean_abs at t0) = {diff_ratio:.3f}×")
    print(f"Verdict: {verdict}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "time_homogeneity.json").write_text(json.dumps(summary, indent=2))
        print(f"Wrote {out_dir / 'time_homogeneity.json'}")

    # Exit code reflects the intended use as a gate.
    if is_stationary_config:
        return 0 if homo_pass else 1
    else:
        return 0 if control_10x else 1


if __name__ == "__main__":
    sys.exit(main())
