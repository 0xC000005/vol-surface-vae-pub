#!/usr/bin/env python
"""
Diagnose whether v1.2-link / v1.2-both / v1.2-noreg learned-link α
differentiates regimes or collapsed to a constant.

Outputs:
  - α distribution stats (std, min, max) across val windows
  - Per-regime α mean (calm vs turb)
  - Regime separation: |mean(α|turb) - mean(α|calm)|
  - Pass/fail vs Stage-B thresholds (std > 0.05, separation > 0.02)
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    ap.add_argument("--test_start", type=int, default=4511)
    ap.add_argument("--val_size", type=int, default=441)
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import load_model
    from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows

    device = torch.device(args.device)
    m, payload = load_model(args.checkpoint, device)

    if m.emission_link is None:
        result = {"skipped": "variant has no learned link; α collapse N/A"}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print("No learned link in this variant; skipping")
        return

    # Load val windows
    raw = np.load(args.data_path)
    surf = torch.from_numpy(raw["surface"].astype(np.float32)).to(device)
    max_train_idx = args.test_start - 30 - 30
    val_idx = np.arange(max_train_idx - args.val_size, max_train_idx)[:args.n_windows]
    hist, _ = build_multistep_windows(val_idx, surf, 30, 30)

    # Regime labels from realized-variance proxy
    hist_np = hist.cpu().numpy()
    dhist = np.diff(hist_np.reshape(hist_np.shape[0], 30, 25), axis=1)
    rv = (dhist ** 2).mean(axis=(1, 2))
    q20, q80 = np.quantile(rv, [0.20, 0.80])
    calm_mask = rv <= q20
    turb_mask = rv >= q80

    # Collect α at step 0 from each val window
    m.eval()
    with torch.no_grad():
        # Use encode_history to get initial cond for each window
        hist_in = hist.to(device)
        cond_B, _ = m.encode_history(hist_in)   # (B, hidden)
        alpha_init = torch.sigmoid(m.emission_link.gate(cond_B))   # (B, D)

    alpha_np = alpha_init.cpu().numpy()
    alpha_flat = alpha_np.reshape(-1)

    result = {
        "checkpoint": args.checkpoint,
        "variant_name": payload.get("variant_name", "?"),
        "n_windows": int(hist.shape[0]),
        "n_calm": int(calm_mask.sum()),
        "n_turb": int(turb_mask.sum()),
        "alpha_overall_stats": {
            "mean": float(alpha_flat.mean()),
            "std": float(alpha_flat.std()),
            "min": float(alpha_flat.min()),
            "max": float(alpha_flat.max()),
        },
        "alpha_per_regime": {
            "calm_mean": float(alpha_np[calm_mask].mean()) if calm_mask.any() else None,
            "turb_mean": float(alpha_np[turb_mask].mean()) if turb_mask.any() else None,
        },
        "regime_separation": float(
            alpha_np[turb_mask].mean() - alpha_np[calm_mask].mean()
        ) if calm_mask.any() and turb_mask.any() else None,
    }

    # Pass/fail gates (design spec Stage B thresholds)
    result["gates"] = {
        "alpha_std_gt_0p05": result["alpha_overall_stats"]["std"] > 0.05,
        "abs_separation_gt_0p02": (
            abs(result["regime_separation"]) > 0.02 if result["regime_separation"] else False
        ),
    }
    result["alpha_collapsed"] = not (
        result["gates"]["alpha_std_gt_0p05"] or result["gates"]["abs_separation_gt_0p02"]
    )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
