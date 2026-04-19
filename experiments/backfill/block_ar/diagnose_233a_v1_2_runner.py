#!/usr/bin/env python
"""Unified v1.2 diagnostic runner — wraps v1 diagnostic logic for v1.2 checkpoints."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant_name", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import torch
    from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import load_model
    from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows

    device = torch.device(args.device)
    model, payload = load_model(args.checkpoint, device)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    results = {"variant_name": args.variant_name, "checkpoint": args.checkpoint}

    # --- Shared data: load val windows once ---
    try:
        raw = np.load("data/vol_surface_with_ret.npz")
        surf = torch.from_numpy(raw["surface"].astype(np.float32)).to(device)
        val_idx = np.arange(4070, 4070 + args.n_windows)
        hist, future = build_multistep_windows(val_idx, surf, 30, 30)
    except Exception as e:
        results["data_error"] = str(e)
        with open(out_dir / "_diagnostic_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
        return

    # --- 1. FiLM logit distribution ---
    try:
        model.eval()
        with torch.no_grad():
            state = model.init_slow_state(hist.reshape(args.n_windows, 30, 25).to(device))
            if hasattr(model, "use_v1_film_pipe") and model.use_v1_film_pipe:
                film_out = model.film(state["s"], state["lam"])
            else:
                film_out = model.film(state["h_slow"])
        logit = film_out["p_jump_logit"]
        results["film_logit_std"] = float(logit.std().item())
        results["film_logit_range"] = [float(logit.min().item()), float(logit.max().item())]
        # Match field name expected by comparator
        results["logit_std_final_seed_42"] = results["film_logit_std"]
    except Exception as e:
        results["film_error"] = str(e)

    # --- 2. Slow-state discrimination (h_slow PC1 AUC on regime) ---
    try:
        hist_np = hist.cpu().numpy().reshape(args.n_windows, 30, 25)
        dhist = np.diff(hist_np, axis=1)
        rv = (dhist ** 2).mean(axis=(1, 2))
        q20, q80 = np.quantile(rv, [0.20, 0.80])
        calm = rv <= q20
        turb = rv >= q80
        with torch.no_grad():
            state = model.init_slow_state(hist.reshape(args.n_windows, 30, 25).to(device))
            h_slow_all = state["h_slow"].cpu().numpy()
        from numpy.linalg import svd
        hs_centered = h_slow_all - h_slow_all.mean(axis=0, keepdims=True)
        u, s, vh = svd(hs_centered, full_matrices=False)
        pc1 = hs_centered @ vh[0]
        if calm.sum() >= 3 and turb.sum() >= 3:
            # Compute AUC: probability that a random turb PC1 > random calm PC1
            calm_pc1 = pc1[calm]
            turb_pc1 = pc1[turb]
            hits = 0
            total = 0
            for ci in calm_pc1:
                for ti in turb_pc1:
                    if ti > ci:
                        hits += 1
                    elif ti == ci:
                        hits += 0.5
                    total += 1
            auc = hits / total if total > 0 else 0.5
            results["h_slow_pc1_auc_seed_42"] = float(max(auc, 1 - auc))
        else:
            results["h_slow_pc1_auc_seed_42"] = None
    except Exception as e:
        results["slow_state_error"] = str(e)

    # --- 3. lag-1 autocorr of deltas at h=29 ---
    try:
        model.eval()
        hist_dev = hist.reshape(args.n_windows, 30, 25).to(device)
        with torch.no_grad():
            out = model.forward_full(hist_dev, future=None, n_members=8, n_steps=30,
                                     p_gt_feedback=0.0, return_teacher_h=False)
        samples = out["samples"]
        deltas = samples[:, :, 1:, :] - samples[:, :, :-1, :]  # (B, K, 29, D)
        # lag-1 autocorr of mean-pooled deltas across timesteps at h=29 (step index 28)
        # Use last 10 steps for stable autocorr estimate
        d_tail = deltas[:, :, -10:, :].mean(dim=-1)  # (B, K, 10)
        d0 = d_tail[:, :, :-1].flatten()
        d1 = d_tail[:, :, 1:].flatten()
        d0_c = d0 - d0.mean()
        d1_c = d1 - d1.mean()
        denom = (d0_c.std() * d1_c.std())
        autocorr = float((d0_c * d1_c).mean() / denom) if denom > 0 else 0.0
        results["model_v1_2"] = {"lag1_autocorr_h29": autocorr}
    except Exception as e:
        results["ar_compounding_error"] = str(e)

    # --- 4. Regime breakdown (read from suite.json if available) ---
    suite_path = Path(f"results/block_ar/233a_v1_2/{args.variant_name}_s42/suite.json")
    if suite_path.exists():
        try:
            with open(suite_path) as f:
                suite = json.load(f)
            cond = suite.get("conditionality", {}).get("per_regime_conditionality", {})
            results["calm_avg_wr"] = cond.get("calm", {}).get("avg_width_ratio")
            results["turb_avg_wr"] = cond.get("turb", {}).get("avg_width_ratio")
            results["regime_inversion_detected"] = bool(
                (results["calm_avg_wr"] or 1.0) > 1.05 and (results["turb_avg_wr"] or 1.0) < 0.95
            )
        except Exception as e:
            results["regime_breakdown_error"] = str(e)
    else:
        results["regime_inversion_detected"] = None

    # --- 5. α-collapse (delegate to emission_link diagnostic if learned link present) ---
    if hasattr(model, "emission_link") and model.emission_link is not None:
        try:
            subprocess.run([
                "python", "experiments/backfill/block_ar/diagnose_233a_v1_2_emission_link.py",
                "--checkpoint", args.checkpoint,
                "--output_json", str(out_dir / "_alpha.json"),
                "--n_windows", str(args.n_windows),
                "--device", args.device,
            ], check=False)
            alpha_path = out_dir / "_alpha.json"
            if alpha_path.exists():
                with open(alpha_path) as f:
                    alpha_result = json.load(f)
                # Lift alpha stats up into main summary for comparator convenience
                results["alpha_overall_stats"] = alpha_result.get("alpha_overall_stats")
        except Exception as e:
            results["emission_link_error"] = str(e)

    # Write diagnostic summary (field names keyed to DIAGNOSTIC_FIELDS in comparator)
    # Also write per-diagnostic JSON files so comparator can find them at expected paths
    # (Comparator expects _diagnostic_film_collapse.json, _diagnostic_slow_state.json, etc.)
    with open(out_dir / "_diagnostic_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Fan out individual diagnostic JSONs expected by comparator
    with open(out_dir / "_diagnostic_film_collapse.json", "w") as f:
        json.dump({"logit_std_final_seed_42": results.get("film_logit_std")}, f, indent=2)
    with open(out_dir / "_diagnostic_emission_link.json", "w") as f:
        json.dump({"alpha_overall_stats": results.get("alpha_overall_stats", {})}, f, indent=2)
    with open(out_dir / "_diagnostic_ar_compounding.json", "w") as f:
        json.dump({"model_v1_2": results.get("model_v1_2", {})}, f, indent=2)
    with open(out_dir / "_diagnostic_slow_state.json", "w") as f:
        json.dump({"h_slow_pc1_auc_seed_42": results.get("h_slow_pc1_auc_seed_42")}, f, indent=2)
    with open(out_dir / "_diagnostic_regime_breakdown.json", "w") as f:
        json.dump({"regime_inversion_detected": results.get("regime_inversion_detected")}, f, indent=2)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
