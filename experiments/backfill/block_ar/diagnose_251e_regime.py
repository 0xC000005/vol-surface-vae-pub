"""
251e regime diagnostic — 3-level causal chain for H1.

L1 Mechanism activation:
  - effective_rank(r_slow) across val windows
  - per-dim r_slow variance, min/max/mean
  - Classification AUC: can r_slow linearly separate calm vs turb (by RV median)?

L2 Causal-chain propagation (CRITICAL):
  - Counterfactual Λ test: for val windows, compute Λ = LoadingHead([h, r_slow_orig]).
    Also compute Λ' = LoadingHead([h, r_slow_swapped]) using r_slow from a different
    window (calm↔turb swap). Report ||ΔΛ||_F / ||Λ||_F.
  - Counterfactual D test: same pattern.

L3 Target metric pass-through:
  - Reference conditionality.turb_calm_ratio and regime_coverage.layer2/3 from eval.

Gate:
  - L1: r_slow eff_rank ≥ 4 (of regime_dim=8); AUC ≥ 0.60
  - L2: counterfactual ||ΔΛ||_F/||Λ||_F ≥ 0.05 (regime routes through heads)
       below 0.01 → routing failure (similar to 233a)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import load_model
from diffusion.block_ar.single_pass_ar import normalize_iv


def effective_rank(X: torch.Tensor) -> float:
    if X.numel() == 0:
        return 0.0
    X32 = X.float()
    try:
        s = torch.linalg.svdvals(X32)
    except RuntimeError:
        return 0.0
    s_sq = s.pow(2)
    p = s_sq / s_sq.sum().clamp(min=1e-12)
    H = -(p * (p.clamp(min=1e-12)).log()).sum()
    return float(torch.exp(H).item())


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--n_windows", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, device)
    model.eval()

    if model.regime_encoder is None:
        print("ERROR: model has no regime_encoder (use_regime=False)")
        return 2

    # Load validation data
    surfaces = np.load(args.data_path)["surface"].astype("float32")
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    # Build history windows (B, T_hist, D)
    hist_list = []
    for i in val_indices[: args.n_windows]:
        hist_list.append(surf_tensor[i:i + args.history_len].reshape(args.history_len, -1))
    history = torch.stack(hist_list, dim=0)  # (n_windows, T_hist, D)

    # Normalize for model input
    hist_norm = normalize_iv(history.unsqueeze(-1)).squeeze(-1) if history.dim() == 3 else normalize_iv(history)
    # In this codebase history is kept in [0,1] IV scale; normalize to [-1,1]
    hist_norm = history * 2.0 - 1.0

    # ----- L1: r_slow statistics -----
    h = model.encode_history(hist_norm)  # (N, bottleneck)
    r_slow = model.regime_encoder(hist_norm)  # (N, regime_dim)
    r_np = r_slow.cpu().numpy()

    eff_rank = effective_rank(r_slow)
    per_dim_std = r_slow.std(dim=0).cpu().numpy()
    r_mean = r_slow.mean(dim=0).cpu().numpy()

    # AUC: calm vs turb by realized vol
    rv = (history.diff(dim=1) ** 2).mean(dim=(1, 2)).cpu().numpy()
    rv_median = np.median(rv)
    labels = (rv > rv_median).astype(int)  # 1 = turb, 0 = calm
    # Linear probe: correlation of each r_slow dim with labels, take max |corr| as AUC proxy
    auc_proxy = 0.0
    for d in range(r_np.shape[1]):
        c = np.corrcoef(r_np[:, d], labels)[0, 1]
        if abs(c) > auc_proxy:
            auc_proxy = abs(c)
    # Better AUC: logistic regression on r_slow
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        clf = LogisticRegression(max_iter=1000)
        clf.fit(r_np, labels)
        pred = clf.predict_proba(r_np)[:, 1]
        auc = float(roc_auc_score(labels, pred))
    except Exception as e:
        auc = float("nan")
        print(f"(sklearn AUC skipped: {e})")

    # ----- L2: counterfactual Λ and D -----
    # Original: Lambda = LoadingHead([h, r_slow])
    h_ext = torch.cat([h, r_slow], dim=-1)
    Lambda_orig = model.loading_head(h_ext)  # (N, T, D, L)
    D_orig = model.idio_head(h_ext)          # (N, T, D)

    # Swap r_slow within extreme pairs: split by RV and swap calm<->turb
    # Take top-20 turb and bottom-20 calm
    sorted_idx = np.argsort(rv)
    calm_idx = sorted_idx[:20]
    turb_idx = sorted_idx[-20:]
    # Build swap: for calm windows, take r_slow from turb windows (and vice versa)
    swap_pairs = list(zip(calm_idx, turb_idx)) + list(zip(turb_idx, calm_idx))
    delta_lambda_rel = []
    delta_D_rel = []
    for orig_i, swap_i in swap_pairs:
        h_i = h[orig_i:orig_i + 1]
        r_swap = r_slow[swap_i:swap_i + 1]  # regime from swap window
        h_ext_swap = torch.cat([h_i, r_swap], dim=-1)
        L_orig_i = Lambda_orig[orig_i:orig_i + 1]
        L_swap = model.loading_head(h_ext_swap)
        D_orig_i = D_orig[orig_i:orig_i + 1]
        D_swap = model.idio_head(h_ext_swap)
        # Relative Frobenius
        d_L = (L_swap - L_orig_i).norm() / L_orig_i.norm().clamp(min=1e-12)
        d_D = (D_swap - D_orig_i).norm() / D_orig_i.norm().clamp(min=1e-12)
        delta_lambda_rel.append(float(d_L.item()))
        delta_D_rel.append(float(d_D.item()))

    dL_mean = float(np.mean(delta_lambda_rel))
    dL_std = float(np.std(delta_lambda_rel))
    dD_mean = float(np.mean(delta_D_rel))
    dD_std = float(np.std(delta_D_rel))

    # Gates
    l1_rank_gate = eff_rank >= 4.0
    l1_auc_gate = (auc >= 0.60) if not np.isnan(auc) else False
    l2_lambda_gate = dL_mean >= 0.05
    l2_D_gate = dD_mean >= 0.05

    summary = {
        "checkpoint": str(args.checkpoint),
        "regime_dim": int(r_slow.shape[1]),
        "L1_activation": {
            "effective_rank_r_slow": eff_rank,
            "per_dim_std": per_dim_std.tolist(),
            "mean_per_dim": r_mean.tolist(),
            "auc_calm_vs_turb": auc,
            "auc_proxy_corr": float(auc_proxy),
            "gate_rank_ge_4": bool(l1_rank_gate),
            "gate_auc_ge_0p60": bool(l1_auc_gate),
        },
        "L2_counterfactual": {
            "n_swap_pairs": len(swap_pairs),
            "delta_lambda_rel_mean": dL_mean,
            "delta_lambda_rel_std": dL_std,
            "delta_D_rel_mean": dD_mean,
            "delta_D_rel_std": dD_std,
            "gate_dL_ge_0p05": bool(l2_lambda_gate),
            "gate_dD_ge_0p05": bool(l2_D_gate),
        },
    }

    print(f"=== L1 Mechanism activation ===")
    print(f"  r_slow eff_rank: {eff_rank:.3f} / {r_slow.shape[1]} (gate ≥ 4)")
    print(f"  per-dim std range: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")
    print(f"  calm/turb linear AUC: {auc:.3f} (gate ≥ 0.60)")
    print(f"=== L2 Causal-chain propagation ===")
    print(f"  ||ΔΛ||_F/||Λ||_F swap (mean ± std): {dL_mean:.4f} ± {dL_std:.4f}  (gate ≥ 0.05)")
    print(f"  ||ΔD||_F/||D||_F swap (mean ± std): {dD_mean:.4f} ± {dD_std:.4f}  (gate ≥ 0.05)")
    print()
    verdict = "PASS" if (l1_rank_gate and l1_auc_gate and l2_lambda_gate) else "FAIL"
    print(f"Verdict (L1 rank AND L1 AUC AND L2 dL): {verdict}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regime_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'regime_diagnostic.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
