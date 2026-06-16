"""Stage-C verdict for the risk_context oracle-injection probe.

On the co-trained slot-on checkpoint:
  - non-regression sanity: probe@natural-risk_context vs 734a (finite, comparable spread);
  - ORACLE injection: force the 4-d risk-state to extreme +/- activity (in-manifold via
    risk_context_proj), measure per-channel separation between +activity and -activity rollouts;
  - NULL band: repeat-sample spread at natural risk_context (different seed) = the noise floor;
  - VERDICT: injection separation clearly above the null band => the slot steers (activity).
"""
from __future__ import annotations

import numpy as np
import torch

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import load_model

PROBE = "experiments/backfill/block_ar/nl_scenario_demo_outputs/734a_riskslot_probe_20260616/best_model.pt"
BASE = "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt"
ARRAYS = "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
DEVICE = torch.device("cpu")
N_WIN, N_SAMP = 8, 16


def _inputs(idx0=0):
    a = np.load(ARRAYS, allow_pickle=True)
    sl = slice(idx0, idx0 + N_WIN)
    t = lambda x: torch.tensor(np.asarray(a[x][sl], dtype=np.float32), device=DEVICE)
    return t("history_level"), t("history_norm"), t("center"), t("scale"), t("drift_feature")


def _sample(model, ins, seed):
    torch.manual_seed(seed)
    hlv, hni, c, s, d = ins
    with torch.no_grad():
        return model.sample_batched(hlv, hni, c, s, n_samples=N_SAMP, n_steps=30, drift_feature=d)


def _inject(model, z_value):
    """Force risk-state = z_value (z-space, 4-d) -> in-manifold risk_context via the proj."""
    pred = torch.full((1, int(model.cfg.risk_state_dim)), float(z_value), device=DEVICE)
    ctx = model.risk_context_proj(pred)
    def _f(*_a, **_k):
        return pred, ctx
    model._risk_state_from_history = _f  # type: ignore[assignment]


def main() -> None:
    ins = _inputs()
    probe, _ = load_model(PROBE, DEVICE); probe.eval()
    base, _ = load_model(BASE, DEVICE); base.eval()

    # --- non-regression sanity: probe@natural vs 734a ---
    out_probe = _sample(probe, ins, seed=11)
    out_base = _sample(base, ins, seed=11)
    finite = float(torch.isfinite(out_probe).float().mean())
    drift = float(torch.mean(torch.abs(out_probe - out_base)))
    probe_std = float(out_probe.std()); base_std = float(out_base.std())

    # --- null band: repeat-sample spread at natural risk_context (two seeds) ---
    n1 = _sample(probe, ins, seed=101)
    n2 = _sample(probe, ins, seed=202)
    null_per_ch = torch.mean(torch.abs(n1 - n2), dim=(0, 1, 2))  # (39,)

    # --- oracle injection: +activity vs -activity (in-manifold) ---
    _inject(probe, +3.0); hi = _sample(probe, ins, seed=303)
    _inject(probe, -3.0); lo = _sample(probe, ins, seed=303)
    inj_per_ch = torch.mean(torch.abs(hi - lo), dim=(0, 1, 2))  # (39,)
    # restore (reload to drop the monkeypatch) not needed; process ends

    ratio_per_ch = (inj_per_ch / null_per_ch.clamp_min(1e-9)).cpu().numpy()
    inj_med = float(np.median(ratio_per_ch)); inj_max = float(np.max(ratio_per_ch))
    n_above = int((ratio_per_ch > 1.0).sum()); n_above2 = int((ratio_per_ch > 2.0).sum())
    # activity check: does +activity increase per-sample spread vs -activity?
    hi_spread = float(hi.std()); lo_spread = float(lo.std())

    print("=== Stage-C risk_context oracle-injection verdict ===")
    print(f"[non-reg] probe finite_rate={finite:.3f} | drift vs 734a mean|.|={drift:.3e} | std probe={probe_std:.3f} base={base_std:.3f}")
    print(f"[null band] median per-channel repeat spread = {float(null_per_ch.median()):.3e}")
    print(f"[injection] +act vs -act per-channel separation: median={float(inj_per_ch.median()):.3e}")
    print(f"[separation / null] median ratio={inj_med:.2f}  max={inj_max:.2f}  | channels>1x: {n_above}/39  >2x: {n_above2}/39")
    print(f"[activity] sample std: +3act={hi_spread:.3f}  -3act={lo_spread:.3f}  (expect +>-)")
    steers = inj_med > 1.5 and n_above >= 20 and finite > 0.99
    print(f"VERDICT: {'GO — slot steers activity past the null band' if steers else 'NO-GO / WEAK — injection inside or near the null band'}")
    print(f"  (median sep/null = {inj_med:.2f}x; {n_above}/39 channels above noise floor)")


if __name__ == "__main__":
    main()
